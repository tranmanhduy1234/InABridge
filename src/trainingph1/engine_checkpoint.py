"""Engine-owned DCP lifecycle shared by single-device and sharded training."""
import io
import json
import os
import random
import shutil
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict, get_state_dict, set_model_state_dict, set_state_dict,
)

from src.utils.checkpoint_metadata import (
    ROOT, FORMAT_VERSION, _architecture, _digest, build_metadata, build_model_tokenizer,
)
from src.utils.seed import seed_everything


def _rank():
    return dist.get_rank() if dist.is_initialized() else 0


def _world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def _rank0(fn):
    result = [None, None]
    if _rank() == 0:
        try:
            result[0] = fn()
        except Exception as exc:
            result[1] = f'{type(exc).__name__}: {exc}'
    if dist.is_initialized():
        dist.broadcast_object_list(result, src=0)
    if result[1]:
        raise RuntimeError(result[1])
    return result[0]


def _each_rank(fn):
    result, error = None, None
    try:
        result = fn()
    except Exception as exc:
        error = f'rank {_rank()}: {type(exc).__name__}: {exc}'
    errors = [error]
    if dist.is_initialized():
        errors = [None] * _world_size()
        dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError('; '.join(e for e in errors if e))
    return result


def _json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _check_model(metadata, model):
    if (_digest(_architecture(model)) != _digest(metadata['architecture']) or
            metadata['trainable'] != [n for n, p in model.named_parameters() if p.requires_grad]):
        raise ValueError('Model architecture or trainable parameters differ from run.json')


def _optimizer_groups(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    return [[names[id(p)] for p in group['params']] for group in optimizer.param_groups]


def create_run(state, parent_checkpoint=None):
    def create():
        cfg = state.settings
        for name in ('MODEL_VERSION', 'TRAINING_MODE', 'RUN_NAME'):
            if not cfg[name] or Path(cfg[name]).name != cfg[name] or cfg[name] in {'.', '..'}:
                raise ValueError(f'{name} must be a single directory name')
        path = ROOT / cfg['CHECKPOINT_DIR'] / cfg['MODEL_VERSION'] / cfg['TRAINING_MODE'] / cfg['RUN_NAME']
        metadata = build_metadata(state.model, state.tokenizer, cfg, parent_checkpoint)
        metadata['backend'] = 'dcp'
        path.mkdir(parents=True, exist_ok=False)
        _json(path / 'run.json', metadata)
        (path / 'snapshots').mkdir()
        return str(path)
    return Path(_rank0(create))


def read_checkpoint(source):
    def read():
        path = ROOT / source
        if path.is_dir() and (path / 'run.json').is_file():
            path = path / 'last.json'
        if path.is_file():
            name = json.loads(path.read_text())['snapshot']
            if len(name) != 32 or any(c not in '0123456789abcdef' for c in name):
                raise ValueError('Invalid checkpoint pointer')
            path = path.parent / 'snapshots' / name
        metadata = json.loads((path.parent.parent / 'run.json').read_text())
        complete = json.loads((path / 'checkpoint.json').read_text())
        if (metadata.get('format_version') != FORMAT_VERSION or metadata.get('backend') != 'dcp'
                or complete.get('run_digest') != _digest(metadata)):
            raise ValueError('Checkpoint format or run.json digest mismatch')
        return str(path), metadata
    path, metadata = _rank0(read)
    return Path(path), metadata


def _rng(device):
    numpy_state = np.random.get_state()
    return dict(python=random.getstate(), numpy=(numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)


def _restore_rng(rng, device):
    random.setstate(rng['python'])
    np.random.set_state((rng['numpy'][0], np.asarray(rng['numpy'][1], dtype=np.uint32), *rng['numpy'][2:]))
    torch.set_rng_state(rng['torch'])
    if rng['cuda'] is not None and device.type == 'cuda':
        torch.cuda.set_rng_state(rng['cuda'], device)


def _device(state):
    return next(state.model.parameters()).device


def _encode(value):
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


def read_training_state(path):
    data = {'extra': b''}
    dcp.load(data, checkpoint_id=path / 'state', no_dist=not dist.is_initialized())
    return torch.load(io.BytesIO(data['extra']), weights_only=True, map_location='cpu')


def save_checkpoint(state, *, best=False):
    def check_boundary():
        _check_model(json.loads((state.run_dir / 'run.json').read_text()), state.model)
        if any(p.grad is not None for p in state.model.parameters()):
            raise ValueError('Save only after zero_grad at an optimizer boundary')
        if any(type(v) is not int or v < 0 for v in state.progress.values()):
            raise ValueError('Progress counters must be non-negative integers')
        batch = state.progress['next_batch']
        if batch > len(state.train_loader) or (batch != len(state.train_loader)
                                               and batch % state.settings['ACCUMULATION_STEPS']):
            raise ValueError('Save requires an accumulation boundary')
    _each_rank(check_boundary)
    path = state.run_dir / 'snapshots' / _rank0(lambda: uuid4().hex)
    _rank0(lambda: path.mkdir())
    started = time.perf_counter()
    device, rng = _device(state), _rng(_device(state))
    logger = getattr(state, 'logger', None)
    if logger:
        logger.info('checkpoint save started step=%d snapshot=%s best=%s', state.progress['global_step'], path.name, best)
    try:
        metrics = getattr(state, 'epoch_metrics', None)
        local = dict(rng=rng, val_generator=state.val_loader.generator.get_state(),
                     epoch_metrics=None if metrics is None else
                     {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()})
        ranks = [local]
        if dist.is_initialized():
            ranks = [None] * _world_size()
            dist.all_gather_object(ranks, local)
        initialized = bool(state.optimizer.state)
        scaler = getattr(state, 'scaler', None)
        extra = dict(scheduler=state.scheduler.state_dict(), scaler=scaler.state_dict() if scaler else None,
                     ema_momentum=state.ema.momentum, progress=state.progress, events=state.events,
                     optimizer_initialized=initialized, optimizer_groups=_optimizer_groups(state.model, state.optimizer),
                     optimizer_type=type(state.optimizer).__name__, scheduler_type=type(state.scheduler).__name__,
                     ranks=ranks, device_type=device.type)
        tensors = dict(model=get_model_state_dict(state.model), ema=get_model_state_dict(state.ema.model),
                       queue=state.queue.state_dict(), extra=_each_rank(lambda: _encode(extra)))
        # State-dict extraction may initialize an empty optimizer with a dummy step.
        if initialized:
            tensors['model'], tensors['optimizer'] = get_state_dict(state.model, state.optimizer)
        dcp.save(tensors, checkpoint_id=path / 'state', no_dist=not dist.is_initialized())

        def publish():
            metadata = json.loads((state.run_dir / 'run.json').read_text())
            _json(path / 'checkpoint.json', dict(run_digest=_digest(metadata)))
            previous = {json.loads(p.read_text())['snapshot'] for p in
                        (state.run_dir / 'last.json', state.run_dir / 'best.json') if p.exists()}
            if best:
                _json(state.run_dir / 'best.json', dict(snapshot=path.name))
            _json(state.run_dir / 'last.json', dict(snapshot=path.name))
            retained = {json.loads(p.read_text())['snapshot'] for p in
                        (state.run_dir / 'last.json', state.run_dir / 'best.json') if p.exists()}
            for old in previous - retained:
                if len(old) == 32 and all(c in '0123456789abcdef' for c in old):
                    shutil.rmtree(state.run_dir / 'snapshots' / old)
        _rank0(publish)
    finally:
        _restore_rng(rng, device)
    elapsed = time.perf_counter() - started
    if logger:
        logger.info('checkpoint saved path=%s elapsed=%.3fs', path, elapsed)
    if hasattr(state, 'tb'):
        state.tb.log_scalars('checkpoint', dict(save_seconds=elapsed), state.progress['global_step'])
        state.tb.flush()
    return path


def load_weights(path, model):
    weights = dict(model=get_model_state_dict(model))
    dcp.load(weights, checkpoint_id=path / 'state', no_dist=not dist.is_initialized())
    set_model_state_dict(model, weights['model'])


def load_pretrained(source):
    path, metadata = read_checkpoint(source)
    model, tokenizer = build_model_tokenizer(metadata)
    load_weights(path, model)
    return model.eval(), tokenizer, metadata['config']


def load_checkpoint(path, metadata, state):
    extra = read_training_state(path)
    def check():
        _check_model(metadata, state.model)
        if extra['optimizer_groups'] != _optimizer_groups(state.model, state.optimizer):
            raise ValueError('Optimizer parameter groups do not match checkpoint')
        if (extra['optimizer_type'] != type(state.optimizer).__name__ or
                extra['scheduler_type'] != type(state.scheduler).__name__):
            raise ValueError('Optimizer/scheduler types do not match checkpoint')
    _each_rank(check)
    tensors = dict(model=get_model_state_dict(state.model), ema=get_model_state_dict(state.ema.model),
                   queue=state.queue.state_dict())
    if extra['optimizer_initialized']:
        tensors['model'], tensors['optimizer'] = get_state_dict(state.model, state.optimizer)
    dcp.load(tensors, checkpoint_id=path / 'state', no_dist=not dist.is_initialized())
    if extra['optimizer_initialized']:
        set_state_dict(state.model, state.optimizer, model_state_dict=tensors['model'], optim_state_dict=tensors['optimizer'])
    else:
        set_model_state_dict(state.model, tensors['model'])
        state.optimizer.state.clear()
    set_model_state_dict(state.ema.model, tensors['ema'])
    state.queue.load_state_dict(tensors['queue'])
    state.scheduler.load_state_dict(extra['scheduler'])
    scaler = getattr(state, 'scaler', None)
    if scaler is not None and extra['scaler'] is not None:
        scaler.load_state_dict(extra['scaler'])
    state.ema.momentum = extra['ema_momentum']
    state.progress, state.events = extra['progress'], extra['events']
    device = _device(state)
    same_topology = len(extra['ranks']) == _world_size() and extra['device_type'] == device.type
    if same_topology:
        local = extra['ranks'][_rank()]
        state.epoch_metrics = local['epoch_metrics']
        state.val_loader.generator.set_state(local['val_generator'])
        _restore_rng(local['rng'], device)
    else:
        # Keep aggregate epoch totals while giving the new topology independent RNG streams.
        metrics = [item['epoch_metrics'] for item in extra['ranks']]
        state.epoch_metrics = None
        if metrics and all(item is not None for item in metrics):
            state.epoch_metrics = dict(
                totals=sum(item['totals'] for item in metrics) if _rank() == 0 else torch.zeros_like(metrics[0]['totals']),
                counts=sum(item['counts'] for item in metrics) if _rank() == 0 else torch.zeros_like(metrics[0]['counts']),
                skipped_steps=max(item['skipped_steps'] for item in metrics))
        seed_everything((state.settings['SEED'] + state.progress['global_step'] + _rank()) % 2**32)
        state.val_loader.generator.manual_seed((state.settings['SEED'] + _rank()) % 2**32)
    return state.progress
