import hashlib
import json
import os
import random
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

import numpy as np
import torch

from src import config

ROOT = Path(__file__).resolve().parents[2]
FORMAT_VERSION = 5


def _architecture(model):
    def model_config(cfg):
        return {k: v for k, v in cfg.to_dict().items() if k not in {'transformers_version', '_name_or_path'}}

    return {
        'args': dict(bert_name=model.bert_name, model_vision_id=model.vision_encoder.model_id,
                     return_layer=model.vision_encoder.return_layer, num_queries=model.qformer_model.num_queries,
                     cross_attn_every=model.qformer_model.cross_attn_every, hidden_dim=model.qformer_model.hidden_dim,
                     itc_dim=model.itc_encoder.query_proj.out_features, normalize_eps=model.itc_encoder.normalize_eps),
        'bert_config': model_config(model.qformer_model.cfg),
        'vision_config': model_config(model.vision_encoder.model.config),
    }


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def create_run(model, tokenizer, *, run_dir=None, settings=None, parent_checkpoint=None):
    """Create a run directory and save its metadata."""
    if not tokenizer.is_fast:
        raise ValueError('A fast tokenizer is required for offline reconstruction')
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    backend['padding'] = backend['truncation'] = None
    metadata = {
        'format_version': FORMAT_VERSION,
        'run_id': uuid4().hex,
        'architecture': _architecture(model),
        'tokenizer': dict(backend=backend, special_tokens=tokenizer.special_tokens_map,
                          padding_side=tokenizer.padding_side, truncation_side=tokenizer.truncation_side,
                          model_max_length=tokenizer.model_max_length, model_input_names=tokenizer.model_input_names,
                          clean_up_tokenization_spaces=tokenizer.clean_up_tokenization_spaces,
                          split_special_tokens=tokenizer.split_special_tokens),
        'trainable': [name for name, p in model.named_parameters() if p.requires_grad],
        'config': {k: v for k, v in (vars(config) | (settings or {})).items() if k.isupper()},
        'parent_checkpoint': str(parent_checkpoint) if parent_checkpoint is not None else None,
    }
    encoded = json.dumps(metadata, indent=2)
    settings = vars(config) | (settings or {})
    run_dir = ROOT / (Path(run_dir) if run_dir is not None else
                      Path(settings['CHECKPOINT_DIR']) / settings['MODEL_VERSION'] / settings['RUN_NAME'])
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / 'run.json').write_text(encoded + '\n')
    return run_dir


def _run(run_dir):
    metadata = json.loads((Path(run_dir) / 'run.json').read_text())
    if metadata.get('format_version') != FORMAT_VERSION:
        raise ValueError('Unsupported checkpoint format; use a run created with the current format')
    return metadata


def _components(optimizer, scheduler, scaler, ema, queue):
    return {name: obj for name, obj in dict(optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, ema=ema, queue=queue).items() if obj is not None}


def _optimizer_groups(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    return [[names[id(p)] for p in group['params']] for group in optimizer.param_groups]


def _check_model(metadata, model):
    if _digest(metadata['architecture']) != _digest(_architecture(model)):
        raise ValueError('Model architecture does not match this run')
    if metadata['trainable'] != [n for n, p in model.named_parameters() if p.requires_grad]:
        raise ValueError('Trainable parameters changed; create a new fine-tune run')


def save_checkpoint(run_dir, model, *, optimizer, scheduler, global_step, epoch, next_batch,
                    scaler=None, ema=None, queue=None, data_state=None, filename='last.pt'):
    """Save at an optimizer-step boundary, after EMA/queue updates and zero_grad."""
    if filename not in {'last.pt', 'best.pt'}:
        raise ValueError('filename must be last.pt or best.pt')
    if any(type(n) is not int or n < 0 for n in (global_step, epoch, next_batch)):
        raise ValueError('Progress counters must be non-negative integers')
    metadata = _run(run_dir)
    _check_model(metadata, model)
    if optimizer is None or scheduler is None:
        raise ValueError('Optimizer and scheduler are required for resumable checkpoints')
    objects = _components(optimizer, scheduler, scaler, ema, queue)
    numpy_state = np.random.get_state()
    payload = {
        'format_version': FORMAT_VERSION, 'run_id': _digest(metadata), 'model': model.state_dict(),
        'states': {name: obj.state_dict() for name, obj in objects.items()},
        'types': {name: f'{type(obj).__module__}.{type(obj).__qualname__}' for name, obj in objects.items()},
        'optimizer_groups': _optimizer_groups(model, optimizer),
        'ema_momentum': ema.momentum if ema is not None else None,
        'global_step': global_step, 'epoch': epoch, 'next_batch': next_batch, 'data_state': data_state,
        'rng': dict(python=random.getstate(), numpy=(numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
                    torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),
    }
    path = Path(run_dir) / filename
    with NamedTemporaryFile(dir=run_dir, suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def _read(path):
    path = ROOT / Path(path)
    metadata = _run(path.parent)
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if payload.get('format_version') != FORMAT_VERSION or payload.get('run_id') != _digest(metadata):
        raise ValueError('Checkpoint does not belong to this run.json')
    return metadata, payload


def _load_weights(model, weights):
    current = model.state_dict()
    if weights.keys() != current.keys() or any(
        not isinstance(weights[name], torch.Tensor) or weights[name].shape != tensor.shape
        for name, tensor in current.items()
    ):
        raise ValueError('Checkpoint parameter names or shapes do not match the model')
    model.load_state_dict(weights, strict=True)


def load_pretrained(path):
    """Load the model and tokenizer offline for inference."""
    from tokenizers import AddedToken, Tokenizer
    from transformers import BertConfig, DINOv3ViTConfig, PreTrainedTokenizerFast
    from src.trainingph1.model import ModelStage1

    metadata, payload = _read(path)
    architecture = metadata['architecture']
    model = ModelStage1(**architecture['args'], bert_config=BertConfig.from_dict(architecture['bert_config']),
                        vision_config=DINOv3ViTConfig.from_dict(architecture['vision_config']))
    _load_weights(model, payload['model'])
    trainable = set(metadata['trainable'])
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in trainable)
    token_config = metadata['tokenizer']
    added = {token['content']: AddedToken(**{k: v for k, v in token.items() if k != 'id'})
             for token in token_config['backend']['added_tokens']}
    specials = {name: [added.get(t, t) for t in value] if isinstance(value, list)
                else added.get(value, value) for name, value in token_config['special_tokens'].items()}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_str(json.dumps(token_config['backend'])), **specials,
        **{k: v for k, v in token_config.items() if k not in {'backend', 'special_tokens'}},
    )
    return model.eval(), tokenizer, metadata['config']


def load_checkpoint(path, model, *, optimizer, scheduler, scaler=None, ema=None, queue=None):
    """Restore training state into components built from the saved run config."""
    metadata, payload = _read(path)
    _check_model(metadata, model)
    objects = _components(optimizer, scheduler, scaler, ema, queue)
    types = {name: f'{type(obj).__module__}.{type(obj).__qualname__}' for name, obj in objects.items()}
    if payload['types'] != types:
        raise ValueError('Resume requires the same optimizer/scheduler/scaler/EMA/queue components')
    if payload['optimizer_groups'] != _optimizer_groups(model, optimizer):
        raise ValueError('Optimizer parameter groups do not match the checkpoint')
    _load_weights(model, payload['model'])
    for name, obj in objects.items():
        obj.load_state_dict(payload['states'][name])
    if ema is not None:
        ema.momentum = payload['ema_momentum']
    rng = payload['rng']
    random.setstate(rng['python'])
    np.random.set_state((rng['numpy'][0], np.asarray(rng['numpy'][1], dtype=np.uint32), *rng['numpy'][2:]))
    torch.set_rng_state(rng['torch'])
    if rng['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng['cuda'])
    return {key: payload[key] for key in ('global_step', 'epoch', 'next_batch', 'data_state')}