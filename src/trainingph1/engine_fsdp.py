"""Run with torchrun --standalone --nproc-per-node=2 -m src.trainingph1.engine_fsdp.

Without a process group, runs an unsharded reference using the same checkpoint format.
"""

from itertools import islice
from pathlib import Path
from types import SimpleNamespace
import math
import os
import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, fully_shard, MixedPrecisionPolicy, register_fsdp_forward_method
from torch.distributed.tensor import DTensor, Partial, Shard
from transformers import AutoTokenizer

from src import config
from src.data.dataloaderph1_multilevel import (
    VLMDatasetStage1, VLMDataCollator, build_dataloader, build_transform, set_dataloader_epoch,
)
from src.losses.lossph1 import Stage1Criterion
from src.queue.ema import EMA
from src.queue.moco import MoCoQueue
from src.trainingph1.model import ModelStage1
from src.trainingph1.engine import (
    get_data_fingerprints, get_optimizer, get_scheduler, get_pseudo_weight, _data_rng, _data_rng_state, _training_batches,
)
from src.trainingph1 import engine_checkpoint as checkpoint
from src.utils.seed import seed_everything
from src.utils.logging_setup import setup_logging, close_logging
from src.utils.tb_logger import TensorBoardLogger
from src.utils.training_diagnostics import TrainingDiagnostics, GRADIENT_GROUPS


@torch.no_grad()
def gradient_diagnostics(model, device):
    sums = torch.zeros(len(GRADIENT_GROUPS), 2, dtype=torch.float64, device=device)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = ('qformer' if name.startswith('itc_encoder.qformer.') else
                 'itm_head' if name.startswith('itm_logit.') else
                 name.split('.')[1] if name.startswith('itc_encoder.') else 'dec_embedding')
        i = GRADIENT_GROUPS.index(group)
        weight = parameter.to_local() if isinstance(parameter, DTensor) else parameter
        sums[i, 1] += weight.double().square().sum()
        if parameter.grad is not None:
            grad = parameter.grad.to_local() if isinstance(parameter.grad, DTensor) else parameter.grad
            sums[i, 0] += grad.double().square().sum()
    if dist.is_initialized():
        dist.all_reduce(sums)
    result = {}
    for name, (grad, weight) in zip(GRADIENT_GROUPS, sums.sqrt().cpu().tolist()):
        result[f'gradients/{name}/norm'] = grad
        if weight > 0:
            result[f'gradients/{name}/grad_to_weight_norm'] = grad / weight
    return result


OBJECTIVES = ("itc", "itm", "itg")


def init_runtime(settings):
    device = torch.device(settings["DEVICE"])
    if device.type == "cuda":
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", device.index or 0)))
        torch.cuda.set_device(device)
    owned = not dist.is_initialized() and "RANK" in os.environ
    if owned:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    active = dist.is_initialized()
    return SimpleNamespace(device=device, rank=dist.get_rank() if active else 0,
                           world_size=dist.get_world_size() if active else 1,
                           distributed=active, owns_process_group=owned)


def _sum(tensor):
    if dist.is_initialized():
        dist.all_reduce(tensor)
    return tensor


def create_ema(model, momentum):
    if any(isinstance(m, FSDPModule) for m in model.modules()) or any(
        isinstance(t, DTensor) for t in (*model.parameters(), *model.buffers())
    ):
        raise ValueError("Create EMA before sharding the online model")
    return EMA(model, momentum)


def _ema_local_pairs(ema, online):
    def local_pair(teacher, source):
        if isinstance(teacher, DTensor) != isinstance(source, DTensor):
            raise ValueError("EMA sharding mismatch")
        if isinstance(teacher, DTensor):
            if (teacher.shape != source.shape or teacher.device_mesh != source.device_mesh
                    or teacher.placements != source.placements
                    or any(isinstance(p, Partial) for p in teacher.placements)):
                raise ValueError("EMA shape/mesh/placements mismatch or partial tensor")
            teacher, source = teacher.to_local(), source.to_local()
        return teacher, source
    return [[local_pair(teacher, source) for teacher, source in pairs]
            for pairs in ema.tensor_pairs(online)]


def update_ema(ema, online):
    ema.update_tensors(*_ema_local_pairs(ema, online))


def enqueue_keys(queue, images, texts, image_ids):
    if any(isinstance(t, DTensor) for t in (images, texts, image_ids, *queue.buffers())):
        raise ValueError("MoCo queue and incoming keys must be full, unsharded tensors")
    queue.enqueue(images, texts, image_ids)


def build_dataset(*args, distributed=False, **kwargs):
    if not distributed:
        return VLMDatasetStage1(*args, **kwargs)
    if not dist.is_initialized():
        raise RuntimeError("distributed=True requires an initialized process group")
    dataset, error = None, [None]
    if dist.get_rank() == 0:
        try:
            dataset = VLMDatasetStage1(*args, **kwargs)
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(error[0])
    return checkpoint._each_rank(lambda: dataset if dataset is not None else
                                 VLMDatasetStage1(*args, **kwargs, prepare_cache=False))


def shard_models(model, ema, mesh, amp_enabled=False):
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16 if amp_enabled else None,
                                  reduce_dtype=torch.float32)
    options = dict(mesh=mesh, reshard_after_forward=True, mp_policy=policy)
    for encoder in (model.itc_encoder, ema.model):
        for block in encoder.qformer.layers:
            fully_shard(block, **options)
        fully_shard(encoder.qformer, **options)
        fully_shard(encoder, **options)
    if not model.vision_encoder.mock:
        for block in model.vision_encoder.model.model.layer:
            fully_shard(block, **options)
    fully_shard(model.vision_encoder, **options)
    # One group owns the tied weight. It reshards after LM projection, the last user.
    fully_shard([model.embeddings, model.lm_head], **options)
    fully_shard(model, **options)
    # The first image encoding must initialize FSDP at the online root, not a child.
    register_fsdp_forward_method(model, "encode_image")
    if model.lm_head.weight is not model.embeddings.word_embeddings.weight:
        raise RuntimeError("FSDP broke embedding/LM-head weight tying")
    for module in (model, ema.model):
        if any(not isinstance(p, DTensor) for p in module.parameters()):
            raise RuntimeError("Every online/teacher parameter must be sharded")
    _ema_local_pairs(ema, model.itc_encoder)


class BatchPartition:
    """Partition stable global batches without changing their order across resumes."""
    def __init__(self, sampler, rank, replicas):
        self.sampler, self.rank, self.replicas = sampler, rank, replicas
        self.batch_size = math.ceil(sampler.batch_size / replicas)

    def __len__(self):
        return len(self.sampler)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        for batch in self.sampler:
            padded = batch + [(batch[0][0], False)] * (self.batch_size * self.replicas - len(batch))
            start = self.rank * self.batch_size
            yield padded[start:start + self.batch_size]


def partition_loader(loader, rank, replicas):
    options = {name: getattr(loader, name) for name in (
        'num_workers', 'pin_memory', 'persistent_workers', 'worker_init_fn', 'collate_fn')}
    if loader.num_workers:
        options['prefetch_factor'] = loader.prefetch_factor
    return torch.utils.data.DataLoader(
        loader.dataset, batch_sampler=BatchPartition(loader.batch_sampler, rank, replicas),
        generator=torch.Generator().manual_seed(loader.generator.initial_seed() + rank), **options)


def get_dataloader(tokenizer, settings, distributed=False):
    if distributed and not dist.is_initialized():
        raise RuntimeError("distributed=True requires an initialized process group")
    rank, replicas = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    cfg = SimpleNamespace(**settings)
    root = Path(__file__).resolve().parents[2]
    collator = VLMDataCollator(tokenizer, cfg.MAX_LENGTH, cfg.TOKENIZER_PADDING, cfg.TOKENIZER_TRUNCATION)
    loaders = []
    for training, manifest in ((True, cfg.JSON_PATH_TRAIN), (False, cfg.JSON_PATH_VAL)):
        transform = build_transform(
            cfg.IMAGE_SIZE, training, cfg.NORMALIZATION, cfg.CROP_SCALE, cfg.CROP_RATIO,
            cfg.INTERPOLATION, cfg.ANTIALIAS, cfg.HORIZONTAL_FLIP_PROBABILITY,
            cfg.COLOR_JITTER, cfg.COLOR_JITTER_PROBABILITY, cfg.GRAYSCALE_PROBABILITY,
            cfg.BLUR_KERNEL_SIZE, cfg.BLUR_SIGMA, cfg.BLUR_PROBABILITY)
        dataset = build_dataset(
            root / manifest, root / cfg.IMAGE_DIR, training, root / cfg.CACHE_DIR,
            cfg.CACHE_CHUNK_SIZE, transform, cfg.DATASET_NAMESPACE, tokenizer, cfg.MAX_LENGTH,
            distributed=distributed)
        def make_loader():
            loader = build_dataloader(
                dataset, settings.get("GLOBAL_BATCH_SIZE", cfg.BATCH_SIZE * replicas), cfg.NUM_WORKERS, cfg.DROP_LAST, cfg.SHUFFLE,
                cfg.PIN_MEMORY, cfg.PERSISTENT_WORKERS and not training, cfg.PREFETCH_FACTOR,
                collator,
                length_aware=cfg.LENGTH_AWARE, mega_batch_mult=cfg.LENGTH_MEGA_BATCH_MULT,
                length_jitter=cfg.LENGTH_JITTER, seed=cfg.SEED)
            return partition_loader(loader, rank, replicas) if distributed else loader
        loaders.append(checkpoint._each_rank(make_loader) if distributed else make_loader())
    return loaders


def prepare_training(settings, runtime):
    if runtime.distributed:
        configs = [None] * runtime.world_size
        dist.all_gather_object(configs, settings)
        if any(cfg != configs[0] for cfg in configs):
            raise ValueError("All ranks must use the same training settings")
    cfg = SimpleNamespace(**settings)
    if cfg.TRAINING_MODE not in {'pretrain', 'finetune'}:
        raise ValueError('TRAINING_MODE must be pretrain or finetune')
    if cfg.INIT_CHECKPOINT and cfg.RESUME_CHECKPOINT:
        raise ValueError('INIT_CHECKPOINT and RESUME_CHECKPOINT are mutually exclusive')
    if not cfg.RESUME_CHECKPOINT and bool(cfg.INIT_CHECKPOINT) != (cfg.TRAINING_MODE == 'finetune'):
        raise ValueError('New finetune requires INIT_CHECKPOINT; new pretrain requires no checkpoint')
    source = cfg.RESUME_CHECKPOINT or cfg.INIT_CHECKPOINT
    source_path, metadata = None, None
    if source:
        source_path, metadata = checkpoint.read_checkpoint(source)
    if cfg.RESUME_CHECKPOINT:
        if metadata['config']['TRAINING_MODE'] != cfg.TRAINING_MODE:
            raise ValueError('TRAINING_MODE must match the resumed run')
        logging_settings = {name: settings[name] for name in (
            'LOG_EVERY_STEPS', 'LOG_LEVEL', 'TENSORBOARD_ENABLED', 'TENSORBOARD_FLUSH_SECS',
            'TENSORBOARD_DIAGNOSTICS_EVERY_STEPS', 'TENSORBOARD_TEXT_SAMPLES')}
        settings = {name: value for name, value in vars(config).items() if name.isupper()} | metadata['config'] | dict(
            DEVICE=cfg.DEVICE, AMP_ENABLED=cfg.AMP_ENABLED, AMP_DTYPE=cfg.AMP_DTYPE,
            INIT_CHECKPOINT=None, RESUME_CHECKPOINT=cfg.RESUME_CHECKPOINT) | logging_settings
        cfg = SimpleNamespace(**settings)
    settings.setdefault("GLOBAL_BATCH_SIZE", cfg.BATCH_SIZE * runtime.world_size)
    if cfg.AMP_ENABLED and cfg.AMP_DTYPE != "bfloat16":
        raise ValueError("This FSDP baseline supports BF16 or FP32, not FP16/GradScaler")
    if any(type(settings[name]) is not int or settings[name] < 1 for name in ('ACCUMULATION_STEPS', 'EPOCHS')):
        raise ValueError("ACCUMULATION_STEPS and EPOCHS must be positive integers")
    for name in ("SAVE_EVERY_STEPS", "VAL_EVERY_STEPS", "LOG_EVERY_STEPS"):
        if type(settings[name]) is not int or settings[name] < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    for name in ('TENSORBOARD_DIAGNOSTICS_EVERY_STEPS', 'TENSORBOARD_TEXT_SAMPLES'):
        if type(settings[name]) is not int or settings[name] < 0:
            raise ValueError(f'{name} must be a non-negative integer')
    if cfg.MAX_GRAD_NORM is not None and (not math.isfinite(cfg.MAX_GRAD_NORM) or cfg.MAX_GRAD_NORM <= 0):
        raise ValueError("MAX_GRAD_NORM must be finite and positive")
    seed_everything(cfg.SEED)
    if metadata:
        model, tokenizer = checkpoint.build_model_tokenizer(metadata)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.TOKENIZER_MODEL_ID, use_fast=cfg.TOKENIZER_USE_FAST)
        model = ModelStage1(
            cfg.BERT_MODEL_ID, cfg.VISION_RETURN_LAYER, cfg.VISION_MODEL_ID, cfg.NUM_QUERIES,
            cfg.CROSS_ATTN_EVERY, cfg.HIDDEN_DIM, cfg.ITC_DIM, cfg.NORMALIZE_EPS,
            vision_mock=cfg.VISION_MOCK)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no PAD/EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    model.train()
    settings.update(NUM_QUERIES=model.qformer_model.num_queries,
                    ITC_DIM=model.itc_encoder.query_proj.out_features, VISION_MOCK=model.vision_encoder.mock)
    ema = create_ema(model.itc_encoder, cfg.MOMENTUM)
    if runtime.distributed:
        mesh = init_device_mesh(runtime.device.type, (runtime.world_size,))
        shard_models(model, ema, mesh, cfg.AMP_ENABLED)
    else:
        model.to(runtime.device)
        ema.to(runtime.device)
    if source_path and not cfg.RESUME_CHECKPOINT:
        checkpoint.load_weights(source_path, model)
        ema.momentum = 0.
        update_ema(ema, model.itc_encoder)
        ema.momentum = cfg.MOMENTUM
    optimizer = get_optimizer(model, settings)
    train_loader, val_loader = get_dataloader(tokenizer, settings, runtime.distributed)
    if runtime.distributed:
        sizes = [None] * runtime.world_size
        dist.all_gather_object(sizes, (len(train_loader), len(val_loader)))
        if any(size != sizes[0] for size in sizes):
            raise ValueError("All ranks must have equal training/validation batch counts")
    scheduler = get_scheduler(optimizer, batches_per_epoch=len(train_loader), settings=settings)
    fingerprints = get_data_fingerprints(settings, (train_loader, val_loader))
    if cfg.RESUME_CHECKPOINT:
        if settings.get('DATA_FINGERPRINTS') != fingerprints:
            raise ValueError('Dataset manifest/cache fingerprint changed; start a new finetune run')
        if settings.get('BATCHES_PER_EPOCH') != len(train_loader):
            raise ValueError('Resume requires the same batches per epoch')
    settings.update(DATA_FINGERPRINTS=fingerprints, BATCHES_PER_EPOCH=len(train_loader), DATA_RNG_VERSION=2)
    queue = MoCoQueue(cfg.QUEUE_SIZE, settings['NUM_QUERIES'], settings['ITC_DIM']).to(runtime.device)
    seed_everything((cfg.SEED + runtime.rank) % 2**32)
    state = SimpleNamespace(
        model=model, ema=ema, optimizer=optimizer, scheduler=scheduler, queue=queue,
        criterion=Stage1Criterion(), tokenizer=tokenizer, train_loader=train_loader,
        val_loader=val_loader, runtime=runtime, settings=settings,
        progress=dict(global_step=0, epoch=0, next_batch=0), events={}, epoch_metrics=None)
    if cfg.RESUME_CHECKPOINT:
        state.run_dir = source_path.parent.parent
        checkpoint.load_checkpoint(source_path, metadata, state)
    else:
        state.run_dir = checkpoint.create_run(state, parent_checkpoint=str(source_path) if source_path else source)
    if runtime.distributed:
        for buffer in queue.buffers():
            dist.broadcast(buffer, src=0)
    return state


@torch.no_grad()
def gather_keys(momentum, image_ids, valid_mask):
    keys = dict(image_features=momentum["image_features"].detach().float(),
                text_features=momentum["text_features"].detach().float(),
                image_ids=image_ids, valid_mask=valid_mask)
    if dist.is_initialized():
        for name, value in keys.items():
            parts = [torch.empty_like(value) for _ in range(dist.get_world_size())]
            dist.all_gather(parts, value.contiguous())
            keys[name] = torch.cat(parts)
    return keys


def _pair_mask(image_ids, valid):
    return valid[:, None] & valid[None, :] & image_ids[:, None].ne(image_ids[None, :])


def _loss_counts(batch):
    valid = batch["valid_mask"]
    pairs = _pair_mask(batch["image_ids"], valid).any(-1)
    tokens = valid[:, None] & batch["attention_mask"][:, 1:].bool() & batch["input_ids"][:, 1:].ne(-100)
    return torch.stack((valid.sum(), 3 * pairs.sum(), tokens.sum()))


def compute_loss(state, batch, pseudo_weight):
    """All ranks must execute the full ITC -> three ITM passes -> ITG sequence."""
    cfg = SimpleNamespace(**state.settings)
    batch = {name: value.to(state.runtime.device, non_blocking=True) for name, value in batch.items()}
    images, ids = batch["images"], batch["image_ids"]
    tokens, mask, valid = batch["input_ids"], batch["attention_mask"], batch["valid_mask"]
    model, criterion = state.model, state.criterion
    image_features = model.encode_image(images)
    online = model(image_features, tokens, mask, "itc")
    with torch.no_grad():
        momentum = state.ema.model(image_features, model.encode_text(tokens), mask)
    keys = gather_keys(momentum, ids, valid)
    image_queue, text_queue, queue_ids = state.queue.get()
    itc = criterion.get_itc_loss(
        online["image_features"], online["text_features"],
        momentum["image_features"], momentum["text_features"], image_queue, text_queue,
        cfg.TEMPERATURE, pseudo_weight, ids, queue_ids, valid,
        global_image_keys=keys["image_features"], global_text_keys=keys["text_features"],
        global_image_ids=keys["image_ids"], global_valid_mask=keys["valid_mask"])
    losses = {name: itc[name] for name in ("loss_itc_sum", "count_itc")}
    diagnostics = getattr(state, 'diagnostics', None)
    if diagnostics is not None:
        diagnostics.itc(itc, online, momentum, keys, queue_ids, ids, valid, cfg.TEMPERATURE, pseudo_weight)
    with torch.no_grad(), torch.autocast(device_type=images.device.type, enabled=False):
        candidates = _pair_mask(ids, valid)
        pair_valid = candidates.any(-1)
        similarity = torch.einsum("bqd,kd->bkq", online["image_features"].float(),
                                  online["text_features"].float()).amax(-1) / cfg.TEMPERATURE
        # A bad forward must reach the global skip decision, not fail in multinomial.
        similarity = torch.where(similarity.isfinite(), similarity, 0.)
        negative_indices = []
        for scores in (similarity, similarity.T):
            scores = scores.masked_fill(~candidates, -torch.inf)
            scores = torch.where(pair_valid[:, None], scores, torch.zeros_like(scores))
            negative_indices.append(torch.multinomial(scores.softmax(-1), 1).squeeze(1))
        negative_text, negative_image = negative_indices
    itm_losses = []
    for features, input_ids, attention_mask, label, branch in (
        (image_features, tokens, mask, 1, 'positive'),
        (image_features[negative_image], tokens, mask, 0, 'negative_image'),
        (image_features, tokens[negative_text], mask[negative_text], 0, 'negative_text'),
    ):
        out = model(features, input_ids, attention_mask, "itm")
        labels = torch.full_like(ids, label)
        itm_losses.append(criterion.get_itm_loss(out["itm_logits"], labels, pair_valid)["loss_itm_sum"])
        if diagnostics is not None:
            diagnostics.itm(branch, out['itm_logits'], pair_valid)
    losses.update(loss_itm_sum=sum(itm_losses), count_itm=3 * pair_valid.sum())
    itg = model(image_features, tokens, mask, "itg")
    labels = tokens[:, 1:].masked_fill(~mask[:, 1:].bool(), -100)
    losses.update(criterion.get_itg_loss(itg["itg_logits"], labels, valid))
    if diagnostics is not None:
        diagnostics.itg(itg['itg_logits'], labels, valid, images)
        diagnostics.losses(losses)
    pending = {name: value[keys["valid_mask"]] for name, value in keys.items() if name != "valid_mask"}
    return losses, pending


@torch.no_grad()
def gradient_norm(model, device):
    squared = torch.zeros((), dtype=torch.float64, device=device)
    for parameter in model.parameters():
        grad = parameter.grad
        if grad is None:
            continue
        if isinstance(grad, DTensor):
            if len(grad.placements) != 1 or not isinstance(grad.placements[0], Shard):
                raise ValueError("Expected gradients sharded on a one-dimensional data-parallel mesh")
            grad = grad.to_local()
        elif dist.is_initialized():
            raise ValueError("Distributed training requires sharded parameter gradients")
        squared += grad.double().square().sum()
    return _sum(squared).sqrt()


@torch.no_grad()
def _clip_gradients(model, norm, max_norm):
    scale = (max_norm / (norm + 1e-6)).clamp(max=1.)
    for parameter in model.parameters():
        if parameter.grad is not None:
            grad = parameter.grad.to_local() if isinstance(parameter.grad, DTensor) else parameter.grad
            grad.mul_(scale.to(grad.dtype))


def _metrics(state, totals, counts):
    totals, counts = _sum(totals.clone()), _sum(counts.clone())
    values = (totals / counts.clamp_min(1)).tolist()
    result = {f"loss_{name}": value for name, value in zip(OBJECTIVES, values)}
    result["loss"] = sum(state.settings[f"{name.upper()}_WEIGHT"] * result[f"loss_{name}"]
                         for name in OBJECTIVES)
    return result


def _autocast(state):
    return torch.autocast(state.runtime.device.type, dtype=torch.bfloat16,
                          enabled=state.settings["AMP_ENABLED"])


def _diagnostics_enabled(state):
    return (hasattr(state, 'tb') and state.settings['TENSORBOARD_ENABLED']
            and state.settings['TENSORBOARD_DIAGNOSTICS_EVERY_STEPS'] > 0)


def train_one_epoch(state, epoch, on_step=None):
    """on_step runs on every rank so it may perform collective operations."""
    cfg = SimpleNamespace(**state.settings)
    if type(cfg.ACCUMULATION_STEPS) is not int or cfg.ACCUMULATION_STEPS < 1:
        raise ValueError("ACCUMULATION_STEPS must be positive")
    logger = getattr(state, 'logger', None)
    epoch_start = time.perf_counter()
    start_batch = state.progress['next_batch'] if epoch == state.progress['epoch'] else 0
    if not 0 <= start_batch <= len(state.train_loader) or (
            start_batch != len(state.train_loader) and start_batch % cfg.ACCUMULATION_STEPS):
        raise ValueError('Resume requires a valid accumulation boundary')
    state.model.train()
    state.ema.eval()
    set_dataloader_epoch(state.train_loader, epoch)
    state.train_loader.generator.manual_seed((cfg.SEED + epoch + state.runtime.rank) % 2**32)
    state.optimizer.zero_grad(set_to_none=True)
    totals = torch.zeros(3, dtype=torch.float64, device=state.runtime.device)
    counts = torch.zeros(3, dtype=torch.long, device=state.runtime.device)
    skipped_steps = 0
    saved_metrics = getattr(state, 'epoch_metrics', None)
    if start_batch and saved_metrics is not None:
        totals, counts = (saved_metrics[name].to(state.runtime.device) for name in ('totals', 'counts'))
        skipped_steps = saved_metrics['skipped_steps']
    batches = _training_batches(state.train_loader, cfg.SEED + epoch + state.runtime.rank, start_batch)
    if logger:
        logger.info('epoch=%d/%d start batch=%d/%d global_step=%d', epoch + 1, cfg.EPOCHS,
                    start_batch, len(state.train_loader), state.progress['global_step'])
    if state.runtime.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(state.runtime.device)
    for start in range(start_batch, len(state.train_loader), cfg.ACCUMULATION_STEPS):
        next_step = state.progress['global_step'] + 1
        collect = _diagnostics_enabled(state) and (next_step == 1 or next_step % cfg.TENSORBOARD_DIAGNOSTICS_EVERY_STEPS == 0)
        state.diagnostics = TrainingDiagnostics(state.runtime.device) if collect else None
        state.step_diagnostics = None
        step_start = time.perf_counter()
        learning_rate = state.optimizer.param_groups[0]['lr']
        group = list(islice(batches, cfg.ACCUMULATION_STEPS))
        global_counts = _sum(sum(_loss_counts(batch) for batch in group).to(state.runtime.device))
        group_totals, group_counts = torch.zeros_like(totals), torch.zeros_like(counts)
        pending, finite = [], torch.ones((), dtype=torch.int32, device=state.runtime.device)
        for offset, batch in enumerate(group):
            alpha = get_pseudo_weight(epoch + (start + offset) / len(state.train_loader), state.settings)
            with _autocast(state):
                losses, keys = compute_loss(state, batch, alpha)
                loss = sum(state.settings[f"{name.upper()}_WEIGHT"] * state.runtime.world_size
                           * losses[f"loss_{name}_sum"] / global_counts[i].clamp_min(1)
                           for i, name in enumerate(OBJECTIVES))
            finite *= loss.detach().isfinite().to(torch.int32)
            loss.backward()
            pending.append(keys)
            for i, name in enumerate(OBJECTIVES):
                group_totals[i] += losses[f"loss_{name}_sum"].detach()
                group_counts[i] += losses[f"count_{name}"]
        if state.runtime.distributed:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        norm = gradient_norm(state.model, state.runtime.device)
        updated = bool(finite.item() and norm.isfinite().item() and global_counts[0].item())
        if updated:
            if state.diagnostics is not None:
                state.step_diagnostics = state.diagnostics.summarize(state.settings, _sum(state.diagnostics.pack()))
                state.step_diagnostics['scalars'].update(gradient_diagnostics(state.model, state.runtime.device))
                clip_scale = min(1., cfg.MAX_GRAD_NORM / (norm.item() + 1e-6)) if cfg.MAX_GRAD_NORM else 1.
                state.step_diagnostics['scalars'].update({'gradients/clip_scale': clip_scale,
                                                          'gradients/clipped': float(clip_scale < 1.)})
            if cfg.MAX_GRAD_NORM is not None:
                _clip_gradients(state.model, norm, cfg.MAX_GRAD_NORM)
            state.optimizer.step()
            state.scheduler.step()
            update_ema(state.ema, state.model.itc_encoder)
            enqueue_keys(state.queue, *(torch.cat([keys[name] for keys in pending])
                                        for name in ("image_features", "text_features", "image_ids")))
            state.progress["global_step"] += 1
            totals += group_totals
            counts += group_counts
        else:
            skipped_steps += 1
            if logger:
                reason = ('nonfinite_loss' if not finite.item() else
                          'nonfinite_gradient' if not norm.isfinite().item() else 'no_valid_samples')
                logger.warning('step skipped: epoch=%d batches=%d:%d global_step=%d reason=%s skipped_steps=%d',
                               epoch + 1, start, start + len(group), state.progress['global_step'], reason, skipped_steps)
        state.optimizer.zero_grad(set_to_none=True)
        state.diagnostics = None
        state.progress.update(epoch=epoch, next_batch=start + len(group))
        state.epoch_metrics = dict(totals=totals, counts=counts, skipped_steps=skipped_steps)
        seconds = time.perf_counter() - step_start
        state.step_stats = dict(lr=learning_rate, grad_norm=norm.item(), step_seconds=seconds,
                                samples_per_second=global_counts[0].item() / max(seconds, 1e-9),
                                tokens_per_second=global_counts[2].item() / max(seconds, 1e-9),
                                epoch=epoch + (start + len(group)) / len(state.train_loader),
                                queue_count=state.queue.count.item(), queue_fill=state.queue.count.item() / state.queue.capacity)
        if state.runtime.device.type == 'cuda':
            state.step_stats.update(cuda_allocated_gb=torch.cuda.memory_allocated(state.runtime.device) / 2**30,
                                    cuda_peak_gb=torch.cuda.max_memory_allocated(state.runtime.device) / 2**30)
        if updated and on_step is not None:
            on_step(state, _metrics(state, totals, counts) | {"skipped_steps": skipped_steps})
    state.progress.update(epoch=epoch + 1, next_batch=0)
    state.epoch_metrics = None
    metrics = _metrics(state, totals, counts) | {"skipped_steps": skipped_steps}
    if logger:
        logger.info('epoch=%d complete global_step=%d elapsed=%.3fs metrics=%s',
                    epoch + 1, state.progress['global_step'], time.perf_counter() - epoch_start, metrics)
    return metrics


@torch.no_grad()
def validate(state, pseudo_weight=None):
    totals = torch.zeros(3, dtype=torch.float64, device=state.runtime.device)
    counts = torch.zeros(3, dtype=torch.long, device=state.runtime.device)
    alpha = state.settings["PSEUDO_WEIGHT"] if pseudo_weight is None else pseudo_weight
    was_training = state.model.training
    state.model.eval()
    generator_state = state.val_loader.generator.get_state()
    devices = [state.runtime.device.index] if state.runtime.device.type == "cuda" else []
    previous_diagnostics = getattr(state, 'diagnostics', None)
    diagnostics = (TrainingDiagnostics(state.runtime.device, state.settings['TENSORBOARD_TEXT_SAMPLES']
                                      if state.runtime.rank == 0 else 0) if _diagnostics_enabled(state) else None)
    state.diagnostics = diagnostics
    state.validation_diagnostics = None
    try:
        with _data_rng(_data_rng_state()), torch.random.fork_rng(devices=devices):
            for batch in state.val_loader:
                with _autocast(state):
                    losses, _ = compute_loss(state, batch, alpha)
                for i, name in enumerate(OBJECTIVES):
                    totals[i] += losses[f"loss_{name}_sum"]
                    counts[i] += losses[f"count_{name}"]
    finally:
        state.diagnostics = previous_diagnostics
        state.model.train(was_training)
        state.val_loader.generator.set_state(generator_state)
    if diagnostics is not None:
        state.validation_diagnostics = diagnostics.summarize(state.settings, _sum(diagnostics.pack()))
    return _metrics(state, totals, counts)


def run_training(settings=None, on_step=None):
    settings = {name: value for name, value in vars(config).items() if name.isupper()} | (settings or {})
    runtime = init_runtime(settings)
    logger, state = None, None
    try:
        logger = setup_logging(label=f'rank-{runtime.rank}', level=settings['LOG_LEVEL'],
                               console_level=None if runtime.rank == 0 else 'WARNING')
        logger.info('initializing mode=%s action=%s device=%s world_size=%d source=%s',
                    settings['TRAINING_MODE'], 'resume' if settings['RESUME_CHECKPOINT'] else 'train',
                    runtime.device, runtime.world_size, settings['RESUME_CHECKPOINT'] or settings['INIT_CHECKPOINT'])
        state = prepare_training(settings, runtime)
        settings = state.settings
        close_logging(logger)
        state.logger = logger = checkpoint._each_rank(lambda: setup_logging(
            state.run_dir, label=f'rank-{runtime.rank}', level=settings['LOG_LEVEL'],
                               console_level=None if runtime.rank == 0 else 'WARNING'))
        state.tb = checkpoint._each_rank(lambda: TensorBoardLogger(
            state.run_dir, enabled=settings['TENSORBOARD_ENABLED'] and runtime.rank == 0,
            resume_step=state.progress['global_step'] if settings['RESUME_CHECKPOINT'] else None,
            flush_secs=settings['TENSORBOARD_FLUSH_SECS']))
        state.tb.log_config(settings)
        logger.info('run=%s mode=%s action=%s device=%s world_size=%d precision=%s progress=%s',
                    state.run_dir, settings['TRAINING_MODE'], 'resume' if settings['RESUME_CHECKPOINT'] else 'train',
                    runtime.device, runtime.world_size, settings['AMP_DTYPE'] if settings['AMP_ENABLED'] else 'float32',
                    state.progress)
        logger.info('model parameters=%d trainable=%d vision_mock=%s; train_samples=%d val_samples=%d '
                    'train_batches=%d val_batches=%d batch_per_rank=%d accumulation=%d workers=%d',
                    sum(p.numel() for p in state.model.parameters()),
                    sum(p.numel() for p in state.model.parameters() if p.requires_grad),
                    settings['VISION_MOCK'], len(state.train_loader.dataset), len(state.val_loader.dataset),
                    len(state.train_loader), len(state.val_loader), settings['BATCH_SIZE'],
                    settings['ACCUMULATION_STEPS'], settings['NUM_WORKERS'])
        logger.debug('effective settings=%s', settings)

        def training_events(metrics=None, final=False):
            step = state.progress['global_step']
            def due(name):
                return settings[name] > 0 and step > 0 and step % settings[name] == 0
            if metrics is not None and due('LOG_EVERY_STEPS'):
                logger.info('step=%d train=%s stats=%s', step, metrics, state.step_stats)
                state.tb.log_scalars('train', metrics | state.step_stats, step)
            should_validate = (due('VAL_EVERY_STEPS') or (final and settings['VAL_EVERY_STEPS'] > 0))
            should_validate = should_validate and state.events.get('last_validation_step') != step
            best = False
            if should_validate:
                started = time.perf_counter()
                logger.info('step=%d validation started', step)
                result = validate(state)
                state.events['last_validation_step'] = step
                logger.info('step=%d validation=%s elapsed=%.3fs', step, result, time.perf_counter() - started)
                state.tb.log_scalars('validation', result | dict(seconds=time.perf_counter() - started), step)
                if state.validation_diagnostics is not None:
                    state.tb.log_diagnostics('validation', state.validation_diagnostics, step)
                    state.tb.log_examples(state.validation_diagnostics['samples'], state.tokenizer,
                                          settings['NORMALIZATION'], step)
                best = math.isfinite(result['loss']) and result['loss'] < state.events.get('best_val_loss', math.inf)
                if best:
                    state.events['best_val_loss'] = result['loss']
                    logger.info('step=%d new best validation loss=%.6f', step, result['loss'])
                elif not math.isfinite(result['loss']):
                    logger.warning('step=%d nonfinite validation loss; best checkpoint unchanged', step)
            if final or due('SAVE_EVERY_STEPS') or should_validate:
                checkpoint.save_checkpoint(state, best=best)

        def step_event(state, metrics):
            if state.step_diagnostics is not None:
                state.tb.log_diagnostics('train', state.step_diagnostics, state.progress['global_step'])
            training_events(metrics)
            if on_step is not None:
                on_step(state, metrics)

        for epoch in range(state.progress['epoch'], settings["EPOCHS"]):
            metrics = train_one_epoch(state, epoch, on_step=step_event)
            state.tb.log_scalars('epoch', metrics | dict(epoch=epoch + 1), state.progress['global_step'])
        training_events(final=True)
        logger.info('training completed run=%s progress=%s best_val_loss=%s',
                    state.run_dir, state.progress, state.events.get('best_val_loss'))
        return state
    except BaseException:
        if logger:
            logger.exception('training interrupted or failed')
        raise
    finally:
        if state is not None and hasattr(state, 'tb'):
            state.tb.close()
        if logger:
            close_logging(logger)
        if runtime.owns_process_group:
            dist.destroy_process_group()


if __name__ == "__main__":
    run_training()
