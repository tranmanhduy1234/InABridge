from itertools import islice
from contextlib import contextmanager
from pathlib import Path
import math
import random
from types import SimpleNamespace
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer
from src.data.dataloaderph1 import (
    VLMDatasetStage1, VLMDataCollator, build_dataloader, build_transform
)
from src import config
from src.trainingph1.model import ModelStage1
from src.queue.ema import EMA
from src.queue.moco import MoCoQueue
from src.utils.checkpoint import create_run, load_pretrained, load_checkpoint
from src.losses.lossph1 import Stage1Criterion
from src.utils.seed import seed_everything

def get_dataloader(tokenizer=None, settings=None):
    if settings is not None and tokenizer is None:
        raise ValueError("Pass the checkpoint tokenizer together with its settings")
    cfg = SimpleNamespace(**(vars(config) | (settings or {})))
    root = Path(__file__).resolve().parents[2]
    image_dir = root / cfg.IMAGE_DIR
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    splits = (
        (True, cfg.JSON_PATH_TRAIN),
        (False, cfg.JSON_PATH_VAL)
    )
    for _, json_path in splits:
        if not (root / json_path).is_file():
            raise FileNotFoundError(root / json_path)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.TOKENIZER_MODEL_ID, use_fast=cfg.TOKENIZER_USE_FAST
        )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no PAD/EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    collator = VLMDataCollator(
        tokenizer, max_length = cfg.MAX_LENGTH,
        padding=cfg.TOKENIZER_PADDING, truncation=cfg.TOKENIZER_TRUNCATION
    )
    loaders = []
    for is_training, json_path in splits:
        transform = build_transform(
            image_size=cfg.IMAGE_SIZE, is_training=is_training,
            normalization=cfg.NORMALIZATION,
            crop_scale=cfg.CROP_SCALE, crop_ratio=cfg.CROP_RATIO,
            interpolation=cfg.INTERPOLATION, antialias=cfg.ANTIALIAS,
            horizontal_flip_probability=cfg.HORIZONTAL_FLIP_PROBABILITY,
            color_jitter=cfg.COLOR_JITTER,
            color_jitter_probability=cfg.COLOR_JITTER_PROBABILITY,
            grayscale_probability=cfg.GRAYSCALE_PROBABILITY,
            blur_kernel_size=cfg.BLUR_KERNEL_SIZE,
            blur_sigma=cfg.BLUR_SIGMA, blur_probability=cfg.BLUR_PROBABILITY,
        )
        dataset = VLMDatasetStage1(
            json_path=root / json_path, image_dir=image_dir,
            is_training=is_training, cache_dir=root / cfg.CACHE_DIR,
            chunk_size=cfg.CACHE_CHUNK_SIZE, transform=transform, dataset_namespace=cfg.DATASET_NAMESPACE,
        )
        loaders.append(build_dataloader(
            dataset=dataset, batch_size=cfg.BATCH_SIZE,
            num_workers=cfg.NUM_WORKERS, drop_last=cfg.DROP_LAST,
            shuffle=cfg.SHUFFLE if is_training else False,
            # Recreate training workers so epoch seeds also reproduce their augmentations.
            pin_memory=cfg.PIN_MEMORY, persistent_workers=cfg.PERSISTENT_WORKERS and not is_training,
            prefetch_factor=cfg.PREFETCH_FACTOR, collate_fn=collator,
        ))
    return loaders[0], loaders[1]

def get_model(settings=None):
    cfg = SimpleNamespace(**(vars(config) | (settings or {})))
    online_model = ModelStage1(
        bert_name=cfg.BERT_MODEL_ID,
        return_layer=cfg.VISION_RETURN_LAYER,
        model_vision_id=cfg.VISION_MODEL_ID,
        num_queries=cfg.NUM_QUERIES,
        cross_attn_every=cfg.CROSS_ATTN_EVERY,
        hidden_dim=cfg.HIDDEN_DIM,
        itc_dim=cfg.ITC_DIM,
        normalize_eps=cfg.NORMALIZE_EPS
    ).train()
    momentum_model = EMA(online_model.itc_encoder, momentum=cfg.MOMENTUM)
    return online_model, momentum_model

def get_criterion():
    return Stage1Criterion()

def get_pseudo_weight(epoch_progress, settings=None):
    cfg = SimpleNamespace(**(vars(config) | (settings or {})))
    if epoch_progress < 0 or cfg.PSEUDO_WARMUP_EPOCHS < 0:
        raise ValueError("Epoch progress and pseudo warmup must be non-negative")
    progress = min(epoch_progress / cfg.PSEUDO_WARMUP_EPOCHS, 1.0) if cfg.PSEUDO_WARMUP_EPOCHS else 1.0
    return cfg.PSEUDO_WEIGHT * progress

def get_optimizer(model, settings=None):
    cfg = SimpleNamespace(**(vars(config) | (settings or {})))
    return AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
        betas=cfg.ADAM_BETAS, eps=cfg.ADAM_EPS,
    )

def get_scheduler(optimizer, warmup_steps=None, total_steps=None,
                  min_lr_ratio=None, *, batches_per_epoch=None, settings=None):
    cfg = SimpleNamespace(**(vars(config) | (settings or {})))
    if batches_per_epoch is not None:
        if batches_per_epoch <= 0 or cfg.ACCUMULATION_STEPS < 1:
            raise ValueError("Batch count and accumulation steps must be positive")
        steps_per_epoch = math.ceil(batches_per_epoch / cfg.ACCUMULATION_STEPS)
        if warmup_steps is None:
            warmup_steps = cfg.WARMUP_EPOCHS * steps_per_epoch
        if total_steps is None:
            total_steps = cfg.EPOCHS * steps_per_epoch
    if total_steps is None or warmup_steps is None or not 0 <= warmup_steps < total_steps:
        raise ValueError("Require 0 <= warmup_steps < total_steps")
    if any(group["lr"] <= 0 for group in optimizer.param_groups):
        raise ValueError("Initial optimizer LR must be positive")
    ratios = ([cfg.MIN_LR / group["lr"] for group in optimizer.param_groups]
              if min_lr_ratio is None else [min_lr_ratio] * len(optimizer.param_groups))
    if any(not 0 <= ratio <= 1 for ratio in ratios):
        raise ValueError("Minimum LR must be between zero and the initial LR")
    def lr_lambda(step, ratio):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return ratio + (1.0 - ratio) * cosine
    return LambdaLR(optimizer, [lambda step, ratio=ratio: lr_lambda(step, ratio) for ratio in ratios])

def prepare_training():
    if config.INIT_CHECKPOINT and config.RESUME_CHECKPOINT:
        raise ValueError("INIT_CHECKPOINT and RESUME_CHECKPOINT are mutually exclusive")
    source = config.RESUME_CHECKPOINT or config.INIT_CHECKPOINT
    settings = {k: v for k, v in vars(config).items() if k.isupper()}
    if source:
        model, tokenizer, saved_settings = load_pretrained(source)
        if config.RESUME_CHECKPOINT:
            settings.update(saved_settings)
            settings["DEVICE"] = config.DEVICE
    cfg = SimpleNamespace(**settings)
    seed_everything(cfg.SEED)
    if not source:
        model, ema = get_model(settings)
        tokenizer = AutoTokenizer.from_pretrained(cfg.TOKENIZER_MODEL_ID, use_fast=cfg.TOKENIZER_USE_FAST)
    model = model.to(cfg.DEVICE).train()
    ema = (EMA(model.itc_encoder, cfg.MOMENTUM) if source else ema).to(cfg.DEVICE)
    train_loader, val_loader = get_dataloader(tokenizer, settings)
    if config.RESUME_CHECKPOINT:
        if saved_settings.get("BATCHES_PER_EPOCH") != len(train_loader):
            raise ValueError("Resume requires the saved BATCHES_PER_EPOCH to match the training loader")
        if saved_settings.get("DATA_RNG_VERSION") != 1:
            raise ValueError("Checkpoint predates reproducible data iteration; use INIT_CHECKPOINT")
    settings.update(BATCHES_PER_EPOCH=len(train_loader), DATA_RNG_VERSION=1)
    optimizer = get_optimizer(model, settings)
    scheduler = get_scheduler(optimizer, batches_per_epoch=len(train_loader), settings=settings)
    queue = get_queue(settings, model).to(cfg.DEVICE)
    if cfg.AMP_DTYPE not in {"float16", "bfloat16"}:
        raise ValueError("AMP_DTYPE must be float16 or bfloat16")
    scaler = (torch.amp.GradScaler(torch.device(cfg.DEVICE).type)
              if cfg.AMP_ENABLED and cfg.AMP_DTYPE == "float16" else None)
    components = dict(optimizer=optimizer, scheduler=scheduler, ema=ema, queue=queue, scaler=scaler)
    progress = dict(global_step=0, epoch=0, next_batch=0, data_state=None)
    if config.RESUME_CHECKPOINT:
        progress = load_checkpoint(source, model, **components)
        run_dir = Path(__file__).resolve().parents[2] / Path(source).parent
    else:
        run_dir = create_run(model, tokenizer, settings=settings, parent_checkpoint=source)
    criterion = get_criterion()
    return SimpleNamespace(model=model, tokenizer=tokenizer, settings=settings,
                           train_loader=train_loader, val_loader=val_loader, criterion=criterion,
                           run_dir=run_dir, progress=progress, **components)

def get_queue(settings=None, model=None):
    cfg = SimpleNamespace(**(vars(config) | (settings or {})))
    num_queries = model.qformer_model.num_queries if model is not None else cfg.NUM_QUERIES
    dim = model.itc_encoder.query_proj.out_features if model is not None else cfg.ITC_DIM
    return MoCoQueue(cfg.QUEUE_SIZE, num_queries, dim)

def hepler_compute_loss(state, batch, pseudo_weight):
    cfg = SimpleNamespace(**state.settings)
    images = batch["images"].to(cfg.DEVICE, non_blocking=True)
    image_ids = batch["image_ids"].to(cfg.DEVICE, non_blocking=True)
    input_ids = batch["input_ids"].to(cfg.DEVICE, non_blocking=True)
    attn_mask = batch["attention_mask"].to(cfg.DEVICE, non_blocking=True)
    model, criterion = state.model, state.criterion
    image_features = model.encode_image(images)
    online = model(image_features, input_ids, attn_mask, "itc")
    with torch.no_grad():
        momentum = state.ema.model(image_features, model.encode_text(input_ids), attn_mask)
    momentum["image_ids"] = image_ids
    image_queue, text_queue, queue_ids = state.queue.get()
    itc = criterion.get_itc_loss(
        online["image_features"], online["text_features"],
        momentum["image_features"], momentum["text_features"],
        image_queue, text_queue, cfg.TEMPERATURE, pseudo_weight, image_ids, queue_ids,
    )
    losses = {"loss_itc": itc["loss_itc"]}
    batch_size = input_ids.size(0)
    same_image = image_ids[:, None] == image_ids[None, :]
    if (~same_image).any():
        with torch.no_grad(), torch.autocast(device_type=images.device.type, enabled=False):
            similarity = torch.einsum("bqd,kd->bkq", online["image_features"].float(),
                                      online["text_features"].float()).amax(-1) / cfg.TEMPERATURE
            similarity.masked_fill_(same_image, -torch.inf)
            negative_text = torch.multinomial(similarity.softmax(-1), 1).squeeze(1)
            negative_image = torch.multinomial(similarity.T.softmax(-1), 1).squeeze(1)
        losses["loss_itm"] = 0
        for features, tokens, mask, label in (
            (image_features, input_ids, attn_mask, 1),
            (image_features[negative_image], input_ids, attn_mask, 0),
            (image_features, input_ids[negative_text], attn_mask[negative_text], 0),
        ):
            itm = model(features, tokens, mask, "itm")
            labels = torch.full((batch_size,), label, device=images.device)
            losses["loss_itm"] += criterion.get_itm_loss(itm["itm_logits"], labels)["loss_itm"] / 3
    else:
        # No negative exists when every caption belongs to the same image.
        losses["loss_itm"] = itc["loss_itc"] * 0
    itg = model(image_features, input_ids, attn_mask, "itg")
    labels = input_ids[:, 1:].masked_fill(~attn_mask[:, 1:].bool(), -100)
    losses.update(criterion.get_itg_loss(itg["itg_logits"], labels))
    losses["loss"] = (cfg.ITC_WEIGHT * losses["loss_itc"]
                      + cfg.ITM_WEIGHT * losses["loss_itm"]
                      + cfg.ITG_WEIGHT * losses["loss_itg"])
    return losses, momentum

def _loss_counts(batch):
    size = batch["input_ids"].size(0)
    has_negative = (batch["image_ids"] != batch["image_ids"][0]).any().item()
    return dict(loss_itc=size, loss_itm=3 * size if has_negative else 0,
                loss_itg=batch["attention_mask"][:, 1:].sum().item())

@torch.no_grad()
def validate(state, pseudo_weight=None):
    cfg = SimpleNamespace(**state.settings)
    pseudo_weight = cfg.PSEUDO_WEIGHT if pseudo_weight is None else pseudo_weight
    totals = dict(loss_itc=0.0, loss_itm=0.0, loss_itg=0.0)
    counts = dict.fromkeys(totals, 0)
    was_training = state.model.training
    state.model.eval()
    try:
        with torch.random.fork_rng():
            for batch in state.val_loader:
                with torch.autocast(device_type=torch.device(cfg.DEVICE).type,
                                    dtype=getattr(torch, cfg.AMP_DTYPE), enabled=cfg.AMP_ENABLED):
                    losses, _ = hepler_compute_loss(state, batch, pseudo_weight)
                sizes = _loss_counts(batch)
                for name, size in sizes.items():
                    totals[name] += losses[name].item() * size
                    counts[name] += size
    finally:
        state.model.train(was_training)
    if not counts["loss_itc"]:
        raise ValueError("Validation loader is empty")
    losses = {name: total / max(counts[name], 1) for name, total in totals.items()}
    losses["loss"] = (cfg.ITC_WEIGHT * losses["loss_itc"]
                      + cfg.ITM_WEIGHT * losses["loss_itm"]
                      + cfg.ITG_WEIGHT * losses["loss_itg"])
    return losses

def _data_rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state()

@contextmanager
def _data_rng(rng):
    previous = _data_rng_state()
    try:
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        yield
    finally:
        random.setstate(previous[0])
        np.random.set_state(previous[1])
        torch.set_rng_state(previous[2])

def _training_batches(loader, seed, start_batch):
    if not 0 <= start_batch <= len(loader):
        raise ValueError("Resume batch is outside the training loader")
    seed %= 2**32
    rng = (random.Random(seed).getstate(), np.random.RandomState(seed).get_state(),
           torch.Generator().manual_seed(seed).get_state())
    # Replay data RNG independently of dropout and negative sampling in the model.
    with _data_rng(rng):
        batches = iter(loader)
        for _ in range(start_batch):
            next(batches)
        rng = _data_rng_state()
    for _ in range(start_batch, len(loader)):
        with _data_rng(rng):
            batch = next(batches)
            rng = _data_rng_state()
        yield batch

def train_one_epoch(state, epoch):
    cfg = SimpleNamespace(**state.settings)
    if cfg.ACCUMULATION_STEPS < 1:
        raise ValueError("ACCUMULATION_STEPS must be positive")
    model, optimizer, scaler = state.model, state.optimizer, state.scaler
    model.train()
    state.ema.eval()
    optimizer.zero_grad(set_to_none=True)
    num_batches = len(state.train_loader)
    start_batch = state.progress["next_batch"] if epoch == state.progress["epoch"] else 0
    weights = dict(loss_itc=cfg.ITC_WEIGHT, loss_itm=cfg.ITM_WEIGHT,
                   loss_itg=cfg.ITG_WEIGHT)
    totals, counts = dict.fromkeys(weights, 0.0), dict.fromkeys(weights, 0)
    if start_batch != num_batches and start_batch % cfg.ACCUMULATION_STEPS:
        raise ValueError("Resume must start at an accumulation boundary")
    batches = _training_batches(state.train_loader, cfg.SEED + epoch, start_batch)
    for group_start in range(start_batch, num_batches, cfg.ACCUMULATION_STEPS):
        group = list(islice(batches, cfg.ACCUMULATION_STEPS))
        sizes = [_loss_counts(batch) for batch in group]
        group_counts = {name: sum(size[name] for size in sizes) for name in weights}
        pending_keys = []
        for offset, (batch, size) in enumerate(zip(group, sizes)):
            pseudo_weight = get_pseudo_weight(epoch + (group_start + offset) / num_batches, state.settings)
            with torch.autocast(device_type=torch.device(cfg.DEVICE).type,
                                dtype=getattr(torch, cfg.AMP_DTYPE), enabled=cfg.AMP_ENABLED):
                losses, momentum = hepler_compute_loss(state, batch, pseudo_weight)
                loss = sum(weights[name] * losses[name] * size[name] / max(group_counts[name], 1)
                           for name in weights)
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()
            pending_keys.append(momentum)
            for name in weights:
                totals[name] += losses[name].detach().item() * size[name]
                counts[name] += size[name]
        if scaler is not None:
            scaler.unscale_(optimizer)
        parameters = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
        updated = bool(parameters) and torch.stack([p.grad.isfinite().all() for p in parameters]).all().item()
        if updated and cfg.MAX_GRAD_NORM is not None:
            norm = torch.nn.utils.clip_grad_norm_(parameters, cfg.MAX_GRAD_NORM)
            updated = norm.isfinite().item()
        if updated:
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
        if scaler is not None:
            scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if updated:
            state.scheduler.step()
            state.ema.update(model.itc_encoder)
            state.queue.enqueue(
                torch.cat([keys["image_features"] for keys in pending_keys]).to(state.queue.image.dtype),
                torch.cat([keys["text_features"] for keys in pending_keys]).to(state.queue.text.dtype),
                torch.cat([keys["image_ids"] for keys in pending_keys]),
            )
            state.progress["global_step"] += 1
        state.progress.update(epoch=epoch, next_batch=group_start + len(group))
    state.progress.update(epoch=epoch + 1, next_batch=0)
    losses = {name: total / max(counts[name], 1) for name, total in totals.items()}
    losses["loss"] = sum(weights[name] * losses[name] for name in weights)
    return losses

def run_training():
    state = prepare_training()
    for epoch in range(state.progress["epoch"], state.settings["EPOCHS"]):
        train_one_epoch(state, epoch)
    return state

if __name__=="__main__":
    run_training()