from pathlib import Path
import math
from types import SimpleNamespace
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer
from src.data.dataloaderph1 import (
    VLMDatasetStage1, VLMDataCollator, build_dataloader, build_transform
)
from src import config
from src.trainingph1.model import ModelStage1
from src.queue.ema import EMA
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
            chunk_size=cfg.CACHE_CHUNK_SIZE, transform=transform,
        )
        loaders.append(build_dataloader(
            dataset=dataset, batch_size=cfg.BATCH_SIZE,
            num_workers=cfg.NUM_WORKERS, drop_last=cfg.DROP_LAST,
            shuffle=cfg.SHUFFLE if is_training else False,
            pin_memory=cfg.PIN_MEMORY, persistent_workers=cfg.PERSISTENT_WORKERS,
            prefetch_factor=cfg.PREFETCH_FACTOR, collate_fn=collator,
        ))
    return loaders[0], loaders[1]

def get_model():
    online_model = ModelStage1(
        bert_name=config.BERT_MODEL_ID,
        return_layer=config.VISION_RETURN_LAYER,
        model_vision_id=config.VISION_MODEL_ID,
        num_queries=config.NUM_QUERIES,
        cross_attn_every=config.CROSS_ATTN_EVERY,
        hidden_dim=config.HIDDEN_DIM,
        itc_dim=config.ITC_DIM,
        normalize_eps=config.NORMALIZE_EPS
    ).train()
    momentum_model = EMA(online_model, momentum=config.MOMENTUM)
    return online_model, momentum_model

def get_criterion(epoch_progress=0.0):
    if epoch_progress < 0 or config.PSEUDO_WARMUP_EPOCHS < 0:
        raise ValueError("Epoch progress and pseudo warmup must be non-negative")
    progress = min(epoch_progress / config.PSEUDO_WARMUP_EPOCHS, 1.0) if config.PSEUDO_WARMUP_EPOCHS else 1.0
    return Stage1Criterion(
        config.ITC_WEIGHT, config.ITM_WEIGHT, config.ITG_WEIGHT,
        config.PSEUDO_WEIGHT * progress,
    )

def get_optimizer(model):
    return AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.LR, weight_decay=config.WEIGHT_DECAY,
        betas=config.ADAM_BETAS, eps=config.ADAM_EPS,
    )

def get_scheduler(optimizer, warmup_steps=None, total_steps=None,
                  min_lr_ratio=None, *, batches_per_epoch=None):
    if batches_per_epoch is not None:
        if batches_per_epoch <= 0 or config.ACCUMULATION_STEPS < 1:
            raise ValueError("Batch count and accumulation steps must be positive")
        steps_per_epoch = math.ceil(batches_per_epoch / config.ACCUMULATION_STEPS)
        if warmup_steps is None:
            warmup_steps = config.WARMUP_EPOCHS * steps_per_epoch
        if total_steps is None:
            total_steps = config.EPOCHS * steps_per_epoch
    if total_steps is None or warmup_steps is None or not 0 <= warmup_steps < total_steps:
        raise ValueError("Require 0 <= warmup_steps < total_steps")
    if any(group["lr"] <= 0 for group in optimizer.param_groups):
        raise ValueError("Initial optimizer LR must be positive")
    ratios = ([config.MIN_LR / group["lr"] for group in optimizer.param_groups]
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

def validate():
    pass

def train_one_epoch():
    pass

def run_training():
    seed_everything()
