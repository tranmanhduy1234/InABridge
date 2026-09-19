from pathlib import Path
import math
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer
from src.data.dataloaderph1 import (
    VLMDatasetStage1, VLMDataCollator, build_dataloader, build_transform,
)
from src import config
from src.trainingph1.model import ModelStage1
from src.queue.ema import EMA

def get_dataloader():
    root = Path(__file__).resolve().parents[2]
    image_dir = root / config.IMAGE_DIR
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    splits = (
        (True, config.JSON_PATH_TRAIN),
        (False, config.JSON_PATH_VAL),
    )
    for _, json_path in splits:
        if not (root / json_path).is_file():
            raise FileNotFoundError(root / json_path)

    tokenizer = AutoTokenizer.from_pretrained(
        config.TOKENIZER_MODEL_ID, use_fast=config.TOKENIZER_USE_FAST,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no PAD/EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    collator = VLMDataCollator(
        tokenizer, max_length=config.MAX_LENGTH,
        padding=config.TOKENIZER_PADDING, truncation=config.TOKENIZER_TRUNCATION,
    )
    loaders = []
    for is_training, json_path in splits:
        transform = build_transform(
            image_size=config.IMAGE_SIZE, is_training=is_training,
            normalization=config.NORMALIZATION,
            crop_scale=config.CROP_SCALE, crop_ratio=config.CROP_RATIO,
            interpolation=config.INTERPOLATION, antialias=config.ANTIALIAS,
            horizontal_flip_probability=config.HORIZONTAL_FLIP_PROBABILITY,
            color_jitter=config.COLOR_JITTER,
            color_jitter_probability=config.COLOR_JITTER_PROBABILITY,
            grayscale_probability=config.GRAYSCALE_PROBABILITY,
            blur_kernel_size=config.BLUR_KERNEL_SIZE,
            blur_sigma=config.BLUR_SIGMA, blur_probability=config.BLUR_PROBABILITY,
        )
        dataset = VLMDatasetStage1(
            json_path=root / json_path, image_dir=image_dir,
            is_training=is_training, cache_dir=root / config.CACHE_DIR,
            chunk_size=config.CACHE_CHUNK_SIZE, transform=transform,
        )
        loaders.append(build_dataloader(
            dataset=dataset, batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS, drop_last=config.DROP_LAST,
            shuffle=config.SHUFFLE if is_training else False,
            pin_memory=config.PIN_MEMORY, persistent_workers=config.PERSISTENT_WORKERS,
            prefetch_factor=config.PREFETCH_FACTOR, collate_fn=collator,
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
        normalize_eps=config.NORMALIZE_EPS,
    ).train()
    momentum_model = EMA(online_model, momentum=config.MOMENTUM)
    return online_model, momentum_model

def get_optimizer():
    pass

def get_criterion():
    pass

def get_scheduler(optimizer, warmup_steps: int, total_steps: int,
                  min_lr_ratio: float = 0.05):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return LambdaLR(optimizer, lr_lambda)