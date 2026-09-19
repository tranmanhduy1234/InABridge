import sqlite3
from pathlib import Path

import ijson
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoTokenizer

from src import config as cfg

def build_transform(image_size, is_training, normalization,
                    crop_scale, crop_ratio, interpolation,
                    antialias, horizontal_flip_probability,
                    color_jitter, color_jitter_probability, grayscale_probability,
                    blur_kernel_size, blur_sigma, blur_probability):
    interp = dict(interpolation=InterpolationMode(interpolation), antialias=antialias)

    aug = [
        transforms.RandomResizedCrop(image_size, scale=crop_scale, ratio=crop_ratio, **interp),
        transforms.RandomHorizontalFlip(horizontal_flip_probability),
        transforms.RandomApply([transforms.ColorJitter(*color_jitter)], p=color_jitter_probability),
        transforms.RandomGrayscale(grayscale_probability),
        transforms.RandomApply([transforms.GaussianBlur(blur_kernel_size, blur_sigma)], p=blur_probability),
    ] if is_training else [
        transforms.Resize((image_size, image_size), **interp),
    ]
    aug.append(transforms.ToTensor())
    if normalization is not None:
        aug.append(transforms.Normalize(*normalization))

    return transforms.Compose(aug)

def build_cache(json_path, cache_path, chunk_size):
    if cache_path.exists():
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp")
    if tmp.exists():
        tmp.unlink()

    conn = sqlite3.connect(tmp)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("""
        CREATE TABLE samples (
            id INTEGER PRIMARY KEY,
            image_name TEXT NOT NULL,
            text TEXT NOT NULL
        )
    """)
    with open(json_path, "rb") as f:
        rows = []
        for i, x in enumerate(ijson.items(f, "item")):
            rows.append((i, x["image_name"], x["text"]))
            if len(rows) >= chunk_size:
                conn.executemany("INSERT INTO samples VALUES (?, ?, ?)", rows)
                rows.clear()
        if rows:
            conn.executemany("INSERT INTO samples VALUES (?, ?, ?)", rows)
    conn.commit()
    conn.close()
    tmp.rename(cache_path)

class VLMDatasetStage1(Dataset):
    def __init__(self, json_path, image_dir, is_training, cache_dir, chunk_size, transform):
        json_path = Path(json_path)
        self.image_dir = Path(image_dir)
        self.is_training = is_training
        self.cache_path = Path(cache_dir) / ("train.sqlite" if is_training else "validation.sqlite")
        self.transform = transform
        self.conn = None
        build_cache(json_path=json_path, cache_path=self.cache_path, chunk_size=chunk_size)

        with sqlite3.connect(self.cache_path) as conn:
            self.length = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]

    def __len__(self):
        return self.length

    def _db(self):
        if self.conn is None:
            self.conn = sqlite3.connect(
                f"file:{self.cache_path}?mode=ro", uri=True, check_same_thread=False
            )
        return self.conn

    def __getitem__(self, idx):
        row = self._db().execute(
            "SELECT image_name, text FROM samples WHERE id=?", (int(idx),)
        ).fetchone()
        if row is None:
            raise IndexError(idx)
        image_name, text = row
        with Image.open(self.image_dir / image_name) as img:
            image = self.transform(img.convert("RGB"))
        return image, text

    def __getstate__(self):
        state = self.__dict__.copy()
        state["conn"] = None
        return state

class VLMDataCollator:
    def __init__(self, tokenizer, max_length, padding, truncation):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
    def __call__(self, batch):
        images, texts = zip(*batch)
        tokens = self.tokenizer(
            texts, padding=self.padding, truncation=self.truncation,
            max_length=self.max_length, return_tensors="pt"
        )
        return {
            "images": torch.stack(images),
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"].bool()
        }

def build_dataloader(dataset: VLMDatasetStage1, batch_size, num_workers,
                     drop_last, shuffle, pin_memory, persistent_workers,
                     prefetch_factor, collate_fn):
    kwargs = dict(
        dataset=dataset, batch_size=batch_size, shuffle=dataset.is_training if shuffle is None else shuffle,
        drop_last=drop_last if dataset.is_training else False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available() if pin_memory is None else pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        collate_fn=collate_fn
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)

def main():
    import matplotlib.pyplot as plt
    root = Path(__file__).resolve().parents[2]
    transforms_aug = build_transform(
        image_size=cfg.IMAGE_SIZE,
        is_training=cfg.IS_TRAINING,
        normalization=cfg.NORMALIZATION,
        crop_scale=cfg.CROP_SCALE,
        crop_ratio=cfg.CROP_RATIO,
        interpolation=cfg.INTERPOLATION,
        antialias=cfg.ANTIALIAS,
        horizontal_flip_probability=cfg.HORIZONTAL_FLIP_PROBABILITY,
        color_jitter=cfg.COLOR_JITTER,
        color_jitter_probability=cfg.COLOR_JITTER_PROBABILITY,
        grayscale_probability=cfg.GRAYSCALE_PROBABILITY,
        blur_kernel_size=cfg.BLUR_KERNEL_SIZE,
        blur_sigma=cfg.BLUR_SIGMA,
        blur_probability=cfg.BLUR_PROBABILITY,
    )
    dataset = VLMDatasetStage1(
        json_path=root / (cfg.JSON_PATH_TRAIN if cfg.IS_TRAINING else cfg.JSON_PATH_VAL),
        image_dir=root / cfg.IMAGE_DIR,
        is_training=cfg.IS_TRAINING,
        cache_dir=root / cfg.CACHE_DIR,
        chunk_size=cfg.CACHE_CHUNK_SIZE,
        transform=transforms_aug,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.TOKENIZER_MODEL_ID, use_fast=cfg.TOKENIZER_USE_FAST)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no PAD/EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    collator = VLMDataCollator(
        tokenizer, max_length=cfg.MAX_LENGTH,
        padding=cfg.TOKENIZER_PADDING, truncation=cfg.TOKENIZER_TRUNCATION,
    )
    loader = build_dataloader(
        dataset=dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        drop_last=cfg.DROP_LAST,
        shuffle=cfg.SHUFFLE,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=cfg.PERSISTENT_WORKERS,
        prefetch_factor=cfg.PREFETCH_FACTOR,
        collate_fn=collator,
    )
    if len(dataset) == 0:
        raise ValueError("Demo dataset is empty")
    for batch in loader:
        print({name: tuple(value.shape) for name, value in batch.items()})

        for i in range(min(cfg.DEMO_NUM_IMAGES, len(batch["images"]))):
            image = batch["images"][i].permute(1, 2, 0)
            caption = tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True)
            print(batch["input_ids"][i])
            print(batch["attention_mask"][i])
            print()
            plt.figure(figsize=cfg.DEMO_FIGURE_SIZE)
            plt.imshow(image)
            plt.title(caption, fontsize=cfg.DEMO_TITLE_FONT_SIZE, wrap=True)
            plt.axis("off")
            plt.tight_layout()
            plt.show()
            plt.close()

if __name__ == "__main__":
    main()
