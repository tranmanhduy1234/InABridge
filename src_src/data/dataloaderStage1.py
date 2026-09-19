import sqlite3
from pathlib import Path

import ijson
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoTokenizer

from src_src import config_model as cfg

def build_transform(size=cfg.IMAGE_SIZE, train=cfg.DATA_TRAIN, normalization=cfg.DATA_NORMALIZATION, *,
                    crop_scale=cfg.CROP_SCALE, crop_ratio=cfg.CROP_RATIO,
                    interpolation=cfg.IMAGE_INTERPOLATION, antialias=cfg.IMAGE_ANTIALIAS,
                    horizontal_flip_probability=cfg.HORIZONTAL_FLIP_PROBABILITY,
                    color_jitter=cfg.COLOR_JITTER, color_jitter_probability=cfg.COLOR_JITTER_PROBABILITY,
                    grayscale_probability=cfg.GRAYSCALE_PROBABILITY,
                    blur_kernel_size=cfg.BLUR_KERNEL_SIZE, blur_sigma=cfg.BLUR_SIGMA,
                    blur_probability=cfg.BLUR_PROBABILITY):
    interp = dict(interpolation=InterpolationMode(interpolation), antialias=antialias)

    aug = [
        transforms.RandomResizedCrop(size, scale=crop_scale, ratio=crop_ratio, **interp),
        transforms.RandomHorizontalFlip(horizontal_flip_probability),
        transforms.RandomApply([transforms.ColorJitter(*color_jitter)], p=color_jitter_probability),
        transforms.RandomGrayscale(grayscale_probability),
        transforms.RandomApply([transforms.GaussianBlur(blur_kernel_size, blur_sigma)], p=blur_probability),
    ] if train else [
        transforms.Resize((size, size), **interp)
    ]

    aug.append(transforms.ToTensor())
    if normalization is not None:
        aug.append(transforms.Normalize(*normalization))

    return transforms.Compose(aug)

def build_cache(json_path, cache_path, chunk_size=cfg.CACHE_CHUNK_SIZE):
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
    def __init__(self, json_path, image_dir, *, train=cfg.DATA_TRAIN,
                 cache_dir=cfg.DATA_CACHE_DIR, transform=None):
        json_path = Path(json_path)
        self.image_dir = Path(image_dir)
        self.train = train
        self.cache_path = Path(cache_dir) / ("train.sqlite" if train else "validation.sqlite")
        self.transform = transform if transform is not None else build_transform(train=train)
        self.conn = None
        build_cache(json_path, self.cache_path)

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
    def __init__(self, tokenizer, max_length=cfg.MAX_INSTRUCTION_LENGTH, padding=cfg.TOKENIZER_PADDING,
                 truncation=cfg.TOKENIZER_TRUNCATION):
        self.tokenizer, self.max_length = tokenizer, max_length
        self.padding, self.truncation = padding, truncation
    def __call__(self, batch):
        images, texts = zip(*batch)
        tokens = self.tokenizer(
            texts, padding=self.padding, truncation=self.truncation,
            max_length=self.max_length, return_tensors="pt",
        )
        return {
            "images": torch.stack(images),
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"].bool(),
        }
def build_dataloader(dataset, model_name=cfg.QFORMER_MODEL_ID, *,
                     batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS,
                     drop_last=cfg.DROP_LAST, shuffle=cfg.DATA_SHUFFLE,
                     pin_memory=cfg.PIN_MEMORY, persistent_workers=cfg.PERSISTENT_WORKERS,
                     prefetch_factor=cfg.PREFETCH_FACTOR, collate_fn=None):

    if collate_fn is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=cfg.TOKENIZER_USE_FAST)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError(f"{model_name} has no PAD/EOS token")
            tokenizer.pad_token = tokenizer.eos_token
        collate_fn = VLMDataCollator(tokenizer)

    kwargs = dict(
        dataset=dataset, batch_size=batch_size, shuffle=dataset.train if shuffle is None else shuffle,
        drop_last=drop_last if dataset.train else False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available() if pin_memory is None else pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        collate_fn=collate_fn,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)

import matplotlib.pyplot as plt

if __name__ == "__main__":
    if cfg.TRAIN_JSON_PATH is None or cfg.TRAIN_IMAGE_DIR is None:
        raise ValueError("Set TRAIN_JSON_PATH and TRAIN_IMAGE_DIR in src_src/config_model.py")

    dataset = VLMDatasetStage1(cfg.TRAIN_JSON_PATH, cfg.TRAIN_IMAGE_DIR, train=False)
    loader = build_dataloader(dataset)
    batch = next(iter(loader))
    tokenizer = loader.collate_fn.tokenizer

    for i in range(5):
        image = batch["images"][i].permute(1, 2, 0)
        caption = tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True)

        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title(caption, fontsize=10, wrap=True)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
