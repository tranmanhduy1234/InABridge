import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path

import ijson
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import BatchSampler, DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoTokenizer

from src import config as cfg
from src.utils.seed import seed_everything, seed_worker


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

def _distributed_info(enabled=True):
    active = enabled and dist.is_available() and dist.is_initialized()
    return active, dist.get_rank() if active else 0, dist.get_world_size() if active else 1

def _safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))

def _cache_signature(json_path, tokenizer, max_length):
    p = Path(json_path).resolve()
    stat = p.stat()
    return {
        "json_path": str(p),
        "json_size": stat.st_size,
        "json_mtime_ns": stat.st_mtime_ns,
        "tokenizer": getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__),
        "max_length": int(max_length),
    }

def _cache_is_valid(cache_path, signature):
    if not cache_path.exists():
        return False
    try:
        with sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True) as conn:
            rows = dict(conn.execute("SELECT key, value FROM metadata"))
            cols = {r[1] for r in conn.execute("PRAGMA table_info(samples)")}
        return cols >= {"id", "image_name", "text", "token_length"} and \
               rows.get("signature") == json.dumps(signature, sort_keys=True)
    except sqlite3.Error:
        return False

def build_cache(json_path, cache_path, chunk_size, tokenizer, max_length, distributed=False):
    distributed, rank, _ = _distributed_info(distributed)
    json_path, cache_path = Path(json_path), Path(cache_path)
    signature = _cache_signature(json_path, tokenizer, max_length)

    if rank == 0 and not _cache_is_valid(cache_path, signature):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        if tmp.exists():
            tmp.unlink()

        conn = sqlite3.connect(tmp)
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("""
            CREATE TABLE samples (
                id INTEGER PRIMARY KEY,
                image_name TEXT NOT NULL,
                text TEXT NOT NULL,
                token_length INTEGER NOT NULL
            )
        """)
        conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute(
            "INSERT INTO metadata VALUES (?, ?)",
            ("signature", json.dumps(signature, sort_keys=True)),
        )
        def flush(rows):
            if not rows:
                return
            texts = [r[2] for r in rows]
            encoded = tokenizer(
                texts,
                add_special_tokens=True,
                padding=False,
                truncation=True,
                max_length=max_length,
                return_length=True,
            )
            lengths = encoded.get("length")
            if lengths is None:
                lengths = [len(x) for x in encoded["input_ids"]]
            conn.executemany(
                "INSERT INTO samples VALUES (?, ?, ?, ?)",
                [(idx, image_name, text, int(length))
                 for (idx, image_name, text), length in zip(rows, lengths)],
            )

        with open(json_path, "rb") as f:
            rows = []
            for i, x in enumerate(ijson.items(f, "item")):
                rows.append((i, x["image_name"], x["text"]))
                if len(rows) >= chunk_size:
                    flush(rows)
                    rows.clear()
            flush(rows)

        conn.commit()
        conn.close()
        tmp.replace(cache_path)

    if distributed:
        dist.barrier()

def load_token_lengths(cache_path):
    with sqlite3.connect(f"file:{Path(cache_path)}?mode=ro", uri=True) as conn:
        n = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        return np.fromiter(
            (row[0] for row in conn.execute("SELECT token_length FROM samples ORDER BY id")),
            dtype=np.uint16,
            count=n,
        )

class VLMDatasetStage1(Dataset):
    def __init__(self, json_path, image_dir, is_training, cache_dir, chunk_size,
                 transform, dataset_namespace, tokenizer, max_length, distributed=False):
        self.image_dir = Path(image_dir).resolve()
        self.dataset_namespace = dataset_namespace
        self.is_training = is_training
        self.transform = transform
        self.conn = None

        split = "train" if is_training else "validation"
        cache_name = f"{_safe_name(dataset_namespace)}_{split}.sqlite"
        self.cache_path = Path(cache_dir) / cache_name

        build_cache(
            json_path=json_path,
            cache_path=self.cache_path,
            chunk_size=chunk_size,
            tokenizer=tokenizer,
            max_length=max_length,
            distributed=distributed,
        )

        with sqlite3.connect(self.cache_path) as conn:
            self.length = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]

    def __len__(self):
        return self.length

    def _db(self):
        if self.conn is None:
            self.conn = sqlite3.connect(
                f"file:{self.cache_path}?mode=ro",
                uri=True,
                check_same_thread=False,
            )
        return self.conn

    def __getitem__(self, idx):
        row = self._db().execute(
            "SELECT image_name, text FROM samples WHERE id=?",
            (int(idx),),
        ).fetchone()
        if row is None:
            raise IndexError(idx)

        image_name, text = row
        image_path = (self.image_dir / image_name).resolve()
        relative_path = image_path.relative_to(self.image_dir).as_posix()

        key = f"{self.dataset_namespace}\0{relative_path}".encode("utf-8")
        image_id = int.from_bytes(
            hashlib.blake2b(key, digest_size=8).digest(),
            "big",
            signed=True,
        )

        with Image.open(image_path) as img:
            image = self.transform(img.convert("RGB"))

        return image, text, image_id

    def __getstate__(self):
        state = self.__dict__.copy()
        state["conn"] = None
        return state

class VLMDataCollator:
    def __init__(self, tokenizer, max_length, padding="longest", truncation=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

    def __call__(self, batch):
        images, texts, image_ids = zip(*batch)
        tokens = self.tokenizer(
            texts,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "images": torch.stack(images),
            "image_ids": torch.tensor(image_ids, dtype=torch.long),
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"].bool(),
        }

class MultiLevelLengthBatchSampler(BatchSampler):
    def __init__(self, lengths, batch_size, drop_last=True, shuffle=True,
                 mega_batch_mult=32, length_jitter=4, seed=0,
                 distributed=False):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if mega_batch_mult <= 0:
            raise ValueError("mega_batch_mult must be > 0")
        if length_jitter < 0:
            raise ValueError("length_jitter must be >= 0")

        self.lengths = np.asarray(lengths)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.mega_batch_mult = int(mega_batch_mult)
        self.length_jitter = int(length_jitter)
        self.seed = int(seed)
        self.epoch = 0

        self.distributed, self.rank, self.world_size = _distributed_info(distributed)
        if self.distributed and not self.drop_last:
            raise ValueError(
                "Distributed training should use drop_last=True so all ranks "
                "execute the same number of optimizer steps."
            )

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _num_global_batches(self):
        n = len(self.lengths)
        batches = n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)
        if self.distributed:
            batches -= batches % self.world_size
        return batches

    def __len__(self):
        return self._num_global_batches() // self.world_size

    def __iter__(self):
        n = len(self.lengths)
        rng = np.random.default_rng(self.seed + self.epoch)

        indices = np.arange(n, dtype=np.int32 if n < 2**31 else np.int64)
        if self.shuffle:
            rng.shuffle(indices)

        global_batch_limit = self._num_global_batches()
        produced_global = 0
        window_size = self.batch_size * self.mega_batch_mult

        for start in range(0, n, window_size):
            if produced_global >= global_batch_limit:
                break

            window = indices[start:start + window_size]
            if len(window) == 0:
                continue

            score = self.lengths[window].astype(np.int32, copy=True)
            if self.shuffle and self.length_jitter:
                score += rng.integers(
                    -self.length_jitter,
                    self.length_jitter + 1,
                    size=len(window),
                    dtype=np.int32,
                )

            order = np.argsort(score, kind="stable")
            grouped = window[order]

            batches = []
            for b_start in range(0, len(grouped), self.batch_size):
                batch = grouped[b_start:b_start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batch = batch.copy()
                if self.shuffle:
                    rng.shuffle(batch)
                batches.append(batch)

            if self.shuffle:
                rng.shuffle(batches)

            for batch in batches:
                if produced_global >= global_batch_limit:
                    break

                global_id = produced_global
                produced_global += 1

                if global_id % self.world_size == self.rank:
                    yield batch.tolist()

def build_dataloader(dataset: VLMDatasetStage1, batch_size, num_workers,
                     drop_last, shuffle, pin_memory, persistent_workers,
                     prefetch_factor, collate_fn, distributed=False,
                     length_aware=True, mega_batch_mult=32,
                     length_jitter=4, seed=0):
    shuffle = dataset.is_training if shuffle is None else shuffle
    distributed_active, rank, _ = _distributed_info(distributed)

    generator = torch.Generator()
    generator.manual_seed(seed + rank)

    common = dict(
        dataset=dataset,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available() if pin_memory is None else pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    if num_workers > 0:
        common["prefetch_factor"] = prefetch_factor

    if dataset.is_training and length_aware:
        lengths = load_token_lengths(dataset.cache_path)
        batch_sampler = MultiLevelLengthBatchSampler(
            lengths=lengths,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            mega_batch_mult=mega_batch_mult,
            length_jitter=length_jitter,
            seed=seed,
            distributed=distributed_active,
        )
        return DataLoader(batch_sampler=batch_sampler, **common)

    sampler = None
    if distributed_active:
        sampler = DistributedSampler(
            dataset,
            shuffle=shuffle if dataset.is_training else False,
            drop_last=drop_last if dataset.is_training else False,
        )

    return DataLoader(
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        drop_last=drop_last if dataset.is_training else False,
        **common,
    )

def set_dataloader_epoch(loader, epoch):
    sampler = getattr(loader, "batch_sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
        return

    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)

def check_batch_length_stats(loader):
    total_range = 0.0
    total_std = 0.0
    total_padding_ratio = 0.0
    num_batches = 0

    for batch in loader:
        lengths = batch["attention_mask"].sum(dim=1).float()

        batch_range = (lengths.max() - lengths.min()).item()
        batch_std = lengths.std(unbiased=False).item()

        padded_tokens = lengths.numel() * lengths.max()
        real_tokens = lengths.sum()
        padding_ratio = (1 - real_tokens / padded_tokens).item()

        total_range += batch_range
        total_std += batch_std
        total_padding_ratio += padding_ratio
        num_batches += 1

    stats = {
        "num_batches": num_batches,
        "avg_length_range": total_range / num_batches,
        "avg_length_std": total_std / num_batches,
        "avg_padding_ratio": total_padding_ratio / num_batches,
    }

    print(f"Num batches        : {stats['num_batches']}")
    print(f"Avg max-min length : {stats['avg_length_range']:.2f} tokens")
    print(f"Avg std length     : {stats['avg_length_std']:.2f} tokens")
    print(f"Avg padding waste  : {stats['avg_padding_ratio'] * 100:.2f}%")

    return stats

def main():
    seed_everything()
    import matplotlib.pyplot as plt

    root = Path(__file__).resolve().parents[2]
    distributed = dist.is_available() and dist.is_initialized()

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.TOKENIZER_MODEL_ID,
        use_fast=cfg.TOKENIZER_USE_FAST,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no PAD/EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    transform = build_transform(
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
        transform=transform,
        dataset_namespace=cfg.DATASET_NAMESPACE,
        tokenizer=tokenizer,
        max_length=cfg.MAX_LENGTH,
        distributed=distributed,
    )

    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    collator = VLMDataCollator(
        tokenizer=tokenizer,
        max_length=cfg.MAX_LENGTH,
        padding=cfg.TOKENIZER_PADDING,
        truncation=cfg.TOKENIZER_TRUNCATION,
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
        distributed=distributed,
        length_aware=cfg.IS_TRAINING,
        mega_batch_mult=getattr(cfg, "LENGTH_MEGA_BATCH_MULT", 32),
        length_jitter=getattr(cfg, "LENGTH_JITTER", 4),
        seed=getattr(cfg, "SEED", 0),
    )
    check_batch_length_stats(loader)
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