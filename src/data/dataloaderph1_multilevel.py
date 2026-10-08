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
from torch.utils.data import BatchSampler, DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoTokenizer

from src import config as cfg
from src.utils.seed import seed_everything, seed_worker

def build_transform(image_size, is_training, normalization, crop_scale, crop_ratio,
                    interpolation, antialias, horizontal_flip_probability,
                    color_jitter, color_jitter_probability, grayscale_probability,
                    blur_kernel_size, blur_sigma, blur_probability):
    interp = dict(interpolation=InterpolationMode(interpolation), antialias=antialias)
    aug = ([
        transforms.RandomResizedCrop(image_size, scale=crop_scale, ratio=crop_ratio, **interp),
        transforms.RandomHorizontalFlip(horizontal_flip_probability),
        transforms.RandomApply([transforms.ColorJitter(*color_jitter)], p=color_jitter_probability),
        transforms.RandomGrayscale(grayscale_probability),
        transforms.RandomApply([transforms.GaussianBlur(blur_kernel_size, blur_sigma)], p=blur_probability),
    ] if is_training else [transforms.Resize((image_size, image_size), **interp)])
    aug.append(transforms.ToTensor())
    if normalization is not None:
        aug.append(transforms.Normalize(*normalization))
    return transforms.Compose(aug)

def _distributed_info(enabled=True):
    if not enabled:
        return False, 0, 1
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("distributed=True requires an initialized process group")
    return True, dist.get_rank(), dist.get_world_size()

def _cache_signature(json_path, tokenizer, max_length):
    p = Path(json_path).resolve()
    stat = p.stat()
    vocab = (tokenizer.backend_tokenizer.to_str() if tokenizer.is_fast
             else json.dumps(tokenizer.get_vocab(), sort_keys=True))
    fingerprint = hashlib.sha256((vocab + json.dumps(
        tokenizer.special_tokens_map, sort_keys=True, default=str
    ) + tokenizer.truncation_side).encode()).hexdigest()
    return dict(path=str(p), size=stat.st_size, mtime=stat.st_mtime_ns,
                tokenizer=fingerprint, max_length=int(max_length))

def _cache_is_valid(path, signature):
    if not path.exists():
        return False
    try:
        with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as db:
            metadata = dict(db.execute("SELECT key, value FROM metadata"))
            columns = {row[1] for row in db.execute("PRAGMA table_info(samples)")}
            return (metadata.get("signature") == json.dumps(signature, sort_keys=True)
                    and columns >= {"id", "image_name", "text", "token_length"})
    except sqlite3.Error:
        return False

def build_cache(json_path, cache_path, chunk_size, tokenizer, max_length, distributed=False):
    active, rank, _ = _distributed_info(distributed)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    path = Path(cache_path).resolve()
    signature = _cache_signature(json_path, tokenizer, max_length)
    error = None

    if rank == 0 and not _cache_is_valid(path, signature):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        try:
            tmp.unlink(missing_ok=True)
            with sqlite3.connect(tmp) as db:
                db.execute("PRAGMA journal_mode=OFF")
                db.execute("PRAGMA synchronous=OFF")
                db.execute("CREATE TABLE samples (id INTEGER PRIMARY KEY, image_name TEXT NOT NULL, "
                           "text TEXT NOT NULL, token_length INTEGER NOT NULL)")
                db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                db.execute("INSERT INTO metadata VALUES (?, ?)",
                           ("signature", json.dumps(signature, sort_keys=True)))

                def flush(rows):
                    encoded = tokenizer([row[2] for row in rows], padding=False,
                                        truncation=True, max_length=max_length)
                    db.executemany("INSERT INTO samples VALUES (?, ?, ?, ?)",
                                   [(idx, image, text, len(ids))
                                    for (idx, image, text), ids in zip(rows, encoded["input_ids"])])

                with open(json_path, "rb") as source:
                    rows = []
                    for idx, item in enumerate(ijson.items(source, "item")):
                        rows.append((idx, item["image_name"], item["text"]))
                        if len(rows) == chunk_size:
                            flush(rows)
                            rows.clear()
                    if rows:
                        flush(rows)
            tmp.replace(path)
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            error = f"Cache construction failed: {type(exc).__name__}: {exc}"

    if active:
        result = [error]
        dist.broadcast_object_list(result, src=0)
        error = result[0]
    if error:
        raise RuntimeError(error)

def load_token_lengths(cache_path):
    path = Path(cache_path).resolve()
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        n = db.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        return np.fromiter((row[0] for row in db.execute(
            "SELECT token_length FROM samples ORDER BY id")), dtype=np.int32, count=n)

class VLMDatasetStage1(Dataset):
    def __init__(self, json_path, image_dir, is_training, cache_dir, chunk_size,
                 transform, dataset_namespace, tokenizer, max_length, distributed=False):
        self.image_dir = Path(image_dir).resolve()
        self.dataset_namespace = dataset_namespace
        self.is_training = is_training
        self.transform = transform
        self.conn = None
        split = "train" if is_training else "validation"
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(dataset_namespace))
        self.cache_path = Path(cache_dir).resolve() / f"{name}_{split}.sqlite"
        build_cache(json_path, self.cache_path, chunk_size, tokenizer, max_length, distributed)
        with sqlite3.connect(self.cache_path) as db:
            self.length = db.execute("SELECT COUNT(*) FROM samples").fetchone()[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx, valid = idx if isinstance(idx, tuple) else (idx, True)
        if self.conn is None:
            self.conn = sqlite3.connect(f"file:{self.cache_path}?mode=ro", uri=True)
        row = self.conn.execute("SELECT image_name, text FROM samples WHERE id=?", (int(idx),)).fetchone()
        if row is None:
            raise IndexError(idx)
        name, text = row
        path = (self.image_dir / name).resolve()
        relative = path.relative_to(self.image_dir).as_posix()
        key = f"{self.dataset_namespace}\0{relative}".encode()
        image_id = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big", signed=True)
        with Image.open(path) as image:
            pixels = self.transform(image.convert("RGB"))
        return pixels, text, image_id, bool(valid)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["conn"] = None
        return state

class VLMDataCollator:
    def __init__(self, tokenizer, max_length, padding="longest", truncation=True):
        self.tokenizer, self.max_length = tokenizer, max_length
        self.padding, self.truncation = padding, truncation

    def __call__(self, batch):
        images, texts, ids = zip(*(sample[:3] for sample in batch))
        tokens = self.tokenizer(texts, padding=self.padding, truncation=self.truncation,
                                max_length=self.max_length, return_tensors="pt")
        return dict(images=torch.stack(images), image_ids=torch.tensor(ids, dtype=torch.long),
                    input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"].bool(),
                    valid_mask=torch.tensor([s[3] if len(s) > 3 else True for s in batch],
                                            dtype=torch.bool))

class MultiLevelLengthBatchSampler(BatchSampler):
    def __init__(self, lengths, batch_size, drop_last=True, shuffle=True,
                 mega_batch_mult=32, length_jitter=4, seed=0,
                 distributed=False, length_aware=True):
        if batch_size <= 0 or mega_batch_mult <= 0 or length_jitter < 0:
            raise ValueError("Invalid sampler parameters")
        self.lengths = np.asarray(lengths)
        self.batch_size, self.drop_last, self.shuffle = int(batch_size), bool(drop_last), bool(shuffle)
        self.mega_batch_mult, self.length_jitter = int(mega_batch_mult), int(length_jitter)
        self.seed, self.epoch, self.length_aware = int(seed), 0, length_aware
        self.distributed, self.rank, self.world_size = _distributed_info(distributed)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        n = len(self.lengths)
        size = self.batch_size * self.world_size
        return n // size if self.drop_last else math.ceil(n / size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        indices = np.arange(len(self.lengths))
        if self.shuffle:
            rng.shuffle(indices)
        step, total = 0, len(self) * self.world_size
        window_size = self.batch_size * self.mega_batch_mult
        for start in range(0, len(indices), window_size):
            window = indices[start:start + window_size]
            if self.length_aware:
                score = self.lengths[window].astype(np.int64)
                if self.shuffle and self.length_jitter:
                    score += rng.integers(-self.length_jitter, self.length_jitter + 1, size=len(window))
                window = window[np.argsort(score, kind="stable")]
            batches = []
            for offset in range(0, len(window), self.batch_size):
                part = window[offset:offset + self.batch_size].copy()
                if len(part) == self.batch_size or not self.drop_last:
                    if self.shuffle:
                        rng.shuffle(part)
                    batches.append(part.tolist())
            if self.shuffle:
                rng.shuffle(batches)
            for batch in batches:
                if step >= total:
                    return
                if step % self.world_size == self.rank:
                    if self.distributed and not self.drop_last:
                        batch = [(idx, True) for idx in batch] + [
                            (batch[0], False)] * (self.batch_size - len(batch))
                    else:
                        batch = [(idx, True) for idx in batch]
                    yield batch
                step += 1
        while step < total:
            if step % self.world_size == self.rank:
                yield [(0, False)] * self.batch_size
            step += 1

def build_dataloader(dataset: VLMDatasetStage1, batch_size, num_workers,
                     drop_last, shuffle, pin_memory, persistent_workers,
                     prefetch_factor, collate_fn, distributed=False,
                     length_aware=True, mega_batch_mult=32, length_jitter=4, seed=0):
    if not len(dataset):
        raise ValueError("Dataset is empty")
    active, rank, _ = _distributed_info(distributed)
    sampler = MultiLevelLengthBatchSampler(
        load_token_lengths(dataset.cache_path) if dataset.is_training and length_aware
        else np.zeros(len(dataset), dtype=np.uint8), batch_size,
        drop_last=drop_last if dataset.is_training else False,
        shuffle=(dataset.is_training if shuffle is None else shuffle) if dataset.is_training else False,
        mega_batch_mult=mega_batch_mult, length_jitter=length_jitter, seed=seed,
        distributed=active, length_aware=dataset.is_training and length_aware)
    generator = torch.Generator().manual_seed(seed + rank)
    kwargs = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available() if pin_memory is None
                  else pin_memory, persistent_workers=bool(persistent_workers and num_workers > 0),
                  worker_init_fn=seed_worker, generator=generator, collate_fn=collate_fn)
    if num_workers:
        kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(dataset, batch_sampler=sampler, **kwargs)
    if not len(loader):
        raise ValueError("No batches; reduce batch_size/world_size or add samples")
    return loader

def set_dataloader_epoch(loader, epoch):
    loader.batch_sampler.set_epoch(epoch)

def check_batch_length_stats(loader):
    lengths = load_token_lengths(loader.dataset.cache_path)
    ranges, stds, wastes = [], [], []
    for batch in loader.batch_sampler:
        values = np.asarray([lengths[idx] for idx, valid in batch if valid], dtype=np.float64)
        if not len(values):
            continue
        ranges.append(float(np.ptp(values)))
        stds.append(float(np.std(values)))
        wastes.append(float(1 - values.sum() / (len(values) * values.max())))
    if not ranges:
        raise ValueError("No valid samples to evaluate")
    stats = dict(num_batches=len(ranges), avg_length_range=float(np.mean(ranges)),
                 avg_length_std=float(np.mean(stds)), avg_padding_ratio=float(np.mean(wastes)))
    print(f"Num batches        : {stats['num_batches']}\n"
          f"Avg max-min length : {stats['avg_length_range']:.2f} tokens\n"
          f"Avg std length     : {stats['avg_length_std']:.2f} tokens\n"
          f"Avg padding waste  : {stats['avg_padding_ratio'] * 100:.2f}%")
    return stats

def main():
    import matplotlib.pyplot as plt
    seed_everything()
    root = Path(__file__).resolve().parents[2]
    distributed = dist.is_available() and dist.is_initialized()
    tokenizer = AutoTokenizer.from_pretrained(cfg.TOKENIZER_MODEL_ID, use_fast=cfg.TOKENIZER_USE_FAST)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no PAD/EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    transform = build_transform(
        cfg.IMAGE_SIZE, cfg.IS_TRAINING, cfg.NORMALIZATION, cfg.CROP_SCALE,
        cfg.CROP_RATIO, cfg.INTERPOLATION, cfg.ANTIALIAS,
        cfg.HORIZONTAL_FLIP_PROBABILITY, cfg.COLOR_JITTER,
        cfg.COLOR_JITTER_PROBABILITY, cfg.GRAYSCALE_PROBABILITY,
        cfg.BLUR_KERNEL_SIZE, cfg.BLUR_SIGMA, cfg.BLUR_PROBABILITY)
    dataset = VLMDatasetStage1(
        root / (cfg.JSON_PATH_TRAIN if cfg.IS_TRAINING else cfg.JSON_PATH_VAL),
        root / cfg.IMAGE_DIR, cfg.IS_TRAINING, root / cfg.CACHE_DIR,
        cfg.CACHE_CHUNK_SIZE, transform, cfg.DATASET_NAMESPACE, tokenizer,
        cfg.MAX_LENGTH, distributed)
    loader = build_dataloader(
        dataset, cfg.BATCH_SIZE, cfg.NUM_WORKERS, cfg.DROP_LAST, cfg.SHUFFLE,
        cfg.PIN_MEMORY, cfg.PERSISTENT_WORKERS, cfg.PREFETCH_FACTOR,
        VLMDataCollator(tokenizer, cfg.MAX_LENGTH, cfg.TOKENIZER_PADDING,
                        cfg.TOKENIZER_TRUNCATION), distributed, cfg.IS_TRAINING,
        getattr(cfg, "LENGTH_MEGA_BATCH_MULT", 32), getattr(cfg, "LENGTH_JITTER", 4),
        getattr(cfg, "SEED", 0))
    check_batch_length_stats(loader)
    for batch in loader:
        print({key: tuple(value.shape) for key, value in batch.items()})
        for i in range(min(cfg.DEMO_NUM_IMAGES, len(batch["images"]))):
            image = batch["images"][i].permute(1, 2, 0)
            if cfg.NORMALIZATION is not None:
                mean, std = cfg.NORMALIZATION
                image = image * torch.tensor(std) + torch.tensor(mean)
            caption = tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True)
            plt.figure(figsize=cfg.DEMO_FIGURE_SIZE)
            plt.imshow(image.clamp(0, 1))
            plt.title(caption, fontsize=cfg.DEMO_TITLE_FONT_SIZE, wrap=True)
            plt.axis("off")
            plt.tight_layout()
            plt.show()
            plt.close()

if __name__ == "__main__":
    main()