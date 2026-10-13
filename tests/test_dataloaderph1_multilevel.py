"""Offline, CPU-only checks for cache, soft length grouping and DDP partitioning."""
import json
import math
import multiprocessing
import pickle
import sqlite3
from collections import Counter
from datetime import timedelta
from multiprocessing.connection import Listener

import numpy as np
import pytest
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader

from src.data import dataloaderph1_multilevel as dl
from src.trainingph1 import engine_fsdp as engine


class DummyTokenizer:
    name_or_path = "dummy-v1"
    pad_token_id = 0
    is_fast = False
    special_tokens_map = {'pad_token': '[PAD]'}
    truncation_side = 'right'

    def __init__(self, name="dummy-v1", return_lengths=True):
        self.name_or_path = name
        self.return_lengths = return_lengths
        self.calls = 0
        self.vocab = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2}

    def get_vocab(self):
        return self.vocab

    def __call__(self, texts, add_special_tokens=True, padding=False,
                 truncation=False, max_length=None, return_length=False,
                 return_tensors=None):
        self.calls += 1
        ids = [[3 + i % 100 for i, _ in enumerate(text.split())] for text in texts]
        if add_special_tokens:
            ids = [[1, *row, 2] for row in ids]
        if truncation:
            ids = [row[:max_length] for row in ids]
        lengths = [len(row) for row in ids]
        width = max_length if padding == "max_length" else max(lengths, default=0)
        mask = [[1] * len(row) for row in ids]
        if padding:
            mask = [row + [0] * (width - len(row)) for row in mask]
            ids = [row + [self.pad_token_id] * (width - len(row)) for row in ids]
        result = {"input_ids": ids, "attention_mask": mask}
        if return_tensors == "pt":
            result = {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}
        if return_length and self.return_lengths:
            result["length"] = lengths
        return result


def image_transform(image):
    return torch.from_numpy(np.array(image.resize((8, 8)), copy=True)).permute(2, 0, 1).float() / 255


@pytest.fixture(autouse=True)
def forbid_pretrained_downloads(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Tests must never download a pretrained tokenizer")
    monkeypatch.setattr(dl.AutoTokenizer, "from_pretrained", fail)


@pytest.fixture
def tokenizer():
    return DummyTokenizer()


@pytest.fixture
def dataset_factory(tmp_path, tokenizer):
    datasets = []

    def make(word_counts=(1, 4, 12, 30, 3), max_length=24, namespace="test", samples=None):
        root = tmp_path / str(len(datasets))
        root.mkdir()
        if samples is None:
            samples = [{"image_name": f"{i}.png", "text": " ".join(["word"] * n)}
                       for i, n in enumerate(word_counts)]
        for i, sample in enumerate(samples):
            path = root / sample["image_name"]
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (10, 12), (i % 256, 40, 90)).save(path)
        manifest = root / "samples.json"
        manifest.write_text(json.dumps(samples), encoding="utf-8")
        dataset = dl.VLMDatasetStage1(
            manifest, root, True, root / "cache", 3, image_transform,
            namespace, tokenizer, max_length,
        )
        datasets.append(dataset)
        return dataset, samples, manifest

    yield make
    for dataset in datasets:
        if dataset.conn is not None:
            dataset.conn.close()


@pytest.fixture
def wide_lengths():
    return np.tile(np.concatenate([np.arange(20, 31), np.arange(50, 61),
                                   np.arange(100, 121), np.arange(150, 161)]), 16)


@pytest.fixture
def partition():
    return dict(rank=0, num_replicas=2, pad_batches=True)


def read_rows(path):
    with sqlite3.connect(path) as conn:
        return conn.execute("SELECT id, image_name, text, token_length FROM samples ORDER BY id").fetchall()


def flattened(batches):
    return [index for batch in batches for index, valid in batch if valid]


def memberships(batches):
    return {frozenset(flattened([batch])) for batch in batches}


def length_metrics(lengths, batches):
    values = [np.asarray(lengths)[flattened([batch])] for batch in batches]
    return (np.mean([np.ptp(v) for v in values]),
            np.mean([1 - v.sum() / (len(v) * v.max()) for v in values]))


def compute_loader_length_stats(loader):
    lengths = [batch["attention_mask"].sum(dim=1).numpy() for batch in loader]
    return {
        "num_batches": len(lengths),
        "avg_range": float(np.mean([np.ptp(v) for v in lengths])) if lengths else 0.0,
        "avg_std": float(np.mean([np.std(v) for v in lengths])) if lengths else 0.0,
        "avg_padding_waste": float(np.mean([1 - v.sum() / (len(v) * v.max())
                                            for v in lengths])) if lengths else 0.0,
        "max_batch_length": int(max(v.max() for v in lengths)) if lengths else 0,
        "min_batch_length": int(min(v.min() for v in lengths)) if lengths else 0,
    }


def make_loader(dataset, tokenizer, max_length=24, **overrides):
    options = dict(batch_size=8, num_workers=0, drop_last=False, shuffle=True,
                   pin_memory=False, persistent_workers=False, prefetch_factor=2,
                   collate_fn=dl.VLMDataCollator(tokenizer, max_length), seed=17,
                   mega_batch_mult=32, length_jitter=2)
    options.update(overrides)
    return dl.build_dataloader(dataset, **options)


@pytest.mark.parametrize("return_lengths", [True, False])
def test_cache_schema_rows_truncation_and_final_chunk(dataset_factory, return_lengths):
    dataset, samples, manifest = dataset_factory(word_counts=[0, 1, 5, 20, 100], max_length=12)
    path = manifest.parent / "separate.sqlite"
    tokenizer = DummyTokenizer(return_lengths=return_lengths)
    dl.build_cache(manifest, path, 2, tokenizer, 12)
    assert path.is_file()
    with sqlite3.connect(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(samples)")}
    assert columns >= {"id", "image_name", "text", "token_length"}
    expected = [(i, s["image_name"], s["text"], min(len(s["text"].split()) + 2, 12))
                for i, s in enumerate(samples)]
    assert read_rows(path) == expected
    assert tokenizer.calls == 3
    assert dl.load_token_lengths(path).tolist() == [row[3] for row in expected]
    assert not path.with_suffix(".sqlite.tmp").exists()


def test_cache_reuses_matching_metadata_without_tokenizing(dataset_factory, tokenizer):
    dataset, _, manifest = dataset_factory()
    before = dataset.cache_path.stat()
    calls = tokenizer.calls
    dl.build_cache(manifest, dataset.cache_path, 1, tokenizer, 24)
    after = dataset.cache_path.stat()
    assert tokenizer.calls == calls
    assert (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)


@pytest.mark.parametrize("change", ["json", "tokenizer", "max_length"])
def test_cache_rebuilds_when_signature_changes(dataset_factory, tokenizer, change):
    dataset, samples, manifest = dataset_factory()
    if change == "json":
        samples.append({"image_name": "extra.png", "text": "new caption"})
        manifest.write_text(json.dumps(samples), encoding="utf-8")
    if change == "tokenizer":
        tokenizer.vocab['new_token'] = 103
    limit = 8 if change == "max_length" else 24
    calls = tokenizer.calls
    dl.build_cache(manifest, dataset.cache_path, 2, tokenizer, limit)
    assert tokenizer.calls > calls
    assert read_rows(dataset.cache_path) == [
        (i, s["image_name"], s["text"], min(len(s["text"].split()) + 2, limit))
        for i, s in enumerate(samples)]
    calls = tokenizer.calls
    dl.build_cache(manifest, dataset.cache_path, 2, tokenizer, limit)
    assert tokenizer.calls == calls


@pytest.mark.parametrize("existing_cache", [False, True])
def test_failed_cache_build_never_promotes_partial_tmp(dataset_factory, existing_cache):
    dataset, _, manifest = dataset_factory()
    path = dataset.cache_path if existing_cache else manifest.parent / "new.sqlite"
    before = path.read_bytes() if path.exists() else None
    class FailingTokenizer(DummyTokenizer):
        def __call__(self, *args, **kwargs):
            if self.calls == 1:
                raise RuntimeError("synthetic tokenization failure")
            return super().__call__(*args, **kwargs)
    tokenizer = FailingTokenizer()
    tokenizer.vocab['new_token'] = 103
    with pytest.raises(RuntimeError, match="synthetic tokenization failure"):
        dl.build_cache(manifest, path, 2, tokenizer, 24)
    if existing_cache:
        assert path.read_bytes() == before
    else:
        assert not path.exists()
    tokenizer = DummyTokenizer()
    tokenizer.vocab['new_token'] = 103
    dl.build_cache(manifest, path, 2, tokenizer, 24)
    assert len(read_rows(path)) == 5
    assert not path.with_suffix(".sqlite.tmp").exists()


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_cache_mkdir_failure_is_broadcast(dataset_factory, tokenizer, monkeypatch, cleanup_fails):
    _, _, manifest = dataset_factory()
    monkeypatch.setattr(engine.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(engine.dist, "get_rank", lambda: 0)
    broadcasts = []
    monkeypatch.setattr(engine.dist, "broadcast_object_list",
                        lambda result, src: broadcasts.append((result.copy(), src)))

    def fail_mkdir(*args, **kwargs):
        raise PermissionError("cannot create cache directory")

    def fail_unlink(*args, **kwargs):
        raise PermissionError("cannot clean up cache")

    with monkeypatch.context() as patches:
        patches.setattr(dl.Path, "mkdir", fail_mkdir)
        if cleanup_fails:
            patches.setattr(dl.Path, "unlink", fail_unlink)
        with pytest.raises(RuntimeError, match="PermissionError: cannot create cache directory") as exc:
            engine.build_dataset(manifest, manifest.parent, True, manifest.parent / "new_cache", 2,
                                 image_transform, "test", tokenizer, 24, distributed=True)
    assert broadcasts == [([str(exc.value)], 0)]


@pytest.mark.parametrize("max_length", [65535, 65536, 70000])
def test_cached_lengths_preserve_large_valid_max_length(dataset_factory, max_length):
    dataset, _, _ = dataset_factory(word_counts=[70000], max_length=max_length)
    assert read_rows(dataset.cache_path)[0][3] == max_length
    assert dl.load_token_lengths(dataset.cache_path).tolist() == [max_length]


def test_corrupt_cache_is_rebuilt(dataset_factory, tokenizer):
    dataset, _, manifest = dataset_factory()
    expected = read_rows(dataset.cache_path)
    dataset.cache_path.write_bytes(b"interrupted or corrupt sqlite file")
    dl.build_cache(manifest, dataset.cache_path, 2, tokenizer, 24)
    assert read_rows(dataset.cache_path) == expected


def test_dataset_can_open_prepared_cache_without_tokenizing(dataset_factory, tokenizer):
    dataset, _, manifest = dataset_factory()
    calls = tokenizer.calls
    reopened = dl.VLMDatasetStage1(manifest, manifest.parent, True, dataset.cache_path.parent, 2,
                                  image_transform, "test", tokenizer, 24, prepare_cache=False)
    assert len(reopened) == len(dataset)
    assert tokenizer.calls == calls


def test_dataset_values_lazy_connection_and_pickle(dataset_factory):
    dataset, samples, _ = dataset_factory()
    assert len(dataset) == len(samples)
    assert dataset.conn is None
    for i, sample in enumerate(samples):
        image, text, image_id, valid = dataset[i]
        assert valid is True
        assert image.shape == (3, 8, 8)
        assert image.dtype == torch.float32
        torch.testing.assert_close(image[:, 0, 0], torch.tensor([i, 40, 90]) / 255)
        assert text == sample["text"]
        assert isinstance(image_id, int)
    connection = dataset.conn
    assert connection is not None
    assert dataset.__getstate__()["conn"] is None
    assert dataset.conn is connection
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored.conn is None
    try:
        assert restored[0][1:] == dataset[0][1:]
    finally:
        restored.conn.close()


def test_image_ids_are_stable_across_instances_relocation_and_path_aliases(dataset_factory):
    samples = [{"image_name": name, "text": "caption"}
               for name in ["image.png", "./image.png", "other.png"]]
    first, _, _ = dataset_factory(samples=samples)
    moved, _, _ = dataset_factory(samples=samples)
    other, _, _ = dataset_factory(samples=samples, namespace="other")
    assert first[0][2] == first[1][2] == moved[0][2]
    assert first[0][2] != first[2][2]
    assert first[0][2] != other[0][2]


@pytest.mark.parametrize("index", [-1, 5, 100])
def test_dataset_out_of_bounds_raises_index_error(dataset_factory, index):
    dataset, _, _ = dataset_factory()
    with pytest.raises(IndexError):
        dataset[index]


@pytest.mark.parametrize("word_counts, limit", [([1, 4, 2], 32), ([0, 4, 1000], 16)])
def test_collator_stacks_preserves_ids_pads_longest_and_truncates(tokenizer, word_counts, limit):
    batch = [(torch.full((3, 8, 8), float(i)), " ".join(["x"] * n), [-9, 7, -9][i])
             for i, n in enumerate(word_counts)]
    result = dl.VLMDataCollator(tokenizer, limit)(batch)
    effective = [min(n + 2, limit) for n in word_counts]
    torch.testing.assert_close(result["images"], torch.stack([row[0] for row in batch]))
    assert result["image_ids"].dtype == torch.long
    assert result["image_ids"].tolist() == [-9, 7, -9]
    assert result["input_ids"].shape == (3, max(effective))
    assert result["attention_mask"].dtype == torch.bool
    assert result["attention_mask"].sum(1).tolist() == effective
    for i, length in enumerate(effective):
        expected = ([1] + [3 + j % 100 for j in range(word_counts[i])] + [2])[:limit]
        assert result["input_ids"][i, :length].tolist() == expected
        assert result["attention_mask"][i, :length].all()
        assert not result["attention_mask"][i, length:].any()
        assert (result["input_ids"][i, length:] == tokenizer.pad_token_id).all()


@pytest.mark.parametrize("n", [0, 1, 7, 8, 9, 63, 65, 137])
@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("mega_batch_mult", [1, 4])
def test_sampler_exact_coverage_batch_sizes_and_len(n, drop_last, shuffle, mega_batch_mult):
    lengths = np.random.default_rng(31).integers(2, 200, n)
    sampler = dl.MultiLevelLengthBatchSampler(lengths, 8, drop_last, shuffle,
                                              mega_batch_mult, 0, seed=13)
    batches = list(sampler)
    indices = flattened(batches)
    assert len(batches) == len(sampler) == (n // 8 if drop_last else math.ceil(n / 8))
    assert all(0 < len(batch) <= 8 for batch in batches)
    if drop_last:
        assert all(len(batch) == 8 for batch in batches)
    assert all(0 <= i < n for i in indices)
    assert len(indices) == len(set(indices))
    assert n - len(indices) == (n % 8 if drop_last else 0)
    if not drop_last:
        assert sorted(indices) == list(range(n))


@pytest.mark.parametrize("lengths", [[10] * 129, [1, 2, 10000, 1000000] * 33])
@pytest.mark.parametrize("jitter", [0, 4])
def test_sampler_equal_or_extreme_lengths_preserve_coverage(lengths, jitter):
    sampler = dl.MultiLevelLengthBatchSampler(lengths, 8, drop_last=False, length_jitter=jitter)
    assert sorted(flattened(list(sampler))) == list(range(len(lengths)))


def test_seed_epoch_reproducibility_and_checkpoint_resume(wide_lengths):
    sampler = dl.MultiLevelLengthBatchSampler(wide_lengths, 8, seed=37)
    sampler.set_epoch(0)
    epoch0 = list(sampler)
    assert list(sampler) == epoch0
    sampler.set_epoch(1)
    epoch1 = list(sampler)
    assert epoch1 != epoch0
    assert list(sampler) == epoch1
    resumed = dl.MultiLevelLengthBatchSampler(wide_lengths, 8, seed=37)
    resumed.set_epoch(1)
    assert list(resumed) == epoch1
    sampler.set_epoch(0)
    assert list(sampler) == epoch0
    assert list(dl.MultiLevelLengthBatchSampler(wide_lengths, 8, seed=38)) != epoch0


def test_multilevel_shuffle_changes_order_membership_and_grouping(wide_lengths):
    sampler = dl.MultiLevelLengthBatchSampler(wide_lengths, 8, seed=17, length_jitter=4)
    epochs = []
    for epoch in range(4):
        sampler.set_epoch(epoch)
        epochs.append(list(sampler))
    for previous, current in zip(epochs, epochs[1:]):
        assert flattened(previous) != flattened(current)
        assert len(memberships(previous) ^ memberships(current)) > len(current) // 2
        assert [min(batch) for batch in previous] != [min(batch) for batch in current]
    # A single mega-bucket with unique lengths and zero jitter fixes membership,
    # isolating batch-order shuffle from changes in grouping.
    lengths = np.arange(1, 257)
    sampler = dl.MultiLevelLengthBatchSampler(lengths, 8, mega_batch_mult=32,
                                              length_jitter=0, seed=17)
    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)
    assert memberships(first) == memberships(second)
    assert [tuple(sorted(b)) for b in first] != [tuple(sorted(b)) for b in second]
    assert any(b != sorted(b) for b in first)


def test_no_shuffle_is_epoch_independent_and_sorted_within_windows(wide_lengths):
    sampler = dl.MultiLevelLengthBatchSampler(wide_lengths, 8, shuffle=False,
                                              drop_last=False, mega_batch_mult=4, length_jitter=100)
    first = list(sampler)
    sampler.set_epoch(9)
    assert list(sampler) == first
    indices = flattened(first)
    for start in range(0, len(indices), 32):
        values = wide_lengths[indices[start:start + 32]]
        assert np.all(values[:-1] <= values[1:])


def test_length_aware_batching_reduces_range_and_padding_waste(wide_lengths):
    custom = list(dl.MultiLevelLengthBatchSampler(wide_lengths, 8, seed=17))
    random_batches = np.random.default_rng(17).permutation(len(wide_lengths)).reshape(-1, 8)
    custom_range, custom_waste = length_metrics(wide_lengths, custom)
    random_batches = [[(int(i), True) for i in batch] for batch in random_batches]
    random_range, random_waste = length_metrics(wide_lengths, random_batches)
    assert custom_range < random_range * 0.5
    assert custom_waste < random_waste * 0.5


def test_larger_mega_buckets_improve_grouping_and_change_composition(wide_lengths):
    small = list(dl.MultiLevelLengthBatchSampler(wide_lengths, 8, mega_batch_mult=1, seed=17))
    large = list(dl.MultiLevelLengthBatchSampler(wide_lengths, 8, mega_batch_mult=32, seed=17))
    assert memberships(small) != memberships(large)
    small_range, small_waste = length_metrics(wide_lengths, small)
    large_range, large_waste = length_metrics(wide_lengths, large)
    assert large_range < small_range * 0.5
    assert large_waste < small_waste * 0.5


def test_jitter_softens_strict_grouping_while_retaining_efficiency():
    lengths = np.arange(20, 1044)
    strict = list(dl.MultiLevelLengthBatchSampler(lengths, 8, mega_batch_mult=128,
                                                 length_jitter=0, seed=17))
    soft = list(dl.MultiLevelLengthBatchSampler(lengths, 8, mega_batch_mult=128,
                                               length_jitter=12, seed=17))
    expected = {frozenset(range(i, i + 8)) for i in range(0, len(lengths), 8)}
    assert memberships(strict) == expected
    assert memberships(soft) != expected
    strict_range, strict_waste = length_metrics(lengths, strict)
    soft_range, soft_waste = length_metrics(lengths, soft)
    random_batches = np.random.default_rng(17).permutation(len(lengths)).reshape(-1, 8)
    random_batches = [[(int(i), True) for i in batch] for batch in random_batches]
    random_range, random_waste = length_metrics(lengths, random_batches)
    assert strict_range < soft_range < random_range * 0.5
    assert strict_waste < soft_waste < random_waste * 0.5
    # Sort only the batches by length: overlap now demonstrates jitter rather
    # than the unrelated shuffling of samples inside each batch.
    ordered = sorted(soft, key=lambda batch: lengths[flattened([batch])].mean())
    assert any(lengths[flattened([a])].max() > lengths[flattened([b])].min() for a, b in zip(ordered, ordered[1:]))


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("n", [1, 7, 8, 17, 32, 65, 137, 259])
def test_distributed_partition_matches_global_stream_and_equal_steps(partition, world_size, n):
    partition["num_replicas"] = world_size
    lengths = np.random.default_rng(7).integers(2, 200, n)
    previous = None
    for epoch in [0, 1]:
        global_sampler = dl.MultiLevelLengthBatchSampler(lengths, 8, seed=11, mega_batch_mult=4)
        global_sampler.set_epoch(epoch)
        global_batches = list(global_sampler)
        limit = len(global_batches) // world_size * world_size
        valid_global = global_batches[:limit]
        partitions = []
        for rank in range(world_size):
            partition["rank"] = rank
            sampler = dl.MultiLevelLengthBatchSampler(lengths, 8, seed=11,
                                                      mega_batch_mult=4, **partition)
            sampler.set_epoch(epoch)
            batches = list(sampler)
            assert batches == list(sampler) == valid_global[rank::world_size]
            assert len(batches) == len(sampler) == n // (8 * world_size)
            assert all(len(batch) == 8 for batch in batches)
            partitions.append(batches)
        reconstructed = [partitions[rank][step] for step in range(limit // world_size)
                         for rank in range(world_size)]
        assert reconstructed == valid_global
        indices = flattened(reconstructed)
        assert len(set(indices)) == len(indices)
        assert set(indices) == set(flattened(valid_global))
        assert n - len(indices) == n % (8 * world_size)
        if previous is not None and limit:
            assert reconstructed != previous
        previous = reconstructed


def test_distributed_pads_partial_batch_mode(partition):
    batches = []
    for rank in range(2):
        partition["rank"] = rank
        sampler = dl.MultiLevelLengthBatchSampler([10] * 17, 8, drop_last=False, **partition)
        local = list(sampler)
        assert len(local) == len(sampler) == 2
        assert all(len(batch) == 8 for batch in local)
        batches.extend(local)
    assert sorted(idx for batch in batches for idx, valid in batch if valid) == list(range(17))


def test_distributed_requested_before_initialization_raises(monkeypatch):
    monkeypatch.setattr(engine.dist, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="initialized process group"):
        engine.get_dataloader(None, {}, distributed=True)


@pytest.mark.parametrize("rank, replicas", [(-1, 2), (2, 2), (0, 0)])
def test_invalid_partition_raises(rank, replicas):
    with pytest.raises(ValueError, match="rank or num_replicas"):
        dl.MultiLevelLengthBatchSampler([10] * 17, 8, rank=rank, num_replicas=replicas)


@pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"batch_size": -1},
                                    {"mega_batch_mult": 0}, {"length_jitter": -1}])
def test_invalid_sampler_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        dl.MultiLevelLengthBatchSampler([10], **({"batch_size": 8} | kwargs))


def require_multiprocessing():
    # Probe OS support before starting workers; never skip a loader failure.
    try:
        context = multiprocessing.get_context("spawn")
        context.Queue().close()
        context.Lock()
        with Listener():
            pass
    except (OSError, PermissionError, NotImplementedError) as exc:
        pytest.skip(f"Multiprocessing primitives unavailable: {exc}")


@pytest.mark.parametrize("num_workers", [0, 2])
def test_full_loader_preserves_images_ids_lengths_and_token_limit(dataset_factory, tokenizer, num_workers):
    if num_workers:
        require_multiprocessing()
    counts = [1, 4, 12, 30, 10000, 7, 2, 15, 80] * 3
    dataset, samples, _ = dataset_factory(word_counts=counts)
    expected = {dataset[i][2]: (min(len(s["text"].split()) + 2, 24), dataset[i][0])
                for i, s in enumerate(samples)}
    dataset.conn.close()
    dataset.conn = None
    sampler = dl.MultiLevelLengthBatchSampler(dl.load_token_lengths(dataset.cache_path), 8,
                                              drop_last=False, seed=17, length_jitter=0)
    options = {"multiprocessing_context": "spawn", "timeout": 30} if num_workers else {}
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers,
                        collate_fn=dl.VLMDataCollator(tokenizer, 24), **options)
    batches = list(loader)
    assert len(batches) == len(loader) == math.ceil(len(dataset) / 8)
    assert [b["image_ids"].tolist() for b in batches] == [
        [dataset[i][2] for i in indices] for indices in sampler]
    seen = []
    for result in batches:
        ids = result["image_ids"].tolist()
        lengths = result["attention_mask"].sum(1)
        assert result["images"].shape == (len(ids), 3, 8, 8)
        assert 0 < len(ids) <= 8
        assert result["image_ids"].dtype == torch.long
        assert result["attention_mask"].dtype == torch.bool
        assert result["input_ids"].shape == result["attention_mask"].shape
        assert result["input_ids"].shape[1] == lengths.max().item() <= 24
        assert lengths.tolist() == [expected[i][0] for i in ids]
        torch.testing.assert_close(result["images"], torch.stack([expected[i][1] for i in ids]))
        seen.extend(ids)
    assert Counter(seen) == Counter(expected.keys())
    stats = compute_loader_length_stats(batches)
    assert stats["avg_range"] < 12
    assert stats["max_batch_length"] == 24


def test_loader_statistics_match_hand_computed_values():
    batches = [{"attention_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)},
               {"attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.bool)}]
    assert compute_loader_length_stats(batches) == {
        "num_batches": 2, "avg_range": 1.0, "avg_std": 0.5,
        "avg_padding_waste": 0.125, "max_batch_length": 4, "min_batch_length": 2,
    }


def test_loader_length_statistics_beat_random_loader(dataset_factory, tokenizer, wide_lengths):
    dataset, _, _ = dataset_factory(word_counts=(wide_lengths - 2).tolist(), max_length=160)
    custom = make_loader(dataset, tokenizer, max_length=160)
    random_loader = make_loader(dataset, tokenizer, max_length=160, length_aware=False)
    grouped, random_stats = map(compute_loader_length_stats, [custom, random_loader])
    assert grouped["num_batches"] == random_stats["num_batches"] == len(dataset) // 8
    for key in ["avg_range", "avg_std", "avg_padding_waste"]:
        assert grouped[key] < random_stats[key] * 0.5
    assert grouped["max_batch_length"] == random_stats["max_batch_length"] == 160
    assert grouped["min_batch_length"] == random_stats["min_batch_length"] == 20


def test_build_loader_epoch_helper_reproduces_batches(dataset_factory, tokenizer):
    dataset, _, _ = dataset_factory(word_counts=list(range(65)))
    loader = make_loader(dataset, tokenizer)
    def ids():
        return [b["image_ids"].tolist() for b in loader]
    dl.set_dataloader_epoch(loader, 0)
    first = ids()
    dl.set_dataloader_epoch(loader, 1)
    second = ids()
    assert first != second
    resumed = make_loader(dataset, tokenizer)
    dl.set_dataloader_epoch(resumed, 1)
    assert second == [b["image_ids"].tolist() for b in resumed]


@pytest.mark.parametrize("length_aware", [False, True])
def test_build_loader_distributed_steps_and_epoch_helper(dataset_factory, tokenizer, partition, length_aware):
    dataset, _, _ = dataset_factory(word_counts=list(range(65)))
    partitions = []
    for rank in range(2):
        partition["rank"] = rank
        loader = make_loader(dataset, tokenizer, **partition, drop_last=True, length_aware=length_aware)
        dl.set_dataloader_epoch(loader, 3)
        sampler = loader.batch_sampler
        assert sampler.epoch == 3
        batches = list(loader)
        assert len(batches) == len(loader) == 4
        assert all(len(b["image_ids"]) == 8 for b in batches)
        partitions.append([i for b in batches for i in b["image_ids"].tolist()])
    assert not set(partitions[0]) & set(partitions[1])
    assert len(set(partitions[0] + partitions[1])) == 64


@pytest.mark.parametrize("length_aware", [False, True])
def test_distributed_loader_honors_seed(dataset_factory, tokenizer, partition, length_aware):
    dataset, _, _ = dataset_factory(word_counts=list(range(65)))
    orders = []
    for seed in (17, 42, 17):
        loader = make_loader(dataset, tokenizer, **partition, drop_last=True,
                             length_aware=length_aware, seed=seed)
        dl.set_dataloader_epoch(loader, 3)
        orders.append([batch["image_ids"].tolist() for batch in loader])
    assert orders[0] == orders[2]
    assert orders[0] != orders[1]


@pytest.mark.parametrize("n, world_size", [(1, 4), (5, 2), (8, 4), (9, 4)])
def test_validation_masks_only_padding(dataset_factory, tokenizer, partition, n, world_size):
    # Repeated IDs are valid captions, not evidence that a slot is padding.
    samples = [{"image_name": "shared.png", "text": "word " * (i + 1)} for i in range(n)]
    dataset, _, _ = dataset_factory(samples=samples)
    dataset.is_training = False
    partition["num_replicas"] = world_size
    valid_indices, batch_shapes, valid_lengths = [], [], []
    for rank in range(world_size):
        partition["rank"] = rank
        loader = make_loader(dataset, tokenizer, **partition, batch_size=2)
        indices = [sample for batch in loader.batch_sampler for sample in batch]
        valid_indices.extend(idx for idx, valid in indices if valid)
        batches = list(loader)
        batch_shapes.append([len(batch["image_ids"]) for batch in batches])
        masks = torch.cat([batch["valid_mask"] for batch in batches])
        assert masks.dtype == torch.bool
        assert masks.tolist() == [valid for _, valid in indices]
        for batch in batches:
            valid_lengths.extend(batch["attention_mask"].sum(1)[batch["valid_mask"]].tolist())
    assert sorted(valid_indices) == list(range(n))
    assert sorted(valid_lengths) == list(range(3, n + 3))
    assert all(shapes == batch_shapes[0] for shapes in batch_shapes)


@pytest.mark.parametrize("is_training", [True, False])
def test_single_process_batches_are_all_valid(dataset_factory, tokenizer, is_training):
    dataset, _, _ = dataset_factory()
    dataset.is_training = is_training
    batches = list(make_loader(dataset, tokenizer))
    assert sum(len(batch["valid_mask"]) for batch in batches) == len(dataset)
    assert all(batch["valid_mask"].all() for batch in batches)


@pytest.mark.parametrize("length_aware", [False, True])
@pytest.mark.parametrize("world_size", [2, 3, 4])
@pytest.mark.parametrize("n", [1, 5, 19, 33])
def test_training_padding_preserves_all_samples(dataset_factory, tokenizer, partition,
                                               length_aware, world_size, n):
    samples = [{"image_name": "shared.png", "text": "word " * (i + 1)} for i in range(n)]
    dataset, _, _ = dataset_factory(samples=samples, max_length=64)
    partition["num_replicas"] = world_size
    for epoch in (0, 1):
        indices, lengths, slots = [], [], 0
        for rank in range(world_size):
            partition["rank"] = rank
            loader = make_loader(dataset, tokenizer, **partition, drop_last=False,
                                 batch_size=3, length_aware=length_aware, max_length=64)
            dl.set_dataloader_epoch(loader, epoch)
            stream = list(loader.batch_sampler)
            assert stream == list(loader.batch_sampler)
            indices.extend(idx for batch in stream for idx, valid in batch if valid)
            batches = list(loader)
            assert len(batches) == len(loader) == math.ceil(n / (3 * world_size))
            for batch in batches:
                assert len(batch["image_ids"]) == 3
                slots += 3
                lengths.extend(batch["attention_mask"].sum(1)[batch["valid_mask"]].tolist())
        assert sorted(indices) == list(range(n))
        assert sorted(lengths) == list(range(3, n + 3))
        assert slots == math.ceil(n / (3 * world_size)) * 3 * world_size


@pytest.mark.parametrize("length_aware", [False, True])
@pytest.mark.parametrize("n, distributed", [(0, False), (1, False), (8, True)])
def test_build_loader_rejects_empty_batches(dataset_factory, tokenizer, partition,
                                           length_aware, n, distributed):
    dataset, _, _ = dataset_factory(word_counts=[1] * n)
    with pytest.raises(ValueError, match="empty|No batches"):
        make_loader(dataset, tokenizer, drop_last=True, **(partition if distributed else {}),
                    length_aware=length_aware)


def _distributed_loader_worker(rank, root):
    dist.init_process_group("gloo", init_method=(root / "rendezvous").as_uri(),
                               rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        partition = dict(rank=rank, num_replicas=2, pad_batches=True)
        tokenizer = DummyTokenizer()
        dataset = engine.build_dataset(root / "samples.json", root, True, root / "cache", 3,
                                     image_transform, "distributed", tokenizer, 24, distributed=True)
        loader = make_loader(dataset, tokenizer, **partition, drop_last=False, batch_size=3)
        dl.set_dataloader_epoch(loader, 2)
        train_ids = []
        for batch in loader:
            assert len(batch["image_ids"]) == 3
            sizes = [torch.zeros((), dtype=torch.long) for _ in range(2)]
            dist.all_gather(sizes, batch["valid_mask"].sum())
            train_ids.extend(batch["image_ids"][batch["valid_mask"]].tolist())
        dataset.is_training = False
        batches = list(make_loader(dataset, tokenizer, **partition))
        validation_ids = [idx for batch in batches
                          for idx in batch["image_ids"][batch["valid_mask"]].tolist()]
        result = dict(train=train_ids, validation=validation_ids,
                      validation_shapes=[len(batch["image_ids"]) for batch in batches],
                      tokenize_calls=tokenizer.calls)
        (root / f"rank{rank}.json").write_text(json.dumps(result))
        if dataset.conn is not None:
            dataset.conn.close()
        # A real filesystem failure on rank 0 must reach both ranks.
        if rank == 0:
            (root / "blocked_cache").write_text("not a directory")
        with pytest.raises(RuntimeError, match="FileExistsError") as exc:
            engine.build_dataset(root / "samples.json", root, True, root / "blocked_cache", 3,
                                 image_transform, "failed", tokenizer, 24, distributed=True)
        errors = [None, None]
        dist.all_gather_object(errors, str(exc.value))
        assert errors[0] == errors[1]
    finally:
        dist.destroy_process_group()


def test_real_process_group_cache_and_partition(tmp_path):
    require_multiprocessing()
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo unavailable")
    samples = [{"image_name": f"{i}.png", "text": "word " * (i + 1)} for i in range(33)]
    for sample in samples:
        Image.new("RGB", (8, 8)).save(tmp_path / sample["image_name"])
    (tmp_path / "samples.json").write_text(json.dumps(samples))
    torch.multiprocessing.spawn(_distributed_loader_worker, args=(tmp_path,), nprocs=2)
    first, second = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    assert len(first["train"] + second["train"]) == 33
    assert not set(first["train"]) & set(second["train"])
    assert len(set(first["validation"] + second["validation"])) == 33
    assert len(first["validation"] + second["validation"]) == 33
    assert first["validation_shapes"] == second["validation_shapes"] == [8, 8, 8]
    # Rank 0 alone tokenizes the shared cache: 11 chunks plus 6 train and 3 val batches.
    assert first["tokenize_calls"] == 20
    assert second["tokenize_calls"] == 9
