"""One checkpoint contract across engines and process-group topologies."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.trainingph1 import engine, engine_checkpoint as checkpoint
from src.queue.ema import EMA
from test_checkpoint_fsdp import fixture, tiny_model, tokenizer, prepare, snapshot, assert_equal


def prepare_single(settings):
    def model(options):
        online = tiny_model()
        return online, EMA(online.itc_encoder, options['MOMENTUM'])
    with patch.object(engine, 'get_model', side_effect=model), \
         patch.object(engine.AutoTokenizer, 'from_pretrained', side_effect=lambda *a, **k: tokenizer()):
        return engine.prepare_training(settings)


def full_snapshot(state):
    from torch.distributed.checkpoint.state_dict import get_state_dict, get_model_state_dict, StateDictOptions
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    model, optimizer = get_state_dict(state.model, state.optimizer, options=options)
    ema = get_model_state_dict(state.ema.model, options=options)
    return dict(model=model, optimizer=optimizer, ema=ema, queue=state.queue.state_dict(),
                scheduler=state.scheduler.state_dict(), progress=state.progress, events=state.events)


def cross_topology_worker(rank, folder):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + folder + '/store', rank=rank, world_size=2)
    try:
        settings = json.loads((Path(folder) / 'settings.json').read_text())
        state = prepare(settings)
        full = full_snapshot(state)
        if rank == 0:
            torch.save(full, Path(folder) / 'loaded.pt')
        # Check that every remaining global batch has exactly the same real samples.
        local_batches = list(state.train_loader.batch_sampler)
        gathered = [None, None]
        dist.all_gather_object(gathered, local_batches)
        if rank == 0:
            merged = [[index for local in gathered for index, valid in local[step] if valid]
                      for step in range(len(local_batches))]
            (Path(folder) / 'batches.json').write_text(json.dumps(merged))
        checkpoint.save_checkpoint(state)
        fresh = prepare(settings | dict(RESUME_CHECKPOINT=None, INIT_CHECKPOINT=str(state.run_dir),
                                        TRAINING_MODE='finetune', RUN_NAME='distributed_finetune'))
        assert not fresh.optimizer.state and fresh.progress['global_step'] == 0
        assert fresh.queue.count.item() == 0
        # A sharded update must survive a subsequent single-device load.
        from src.trainingph1.engine_fsdp import train_one_epoch
        train_one_epoch(state, state.progress['epoch'])
        full = full_snapshot(state)
        if rank == 0:
            torch.save(full, Path(folder) / 'updated.pt')
        checkpoint.save_checkpoint(state)
    finally:
        dist.destroy_process_group()


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = fixture(self.temp.name) | dict(RUN_NAME='single', TENSORBOARD_ENABLED=False)

    def test_single_and_reference_engine_share_resume_and_finetune_format(self):
        single = prepare_single(self.settings)
        engine.train_one_epoch(single, 0)
        checkpoint.save_checkpoint(single, best=True)
        expected = snapshot(single)
        source = str(single.run_dir)
        reference = prepare(self.settings | dict(RESUME_CHECKPOINT=source))
        assert_equal(snapshot(reference), expected)
        checkpoint.save_checkpoint(reference)
        resumed = prepare_single(self.settings | dict(RESUME_CHECKPOINT=source))
        assert_equal(snapshot(resumed), expected)
        fresh = prepare_single(self.settings | dict(TRAINING_MODE='finetune', INIT_CHECKPOINT=source,
                                                    RUN_NAME='finetune', LR=.002))
        assert_equal(snapshot(fresh)['model'], expected['model'])
        self.assertFalse(fresh.optimizer.state)
        self.assertEqual(fresh.optimizer.param_groups[0]['lr'], .002)
        self.assertEqual(fresh.progress, dict(epoch=0, next_batch=0, global_step=0))
        self.assertEqual(fresh.queue.count.item(), 0)
        self.assertEqual(fresh.events, {})
        for name, parameter in fresh.ema.model.named_parameters():
            torch.testing.assert_close(parameter, dict(fresh.model.itc_encoder.named_parameters())[name], rtol=0, atol=0)

    def test_save_is_atomic_and_preserves_rng_and_empty_optimizer(self):
        state = prepare_single(self.settings)
        before = snapshot(state)
        rng = checkpoint._rng(torch.device('cpu'))
        checkpoint.save_checkpoint(state, best=True)
        assert_equal(checkpoint._rng(torch.device('cpu')), rng)
        assert_equal(snapshot(state), before)
        pointers = {name: (state.run_dir / name).read_bytes() for name in ('last.json', 'best.json')}
        with patch.object(checkpoint.dcp, 'save', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                checkpoint.save_checkpoint(state, best=True)
        for name, value in pointers.items():
            self.assertEqual((state.run_dir / name).read_bytes(), value)
        resumed = prepare_single(self.settings | dict(RESUME_CHECKPOINT=str(state.run_dir)))
        assert_equal(snapshot(resumed), before)

    def test_offline_pretrained_and_metadata_validation(self):
        state = prepare_single(self.settings)
        checkpoint.save_checkpoint(state)
        with patch('transformers.BertModel.from_pretrained', side_effect=AssertionError('network')):
            model, token, settings = checkpoint.load_pretrained(str(state.run_dir))
        assert_equal(model.state_dict(), state.model.state_dict())
        self.assertEqual(token('caption caption'), state.tokenizer('caption caption'))
        path, metadata = checkpoint.read_checkpoint(str(state.run_dir))
        metadata['architecture']['args']['num_queries'] += 1
        with self.assertRaisesRegex(RuntimeError, 'architecture'):
            checkpoint.load_checkpoint(path, metadata, state)
        (path / 'checkpoint.json').unlink()
        with self.assertRaises(RuntimeError):
            checkpoint.read_checkpoint(str(state.run_dir))

    def test_checkpoint_directory_failure_is_broadcast(self):
        with patch.object(checkpoint, '_rank', return_value=0), \
             patch.object(checkpoint.dist, 'is_initialized', return_value=True), \
             patch.object(checkpoint.dist, 'broadcast_object_list') as broadcast, \
             patch.object(Path, 'mkdir', side_effect=PermissionError('denied')):
            from types import SimpleNamespace
            state = SimpleNamespace(settings=self.settings, model=tiny_model(), tokenizer=tokenizer())
            # Config defaults are needed before create_run validation.
            from src import config
            state.settings = vars(config) | self.settings
            with self.assertRaisesRegex(RuntimeError, 'denied'):
                checkpoint.create_run(state)
        broadcast.assert_called_once()

    @unittest.skipUnless(os.environ.get('RUN_DISTRIBUTED_TESTS') == '1', 'requires local process sockets')
    def test_single_to_two_shards_and_back_mid_epoch(self):
        # Odd global size also exercises padding when the topology changes.
        state = prepare_single(self.settings | dict(BATCH_SIZE=3, ACCUMULATION_STEPS=1))
        class Stop(Exception):
            pass
        def stop(current, metrics):
            checkpoint.save_checkpoint(current)
            raise Stop()
        with self.assertRaises(Stop):
            engine.train_one_epoch(state, 0, on_step=stop)
        expected = full_snapshot(state)
        batches = [[index for index, valid in batch if valid] for batch in state.train_loader.batch_sampler]
        settings = self.settings | dict(RESUME_CHECKPOINT=str(state.run_dir))
        (Path(self.temp.name) / 'settings.json').write_text(json.dumps(settings))
        mp.spawn(cross_topology_worker, args=(self.temp.name,), nprocs=2, join=True)
        assert_equal(torch.load(Path(self.temp.name) / 'loaded.pt', weights_only=True), expected)
        self.assertEqual(json.loads((Path(self.temp.name) / 'batches.json').read_text()), batches)
        restored = prepare_single(settings)
        assert_equal(full_snapshot(restored), torch.load(Path(self.temp.name) / 'updated.pt', weights_only=True))
        fresh = prepare_single(self.settings | dict(TRAINING_MODE='finetune', INIT_CHECKPOINT=str(state.run_dir),
                                                    RUN_NAME='single_finetune'))
        assert_equal(fresh.model.state_dict(), restored.model.state_dict())
        self.assertFalse(fresh.optimizer.state)
        engine.train_one_epoch(restored, restored.progress['epoch'])
        self.assertGreater(restored.progress['global_step'], state.progress['global_step'])
