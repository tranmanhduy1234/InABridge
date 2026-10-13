import copy
import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import BertConfig, DINOv3ViTConfig, PreTrainedTokenizerFast

from src import config
from src.trainingph1.model import ModelStage1
from src.trainingph1.engine_fsdp import compute_loss, init_runtime, prepare_training, run_training, train_one_epoch, validate
from src.trainingph1 import engine_checkpoint as checkpoint


def tiny_model(*args, **kwargs):
    return ModelStage1('tiny', -1, 'tiny', 2, 1, 8, 4, 1e-12,
                      bert_config=BertConfig(vocab_size=5, hidden_size=8, num_hidden_layers=1,
                                             num_attention_heads=2, intermediate_size=16,
                                             hidden_dropout_prob=.1, attention_probs_dropout_prob=.1),
                      vision_config=DINOv3ViTConfig(hidden_size=8, num_hidden_layers=1,
                                                   num_attention_heads=2, intermediate_size=16,
                                                   image_size=16, patch_size=8),
                      vision_mock=kwargs.get('vision_mock', True))


def tokenizer():
    backend = Tokenizer(models.WordLevel(
        {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, 'caption': 4}, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single='[CLS] $A [SEP]', special_tokens=[('[CLS]', 2), ('[SEP]', 3)])
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]')


def fixture(folder):
    folder = Path(folder)
    records = []
    for i in range(9):
        name = f'{i}.png'
        pixels = np.random.default_rng(i).integers(0, 256, (20, 20, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(folder / name)
        records.append(dict(image_name=name, text='caption ' * (i % 5 + 1)))
    manifest = folder / 'manifest.json'
    manifest.write_text(json.dumps(records))
    return dict(DEVICE='cpu', AMP_ENABLED=False, EPOCHS=2, WARMUP_EPOCHS=0, TRAINING_MODE='pretrain',
                ACCUMULATION_STEPS=2, JSON_PATH_TRAIN=str(manifest), JSON_PATH_VAL=str(manifest),
                IMAGE_DIR=str(folder), CACHE_DIR=str(folder / 'cache'), IMAGE_SIZE=16,
                BATCH_SIZE=2, DROP_LAST=False, NUM_WORKERS=0, MAX_LENGTH=8, QUEUE_SIZE=9,
                INIT_CHECKPOINT=None, RESUME_CHECKPOINT=None, SAVE_EVERY_STEPS=1,
                VAL_EVERY_STEPS=1, LOG_EVERY_STEPS=0, CHECKPOINT_DIR=str(folder / 'checkpoints'))


def prepare(settings):
    settings = {k: v for k, v in vars(config).items() if k.isupper()} | settings
    return prepare_training(settings, init_runtime(settings))


def snapshot(state):
    def clone(value):
        if isinstance(value, torch.Tensor):
            if hasattr(value, 'to_local'):
                value = value.to_local()
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: clone(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clone(v) for v in value]
        return copy.deepcopy(value)
    result = {name: clone(getattr(state, name).state_dict())
              for name in ('model', 'optimizer', 'scheduler', 'ema', 'queue')}
    result.update(progress=copy.deepcopy(state.progress), events=copy.deepcopy(state.events))
    return result


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    else:
        assert left == right, (left, right)


class Interrupted(Exception):
    pass


def resume_scenario(settings, mode='pretrain', source=None):
    settings = settings | dict(TRAINING_MODE=mode, INIT_CHECKPOINT=source)
    with patch('src.trainingph1.engine_fsdp.AutoTokenizer.from_pretrained', side_effect=lambda *a, **k: tokenizer()), \
         patch('src.trainingph1.engine_fsdp.ModelStage1', side_effect=tiny_model):
        baseline = run_training(settings | dict(RUN_NAME=mode + '_baseline'))
        expected = snapshot(baseline)
        expected_rng = (random.random(), np.random.rand(), torch.rand(3))
        captured = {}

        def interrupt(state, metrics):
            captured['run_dir'] = state.run_dir
            captured['metrics'] = metrics
            raise Interrupted()

        try:
            run_training(settings | dict(RUN_NAME=mode + '_interrupted'), on_step=interrupt)
        except Interrupted:
            pass
        else:
            raise AssertionError('Missing interruption')
    path = captured['run_dir'] / 'last.json'
    checkpoint_path, _ = checkpoint.read_checkpoint(path)
    shared = checkpoint.read_training_state(checkpoint_path)
    assert shared['progress'] == dict(global_step=1, epoch=0, next_batch=2)
    with patch('transformers.BertModel.from_pretrained', side_effect=AssertionError('network')), \
         patch('src.trainingph1.engine_fsdp.AutoTokenizer.from_pretrained', side_effect=AssertionError('network')):
        resumed = run_training(settings | dict(INIT_CHECKPOINT=None, RESUME_CHECKPOINT=str(path),
                                               LR=.5, EPOCHS=99, RUN_NAME='ignored',
                                               TENSORBOARD_DIAGNOSTICS_EVERY_STEPS=1, LOG_LEVEL='WARNING'))
    assert resumed.run_dir == captured['run_dir']
    assert resumed.settings['EPOCHS'] == settings['EPOCHS']
    assert resumed.settings['TENSORBOARD_DIAGNOSTICS_EVERY_STEPS'] == 1
    assert resumed.settings['LOG_LEVEL'] == 'WARNING'
    assert_equal(snapshot(resumed), expected)
    assert_equal([random.random(), np.random.rand(), torch.rand(3)], list(expected_rng))
    assert len(list(resumed.run_dir.joinpath('snapshots').iterdir())) <= 2
    return resumed


def distributed_worker(rank, folder):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + folder + '/store', rank=rank, world_size=2)
    try:
        settings = json.loads((Path(folder) / 'settings.json').read_text())
        try:
            prepare(settings | dict(LR=.001 * (rank + 1)))
        except ValueError as exc:
            assert 'same training settings' in str(exc)
        else:
            raise AssertionError('Rank-dependent settings must be rejected')
        state = resume_scenario(settings)
        resume_scenario(settings, 'finetune', str(state.run_dir / 'best.json'))
        before = (state.run_dir / 'last.json').read_bytes()
        parameter = next(p for p in state.model.parameters() if p.requires_grad)
        if rank == 1:
            parameter.grad = torch.ones_like(parameter)
        try:
            checkpoint.save_checkpoint(state)
        except RuntimeError as exc:
            assert 'rank 1' in str(exc) and 'zero_grad' in str(exc)
        else:
            raise AssertionError('Invalid save boundary must fail on every rank')
        finally:
            parameter.grad = None
        assert (state.run_dir / 'last.json').read_bytes() == before
        original_save = torch.save

        def fail_one_rank(value, path, *args, **kwargs):
            if rank == 1 and hasattr(path, 'getvalue'):
                raise OSError('rank-local disk full')
            return original_save(value, path, *args, **kwargs)

        with patch.object(checkpoint.torch, 'save', side_effect=fail_one_rank):
            try:
                checkpoint.save_checkpoint(state)
            except RuntimeError as exc:
                assert 'rank-local disk full' in str(exc)
            else:
                raise AssertionError('Checkpoint failure must reach every rank')
        assert (state.run_dir / 'last.json').read_bytes() == before
    finally:
        dist.destroy_process_group()


class CheckpointFSDPTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = fixture(self.temp.name)
        self.token_patch = patch('src.trainingph1.engine_fsdp.AutoTokenizer.from_pretrained',
                                side_effect=lambda *a, **k: tokenizer())
        self.model_patch = patch('src.trainingph1.engine_fsdp.ModelStage1', side_effect=tiny_model)
        self.token_patch.start()
        self.model_mock = self.model_patch.start()
        self.addCleanup(self.token_patch.stop)
        self.addCleanup(self.model_patch.stop)

    def test_real_backbone_training_updates_only_trainable_parameters(self):
        state = prepare(self.settings | dict(RUN_NAME='updates', VISION_MOCK=False))
        before = {name: p.detach().clone() for name, p in state.model.named_parameters()}
        trainable = {id(p) for p in state.model.parameters() if p.requires_grad}
        optimized = [id(p) for group in state.optimizer.param_groups for p in group['params']]
        self.assertEqual(set(optimized), trainable)
        self.assertEqual(len(optimized), len(trainable))
        metrics = train_one_epoch(state, 0)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        changed = set()
        for name, parameter in state.model.named_parameters():
            self.assertIsNone(parameter.grad)
            if parameter.requires_grad:
                if not torch.equal(parameter, before[name]):
                    changed.add(name)
            else:
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        for prefix in ('dec_embedding', 'itc_encoder.qformer.', 'itc_encoder.query_proj.',
                       'itc_encoder.text_proj.', 'itm_logit.'):
            self.assertTrue(any(name.startswith(prefix) for name in changed), prefix)
        self.assertIs(state.model.lm_head.weight, state.model.embeddings.word_embeddings.weight)
        self.assertFalse(state.model.vision_encoder.model.training)
        self.assertFalse(state.model.embeddings.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in state.ema.parameters()))
        self.assertEqual(state.progress, dict(epoch=1, next_batch=0, global_step=3))
        self.assertEqual(state.scheduler.last_epoch, 3)
        self.assertEqual(state.queue.count.item(), 9)

    def test_validation_preserves_training_state_and_rng_even_on_failure(self):
        state = prepare(self.settings | dict(RUN_NAME='validation'))
        train_one_epoch(state, 0)
        for training in (True, False):
            for fail in (False, True):
                with self.subTest(training=training, fail=fail):
                    state.model.train(training)
                    before = snapshot(state)
                    rng = checkpoint._rng(state.runtime.device)
                    generator = state.val_loader.generator.get_state().clone()
                    sentinel = object()
                    state.diagnostics = sentinel
                    if fail:
                        with patch('src.trainingph1.engine_fsdp.compute_loss', side_effect=RuntimeError('validation failed')):
                            with self.assertRaisesRegex(RuntimeError, 'validation failed'):
                                validate(state)
                    else:
                        first = validate(state)
                        self.assertEqual(first, validate(state))
                        self.assertTrue(all(np.isfinite(v) for v in first.values()))
                    assert_equal(snapshot(state), before)
                    assert_equal(checkpoint._rng(state.runtime.device), rng)
                    torch.testing.assert_close(state.val_loader.generator.get_state(), generator, rtol=0, atol=0)
                    self.assertEqual(state.model.training, training)
                    self.assertIs(state.diagnostics, sentinel)

    def test_final_checkpoint_without_periodic_events_resumes_completed_run(self):
        settings = self.settings | dict(RUN_NAME='final_only', EPOCHS=1, SAVE_EVERY_STEPS=0,
                                        VAL_EVERY_STEPS=0, TENSORBOARD_ENABLED=False)
        state = run_training(settings)
        self.assertTrue((state.run_dir / 'last.json').is_file())
        self.assertFalse((state.run_dir / 'best.json').exists())
        self.assertEqual(state.events, {})
        expected = snapshot(state)
        with patch('src.trainingph1.engine_fsdp.train_one_epoch', side_effect=AssertionError('already complete')):
            resumed = run_training(settings | dict(RESUME_CHECKPOINT=str(state.run_dir)))
        assert_equal(snapshot(resumed), expected)
        self.assertEqual(len(list((state.run_dir / 'snapshots').iterdir())), 1)

    def test_invalid_numeric_settings_fail_before_model_loading(self):
        cases = [('AMP_ENABLED', True, 'BF16')]
        for name in ('ACCUMULATION_STEPS', 'EPOCHS'):
            cases.extend((name, value, 'positive integers') for value in (0, -1, 1.5, True))
        for name in ('SAVE_EVERY_STEPS', 'VAL_EVERY_STEPS', 'LOG_EVERY_STEPS',
                     'TENSORBOARD_DIAGNOSTICS_EVERY_STEPS', 'TENSORBOARD_TEXT_SAMPLES'):
            cases.extend((name, value, name) for value in (-1, 1.5, True))
        cases.extend(('MAX_GRAD_NORM', value, 'MAX_GRAD_NORM') for value in (0, -1, float('nan'), float('inf')))
        for name, value, message in cases:
            with self.subTest(name=name, value=value):
                self.model_mock.reset_mock()
                with self.assertRaisesRegex(ValueError, message):
                    prepare(self.settings | {name: value, 'AMP_DTYPE': 'float16'})
                self.model_mock.assert_not_called()

    def test_checkpoint_rejects_pending_gradients_and_partial_accumulation(self):
        state = prepare(self.settings | dict(RUN_NAME='boundary'))
        checkpoint.save_checkpoint(state)
        pointer = state.run_dir / 'last.json'
        original = pointer.read_bytes()
        parameter = next(p for p in state.model.parameters() if p.requires_grad)
        parameter.grad = torch.ones_like(parameter)
        with self.assertRaisesRegex(RuntimeError, 'zero_grad'):
            checkpoint.save_checkpoint(state)
        parameter.grad = None
        state.progress['next_batch'] = 1
        with self.assertRaisesRegex(RuntimeError, 'accumulation boundary'):
            checkpoint.save_checkpoint(state)
        self.assertEqual(pointer.read_bytes(), original)
        self.assertEqual(len(list((state.run_dir / 'snapshots').iterdir())), 1)

    def test_pretrain_and_finetune_resume_exactly(self):
        state = resume_scenario(self.settings)
        source = str(state.run_dir / 'last.json')
        fresh = prepare(self.settings | dict(TRAINING_MODE='finetune', INIT_CHECKPOINT=source, RUN_NAME='fresh',
                                             LR=.002, NUM_QUERIES=99, ITC_DIM=99))
        assert_equal(snapshot(fresh)['model'], snapshot(state)['model'])
        self.assertFalse(fresh.optimizer.state)
        self.assertEqual(fresh.optimizer.param_groups[0]['lr'], .002)
        self.assertEqual(fresh.queue.count.item(), 0)
        self.assertEqual(fresh.progress, dict(epoch=0, next_batch=0, global_step=0))
        for name, p in fresh.ema.model.named_parameters():
            torch.testing.assert_close(p, dict(fresh.model.itc_encoder.named_parameters())[name], rtol=0, atol=0)
        resume_scenario(self.settings, 'finetune', source)

    def test_resume_replays_augmentation_with_workers(self):
        resume_scenario(self.settings | dict(NUM_WORKERS=2, PERSISTENT_WORKERS=True))

    def test_save_failure_keeps_previous_checkpoint_and_empty_optimizer(self):
        state = prepare(self.settings | dict(RUN_NAME='empty'))
        before = snapshot(state)
        checkpoint.save_checkpoint(state)
        self.assertFalse(state.optimizer.state)
        assert_equal(snapshot(state), before)
        pointer = state.run_dir / 'last.json'
        original = pointer.read_bytes()
        with patch.object(checkpoint.dcp, 'save', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                checkpoint.save_checkpoint(state)
        self.assertEqual(pointer.read_bytes(), original)
        restored = prepare(self.settings | dict(RESUME_CHECKPOINT=str(pointer)))
        assert_equal(snapshot(restored), before)
        self.assertFalse(restored.optimizer.state)

    def test_resume_after_nonfinite_epoch_preserves_empty_optimizer(self):
        state = prepare(self.settings | dict(RUN_NAME='skipped'))

        def nonfinite(*args):
            losses, keys = compute_loss(*args)
            losses['loss_itc_sum'] = losses['loss_itc_sum'] * float('nan')
            return losses, keys

        with patch('src.trainingph1.engine_fsdp.compute_loss', side_effect=nonfinite):
            metrics = train_one_epoch(state, 0)
        self.assertEqual(metrics['skipped_steps'], 3)
        self.assertEqual(metrics['loss'], 0.)
        checkpoint.save_checkpoint(state)
        self.assertFalse(state.optimizer.state)
        train_one_epoch(state, 1)
        expected = snapshot(state)
        resumed = prepare(self.settings | dict(RESUME_CHECKPOINT=str(state.run_dir)))
        self.assertEqual(resumed.progress, dict(epoch=1, next_batch=0, global_step=0))
        self.assertFalse(resumed.optimizer.state)
        train_one_epoch(resumed, 1)
        assert_equal(snapshot(resumed), expected)

    def test_training_logs_metrics_skips_validation_and_checkpoints(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        calls = 0

        def nonfinite_first_batch(*args):
            nonlocal calls
            losses, keys = compute_loss(*args)
            if calls == 0:
                losses['loss_itc_sum'] = losses['loss_itc_sum'] * float('nan')
            calls += 1
            return losses, keys

        with patch('src.trainingph1.engine_fsdp.compute_loss', side_effect=nonfinite_first_batch):
            state = run_training(self.settings | dict(RUN_NAME='logging', EPOCHS=1, LOG_EVERY_STEPS=1))
        log = (state.run_dir / 'logs/rank-0.log').read_text()
        for message in ('reason=nonfinite_loss', 'train=', 'new best validation', 'checkpoint saved',
                        'training completed', 'skipped_steps'):
            self.assertIn(message, log)
        events = EventAccumulator(str(state.run_dir / 'tensorboard')).Reload()
        self.assertTrue({'train/loss', 'train/lr', 'train/grad_norm', 'train/queue_count',
                         'train/samples_per_second', 'validation/loss', 'checkpoint/save_seconds',
                         'epoch/skipped_steps'} <= set(events.Tags()['scalars']))
        self.assertEqual(events.Scalars('epoch/skipped_steps')[-1].value, 1.)
        self.assertTrue({'train/diagnostics/itc/i2t/recall_at_1', 'train/diagnostics/gradients/qformer/norm',
                         'validation/diagnostics/itm/balanced_accuracy', 'validation/diagnostics/itg/token_accuracy',
                         'validation/diagnostics/itg/perplexity'} <= set(events.Tags()['scalars']))
        self.assertIn('validation/distributions/itm/positive_probability', events.Tags()['histograms'])
        self.assertIn('validation/examples/image_0', events.Tags()['images'])
        self.assertIn('validation/examples/teacher_forced/text_summary', events.Tags()['tensors'])
        self.assertIsNone(state.tb.writer)
        self.assertFalse(state.logger.handlers)

    def test_real_vision_architecture_is_reconstructed_offline(self):
        state = prepare(self.settings | dict(RUN_NAME='real_vision', VISION_MOCK=False))
        self.assertFalse(state.model.vision_encoder.mock)
        checkpoint.save_checkpoint(state)
        with patch('transformers.DINOv3ViTModel.from_pretrained', side_effect=AssertionError('network')):
            restored = prepare(self.settings | dict(RESUME_CHECKPOINT=str(state.run_dir)))
        self.assertFalse(restored.model.vision_encoder.mock)
        assert_equal(snapshot(restored), snapshot(state))

    def test_invalid_modes_metadata_and_data_are_rejected(self):
        for values in (dict(TRAINING_MODE='other'), dict(TRAINING_MODE='finetune'),
                       dict(INIT_CHECKPOINT='a'), dict(INIT_CHECKPOINT='a', RESUME_CHECKPOINT='b')):
            with self.subTest(values=values), self.assertRaises(ValueError):
                prepare(self.settings | values)
        state = prepare(self.settings | dict(RUN_NAME='checks'))
        checkpoint.save_checkpoint(state)
        source = str(state.run_dir)
        with self.assertRaisesRegex(ValueError, 'TRAINING_MODE'):
            prepare(self.settings | dict(TRAINING_MODE='finetune', RESUME_CHECKPOINT=source))
        path, metadata = checkpoint.read_checkpoint(source)
        manifest = Path(self.settings['JSON_PATH_TRAIN'])
        manifest.write_text(manifest.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            prepare(self.settings | dict(RESUME_CHECKPOINT=source))
        metadata['run_id'] = 'changed'
        (state.run_dir / 'run.json').write_text(json.dumps(metadata))
        with self.assertRaisesRegex(RuntimeError, 'digest'):
            checkpoint.read_checkpoint(source)

    @unittest.skipUnless(os.environ.get('RUN_DISTRIBUTED_TESTS') == '1', 'requires local process sockets')
    def test_two_rank_checkpoint_and_resume(self):
        (Path(self.temp.name) / 'settings.json').write_text(json.dumps(self.settings))
        mp.spawn(distributed_worker, args=(self.temp.name,), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
