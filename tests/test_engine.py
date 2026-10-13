import copy
import os
import random

import numpy as np
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import BertConfig, DINOv3ViTConfig

from src import config
from src.trainingph1 import engine as single
from src.trainingph1 import engine_fsdp as fsdp
from src.losses.lossph1 import Stage1Criterion
from src.queue.ema import EMA
from src.queue.moco import MoCoQueue
from src.trainingph1.engine import hepler_compute_loss, train_one_epoch, run_training, validate, _training_events, get_data_fingerprints
from src.trainingph1.model import ModelStage1


class EngineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        model = ModelStage1(
            'tiny-bert', -1, 'tiny-vision', 2, 1, 8, 4, 1e-12,
            bert_config=BertConfig(vocab_size=8, hidden_size=8, num_hidden_layers=1,
                                   num_attention_heads=2, intermediate_size=16),
            vision_config=DINOv3ViTConfig(hidden_size=8, num_hidden_layers=1,
                                         num_attention_heads=2, intermediate_size=16,
                                         image_size=16, patch_size=8),
        )
        settings = {k: v for k, v in vars(config).items() if k.isupper()}
        settings.update(DEVICE='cpu', AMP_ENABLED=False, EPOCHS=2, ACCUMULATION_STEPS=2,
                        MAX_GRAD_NORM=None, ITC_WEIGHT=1, ITM_WEIGHT=2, ITG_WEIGHT=3)
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=.01)
        self.state = SimpleNamespace(
            model=model, ema=EMA(model.itc_encoder, .9), queue=MoCoQueue(16, 2, 4),
            criterion=Stage1Criterion(), settings=settings, optimizer=optimizer,
            scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.),
            scaler=None, progress=dict(epoch=0, next_batch=0, global_step=0), events={}, epoch_metrics=None,
            train_loader=[self.batch(2), self.batch(2), self.batch(1)],
        )

    def batch(self, size):
        ids = torch.tensor([[1, i + 2, 7, 0] for i in range(size)])
        return dict(images=torch.randn(size, 3, 16, 16), image_ids=torch.arange(size), input_ids=ids,
                    attention_mask=ids.ne(0))

    def test_helper_excludes_padded_samples_from_loss_and_queue_keys(self):
        self.state.model.eval()
        batch = self.batch(3)
        batch['valid_mask'] = torch.tensor([True, True, False])
        trimmed = {name: value[:2] for name, value in batch.items()}
        expected, _ = hepler_compute_loss(self.state, trimmed, .2)
        actual, keys = hepler_compute_loss(self.state, batch, .2)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name])
        self.assertEqual(keys['image_ids'].tolist(), [0, 1])
        batch['valid_mask'].zero_()
        actual, keys = hepler_compute_loss(self.state, batch, .2)
        self.assertTrue(all(value.item() == 0 for value in actual.values()))
        self.assertEqual(len(keys['image_ids']), 0)
        actual['loss'].backward()

    def test_helper_samples_online_inbatch_and_does_not_update_state(self):
        state, batch = self.state, self.batch(3)
        state.model.eval()
        state.queue.enqueue(torch.randn(5, 2, 4), torch.randn(5, 4), torch.arange(5))
        before = copy.deepcopy(state.queue.state_dict())
        with torch.no_grad():
            state.ema.model.text_proj.weight.neg_()
            features = state.model.encode_image(batch['images'])
            online = state.model(features, batch['input_ids'], batch['attention_mask'], 'itc')
            scores = torch.einsum('bqd,kd->bkq', online['image_features'], online['text_features']).amax(-1)
            scores = scores / state.settings['TEMPERATURE']
            scores.fill_diagonal_(-torch.inf)
        with patch('torch.multinomial', wraps=torch.multinomial) as sample, \
             patch.object(state.model, 'forward', wraps=state.model.forward) as forward:
            losses, momentum = hepler_compute_loss(state, batch, .2)
        torch.testing.assert_close(sample.call_args_list[0].args[0], scores.softmax(-1))
        torch.testing.assert_close(sample.call_args_list[1].args[0], scores.T.softmax(-1))
        itm_calls = [c for c in forward.call_args_list if c.args[-1] == 'itm']
        self.assertEqual(len(itm_calls), 3)
        self.assertTrue(all(c.args[0].shape[0] == 3 for c in itm_calls))
        self.assertTrue(torch.all(itm_calls[2].args[1][:, 1] != batch['input_ids'][:, 1]))
        self.assertTrue(torch.all((itm_calls[1].args[0] - features).flatten(1).abs().sum(1) > 0))
        with torch.no_grad():
            logits = torch.cat([state.model(*c.args)['itm_logits'] for c in itm_calls])
            labels = torch.tensor([1.] * 3 + [0.] * 6)
            out = state.criterion.get_itm_loss(logits, labels)
            expected = out['loss_itm_sum'] / out['count_itm']
        torch.testing.assert_close(losses['loss_itm'], expected)
        torch.testing.assert_close(losses['loss'], losses['loss_itc'] + 2*losses['loss_itm'] + 3*losses['loss_itg'])
        losses['loss'].backward()
        self.assertTrue(state.model.itm_logit.weight.grad.isfinite().all())
        self.assertTrue(all(p.grad is None for p in state.ema.parameters()))
        self.assertTrue(all(not t.requires_grad for t in momentum.values()))
        for name, value in before.items():
            torch.testing.assert_close(value, state.queue.state_dict()[name])

    def test_singleton_skips_itm_and_helper_supports_no_grad(self):
        with torch.no_grad(), patch.object(self.state.model, 'forward', wraps=self.state.model.forward) as forward:
            losses, _ = hepler_compute_loss(self.state, self.batch(1), 0.)
        self.assertEqual(losses['loss_itm'].item(), 0.)
        self.assertTrue(losses['loss'].isfinite())
        self.assertNotIn('itm', [call.args[-1] for call in forward.call_args_list])

    def test_itm_excludes_every_caption_of_same_image(self):
        batch = self.batch(3)
        batch['image_ids'] = torch.tensor([10, 10, 20])
        with patch('torch.multinomial', wraps=torch.multinomial) as sample:
            losses, _ = hepler_compute_loss(self.state, batch, 0.)
        for call in sample.call_args_list:
            weights = call.args[0]
            torch.testing.assert_close(weights[:2], torch.tensor([[0., 0., 1.], [0., 0., 1.]]))
            self.assertEqual(weights[2, 2].item(), 0.)
        self.assertTrue(losses['loss'].isfinite())
        batch['image_ids'].fill_(10)
        with patch('torch.multinomial', side_effect=AssertionError('no negative available')):
            losses, _ = hepler_compute_loss(self.state, batch, 0.)
        self.assertEqual(losses['loss_itm'].item(), 0.)

    def test_run_training_updates_all_losses_ema_and_queue(self):
        state = self.state
        before = state.model.itc_encoder.query_proj.weight.detach().clone()
        with patch('src.trainingph1.engine.prepare_training', return_value=state), \
             patch('src.trainingph1.engine._save_state') as save, \
             patch('src.trainingph1.engine.validate', return_value={'loss': 1.}):
            result = run_training()
        self.assertIs(result, state)
        self.assertEqual(state.progress, dict(epoch=2, next_batch=0, global_step=4))
        self.assertEqual(save.call_args.args, (state,))
        self.assertEqual(state.scheduler.last_epoch, 4)
        self.assertEqual(state.queue.count.item(), 10)
        self.assertFalse(torch.equal(before, state.model.itc_encoder.query_proj.weight))
        self.assertTrue(all(p.grad is None for p in state.model.parameters()))

    def test_validation_preserves_training_state_and_rng(self):
        state = self.state
        state.val_loader = [self.batch(3), self.batch(1)]
        state.model.train()
        before = copy.deepcopy(state.model.state_dict())
        queue_before = copy.deepcopy(state.queue.state_dict())
        progress = state.progress.copy()
        rng = torch.get_rng_state()
        result = validate(state)
        self.assertEqual(set(result), {'loss', 'loss_itc', 'loss_itm', 'loss_itg'})
        self.assertTrue(all(isinstance(v, float) and torch.isfinite(torch.tensor(v)) for v in result.values()))
        self.assertTrue(state.model.training)
        self.assertEqual(state.progress, progress)
        self.assertTrue(all(p.grad is None for p in state.model.parameters()))
        torch.testing.assert_close(torch.get_rng_state(), rng)
        for name, value in before.items():
            torch.testing.assert_close(value, state.model.state_dict()[name])
        for name, value in queue_before.items():
            torch.testing.assert_close(value, state.queue.state_dict()[name])
        self.assertEqual(validate(state), result)

    def test_step_events_run_after_updates_and_skip_failed_steps(self):
        state = self.state
        state.settings.update(SAVE_EVERY_STEPS=1, VAL_EVERY_STEPS=1, LOG_EVERY_STEPS=1)
        saved = []
        def save(current, *, best=False):
            self.assertTrue(all(p.grad is None for p in current.model.parameters()))
            self.assertEqual(current.scheduler.last_epoch, current.progress['global_step'])
            saved.append((best, current.progress['global_step'], current.progress['next_batch'],
                          current.queue.count.item()))
        with patch('src.trainingph1.engine._save_state', side_effect=save), \
             patch('src.trainingph1.engine.validate', side_effect=[{'loss': 1.}, {'loss': 2.}]) as validation, \
             patch('builtins.print') as log:
            train_one_epoch(state, 0, on_step=_training_events)
            _training_events(state, None, final=True)
            self.assertEqual(saved, [(True, 1, 2, 4), (False, 2, 3, 5), (False, 2, 0, 5)])
            self.assertEqual(validation.call_count, 2)
            self.assertEqual(log.call_count, 4)
            self.assertEqual(state.events['best_val_loss'], 1.)
            hook = state.model.itc_encoder.query_proj.weight.register_hook(lambda grad: grad * float('inf'))
            try:
                train_one_epoch(state, 1, on_step=_training_events)
            finally:
                hook.remove()
            self.assertEqual(len(saved), 3)
            self.assertEqual(validation.call_count, 2)

    def test_disabled_intervals_still_save_final_checkpoint(self):
        self.state.settings.update(SAVE_EVERY_STEPS=0, VAL_EVERY_STEPS=0, LOG_EVERY_STEPS=0)
        with patch('src.trainingph1.engine._save_state') as save, \
             patch('src.trainingph1.engine.validate') as validation:
            train_one_epoch(self.state, 0, on_step=_training_events)
            save.assert_not_called()
            _training_events(self.state, None, final=True)
            save.assert_called_once_with(self.state, best=False)
            validation.assert_not_called()

    def test_data_fingerprints_detect_same_size_changes_and_ignore_location(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = dict(JSON_PATH_TRAIN=str(root / 'train.json'),
                            JSON_PATH_VAL=str(root / 'val.json'), CACHE_DIR=str(root))
            files = ['train.json', 'val.json', 'custom_train.sqlite', 'custom_validation.sqlite']
            loaders = [SimpleNamespace(dataset=SimpleNamespace(cache_path=root / name)) for name in files[2:]]
            for name in files:
                (root / name).write_bytes(b'original')
            original = get_data_fingerprints(settings, loaders)
            for name in files:
                (root / name).write_bytes(b'modified')
                self.assertNotEqual(get_data_fingerprints(settings, loaders), original)
                (root / name).write_bytes(b'original')
            moved = root / 'moved'
            moved.mkdir()
            for name in files:
                (root / name).rename(moved / name)
            relocated = {key: str(moved / Path(value).name) if key != 'CACHE_DIR' else str(moved)
                         for key, value in settings.items()}
            relocated_loaders = [SimpleNamespace(dataset=SimpleNamespace(cache_path=moved / name)) for name in files[2:]]
            self.assertEqual(get_data_fingerprints(relocated, relocated_loaders), original)

    def test_validation_averages_by_samples_pairs_and_tokens(self):
        state = self.state
        state.val_loader = [self.batch(2), self.batch(1)]
        state.val_loader[1]['attention_mask'][:, 2:] = False
        outputs = [({'loss_itc': torch.tensor(2.), 'loss_itm': torch.tensor(3.),
                     'loss_itg': torch.tensor(4.)}, {}),
                   ({'loss_itc': torch.tensor(8.), 'loss_itm': torch.tensor(0.),
                     'loss_itg': torch.tensor(9.)}, {})]
        with patch('src.trainingph1.engine.hepler_compute_loss', side_effect=outputs) as helper:
            losses = validate(state, pseudo_weight=.1)
        self.assertEqual(losses, dict(loss_itc=4., loss_itm=3., loss_itg=5., loss=25.))
        self.assertEqual(helper.call_args.args[-1], .1)
        state.val_loader = []
        state.model.eval()
        with self.assertRaisesRegex(ValueError, 'empty'):
            validate(state)
        self.assertFalse(state.model.training)
        state.model.train()
        state.val_loader = [self.batch(2)]
        with patch('src.trainingph1.engine.hepler_compute_loss', side_effect=RuntimeError('failure')):
            with self.assertRaisesRegex(RuntimeError, 'failure'):
                validate(state)
        self.assertTrue(state.model.training)

    def test_accumulation_tail_uses_actual_group_size(self):
        state = self.state
        state.model = torch.nn.Module()
        state.model.itc_encoder = torch.nn.Linear(1, 1, bias=False)
        state.model.itc_encoder.weight.data.fill_(1.)
        state.settings.update(ITC_WEIGHT=1, ITM_WEIGHT=0, ITG_WEIGHT=0)
        state.ema = EMA(state.model.itc_encoder, .9)
        state.optimizer = torch.optim.SGD(state.model.parameters(), lr=.1)
        state.scheduler = torch.optim.lr_scheduler.LambdaLR(state.optimizer, lambda step: 1.)
        def loss(*args):
            return {name: state.model.itc_encoder.weight.square().sum()
                    for name in ('loss_itc', 'loss_itm', 'loss_itg')}, {
                'image_features': torch.zeros(1, 2, 4), 'text_features': torch.zeros(1, 4), 'image_ids': torch.zeros(1, dtype=torch.long)}
        with patch('src.trainingph1.engine.hepler_compute_loss', side_effect=loss):
            train_one_epoch(state, 0)
        torch.testing.assert_close(state.model.itc_encoder.weight, torch.tensor([[.64]]))
        self.assertEqual(state.progress['global_step'], 2)

    def test_accumulation_weights_each_objective_by_its_count(self):
        state = self.state
        state.model = torch.nn.Module()
        state.model.itc_encoder = torch.nn.Linear(1, 1, bias=False)
        state.model.itc_encoder.weight.data.fill_(1.)
        state.settings.update(ITC_WEIGHT=1, ITM_WEIGHT=1, ITG_WEIGHT=1)
        state.ema = EMA(state.model.itc_encoder, .9)
        state.optimizer = torch.optim.SGD(state.model.parameters(), lr=.1)
        state.scheduler = torch.optim.lr_scheduler.LambdaLR(state.optimizer, lambda step: 1.)
        state.train_loader = [self.batch(2), self.batch(1)]
        state.train_loader[1]['attention_mask'][:, 2:] = False
        coefficients = iter([(1., 2., 3.), (5., 0., 7.)])
        def loss(*args):
            losses = {name: state.model.itc_encoder.weight.sum() * value for name, value in
                      zip(('loss_itc', 'loss_itm', 'loss_itg'), next(coefficients))}
            return losses, {'image_features': torch.zeros(1, 2, 4), 'text_features': torch.zeros(1, 4),
                            'image_ids': torch.zeros(1, dtype=torch.long)}
        with patch('src.trainingph1.engine.hepler_compute_loss', side_effect=loss):
            metrics = train_one_epoch(state, 0)
        expected = 7/3 + 2 + 19/5
        torch.testing.assert_close(state.model.itc_encoder.weight, torch.tensor([[1 - .1 * expected]]))
        self.assertAlmostEqual(metrics['loss'], expected)

    def test_ema_only_contains_independent_frozen_itc_encoder(self):
        online, teacher = self.state.model.itc_encoder, self.state.ema.model
        self.assertEqual(set(teacher._modules), {'qformer', 'query_proj', 'text_proj'})
        for name, parameter in teacher.named_parameters():
            original = dict(online.named_parameters())[name]
            self.assertNotEqual(parameter.data_ptr(), original.data_ptr())
            self.assertFalse(parameter.requires_grad)
            torch.testing.assert_close(parameter, original)
        before = teacher.query_proj.weight.clone()
        with torch.no_grad():
            online.query_proj.weight.add_(1.)
        self.state.ema.update(online)
        torch.testing.assert_close(teacher.query_proj.weight, before + .1)
        self.state.ema.train()
        self.assertFalse(teacher.training)

    def test_nonfinite_gradients_without_scaler_skip_all_updates(self):
        for value in (float('nan'), float('inf')):
            with self.subTest(value=value):
                state = self.state
                state.train_loader = state.train_loader[:1]
                before = copy.deepcopy(state.model.state_dict())
                ema_before = copy.deepcopy(state.ema.state_dict())
                hook = state.model.itc_encoder.query_proj.weight.register_hook(lambda grad: grad * value)
                try:
                    train_one_epoch(state, state.progress['epoch'])
                finally:
                    hook.remove()
                self.assertEqual(state.progress['global_step'], 0)
                self.assertEqual(state.scheduler.last_epoch, 0)
                self.assertEqual(state.queue.count.item(), 0)
                self.assertTrue(all(p.grad is None for p in state.model.parameters()))
                for name, tensor in before.items():
                    torch.testing.assert_close(tensor, state.model.state_dict()[name])
                for name, tensor in ema_before.items():
                    torch.testing.assert_close(tensor, state.ema.state_dict()[name])

    def test_cpu_bfloat16_autocast_training(self):
        self.state.settings['AMP_ENABLED'] = True
        metrics = train_one_epoch(self.state, 0)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))
        self.assertEqual(self.state.progress['global_step'], 2)

    def test_all_padding_skips_optimizer_scheduler_ema_and_queue(self):
        state = self.state
        state.train_loader = [self.batch(2)]
        state.train_loader[0]['valid_mask'] = torch.zeros(2, dtype=torch.bool)
        before = {name: copy.deepcopy(getattr(state, name).state_dict())
                  for name in ('model', 'ema', 'queue', 'optimizer', 'scheduler')}
        with patch.object(state.optimizer, 'step', wraps=state.optimizer.step) as step:
            metrics = train_one_epoch(state, 0)
        step.assert_not_called()
        self.assertEqual(state.progress['global_step'], 0)
        self.assertTrue(all(value == 0 for value in metrics.values()))
        from test_checkpoint_fsdp import assert_equal
        for name, expected in before.items():
            assert_equal(getattr(state, name).state_dict(), expected)

    def test_nonfinite_loss_with_finite_gradients_is_skipped_and_metrics_recover(self):
        state = self.state
        state.settings['ACCUMULATION_STEPS'] = 1
        state.train_loader = state.train_loader[:2]
        calls = 0
        def poisoned_loss(*args):
            nonlocal calls
            losses, keys = hepler_compute_loss(*args)
            if calls == 0:
                losses['loss_itc'] = losses['loss_itc'] + float('nan')
            calls += 1
            return losses, keys
        with patch.object(single, 'hepler_compute_loss', side_effect=poisoned_loss):
            metrics = train_one_epoch(state, 0)
        self.assertEqual(state.progress['global_step'], 1)
        self.assertEqual(state.queue.count.item(), 2)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_nonfinite_features_reach_skip_without_multinomial_failure(self):
        state = self.state
        state.train_loader = state.train_loader[:1]
        def poison(module, args, result):
            if args[-1] == 'itc':
                result['image_features'] = result['image_features'] * float('nan')
            return result
        handle = state.model.register_forward_hook(poison)
        try:
            metrics = train_one_epoch(state, 0)
        finally:
            handle.remove()
        self.assertEqual(state.progress['global_step'], 0)
        self.assertEqual(state.queue.count.item(), 0)
        self.assertEqual(metrics['loss'], 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'requires one CUDA GPU')
    def test_single_gpu_training(self):
        for dtype in ('float16', 'bfloat16'):
            with self.subTest(dtype=dtype):
                state = copy.deepcopy(self.state)
                state.settings.update(DEVICE='cuda:0', AMP_ENABLED=True, AMP_DTYPE=dtype)
                state.model.cuda()
                state.ema.cuda()
                state.queue.cuda()
                state.optimizer = torch.optim.SGD((p for p in state.model.parameters() if p.requires_grad), lr=.01)
                state.scheduler = torch.optim.lr_scheduler.LambdaLR(state.optimizer, lambda _: 1.)
                state.scaler = torch.amp.GradScaler('cuda', init_scale=128) if dtype == 'float16' else None
                metrics = train_one_epoch(state, 0)
                self.assertEqual(state.progress['global_step'], 2)
                self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_amp_overflow_does_not_advance_scheduler_ema_or_queue(self):
        state = self.state
        state.scaler = torch.amp.GradScaler('cpu')
        state.train_loader = state.train_loader[:1]
        before = copy.deepcopy(state.ema.state_dict())
        hook = state.model.itc_encoder.query_proj.weight.register_hook(lambda grad: grad * float('inf'))
        try:
            train_one_epoch(state, 0)
        finally:
            hook.remove()
        self.assertEqual(state.progress['global_step'], 0)
        self.assertEqual(state.scheduler.last_epoch, 0)
        self.assertEqual(state.queue.count.item(), 0)
        for name, value in before.items():
            torch.testing.assert_close(value, state.ema.state_dict()[name])


class EngineIntegrationTests(unittest.TestCase):
    def setUp(self):
        from test_checkpoint_fsdp import fixture, tokenizer, tiny_model
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = {k: v for k, v in vars(config).items() if k.isupper()} | fixture(self.temp.name)
        self.settings.update(DATASET_NAMESPACE='custom/test', SEED=19, LENGTH_AWARE=False,
                             LENGTH_MEGA_BATCH_MULT=2, LENGTH_JITTER=0, TENSORBOARD_ENABLED=False)
        self.tokenizer = tokenizer()
        for target, replacement in (
            ('src.trainingph1.engine.ModelStage1', tiny_model),
            ('src.trainingph1.engine.AutoTokenizer.from_pretrained', lambda *a, **k: tokenizer()),
        ):
            patcher = patch(target, side_effect=replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_real_loaders_pass_settings_and_fingerprint_actual_cache(self):
        loaders = single.get_dataloader(self.tokenizer, self.settings)
        references = fsdp.get_dataloader(self.tokenizer, self.settings)
        for loader, reference in zip(loaders, references):
            sampler = loader.batch_sampler
            self.assertEqual((sampler.seed, sampler.mega_batch_mult, sampler.length_jitter), (19, 2, 0))
            self.assertFalse(sampler.length_aware)
            self.assertEqual(list(sampler), list(reference.batch_sampler))
            self.assertEqual(sampler.num_replicas, 1)
            self.assertFalse(sampler.pad_batches)
            self.assertEqual(len(loader), 5)
            self.assertTrue(all(batch['valid_mask'].all() for batch in loader))
        self.assertEqual(loaders[0].dataset.cache_path.name, 'custom_test_train.sqlite')
        self.assertEqual(single.get_data_fingerprints(self.settings, loaders),
                         fsdp.get_data_fingerprints(self.settings, loaders))
        loaders[0].batch_sampler.set_epoch(1)
        self.assertNotEqual(list(loaders[0].batch_sampler), list(references[0].batch_sampler))

    def test_real_validation_preserves_loader_generator_and_all_rng_on_failure(self):
        from test_checkpoint_fsdp import assert_equal
        state = single.prepare_training(self.settings | dict(RUN_NAME='validation'))
        expected_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        generator = state.val_loader.generator.get_state()
        def fail(*args):
            random.random()
            np.random.rand()
            torch.rand(1)
            raise RuntimeError('validation failed')
        with patch.object(single, 'hepler_compute_loss', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'validation failed'):
                single.validate(state)
        self.assertTrue(state.model.training)
        self.assertEqual(random.getstate(), expected_rng[0])
        np.testing.assert_equal(np.random.get_state(), expected_rng[1])
        assert_equal(torch.get_rng_state(), expected_rng[2])
        assert_equal(state.val_loader.generator.get_state(), generator)
        self.assertEqual(single.validate(state), single.validate(state))

    def test_real_single_device_training_and_resume_in_second_epoch(self):
        from test_checkpoint_fsdp import assert_equal
        for workers in (0, 2):
            with self.subTest(workers=workers):
                options = self.settings | dict(NUM_WORKERS=workers, PERSISTENT_WORKERS=True)
                baseline_metrics = []
                baseline = single.run_training(options | dict(RUN_NAME=f'baseline-{workers}'),
                                               on_step=lambda state, metrics: baseline_metrics.append(metrics))
                expected_rng = (random.random(), np.random.rand(), torch.rand(3))
                captured = {}
                def interrupt(state, metrics):
                    if state.progress['global_step'] == 4:
                        captured['path'] = str(state.run_dir / 'last.json')
                        raise RuntimeError('interrupted')
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    single.run_training(options | dict(RUN_NAME=f'interrupted-{workers}'), on_step=interrupt)
                resumed_metrics = []
                resumed = single.run_training(options | dict(RESUME_CHECKPOINT=captured['path']),
                                              on_step=lambda state, metrics: resumed_metrics.append(metrics))
                self.assertEqual(resumed_metrics, baseline_metrics[4:])
                self.assertEqual(resumed.progress, baseline.progress)
                self.assertEqual(resumed.progress['global_step'], 6)
                self.assertEqual(resumed.train_loader.batch_sampler.epoch, 1)
                for name in ('model', 'ema', 'queue', 'optimizer', 'scheduler'):
                    assert_equal(getattr(resumed, name).state_dict(), getattr(baseline, name).state_dict())
                self.assertEqual((random.random(), np.random.rand()), expected_rng[:2])
                assert_equal(torch.rand(3), expected_rng[2])

    def test_engines_match_loss_gradients_and_keys_without_sharding(self):
        from test_engine_fsdp import tiny_model, state_for, batch
        for ids, mask in (([10, 20, 30], [True, True, False]),
                          ([10, 10, 10], [True, False, True]),
                          ([10, 20, 30], [False, False, False])):
            with self.subTest(ids=ids, mask=mask):
                local = state_for(tiny_model())
                local.settings['DEVICE'] = 'cpu'
                reference = state_for(copy.deepcopy(local.model))
                local.model.eval()
                reference.model.eval()
                data = batch(ids, mask)
                data['attention_mask'][1, -1] = False
                torch.manual_seed(17)
                actual, local_keys = single.hepler_compute_loss(local, data, .4)
                torch.manual_seed(17)
                sums, global_keys = fsdp.compute_loss(reference, data, .4)
                expected = {name: sums[name + '_sum'] / sums['count_' + name[5:]].clamp_min(1)
                            for name in ('loss_itc', 'loss_itm', 'loss_itg')}
                for name in expected:
                    torch.testing.assert_close(actual[name], expected[name])
                actual['loss'].backward()
                sum(reference.settings[name[5:].upper() + '_WEIGHT'] * value
                    for name, value in expected.items()).backward()
                for a, b in zip(local.model.parameters(), reference.model.parameters()):
                    ga = torch.zeros_like(a) if a.grad is None else a.grad
                    gb = torch.zeros_like(b) if b.grad is None else b.grad
                    torch.testing.assert_close(ga, gb, rtol=2e-4, atol=2e-5)
                for name in local_keys:
                    torch.testing.assert_close(local_keys[name], global_keys[name])

    def test_runtime_and_settings_fail_before_loading_models(self):
        with patch.dict(os.environ, WORLD_SIZE='2'):
            with self.assertRaisesRegex(ValueError, 'engine_fsdp'):
                single.prepare_training(self.settings)
        invalid = [dict(ACCUMULATION_STEPS=0), dict(EPOCHS=True), dict(MAX_GRAD_NORM=float('nan')),
                   dict(TRAINING_MODE='invalid'), dict(TRAINING_MODE='finetune'), dict(AMP_DTYPE='float32')]
        with patch.object(single, 'get_model', side_effect=AssertionError('model loaded')):
            for values in invalid:
                with self.subTest(values=values), self.assertRaises(ValueError):
                    single.prepare_training(self.settings | values)


if __name__ == '__main__':
    unittest.main()
