import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import BertConfig, DINOv3ViTConfig

from src import config
from src.losses.lossph1 import Stage1Criterion
from src.queue.ema import EMA
from src.queue.moco import MoCoQueue
from src.trainingph1.engine import hepler_compute_loss, train_one_epoch, run_training, validate
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
            scaler=None, progress=dict(epoch=0, next_batch=0, global_step=0, data_state=None),
            train_loader=[self.batch(2), self.batch(2), self.batch(1)],
        )

    def batch(self, size):
        ids = torch.tensor([[1, i + 2, 7, 0] for i in range(size)])
        return dict(images=torch.randn(size, 3, 16, 16), image_ids=torch.arange(size), input_ids=ids,
                    attention_mask=ids.ne(0))

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
        itm_call = next(c for c in forward.call_args_list if c.args[-1] == 'itm')
        self.assertEqual(itm_call.args[0].shape[0], 9)
        self.assertTrue(torch.all(itm_call.args[1][6:, 1] != batch['input_ids'][:, 1]))
        self.assertTrue(torch.all((itm_call.args[0][3:6] - features).flatten(1).abs().sum(1) > 0))
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
        with patch('src.trainingph1.engine.prepare_training', return_value=state):
            result = run_training()
        self.assertIs(result, state)
        self.assertEqual(state.progress, dict(epoch=2, next_batch=0, global_step=4, data_state=None))
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


if __name__ == '__main__':
    unittest.main()
