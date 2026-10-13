import copy
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from transformers import BertConfig, DINOv3ViTConfig
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from PIL import Image

from src import config
from src.losses.lossph1 import Stage1Criterion
from src.queue.ema import EMA
from src.queue.moco import MoCoQueue
from src.trainingph1.model import ModelStage1
from src.trainingph1.engine_fsdp import (
    compute_loss, _loss_counts, shard_models, gradient_norm, train_one_epoch, validate, run_training,
    update_ema, get_dataloader,
)


def tiny_model():
    return ModelStage1(
        'tiny', -1, 'tiny', 2, 1, 8, 4, 1e-12,
        bert_config=BertConfig(vocab_size=12, hidden_size=8, num_hidden_layers=2,
                               num_attention_heads=2, intermediate_size=16,
                               hidden_dropout_prob=0., attention_probs_dropout_prob=0.),
        vision_config=DINOv3ViTConfig(hidden_size=8, num_hidden_layers=1,
                                     num_attention_heads=2, intermediate_size=16,
                                     image_size=16, patch_size=8))


def batch(ids, valid, length=4):
    size = len(ids)
    return dict(images=torch.randn(size, 3, 16, 16), image_ids=torch.tensor(ids),
                input_ids=torch.randint(0, 12, (size, length)),
                attention_mask=torch.ones(size, length, dtype=torch.bool),
                valid_mask=torch.tensor(valid))


class Batches(list):
    def __init__(self, values):
        super().__init__(values)
        self.generator = torch.Generator()
        self.batch_sampler = self
        self.epoch = None

    def set_epoch(self, epoch):
        self.epoch = epoch


def state_for(model, distributed=False, device='cpu'):
    settings = {k: v for k, v in vars(config).items() if k.isupper()}
    settings.update(AMP_ENABLED=False, ACCUMULATION_STEPS=2, MAX_GRAD_NORM=None)
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=.01)
    return SimpleNamespace(
        model=model, ema=EMA(model.itc_encoder, .9), settings=settings,
        optimizer=optimizer, scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.),
        queue=MoCoQueue(9, 2, 4).to(device), criterion=Stage1Criterion(),
        runtime=SimpleNamespace(device=torch.device(device), distributed=distributed,
                                rank=dist.get_rank() if distributed else 0,
                                world_size=dist.get_world_size() if distributed else 1),
        progress=dict(epoch=0, next_batch=0, global_step=0))


def distributed_worker(rank, path, device_type):
    torch.set_num_threads(1)
    if device_type == 'cuda':
        torch.cuda.set_device(rank)
    device = torch.device(device_type, rank) if device_type == 'cuda' else torch.device('cpu')
    dist.init_process_group('nccl' if device_type == 'cuda' else 'gloo',
                            init_method='file://' + path, rank=rank, world_size=2)
    try:
        torch.manual_seed(4)
        model = tiny_model()
        state = state_for(model, True, device)
        reference = state_for(copy.deepcopy(model).to(device), True, device)
        shard_models(model, state.ema, init_device_mesh(device_type, (2,)))
        state.optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=.01)
        state.scheduler = torch.optim.lr_scheduler.LambdaLR(state.optimizer, lambda _: 1.)
        assert all(isinstance(p, DTensor) for p in model.parameters())
        assert model.lm_head.weight is model.embeddings.word_embeddings.weight
        # Rank 1 has fewer valid tokens and no ITM negative.
        local = batch([10, 20], [True, True], 4) if rank == 0 else batch([30, 30], [True, False], 3)
        counts = _loss_counts(local).to(device)
        dist.all_reduce(counts)
        torch.manual_seed(21)
        ref_loss, ref_keys = compute_loss(reference, local, .4)
        expected = sum(2 * ref_loss[f'loss_{name}_sum'] / counts[i].clamp_min(1)
                       for i, name in enumerate(('itc', 'itm', 'itg')))
        expected.backward()
        torch.manual_seed(21)
        loss, keys = compute_loss(state, local, .4)
        assert all(isinstance(p, DTensor) for p in model.parameters())
        assert all(isinstance(p, DTensor) for p in state.ema.model.parameters())
        actual = sum(2 * loss[f'loss_{name}_sum'] / counts[i].clamp_min(1)
                     for i, name in enumerate(('itc', 'itm', 'itg')))
        torch.testing.assert_close(actual, expected)
        actual.backward()
        for name, p in reference.model.named_parameters():
            if not p.requires_grad:
                continue
            dist.all_reduce(p.grad)
            p.grad.div_(2)
            sharded = dict(model.named_parameters())[name]
            torch.testing.assert_close(sharded.grad.full_tensor(), p.grad, rtol=2e-4, atol=2e-5)
        assert reference.model.itc_encoder.qformer.query_tokens.grad.abs().sum() > 0
        norm = gradient_norm(model, device)
        target_norm = sum(p.grad.double().square().sum() for p in reference.model.parameters()
                          if p.grad is not None).sqrt()
        torch.testing.assert_close(norm, target_norm, rtol=2e-4, atol=2e-5)
        state.optimizer.step()
        reference.optimizer.step()
        update_ema(state.ema, model.itc_encoder)
        reference.ema.update(reference.model.itc_encoder)
        for name, p in state.ema.model.named_parameters():
            torch.testing.assert_close(p.full_tensor(), dict(reference.ema.model.named_parameters())[name],
                                       rtol=2e-4, atol=2e-5)
        state.optimizer.zero_grad(set_to_none=True)
        padded = copy.deepcopy(local)
        if rank == 1:
            padded['valid_mask'].fill_(False)
        state.train_loader = Batches([local, padded, local])
        state.val_loader = Batches([padded])
        state.tb = SimpleNamespace()
        state.settings['TENSORBOARD_DIAGNOSTICS_EVERY_STEPS'] = 1
        train_one_epoch(state, 0)
        assert state.progress['global_step'] == 2
        assert state.queue.count.item() == 8
        reports = [None, None]
        dist.all_gather_object(reports, state.step_diagnostics['scalars'])
        assert reports[0] == reports[1]
        assert reports[0]['counts/itc_queries'] == 3
        assert reports[0]['gradients/qformer/norm'] > 0
        for buf in state.queue.buffers():
            other = buf.clone()
            dist.broadcast(other, 0)
            torch.testing.assert_close(buf, other)
        before = copy.deepcopy(state.queue.state_dict())
        validate(state)
        assert state.validation_diagnostics['scalars']['counts/itc_queries'] == 2
        for name, buf in state.queue.state_dict().items():
            torch.testing.assert_close(buf, before[name])
        # One rank reports a nonfinite gradient: all ranks must skip together.
        before_queue = copy.deepcopy(state.queue.state_dict())
        before_ema = {k: p.to_local().clone() for k, p in state.ema.model.named_parameters()}
        state.train_loader = Batches([local])
        original_step = state.progress['global_step']
        parameter = next(p for p in model.parameters() if p.requires_grad and p.to_local().numel())
        handle = parameter.register_post_accumulate_grad_hook(
            lambda p: p.grad.to_local().fill_(float('nan')) if rank == 1 else None)
        metrics = train_one_epoch(state, 1)
        handle.remove()
        assert metrics['skipped_steps'] == 1
        assert metrics['loss'] == 0
        assert state.progress['global_step'] == original_step
        for name, buf in state.queue.state_dict().items():
            torch.testing.assert_close(buf, before_queue[name])
        for name, p in state.ema.model.named_parameters():
            torch.testing.assert_close(p.to_local(), before_ema[name])
        state.settings['ACCUMULATION_STEPS'] = 1
        state.train_loader = Batches([local, local])
        calls = 0

        def nonfinite_on_one_rank(*args):
            nonlocal calls
            losses, keys = compute_loss(*args)
            if calls == 0 and rank == 1:
                losses['loss_itc_sum'] = losses['loss_itc_sum'] * float('nan')
            calls += 1
            return losses, keys

        with patch('src.trainingph1.engine_fsdp.compute_loss', side_effect=nonfinite_on_one_rank):
            metrics = train_one_epoch(state, 2)
        assert metrics['skipped_steps'] == 1
        assert state.progress['global_step'] == original_step + 1
        assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
        rank_metrics = [None, None]
        dist.all_gather_object(rank_metrics, metrics)
        assert rank_metrics[0] == rank_metrics[1]
        # Exercise FSDP's actual mixed-precision policy, including the frozen tied weight.
        mixed = state_for(tiny_model(), True, device)
        mixed.settings['AMP_ENABLED'] = True
        shard_models(mixed.model, mixed.ema, init_device_mesh(device_type, (2,)), amp_enabled=True)
        mixed.optimizer = torch.optim.SGD((p for p in mixed.model.parameters() if p.requires_grad), lr=.01)
        mixed.scheduler = torch.optim.lr_scheduler.LambdaLR(mixed.optimizer, lambda _: 1.)
        mixed.train_loader = Batches([local, padded])
        metrics = train_one_epoch(mixed, 0)
        assert mixed.progress['global_step'] == 1
        assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
        # A rank-local loader failure must reach every rank before the next split.
        def loader_failure(*args, **kwargs):
            if rank == 1:
                raise ValueError('rank-local loader failure')
            return object()
        with patch('src.trainingph1.engine_fsdp.build_dataset', return_value=object()), \
             patch('src.trainingph1.engine_fsdp.build_dataloader', side_effect=loader_failure):
            try:
                get_dataloader(None, state.settings, distributed=True)
            except RuntimeError as exc:
                assert 'rank-local loader failure' in str(exc)
            else:
                raise AssertionError('Loader failure must reach all ranks')
    finally:
        dist.destroy_process_group()


class EngineFSDPTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)

    def test_masks_counts_and_dummy_itm(self):
        state = state_for(tiny_model())
        local = batch([10, 10, 20], [True, True, False])
        counts = _loss_counts(local)
        self.assertEqual(counts.tolist(), [2, 0, 6])
        called = []
        handle = state.model.register_forward_pre_hook(lambda m, args: called.append(args[-1]))
        losses, keys = compute_loss(state, local, .4)
        handle.remove()
        self.assertEqual(called, ['itc', 'itm', 'itm', 'itm', 'itg'])
        self.assertEqual(keys['image_ids'].tolist(), [10, 10])
        self.assertEqual(losses['loss_itm_sum'].item(), 0)
        sum(losses[f'loss_{name}_sum'] for name in ('itc', 'itm', 'itg')).backward()
        self.assertIsNotNone(state.model.itm_logit.weight.grad)
        self.assertEqual(state.model.itm_logit.weight.grad.count_nonzero().item(), 0)

    def test_accumulation_queue_and_validation(self):
        state = state_for(tiny_model())
        state.train_loader = Batches([batch([1, 2], [True, True]), batch([3, 4], [True, False]),
                                     batch([5, 6], [True, True])])
        state.val_loader = Batches([batch([7, 8], [True, False])])
        train_one_epoch(state, 3)
        self.assertEqual(state.progress, dict(epoch=4, next_batch=0, global_step=2))
        self.assertEqual(state.train_loader.epoch, 3)
        self.assertEqual(state.queue.get()[2].tolist(), [1, 2, 3, 5, 6])
        before = copy.deepcopy(state.queue.state_dict())
        rng = torch.get_rng_state()
        result = validate(state)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in result.values()))
        self.assertTrue(state.model.training)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        for name, value in state.queue.state_dict().items():
            torch.testing.assert_close(value, before[name])

    def test_padding_matches_removing_samples_from_loss_and_queue(self):
        state = state_for(tiny_model())
        reference = state_for(copy.deepcopy(state.model))
        padded = [batch([1, 2, 1], [True, True, False]),
                  batch([3, 99, 4], [True, False, True], length=5)]
        real = [{name: value[local['valid_mask']] for name, value in local.items()}
                for local in padded]
        for local, filtered in zip(padded, real):
            state.model.zero_grad(set_to_none=True)
            reference.model.zero_grad(set_to_none=True)
            actual, keys = compute_loss(state, local, .4)
            expected, expected_keys = compute_loss(reference, filtered, .4)
            torch.testing.assert_close(_loss_counts(local), _loss_counts(filtered))
            for name in actual:
                torch.testing.assert_close(actual[name], expected[name])
            for name in keys:
                torch.testing.assert_close(keys[name], expected_keys[name])
            for losses in (actual, expected):
                sum(losses[f'loss_{name}_sum'] for name in ('itc', 'itm', 'itg')).backward()
            for parameter, target in zip(state.model.parameters(), reference.model.parameters()):
                if parameter.requires_grad:
                    torch.testing.assert_close(parameter.grad, target.grad, rtol=2e-4, atol=2e-5)

        state.train_loader, reference.train_loader = Batches(padded), Batches(real)
        actual = train_one_epoch(state, 0)
        expected = train_one_epoch(reference, 0)
        for name in actual:
            self.assertAlmostEqual(actual[name], expected[name], places=5)
        for component in ('model', 'ema', 'queue'):
            expected_state = getattr(reference, component).state_dict()
            for name, value in getattr(state, component).state_dict().items():
                torch.testing.assert_close(value, expected_state[name], rtol=2e-4, atol=2e-5)
        self.assertEqual(state.queue.get()[2].tolist(), [1, 2, 3, 4])
        self.assertEqual(state.queue.count.item(), 4)
        self.assertEqual(state.queue.ptr.item(), 4)
        state.val_loader, reference.val_loader = Batches(padded), Batches(real)
        actual, expected = validate(state), validate(reference)
        for name in actual:
            self.assertAlmostEqual(actual[name], expected[name], places=5)

    def test_all_padding_skips_update(self):
        state = state_for(tiny_model())
        state.train_loader = Batches([batch([1, 2], [False, False])])
        before = copy.deepcopy(state.model.state_dict())
        result = train_one_epoch(state, 0)
        self.assertEqual(state.progress['global_step'], 0)
        self.assertEqual(state.queue.count.item(), 0)
        self.assertEqual(result['loss'], 0)
        self.assertEqual(result['skipped_steps'], 1)
        for name, value in state.model.state_dict().items():
            torch.testing.assert_close(value, before[name])

    def test_skipped_accumulation_does_not_pollute_metrics(self):
        for failure in ('nan', 'inf', 'gradient'):
            with self.subTest(failure=failure):
                state = state_for(tiny_model())
                state.train_loader = Batches([
                    batch([1, 2], [True, True], 5), batch([3, 4], [True, True], 4),
                    batch([5, 6], [True, False], 3), batch([7, 8], [True, True], 6),
                ])
                accepted, events = [], []
                calls = 0

                def inject_failure(*args):
                    nonlocal calls
                    losses, keys = compute_loss(*args)
                    if calls == 1:
                        if failure == 'gradient':
                            losses['loss_itc_sum'].register_hook(lambda grad: grad * float('nan'))
                        else:
                            losses['loss_itc_sum'] = losses['loss_itc_sum'] * float(failure)
                    if calls >= 2:
                        accepted.append({k: v.detach().clone() for k, v in losses.items()})
                    calls += 1
                    return losses, keys

                with patch('src.trainingph1.engine_fsdp.compute_loss', side_effect=inject_failure):
                    result = train_one_epoch(state, 0, on_step=lambda _, metrics: events.append(metrics))
                self.assertEqual(state.progress['global_step'], 1)
                self.assertEqual(state.scheduler.last_epoch, 1)
                self.assertEqual(state.queue.get()[2].tolist(), [5, 7, 8])
                self.assertEqual(result['skipped_steps'], 1)
                self.assertEqual(events, [result])
                for name in ('itc', 'itm', 'itg'):
                    expected = (sum(x[f'loss_{name}_sum'].double() for x in accepted)
                                / sum(x[f'count_{name}'] for x in accepted).clamp_min(1))
                    self.assertAlmostEqual(result[f'loss_{name}'], expected.item())
                self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in result.values()))

    def test_all_nonfinite_steps_report_skips_without_metrics_nan(self):
        state = state_for(tiny_model())
        state.train_loader = Batches([batch([1, 2], [True, True])])
        before = copy.deepcopy(state.model.state_dict())
        before_ema = copy.deepcopy(state.ema.state_dict())

        def nonfinite_loss(*args):
            losses, keys = compute_loss(*args)
            losses['loss_itc_sum'] = losses['loss_itc_sum'] * float('nan')
            return losses, keys

        with patch('src.trainingph1.engine_fsdp.compute_loss', side_effect=nonfinite_loss):
            result = train_one_epoch(state, 0)
        self.assertEqual(result, dict(loss_itc=0., loss_itm=0., loss_itg=0., loss=0., skipped_steps=1))
        self.assertEqual(state.progress['global_step'], 0)
        self.assertEqual(state.scheduler.last_epoch, 0)
        self.assertEqual(state.queue.count.item(), 0)
        self.assertFalse(state.optimizer.state)
        for name, value in state.model.state_dict().items():
            torch.testing.assert_close(value, before[name])
        for name, value in state.ema.state_dict().items():
            torch.testing.assert_close(value, before_ema[name])

    def test_accumulation_matches_token_weighted_reference_with_clipping(self):
        state = state_for(tiny_model())
        reference = state_for(copy.deepcopy(state.model))
        state.settings['MAX_GRAD_NORM'] = .2
        state.train_loader = Batches([batch([1, 2], [True, True], 5),
                                     batch([3, 4], [True, False], 3)])
        counts = sum(_loss_counts(b) for b in state.train_loader)
        from src.trainingph1.engine import get_pseudo_weight
        rng = torch.get_rng_state()
        for i, local in enumerate(state.train_loader):
            alpha = get_pseudo_weight(i / 2, state.settings)
            losses, _ = compute_loss(reference, local, alpha)
            sum(losses[f'loss_{name}_sum'] / counts[j].clamp_min(1)
                for j, name in enumerate(('itc', 'itm', 'itg'))).backward()
        torch.nn.utils.clip_grad_norm_(reference.model.parameters(), .2)
        reference.optimizer.step()
        torch.set_rng_state(rng)
        train_one_epoch(state, 0)
        for name, p in state.model.named_parameters():
            torch.testing.assert_close(p, dict(reference.model.named_parameters())[name])

    def test_bfloat16_forward_and_backward(self):
        state = state_for(tiny_model())
        state.settings['AMP_ENABLED'] = True
        state.train_loader = Batches([batch([1, 2], [True, True])])
        metrics = train_one_epoch(state, 0)
        self.assertEqual(state.progress['global_step'], 1)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))

    def test_diagnostics_do_not_change_updates_or_rng(self):
        state = state_for(tiny_model())
        reference = state_for(copy.deepcopy(state.model))
        state.tb = SimpleNamespace()
        state.settings['TENSORBOARD_DIAGNOSTICS_EVERY_STEPS'] = 1
        state.train_loader = Batches([batch([1, 2], [True, True]), batch([3, 4], [True, False])])
        reference.train_loader = copy.deepcopy(state.train_loader)
        rng = torch.get_rng_state()
        train_one_epoch(reference, 0)
        expected_rng = torch.get_rng_state()
        torch.set_rng_state(rng)
        train_one_epoch(state, 0)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        for name, parameter in state.model.named_parameters():
            torch.testing.assert_close(parameter, dict(reference.model.named_parameters())[name], rtol=0, atol=0)
        self.assertEqual(state.step_diagnostics['scalars']['counts/itc_queries'], 3)

    def test_run_training_with_multilevel_loader(self):
        tokenizer = Tokenizer(models.WordLevel(
            {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, 'caption': 4}, unk_token='[UNK]'))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.post_processor = processors.TemplateProcessing(
            single='[CLS] $A [SEP]', special_tokens=[('[CLS]', 2), ('[SEP]', 3)])
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token='[PAD]', unk_token='[UNK]')
        with tempfile.TemporaryDirectory() as folder:
            records = []
            for i in range(5):
                name = f'{i}.png'
                Image.new('RGB', (16, 16), (i * 20, 0, 0)).save(folder + '/' + name)
                records.append(dict(image_name=name, text='caption ' * (i + 1)))
            manifest = folder + '/manifest.json'
            with open(manifest, 'w') as stream:
                json.dump(records, stream)
            settings = dict(DEVICE='cpu', AMP_ENABLED=False, EPOCHS=1, WARMUP_EPOCHS=0,
                            ACCUMULATION_STEPS=2, JSON_PATH_TRAIN=manifest, JSON_PATH_VAL=manifest,
                            IMAGE_DIR=folder, CACHE_DIR=folder + '/cache', IMAGE_SIZE=16,
                            BATCH_SIZE=2, DROP_LAST=False, NUM_WORKERS=0, MAX_LENGTH=8,
                            NUM_QUERIES=2, ITC_DIM=4, QUEUE_SIZE=9, INIT_CHECKPOINT=None,
                            RESUME_CHECKPOINT=None, VAL_EVERY_STEPS=1, LOG_EVERY_STEPS=0)
            settings.update(CHECKPOINT_DIR=folder + '/checkpoints')
            with patch('src.trainingph1.engine_fsdp.AutoTokenizer.from_pretrained', return_value=tokenizer), \
                 patch('src.trainingph1.engine_fsdp.ModelStage1', side_effect=lambda *args, **kwargs: tiny_model()):
                state = run_training(settings)
            self.assertEqual(state.progress['global_step'], 2)
            self.assertEqual(state.queue.count.item(), 5)
            self.assertEqual(state.train_loader.batch_sampler.epoch, 0)

    @unittest.skipUnless(os.environ.get('RUN_DISTRIBUTED_TESTS') == '1', 'requires local process sockets')
    def test_two_rank_fsdp(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(distributed_worker, args=(folder + '/store', 'cpu'), nprocs=2, join=True)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, 'requires two CUDA GPUs')
    def test_two_gpu_fsdp(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(distributed_worker, args=(folder + '/store', 'cuda'), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
