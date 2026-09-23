import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import BertConfig, DINOv3ViTConfig, PreTrainedTokenizerFast

from src import config
from src.queue.ema import EMA
from src.queue.moco import MoCoQueue
from src.trainingph1.model import ModelStage1
from src.trainingph1.engine import prepare_training, get_pseudo_weight
from src.utils.checkpoint import create_run, save_checkpoint, load_pretrained, load_checkpoint


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = ModelStage1(
            'tiny-bert', -1, 'tiny-vision', 2, 1, 8, 4, 1e-12,
            bert_config=BertConfig(vocab_size=5, hidden_size=8, num_hidden_layers=1,
                                   num_attention_heads=2, intermediate_size=16),
            vision_config=DINOv3ViTConfig(hidden_size=8, num_hidden_layers=1,
                                         num_attention_heads=2, intermediate_size=16,
                                         image_size=16, patch_size=8),
        )
        backend = Tokenizer(WordLevel({'[UNK]': 0, '[PAD]': 1, '[CLS]': 2, 'a': 3, 'b': 4}, unk_token='[UNK]'))
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]')
        self.run = create_run(self.model, self.tokenizer, run_dir=self.root / 'pretrain')
        self.objects = self.components(self.model)
        self.step(self.model, self.objects)
        self.path = self.save(self.run, self.model, self.objects, 1)

    def components(self, model):
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
        return dict(optimizer=optimizer, scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: 0.9**n),
                    scaler=torch.amp.GradScaler('cpu'), ema=EMA(model.itc_encoder, 0.9), queue=MoCoQueue(4, 2, 4))

    def step(self, model, objects):
        model.train()
        images, ids = torch.randn(2, 3, 16, 16), torch.tensor([[2, 3], [2, 4]])
        loss = model(model.encode_image(images), ids, torch.ones_like(ids), 'itm')['itm_logits'].square().mean()
        if objects['scaler'] is None:
            loss.backward()
            objects['optimizer'].step()
        else:
            objects['scaler'].scale(loss).backward()
            objects['scaler'].step(objects['optimizer'])
            objects['scaler'].update()
        objects['scheduler'].step()
        objects['optimizer'].zero_grad(set_to_none=True)
        objects['ema'].update(model.itc_encoder)
        objects['queue'].enqueue(torch.randn(2, 2, 4), torch.randn(2, 4), torch.tensor([11, 22]))

    def save(self, run, model, objects, step, **kwargs):
        return save_checkpoint(run, model, **objects, global_step=step, epoch=0, next_batch=step,
                               data_state={'sampler_order': [2, 0, 1]}, **kwargs)

    def assert_state_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(left, right)

    def assert_resume_matches_next_update(self, path, model, objects):
        expected_random = random.random(), np.random.rand()
        self.step(model, objects)
        restored, _, _ = load_pretrained(path)
        restored_objects = self.components(restored)
        progress = load_checkpoint(path, restored, **restored_objects)
        self.assertEqual(progress['data_state']['sampler_order'], [2, 0, 1])
        self.assertEqual((random.random(), np.random.rand()), expected_random)
        self.step(restored, restored_objects)
        self.assert_state_equal(model.state_dict(), restored.state_dict())
        for name in objects:
            self.assert_state_equal(objects[name].state_dict(), restored_objects[name].state_dict())
        self.assertEqual(objects['ema'].momentum, restored_objects['ema'].momentum)
        return progress

    def test_resume_pretrain_restores_complete_state_and_rng(self):
        progress = self.assert_resume_matches_next_update(self.path, self.model, self.objects)
        self.assertEqual((progress['global_step'], progress['epoch'], progress['next_batch']), (1, 0, 1))

    def test_finetune_and_resume_finetune(self):
        with patch.object(config, 'CROP_SCALE', (0.5, 1.0)), patch.object(config, 'NUM_QUERIES', 99), \
             patch('transformers.BertModel.from_pretrained', side_effect=AssertionError('network')), \
             patch('transformers.DINOv3ViTModel.from_pretrained', side_effect=AssertionError('network')):
            model, tokenizer, _ = load_pretrained(self.path)
            self.assertEqual(model.qformer_model.num_queries, 2)
            self.assertEqual(tokenizer('a')['input_ids'], self.tokenizer('a')['input_ids'])
            self.assert_state_equal(model.state_dict(), self.model.state_dict())
            model.itm_logit.bias.requires_grad_(False)
            objects = self.components(model)
            self.assertFalse(objects['optimizer'].state)
            run = create_run(model, tokenizer, run_dir=self.root / 'finetune', parent_checkpoint=self.path)
            self.step(model, objects)
            path = self.save(run, model, objects, 1)
            metadata = json.loads((run / 'run.json').read_text())
            self.assertEqual(metadata['config']['CROP_SCALE'], [0.5, 1.0])
            self.assertEqual(metadata['parent_checkpoint'], str(self.path))
            self.assertTrue(self.path.exists())
            self.assert_resume_matches_next_update(path, model, objects)

    def test_atomic_last_and_optional_best(self):
        before = self.path.read_bytes()
        with patch('src.utils.checkpoint.torch.save', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.save(self.run, self.model, self.objects, 2)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(list(self.run.glob('*.tmp')))
        self.save(self.run, self.model, self.objects, 2)
        best = self.save(self.run, self.model, self.objects, 2, filename='best.pt')
        self.assertTrue(best.exists())
        self.assertEqual(len(list(self.run.glob('*.pt'))), 2)
        with self.assertRaises(FileExistsError):
            create_run(self.model, self.tokenizer, run_dir=self.run)

    def test_prepare_finetune_and_resume_uses_saved_settings(self):
        settings = dict(INIT_CHECKPOINT=str(self.path), RESUME_CHECKPOINT=None,
                        CHECKPOINT_DIR=str(self.root), MODEL_VERSION='v1', RUN_NAME='finetune',
                        DEVICE='cpu', LR=0.002, MIN_LR=0.0001, EPOCHS=4, WARMUP_EPOCHS=1,
                        ACCUMULATION_STEPS=2, MOMENTUM=0.95, QUEUE_SIZE=4, AMP_ENABLED=False,
                        PSEUDO_WEIGHT=0.6, PSEUDO_WARMUP_EPOCHS=2, NUM_QUERIES=99, ITC_DIM=99)
        with patch.multiple(config, **settings), \
             patch('src.trainingph1.engine.get_dataloader', return_value=([0] * 4, [])), \
             patch('transformers.BertModel.from_pretrained', side_effect=AssertionError('network')), \
             patch('transformers.DINOv3ViTModel.from_pretrained', side_effect=AssertionError('network')):
            state = prepare_training()
            self.assertEqual(state.run_dir, self.root / 'v1' / 'finetune')
            self.assertEqual(state.queue.image.shape, (4, 2, 4))
            self.assertFalse(state.optimizer.state)
            self.assertEqual(state.progress['global_step'], 0)
            self.assert_state_equal(state.model.state_dict(), self.model.state_dict())
            objects = {name: getattr(state, name) for name in ('optimizer', 'scheduler', 'ema', 'queue', 'scaler')}
            self.step(state.model, objects)
            path = self.save(state.run_dir, state.model, objects, 1)
            with patch.multiple(config, INIT_CHECKPOINT=None, RESUME_CHECKPOINT=str(path),
                                LR=0.5, EPOCHS=100, WARMUP_EPOCHS=10, ACCUMULATION_STEPS=1,
                                MOMENTUM=0.1, QUEUE_SIZE=99, RUN_NAME='ignored', MODEL_VERSION='ignored'):
                restored = prepare_training()
            self.assertEqual(restored.run_dir, state.run_dir)
            self.assertEqual(restored.settings['LR'], 0.002)
            self.assertEqual(restored.progress['next_batch'], 1)
            epoch_progress = restored.progress['epoch'] + restored.progress['next_batch'] / len(restored.train_loader)
            self.assertAlmostEqual(get_pseudo_weight(epoch_progress, restored.settings), 0.075)
            restored_objects = {name: getattr(restored, name) for name in objects}
            rng = torch.get_rng_state()
            self.step(state.model, objects)
            torch.set_rng_state(rng)
            self.step(restored.model, restored_objects)
            self.assert_state_equal(state.model.state_dict(), restored.model.state_dict())
            for name in ('optimizer', 'scheduler', 'ema', 'queue'):
                self.assert_state_equal(objects[name].state_dict(), restored_objects[name].state_dict())

    def test_prepare_new_run_and_rejects_ambiguous_source(self):
        with patch.multiple(config, INIT_CHECKPOINT=None, RESUME_CHECKPOINT=None,
                            CHECKPOINT_DIR=str(self.root), MODEL_VERSION='v2', RUN_NAME='pretrain',
                            DEVICE='cpu', AMP_ENABLED=False, QUEUE_SIZE=4), \
             patch('src.trainingph1.engine.get_model', return_value=(self.model, EMA(self.model.itc_encoder, 0.99))), \
             patch('src.trainingph1.engine.AutoTokenizer.from_pretrained', return_value=self.tokenizer), \
             patch('src.trainingph1.engine.get_dataloader', return_value=([0] * 4, [])):
            state = prepare_training()
            self.assertEqual(state.run_dir, self.root / 'v2' / 'pretrain')
            self.assertIsNone(json.loads((state.run_dir / 'run.json').read_text())['parent_checkpoint'])
            self.assertEqual(state.progress['epoch'], 0)
            self.assertTrue(state.model.training)
            self.assertFalse(state.ema.training)
        with patch.multiple(config, INIT_CHECKPOINT='source.pt', RESUME_CHECKPOINT='last.pt'):
            with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
                prepare_training()

    def test_rejects_wrong_run_architecture_and_missing_components(self):
        other = create_run(self.model, self.tokenizer, run_dir=self.root / 'other')
        copied = other / 'last.pt'
        copied.write_bytes(self.path.read_bytes())
        with self.assertRaisesRegex(ValueError, 'run.json'):
            load_pretrained(copied)
        self.model.vision_encoder.return_layer = -2
        with self.assertRaisesRegex(ValueError, 'architecture'):
            load_checkpoint(self.path, self.model, **self.objects)
        self.model.vision_encoder.return_layer = -1
        with self.assertRaisesRegex(ValueError, 'components'):
            load_checkpoint(self.path, self.model, optimizer=self.objects['optimizer'], scheduler=self.objects['scheduler'])
        before = copy.deepcopy(self.model.state_dict())
        self.objects['optimizer'].param_groups[0]['params'].reverse()
        with self.assertRaisesRegex(ValueError, 'groups'):
            load_checkpoint(self.path, self.model, **self.objects)
        self.assert_state_equal(before, self.model.state_dict())


if __name__ == '__main__':
    unittest.main()
