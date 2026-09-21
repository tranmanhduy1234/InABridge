import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import BertConfig, DINOv3ViTConfig, PreTrainedTokenizerFast

from src import config
from src.trainingph1.model import ModelStage1
from src.trainingph1.engine import get_optimizer, get_dataloader, get_scheduler, get_criterion
from src.utils import checkpoint as cp


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        checkpoint_dir = patch.object(config, 'CHECKPOINT_DIR', self.directory.name)
        checkpoint_dir.start()
        self.addCleanup(checkpoint_dir.stop)
        manager = Path(self.directory.name) / 'manager.json'
        manager.write_text(cp.MANAGER_PATH.read_text())
        self.manager_patch = patch.object(cp, 'MANAGER_PATH', manager)
        self.manager_patch.start()
        self.addCleanup(self.manager_patch.stop)
        self.model = ModelStage1(
            'tiny-bert', -1, 'tiny-vision', 2, 1, 8, 4, 1e-12,
            bert_config=BertConfig(vocab_size=5, hidden_size=8, num_hidden_layers=1,
                                   num_attention_heads=2, intermediate_size=16),
            vision_config=DINOv3ViTConfig(hidden_size=8, num_hidden_layers=1,
                                         num_attention_heads=2, intermediate_size=16,
                                         image_size=16, patch_size=8),
        ).eval()
        backend = Tokenizer(WordLevel({'[UNK]': 0, '[PAD]': 1, '[CLS]': 2, 'a': 3, 'b': 4}, unk_token='[UNK]'))
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]',
                                                pad_token='[PAD]', cls_token='[CLS]')
        self.settings = dict(vars(config), IMAGE_SIZE=16, MAX_LENGTH=8)
        self.path = cp.save_checkpoint(self.model, self.tokenizer, 12,
                                      {'pretrain_optimizer': True}, settings=self.settings)

    def assert_rejected_unchanged(self, path, tokenizer=None):
        before = copy.deepcopy(self.model.state_dict())
        with self.assertRaises(ValueError):
            cp.load_checkpoint(path, self.model, tokenizer or self.tokenizer)
        for key, value in before.items():
            self.assertTrue(torch.equal(value, self.model.state_dict()[key]))

    def test_finetune_restores_offline_despite_changed_global_config(self):
        with patch.object(config, 'NUM_QUERIES', 99), patch.object(config, 'IMAGE_SIZE', 1024), \
             patch('transformers.BertModel.from_pretrained', side_effect=AssertionError('network')), \
             patch('transformers.DINOv3ViTModel.from_pretrained', side_effect=AssertionError('network')), \
             patch('transformers.AutoTokenizer.from_pretrained', side_effect=AssertionError('network')):
            restored, tokenizer, settings = cp.load_pretrained(self.path)
            self.assertEqual(settings['NUM_QUERIES'], 2)
            self.assertEqual(settings['IMAGE_SIZE'], 16)
            self.assertEqual(settings['BERT_MODEL_ID'], 'tiny-bert')
            self.assertEqual(config.NUM_QUERIES, 99)
            self.assertEqual(tokenizer('a')['input_ids'], self.tokenizer('a')['input_ids'])
            for key, value in self.model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]))
            images, ids = torch.randn(1, 3, 16, 16), torch.tensor([[2, 3]])
            with torch.no_grad():
                original = self.model(self.model.encode_image(images), ids, torch.ones_like(ids), 'itc')
                loaded = restored(restored.encode_image(images), ids, torch.ones_like(ids), 'itc')
            for key in original:
                torch.testing.assert_close(original[key], loaded[key], rtol=0, atol=0)
            restored.train()
            optimizer = get_optimizer(restored)
            self.assertFalse(optimizer.state)
            before = restored.itm_logit.weight.detach().clone()
            features = restored.encode_image(images)
            restored(features, ids, torch.ones_like(ids), 'itm')['itm_logits'].sum().backward()
            optimizer.step()
            self.assertFalse(torch.equal(before, restored.itm_logit.weight))
            # Tokenizer batching must not change the compatibility fingerprint.
            tokenizer(['a', 'a b'], padding=True, truncation=True, max_length=8)
            new_path = cp.save_checkpoint(restored, tokenizer, 13)
            self.assertEqual(new_path.parent, self.path.parent)
            state = cp.load_checkpoint(new_path, restored, tokenizer)
            self.assertIsNone(state['training_state'])

    def test_rejects_incompatible_model_and_tokenizer(self):
        self.model.checkpoint_config = self.settings
        self.model.vision_encoder.return_layer = -2
        self.assert_rejected_unchanged(self.path)
        self.model.vision_encoder.return_layer = -1
        other = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
            {'[UNK]': 0, '[PAD]': 1, '[CLS]': 2, 'a': 4, 'b': 3}, unk_token='[UNK]')),
            unk_token='[UNK]', pad_token='[PAD]', cls_token='[CLS]')
        self.assert_rejected_unchanged(self.path, other)

    def test_rejects_bad_metadata_and_weights(self):
        self.model.checkpoint_config = self.settings
        bad = Path(self.directory.name) / 'bad.pt'
        torch.save(self.model.state_dict(), bad)
        self.assert_rejected_unchanged(bad)
        payload = torch.load(self.path, weights_only=True)
        key = next(iter(payload['model']))
        payload['model'][key] = torch.zeros(1)
        torch.save(payload, bad)
        self.assert_rejected_unchanged(bad)
        payload['fingerprint'] = 'invalid'
        torch.save(payload, bad)
        self.assert_rejected_unchanged(bad)

    def test_rejects_overwrite_and_runtime_changes(self):
        with self.assertRaises(FileExistsError):
            cp.save_checkpoint(self.model, self.tokenizer, 12, settings=self.settings)
        self.assertFalse(list(self.path.parent.glob('*.tmp')))
        with patch.object(cp, 'version', return_value='different'):
            with self.assertRaisesRegex(ValueError, 'dependency'):
                cp.load_pretrained(self.path)
        manager = json.loads(cp.MANAGER_PATH.read_text())
        manager['schema_version'] = 1
        cp.MANAGER_PATH.write_text(json.dumps(manager))
        with self.assertRaisesRegex(ValueError, 'schema'):
            cp.load_pretrained(self.path)

    def test_config_coverage_and_resume_settings(self):
        payload = torch.load(self.path, weights_only=True)
        self.assertEqual(set(payload['run_config']), {k for k in vars(config) if k.isupper()})
        self.model.checkpoint_config = self.settings
        for name, value in [('LR', 0.01), ('ACCUMULATION_STEPS', 8), ('QUEUE_SIZE', 32),
                            ('TEMPERATURE', 0.1), ('PSEUDO_WARMUP_EPOCHS', 3), ('CROP_SCALE', (0.5, 1.0))]:
            with self.subTest(name=name), patch.object(config, name, value):
                with self.assertRaisesRegex(ValueError, name):
                    cp.load_checkpoint(self.path, self.model, self.tokenizer)
                cp.load_pretrained(self.path)
        with patch.object(config, 'LOG_EVERY_STEPS', 999):
            cp.load_checkpoint(self.path, self.model, self.tokenizer)
        with patch.object(config, 'NEW_SETTING', 1, create=True):
            with self.assertRaisesRegex(ValueError, 'classification'):
                cp.load_pretrained(self.path)

    def test_rejects_invalid_input_settings(self):
        for name, value in [('MAX_LENGTH', 10000), ('IMAGE_SIZE', 15), ('TOKENIZER_USE_FAST', False)]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                cp.save_checkpoint(self.model, self.tokenizer, 20, settings=self.settings | {name: value})
        self.model.vision_encoder.return_layer = -100
        with self.assertRaisesRegex(ValueError, 'VISION_RETURN_LAYER'):
            cp.save_checkpoint(self.model, self.tokenizer, 20, settings=self.settings)

    def test_dataloader_uses_saved_preprocessing(self):
        root = Path(self.directory.name)
        Image.new('RGB', (24, 24)).save(root / 'image.png')
        manifest = root / 'data.json'
        manifest.write_text(json.dumps([{'image_name': 'image.png', 'text': 'a'}] * 3))
        settings = self.settings | {
            'IMAGE_DIR': str(root), 'JSON_PATH_TRAIN': str(manifest), 'JSON_PATH_VAL': str(manifest),
            'CACHE_DIR': str(root / 'cache'), 'BATCH_SIZE': 2, 'DROP_LAST': True,
            'MAX_LENGTH': 4, 'TOKENIZER_PADDING': 'max_length',
        }
        with patch('transformers.AutoTokenizer.from_pretrained', side_effect=AssertionError('network')):
            train, val = get_dataloader(self.tokenizer, settings)
        self.assertEqual(len(train), 1)
        self.assertEqual(len(val), 2)
        batch = next(iter(val))
        self.assertEqual(batch['images'].shape, (2, 3, 16, 16))
        self.assertEqual(batch['input_ids'].shape, (2, 4))
        if train.dataset.conn is not None:
            train.dataset.conn.close()
        val.dataset.conn.close()

    def test_scheduler_and_pseudo_weight_follow_config(self):
        with patch.multiple(config, LR=0.01, MIN_LR=0.002, EPOCHS=3, WARMUP_EPOCHS=1,
                            ACCUMULATION_STEPS=2, PSEUDO_WARMUP_EPOCHS=2):
            optimizer = get_optimizer(self.model)
            scheduler = get_scheduler(optimizer, batches_per_epoch=3)
            self.assertEqual(optimizer.param_groups[0]['lr'], 0)
            rates = []
            for _ in range(6):
                optimizer.step()
                scheduler.step()
                rates.append(optimizer.param_groups[0]['lr'])
            self.assertAlmostEqual(rates[1], 0.01)
            self.assertAlmostEqual(rates[-1], 0.002)
            self.assertEqual(get_criterion(0).pseudo_weight, 0)
            self.assertEqual(get_criterion(1).pseudo_weight, config.PSEUDO_WEIGHT / 2)
            self.assertEqual(get_criterion(3).pseudo_weight, config.PSEUDO_WEIGHT)


if __name__ == '__main__':
    unittest.main()
