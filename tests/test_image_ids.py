import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch
from PIL import Image

from src.data.dataloaderph1_multilevel import VLMDatasetStage1, VLMDataCollator, build_cache
from src.queue.moco import MoCoQueue
from test_checkpoint_fsdp import tokenizer as make_tokenizer


class ImageIdTests(unittest.TestCase):
    def test_cache_creation_and_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, cache = root / 'data.json', root / 'cache.sqlite'
            records = [{'image_name': f'{i}.jpg', 'text': str(i)} for i in range(3)]
            manifest.write_text(json.dumps(records))
            tokenizer = make_tokenizer()
            for _ in range(2):
                build_cache(manifest, cache, 2, tokenizer, 8)
                with sqlite3.connect(cache) as conn:
                    self.assertEqual(conn.execute('SELECT image_name, text FROM samples ORDER BY id').fetchall(),
                                     [(record['image_name'], record['text']) for record in records])
            self.assertFalse(cache.with_suffix('.sqlite.tmp').exists())

    def test_ids_survive_relocation_and_collator_preserves_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = []
            for name, namespace in [('original', 'dataset-a'), ('moved', 'dataset-a'), ('other', 'dataset-b')]:
                image_dir = root / name
                image_dir.mkdir()
                Image.new('RGB', (2, 2)).save(image_dir / 'image.png')
                manifest = image_dir / 'data.json'
                manifest.write_text(json.dumps([{'image_name': p, 'text': 'caption'}
                                                 for p in ('image.png', './image.png')]))
                dataset = VLMDatasetStage1(manifest, image_dir, True, image_dir / 'cache', 10,
                                           lambda image: torch.zeros(3, 2, 2), namespace, make_tokenizer(), 8)
                try:
                    samples.append([dataset[0], dataset[1]])
                finally:
                    dataset.conn.close()
            self.assertEqual(samples[0][0][2], samples[0][1][2])
            self.assertEqual(samples[0][0][2], samples[1][0][2])
            self.assertNotEqual(samples[0][0][2], samples[2][0][2])
            tokenizer = Mock(return_value={'input_ids': torch.ones(2, 3, dtype=torch.long),
                                           'attention_mask': torch.ones(2, 3)})
            batch = VLMDataCollator(tokenizer, 8, True, True)(samples[0])
            self.assertEqual(batch['image_ids'].dtype, torch.long)
            self.assertEqual(batch['image_ids'].tolist(), [s[2] for s in samples[0]])

    def test_queue_ids_follow_features_through_wrap_and_reload(self):
        queue = MoCoQueue(3, 2, 1)
        self.assertEqual(queue.get()[2].numel(), 0)
        for values in ([10, 20], [30, 40], [50, 60, 70, 80]):
            ids = torch.tensor(values)
            queue.enqueue(ids.float()[:, None, None].expand(-1, 2, 1), ids.float()[:, None], ids)
            images, texts, actual_ids = queue.get()
            torch.testing.assert_close(images[:, 0, 0], actual_ids.float())
            torch.testing.assert_close(texts[:, 0], actual_ids.float())
        self.assertEqual(queue.get()[2].tolist(), [60, 70, 80])
        restored = MoCoQueue(3, 2, 1)
        restored.load_state_dict(queue.state_dict())
        for a, b in zip(queue.get(), restored.get()):
            torch.testing.assert_close(a, b)


if __name__ == '__main__':
    unittest.main()
