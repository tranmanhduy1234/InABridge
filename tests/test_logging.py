import io
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.utils.logging_setup import setup_logging, close_logging
from src.utils.tb_logger import TensorBoardLogger


class LoggingTests(unittest.TestCase):
    def test_rank_files_append_without_duplicate_handlers_or_root_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root_handlers = logging.getLogger().handlers[:]
            output = io.StringIO()
            with patch('sys.stdout', output):
                logger = setup_logging(folder, label='rank-1', level='DEBUG', console_level='WARNING')
                logger.info('worker-only detail')
                logger.warning('visible warning')
                logger = setup_logging(folder, label='rank-1', level='DEBUG', console_level='WARNING')
                logger.info('resumed')
                close_logging(logger)
            text = (Path(folder) / 'logs/rank-1.log').read_text()
            self.assertEqual(text.count('worker-only detail'), 1)
            self.assertEqual(text.count('visible warning'), 1)
            self.assertEqual(text.count('resumed'), 1)
            self.assertIn('rank-1', text)
            self.assertNotIn('worker-only detail', output.getvalue())
            self.assertIn('visible warning', output.getvalue())
            self.assertEqual(logging.getLogger().handlers, root_handlers)
            self.assertFalse(logger.handlers)

    def test_tensorboard_resume_purges_future_steps_and_filters_nan(self):
        with tempfile.TemporaryDirectory() as folder:
            logger = TensorBoardLogger(folder)
            logger.log_config({'TRAINING_MODE': 'pretrain'})
            for step in (1, 2, 3):
                logger.log_scalars('train', {'loss': step, 'bad': float('nan')}, step)
            logger.close()
            resumed = TensorBoardLogger(folder, resume_step=1)
            resumed.log_scalars('train', {'loss': 22}, 2)
            resumed.close()
            events = EventAccumulator(str(Path(folder) / 'tensorboard')).Reload()
            self.assertEqual([(x.step, x.value) for x in events.Scalars('train/loss')], [(1, 1.), (2, 22.)])
            self.assertNotIn('train/bad', events.Tags()['scalars'])
            self.assertIn('run/config/text_summary', events.Tags()['tensors'])

    def test_disabled_or_nonzero_rank_does_not_create_event_files(self):
        with tempfile.TemporaryDirectory() as folder:
            for kwargs in (dict(enabled=False),):
                logger = TensorBoardLogger(folder, **kwargs)
                logger.log_scalars('train', {'loss': 1}, 0)
                logger.flush()
                logger.close()
            self.assertFalse((Path(folder) / 'tensorboard').exists())


if __name__ == '__main__':
    unittest.main()
