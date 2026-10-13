"""Run-scoped logging without changing the application's root logger."""

import logging
import sys
from pathlib import Path


def setup_logging(run_dir=None, *, label='main', level='INFO', console_level=None):
    if level not in {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}:
        raise ValueError('LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR or CRITICAL')
    name = str(Path(run_dir).resolve()) if run_dir is not None else 'startup'
    logger = logging.getLogger(f'ina.{name}.{label}')
    close_logging(logger)
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter(f'%(asctime)s | %(levelname)s | {label} | %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level or level)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if run_dir is not None:
        folder = Path(run_dir) / 'logs'
        folder.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(folder / f'{label}.log', mode='a', encoding='utf-8')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def close_logging(logger):
    for handler in logger.handlers[:]:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)
