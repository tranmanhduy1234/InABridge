import random
import numpy as np
import torch
from src import config

def seed_everything(seed=None):
    seed = config.SEED if seed is None else seed
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32)")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # Seeds CPU and all CUDA devices.

def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
