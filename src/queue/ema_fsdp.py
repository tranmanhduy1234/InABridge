from copy import deepcopy

import torch
from torch import nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor, Partial

class EMA(nn.Module):
    def __init__(self, model: nn.Module, momentum: float):
        super().__init__()
        if not 0 <= momentum <= 1:
            raise ValueError("Momentum should be between 0 and 1")
        if any(isinstance(m, FSDPModule) for m in model.modules()) or any(
            isinstance(t, DTensor) for t in (*model.parameters(), *model.buffers())
        ):
            raise ValueError("Create EMA before sharding the online model")
        self.momentum = momentum
        self.model = deepcopy(model).requires_grad_(False)
        self.train(False)

    @staticmethod
    def _local_pair(name, teacher, online):
        if (teacher.shape != online.shape or teacher.dtype != online.dtype
                or teacher.device != online.device or teacher.is_meta or online.is_meta):
            raise ValueError(f"EMA tensor metadata mismatch or uninitialized tensor: {name}")
        if isinstance(teacher, DTensor) != isinstance(online, DTensor):
            raise ValueError(f"EMA sharding mismatch: {name}")
        if isinstance(teacher, DTensor):
            if (teacher.device_mesh != online.device_mesh or teacher.placements != online.placements
                    or any(isinstance(p, Partial) for p in teacher.placements)):
                raise ValueError(f"EMA mesh/placements mismatch or partial tensor: {name}")
            teacher, online = teacher.to_local(), online.to_local()
        if teacher.shape != online.shape:
            raise ValueError(f"EMA local shard shape mismatch: {name}")
        return teacher, online.detach()

    @torch.no_grad()
    def update(self, online: nn.Module):
        pairs = []
        for teacher, source in (
            (dict(self.model.named_parameters()), dict(online.named_parameters())),
            (dict(self.model.named_buffers()), dict(online.named_buffers())),
        ):
            if teacher.keys() != source.keys():
                raise ValueError("EMA parameter/buffer names do not match")
            pairs.append([self._local_pair(name, value, source[name])
                          for name, value in teacher.items()])
        # Validate every pair before mutating any teacher state.
        for teacher, source in pairs[0]:
            teacher.lerp_(source, 1 - self.momentum)
        for teacher, source in pairs[1]:
            teacher.copy_(source)

    def train(self, mode: bool = True):
        super().train(False)
        return self
