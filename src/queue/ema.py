from copy import deepcopy
import torch
from torch import nn

class EMA(nn.Module):
    def __init__(self, model: nn.Module, momentum: float):
        super().__init__()
        if not 0 <= momentum <= 1:
            raise ValueError("Momentum should be between 0 and 1")
        self.momentum = momentum
        self.model = deepcopy(model).requires_grad_(False)
        self.train(False)

    def tensor_pairs(self, online):
        pairs = []
        for teacher, source in (
            (dict(self.model.named_parameters()), dict(online.named_parameters())),
            (dict(self.model.named_buffers()), dict(online.named_buffers())),
        ):
            if teacher.keys() != source.keys():
                raise ValueError("EMA parameter/buffer names do not match")
            pairs.append([(value, source[name]) for name, value in teacher.items()])
        return pairs

    def update(self, online: nn.Module):
        self.update_tensors(*self.tensor_pairs(online))

    @torch.no_grad()
    def update_tensors(self, parameters, buffers):
        parameters, buffers = list(parameters), list(buffers)
        # Validate every pair before mutating any teacher state.
        for teacher, source in (*parameters, *buffers):
            if (teacher.shape != source.shape or teacher.dtype != source.dtype
                    or teacher.device != source.device or teacher.is_meta or source.is_meta):
                raise ValueError("EMA tensor metadata mismatch or uninitialized tensor")
        for teacher, source in parameters:
            teacher.lerp_(source.detach(), 1 - self.momentum)
        for teacher, source in buffers:
            teacher.copy_(source.detach())

    def train(self, mode: bool = True):
        super().train(False)
        return self
