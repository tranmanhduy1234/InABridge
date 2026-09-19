from copy import deepcopy
import torch
import torch.nn as nn

class EMA(nn.Module):
    def __init__(self, model: nn.Module, momentum: float):
        super().__init__()
        if not 0 <= momentum <= 1:
            raise ValueError("Momentum should be between 0 and 1")

        self.momentum = momentum
        self.model = deepcopy(model).requires_grad_(False)
        self.train(False)

    @torch.no_grad()
    def update(self, online: nn.Module):
        online_params = dict(online.named_parameters())
        for name, p in self.model.named_parameters():
            p.lerp_(online_params[name].detach(), 1 - self.momentum)

        online_buffers = dict(online.named_buffers())
        for name, b in self.model.named_buffers():
            b.copy_(online_buffers[name])

    def train(self, mode: bool = True):
        super().train(False)
        return self