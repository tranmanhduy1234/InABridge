import torch
from torch import nn

class MoCoQueue(nn.Module):
    def __init__(self, capacity: int, num_queries: int, dim: int):
        super().__init__()
        self.capacity = capacity
        self.register_buffer("image", torch.zeros(capacity, num_queries, dim))
        self.register_buffer("text", torch.zeros(capacity, dim))
        self.register_buffer("ptr", torch.zeros((), dtype=torch.long))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, images, texts):
        b = images.size(0)
        n = min(b, self.capacity)

        start = (self.ptr.item() + b - n) % self.capacity
        idx = (torch.arange(n, device=images.device) + start) % self.capacity

        self.image[idx] = images[-n:].detach()
        self.text[idx] = texts[-n:].detach()
        self.ptr.fill_((self.ptr.item() + b) % self.capacity)
        self.count.fill_(min(self.capacity, self.count.item() + b))

    def get(self):
        n = self.count.item()
        if n < self.capacity:
            return self.image[:n], self.text[:n]
        idx = (
            torch.arange(self.capacity, device=self.image.device)
            + self.ptr
        ) % self.capacity
        return self.image[idx], self.text[idx]