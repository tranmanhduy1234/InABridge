import torch
from torch import nn
from torch.distributed.tensor import DTensor


class MoCoQueue(nn.Module):
    def __init__(self, capacity: int, num_queries: int, dim: int):
        super().__init__()
        if any(type(value) is not int or value <= 0 for value in (capacity, num_queries, dim)):
            raise ValueError("capacity, num_queries and dim must be positive integers")
        self.capacity = capacity
        self.register_buffer("image", torch.zeros(capacity, num_queries, dim))
        self.register_buffer("text", torch.zeros(capacity, dim))
        self.register_buffer("image_ids", torch.zeros(capacity, dtype=torch.long))
        self.register_buffer("ptr", torch.zeros((), dtype=torch.long))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, images, texts, image_ids, valid_mask=None):
        tensors = (images, texts, image_ids) + (() if valid_mask is None else (valid_mask,))
        if any(isinstance(t, DTensor) for t in (*tensors, *self.buffers())):
            raise ValueError("MoCo queue and incoming keys must be full, unsharded tensors")
        if images.ndim != 3:
            raise ValueError("images must have shape [batch, num_queries, dim]")
        b = images.size(0)
        if (images.shape[1:] != self.image.shape[1:] or texts.shape != (b, self.text.size(1))
                or image_ids.shape != (b,)):
            raise ValueError("Image/text/ID shapes must match the queue and batch size")
        if not images.is_floating_point() or not texts.is_floating_point() or image_ids.dtype != torch.long:
            raise ValueError("Features must be floating point and image_ids must be int64")
        if any(t.device != self.image.device for t in tensors):
            raise ValueError("Keys, IDs and mask must be on the queue device")
        if valid_mask is not None:
            if valid_mask.shape != (b,) or valid_mask.dtype != torch.bool:
                raise ValueError("valid_mask must be boolean with shape [batch]")
            images, texts, image_ids = images[valid_mask], texts[valid_mask], image_ids[valid_mask]
            b = images.size(0)
        if b == 0:
            return
        n = min(b, self.capacity)
        start = (self.ptr.item() + b - n) % self.capacity
        idx = (torch.arange(n, device=self.image.device) + start) % self.capacity
        self.image[idx] = images[-n:].detach().to(self.image.dtype)
        self.text[idx] = texts[-n:].detach().to(self.text.dtype)
        self.image_ids[idx] = image_ids[-n:]
        self.ptr.fill_((self.ptr.item() + b) % self.capacity)
        self.count.fill_(min(self.capacity, self.count.item() + b))

    def get(self):
        n = self.count.item()
        if n < self.capacity:
            return self.image[:n], self.text[:n], self.image_ids[:n]
        idx = (torch.arange(self.capacity, device=self.image.device) + self.ptr) % self.capacity
        return self.image[idx], self.text[idx], self.image_ids[idx]
