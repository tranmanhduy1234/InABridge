import torch
import torch.nn as nn
import torch.nn.functional as F

# Fallback cho RMSNorm nếu chạy ở phiên bản PyTorch < 2.4
if hasattr(nn, "RMSNorm"):
    RMSNorm = nn.RMSNorm
else:
    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
        def forward(self, x):
            variance = x.pow(2).mean(-1, keepdim=True)
            return x * torch.rsqrt(variance + self.eps) * self.weight

class QwenProjector(nn.Module):
    def __init__(
        self,
        qformer_dim: int = 768,
        hidden_dim: int = 1536,
        llm_dim: int = 3584,  
        zero_init_last_layer: bool = True
    ):
        super().__init__()
        self.norm = RMSNorm(qformer_dim)
        self.fc1 = nn.Linear(qformer_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, llm_dim)
        self.residual_proj = nn.Linear(qformer_dim, llm_dim)
        self.gate = nn.Linear(qformer_dim, llm_dim)
        if zero_init_last_layer:
            self._reset_parameters()

    def _reset_parameters(self):
        # Zero-initialize fc2 để ở bước đầu tiên, output mượt mà dựa vào residual_proj
        nn.init.zeros_(self.fc2.weight)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)
    def forward(self, x: torch.Tensor):
        residual = self.residual_proj(x)
        x_norm = self.norm(x)
        
        # SwiGLU Branch
        a, b = self.fc1(x_norm).chunk(2, dim=-1)
        x_swiglu = F.silu(a) * b
        x_proj = self.fc2(x_swiglu)
        # Gating Mechanism
        gate = torch.sigmoid(self.gate(x_norm))
        out = residual + gate * x_proj
        return out, gate