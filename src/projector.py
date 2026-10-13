import torch
from torch import nn
import torch.nn.functional as F


class QwenProjector(nn.Module):
    def __init__(self, qformer_dim, llm_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.RMSNorm(qformer_dim)

        self.skip = nn.Linear(qformer_dim, llm_dim, bias=False)
        self.gate_proj = nn.Linear(qformer_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(qformer_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, llm_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, qformer_tensor):
        x = self.norm(qformer_tensor)
        h = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.skip(x) + self.scale * self.down_proj(self.dropout(h))