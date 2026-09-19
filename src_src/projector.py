import torch
import torch.nn as nn
import torch.nn.functional as F
from src_src import config_model as cfg

class QwenProjector(nn.Module):
    def __init__(
        self,
        qformer_dim: int,
        llm_dim: int,
        hidden_dim: int | None = None,
        dropout: float = cfg.PROJECTOR_DROPOUT,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = cfg.PROJECTOR_HIDDEN_MULTIPLIER * qformer_dim
        self.norm = nn.RMSNorm(qformer_dim)
        self.fc1 = nn.Linear(qformer_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, llm_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.dropout(F.silu(self.fc1(x)))
        return self.fc2(x)