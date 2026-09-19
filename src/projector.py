import torch
import torch.nn as nn
import torch.nn.functional as F

class QwenProjector(nn.Module):
    def __init__(self, qformer_dim, llm_dim, hidden_dim, dropout):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_dim)
        self.fc1 = nn.Linear(qformer_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, llm_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, qformer_tensor):
        x = self.norm(qformer_tensor)
        x = self.dropout(F.silu(self.fc1(x)))
        return self.fc2(x)