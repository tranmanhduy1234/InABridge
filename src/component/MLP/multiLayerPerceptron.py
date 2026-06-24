import torch
import torch.nn as nn
import torch.nn.functional as F

class QwenMultiPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(768, 3584)
        self.gate = nn.Linear(3584, 18944)
        self.up   = nn.Linear(3584, 18944)
        self.down = nn.Linear(18944, 3584)
        self.norm = nn.RMSNorm(3584)

    def forward(self, x):
        x = self.input_proj(x)
        residual = x
        x = F.silu(self.gate(x)) * self.up(x)
        x = self.down(x)
        return self.norm(x + residual)

if __name__ == "__main__":
    x = torch.randn(2, 64, 768)
    projector = QwenMultiPerceptron()
    y = projector(x)
    print(y.shape)