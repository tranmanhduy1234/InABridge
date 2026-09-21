import torch
import torch.nn as nn
from transformers import DINOv3ViTModel

class ImageEncoder(nn.Module):
    def __init__(self, return_layer, model_id, model_config=None):
        super().__init__()
        self.return_layer = return_layer
        self.model_id = model_id
        self.model = (DINOv3ViTModel.from_pretrained(model_id) if model_config is None
                      else DINOv3ViTModel(model_config))
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def compute_grid_shape(self, height: int, width: int) -> tuple[int, int]:
        patch_size = self.model.config.patch_size
        if height <= 0 or width <= 0 or height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                f"Input size {height}x{width} must be positive "
                f"and divisible by {patch_size}."
            )
        return height // patch_size, width // patch_size

    def count_parameters(self, trainable: bool = False) -> int:
        return sum(p.numel() for p in self.model.parameters()) if not trainable \
            else sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values, output_hidden_states=True)
        features = outputs.hidden_states[self.return_layer]
        num_prefix_tokens = 1 + self.model.config.num_register_tokens
        return features[:, num_prefix_tokens:, :]

if __name__ == "__main__":
    from src.utils.seed import seed_everything

    seed_everything()
    encoder = ImageEncoder(-1, "facebook/dinov3-vitl16-pretrain-lvd1689m").to("cuda")
    encoder.train()
    print("Compute grid shape: ", encoder.compute_grid_shape(512, 512))
    print("encoder training:", encoder.training)
    print("DINO training   :", encoder.model.training)
    print(
        "trainable params:",
        sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    )
    x = torch.randn(2, 3, 512, 512, device=encoder.device)
    y = encoder(x)
    print("Input :", x.shape)
    print("Output:", y.shape)
    print("requires_grad:", y.requires_grad)
