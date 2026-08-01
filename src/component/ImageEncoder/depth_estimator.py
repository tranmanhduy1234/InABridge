import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Import class ImageEncoder từ module của bạn (ví dụ đặt tên file cũ là dinov3_encoder.py)
from src.component.ImageEncoder.imageEncoder import ImageEncoder, TrainMode

class DepthDecoderHead(nn.Module):
    """
    Head giải mã nhẹ (Lightweight Conv-Decoder) biến đổi patch features
    thành Depth Map ở độ phân giải gốc.
    """
    def __init__(self, in_channels: int, patch_size: int = 14):
        super().__init__()
        self.patch_size = patch_size
        
        # Mạng Conv Upsampling nâng số kênh và khôi phục độ phân giải
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1), # Output 1 channel (Depth map)
            nn.Sigmoid() # Normalize depth trong khoảng [0, 1]
        )

    def forward(self, x: torch.Tensor, original_size: tuple[int, int]) -> torch.Tensor:
        """
        x: Tensor 3D (B, L, D) - Patch tokens từ backbone (đã strip special tokens)
        original_size: (H, W) của ảnh đầu vào
        """
        B, L, D = x.shape
        H_orig, W_orig = original_size
        
        # Kích thước lưới patch (grid size)
        grid_h = H_orig // self.patch_size
        grid_w = W_orig // self.patch_size
        
        # Reshape từ 3D (B, L, D) sang 4D (B, D, H_grid, W_grid)
        x = x.transpose(1, 2).reshape(B, D, grid_h, grid_w)
        
        # Upsample không gian đặc trưng về độ phân giải ảnh ban đầu
        x = F.interpolate(x, size=(H_orig, W_orig), mode="bilinear", align_corners=False)
        
        # Đưa qua Conv Decoder
        depth = self.decoder(x)
        return depth.squeeze(1)  # Shape: (B, H, W)


class DINOv3DepthEstimator(nn.Module):
    """
    Mô hình hoàn chỉnh kết hợp DINOv3 Backbone + Depth Decoder Head.
    """
    def __init__(
        self,
        model_id: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",  # Có thể thay bằng Weights/Model ID DINOv3
        return_layer: int = -1,
    ):
        super().__init__()
        # Khởi tạo backbone encoder từ module của bạn
        self.encoder = ImageEncoder(model_id=model_id, return_layer=return_layer)
        
        # Đóng băng hoặc huấn luyện một phần backbone tùy nhu cầu
        self.encoder.set_train_mode(TrainMode.FROZEN)
        
        # Khởi tạo Depth Head với embed_dim nhận từ encoder
        self.depth_head = DepthDecoderHead(
            in_channels=self.encoder.embed_dim,
            patch_size=self.encoder.patch_size
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Input: pixel_values shape (B, C, H, W)
        Output: depth_map shape (B, H, W)
        """
        _, _, h, w = pixel_values.shape
        
        # 1. Trích xuất patch features từ DINOv3 Backbone
        feats = self.encoder(pixel_values, drop_special_tokens=True)
        
        # 2. Dự đoán depth map thông qua decoder head
        depth_map = self.depth_head(feats, original_size=(h, w))
        
        return depth_map