import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from src.component.ImageEncoder.depth_estimator import DINOv3DepthEstimator

def preprocess_image(image_path: str, img_size: int = 518) -> tuple[torch.Tensor, Image.Image]:
    """Tải và tiền xử lý ảnh theo chuẩn ViT/DINOv3 (kích thước chia hết cho patch_size)."""
    raw_image = Image.open(image_path).convert("RGB")
    
    # DINOv3 thường yêu cầu ảnh vuông có kích thước chia hết cho patch_size (vd: 14 -> 518x518)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    input_tensor = transform(raw_image).unsqueeze(0)  # Thêm Batch dim: (1, 3, H, W)
    return input_tensor, raw_image.resize((img_size, img_size))


def visualize_depth(original_img: Image.Image, depth_tensor: torch.Tensor, save_path: str = "depth_result.png"):
    """Hiển thị và lưu kết quả so sánh giữa Ảnh gốc và Depth Map."""
    # Chuyển depth tensor về Numpy Array
    depth_map = depth_tensor.squeeze(0).cpu().numpy()
    
    # Chuẩn hóa min-max để hiển thị sắc nét hơn
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_normalized = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)

    # Plot kết quả
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original_img)
    axes[0].set_title("Ảnh đầu vào (Input RGB)")
    axes[0].axis("off")

    im = axes[1].imshow(depth_normalized, cmap="inferno")
    axes[1].set_title("Ước lượng chiều sâu DINOv3 (Depth Map)")
    axes[1].axis("off")
    
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="Depth Relative")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✓ Đã lưu kết quả Depth Map tại: {save_path}")
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang chạy trên thiết bị: {device}")

    # 1. Khởi tạo mô hình DINOv3 Depth Estimator
    # Lưu ý: Thay 'facebook/dinov2-base' bằng checkpoint DINOv3 tương ứng nếu có
    MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m" 
    
    print("\nKhởi tạo DINOv3 Depth Estimator...")
    model = DINOv3DepthEstimator(model_id=MODEL_ID, return_layer=-1).to(device)
    model.eval()

    # In tóm tắt cấu hình mô hình
    model.encoder.summary()

    # 2. Load và chuẩn bị ảnh
    image_path = "/home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/runtime/image.png"  # Thay bằng đường dẫn ảnh của bạn
    try:
        input_tensor, original_img = preprocess_image(image_path, img_size=512)
        input_tensor = input_tensor.to(device)
    except FileNotFoundError:
        print(f"⚠ Không tìm thấy file {image_path}. Tạo ảnh mẫu để test...")
        # Tạo ảnh ngẫu nhiên làm Demo nếu chưa có file ảnh real
        input_tensor = torch.randn(1, 3, 518, 518).to(device)
        original_img = Image.fromarray((np.random.rand(518, 518, 3) * 255).astype(np.uint8))

    # 3. Dự đoán Depth Map (Inference)
    print("\nĐang thực hiện inference ước lượng chiều sâu...")
    with torch.no_grad():
        depth_map = model(input_tensor)

    print(f"Kích thước Depth Map đầu ra: {list(depth_map.shape)}")

    # 4. Hiển thị kết quả
    visualize_depth(original_img, depth_map)

if __name__ == "__main__":
    main()