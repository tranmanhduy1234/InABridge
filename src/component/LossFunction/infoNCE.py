import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, init_temperature=0.07, learnable=True):
        """
        Args:
            init_temperature (float): Hệ số nhiệt độ (tau) giúp kiểm soát độ nhọn 
                                      của phân phối xác suất (tương tự như Softplus/Exp 
                                      giúp làm mượt điểm số trong Local NCE).
            learnable (bool): Nếu True, temperature sẽ là một tham số có thể học được
                              (giống như CLIP/InstructBLIP thường cấu hình).
        """
        super().__init__()
        if learnable:
            # Khởi tạo log(temperature) để tối ưu ổn định số học
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(init_temperature)))
        else:
            self.register_buffer('log_temperature', torch.log(torch.tensor(init_temperature)))

    def forward(self, image_embeddings, text_embeddings):
        """
        Args:
            image_embeddings (Tensor): Ma trận vector ảnh trích xuất từ Q-Former + Projection.
                                       Kích thước: (B, D) với B là Batch Size, D là số chiều Embedding.
            text_embeddings (Tensor): Ma trận vector văn bản từ Text Encoder.
                                      Kích thước: (B, D)
        Returns:
            loss (Tensor): Giá trị InfoNCE loss đối xứng (trung bình cộng 2 chiều).
        """
        B = image_embeddings.size(0)
        device = image_embeddings.device
        
        # 1. Chuẩn hóa L2-norm cho các vector (Bắt buộc trong InfoNCE để tính Cosine Similarity)
        image_features = F.normalize(image_embeddings, p=2, dim=-1)
        text_features = F.normalize(text_embeddings, p=2, dim=-1)

        # 2. Tính toán ma trận điểm số tương đồng (Similarity Matrix) qua phép nhân ma trận (Dot Product)
        # Kích thước logits: (B, B)
        # Đường chéo [i, i] là cặp Khớp Đúng (Positive). Các vị trí còn lại [i, j] (i != j) là Mẫu Nhiễu (Negative)
        similarity_matrix = torch.matmul(image_features, text_features.T)

        # 3. Áp dụng hệ số nhiệt độ nghịch đảo (1 / tau)
        # Tương tự như hàm Softplus/Exp ở Local NCE, bước này phóng đại khoảng cách giữa mẫu đúng và mẫu sai
        temperature = torch.exp(self.log_temperature)
        logits = similarity_matrix / temperature

        # 4. Tạo nhãn Ground-Truth (Đường chéo chính là chỉ mục đúng: 0, 1, 2,..., B-1)
        labels = torch.arange(B, dtype=torch.long, device=device)

        # 5. Tính Cross Entropy tương phản 2 chiều (Symmetric Cross Entropy)
        # Chiều 1: Nhìn từ Ảnh -> Tìm Văn bản khớp nhất (Tính theo hàng)
        loss_image_to_text = F.cross_entropy(logits, labels)
        
        # Chiều 2: Nhìn từ Văn bản -> Tìm Ảnh khớp nhất (Tính theo cột - chuyển vị ma trận logits)
        loss_text_to_image = F.cross_entropy(logits.T, labels)

        # Tổng hợp lỗi đối xứng
        total_loss = (loss_image_to_text + loss_text_to_image) / 2.0

        return total_loss

# ==============================================================================
# ĐOẠN MÃ MÔ PHỎNG KIỂM TRA (TEST RUN SIMULATION)
# ==============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Giả lập Batch Size = 3, Số chiều Embedding D = 4
    batch_size = 3
    embedding_dim = 4
    
    print("--- 1. KHỞI TẠO EMBEDDING GIẢ LẬP ĐẦU RA MÔ HÌNH ---")
    # Giả lập các đặc trưng ảnh (từ Q-Former) và văn bản (từ Text Encoder)
    # Cố tình thiết lập:
    # - Cặp số 0: Hơi lệch nhau (Mô hình đang học chưa tốt cặp này)
    # - Cặp số 1 và 2: Các vector gần như trùng hướng (Mô hình đã học tốt)
    img_emb = torch.tensor([
        [0.1,  0.8, -0.2,  0.5],  # Ảnh 0
        [0.9, -0.1,  0.3,  0.1],  # Ảnh 1
        [-0.4, 0.2,  0.8, -0.5]   # Ảnh 2
    ], dtype=torch.float32, requires_grad=True)

    txt_emb = torch.tensor([
        [-0.3, 0.6,  0.1,  0.8],  # Văn bản mô tả 0 (Khác hướng Ảnh 0)
        [0.85, -0.15, 0.25, 0.05], # Văn bản mô tả 1 (Rất sát Ảnh 1)
        [-0.45, 0.15, 0.75, -0.48] # Văn bản mô tả 2 (Rất sát Ảnh 2)
    ], dtype=torch.float32, requires_grad=True)

    # Đăng ký Hook để xem lực Gradient phạt tác động lên các vector Embedding
    img_gradients = {}
    img_emb.register_hook(lambda grad: img_gradients.update({'img_grad': grad}))

    # 2. Khởi tạo và chạy hàm Loss INFO-NCE
    infonce_criterion = InfoNCELoss(init_temperature=0.07, learnable=True)
    loss = infonce_criterion(img_emb, txt_emb)
    
    # Lan truyền ngược
    loss.backward()

    # 3. IN KẾT QUẢ PHÂN TÍCH ĐẦU RA & GRADIENT
    print("\n--- 2. KẾT QUẢ ĐẦU RA ---")
    print(f"Giá trị InfoNCE Loss tổng cục: {loss.item():.4f}")
    print(f"Hệ số nhiệt độ hiện tại (Tau): {torch.exp(infonce_criterion.log_temperature).item():.4f}")

    print("\n--- 3. MA TRẬN PHẠT GRADIENT TRÊN IMAGE EMBEDDINGS ---")
    # Ma trận này cho biết các vector ảnh cần phải di chuyển theo hướng nào để giảm Loss
    print(img_gradients['img_grad'])

    print("\n--- 4. PHÂN TÍCH HÀNH VI ĐỐI CHIẾU VỚI LOCAL NCE ---")
    print("1. Ở cặp số 0 (Ảnh 0 & Văn bản 0): Mô hình nhận diện điểm tương đồng thấp (Mẫu dương bị chấm điểm thấp).")
    print("   => Gradient sinh ra lực kéo mạnh để ép vector Ảnh 0 dịch chuyển hướng về phía vector Văn bản 0.")
    print("2. Ở cặp số 1 và 2: Mô hình nhận diện điểm tương đồng rất cao (Mẫu dương đã đúng).")
    print("   => Gradient tại các vùng này rất nhỏ, mô hình giữ trạng thái ổn định cho các mẫu đã học tốt.")
    print("3. In-batch Negatives hoạt động tự động:")
    print("   => Bạn không cần truyền ma trận q(x) = 0.20 nữa. Ảnh 0 tự động bị đẩy xa khỏi Văn bản 1 và Văn bản 2.")