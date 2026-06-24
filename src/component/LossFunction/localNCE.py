import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalNCELoss(nn.Module):
    def __init__(self, k_noise_samples=5):
        """
        Args:
            k_noise_samples (int, optional): Số lượng mẫu nhiễu (k) cho mỗi mẫu thật.
                                            Nếu đặt là None, k sẽ được tự động tính bằng B - 1
                                            (số lượng mẫu âm thực tế trong mỗi batch).
        """
        super().__init__()
        self.k = k_noise_samples

    def forward(self, scoring_function_matrix, noise_dist_matrix):
        """
        Args:
            scoring_function_matrix (Tensor): Ma trận điểm số f_theta(x, c) của TOÀN BỘ các cặp trong Batch.
                                              Kích thước: (B, B) trong đó:
                                              - Đường chéo [i, i] là cặp Thật (Positive)
                                              - Các vị trí còn lại [i, j] là cặp Nhiễu (Negative)
            noise_dist_matrix (Tensor): Phân phối nhiễu q(x) tương ứng cho các phần tử.
                                        Kích thước: (B, B)
        Returns:
            loss (Tensor): Giá trị trung bình hàm lỗi Local NCE tự bình ổn.
        """
        B = scoring_function_matrix.size(0)
        device = scoring_function_matrix.device

        # Xác định k tĩnh (lấy từ self.k) hoặc k động (lấy từ B - 1)
        k = self.k if self.k is not None else (B - 1)

        # --------------------------------------------------------------------------
        # 1. Trích xuất Mẫu Dương (Positive Samples - Dữ liệu Thật D=1)
        # --------------------------------------------------------------------------
        f_pos = torch.diagonal(scoring_function_matrix) # Kích thước: (B,)
        q_pos = torch.diagonal(noise_dist_matrix) # Kích thước: (B,)

        # --------------------------------------------------------------------------
        # 2. Trích xuất Mẫu Âm (Negative Samples - Dữ liệu Nhiễu D=0)
        # --------------------------------------------------------------------------
        mask_neg = ~torch.eye(B, dtype=torch.bool, device=device)
        print(f"Ma trận Mask Neg (True là vị trí mẫu âm):\n{mask_neg}\n")
        
        f_neg = scoring_function_matrix[mask_neg] # Kích thước: (B * (B - 1),)
        q_neg = noise_dist_matrix[mask_neg] # Kích thước: (B * (B - 1),)

        # --------------------------------------------------------------------------
        # 3. Tính toán trên Log-Space nhằm đảm bảo ổn định số học (Numerical Stability)
        # --------------------------------------------------------------------------
        log_f_pos = torch.log(f_pos + 1e-8)
        log_q_pos = torch.log(q_pos + 1e-8)
        log_f_neg = torch.log(f_neg + 1e-8)
        log_q_neg = torch.log(q_neg + 1e-8)
        log_k = torch.log(torch.tensor(k, dtype=torch.float32, device=device))

        # Áp dụng công thức Bayes + Giả định Self-Normalization Z(c)=1:
        # -log p(D=1 | x, c) = -log_f_pos + log(f_pos + k * q_pos)
        loss_pos = -log_f_pos + torch.logaddexp(log_f_pos, log_k + log_q_pos)
        
        # -log p(D=0 | x', c) = -log(k * q_neg) + log(f_neg + k * q_neg)
        loss_neg = -(log_k + log_q_neg) + torch.logaddexp(log_f_neg, log_k + log_q_neg)

        # --------------------------------------------------------------------------
        # 4. Tổng hợp Hàm Lỗi Toàn Cục (Đã sửa hệ số trọng số mẫu âm chuẩn lý thuyết)
        # --------------------------------------------------------------------------
        # Kỳ vọng trên mẫu dương và mẫu âm: L = E[-log P(D=1|pos)] + k * E[-log P(D=0|neg)]
        total_loss = loss_pos.mean() + k * loss_neg.mean()
        
        return total_loss

# ==============================================================================
# ĐOẠN MÃ MÔ PHỎNG LUỒNG CHẠY THỰC TẾ (Sử dụng Hook để soi Gradient thông minh)
# ==============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    k_samples = 5
    
    print("--- KHỞI TẠO BIẾN GIẢ LẬP ---")
    # Giả lập ma trận điểm số f_theta(x, c) do mạng neural sinh ra (đã qua lớp phi tuyến tính dương như Exp/Softplus)
    # Cố tình đặt phần tử [0, 0] (Mẫu thật đầu tiên) có điểm cực thấp (0.01) -> Mô hình đang "học dốt" cặp này
    f_theta = torch.tensor([
        [0.01, 1.20, 0.90], # Ngữ cảnh c_0 tương tác với các đối tượng x_0, x_1, x_2
        [0.10, 4.50, 0.30], # Ngữ cảnh c_1 (Mẫu thật [1,1]=4.50 -> Mô hình đang "học tốt")
        [0.20, 0.15, 3.80]  # Ngữ cảnh c_2 (Mẫu thật [2,2]=3.80 -> Mô hình đang "học tốt")
    ], dtype=torch.float32, requires_grad=True)
    print("Ma trận điểm số f_theta (trước khi tính Loss) có kích thước: " + f_theta.shape.__str__() + "\n")
    
    batch_size = f_theta.size(0) # batch_size = 3 khớp với f_theta thực tế

    # Giả lập phân phối nhiễu tĩnh q(x) (Ví dụ: Tần suất unigram đồng đều)W
    q_x = torch.full((batch_size, batch_size), 0.20, dtype=torch.float32)

    # Đăng ký Hook để theo dõi và in ra ma trận Gradient phạt của mạng Neural
    gradients = {}
    f_theta.register_hook(lambda grad: gradients.update({'f_theta_grad': grad}))

    # Khởi tạo lớp Loss Local NCE
    local_nce_criterion = LocalNCELoss(k_noise_samples=k_samples)

    # Tính toán Loss
    loss = local_nce_criterion(f_theta, q_x)
    
    # Lan truyền ngược để sinh Gradient
    loss.backward()

    # --------------------------------------------------------------------------
    # IN KẾT QUẢ ĐỂ KIỂM CHỨNG LÝ THUYẾT ĐỘ DỐC THÔNG MINH
    # --------------------------------------------------------------------------
    print(f"Tổng giá trị Loss: {loss.item():.4f}\n")
    print("Ma trận Điểm số f_theta gốc:")
    print(f_theta.data)
    print("\nMa trận Gradient thu được sau Backward (Lực cập nhật trọng số):")
    print(gradients['f_theta_grad'])
    
    print("\n--- PHÂN TÍCH HÀNH VI ---")
    print(f"1. Tại vị trí [0, 0] (Mẫu thật nhưng điểm quá thấp = 0.01):")
    print(f"   => Lực Gradient âm cực lớn: {gradients['f_theta_grad'][0, 0].item():.4f}")
    print(f"   => Khi cập nhật (theta - eta * grad), nó sẽ thành một lực CỘNG cực mạnh đẩy điểm 0.01 lên!")
    
    print(f"\n2. Tại vị trí [1, 1] (Mẫu thật đã có điểm cao = 4.50):")
    print(f"   => Lực Gradient triệt tiêu về sát 0: {gradients['f_theta_grad'][1, 1].item():.4f}")
    print(f"   => Mô hình ổn định, không tác động nhiều vào vùng này nữa.")
 
    print(f"\n3. Tại vị trí [0, 1] (Mẫu nhiễu bị chấm điểm cao nhầm = 1.20):")
    print(f"   => Lực Gradient dương (+): {gradients['f_theta_grad'][0, 1].item():.4f}")
    print(f"   => Khi cập nhật (theta - eta * grad), dấu (+) chuyển thành lực TRỪ kéo sập điểm mẫu nhiễu này xuống!")