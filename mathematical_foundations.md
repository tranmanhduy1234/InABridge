# NỀN TẢNG TOÁN HỌC CỐT LÕI CỦA KIẾN TRÚC MÔ HÌNH INSTRUCTBLIP VQA
*Phân tích chi tiết các nguyên lý toán học, hàm mất mát và cơ chế tối ưu hóa SOTA*

---

Sự thành công vượt trội của các mô hình Vision-Language thế hệ mới nói chung và kiến trúc **InstructBLIP / GQAModel** nói riêng không đơn thuần đến từ việc tăng quy mô tham số, mà nằm ở sự kết hợp chặt chẽ của các **nền tảng toán học** vững chắc. 

Tài liệu này phân tích chi tiết các nguyên lý toán học cốt lõi từ cơ chế Attention, học tương phản tương quan, hàm kích hoạt phi tuyến tính dạng cổng (Gated Activation), cho tới các phương pháp tối ưu hóa bộ nhớ cấp thấp.

---

## 1. Cơ sở Toán học của Scaled Dot-Product Attention

Cơ chế chú ý tự scaled (Scaled Dot-Product Attention) là hạt nhân của cả Vision Encoder, Q-Former và LLM. Given các ma trận đầu vào Query $Q \in \mathbb{R}^{T_q \times d_k}$, Key $K \in \mathbb{R}^{T_k \times d_k}$, và Value $V \in \mathbb{R}^{T_k \times d_v}$, hàm Attention được định nghĩa là:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 1.1. Chứng minh toán học về sự cần thiết của hệ số tỷ lệ $\frac{1}{\sqrt{d_k}}$
Tại sao lại là hệ số $\frac{1}{\sqrt{d_k}}$ chứ không phải một giá trị nào khác? 

Giả sử các phần tử của vector $q \in Q$ và $k \in K$ là các biến ngẫu nhiên độc lập có cùng phân phối (i.i.d) với kỳ vọng bằng $0$ và phương sai bằng $1$:
$$\mathbb{E}[q_i] = \mathbb{E}[k_i] = 0, \quad \text{Var}(q_i) = \text{Var}(k_i) = 1$$

Tích vô hướng của một hàng $q$ và một cột $k$ trong không gian $d_k$ chiều được tính bằng:
$$u = q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

Áp dụng các tính chất cơ bản của Kỳ vọng ($\mathbb{E}$) và Phương sai ($\text{Var}$) đối với tích các biến độc lập:
1. **Kỳ vọng của từng số hạng:**
   $$\mathbb{E}[q_i k_i] = \mathbb{E}[q_i] \mathbb{E}[k_i] = 0 \cdot 0 = 0$$
   Do đó, kỳ vọng của tích vô hướng:
   $$\mathbb{E}[u] = \sum_{i=1}^{d_k} \mathbb{E}[q_i k_i] = 0$$

2. **Phương sai của từng số hạng:**
   Do các biến độc lập và có kỳ vọng bằng 0:
   $$\text{Var}(q_i k_i) = \mathbb{E}[(q_i k_i)^2] - (\mathbb{E}[q_i k_i])^2 = \mathbb{E}[q_i^2] \mathbb{E}[k_i^2] - 0$$
   Vì $\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 \implies \mathbb{E}[X^2] = \text{Var}(X) + (\mathbb{E}[X])^2$:
   $$\mathbb{E}[q_i^2] = 1 + 0^2 = 1, \quad \mathbb{E}[k_i^2] = 1 + 0^2 = 1$$
   Suy ra:
   $$\text{Var}(q_i k_i) = 1 \cdot 1 = 1$$

3. **Phương sai của tổng $d_k$ số hạng độc lập:**
   $$\text{Var}(u) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$$

Như vậy, khi số chiều ẩn $d_k$ của mô hình tăng lên rất lớn (ví dụ: $d_k = 768$ ở Q-Former hoặc $d_k = 4096$ ở LLM), phương sai của tích vô hướng $u = q \cdot k$ sẽ tăng tuyến tính theo $d_k$ ($\text{Var}(u) = d_k$). Điều này dẫn tới hiện tượng tích vô hướng có trị tuyệt đối cực kỳ lớn.

### 1.2. Hiện tượng triệt tiêu Gradient (Vanishing Gradients) qua Softmax
Hàm Softmax chuyển đổi vector đầu vào $x$ thành phân phối xác suất:
$$s_i = \text{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$$

Đạo hàm riêng của $s_i$ theo phần tử đầu vào $x_k$ là:
$$\frac{\partial s_i}{\partial x_k} = s_i (\delta_{ik} - s_k)$$
Trong đó $\delta_{ik}$ là ký hiệu Kronecker delta ($\delta_{ik}=1$ nếu $i=k$, ngược lại bằng $0$).

* **Hệ quả toán học:** Nếu phương sai của $x$ cực lớn ($d_k$ lớn), Softmax sẽ bị thống trị bởi một vài phần tử cực đại $x_{\max}$, dẫn tới xác suất tương ứng $s_{\max} \approx 1.0$ và các xác suất còn lại $s_{j \neq \max} \approx 0.0$.
* Khi đó, đạo hàm:
  $$\frac{\partial s_i}{\partial x_k} \approx 0 \quad (\forall i, k)$$
* Điều này làm triệt tiêu dòng gradient chảy ngược về mạng (Vanishing Gradient), khiến mô hình không thể học được.

Bằng cách chia tích vô hướng cho $\sqrt{d_k}$, ta chuẩn hóa phương sai của giá trị đầu vào hàm Softmax về mức tiêu chuẩn bằng $1$:
$$\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{1}{d_k} \text{Var}(q \cdot k) = \frac{d_k}{d_k} = 1$$
Đây là nền tảng toán học cốt lõi giúp các mạng Transformer có khả năng hội tụ cực kỳ ổn định ở các số chiều ẩn khổng lồ.

### 1.3. Cơ chế Toán học của Mặt nạ Chú ý Hướng dẫn Ngữ cảnh (InstructBLIP Attention Mask)
Trong Q-Former của InstructBLIP, câu hỏi hướng dẫn $T$ (chiều dài $L_{\text{text}}$) và các truy vấn tự học $Q$ (gồm $N$ queries) cùng đi qua lớp Shared Self-Attention. Để Q-Former có khả năng trích xuất đặc trưng hướng câu hỏi mà không làm hỏng tính biểu diễn độc lập của ngôn ngữ, một mặt nạ chú ý bán cấu trúc (Semi-structured Attention Mask) được thiết lập:

Gọi $M \in \mathbb{R}^{(N + L_{\text{text}}) \times (N + L_{\text{text}})}$ là ma trận mặt nạ chú ý:
$$M_{i, j} = \begin{cases} 
\text{True} & \text{nếu } i < N \quad (\text{Queries được chú ý đến tất cả mọi thứ}) \\
\text{True} & \text{nếu } i \ge N \text{ và } j \ge N \quad (\text{Text tokens được tự chú ý lẫn nhau}) \\
\text{False} & \text{nếu } i \ge N \text{ và } j < N \quad (\text{Text tokens cấm chú ý đến Queries})
\end{cases}$$

Ma trận mặt nạ $M$ này đảm bảo:
1. Các query tokens có thể tự do hấp thụ thông tin ngữ cảnh từ câu hỏi hướng dẫn.
2. Các text tokens đóng vai trò là tiền điều kiện tĩnh, không bị trôi đặc trưng bởi các visual queries trong các lớp self-attention.

---

## 2. Bản chất Toán học của Học Tương phản (InfoNCE Loss)

Trong Giai đoạn 1 (Representation Learning), Q-Former sử dụng hàm mất mát tương phản **Image-Text Contrastive Loss (ITC)**. ITC thực chất là một biến thể của **InfoNCE Loss** (Information Noise-Contrastive Estimation).

### 2.1. Công thức InfoNCE Loss
Với một batch gồm $B$ cặp ảnh-chữ khớp nhau (positive pairs), hàm loss đối chiếu ảnh-sang-chữ được biểu diễn dưới dạng:

$$\mathcal{L}_{\text{InfoNCE}} = - \frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(\cos(z_i^I, z_i^T) / \tau)}{\sum_{j=1}^{B} \exp(\cos(z_i^I, z_j^T) / \tau)}$$

Trong đó:
* $z_i^I, z_i^T$ là vector đặc trưng chuẩn hóa L2 của ảnh và chữ thứ $i$.
* $\cos(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$ là độ tương đồng Cosine.
* $\tau$ là tham số nhiệt độ điều trị (Temperature).

### 2.2. Tối đa hóa Thông tin Hỗ tương (Mutual Information Maximization)
Về mặt toán học lý thuyết thông tin, việc tối thiểu hóa InfoNCE Loss tương đương với việc tối đa hóa **cận dưới** (lower bound) của **Thông tin Hỗ tương (Mutual Information - MI)** giữa hai biến ngẫu nhiên $X$ (Thị giác) và $Y$ (Ngôn ngữ):

$$I(X; Y) = \iint p(x, y) \log \frac{p(x, y)}{p(x) p(y)} \,dx\,dy$$

Chứng minh được chỉ ra rằng:
$$I(X; Y) \ge \log(B) - \mathcal{L}_{\text{InfoNCE}}$$
Trong đó $B$ là kích thước batch (số lượng mẫu phủ định là $B-1$). 

* **Ý nghĩa:** Khi ta giảm thiểu $\mathcal{L}_{\text{InfoNCE}}$, cận dưới của thông tin chung giữa ảnh và mô tả chữ được đẩy lên cao nhất, ép mô hình phải tìm ra các đặc trưng tương hợp sâu sắc giữa hai phương thức dữ liệu khác nhau.

### 2.3. Vai trò của Tham số Nhiệt độ tự học $\tau$ (Learnable Temperature)
Tham số $\tau$ đóng vai trò cực kỳ tinh tế trong toán học phân phối:
* Đạo hàm của loss InfoNCE đối với độ tương đồng của một cặp âm bản $s_{ij} = \cos(z_i^I, z_j^T)$ tỉ lệ nghịch với $\tau$:
  $$\frac{\partial \mathcal{L}}{\partial s_{ij}} \propto \frac{1}{\tau}$$
* **Nếu $\tau$ quá lớn:** Phân phối xác suất Softmax bị san phẳng, mô hình đối xử các mẫu âm bản giống hệt nhau, không tập trung vào các mẫu khó (Hard Negatives).
* **Nếu $\tau$ quá nhỏ:** Phân phối Softmax bị nhọn sắc quá mức, gradient bị thống trị bởi một vài mẫu cực kỳ nhiễu, làm cho việc huấn luyện bị mất ổn định.
* **Giải pháp tự học:** Đặt $\tau$ là một tham số có thể huấn luyện (Learnable Parameter) dưới dạng hàm mũ $\tau = e^{\theta}$ giúp tối ưu hóa động độ dốc của gradient trong suốt quá trình học.

---

## 3. Cơ chế Gating của Hàm Kích Hoạt SwiGLU

Trong các khối Transformer hiện đại, đặc biệt là trong kiến trúc FFN của GQAModel, lớp kích hoạt truyền thống (ReLU hoặc GELU) được thay thế bằng **SwiGLU (Swish Gated Linear Unit)**.

### 3.1. Định nghĩa Toán học của GLU và SwiGLU
Một bộ tuyến tính dạng cổng (Gated Linear Unit - GLU) là tích chập Hadamard ($\otimes$) của hai phép biến đổi tuyến tính, trong đó một phép được lọc qua hàm kích hoạt sigmoid ($\sigma$):
$$\text{GLU}(x, W, V, b, c) = \sigma(x W + b) \otimes (x V + c)$$

SwiGLU thay thế hàm kích hoạt $\sigma$ bằng hàm **Swish** ($\text{Swish}_\beta(x) = x \cdot \text{sigmoid}(\beta x)$), với cấu hình phi tuyến tính không bias ($b = c = 0$ và $\beta = 1$ hay SiLU):

$$\text{SwiGLU}(x) = \text{Swish}_1(x W) \otimes (x V) = (x W \cdot \text{sigmoid}(x W)) \otimes (x V)$$

Sau đó, kết quả được chiếu lại không gian gốc thông qua ma trận trọng số thứ hai $W_2$:

$$\text{FFN}_{\text{SwiGLU}}(x) = \left( \text{Swish}_1(x W) \otimes (x V) \right) W_2$$

```
                          Đầu vào x
                         /         \
                        /           \
                 Tuyến tính (W)   Tuyến tính (V)
                      │               │
                  Swish (SiLU)        │
                      │               │
                      └───────┬───────┘
                              ▼
                        Nhân Hadamard (x)
                              │
                        Tuyến tính (W2)
                              │
                           Đầu ra
```

### 3.2. Tại sao SwiGLU vượt trội hơn ReLU và GELU về mặt toán học?
1. **Khả năng kiểm soát luồng thông tin (Dynamic Gating):** 
   * ReLU chỉ đơn thuần triệt tiêu phần âm: $\max(0, x)$.
   * SwiGLU sử dụng nhánh tuyến tính $x V$ nhân với nhánh cổng $\text{Swish}(x W)$. Nhánh cổng đóng vai trò như một bộ lọc động kiểm soát xem bao nhiêu phần trăm thông tin từ đặc trưng $x V$ được phép truyền qua lớp kế tiếp dựa trên ngữ cảnh.
2. **Khắc phục hiện tượng "chết" neuron (Dying ReLU):**
   * Đạo hàm của ReLU bằng $0$ tại mọi điểm âm, khiến các neuron rơi vào trạng thái "chết" hoàn toàn nếu nhận giá trị âm liên tục.
   * Swish có đạo hàm khác $0$ trên toàn trục số thực (kể cả vùng âm nhỏ nhờ tính chất smooth non-monotonicity), giúp dòng gradient luôn duy trì liên tục.
3. **Mặt cong lỗi trơn tru (Smoother Loss Landscape):**
   * Hàm Swish có đạo hàm bậc một và bậc hai trơn tru tuyệt đối, giúp mặt cong tối ưu hóa của hàm loss ít bị đứt gãy hơn so với ReLU, hỗ trợ các thuật toán tối ưu (AdamW) hội tụ nhanh hơn.

---

## 4. Tối ưu toán học của FlashAttention (Memory-Efficient SDPA)

Lớp `OptimizedFlashMHA` trong dự án GQAModel sử dụng nhân tối ưu hóa bộ nhớ **Scaled Dot Product Attention (SDPA)**. Nguyên lý cốt lõi dựa trên bài báo **FlashAttention**.

### 4.1. Bài toán nghẽn cổ chai bộ nhớ (Memory Bottleneck)
Phép tính Attention thông thường yêu cầu lưu trữ ma trận tương đồng kích thước trung gian khổng lồ:
$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{B \times H \times T \times T}$$

* **Độ phức tạp bộ nhớ:** $O(T^2)$ với $T$ là chiều dài sequence. Với ngữ cảnh dài hoặc ảnh độ phân giải cao ($T=729$ ở SigLIP), ma trận tương đồng này chiếm cực kỳ nhiều bộ nhớ HBM của GPU, dẫn đến việc đọc/ghi liên tục giữa bộ nhớ nhanh SRAM và bộ nhớ chậm HBM (gây thắt nút cổ chai I/O).

### 4.2. Giải pháp Toán học của FlashAttention: Tiling và Online Softmax
FlashAttention không tính toán và lưu trữ toàn bộ ma trận $A$ kích thước $T \times T$ vào bộ nhớ HBM. Thay vào đó, nó chia ma trận $Q, K, V$ thành các khối nhỏ (Tiles) có thể nạp vừa vào bộ nhớ cực nhanh SRAM ($O(1)$ latency), thực hiện tính toán Attention cục bộ và tích lũy kết quả.

Để làm được điều này mà không làm sai lệch kết quả Softmax toàn cục, FlashAttention sử dụng thuật toán **Online Softmax**.

#### Thuật toán Online Softmax biểu diễn toán học:
Thông thường, để tính Softmax của vector $x = [x_1, \dots, x_N]$, ta cần tìm giá trị cực đại toàn cục $m = \max_i x_i$ để tránh tràn số (numerical overflow):
$$m = \max_{i} x_i, \quad d = \sum_{i} e^{x_i - m}, \quad \text{softmax}(x)_i = \frac{e^{x_i - m}}{d}$$

Khi chia nhỏ dữ liệu thành các block, ta không có $m$ toàn cục ngay lập tức. Giả sử ta đang có kết quả của block trước với giá trị cực đại $m^{\text{old}}$ và tổng lũy kế $d^{\text{old}}$. Khi tiếp nhận block mới có giá trị cực đại cục bộ $m^{\text{new}}$ và tổng cục bộ $d^{\text{new}}$:

1. **Cập nhật giá trị cực đại toàn cục mới:**
   $$m^{\text{next}} = \max(m^{\text{old}}, m^{\text{new}})$$

2. **Cập nhật tổng lũy kế chuẩn hóa:**
   $$d^{\text{next}} = d^{\text{old}} \cdot e^{m^{\text{old}} - m^{\text{next}}} + d^{\text{new}} \cdot e^{m^{\text{new}} - m^{\text{next}}}$$

3. **Cập nhật ma trận đặc trưng đầu ra tương ứng:**
   Cách cập nhật này cho phép tính toán chính xác giá trị Softmax toàn cục theo cơ chế streaming/tiling, giúp giảm độ phức tạp bộ nhớ từ $O(T^2)$ xuống **$O(T)$** mà không làm thay đổi bất kỳ một bit kết quả nào của phép tính Attention gốc.

---

## 5. Tối thiểu hóa KL Divergence dẫn xuất đến Cross-Entropy Loss

Trong Giai đoạn 2 (VQA Generative Training), mô hình tối ưu hóa hàm mất mát **Language Modeling Loss ($\mathcal{L}_{\text{LM}}$)**.

### 5.1. Định lý Tích Xác suất và Suy luận tự hồi quy
Mô hình sinh câu trả lời $A = (a_1, a_2, \dots, a_N)$ từ hình ảnh $I$ và câu hỏi $Q$. Theo quy tắc tích xác suất trong lý thuyết xác suất:
$$P(A | I, Q) = \prod_{k=1}^{N} P(a_k | a_{<k}, I, Q)$$

### 5.2. Khoảng cách Kullback-Leibler (KL Divergence)
Mục tiêu là làm cho phân phối xác suất do mô hình dự đoán $P_{\theta}(a_k | \cdot)$ tiệm cận phân phối xác suất thực tế Ground-Truth $P_{\text{data}}(a_k | \cdot)$. Thước đo khoảng cách toán học giữa hai phân phối này là **KL Divergence**:

$$D_{\text{KL}}(P_{\text{data}} \parallel P_{\theta}) = \sum_{a_k} P_{\text{data}}(a_k | \cdot) \log \frac{P_{\text{data}}(a_k | \cdot)}{P_{\theta}(a_k | \cdot)}$$

$$D_{\text{KL}}(P_{\text{data}} \parallel P_{\theta}) = \sum_{a_k} P_{\text{data}}(a_k | \cdot) \log P_{\text{data}}(a_k | \cdot) - \sum_{a_k} P_{\text{data}}(a_k | \cdot) \log P_{\theta}(a_k | \cdot)$$

* Số hạng thứ nhất đại diện cho âm Entropy của dữ liệu gốc (Hằng số không phụ thuộc vào tham số $\theta$ của mô hình).
* Số hạng thứ hai chính là **Cross-Entropy (Entropy chéo)** giữa phân phối thực tế và phân phối dự đoán.
* Do đó, tối thiểu hóa khoảng cách KL Divergence tương đương với tối thiểu hóa Cross-Entropy Loss:

$$\mathcal{L}_{\text{LM}}(\theta) = - \sum_{k=1}^{N} \log P_{\theta}(a_k | a_{<k}, I, Q)$$

Đây là nền tảng toán học tối ưu giúp LLM học phân phối sinh từ chuẩn xác nhất dựa trên các visual prompts được truyền tải từ Q-Former.
