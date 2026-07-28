# Danh sách Paper đề xuất để nắm bắt hành vi của mô hình (InA-Bridge / GQAModel)

Dựa trên cấu trúc mã nguồn và tài liệu thiết kế của dự án, mô hình **InA-Bridge** (hay **GQAModel / SDQ-VLM**) là một Vision-Language Model (VLM) lai ghép độc đáo, kết hợp giữa **DINOv2** (hoặc **SigLIP2**), một **Q-Former** tùy chỉnh dựa trên **DeBERTa-v3**, và một LLM decoder (**Qwen2.5**).

Để hiểu sâu sắc hành vi, thuật toán, hàm loss và các lỗi tiềm ẩn (chẳng hạn như xung đột attention hay cold-start alignment), dưới đây là các nhóm paper khoa học cốt lõi cần nghiên cứu:

---

## 1. Kiến trúc Căn bản của Q-Former (Bridge)

Q-Former đóng vai trò cầu nối rút gọn đặc trưng hình ảnh và căn chỉnh với không gian ngôn ngữ.

### [1] BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
* **Tác giả:** Junnan Li et al. (Salesforce Research, 2023)
* **Ý nghĩa:** Đây là paper khai sinh ra kiến trúc Q-Former với quy trình huấn luyện hai giai đoạn (Representation Learning và Generative Learning).
* **Liên quan trong Codebase:** 
  * Định hình luồng dữ liệu của 3 hàm loss cốt lõi ở Stage 1: **ITC** (Image-Text Contrastive), **ITM** (Image-Text Matching) và **ITG** (Image-Grounded Text Generation).
  * Mô phỏng cấu trúc 2 nhánh song song chia sẻ Self-Attention trong [demopipeline.py](file:///home/tranmanhduy/Workspace/chuyen_nganh/InA-Bridge/src/component/demopipeline.py).

### [2] InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning
* **Tác giả:** Wenliang Dai et al. (Salesforce Research, 2023)
* **Ý nghĩa:** Giới thiệu cơ chế **Instruction-aware Q-Former**, truyền thêm prompt hướng dẫn của người dùng vào lớp Self-Attention của Q-Former để trích xuất đặc trưng thị giác bám sát câu hỏi.
* **Liên quan trong Codebase:**
  * Giúp hiểu cơ chế hoạt động của mặt nạ chú ý hướng dẫn ngữ cảnh (Semi-structured Attention Mask) được phân tích trong [mathematical_foundations.md:L63-77](file:///home/tranmanhduy/Workspace/chuyen_nganh/InA-Bridge/mathematical_foundations.md#L63-L77).

---

## 2. Cơ chế Disentangled Attention của DeBERTa (Backbone của Q-Former)

Dự án sử dụng trọng số khởi tạo của **DeBERTa-v3-base** cho Q-Former, dẫn đến một vấn đề xung đột kỹ thuật nghiêm trọng.

### [3] DeBERTa: Decoding-enhanced BERT with Disentangled Attention
* **Tác giả:** Pengcheng He et al. (Microsoft Research, 2020)
* **Ý nghĩa:** Giải thích cơ chế **Disentangled Attention** - tách biệt biểu diễn Nội dung (Content) và Vị trí tương đối (Relative Position) thành hai vector độc lập trong tính toán Attention.
* **Liên quan trong Codebase:**
  * Đọc để hiểu lớp [DisentangledSelfAttn](file:///home/tranmanhduy/Workspace/chuyen_nganh/InA-Bridge/src/component/Q_former/qformer.py#L33) trong file [qformer.py](file:///home/tranmanhduy/Workspace/chuyen_nganh/InA-Bridge/src/component/Q_former/qformer.py).
  * Giải thích **Rủi ro lớn [KT-1] (Disentangled attention silent failure)** được nêu trong [risk.txt](file:///home/tranmanhduy/Workspace/chuyen_nganh/InA-Bridge/documents/risk.txt) và [strategy.txt](file:///home/tranmanhduy/Workspace/chuyen_nganh/InA-Bridge/documents/strategy.txt): Đặc trưng của patch ảnh từ DINOv2 không có thứ tự Relative Position tự nhiên như văn bản, việc áp dụng disentangled attention trực tiếp lên Cross-Attention sẽ tạo ra nhiễu thông tin vị trí rác, khiến mô hình bị suy giảm khả năng hiểu không gian ảnh.

---

## 3. Đặc trưng Thị giác (Vision Encoder)

### [4] DINOv2: Learning Robust Visual Features without Supervision
* **Tác giả:** Maxime Oquab et al. (Meta AI, 2023)
* **Ý nghĩa:** Giới thiệu phương pháp học tự giám sát thuần thị giác (self-supervised) để thu được đặc trưng biên dạng, chiều sâu và ngữ nghĩa không gian cực tốt.
* **Liên quan trong Codebase:**
  * Mã nguồn lớp [ImageEncoder](file:///home/tranmanhduy/Workspace/chuyen_nganh/InA-Bridge/src/component/ImageEncoder/imageEncoder.py#L12) trong [imageEncoder.py](file:///home/tranmanhduy/Workspace/chuyen_nganh/InA-Bridge/src/component/ImageEncoder/imageEncoder.py) load mô hình DINOv2.
  * Hiểu lý do xảy ra **Rủi ro [CL-1] (Alignment cold-start problem)** trong [risk.txt](file:///home/tranmanhduy/Workspace/chuyen_nganh/InA-Bridge/documents/risk.txt): DINOv2 không được căn chỉnh ngôn ngữ từ trước (như CLIP), nên Q-Former cần nhiều dữ liệu huấn luyện hơn để tự bootstrap mối liên kết ảnh-chữ từ đầu.

### [5] Sigmoid Loss for Language-Image Pre-Training (SigLIP)
* **Tác giả:** Xiaohua Zhai et al. (Google DeepMind, 2023)
* **Ý nghĩa:** Đề xuất hàm loss phân loại nhị phân Sigmoid thay thế cho Softmax toàn cục của CLIP, tối ưu hóa hiệu năng căn chỉnh ảnh-chữ với batch size nhỏ hơn.
* **Liên quan trong Codebase:**
  * Định hướng phát triển Next-Gen VLM trong tài liệu [DataStrategy.md](file:///home/tranmanhduy/Workspace/chuyen_nganh/InA-Bridge/DataStrategy.md) đề xuất dùng SigLIP2 kết hợp Qwen2.5 để nâng cấp khả năng biểu diễn.

---

## 4. Mô hình Ngôn ngữ và Bộ chiếu (LLM & Projector)

### [6] Qwen2.5 Technical Report
* **Tác giả:** Qwen Team (Alibaba, 2024)
* **Ý nghĩa:** Cung cấp thông tin chi tiết về kiến trúc Qwen2.5 (SwiGLU activation, RoPE, Grouped-Query Attention) và định dạng Chat Template.
* **Liên quan trong Codebase:**
  * Cấu hình LLM trong [llm.py](file:///home/tranmanhduy/Workspace/chuyen_nganh/InA-Bridge/src/component/LLM/llm.py) và [InABridgeModel.py](file:///home/tranmanhduy/Workspace/chuyen_nganh/InA-Bridge/src/component/InABridgeModel.py).
  * Việc áp dụng hàm kích hoạt **SwiGLU** và **RMSNorm** trong bộ chiếu [QwenProjector](file:///home/tranmanhduy/Workspace/chuyen_nganh/InA-Bridge/src/component/MLP/multiLayerPerceptron.py#L5) nhằm mục đích đồng nhất hóa không gian toán học với Qwen2.5-7B (phân tích chi tiết tại [mathematical_foundations.md:L114-158](file:///home/tranmanhduy/Workspace/chuyen_nganh/InA-Bridge/mathematical_foundations.md#L114-L158)).

---

## 5. Nền tảng Toán học và Học Tương phản

### [7] Representation Learning with Contrastive Predictive Coding (InfoNCE Loss)
* **Tác giả:** Aaron van den Oord et al. (Google DeepMind, 2018)
* **Ý nghĩa:** Định nghĩa toán học của **InfoNCE Loss** dưới góc nhìn Lý thuyết thông tin (tối đa hóa cận dưới của Mutual Information).
* **Liên quan trong Codebase:**
  * Hàm mất mát ITC (Stage 1 Q-Former) được phân tích chi tiết trong [mathematical_foundations.md:L79-112](file:///home/tranmanhduy/Workspace/chuyen_nganh/InA-Bridge/mathematical_foundations.md#L79-L112).

### [8] Noise-contrastive estimation: A new estimation principle for unnormalized statistical models
* **Tác giả:** Michael Gutmann & Aapo Hyvärinen (AISTATS, 2010)
* **Ý nghĩa:** Khởi nguồn toán học của phương pháp ước lượng tương phản nhiễu (NCE) phục vụ huấn luyện các mô hình xác suất mà không cần tính mẫu số chuẩn hóa phân chia (Partition Function) đắt đỏ.
* **Liên quan trong Codebase:**
  * Đọc để nắm vững bản chất toán học về độ dốc gradient và khả năng tự học tự chuẩn hóa của mô hình ngôn ngữ (xem tài liệu [note.txt](file:///home/tranmanhduy/Workspace/chuyen_nganh/InA-Bridge/documents/note.txt)).

---

## Tóm tắt Lộ trình Đọc nhanh (Quick Reading Path)

Để tiếp cận dự án một cách nhanh nhất, bạn nên đọc theo trình tự sau:
1. **Tuần 1 (Module đơn lẻ):** Đọc phần Disentangled Attention trong paper **DeBERTa [3]** + paper **DINOv2 [4]** để hiểu sự xung đột giữa hai module này.
2. **Tuần 2 (Mô hình cầu nối):** Đọc **BLIP-2 [1]** và **InstructBLIP [2]** để nắm vững luồng xử lý đa nhiệm của Q-Former.
3. **Tuần 3 (Toán học nền tảng):** Đọc paper **InfoNCE [7]** và tài liệu [mathematical_foundations.md](file:///home/tranmanhduy/Workspace/chuyen_nganh/InA-Bridge/mathematical_foundations.md) của dự án.
