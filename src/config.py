# ====================================================================================================
# CẤU HÌNH DÙNG CHUNG
# ====================================================================================================

# MODEL / BACKBONE
BERT_MODEL_ID = "bert-base-uncased"  # Ví dụ: "bert-base-uncased"; pretrained BERT dùng cho embeddings và khởi tạo Q-Former.
VISION_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"  # Ví dụ: "facebook/dinov3-vitl16-pretrain-lvd1689m"; pretrained vision encoder DINOv3.
VISION_RETURN_LAYER = -1  # Ví dụ: -1; lấy hidden states của lớp cuối vision encoder.
NUM_QUERIES = 128  # Ví dụ: 128; số query tokens học được của Q-Former.
CROSS_ATTN_EVERY = 2  # Ví dụ: 2; thêm cross-attention mỗi 2 block, bắt đầu từ block đầu tiên.
HIDDEN_DIM = 768  # Ví dụ: 768; chiều hidden của Q-Former, phải khớp hidden_size của BERT.

# TOKENIZER CHO BERT / Q-FORMER
TOKENIZER_MODEL_ID = BERT_MODEL_ID  # Ví dụ: "bert-base-uncased"; tên hoặc đường dẫn tokenizer, phải khớp với text embeddings của model.
TOKENIZER_USE_FAST = True  # Checkpoint yêu cầu fast tokenizer để lưu và khôi phục đầy đủ backend.

# RUNTIME
SEED = 42  # Seed cho Python, NumPy và PyTorch khi khởi tạo training.
DEVICE = "cuda"  # Thiết bị training: "cuda", "cuda:0" hoặc "cpu".

# ====================================================================================================
# STAGE 1
# ====================================================================================================

# ITC FEATURES VÀ EMA
ITC_DIM = 256  # Ví dụ: 256; chiều embedding ảnh và text dùng cho ITC.
NORMALIZE_EPS = 1e-12  # Ví dụ: 1e-12; ngưỡng tránh chia cho 0 khi chuẩn hóa embedding ITC.
MOMENTUM = 0.995  # Ví dụ: 0.995; hệ số EMA, cập nhật momentum = m * momentum + (1 - m) * online.

# DATALOADER
DATASET_NAMESPACE = "blip3o"  # Định danh dataset ổn định giữa các máy và train/validation; dùng cùng namespace cho cùng ảnh.
IS_TRAINING = True  # Ví dụ: True; bật augmentation train và chọn cache train.sqlite.
JSON_PATH_TRAIN = "DatasetProject/BLIP3o/dataset_metadata.json"  # Ví dụ: "data/train.json"; manifest ảnh–text, đường dẫn tương đối từ gốc dự án hoặc tuyệt đối.
IMAGE_DIR = "DatasetProject/BLIP3o/Image"  # Ví dụ: "data/images"; thư mục ảnh dùng chung cho train và validation, hai tập được phân chia bằng JSON.
JSON_PATH_VAL = "DatasetProject/BLIP3o/dataset_metadata.json"  # Ví dụ: "data/validation.json"; manifest validation, dùng tập riêng khi đánh giá mô hình.
CACHE_DIR = "cache/dataloader_demo"  # Ví dụ: "cache/train"; thư mục SQLite cache, cần đổi hoặc xóa cache khi đổi manifest.
CACHE_CHUNK_SIZE = 1000  # Ví dụ: 1000; số bản ghi mỗi lần ghi metadata vào SQLite.

IMAGE_SIZE = 640  # Ví dụ: 640; chiều cao và rộng ảnh đầu ra, tính bằng pixel.
NORMALIZATION = None  # (mean, std) theo RGB; None bỏ chuẩn hóa, không tự lấy từ vision encoder.
CROP_SCALE = (0.85, 1.0)  # Ví dụ: (0.85, 1.0); khoảng tỷ lệ diện tích crop so với ảnh gốc.
CROP_RATIO = (0.9, 1.1)  # Ví dụ: (0.9, 1.1); khoảng tỷ lệ chiều rộng/chiều cao của crop.
INTERPOLATION = "bicubic"  # Ví dụ: "bicubic"; phương pháp nội suy khi resize ảnh.
ANTIALIAS = True  # Ví dụ: True; giảm răng cưa khi resize với nội suy hỗ trợ.
HORIZONTAL_FLIP_PROBABILITY = 0.3  # Ví dụ: 0.3; xác suất lật ngang ảnh khi train.
COLOR_JITTER = (0.15, 0.15, 0.15, 0.03)  # Ví dụ: (0.15, 0.15, 0.15, 0.03); biên độ thay đổi brightness, contrast, saturation và hue.
COLOR_JITTER_PROBABILITY = 0.4  # Ví dụ: 0.4; xác suất áp dụng biến đổi màu khi train.
GRAYSCALE_PROBABILITY = 0.05  # Ví dụ: 0.05; xác suất chuyển ảnh sang thang xám, vẫn giữ ba kênh.
BLUR_KERNEL_SIZE = 5  # Ví dụ: 5; kích thước kernel Gaussian blur, là số nguyên dương lẻ.
BLUR_SIGMA = (0.1, 1.5)  # Ví dụ: (0.1, 1.5); khoảng sigma được lấy ngẫu nhiên cho Gaussian blur.
BLUR_PROBABILITY = 0.1  # Ví dụ: 0.1; xác suất làm mờ ảnh khi train.

# TOKENIZATION
MAX_LENGTH = 128  # Ví dụ: 128; giới hạn số token mỗi văn bản khi bật truncation.
TOKENIZER_PADDING = True  # Ví dụ: True hoặc "max_length"; pad đến câu dài nhất trong batch hoặc đến MAX_LENGTH.
TOKENIZER_TRUNCATION = True  # Ví dụ: True; cắt văn bản vượt quá MAX_LENGTH.

BATCH_SIZE = 8  # Ví dụ: 8; số mẫu mỗi batch, cần ít nhất 2 cho hard negative sampling.
NUM_WORKERS = 0  # Ví dụ: 4; số worker đọc dữ liệu, 0 để chạy trong tiến trình chính.
DROP_LAST = False  # Ví dụ: True; bỏ batch cuối thiếu mẫu khi train, validation luôn giữ.
SHUFFLE = False  # Ví dụ: None; None tự bật theo IS_TRAINING, True/False để chỉ định trực tiếp.
PIN_MEMORY = False  # Ví dụ: None; None tự bật khi có CUDA, True/False để chỉ định trực tiếp.
PERSISTENT_WORKERS = False  # Ví dụ: True; giữ worker giữa các lượt duyệt dữ liệu, chỉ áp dụng khi NUM_WORKERS > 0.
PREFETCH_FACTOR = 2  # Ví dụ: 2; số batch mỗi worker nạp trước, chỉ áp dụng khi NUM_WORKERS > 0.

# DATALOADER DEMO
DEMO_NUM_IMAGES = 5  # Ví dụ: 5; số ảnh tối đa hiển thị trong mỗi batch của demo.
DEMO_FIGURE_SIZE = (8, 8)  # Ví dụ: (8, 8); kích thước khung hình demo theo inch.
DEMO_TITLE_FONT_SIZE = 10  # Ví dụ: 10; cỡ chữ caption trong hình demo.

# ENGINE TRAINING
EPOCHS = 20  # Tổng số epoch training.
WARMUP_EPOCHS = 1  # Số epoch tăng learning rate tuyến tính từ 0 lên LR trước cosine decay.
ACCUMULATION_STEPS = 1  # Số micro-batch mỗi optimizer step; batch hiệu dụng = BATCH_SIZE * giá trị này trên mỗi GPU.

# AdamW và warmup + cosine scheduler; scheduler.step() theo optimizer step.
LR = 1e-4  # Learning rate đỉnh sau warmup.
MIN_LR = 5e-6  # Learning rate cuối cosine decay; min_lr_ratio = MIN_LR / LR.
WEIGHT_DECAY = 0.05  # Weight decay cho các trọng số được regularize.
ADAM_BETAS = (0.9, 0.999)  # Hệ số trung bình động gradient và bình phương gradient của AdamW.
ADAM_EPS = 1e-8  # Hằng số ổn định số học trong mẫu số AdamW.
MAX_GRAD_NORM = 1.0  # Ngưỡng clip gradient sau unscale và trước optimizer step; None để tắt.

AMP_ENABLED = True  # Bật mixed precision khi thiết bị hỗ trợ.
AMP_DTYPE = "bfloat16"  # "bfloat16" hoặc "float16"; float16 cần GradScaler trên CUDA.

# Stage1Criterion và MoCoQueue.
ITC_WEIGHT = 1.0  # Trọng số image-text contrastive loss.
ITM_WEIGHT = 1.0  # Trọng số image-text matching loss.
ITG_WEIGHT = 1.0  # Trọng số image-grounded text generation loss.
PSEUDO_WEIGHT = 0.4  # Tỷ lệ soft targets đích từ momentum model trong ITC, thuộc [0, 1]; giữ cố định sau warmup.
PSEUDO_WARMUP_EPOCHS = 1  # Tăng tuyến tính pseudo weight từ 0 lên PSEUDO_WEIGHT theo tiến độ trong số epoch này; 0 để áp dụng mức đích ngay.
TEMPERATURE = 0.07  # Temperature dương dùng để tính logits ITC.
QUEUE_SIZE = 4096  # Số cặp feature ảnh–text tối đa lưu trong MoCoQueue.

# CHECKPOINT VÀ LOG
OUTPUT_DIR = "outputs/stage1"  # Thư mục log và kết quả training, tính từ gốc dự án.
CHECKPOINT_DIR = "src/checkpoints/stage1"  # Thư mục gốc chứa các lần chạy riêng biệt.
MODEL_VERSION = "v1"  # Nhãn kiến trúc; dùng Git tag/commit tương ứng khi thay đổi implementation.
RUN_NAME = "pretrain_001"  # Run mới: <CHECKPOINT_DIR>/<MODEL_VERSION>/<RUN_NAME>/.
INIT_CHECKPOINT = None  # Checkpoint nguồn để fine-tune; optimizer và tiến độ bắt đầu mới.
RESUME_CHECKPOINT = None  # Resume run cũ theo cấu hình đã lưu; không dùng cùng INIT_CHECKPOINT.
# Khi resume, DEVICE lấy từ config hiện tại; MODEL_VERSION/RUN_NAME không đổi nơi lưu của run cũ.
# Step là số lần cập nhật optimizer thành công sau accumulation, không tính step bị AMP bỏ qua.
# Đếm global step xuyên suốt các epoch và khôi phục từ checkpoint khi resume.
SAVE_EVERY_STEPS = 1000  # Chu kỳ lưu last.pt theo optimizer step; 0 tắt lịch, vẫn lưu cuối training.
VAL_EVERY_STEPS = 1000  # Chu kỳ validation và chọn best.pt; 0 tắt, nếu bật chạy thêm cuối training.
LOG_EVERY_STEPS = 10  # Chu kỳ in loss training ra stdout theo optimizer step; 0 tắt.

# ====================================================================================================
# STAGE 2
# ====================================================================================================

# Bổ sung cấu hình riêng cho Stage 2 tại đây khi triển khai.
