from dataclasses import dataclass
from pathlib import Path
import torch
# Runtime
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Thiết bị chạy mô hình: ưu tiên GPU CUDA nếu có.

# Image encoder
IMAGE_ENCODER_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"  # Mô hình pretrained dùng để trích xuất đặc trưng ảnh.
IMAGE_SIZE = 512  # Kích thước ảnh đầu vào sau xử lý (pixel).
CENTER_CROP = False  # Bật cắt vùng giữa ảnh khi tiền xử lý.
RETURN_LAYER = -1  # Lớp lấy đặc trưng ảnh; -1 là lớp cuối.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}  # Các phần mở rộng tệp ảnh được chấp nhận.

# QFormer and projector
QFORMER_MODEL_ID = "bert-base-uncased"  # Mô hình BERT pretrained dùng để khởi tạo QFormer.
NUM_QUERIES = 128  # Số query token học được để tổng hợp thông tin ảnh.
CROSS_ATTN_EVERY = 2  # Khoảng cách giữa các lớp cross-attention của QFormer.
ITC_DIM = 256  # Số chiều đặc trưng dùng cho đối sánh tương phản ảnh–văn bản.
ITC_NORMALIZE_EPS = 1e-12  # Hằng số nhỏ tránh chia cho 0 khi chuẩn hóa đặc trưng ITC.
QFORMER_PADDING_ONLY_KEY = True  # Chỉ che padding ở phía key trong attention mask.
QFORMER_DEFAULT_OBJECTIVE = "itm"  # Mục tiêu mặc định quyết định kiểu attention mask (itc/itm/itg).
PROJECTOR_HIDDEN_MULTIPLIER = 2  # Hệ số nhân chiều QFormer để tính chiều ẩn của projector.
PROJECTOR_DROPOUT = 0.0  # Xác suất dropout trong projector.

# LLM
LLM_MODEL_ID = "Qwen/Qwen3-4B"  # Mô hình ngôn ngữ pretrained dùng để sinh văn bản.
COMPUTE_TYPE = torch.bfloat16  # Kiểu dữ liệu tính toán của mô hình ngôn ngữ.
QLORA = False  # Bật lượng tử hóa 4-bit và gắn adapter LoRA cho LLM.
QLORA_TRAINABLE = False  # Cho phép cập nhật trọng số adapter LoRA.
QLORA_QUANT_TYPE = "nf4"  # Kiểu lượng tử hóa 4-bit của QLoRA.
QLORA_DOUBLE_QUANT = True  # Lượng tử hóa thêm các hệ số lượng tử để tiết kiệm bộ nhớ.
GRADIENT_CHECKPOINTING = False  # Tính lại activation khi backward để giảm bộ nhớ huấn luyện.
TUNE_QFORMER_STAGE2 = False  # Cho phép tinh chỉnh QFormer ở giai đoạn 2.

@dataclass
class QLoRAConfig:
    r: int = 16  # Hạng của ma trận cập nhật LoRA.
    alpha: int = 32  # Hệ số điều chỉnh độ lớn cập nhật LoRA (tỷ lệ alpha/r).
    dropout: float = 0.05  # Xác suất dropout trong adapter LoRA.
    targets: tuple[str, ...] = (  # Các lớp tuyến tính của LLM được gắn adapter LoRA.
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

# Data
BATCH_SIZE = 32  # Số mẫu trong mỗi batch.
NUM_WORKERS = 4  # Số tiến trình con tải và xử lý dữ liệu.
MAX_INSTRUCTION_LENGTH = 128  # Số token tối đa khi mã hóa chỉ dẫn hoặc văn bản đầu vào.
MAX_TEXT_LENGTH = 512  # Giới hạn số token của chuỗi văn bản.
SEED = 42  # Seed ngẫu nhiên giúp tái lập kết quả.
CROP_SCALE = (0.85, 1.0)  # Khoảng tỷ lệ diện tích ảnh được giữ khi cắt ngẫu nhiên.
CROP_RATIO = (0.9, 1.1)  # Khoảng tỷ lệ rộng/cao của vùng cắt ngẫu nhiên.
IMAGE_INTERPOLATION = "bicubic"  # Phương pháp nội suy khi thay đổi kích thước ảnh.
IMAGE_ANTIALIAS = True  # Bật khử răng cưa khi thay đổi kích thước ảnh.
HORIZONTAL_FLIP_PROBABILITY = 0.3  # Xác suất lật ngang ảnh.
COLOR_JITTER = (0.15, 0.15, 0.15, 0.03)  # Mức biến đổi độ sáng, tương phản, bão hòa và sắc độ.
COLOR_JITTER_PROBABILITY = 0.4  # Xác suất áp dụng biến đổi màu.
GRAYSCALE_PROBABILITY = 0.05  # Xác suất chuyển ảnh sang thang xám.
BLUR_KERNEL_SIZE = 5  # Kích thước kernel làm mờ Gaussian.
BLUR_SIGMA = (0.1, 1.5)  # Khoảng độ lệch chuẩn của bộ lọc Gaussian.
BLUR_PROBABILITY = 0.1  # Xác suất áp dụng làm mờ ảnh.
DATA_NORMALIZATION = None  # Cặp (mean, std) chuẩn hóa ảnh; None là không chuẩn hóa.
DATA_TRAIN = True  # Bật chế độ dữ liệu huấn luyện với tăng cường ảnh.
DATA_SHUFFLE = None  # Xáo trộn dữ liệu; None thì bật theo chế độ huấn luyện.
DROP_LAST = True  # Bỏ batch cuối chưa đủ mẫu khi huấn luyện.
PIN_MEMORY = None  # Ghim bộ nhớ để truyền dữ liệu lên GPU; None tự bật nếu có CUDA.
PERSISTENT_WORKERS = True  # Giữ các worker tải dữ liệu giữa các epoch.
PREFETCH_FACTOR = 2  # Số batch mỗi worker tải trước.
DATA_CACHE_DIR = Path(__file__).resolve().parents[1] / "cache" / "stage1"  # Thư mục lưu cache dữ liệu giai đoạn 1.
CACHE_CHUNK_SIZE = 10000  # Số mẫu xử lý trong mỗi khối khi tạo cache.
TOKENIZER_USE_FAST = True  # Dùng phiên bản tokenizer nhanh nếu có.
TOKENIZER_PADDING = True  # Thêm token đệm để các chuỗi trong batch có cùng độ dài.
TOKENIZER_TRUNCATION = True  # Cắt chuỗi vượt giới hạn token.

# MoCo
QUEUE_SIZE = 4096  # Số đặc trưng lưu trong hàng đợi MoCo.
MOMENTUM = 0.995  # Hệ số EMA cập nhật encoder momentum từ encoder đang học.
ITC_TEMPERATURE = 0.07  # Nhiệt độ điều chỉnh độ sắc của phân phối tương đồng ITC.
DISTILL_WEIGHT = 0.4  # Trọng số distillation tối đa; tăng tuyến tính từ 0 trong epoch đầu.
UPDATE_QUEUE = True  # Bật cập nhật hàng đợi đặc trưng MoCo.

# Stage 1 loss
ITC_LOSS_WEIGHT = 1.0  # Trọng số loss tương phản ảnh–văn bản.
ITM_LOSS_WEIGHT = 1.0  # Trọng số loss phân loại cặp ảnh–văn bản khớp nhau.
ITG_LOSS_WEIGHT = 1.0  # Trọng số loss sinh văn bản dựa trên ảnh.
ITG_IGNORE_INDEX = -100  # Giá trị nhãn được bỏ qua khi tính loss sinh văn bản.

# Stage 1 training
TRAIN_JSON_PATH = "/home/tranmanhduy/Workspace/chuyen_nganh/InA-Bridge/DatasetProject/BLIP3o/dataset_metadata.json"  # Đường dẫn JSON chứa metadata dữ liệu huấn luyện.
TRAIN_IMAGE_DIR = "/home/tranmanhduy/Workspace/chuyen_nganh/InA-Bridge/DatasetProject/BLIP3o/Image"  # Thư mục chứa ảnh huấn luyện.
TRAIN_OUTPUT_DIR = "outputs/stage1"  # Thư mục lưu kết quả và checkpoint giai đoạn 1.
TRAIN_EPOCHS = 10  # Số lượt huấn luyện qua toàn bộ tập dữ liệu.
LEARNING_RATE = 1e-4  # Tốc độ cập nhật trọng số của optimizer.
WEIGHT_DECAY = 0.05  # Hệ số giảm trọng số để hạn chế overfitting.
MAX_GRAD_NORM = 1.0  # Ngưỡng chuẩn gradient dùng để cắt gradient.
LOG_EVERY = 10  # Số bước huấn luyện giữa hai lần ghi log.

# Generation
SYSTEM_PROMPT = None  # Chỉ dẫn hệ thống cho LLM; None là không thêm chỉ dẫn.
ENABLE_THINKING = False  # Bật chế độ suy luận thinking trong mẫu hội thoại.
MAX_NEW_TOKENS = 256  # Số token mới tối đa được sinh.
TEMPERATURE = 0.0  # Độ ngẫu nhiên khi sinh; 0 chọn token có xác suất cao nhất.
TOP_P = 0.9  # Ngưỡng xác suất tích lũy để lấy mẫu nucleus sampling.
REPETITION_PENALTY = 1.05  # Hệ số phạt lặp token; lớn hơn 1 giảm lặp.

# Feature visualization
FEATURE_IMAGE_FOLDER = Path(__file__).resolve().parents[1] / "DatasetProject/BLIP3o/Image"  # Thư mục ảnh dùng để trực quan hóa đặc trưng.
FEATURE_MAX_LONG_SIDE = 640  # Giới hạn cạnh dài của ảnh khi trực quan hóa (pixel).
FEATURE_PCA_OUTPUT = "vision_feature_pca.png"  # Đường dẫn ảnh kết quả trực quan hóa đặc trưng bằng PCA.
