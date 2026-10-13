"""Training ModelStage1 với FSDP2.

Chạy: torchrun --standalone --nproc-per-node=2 -m src.trainingph1.engine_fsdp

                      TRAINING_MODE   INIT_CHECKPOINT   RESUME_CHECKPOINT
Pretrain mới          "pretrain"      None              None
Resume pretrain       "pretrain"      None              checkpoint pretrain
Finetune mới          "finetune"      checkpoint nguồn  None
Resume finetune       "finetune"      None              checkpoint finetune

Pretrain mới khởi tạo run/step từ 0, vẫn dùng pretrained backbone theo model ID.
Finetune giữ architecture/tokenizer/weights, reset toàn bộ trạng thái training.
Resume dùng config đã lưu; DEVICE và tùy chọn logging lấy từ đây, tiếp tục trong run cũ.
"""

# RUN / CHECKPOINT
TRAINING_MODE = "pretrain"  # "pretrain" hoặc "finetune".
INIT_CHECKPOINT = None  # Run dir, last.json/best.json, snapshot dir hoặc legacy format-5 .pt.
RESUME_CHECKPOINT = None  # Chỉ checkpoint FSDP; không đặt cùng INIT_CHECKPOINT.
CHECKPOINT_DIR = "src/checkpoints/stage1"  # Shared filesystem giữa mọi rank.
MODEL_VERSION = "v1"  # Nhãn phiên bản model, không phải checkpoint format.
RUN_NAME = "pretrain_001"  # Run mới phải có tên chưa tồn tại.
# Layout: CHECKPOINT_DIR / MODEL_VERSION / TRAINING_MODE / RUN_NAME.
# last.json/best.json trỏ tới snapshot hoàn chỉnh; tự thu hồi snapshot bị thay thế.

# RUNTIME / FSDP2
# torchrun --standalone --nproc-per-node=2 -m src.trainingph1.engine_fsdp
# torchrun cấp RANK/LOCAL_RANK/WORLD_SIZE; không cấu hình lại các giá trị này ở đây.
# Có process group: shard toàn bộ online và EMA, kể cả frozen weights.
# Không có process group: chạy model thường để đối chiếu; CUDA dùng NCCL, CPU dùng Gloo.
SEED = 42  # Seed khởi tạo model chung; RNG training và loader được phân biệt theo rank.
DEVICE = "cuda"  # "cuda" hoặc "cpu"; khi torchrun, GPU được chọn theo LOCAL_RANK.
AMP_ENABLED = True  # False: FP32; True: BF16 trong engine_fsdp.
AMP_DTYPE = "bfloat16"  # engine_fsdp chưa hỗ trợ float16/GradScaler.
# FSDP giảm gradient bằng FP32; loss nhạy số học và queue cũng dùng FP32.

# MODEL / BACKBONE
BERT_MODEL_ID = "bert-base-uncased"
VISION_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
VISION_MOCK = True  # Giữ baseline hiện tại; False để dùng DINOv3 thật.
VISION_RETURN_LAYER = -1  # Layer vision lấy features; -1 là layer cuối.
NUM_QUERIES = 128
CROSS_ATTN_EVERY = 2  # Cross-attention mỗi N block Q-Former, bắt đầu từ block 0.
HIDDEN_DIM = 768  # Phải khớp hidden_size của BERT.

# TOKENIZER
TOKENIZER_MODEL_ID = BERT_MODEL_ID
TOKENIZER_USE_FAST = True
MAX_LENGTH = 128  # Độ dài sau tokenization, gồm cả special tokens.
TOKENIZER_PADDING = True  # True/"longest" hoặc "max_length"; batch phải có tensor chữ nhật.
TOKENIZER_TRUNCATION = True

# STAGE 1: DATA
DATASET_NAMESPACE = "blip3o"  # Tham gia image ID và tên cache, phải giống nhau giữa rank.
JSON_PATH_TRAIN = "DatasetProject/BLIP3o/dataset_metadata.json"
JSON_PATH_VAL = "DatasetProject/BLIP3o/dataset_metadata.json"  # Hiện dùng chung để thử; đặt manifest riêng khi đánh giá.
IMAGE_DIR = "DatasetProject/BLIP3o/Image"
# Đường dẫn tương đối tính từ root dự án; cũng chấp nhận đường dẫn tuyệt đối.
CACHE_DIR = "cache/dataloader_demo"  # Các rank dùng chung thư mục trên một node.
CACHE_CHUNK_SIZE = 1000  # Số records xử lý mỗi lần xây cache.
# Multilevel cache: <namespace>_train.sqlite và <namespace>_validation.sqlite;
# tự xây lại khi signature manifest/tokenizer/max_length thay đổi.

BATCH_SIZE = 8  # Số mẫu mỗi rank mỗi micro-batch, không chia cho WORLD_SIZE.
DROP_LAST = True  # Train: bỏ phần dư global batch. False: đệm kèm valid_mask để giữ mọi mẫu thật.
# Validation luôn giữ mọi mẫu thật và dùng valid_mask cho slots đệm.
SHUFFLE = True  # Chỉ áp dụng train; None cũng bật shuffle train. Validation luôn giữ thứ tự.
LENGTH_AWARE = True  # Gom train batch theo độ dài token; False: batching thông thường.
LENGTH_MEGA_BATCH_MULT = 32  # Sắp độ dài trong cửa sổ BATCH_SIZE * giá trị này.
LENGTH_JITTER = 4  # Nhiễu độ dài ±N khi shuffle; 0 tắt nhiễu.
NUM_WORKERS = 0  # Số worker mỗi rank; 0 đọc dữ liệu trong process training.
PIN_MEMORY = False  # None: tự bật nếu CUDA có sẵn; True/False: đặt trực tiếp.
PERSISTENT_WORKERS = False  # Engine chỉ giữ workers validation; train tạo lại mỗi epoch.
PREFETCH_FACTOR = 2  # Số batch nạp trước mỗi worker; chỉ dùng khi NUM_WORKERS > 0.

# IMAGE TRANSFORMS
IMAGE_SIZE = 640
NORMALIZATION = (
    (0.485, 0.456, 0.406),
    (0.229, 0.224, 0.225)
)

# Geometric augmentation
CROP_SCALE = (0.98, 1.0)
CROP_RATIO = (0.5, 2.0)
INTERPOLATION = "bicubic"
ANTIALIAS = True
HORIZONTAL_FLIP_PROBABILITY = 0.0

# Color augmentation
COLOR_JITTER = (0.05, 0.05, 0.03, 0.0)
COLOR_JITTER_PROBABILITY = 0.2

# Grayscale
GRAYSCALE_PROBABILITY = 0.0

# Gaussian blur
BLUR_KERNEL_SIZE = 5
BLUR_SIGMA = (0.1, 0.5)
BLUR_PROBABILITY = 0.0

# ITC / EMA / QUEUE
ITC_DIM = 256
NORMALIZE_EPS = 1e-12
MOMENTUM = 0.995  # teacher = m * teacher + (1 - m) * online, sau step thành công.
TEMPERATURE = 0.07  # Phải hữu hạn và > 0.
PSEUDO_WEIGHT = 0.4  # Trọng số teacher soft targets trong [0, 1].
PSEUDO_WARMUP_EPOCHS = 1  # Tăng tuyến tính đến PSEUDO_WEIGHT; 0 dùng ngay giá trị đích.
QUEUE_SIZE = 4096  # Capacity K đầy đủ trên MỖI rank; không chia/nhân K theo WORLD_SIZE.
# Queue replicated đồng nhất; ITC dùng global keys micro-batch hiện tại + queue cũ.
# Chỉ enqueue mẫu hợp lệ sau step thành công; queue bất biến trong accumulation group.
ITC_WEIGHT = 1.0
ITM_WEIGHT = 1.0  # Negatives lấy trong batch local; không có negative hợp lệ thì ITM count = 0.
ITG_WEIGHT = 1.0

# OPTIMIZER / SCHEDULER
EPOCHS = 20
WARMUP_EPOCHS = 1  # 0 <= WARMUP_EPOCHS < EPOCHS; scheduler tiến theo optimizer step thành công.
ACCUMULATION_STEPS = 1
# Batch global danh nghĩa = BATCH_SIZE * WORLD_SIZE * ACCUMULATION_STEPS.
# Loss chuẩn hóa bằng số samples/pairs/tokens hợp lệ thực tế của cả group trên mọi rank.
LR = 1e-4
MIN_LR = 5e-6  # 0 <= MIN_LR <= LR; cosine decay sau warmup.
WEIGHT_DECAY = 0.05
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
MAX_GRAD_NORM = 1.0  # Clip theo norm toàn bộ gradient trên mọi shard; None để tắt.

# SAVE / VALIDATION / LOGGING
# Các chu kỳ tính theo optimizer step thành công, không tính micro-batch/step skip.
SAVE_EVERY_STEPS = 1000  # 0 tắt lưu định kỳ; luôn lưu last khi training kết thúc.
VAL_EVERY_STEPS = 1000  # 0 tắt; nếu bật, đánh giá định kỳ/cuối run và lưu best loss hữu hạn.
LOG_EVERY_STEPS = 10  # 0 tắt log loss định kỳ; rank 0 vẫn báo skip cuối epoch.
LOG_LEVEL = "INFO"  # DEBUG/INFO/WARNING/ERROR/CRITICAL; file riêng logs/rank-N.log.
TENSORBOARD_ENABLED = True  # Chỉ rank 0 ghi vào <run>/tensorboard; resume xóa phần event vượt checkpoint.
TENSORBOARD_FLUSH_SECS = 30
TENSORBOARD_DIAGNOSTICS_EVERY_STEPS = 50  # Bước 1 và mỗi N step; validation đo toàn bộ. 0 tắt diagnostics.
TENSORBOARD_TEXT_SAMPLES = 3  # Ảnh/caption validation của rank 0; prediction là teacher-forced. 0 tắt ví dụ.
