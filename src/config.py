# MODEL

# DATALOADER
IS_TRAINING = True  # Ví dụ: True; bật augmentation train và chọn cache train.sqlite.
JSON_PATH = "DatasetProject/BLIP3o/dataset_metadata.json"  # Ví dụ: "data/train.json"; manifest ảnh–text, đường dẫn tương đối từ gốc dự án hoặc tuyệt đối.
IMAGE_DIR = "DatasetProject/BLIP3o/Image"  # Ví dụ: "data/images"; thư mục chứa ảnh được tham chiếu trong manifest.
CACHE_DIR = "cache/dataloader_demo"  # Ví dụ: "cache/train"; thư mục SQLite cache, cần đổi hoặc xóa cache khi đổi manifest.
CACHE_CHUNK_SIZE = 1000  # Ví dụ: 1000; số bản ghi mỗi lần ghi metadata vào SQLite.

IMAGE_SIZE = 640  # Ví dụ: 640; chiều cao và rộng ảnh đầu ra, tính bằng pixel.
NORMALIZATION = None  # Ví dụ: ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)); (mean, std) theo RGB, None để bỏ chuẩn hóa; khi train dùng giá trị của vision encoder.
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

TOKENIZER_MODEL_ID = "bert-base-uncased"  # Ví dụ: "bert-base-uncased"; tên hoặc đường dẫn tokenizer, phải khớp với text embeddings của model.
TOKENIZER_USE_FAST = True  # Ví dụ: True; ưu tiên phiên bản tokenizer fast nếu có.
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
