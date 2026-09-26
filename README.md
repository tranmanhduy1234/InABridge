# InA-Bridge

InA-Bridge xây dựng mô hình thị giác–ngôn ngữ (VLM) với **DINOv3 + Q-Former khởi tạo từ BERT**. Giai đoạn 1 học biểu diễn ảnh–văn bản qua ba mục tiêu: image-text contrastive (ITC), image-text matching (ITM) và image-grounded text generation (ITG).

Repository hiện có model, xử lý dữ liệu, ba loss ITC/ITM/ITG, quản lý checkpoint và khung training. **Lịch validation, logging và lưu checkpoint tự động trong vòng lặp chưa được triển khai.** README mô tả mã hiện có; các phần chưa triển khai được đánh dấu riêng.

## 1. Cấu trúc thư mục

```text
InA-Bridge/
├── README.md
├── .gitignore
├── src/
│   ├── config.py                  # Cấu hình model, dữ liệu và training
│   ├── vision.py                  # ImageEncoder: DINOv3 đóng băng
│   ├── qformer.py                 # Query tokens, attention masks và Q-Former blocks
│   ├── projector.py               # QwenProjector, chưa nối vào pipeline
│   ├── data/
│   │   └── dataloaderph1.py        # JSON → SQLite cache → ảnh/text → batch tensor
│   ├── losses/
│   │   └── lossph1.py             # Stage1Criterion: hiện có loss ITC
│   ├── queue/
│   │   ├── ema.py                 # Momentum model, cập nhật bằng EMA
│   │   └── moco.py                # Queue vòng lưu image/text features
│   ├── trainingph1/
│   │   ├── model.py               # ModelStage1 và các nhánh ITC/ITM/ITG
│   │   ├── engine.py              # Các hàm khởi tạo; vòng lặp train còn thiếu
│   │   └── finetune.py            # File trống, chưa triển khai
│   ├── utils/
│   │   ├── checkpoint.py          # Tạo run, lưu, nạp model và resume training
│   │   ├── seed.py                # Seed Python, NumPy, PyTorch và worker
│   │   ├── artifacts.py           # File trống
│   │   ├── logging_setup.py       # File trống
│   │   └── tb_logger.py           # File trống
│   └── checkpoints/
│       └── stage1/
│           └── manager.json       # Tài liệu quy ước checkpoint format 5
├── tests/
│   └── test_checkpoint.py         # Kiểm thử checkpoint bằng model nhỏ trên CPU
├── DatasetProject/                # Dữ liệu cục bộ, không đưa vào Git
├── cache/                         # SQLite cache cục bộ, không đưa vào Git
└── documents/                     # Tài liệu nghiên cứu cục bộ, không đưa vào Git
```

Các thư mục run bên trong `src/checkpoints/stage1/` được tạo khi gọi `create_run()`. `outputs/` là đường dẫn cấu hình cho kết quả/log, chưa có tiện ích tự tạo và ghi log.

## 2. Kiến trúc và luồng tensor

```text
Ảnh [B, 3, H, W]
    │
    ▼
DINOv3 — đóng băng
    │ patch features [B, N, C]
    │
    ├───────────────────────────┐
    │                           │
    ▼                           ▼
Query tokens học được      Cross-attention
    │                           │
    └────────── Q-Former ────────┘ ◄── BERT embeddings ◄── Token IDs [B, T]
                    │                    đóng băng
                    ├── ITC: image/text embeddings
                    ├── ITM: logits khớp ảnh–text
                    └── ITG: logits token kế tiếp
```

Ký hiệu: `B` là batch size, `N` số patch ảnh, `C` chiều đặc trưng vision, `Q` số query, `T` số token text, `D` chiều embedding ITC và `V` kích thước từ vựng.

### `src/vision.py` — ImageEncoder

- Nạp `DINOv3ViTModel` theo model ID; hoặc dựng từ `model_config` khi khôi phục checkpoint offline.
- Đóng băng trọng số, giữ vision model ở chế độ `eval()` và chạy forward trong `no_grad`.
- Lấy hidden states tại `return_layer`, loại CLS token và các register token, trả patch features `[B, N, C]`.
- Cung cấp `hidden_size`, `device`, `count_parameters()` và `compute_grid_shape()`. Hàm tính grid kiểm tra kích thước ảnh dương và chia hết cho patch size.

### `src/qformer.py` — QFormer

`QFormer` dùng các lớp encoder của BERT, bổ sung query tokens học được. Mỗi `QFormerBlock` gồm self-attention, các nhánh feed-forward riêng cho query/text và cross-attention tùy vị trí block.

`CROSS_ATTN_EVERY` quyết định khoảng cách giữa các block có cross-attention, bắt đầu từ block đầu tiên. Cross-attention cho query truy cập patch features của ảnh. `HIDDEN_DIM` phải khớp `BERT.hidden_size`.

| Objective | Quy tắc attention |
| --- | --- |
| `itc` | Query và text self-attention riêng; query nhận thông tin ảnh qua cross-attention |
| `itm` | Query và text tương tác hai chiều |
| `itg` | Query không nhìn text; text nhìn query và các token text tới vị trí hiện tại |

`generate_mask_qformer()` kết hợp cấu trúc trên với padding mask; `True` biểu thị vị trí hợp lệ. `to_additive_mask()` chuyển sang dạng mask cho BERT attention.

Q-Former trả dictionary chứa `query_output`, `text_output` và `hidden_states` đã ghép query/text.

### `src/trainingph1/model.py` — ModelStage1

`ModelStage1` kết hợp vision encoder, BERT embeddings, Q-Former và các head:

| Objective | Thành phần | Kết quả |
| --- | --- | --- |
| `itc` | Chiếu từng query và token text đầu tiên, rồi chuẩn hóa L2 | `image_features`: `[B, Q, D]`; `text_features`: `[B, D]` |
| `itm` | Linear head 1 logit trên mỗi query, lấy trung bình logits theo query | `itm_logits`: `[B]` |
| `itg` | LM head dùng chung trọng số với word embeddings | `itg_logits`: `[B, T-1, V]` |

Vision encoder, BERT embeddings và LM head được đóng băng. Q-Former, các projection ITC và ITM head có thể huấn luyện. Đầu ra ITG bỏ vị trí cuối; engine cần chuẩn bị labels dịch một token và mask padding khi tính loss.

`forward()` nhận **image features đã encode**, không nhận trực tiếp ảnh:

```python
image_features = model.encode_image(batch["images"])
outputs = model(
    image_features=image_features,
    input_ids=batch["input_ids"],
    attn_mask=batch["attention_mask"],
    objective="itc",
)
```

### `src/projector.py` — QwenProjector

Projector ánh xạ biểu diễn Q-Former sang chiều của LLM qua RMSNorm, hai linear layer, SiLU và dropout. Module này chưa được nối vào `ModelStage1` hoặc engine; chưa có pipeline Qwen/LLM trong repository.

Trong implementation hiện tại, RMSNorm dùng `hidden_dim` trực tiếp trên đầu vào; khi khởi tạo cần để chiều đó khớp chiều tensor Q-Former.

## 3. Dữ liệu và dataloader

### Định dạng manifest

`VLMDatasetStage1` nhận một JSON array:

```json
[
  {"image_name": "image_001.jpg", "text": "A dog is running on the grass."},
  {"image_name": "image_002.jpg", "text": "Two people are sitting at a table."}
]
```

`image_name` được nối với `IMAGE_DIR`. Đường dẫn manifest, ảnh và cache trong engine có thể là đường dẫn tuyệt đối hoặc tương đối từ gốc dự án.

### Các thành phần trong `src/data/dataloaderph1.py`

| Thành phần | Trách nhiệm |
| --- | --- |
| `build_cache()` | Đọc JSON theo luồng bằng `ijson`, ghi metadata theo chunk vào SQLite |
| `VLMDatasetStage1` | Đọc một bản ghi từ cache, mở ảnh RGB, áp dụng transform, trả `(image, text, image_id)` |
| `build_transform()` | Dựng augmentation train hoặc resize validation, chuyển tensor và normalization tùy cấu hình |
| `VLMDataCollator` | Tokenize một batch text, padding/truncation và ghép ảnh thành tensor |
| `build_dataloader()` | Cấu hình batch, shuffle, worker, pin memory, prefetch và seed worker |
| `main()` | Demo đọc batch và hiển thị ảnh/caption |

Training dùng random resized crop, horizontal flip, color jitter, grayscale và Gaussian blur theo xác suất cấu hình. Validation chỉ resize trước khi chuyển tensor và normalization. `NORMALIZATION=None` nghĩa là bỏ chuẩn hóa; code không tự lấy mean/std từ vision encoder.

Collator trả:

| Key | Shape / kiểu |
| --- | --- |
| `images` | `[B, 3, IMAGE_SIZE, IMAGE_SIZE]` |
| `image_ids` | `[B]`, `torch.long`; ID ảnh ổn định |
| `input_ids` | `[B, T]`, token IDs |
| `attention_mask` | `[B, T]`, boolean; `True` là token hợp lệ |

`DATASET_NAMESPACE` trong `config.py` định danh nguồn dữ liệu. Image ID dùng BLAKE2b 8 byte của `namespace + "\0" + relative_path`, chuyển thành signed int64. Đường dẫn được resolve rồi lấy tương đối với `IMAGE_DIR`, dùng dấu `/`; ảnh phải nằm trong thư mục gốc này. Cùng namespace và đường dẫn tương đối sẽ giữ ID khi chuyển dataset sang máy/thư mục khác. Dùng cùng namespace cho train/validation nếu cùng nguồn ảnh; đổi namespace cho dataset khác. ID nhận diện đường dẫn, không nhận diện ảnh trùng nội dung ở hai đường dẫn khác nhau. Hash 64-bit có xác suất collision rất nhỏ, không bảo đảm duy nhất tuyệt đối.

Cache được đặt tên `train.sqlite` và `validation.sqlite`, chỉ chứa metadata, không chứa ảnh đã decode. Cache đã tồn tại sẽ được dùng lại; khi đổi manifest cần dùng thư mục cache khác hoặc chủ động xóa cache cũ.

`get_dataloader()` trong engine luôn tạo cả train và validation loader. Validation không shuffle và giữ batch cuối. `PERSISTENT_WORKERS` và `PREFETCH_FACTOR` chỉ có tác dụng khi `NUM_WORKERS > 0`.

Hiện `JSON_PATH_TRAIN` và `JSON_PATH_VAL` trỏ cùng một manifest demo; code **không tự chia tập**. Cần cấu hình hai manifest riêng để đánh giá trên tập validation độc lập.

## 4. Loss, momentum model và queue

### `src/losses/lossph1.py` — Stage1Criterion

Criterion không giữ trọng số loss; engine đọc `ITC_WEIGHT`, `ITM_WEIGHT`, `ITG_WEIGHT` từ settings; `pseudo_weight` được truyền vào mỗi lần gọi `get_itc_loss()`. `get_itc_loss()` thực hiện:

1. Ghép momentum features của batch hiện tại với queue để tạo image/text banks.
2. Tính similarity ảnh–text bằng tích vô hướng, lấy giá trị lớn nhất trên các query ảnh và chia cho temperature.
3. So khớp `image_ids` với ID của batch và queue, chia đều hard target cho mọi phần tử cùng ID; tạo soft targets từ momentum model.
4. Trộn targets theo `pseudo_weight`, tính soft-target cross-entropy cho hai chiều ảnh→text và text→ảnh.

```text
target = (1 - pseudo_weight) × positive_target
       + pseudo_weight × softmax(momentum_logits)

loss_itc = (loss_i2t + loss_t2i) / 2
```

`get_itc_loss()` nhận thêm `image_ids` của batch và `queue_ids` theo đúng thứ tự queue. Momentum features, queue và targets không giữ gradient. Các phép tính similarity/loss sử dụng float32. Hàm trả `loss_itc`, `loss_i2t`, `loss_t2i`, `logits_i2t`, `logits_t2i`.

Hai hàm còn lại chỉ tính loss từ logits và labels do engine chuẩn bị:

| Hàm | Input | Kết quả |
| --- | --- | --- |
| `get_itm_loss(itm_logits, labels)` | Logits `[N]`, labels `[N]` được chuyển sang float: `0` không khớp, `1` khớp | `{"loss_itm": ...}`: BCE with logits trung bình trên các cặp |
| `get_itg_loss(itg_logits, labels)` | Logits `[B, T-1, V]`, labels long `[B, T-1]` đã dịch một token; `-100` là vị trí bỏ qua | `{"loss_itg": ...}`: cross-entropy trung bình trên token hợp lệ |

Engine thực hiện hard-negative sampling và gán labels cho ITM. Với ITG, engine chuẩn bị labels như sau; hàm loss không dịch token lần nữa:

```python
labels = input_ids[:, 1:].masked_fill(~attn_mask[:, 1:].bool(), -100)
loss_itg = criterion.get_itg_loss(itg_logits, labels)["loss_itg"]
```

Dùng attention mask để bỏ padding, giữ EOS hợp lệ ngay cả khi PAD và EOS dùng chung token ID. Nếu không có token hợp lệ, ITG trả loss bằng 0 và vẫn backward được. Các hàm criterion trả loss chưa nhân trọng số; `hepler_compute_loss()` trong engine tổng hợp theo `ITC_WEIGHT`, `ITM_WEIGHT`, `ITG_WEIGHT` trong settings. ITM lấy một negative image và một negative text cho mỗi positive từ similarity online–online trong batch; mask toàn bộ cặp cùng image ID trước softmax và sampling, không lấy candidates từ queue. Nếu batch chỉ có một image ID duy nhất, bỏ qua ITM (loss bằng 0), kể cả khi có nhiều caption.

### `src/queue/ema.py` — EMA

`EMA(model.itc_encoder, momentum)` chỉ deepcopy Q-Former và hai projection head ITC (`query_proj`, `text_proj`), đóng băng bản sao và giữ ở chế độ eval. Vision encoder, text embeddings, ITM head và LM head không nằm trong EMA. Teacher dùng lại vision features và text embeddings đã freeze của online model. `ema.update(model.itc_encoder)` cập nhật tham số:

```text
momentum_parameter = m × momentum_parameter + (1 - m) × online_parameter
```

Buffers được copy từ online ITC encoder. Engine gọi `update()` sau optimizer step thành công. State EMA cũ chứa toàn model không nạp trực tiếp vào EMA mới; chưa triển khai chuyển đổi.

### `src/queue/moco.py` — MoCoQueue

Queue lưu feature ảnh `[capacity, Q, D]`, text `[capacity, D]`, `image_ids` `[capacity]` kiểu int64, cùng con trỏ ghi và số phần tử hợp lệ dưới dạng buffers.

- `enqueue(images, texts, image_ids)` ghi features đã detach và ID đồng bộ vào queue vòng.
- `get()` trả `(images, texts, image_ids)` hợp lệ theo thứ tự từ cũ đến mới.
- Khi batch lớn hơn capacity, chỉ giữ các phần tử mới nhất.
- Trạng thái queue có thể lưu/khôi phục bằng `state_dict()`.

## 5. Engine và cấu hình training

### API hiện có trong `src/trainingph1/engine.py`

| Hàm | Hành vi hiện tại |
| --- | --- |
| `get_dataloader(tokenizer=None, settings=None)` | Tạo train/validation loaders; `settings` ghi đè config và yêu cầu truyền tokenizer cùng |
| `get_model(settings=None)` | Tạo online `ModelStage1` và `EMA` theo config; chưa chuyển device |
| `get_criterion()` | Tạo criterion không giữ trọng số loss |
| `get_pseudo_weight(epoch_progress, settings=None)` | Tính pseudo weight theo tiến độ epoch để truyền vào ITC loss |
| `get_optimizer(model, settings=None)` | AdamW, một learning rate chung cho mọi tham số `requires_grad=True` |
| `get_scheduler(...)` | Linear warmup rồi cosine decay, dùng `LambdaLR` |
| `prepare_training()` | Đọc config, chuẩn bị pretrain/fine-tune/resume; trả model, loaders, training components, run directory và tiến độ |
| `validate(state, pseudo_weight=None)` | Trả dict `loss_itc`, `loss_itm`, `loss_itg`, `loss`; không cập nhật training state |
| `hepler_compute_loss(state, batch, pseudo_weight)` | Trả `(losses, momentum_features)`; tính ba loss và tổng có trọng số, không cập nhật EMA/queue/optimizer |
| `train_one_epoch(state, epoch)` | Training với autocast, accumulation, clipping, scheduler và EMA/queue; trả loss trung bình theo mẫu/cặp/token như validation |
| `run_training()` | Gọi `prepare_training()` một lần, chạy các epoch còn lại và trả state |

`src/trainingph1/finetune.py` hiện là file trống. Dùng `run_training()` để điều phối khung training; chưa có CLI riêng.

### Các nhóm trong `src/config.py`

| Nhóm | Tham số chính |
| --- | --- |
| Kiến trúc | `BERT_MODEL_ID`, `VISION_MODEL_ID`, `VISION_RETURN_LAYER`, `NUM_QUERIES`, `CROSS_ATTN_EVERY`, `HIDDEN_DIM`, `ITC_DIM`, `NORMALIZE_EPS` |
| Dữ liệu và cache | `DATASET_NAMESPACE`, `JSON_PATH_TRAIN`, `JSON_PATH_VAL`, `IMAGE_DIR`, `CACHE_DIR`, `CACHE_CHUNK_SIZE` |
| Transform | `IMAGE_SIZE`, `NORMALIZATION`, `CROP_SCALE`, `CROP_RATIO`, `INTERPOLATION`, `ANTIALIAS`, các xác suất augmentation |
| Tokenizer | `TOKENIZER_MODEL_ID`, `TOKENIZER_USE_FAST`, `MAX_LENGTH`, `TOKENIZER_PADDING`, `TOKENIZER_TRUNCATION` |
| Loader | `BATCH_SIZE`, `NUM_WORKERS`, `DROP_LAST`, `SHUFFLE`, `PIN_MEMORY`, `PERSISTENT_WORKERS`, `PREFETCH_FACTOR` |
| Optimizer/scheduler | `LR`, `MIN_LR`, `WEIGHT_DECAY`, `ADAM_BETAS`, `ADAM_EPS`, `EPOCHS`, `WARMUP_EPOCHS` |
| Training | `SEED`, `DEVICE`, `ACCUMULATION_STEPS`, `AMP_ENABLED`, `AMP_DTYPE`, `MAX_GRAD_NORM` |
| Loss và momentum | `ITC_WEIGHT`, `ITM_WEIGHT`, `ITG_WEIGHT`, `PSEUDO_WEIGHT`, `PSEUDO_WARMUP_EPOCHS`, `TEMPERATURE`, `MOMENTUM`, `QUEUE_SIZE` |
| Run/checkpoint/log | `CHECKPOINT_DIR`, `MODEL_VERSION`, `RUN_NAME`, `INIT_CHECKPOINT`, `RESUME_CHECKPOINT`, `OUTPUT_DIR`, `SAVE_EVERY_STEPS`, `VAL_EVERY_STEPS`, `LOG_EVERY_STEPS` |
| Demo | `IS_TRAINING`, `DEMO_NUM_IMAGES`, `DEMO_FIGURE_SIZE`, `DEMO_TITLE_FONT_SIZE` |

Mặc định model dùng BERT base uncased, DINOv3 ViT-L/16, 128 query, hidden size 768 và ITC dimension 256. Các giá trị cụ thể và chú thích nằm trong `config.py`.

### Quy ước epoch và step

`global_step` là số lần optimizer cập nhật thành công sau accumulation, không phải số micro-batch. Logging, save và validation được cấu hình theo step; tổng thời gian training, LR warmup và pseudo warmup vẫn tính theo epoch.

Khi gọi `get_scheduler(optimizer, batches_per_epoch=len(train_loader))`:

```text
steps_per_epoch = ceil(batches_per_epoch / ACCUMULATION_STEPS)
warmup_steps    = WARMUP_EPOCHS × steps_per_epoch
total_steps     = EPOCHS × steps_per_epoch
```

Cách tính này giả định engine thực hiện optimizer step cho nhóm accumulation cuối epoch dù thiếu micro-batch. Scheduler tăng LR từ 0 lên LR ban đầu, rồi giảm cosine về `MIN_LR`. Có thể truyền trực tiếp `warmup_steps`, `total_steps` và `min_lr_ratio` thay cho cách tính từ config.

Pseudo weight tăng tuyến tính từ 0 đến `PSEUDO_WEIGHT` trong `PSEUDO_WARMUP_EPOCHS`, sau đó giữ nguyên. Đặt thời gian warmup bằng 0 để dùng mức đích ngay. Criterion được tạo một lần. Trong mỗi batch, tính `epoch_progress = epoch + batch_idx / len(train_loader)`, gọi `get_pseudo_weight(epoch_progress, state.settings)` và truyền kết quả qua đối số `pseudo_weight` của `get_itc_loss()`. Khi resume, dùng epoch và batch index thực tế đã khôi phục.

`prepare_training()` áp dụng `DEVICE`, dựng scaler khi bật AMP float16 và khôi phục checkpoint khi resume. `train_one_epoch()` thực thi autocast, gradient clipping, accumulation và cập nhật pseudo weight theo batch. Scheduler, EMA, queue và global step chỉ cập nhật khi optimizer step thành công; momentum keys của các micro-batch được enqueue sau bước đó. Trong mỗi nhóm accumulation (kể cả nhóm cuối), ITC chuẩn hóa theo tổng số mẫu, ITM theo số cặp có negative, ITG theo số token hợp lệ. Engine gom các batch input của nhóm trước forward để biết mẫu số, không giữ đồ thị gradient qua các micro-batch. Candidates ITC/ITM vẫn theo từng micro-batch, không mở rộng thành một batch contrastive lớn. Gradient NaN/Inf hoặc norm clipping không hữu hạn sẽ bỏ qua optimizer/scheduler/EMA/queue/global step, rồi xóa gradient; quy tắc áp dụng cả FP32 và bfloat16 khi không có GradScaler. Lịch log/save/validate chưa được nối vào vòng lặp. Các factory model/loss/optimizer/scheduler nhận `settings` tùy chọn, mặc định dùng `config.py`.

## 6. Quản lý checkpoint

Cấu hình người dùng nằm trong **`src/config.py`**. `run.json` là snapshot tự sinh, không phải file cần chỉnh tay.

Mỗi phiên bản kiến trúc chứa các run riêng; mỗi run chỉ giữ checkpoint gần nhất và checkpoint tốt nhất nếu có:

```text
src/checkpoints/stage1/
├── manager.json
└── v1/
    ├── pretrain_001/
    │   ├── run.json
    │   ├── last.pt
    │   └── best.pt
    └── finetune_caption_001/
        ├── run.json
        └── last.pt
```

- `MODEL_VERSION`: nhãn kiến trúc, ví dụ `v1`. Khi đổi implementation, dùng nhãn mới và Git tag/commit tương ứng để giữ lại code cũ. Loader dùng class `ModelStage1` hiện tại, không tự checkout hay lựa chọn implementation theo nhãn.
- `RUN_NAME`: lần chạy, ví dụ `pretrain_001`; đổi dataset, learning rate hoặc seed để thử nghiệm thì tạo run mới.
- `FORMAT_VERSION`: định dạng checkpoint nội bộ, không phải phiên bản mô hình; không cần sửa khi tạo run mới.

### Vai trò từng file

| File | Nội dung |
| --- | --- |
| `manager.json` | Mô tả quy ước format 5; không phải registry bắt buộc phân loại các biến config |
| `run.json` | Run ID, kiến trúc thực tế và config BERT/DINOv3, fast tokenizer, danh sách tham số trainable, snapshot config và checkpoint nguồn nếu có |
| `last.pt` | Model, optimizer, scheduler, các thành phần tùy chọn, tiến độ training và trạng thái RNG gần nhất |
| `best.pt` | Cùng định dạng với `last.pt`, được lưu khi engine chủ động yêu cầu |

`run.json` được tạo một lần. Checkpoint chứa digest của metadata này để phát hiện ghép nhầm file của run khác; không sửa `run.json` sau khi đã lưu checkpoint. Khi chuyển sang máy khác, sao chép cả thư mục run.

### API trong `src/utils/checkpoint.py`

| Hàm | Mục đích |
| --- | --- |
| `create_run(model, tokenizer, *, run_dir=None, settings=None, parent_checkpoint=None)` | Tạo thư mục mới và `run.json`; mặc định dùng `CHECKPOINT_DIR/MODEL_VERSION/RUN_NAME`; báo lỗi nếu thư mục đã tồn tại |
| `save_checkpoint(run_dir, model, *, optimizer, scheduler, global_step, epoch, next_batch, ...)` | Lưu đầy đủ trạng thái; mặc định cập nhật `last.pt`, hoặc `filename="best.pt"` |
| `load_pretrained(path)` | Dựng lại model/tokenizer offline; trả `(model, tokenizer, saved_config)` |
| `load_checkpoint(path, model, *, optimizer, scheduler, ...)` | Khôi phục training state và RNG vào các đối tượng đã tạo; trả tiến độ để engine tiếp tục |

Các đối số tùy chọn của save/resume là `scaler`, `ema`, `queue`. Save còn nhận `data_state` cho sampler/generator hoặc thông tin cần thiết để tiếp tục duyệt dữ liệu.

`load_pretrained()` khôi phục cả trạng thái đóng băng tham số, trả model trên CPU ở chế độ eval. Caller chuyển device và gọi `.train()` khi cần. Hàm này không khôi phục optimizer, scheduler hay global step.

### Chọn chế độ trong `src/config.py`

**Pretrain mới** — nạp trọng số BERT/DINO ban đầu, tạo toàn bộ training state mới:

```python
MODEL_VERSION = "v1"
RUN_NAME = "pretrain_001"
INIT_CHECKPOINT = None
RESUME_CHECKPOINT = None
```

**Fine-tune mới** — nạp model/tokenizer từ run nguồn, dùng hyperparameters và dữ liệu trong config hiện tại, tạo optimizer/EMA/queue và tiến độ mới:

```python
MODEL_VERSION = "v1"
RUN_NAME = "finetune_caption_001"
INIT_CHECKPOINT = "src/checkpoints/stage1/v1/pretrain_001/best.pt"
RESUME_CHECKPOINT = None
```

Dùng `last.pt` nếu run nguồn chưa có `best.pt`. Kiến trúc và trạng thái đóng băng tham số kế thừa checkpoint nguồn; các biến kiến trúc trong config hiện tại không thay đổi model đã load. Queue lấy số query và chiều feature từ model thực tế.

**Resume pretrain hoặc fine-tune** — dùng cấu hình đã lưu, khôi phục training state và lưu tiếp vào thư mục cũ:

```python
INIT_CHECKPOINT = None
RESUME_CHECKPOINT = "src/checkpoints/stage1/v1/finetune_caption_001/last.pt"
DEVICE = "cuda"
```

Khi resume, `DEVICE` lấy từ config hiện tại; các thiết lập đã lưu khác như LR, epoch, scheduler, AMP, dữ liệu và queue được lấy từ run cũ. `MODEL_VERSION` và `RUN_NAME` hiện tại không chuyển run sang thư mục khác. Muốn đổi thiết lập training để thử nghiệm, dùng fine-tune với run mới. Không đặt đồng thời `INIT_CHECKPOINT` và `RESUME_CHECKPOINT`.

### Chuẩn bị training và lưu checkpoint

Sau khi chỉnh `config.py`, cả ba chế độ đều gọi cùng một hàm:

```python
from src.trainingph1.engine import prepare_training

state = prepare_training()
model = state.model
train_loader, val_loader = state.train_loader, state.val_loader
print(state.run_dir, state.progress)
```

`state` còn chứa `tokenizer`, `settings`, `criterion`, `optimizer`, `scheduler`, `ema`, `queue`, `scaler`. `scaler` là `None` khi AMP tắt hoặc dùng bfloat16. Đây là bước **chuẩn bị**, chưa chạy training. Gọi `train_one_epoch(state, epoch)` để chạy một epoch; hoặc gọi trực tiếp `run_training()` để tự chuẩn bị và chạy toàn bộ các epoch còn lại. Chỉ gọi `prepare_training()` một lần cho mỗi lần khởi động; run mới trùng thư mục sẽ báo lỗi.

Trong vòng lặp, dùng `state.settings` làm cấu hình hiệu lực, đặc biệt khi resume. Cập nhật `state.progress` gồm `global_step`, `epoch`, `next_batch`, `data_state` sau bước training hoàn tất, rồi lưu:

```python
from src.utils.checkpoint import save_checkpoint

save_checkpoint(
    state.run_dir, state.model,
    optimizer=state.optimizer, scheduler=state.scheduler,
    ema=state.ema, queue=state.queue, scaler=state.scaler,
    **state.progress,
)
```

Thêm `filename="best.pt"` khi validation cải thiện. `last.pt` được thay thế sau mỗi lần lưu thành công; không sinh thêm file theo step. Khi chuyển máy, copy cả thư mục run, gồm `run.json`. Với run được tạo từ API cấp thấp bằng training components khác, tiếp tục dùng API cấp thấp để dựng đúng các components đó khi resume.

Chỉ load model cho inference hoặc sử dụng riêng:

```python
from src.utils.checkpoint import load_pretrained

model, tokenizer, saved_config = load_pretrained(
    "src/checkpoints/stage1/v1/pretrain_001/best.pt"
)
model = model.to("cuda").eval()
```

Để chạy khung training từ cấu hình hiện tại:

```python
from src.trainingph1.engine import run_training

state = run_training()
```

Hàm tự lưu `last.pt` theo `SAVE_EVERY_STEPS`, gọi validation theo `VAL_EVERY_STEPS` và in loss training ra stdout theo `LOG_EVERY_STEPS`. Các chu kỳ tính theo optimizer step thành công; đặt `0` để tắt từng lịch. Khi validation loss hữu hạn và thấp hơn mức tốt nhất, engine lưu `best.pt`; giá trị tốt nhất và step validation được giữ trong `data_state` để resume. Mỗi lần validation cũng cập nhật `last.pt`. Cuối training luôn lưu `last.pt` và chạy validation nếu bật, trừ khi step đó đã được đánh giá. Log training là loss trung bình từ đầu phần epoch đang chạy. Không gọi thêm `prepare_training()` trước `run_training()` vì hàm đã tự chuẩn bị run.

Validation có thể gọi riêng:

```python
from src.trainingph1.engine import validate

losses = validate(state)
```

ITC lấy trung bình theo số mẫu, ITM theo số cặp có negative hợp lệ, ITG theo số token không padding. `loss` là tổng ba giá trị trung bình đã nhân trọng số. Nhánh không có phần tử hợp lệ trả 0; loader rỗng báo lỗi. Mặc định pseudo weight cố định bằng `state.settings["PSEUDO_WEIGHT"]`, có thể truyền giá trị khác qua `pseudo_weight`. Validation dùng queue hiện tại để đọc và dùng hard-negative sampling trong batch; không enqueue hoặc cập nhật EMA/optimizer. Hàm tắt gradient, bật eval, khôi phục chế độ model và Torch RNG sau khi chạy. Kết quả ITC vẫn phụ thuộc queue hiện tại.

### Resume và điều kiện lưu

Checkpoint khôi phục optimizer/scheduler, AMP scaler nếu có, EMA cùng momentum, queue, RNG Python/NumPy/PyTorch CPU/CUDA. `load_checkpoint()` trả `global_step`, `epoch`, `next_batch`, `data_state`.

- Tạo optimizer sau khi đã chuyển model sang device. Khi resume, dựng đúng các training components đã sử dụng rồi mới nạp state.
- `epoch` và `next_batch` xác định batch tiếp theo cần xử lý. Engine chịu trách nhiệm khôi phục sampler, vị trí và thứ tự dữ liệu.
- Chỉ lưu ở ranh giới optimizer step: đã cập nhật scheduler/EMA/queue và `zero_grad`, không còn gradient tích lũy chưa áp dụng.
- Không gọi lại seed sau khi `load_checkpoint()` đã khôi phục RNG.
- Dựng lại cùng lịch scheduler khi resume: `LambdaLR.state_dict()` không lưu nội dung hàm lambda.

Loader kiểm tra kiến trúc, danh sách trainable, tên/kích thước trọng số, nhóm tham số optimizer và tập hợp/loại component được resume. **Không khóa augmentation, đường dẫn dữ liệu, hash mã nguồn hoặc phiên bản thư viện.** Config snapshot dùng để tham khảo và dựng lại training, không phải điều kiện bắt buộc mọi biến phải bằng nhau.

Engine lưu SHA-256 nội dung hai manifest và hai SQLite cache trong `DATA_FINGERPRINTS` của config snapshot. Khi resume, fingerprint và số batch phải khớp; thay đổi nội dung hoặc thứ tự dữ liệu vẫn bị phát hiện dù số batch không đổi. Run cũ thiếu fingerprint phải dùng `INIT_CHECKPOINT` để bắt đầu run mới. Fingerprint không phụ thuộc đường dẫn tuyệt đối và không băm nội dung từng file ảnh; cần giữ nguyên ảnh khi resume. Việc đổi manifest cho run mới vẫn cần đổi/xóa cache cũ như hướng dẫn dataloader.

Ghi checkpoint dùng file tạm rồi thay thế file đích; nếu ghi lỗi, bản `last.pt` trước đó còn nguyên. API chỉ tạo `last.pt` và `best.pt`, không tích lũy một file cho mỗi step. Format 5 lưu thêm queue image IDs và namespace trong config snapshot. Checkpoint format 4 bị từ chối; chưa có chuyển đổi tự động. Giữ nguyên namespace và cấu trúc đường dẫn tương đối khi tiếp tục dùng queue đã lưu.

RNG toàn cục không bao gồm toàn bộ trạng thái augmentation đang nằm trong worker/prefetch giữa epoch; khôi phục một run nhiều worker giống từng batch cần engine quản lý thêm trạng thái dữ liệu. Bộ checkpoint hiện chưa tự triển khai phần đó.

## 7. Tiện ích dùng chung

`src/utils/seed.py` cung cấp:

- `seed_everything(seed=None)`: seed Python, NumPy và PyTorch CPU/CUDA; mặc định lấy `config.SEED`.
- `seed_worker(worker_id)`: lấy seed worker do PyTorch cấp để seed Python và NumPy, đã được nối vào dataloader.

Gọi seed trước khi tạo model/dataloader. Seed không tự bật chế độ thuật toán deterministic.

`artifacts.py`, `logging_setup.py` và `tb_logger.py` hiện trống; chưa có cấu hình logger, ghi TensorBoard hoặc quản lý artifacts.

## 8. Chạy demo và kiểm thử

Chạy lệnh từ thư mục gốc dự án. Các thư viện được mã nguồn sử dụng gồm PyTorch, torchvision, transformers, tokenizers, NumPy, Pillow, ijson và matplotlib cho demo ảnh. SQLite dùng thư viện chuẩn Python. Repository chưa có `requirements.txt` hoặc `pyproject.toml` khóa dependencies; môi trường cần hỗ trợ DINOv3 trong transformers, RMSNorm và scaled-dot-product attention trong PyTorch.

```bash
# Demo Q-Former với image features giả lập
python -m src.qformer

# Demo forward ITC/ITM/ITG của ModelStage1
python -m src.trainingph1.model

# Demo queue với dữ liệu giả lập
python -m src.queue.moco

# Đọc dataset cấu hình và hiển thị ảnh/caption
python -m src.data.dataloaderph1

# Demo vision encoder; đoạn demo hiện yêu cầu CUDA
python -m src.vision

# Kiểm thử checkpoint trên CPU, không tải pretrained weights
python -m unittest discover -s tests -v
```

Demo model/Q-Former cần tải hoặc có sẵn pretrained weights. Một số demo model dùng tham số trực tiếp trong `main()`, không đọc toàn bộ `config.py`; chúng phục vụ kiểm tra tensor, không phải entry point training. Demo dữ liệu yêu cầu manifest và ảnh tồn tại theo config.

`tests/test_checkpoint.py` dùng BERT/DINOv3 nhỏ để kiểm tra:

- Resume pretrain khôi phục RNG và cho bước cập nhật kế tiếp giống chạy liên tục.
- Fine-tune tạo optimizer/run mới, cho phép đổi config và tiếp tục resume fine-tune.
- `prepare_training()` tổ chức thư mục version/run, dùng saved settings khi resume dù config hiện tại đã thay đổi và từ chối chọn hai nguồn cùng lúc.
- Lỗi ghi file không làm mất checkpoint trước; lưu `last.pt`/`best.pt` và ngăn tạo đè run.
- Từ chối ghép sai run, sai kiến trúc, thiếu component hoặc sai nhóm tham số optimizer.

## 9. Các phần còn cần hoàn thiện

| Phần | Công việc còn thiếu |
| --- | --- |
| Data resume | Khôi phục chính xác sampler/augmentation giữa epoch; hiện chỉ bỏ qua batch trước `next_batch` |
| Fine-tune | Entry point trong `trainingph1/finetune.py` |
| Logging/artifacts | Logger, TensorBoard và ghi kết quả |
| LLM integration | Nối projector với LLM và pipeline cho giai đoạn tiếp theo |

Các thành phần đã có có thể dùng và kiểm thử riêng; đặt tham số trong `config.py` chưa đồng nghĩa với một tính năng đã được thực thi trong vòng lặp training.
