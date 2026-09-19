# InA-Bridge

InA-Bridge là dự án xây dựng mô hình thị giác–ngôn ngữ, sử dụng DINOv3 để trích xuất đặc trưng ảnh và Q-Former để học biểu diễn liên kết ảnh–văn bản. Mã hiện tại tập trung vào giai đoạn 1 với ba mục tiêu ITC, ITM và ITG. Module `QwenProjector` được chuẩn bị để ánh xạ đầu ra Q-Former sang chiều biểu diễn của mô hình ngôn ngữ; luồng tích hợp LLM chưa được triển khai trong `src`.

## Trạng thái

- `src/` là mã đang được tổ chức lại: đã có vision encoder, Q-Former, model giai đoạn 1, dataloader, EMA và MoCo queue.
- `src/losses/lossph1.py` chưa triển khai criterion; `src/config.py` chưa có cấu hình và `src/trainingph1/engine.py` chưa được tạo.
- `src_src/` lưu bản gốc để đối chiếu hành vi, bao gồm loss và engine giai đoạn 1.
- Thiết kế criterion và engine bên dưới là hướng triển khai đã thống nhất, chưa phải API chạy được.

## Kiến trúc

```text
Ảnh ── DINOv3 (đóng băng) ── patch features ──┐
                                             ├── Q-Former ── ITC / ITM / ITG
Văn bản ── BERT embeddings (đóng băng) ────────┘
```

DINOv3 trả patch features sau khi loại CLS token và register tokens. Q-Former khởi tạo các lớp từ BERT, bổ sung query tokens học được và cross-attention từ query sang đặc trưng ảnh. Attention mask thay đổi theo từng mục tiêu:

| Mục tiêu | Cách xử lý | Đầu ra của model |
| --- | --- | --- |
| ITC — Image-Text Contrastive | Query và text self-attention riêng biệt; chiếu và chuẩn hóa biểu diễn | Image features `[B, Q, D]`, text features `[B, D]` |
| ITM — Image-Text Matching | Query và text tương tác hai chiều | Logits khớp/không khớp `[B, 2]` |
| ITG — Image-grounded Text Generation | Text nhìn query và các token văn bản trước đó | Logits dự đoán token kế tiếp `[B, T-1, V]` |

`B` là batch size, `Q` là số query, `D` là chiều embedding ITC, `T` là độ dài token và `V` là kích thước từ vựng. Khi ghép positive và negative cho ITM, batch đầu vào nhánh này có kích thước `3B`.

Vision encoder và BERT embeddings được đóng băng. LM head dùng chung trọng số với word embeddings và cũng được đóng băng; gradient từ ITG vẫn truyền về Q-Former.

## Thiết kế huấn luyện giai đoạn 1

### Engine điều phối

Engine sở hữu online model, momentum model, queue và thứ tự thực hiện một training step:

1. Forward online model để lấy biểu diễn ITC; forward momentum model trong `no_grad` để lấy keys.
2. Ghép keys của batch hiện tại với keys cũ trong queue, tạo image bank và text bank.
3. Tính similarity logits hai chiều, positive targets và pseudo targets từ momentum model.
4. Gọi hàm tính loss ITC của criterion.
5. Chọn hard negative từ similarity của nhánh ITC; ghép batch và labels cho ITM.
6. Forward ITM và gọi hàm tính loss ITM.
7. Forward ITG, chuẩn bị labels đã dịch một token và mask padding, rồi gọi hàm tính loss ITG.
8. Tổng hợp loss theo trọng số, backward và thực hiện optimizer step.
9. Cập nhật EMA và enqueue keys đã tính trong step khi optimizer step thành công. Nếu AMP bỏ qua step do overflow, bỏ qua cả hai cập nhật này.

Pseudo target được dựng tại engine theo công thức:

```text
target = (1 - distill_weight) × positive_target
       + distill_weight × softmax(momentum_logits)
```

Teacher distribution và targets không giữ gradient. Engine quản lý `temperature`, lịch `distill_weight` và dữ liệu trung gian.

### Hard negative sampling

Similarity ảnh–text là tích vô hướng giữa từng query ảnh và text embedding, lấy giá trị lớn nhất trên các query rồi chia cho temperature.

Theo bản gốc, ITC loss so sánh online features với momentum feature banks. Hard negative sampling dùng một ma trận similarity riêng giữa **online image features và online text features trong cùng batch**; không dùng scalar loss ITC hoặc thay bằng phần logits online–momentum.

Engine mask đường chéo positive bằng `-inf` trước softmax, sau đó dùng `multinomial` để lấy một negative text cho mỗi ảnh và một negative image cho mỗi text. Đây là sampling theo similarity, không phải chọn top-1. Batch phải có ít nhất hai mẫu; bản gốc giả định các cặp ngoài đường chéo là negative.

Batch ITM giữ thứ tự:

1. Ảnh đúng – text đúng, label `1`.
2. Ảnh đúng – text negative, label `0`.
3. Ảnh negative – text đúng, label `0`.

Việc chọn indices chạy trong `no_grad`; các features được chọn để forward ITM vẫn giữ đường gradient vốn có.

### Criterion tính loss

Dự kiến dùng một class `Stage1Criterion(nn.Module)` trong `src/losses/lossph1.py`, khởi tạo một lần và cung cấp các hàm riêng:

| Hàm dự kiến | Đầu vào | Trách nhiệm |
| --- | --- | --- |
| `loss_itc` | Logits và targets của hai chiều | Soft-target cross-entropy, lấy trung bình hai chiều |
| `loss_itm` | ITM logits và labels | Cross-entropy phân loại |
| `loss_itg` | ITG logits và labels đã chuẩn bị | Cross-entropy token với `ignore_index` |
| `total_loss` | Các loss đã tính | Tổng có trọng số của ITC, ITM và ITG |

Engine gọi từng hàm đúng thời điểm, có bước mining và forward ITM xen giữa các lần gọi. Không gom toàn bộ pipeline vào một `forward()` tính đồng thời ba loss.

Criterion giữ trọng số loss và `ignore_index`; không giữ model, queue hoặc dữ liệu batch, không dựng pseudo target và không sampling negative.

```text
loss = itc_weight × loss_itc
     + itm_weight × loss_itm
     + itg_weight × loss_itg
```

## Cấu trúc mã

```text
src/
├── config.py                 # Chưa có cấu hình
├── vision.py                 # DINOv3 image encoder
├── qformer.py                # Query tokens, attention masks và Q-Former blocks
├── projector.py              # QwenProjector
├── data/
│   └── dataloaderph1.py       # Dataset, SQLite cache, augmentation và collator
├── losses/
│   └── lossph1.py             # Vị trí triển khai criterion
├── queue/
│   ├── ema.py                # Momentum model
│   └── moco.py               # Queue lưu image/text keys
└── trainingph1/
    └── model.py              # ModelStage1 và các nhánh ITC/ITM/ITG

src_src/                     # Bản gốc tham chiếu
documents/                   # Tài liệu nghiên cứu
DatasetProject/              # Dữ liệu cục bộ
cache/                       # Cache cục bộ
```

## Dữ liệu

Dataloader nhận một JSON array gồm tên ảnh và văn bản tương ứng:

```json
[
  {"image_name": "image_001.jpg", "text": "A dog is running on the grass."},
  {"image_name": "image_002.jpg", "text": "Two people are sitting at a table."}
]
```

`image_name` được nối với `image_dir` để mở ảnh. Metadata được đọc theo luồng bằng `ijson` và lưu vào SQLite. Collator trả:

- `images`: tensor `[B, 3, H, W]`.
- `input_ids`: tensor `[B, T]`.
- `attention_mask`: tensor boolean `[B, T]`, `True` là token hợp lệ.

Cache hiện được đặt tên `train.sqlite` hoặc `validation.sqlite` trong `cache_dir` và tái sử dụng nếu đã tồn tại. Khi đổi manifest, cần dùng thư mục cache khác hoặc xóa cache cũ để tạo lại.

## Chạy các demo hiện có

Các thư viện được mã nguồn sử dụng gồm `torch`, `torchvision`, `transformers`, `Pillow`, `ijson` và `matplotlib` cho demo dữ liệu. Repository chưa có dependency manifest khóa phiên bản; môi trường cần hỗ trợ `DINOv3ViTModel`, `nn.RMSNorm` và scaled dot-product attention.

Chạy từ thư mục gốc dự án:

```bash
# Q-Former với image features giả lập và BERT pretrained
python -m src.qformer

# Forward ba objective với ảnh giả lập, DINOv3 và BERT pretrained
python -m src.trainingph1.model

# Đọc dữ liệu, in shape và hiển thị ảnh/caption
python -m src.data.dataloaderph1
```

Các demo model cần tải hoặc có sẵn pretrained weights; quyền truy cập model trên Hugging Face cần phù hợp với model được sử dụng. Demo dataloader hiện trỏ tới `DatasetProject/BLIP3o/dataset_metadata.json` và thư mục ảnh `DatasetProject/BLIP3o/Image`.

Demo forward dùng ảnh ngẫu nhiên để kiểm tra luồng tensor, không đánh giá chất lượng mô hình. Luồng train hoàn chỉnh trong `src/` sẽ được bổ sung sau khi triển khai criterion, engine và cấu hình.
