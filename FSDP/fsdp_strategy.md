# Kế hoạch FSDP cho InA-Bridge

**Chốt phương án:** dùng FSDP2 cho online model và EMA teacher; giữ MoCo queue đầy đủ, giống nhau trên mỗi GPU; tự đồng bộ keys, loss và trạng thái training. Triển khai Stage 1 trên một node trước. Đây là kế hoạch, chưa phải tính năng đã có.

FSDP chia parameters, gradients và optimizer state thuộc phần model được shard; không tự quản lý queue, dữ liệu hay tiến độ training. Dataloader chia dữ liệu bằng `DistributedSampler`, dùng được với FSDP mà không cần bọc thêm DDP. API triển khai bám Torch 2.12 trong [environment.yml](../environment.yml) và [tài liệu FSDP2](https://docs.pytorch.org/docs/2.12/distributed.fsdp.fully_shard.html).

## 1. Mỗi thành phần sẽ nằm ở đâu?

| Thành phần hiện tại | Phương án | Việc cần làm thêm |
| --- | --- | --- |
| Online Q-Former, projection/head trainable, `dec_embedding` | Shard bằng FSDP2 | Shard từng `QFormerBlock` rồi các module cha; tạo optimizer sau sharding |
| EMA của `itc_encoder` | Shard riêng, cùng layout với online ITC | Cập nhật trên shard tương ứng sau optimizer step |
| MoCo queue: image/text features, IDs, `ptr`, `count` | Mỗi GPU giữ một bản giống nhau | Gather keys rồi enqueue cùng dữ liệu, cùng thứ tự |
| Frozen vision, BERT embeddings/LM head | Giữ replicated ở bước đầu | Khởi tạo nhất quán; giữ alias giữa LM head và word embeddings |
| Batch, activations, logits | Riêng từng rank | FSDP không tự chia nhỏ các tensor này; đo peak memory riêng |
| Scheduler, scaler nếu có, `global_step` | Mỗi rank giữ một bản | Mọi rank cùng step hoặc cùng skip |
| RNG, dataloader generator, vị trí dữ liệu | Riêng từng rank | Lưu/khôi phục đúng rank khi resume |
| Metrics, log, metadata | Reduce metrics; rank 0 ghi | Mọi rank vẫn tham gia train, validation và checkpoint phân tán |

Trong [model.py](../src/trainingph1/model.py), vision đang chạy `mock=True`; cần đo lại bằng backbone thật. Nếu frozen vision quá lớn thì shard riêng ở bước tối ưu. Các tham số frozen giữ replicated phải được loại khỏi phạm vi shard của root; mọi tham số trainable phải có owner FSDP. Những đường gọi `encode_image()`/`encode_text()` ngoài root forward cần rà soát nếu chuyển phần đó sang sharded.

## 2. MoCo queue: giữ bản sao, đồng bộ nội dung

[MoCoQueue](../src/queue/moco.py) là module riêng chứa buffers. Bọc online model bằng FSDP không làm queue được shard hay tự đồng bộ.

Với `K=4096`, `Q=128`, `D=256`, FP32, image queue chiếm **512 MiB**, text queue **4 MiB**, tổng khoảng **516 MiB/GPU**, chưa tính IDs và tensor tạm. `get()` khi queue đầy còn tạo bản sao theo thứ tự vòng; loss tiếp tục tạo bank và similarity tensors.

Quy tắc cập nhật đề xuất:

1. Mỗi rank tính teacher keys cho micro-batch của mình, không có gradient.
2. All-gather image keys, text keys và image IDs theo thứ tự rank. Bước đầu dùng batch cùng kích thước giữa các rank.
3. Dùng keys toàn cục cho ITC bank; giữ chúng trong `pending_keys` tới cuối accumulation.
4. Sau optimizer step thành công, mọi rank enqueue cùng keys theo thứ tự `(micro-batch, rank, sample)`. Keys được tính trước lần cập nhật EMA này, giữ cách hoạt động hiện tại.
5. Nếu step bị skip, tất cả bỏ pending keys. Validation chỉ đọc queue.

`K` là capacity của **một queue toàn cục được sao chép**, không tăng thành `K × world_size`. Không gather lại keys đã global; không broadcast cả queue mỗi step. Broadcast queue khi khởi tạo/resume, sau đó cập nhật giống nhau; kiểm tra `ptr`, `count`, IDs và checksum features trong test.

**Chưa shard queue ở vòng đầu.** Shard queue cần thiết kế loss phân tán để softmax và targets vẫn xét toàn bank; chỉ tính loss trên phần queue cục bộ sẽ đổi objective. Nếu queue là nút thắt, đo giảm capacity/dtype hoặc tính similarity theo chunk trước; lưu BF16 chưa chắc giảm peak tương ứng vì criterion hiện chuyển bank về FP32.

## 3. EMA teacher và contrastive loss

### EMA teacher

[EMA](../src/queue/ema.py) chỉ sao chép `itc_encoder`: Q-Former và hai projection ITC, không phải toàn bộ `ModelStage1`.

- Tạo teacher từ online weights **trước khi shard**, đặt frozen và `eval()`.
- Shard online ITC và teacher theo cùng mesh, block boundaries và parameter placements. Không deepcopy model đã shard.
- Sau step thành công, cập nhật `teacher_shard = m × teacher_shard + (1 − m) × online_shard`; kiểm tra tên/shape/layout trước khi dùng cách này.
- Teacher chạy `no_grad()` nên không có backward để thu hồi weights; bảo đảm reshard sau forward. Buffers nếu có vẫn cần chính sách copy/đồng bộ riêng.

### ITC / ITM / ITG

[Criterion hiện tại](../src/losses/lossph1.py) giả định teacher keys của batch cũng là queries cho soft targets. Không thể chỉ thay keys cục bộ bằng keys đã gather mà giữ nguyên hàm loss.

| Objective | Thay đổi cần thiết |
| --- | --- |
| ITC | Online queries và teacher queries vẫn local; candidate bank = teacher keys toàn cục + queue. Tách hai loại input này trong criterion. Positive mask dùng image ID để xử lý nhiều caption/cùng ảnh ở nhiều rank |
| ITM | Giữ negative sampling trong batch local ở vòng đầu. Tất cả rank gọi cùng số lượt forward ITM; rank không có negative dùng dummy pairs và mask loss/count về 0, không bỏ nhánh forward/backward |
| ITG | Chuẩn hóa bằng tổng token hợp lệ toàn cục; bỏ padding và labels `-100` |

Teacher keys đã detached nên gather không cần autograd. ITM local có tập negatives khác baseline một GPU dùng cả global batch; phép so sánh phải mô phỏng cùng phạm vi negatives.

Cho mỗi objective, dùng loss **sum** và valid count riêng. Với FSDP trung bình gradient trên `W` rank, mỗi micro-batch backward bằng `W × local_loss_sum / global_count`, trong đó `global_count` cộng trên toàn bộ rank và micro-batch của optimizer step. Không chia thêm cho số accumulation steps. Count bằng 0 thì trả zero loss có graph và giữ lịch gọi module thống nhất.

## 4. Dataloader và vòng lặp training

[Dataloader hiện tại](../src/data/dataloaderph1.py) đã có `DistributedSampler`, nhưng chưa nối xong với engine:

- Sửa `generatorr=generator` thành `generator=generator`.
- Khởi tạo process group và chọn GPU bằng `LOCAL_RANK` trước khi dựng model/loader; engine truyền `distributed=True`.
- Gọi `sampler.set_epoch(epoch)` trước khi tạo iterator. Tách seed model dùng chung và seed RNG/augmentation theo rank; sampler dùng cùng seed giữa các rank. Đây là yêu cầu shuffle của [DistributedSampler](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler).
- Training dùng `drop_last=True` ở sampler và loader cho baseline có cùng batch size/số lượt giữa các rank; kiểm tra loader không rỗng.
- Cache hiện do rank 0 tạo rồi barrier: phù hợp khi các rank thấy cùng đường dẫn. Khi mở rộng nhiều node phải dùng shared cache hoặc một người ghi trên mỗi node.

Thứ tự một optimizer step:

```text
Lấy nhóm micro-batch → reduce counts toàn cục → zero_grad
  Với mỗi micro-batch:
    online + teacher forward → gather teacher keys/IDs
    ITC + ITM + ITG → chuẩn hóa loss → backward
Kiểm tra finite toàn cục → clip bằng gradient norm toàn model
  Thành công: optimizer → scheduler → EMA → enqueue → global_step += 1
  Thất bại: tất cả cùng skip và bỏ pending keys
Cập nhật vị trí dữ liệu → validation/checkpoint nếu đến lịch
```

Bắt đầu với BF16 compute, FP32 reduction và loss nhạy số học; đồng bộ gradient ở mỗi backward để dễ kiểm tra accumulation. Norm phải tính trên toàn bộ shards, không dùng norm cục bộ như norm toàn model. Nếu dùng FP16, overflow và scaler phải thống nhất giữa các rank.

Validation chạy trên mọi rank, reduce **sum/count** rồi rank 0 ghi kết quả. Sampler validation có thể thêm mẫu đệm: loại các mẫu đó khỏi metrics, ITC candidate bank và ITM negatives nhưng vẫn giữ số lượt collective bằng nhau. Không cập nhật EMA/queue và phải bảo toàn RNG training.

## 5. Checkpoint phải chứa cả trạng thái ngoài model

Dùng [Distributed Checkpoint (DCP)](https://docs.pytorch.org/docs/2.12/distributed.checkpoint.html) cho model/optimizer/teacher sharded; tất cả rank tham gia save/load. Tạo format distributed mới, giữ format 5 hiện tại cho load/migration.

| Phạm vi lưu | Nội dung |
| --- | --- |
| Sharded | Online weights, optimizer state, EMA teacher weights |
| Một bản dùng chung, phân phối lại khi load | Queue đầy đủ; EMA momentum; scheduler/scaler; epoch, next batch, global step, best metric; config, tokenizer, data fingerprints và topology |
| Theo từng rank | RNG Python/NumPy/Torch CPU/CUDA, dataloader generator và thông tin replay dữ liệu/augmentation |

Chỉ save sau optimizer boundary hoàn chỉnh, khi không còn gradient hoặc pending keys. Ghi checkpoint mới xong trên mọi rank mới cập nhật `last`; rank 0 quản lý metadata/log, không để nhiều rank ghi đè một `last.pt`.

Resume ban đầu yêu cầu cùng world size, batch/accumulation, sampler và dữ liệu. Cần kiểm thử replay workers/augmentation; chỉ lưu global seed là chưa đủ. DCP có thể reshard khi đổi topology nhưng không bảo đảm chuỗi dữ liệu và RNG giữ nguyên. Full model export để inference/chuyển stage là đường riêng.

## 6. Thứ tự triển khai và nghiệm thu

| Bước | Phạm vi sửa | Điều kiện hoàn thành |
| --- | --- | --- |
| 1. Nối runtime/data | Thêm helper distributed tối thiểu; sửa `engine.py`, `dataloaderph1.py` | Hai rank dùng đúng GPU, nhận dữ liệu theo sampler, cùng số batch; khởi tạo weights nhất quán |
| 2. Shard online/teacher | `model.py`, `ema.py`, setup optimizer | Forward/backward đủ ba objective; teacher shard khớp EMA tham chiếu; tied weights giữ đúng |
| 3. Global ITC và queue | `lossph1.py`, helper loss/enqueue trong engine | Queue giống nhau sau accumulation, wrap-around và global batch lớn hơn capacity; positive IDs xuyên rank đúng |
| 4. Hoàn thiện training state | Loss reduction, finite/clip, validation, checkpoint | Một rank NaN khiến tất cả skip; token/count khác nhau vẫn đúng; resume khớp batch tiếp theo và trạng thái training |
| 5. Đo rồi tối ưu | Backbone thật, activation checkpointing, reshard policy, queue | Ghi peak memory từng GPU, samples/s và thời gian checkpoint với cùng cấu hình bài toán |

Đối chiếu số học bằng model nhỏ, dữ liệu cố định, tắt dropout/augmentation trước; kiểm tra RNG/resume riêng. Bao gồm ca một rank không có ITM negative, zero valid token và nhóm accumulation cuối thiếu micro-batch.

Stage 2 sẽ dùng lại runtime, sharding và checkpoint sau khi Stage 1 chạy đúng. Khi đó thêm shard plan cho LLM và chính sách freeze/projector; mặc định không mang EMA/queue sang nếu objective không cần. Chưa tách engine tổng quát hoặc thêm backend khác trong đợt này.
