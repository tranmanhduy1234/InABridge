# Kế hoạch FSDP/ZeRO-3 và đồng bộ queue cho Stage 1

Trạng thái: tài liệu thiết kế, chưa phải tính năng đã triển khai. Cập nhật theo thảo luận ngày 2026-10-06.

Mục tiêu là shard online model, giữ EMA teacher đúng phạm vi ITC Encoder, và chia hai bank ảnh/text mà vẫn bảo toàn objective contrastive toàn cục. Trong tài liệu này, “shard” nghĩa là chia trạng thái giữa các rank; không phải mỗi GPU luôn giữ một bản đầy đủ.

[Kế hoạch trước](fsdp_strategy.md) chọn replicated queue cho vòng đầu. Tài liệu này phát triển hướng **sharded queue** đang thảo luận; replicated queue chỉ còn là baseline để đối chiếu correctness. Backend cuối cùng và cách đọc queue tiết kiệm memory chưa được chốt.

## 1. Hiện trạng code

| Thành phần | Hiện trạng | Hệ quả khi chuyển distributed |
| --- | --- | --- |
| [ModelStage1](../src/trainingph1/model.py) | Vision frozen, embeddings frozen, Q-Former dùng chung cho ITC/ITM/ITG, các head tương ứng | Phải thiết kế ranh giới shard theo đường gọi thực tế |
| [EMA](../src/queue/ema.py) | Deepcopy `online_model.itc_encoder` | Teacher chỉ gồm Q-Former và hai projection ITC; không nhân đôi toàn bộ online model |
| Teacher inputs | Dùng lại image features và frozen text embeddings từ online | Không cần teacher vision encoder riêng trong thiết kế hiện tại |
| [MoCoQueue](../src/queue/moco.py) | Image/text buffers dùng chung IDs, `ptr`, `count` | Đây là hai bank của một FIFO cặp ảnh–text, không phải hai queue độc lập |
| [ITC loss](../src/losses/lossph1.py) | Candidates là teacher keys của batch local cộng queue local | Chưa có candidate bank toàn cục xuyên rank |
| [Engine](../src/trainingph1/engine.py) | Giữ `pending_keys`, cập nhật EMA và enqueue sau optimizer step thành công | Cần thống nhất quyết định step/skip và enqueue trên mọi rank |
| Dataloader trong engine | Vẫn import `dataloaderph1.py` | Bản `dataloaderph1_multilevel.py` đã test chưa được nối vào engine |
| Checkpoint | Lưu state dict vào file theo cơ chế hiện tại | Chưa đủ cho trạng thái model/optimizer/teacher/queue phân tán |

Hai chi tiết ảnh hưởng trực tiếp tới sharding:

- `lm_head.weight` và `embeddings.word_embeddings.weight` là cùng parameter. Phải giữ weight tying, không tạo hai owner độc lập cho cùng trọng số.
- Vision hiện được dựng với `mock=True`. Kết quả memory/throughput của mock không đại diện backbone thật.

## 2. FSDP2 và ZeRO-3 giải quyết phần nào?

Cả hai hướng đều chia parameters, gradients và optimizer state thuộc phạm vi quản lý; weights cần thiết được tập hợp để tính toán. Chúng không tự thiết kế global contrastive bank, queue FIFO, sampler hay trạng thái tiến độ training của ứng dụng.

| Vấn đề | FSDP2 | DeepSpeed ZeRO-3 |
| --- | --- | --- |
| Cách tích hợp | `fully_shard` theo module/block, kết hợp DeviceMesh và DTensor | DeepSpeed engine quản lý training và partition trạng thái |
| Ranh giới giao tiếp | Theo các module được shard | Theo cơ chế partition/fetch và cấu hình ZeRO-3 |
| Optimizer | Tạo sau khi shard theo workflow FSDP2 | Khởi tạo/tích hợp theo workflow DeepSpeed |
| EMA | Có thể cập nhật shard tương ứng khi tên, shape, mesh và placements khớp | Cần đường cập nhật tương thích partition của backend; không giả định EMA tensor thường dùng nguyên xi |
| Truy cập weights ngoài forward | Cần đúng hooks hoặc đường unshard có kiểm soát | Cần parameter coordination khi truy cập ngoài module sở hữu |
| Queue buffers | Ứng dụng tự quản lý | Ứng dụng tự quản lý |
| Checkpoint | Thiết kế quanh distributed state dict/DCP | Dùng checkpoint của DeepSpeed cho trạng thái do engine quản lý; lưu thêm trạng thái ứng dụng |

Chọn **một backend cho mỗi lần training**, không bọc chồng FSDP và ZeRO-3 lên cùng parameters. Đề xuất triển khai FSDP2 trước vì engine hiện dùng PyTorch trực tiếp; giữ semantics queue/loss độc lập để có thể triển khai ZeRO-3 sau. Không cần xây abstraction nhiều backend ngay từ đầu.

API phải đối chiếu phiên bản môi trường thực tế. Tham khảo [FSDP2, PyTorch 2.12](https://docs.pytorch.org/docs/2.12/distributed.fsdp.fully_shard.html), [ZeRO-3 và parameter coordination](https://deepspeed.readthedocs.io/en/stable/zero3.html), [DeepSpeed training API](https://deepspeed.readthedocs.io/en/stable/training.html).

## 3. Bố trí model và EMA

Online model có một bộ weights logic chung. Mỗi rank xử lý batch riêng; backend phân phối trạng thái và tổng hợp gradient để cập nhật cùng mô hình logic.

- Shard Q-Former theo block, rồi các module cha/projection/head còn lại.
- Nếu mục tiêu là shard toàn bộ online model, frozen vision và embeddings cũng cần shard plan rõ ràng. Frozen chỉ có nghĩa không tính gradient, không có nghĩa không chiếm memory.
- Giữ weight tying giữa embeddings và LM head trong một ownership plan nhất quán.
- Rà soát `encode_image()` và `encode_text()` đang được gọi ngoài root forward. Parameters đã shard phải được tập hợp qua đúng đường gọi; chỉ shard root rồi giữ nguyên mọi helper chưa đủ.
- Giữ một Q-Former online dùng chung cho ba objective, không tạo ba bản online riêng.

EMA teacher chỉ gồm `ITCEncoder`. Tạo teacher từ online weights trước khi shard, đặt `requires_grad=False`, giữ `eval()`.

Với FSDP2, dùng cùng mesh và ranh giới shard tương ứng cho online ITC và teacher ITC. Sau optimizer step thành công:

```text
teacher_shard = momentum * teacher_shard + (1 - momentum) * online_shard
```

Phải kiểm tra tên parameter, shape, dtype, placements và ownership trước khi dùng phép cập nhật này. Buffers cần chính sách copy riêng. Teacher không có backward nên cần reshard sau forward để tránh giữ full weights ngoài ý muốn.

Với ZeRO-3, cần kiểm chứng riêng cách đọc/cập nhật EMA từ partitioned weights. Không dùng `.data` hoặc ghép shards theo suy đoán. Gather weights có kiểm soát có thể làm baseline correctness, nhưng có nguy cơ tăng peak memory.

## 4. Hai bank phải là một FIFO toàn cục

Mỗi slot chứa một record:

```text
(image_features[Q,D], text_features[D], image_id)
```

Image features, text features và ID của cùng record phải ở cùng owner và cùng vị trí logic. Không chia hai bank bằng hai quy tắc khác nhau; không dùng pointer hoặc thời điểm commit riêng.

Quy ước:

- `K`: tổng capacity toàn hệ thống, không phải capacity trên mỗi rank.
- `W`: số rank trong nhóm data parallel/queue.
- `global_ptr`: slot sẽ được ghi tiếp theo.
- `global_count`: số record hợp lệ, tối đa `K`.
- `queue_version`: số lần commit queue thành công.

Metadata nhỏ được giữ nhất quán trên mọi rank. Payload được shard theo sample. Nếu sau này có tensor/pipeline parallel, queue group phải được xác định theo nhóm dữ liệu, không mặc định lấy toàn bộ process group.

### Chi phí bộ nhớ hiện tại

Với `K=4096`, `Q=128`, `D=256`, FP32:

| Buffer | Toàn queue | Mỗi rank nếu chia đều 4 rank |
| --- | ---: | ---: |
| Image features | 512 MiB | 128 MiB |
| Text features | 4 MiB | 1 MiB |
| IDs int64 | 32 KiB | 8 KiB |

Đây chỉ là storage thường trực. `get()` hiện có thể tạo bản sao theo thứ tự FIFO; criterion còn cast FP32, nối bank và tạo similarities. Queue BF16 không tự bảo đảm peak giảm một nửa nếu loss lại materialize toàn bank FP32.

## 5. Đồng bộ ghi queue

Thứ tự enqueue toàn cục cố định:

```text
optimizer step → micro-batch → rank → sample trong batch
```

Với record thứ `j` của một lần commit:

```text
slot       = (global_ptr + j) % K
owner_rank = slot % W
local_slot = slot // W
```

Ví dụ `K=8`, `W=2`:

```text
rank 0 sở hữu global slots: 0, 2, 4, 6
rank 1 sở hữu global slots: 1, 3, 5, 7
```

Quy tắc này vẫn xác định được ownership khi `K` không chia hết `W`; số slot mỗi rank có thể lệch một. Bước đầu có thể yêu cầu `K % W == 0` để đơn giản buffer giao tiếp, nhưng phải validate cấu hình rõ ràng.

Quy trình:

1. Mỗi rank tạo teacher keys detached cho micro-batch local.
2. Trao đổi keys và IDs theo cùng thứ tự rank. Với batch size khác nhau phải trao đổi sizes và mask phần padding; baseline dùng batch bằng nhau.
3. Giữ keys pending tới optimizer boundary; mỗi record chỉ được enqueue một lần.
4. Khi tất cả rank thống nhất step thành công, mỗi rank chỉ ghi các slot mình sở hữu.
5. Cập nhật metadata bằng cùng tổng số record mới `G`:

```text
global_ptr   = (global_ptr + G) % K
global_count = min(K, global_count + G)
queue_version += 1
```

Nếu `G > K`, chỉ giữ `K` record cuối trong thứ tự enqueue, nhưng pointer vẫn tiến theo toàn bộ `G`. Với `G == 0`, không ghi dữ liệu hay tăng version.

Đường triển khai đơn giản là all-gather keys mới rồi lọc owner. Khi đã đúng, có thể route payload tới owner bằng `all_to_all`. Routing chỉ tối ưu khâu ghi: ITC vẫn cần nhìn đủ candidates toàn cục khi đọc.

Không all-reduce trung bình features của cùng slot: những features đó đại diện các samples khác nhau. Không broadcast toàn queue ở mỗi step. Khi resume phải khôi phục từng shard cùng metadata trước khi bắt đầu đọc.

## 6. Đọc queue và giữ nguyên ITC objective

Thiết kế inputs cho mỗi rank:

```text
online queries  = online features local
teacher queries = teacher features local
candidate bank  = teacher keys global của micro-batch hiện tại + global queue
```

Cần tách teacher queries local khỏi candidate keys global trong API criterion. Hàm hiện tại dùng cùng đầu vào cho hai vai trò này; chỉ thay tensor local bằng tensor all-gather sẽ làm sai kích thước hoặc semantics.

Positive mask dùng `image_id`, bao gồm positives ở rank khác hoặc có sẵn trong queue. Số positive để chuẩn hóa hard targets phải tính trên toàn bank. Không dùng index đường chéo local làm positive duy nhất.

Nếu chỉ tính loss trên queue shard local, số negatives, positive mask, denominator softmax và teacher pseudo targets đều thay đổi. Đồng bộ gradient model không khắc phục được thay đổi objective đó.

### Hai phương án đọc

| Phương án | Vai trò | Giới hạn |
| --- | --- | --- |
| Gather toàn bank trước loss | Baseline correctness, dễ so loss/gradient với single-process | Shard storage nhưng có thể không giảm peak |
| Truyền queue theo chunk qua các rank | Hướng tiết kiệm memory thực tế | Cần thiết kế cả forward, backward và thứ tự collectives |

Với đọc theo chunk, mỗi rank giữ queries local và lần lượt đọc đủ candidate chunks. Với logits `z_j`, softmax phải dùng cùng denominator toàn bank:

```text
logZ = logsumexp(z_j trên mọi candidate hợp lệ)
log_probability_j = z_j - logZ
```

Không lấy trung bình các loss softmax đã chuẩn hóa riêng từng chunk. Teacher soft targets cũng cần normalization toàn bank; masked slots chưa hợp lệ không được tham gia.

Không all-reduce trực tiếp denominator của queries local giữa các rank: hàng số 0 ở hai rank thường là hai sample khác nhau. Phương án broadcast/ring candidate chunks giúp mỗi rank tự tính normalization cho đúng queries của mình. Nếu chọn phân phối query computation theo hướng khác, phải thiết kế thêm gradient communication tương ứng.

Chỉ viết vòng lặp chunk chưa đủ: autograd có thể giữ toàn bộ keys để backward. Muốn giảm peak thực sự phải dùng recomputation hoặc custom backward, kiểm tra gradient theo cả hai chiều ITC và xử lý `amax` trên query dimension đúng như baseline. Các lượt giao tiếp khi recompute phải thống nhất giữa rank.

Queue phải bất biến từ lúc loss bắt đầu đọc đến khi backward/recomputation dùng xong dữ liệu đó.

## 7. Optimizer step, accumulation và EMA

Giữ semantics hiện tại: keys được tính trước optimizer/EMA update; queue chỉ commit sau step thành công.

```text
Lấy nhóm micro-batch → tính global valid counts → zero_grad
  Với mỗi micro-batch:
    online forward + teacher forward
    trao đổi current keys/IDs
    tính ITC, ITM, ITG với queue chưa thay đổi
    backward
    giữ pending keys
Thống nhất finite/overflow → tính global gradient norm/clip
  Thành công:
    optimizer → scheduler → EMA → queue commit → global_step += 1
  Thất bại:
    mọi rank cùng skip, bỏ pending keys
Cập nhật vị trí dữ liệu đã tiêu thụ
Validation/checkpoint nếu đến lịch
```

- Quyết định `updated` hiện tại là local; phải chuyển sang quyết định toàn cục. Một rank overflow thì mọi rank cùng skip.
- Với FP16, scaler và kết quả optimizer step phải thống nhất trước khi cập nhật EMA/queue. Không chỉ dựa vào một cờ kiểm tra trước `scaler.step()` nếu backend có thể skip bên trong.
- Gradient norm phải xét toàn model sharded, không lấy norm shard local thay cho global norm.
- Gradient accumulation không tự biến nhiều micro-batch thành một contrastive batch lớn. Thiết kế trên dùng candidates của global micro-batch hiện tại và queue cũ. Muốn negatives xuyên các micro-batch cùng optimizer step là một thay đổi objective riêng.
- Queue được giữ nguyên trong cả accumulation group. Có thể giảm pending memory bằng giữ payload thuộc owner sau khi đã dùng cho current bank, thay vì lưu toàn bộ gathered payload trên mọi rank.

Tốc độ thay mới queue phụ thuộc global batch: một step thành công thêm khoảng `W * local_batch * accumulation` records. Giữ nguyên `K` khi tăng world size sẽ rút ngắn tuổi queue tính theo optimizer steps; cần xem lại `K` và momentum dựa trên semantics mong muốn.

## 8. Các điểm phải sửa trong training pipeline

### ITM và lịch forward

Engine hiện bỏ nhánh ITM khi batch local không có ảnh khác nhau để lấy negative. Khi một rank bỏ nhánh nhưng rank khác chạy ba forward ITM, thứ tự collective của sharded model có thể lệch.

Baseline giữ negative sampling local, nhưng mọi rank phải gọi cùng lịch module. Rank không có negative dùng dummy pairs và mask loss/count về 0; vẫn giữ graph phù hợp. Baseline kiểm chứng một GPU phải mô phỏng cùng phạm vi negatives local, không mặc định dùng global negatives.

### Chuẩn hóa loss

Mỗi objective dùng local loss sum và valid count riêng. ITG cần tổng token hợp lệ toàn cục; length-aware batches có số token khác nhau giữa rank.

Nếu backend trung bình gradient trên `W` rank, scale mỗi micro-batch bằng:

```text
scaled_loss = W * local_loss_sum / global_valid_count_of_accumulation_group
```

Không chia thêm cho accumulation steps khi denominator đã gồm toàn group. Count bằng 0 phải trả zero loss có graph phù hợp và không làm lệch lịch collective. Xác nhận quy ước reduction/scaling thực tế của backend, tránh DeepSpeed và engine cùng chia loss hai lần.

### Dataloader và seed

- Khởi tạo process group, device theo local rank, rồi mới dựng distributed loader/model.
- Nối `dataloaderph1_multilevel.py` vào engine và truyền distributed mode rõ ràng.
- Các rank dùng cùng lengths, sampler config, seed và epoch; gọi `set_epoch()` trước tạo iterator, kể cả resume.
- Training dùng `drop_last=True`, cùng số batch và optimizer boundaries trên mọi rank; từ chối loader rỗng.
- Seed khởi tạo model nhất quán; RNG augmentation/dropout và trạng thái replay được quản lý theo rank.
- Cache chỉ do rank 0 xây hiện cần đường dẫn dùng chung. Nhiều node với local disk cần shared cache hoặc một writer mỗi node.

Bộ [test dataloader](../tests/test_dataloaderph1_multilevel.py) đã kiểm tra partition 2/4 rank bằng mock, coverage, epoch/resume và workers. Đây chưa phải test training distributed thật. Lỗi đã phát hiện: `load_token_lengths()` dùng `uint16`, gây overflow khi token length vượt 65.535; cần xử lý trước khi hỗ trợ `max_length` lớn hơn giới hạn này.

### Validation và logging

Validation phải tham gia đúng lịch collective, chỉ đọc queue, không cập nhật EMA. Reduce metric bằng sum/count; nếu sampler thêm mẫu đệm thì loại chúng khỏi metrics và candidate semantics. Rank 0 ghi log/metadata; không chỉ cho rank 0 chạy forward của model sharded.

## 9. Checkpoint và resume

| Phạm vi | Trạng thái phải lưu |
| --- | --- |
| Model/backend | Online parameters, optimizer, teacher ITC shards |
| Queue shards | Image/text payload, IDs, mapping global slots hoặc thông tin đủ để tái dựng |
| Metadata chung | Capacity, pointer, count, queue version, ownership rule, world size/mesh |
| Training chung | Scheduler, scaler nếu có, EMA momentum, epoch, next batch, global step, config, fingerprints |
| Theo rank | RNG Python/NumPy/Torch CPU/CUDA, dataloader generator và trạng thái replay |

Chỉ checkpoint tại optimizer boundary hoàn chỉnh, không còn gradients hoặc pending keys. Tất cả rank hoàn thành ghi trước khi đánh dấu checkpoint hợp lệ; tránh nhiều rank cùng ghi đè `last.pt`.

Với FSDP2, dùng [Distributed Checkpoint](https://docs.pytorch.org/docs/2.12/distributed.checkpoint.html) cho trạng thái sharded phù hợp. Với ZeRO-3, dùng cơ chế checkpoint của backend và bổ sung queue/application state theo cùng commit boundary.

Baseline resume yêu cầu cùng world size, batch/accumulation, sampler và dữ liệu. Đổi world size cần remap queue ownership và reshard model; ngay cả khi khôi phục được weights, chuỗi batch/RNG không mặc nhiên giống lần chạy cũ.

## 10. Cấu hình cần chốt

Các tên dưới đây là đề xuất thiết kế, chưa phải config đã được code hỗ trợ.

| Nhóm | Nội dung cần cấu hình |
| --- | --- |
| Backend | FSDP2 hoặc ZeRO-3, phiên bản, mesh/process group, device |
| Shard plan | Module boundaries, frozen modules, tied weights, reshard/prefetch |
| Precision | Compute dtype, gradient reduction dtype, queue dtype, FP32 cho loss nhạy số học |
| Queue | Global capacity, replicated/sharded, ownership rule, candidate chunk size |
| ITC | Global current candidates, temperature, pseudo weight/warmup |
| EMA | Momentum, thời điểm update, dtype/layout tương ứng |
| Training | Local batch size, accumulation, clip norm, global skip policy |
| Data | Sampler seed, mega-bucket, jitter, max length, drop_last, cache scope |
| Recovery | Checkpoint format, topology policy, RNG/data replay |

Bước đầu ưu tiên BF16 nếu phần cứng hỗ trợ, giữ cùng batch size giữa các rank và chưa bật offload. Chỉ chọn prefetch/offload/chunk size sau khi đo memory và throughput với backbone thật.

## 11. Lộ trình và tiêu chí nghiệm thu

1. **Distributed runtime và model:** nối loader, shard online/EMA, giữ tied weights, lịch forward ITM nhất quán. Hai rank chạy đủ ba objective, có cùng số optimizer steps.
2. **Global ITC baseline:** gather current keys, dùng full candidate bank để đối chiếu loss và gradient với model nhỏ single-process; tắt dropout/augmentation khi so số học.
3. **Sharded queue:** implement ownership và commit; ghép shards phải tái tạo đúng FIFO tham chiếu.
4. **Đọc theo chunk:** giữ nguyên loss/gradient và giảm peak memory đo được, kể cả backward.
5. **Training state:** global skip/clip, validation, checkpoint và resume; sau đó mới tối ưu giao tiếp hoặc triển khai backend thứ hai.

Các ca kiểm thử bắt buộc:

- Queue rỗng, chưa đầy, đầy, wrap-around; số keys mới lớn hơn capacity.
- ID lặp giữa rank, nhiều caption cùng ảnh; image/text/ID luôn giữ đúng cặp.
- Với cùng đầu vào, thứ tự `(micro-batch, rank, sample)` tái lập chính xác.
- Các shard ghép lại đúng global queue; `ptr`, `count`, version đồng nhất.
- Một rank không có ITM negative, token counts khác nhau, zero valid tokens.
- Accumulation cuối thiếu micro-batch; không enqueue trước boundary.
- Một rank NaN/overflow: mọi rank cùng skip, EMA/queue/scheduler không tiến.
- Loss và gradient của queue sharded/chunked khớp baseline full bank trong tolerance đã định.
- Resume tại boundary khớp queue, teacher, next batch và bước cập nhật tiếp theo.
- Test collectives thật trên nhiều process; mock rank chỉ đủ kiểm tra logic partition.

Điều kiện đồng bộ đúng của sharded queue là: **ghép các shard tái tạo đúng một global FIFO, và mọi query tính loss trên cùng candidate bank logic**. Nội dung queue local giữa các GPU được phép khác nhau vì mỗi rank sở hữu một phần khác nhau.
