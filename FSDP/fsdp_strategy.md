# Stage 1 FSDP: tiến độ hiện tại và công việc cần làm

Cập nhật 2026-10-07. Tài liệu hợp nhất quyết định, tiến độ và kế hoạch FSDP/ZeRO-3; thay thế bản thiết kế sharded queue trước đây.

## Tiến độ hiện tại

Theo hiện trạng được ghi nhận trong hai tài liệu nguồn:

- [x] Dataloader multilevel: seed/epoch, báo lỗi nếu distributed chưa init, chặn loader rỗng, token lengths int64 và valid mask cho padding train/validation; chưa nối vào training engine.
- [x] Kiểm tra module dataloader: 162 tests qua với dữ liệu/tokenizer giả lập trên CPU, gồm 2 workers và process group Gloo thật 2 rank; chưa phải nghiệm thu FSDP/GPU.
- [x] Chốt shard toàn bộ trọng số online model và EMA teacher, kể cả frozen vision, BERT embeddings/LM head; giữ weight tying.
- [x] Chốt MoCo queue replicated: mỗi rank giữ đủ features, IDs, pointer và count giống nhau.
- [x] Lập kế hoạch ownership, global ITC, đồng bộ step/queue, validation, checkpoint và tiêu chí nghiệm thu bên dưới.
- [ ] Triển khai distributed training. Baseline đề xuất là FSDP2 trên một node, một process/GPU; chưa được nghiệm thu trên hai GPU.

## Công việc cần làm theo thứ tự

- [ ] **Runtime và data:** init distributed/device/mesh; nối loader multilevel; đồng bộ sampler, epoch và cấu hình; sửa fingerprints/replay.
- [ ] **Model và EMA:** kiểm chứng ownership, tied weights và nhiều lượt forward trên hai GPU; shard cả online/frozen/teacher; tạo optimizer sau sharding.
- [ ] **Loss, queue và optimizer step:** global ITC keys, counts theo objective, lịch ITM thống nhất, global finite/clip và cùng step/skip; commit queue giống nhau trên mọi rank.
- [ ] **Validation và checkpoint:** loại padding khỏi loss/metrics/candidates; DCP, state/RNG theo rank, kiểm chứng exact resume cùng topology.
- [ ] **Nghiệm thu backbone thật:** đo peak VRAM/throughput từng rank và chạy các kiểm thử ở mục 8 trước khi tối ưu.

Bước tiếp theo là runtime và tích hợp dataloader; chỉ nối training đầy đủ sau khi cổng kiểm chứng model ở mục 4 chạy qua. Các mục dưới đây mô tả yêu cầu triển khai, không phải tính năng đã hoàn thành.

## 1. Quyết định và phạm vi

**Đã chốt:** dataloader multilevel là đầu vào sẵn có; shard toàn bộ trọng số online model và EMA teacher, gồm cả phần frozen; riêng MoCo queue giữ một bản đầy đủ, đồng nhất trên mỗi rank.

“Shard hết” áp dụng cho parameters và training state tương ứng của model. Batch, activations, logits và RNG vẫn riêng từng rank. Scheduler, tiến độ training và metadata nhỏ vẫn replicated nhưng phải nhất quán. Model buffers cần chính sách riêng; không mặc định mọi buffer đều được shard.

**Đề xuất baseline:** FSDP2, một node, một process/GPU, một nhóm data parallel; BF16 compute nếu GPU hỗ trợ, FP32 reduction/loss nhạy số học, queue FP32. Dùng FP32 để kiểm chứng nếu không có BF16. Chưa đưa FP16/scaler, ZeRO-3, offload, tensor/pipeline parallel hay Stage 2 vào lần triển khai đầu.

FSDP2 quản lý parameters, gradients và optimizer state trong phạm vi shard; optimizer được tạo sau sharding. API cần khớp Torch 2.12 được khai báo trong [environment.yml](../environment.yml), đồng thời kiểm tra phiên bản thực tế trước triển khai. Tham khảo [FSDP2 2.12](https://docs.pytorch.org/docs/2.12/distributed.fsdp.fully_shard.html).

## 2. Phân bố trạng thái

| Thành phần | Phân bố dự kiến | Ràng buộc |
| --- | --- | --- |
| Online Q-Former, ITC projections, ITM head, `dec_embedding` | Shard | Một Q-Former dùng chung ITC/ITM/ITG; mỗi parameter có một owner |
| Frozen vision | Shard | Giữ frozen/eval; thu hồi full weights sau forward |
| Frozen BERT embeddings và LM head | Shard | Giữ nguyên `lm_head.weight is embeddings.word_embeddings.weight`; cùng ownership |
| EMA `itc_encoder` | Shard riêng | Cùng mesh, ranh giới và placements với online ITC; không có teacher vision riêng |
| Gradients và optimizer state | Shard theo parameters trainable | Frozen weights không cần gradient/optimizer state |
| Queue image/text/IDs, `ptr`, `count` | Replicated | Mọi rank giữ cùng dữ liệu và cùng thứ tự FIFO |
| Scheduler, epoch, next batch, global step | Replicated | Tất cả cùng step/skip; vị trí dữ liệu vẫn tiến khi bỏ optimizer step |
| Batch, activations, logits, RNG, loader generator | Local | Quản lý memory và lưu RNG/replay theo rank |
| Metrics và checkpoint metadata | Reduce/gather; rank 0 ghi metadata | Các collective vẫn cần mọi rank tham gia |

Với `K=4096`, `Q=128`, `D=256`, queue FP32 dùng khoảng **516 MiB/GPU** chưa tính IDs và tensor tạm. Shard model không giảm phần này. `get()` khi queue đầy tạo bản sao FIFO; loss còn nối/cast bank và tạo logits. Cần đo cả peak của bank, pending keys và backward.

MoCo queue hoàn toàn có thể shard bằng logic của ứng dụng. Chọn replicated là đánh đổi VRAM để đơn giản hóa và giảm giao tiếp lúc đọc, không phải giới hạn về khả năng shard queue.

| Cách phân bố queue | Ghi keys mới | Đọc để tính ITC trên đủ candidates |
| --- | --- | --- |
| Replicated, kế hoạch hiện tại | Gather keys mới; mỗi rank cập nhật cùng FIFO | Đọc queue local, không trao đổi payload queue cũ |
| Sharded, gather bank | Phân phối records theo owner, đồng bộ pointer/count | Gather toàn bank; tốn giao tiếp và lại materialize full queue trên mỗi rank |
| Sharded, tính loss phân tán | Như trên | Truyền candidate chunks hoặc phân phối queries rồi reduce thống kê softmax/gradient; tránh full bank nhưng phức tạp hơn |

Chi phí thêm chủ yếu ở việc **đọc toàn bank logic và backward**, không phải đồng bộ `ptr/count`. Với cách gather bank đơn giản, payload đọc tỷ lệ với capacity `K`, trong khi replicated chỉ trao đổi keys mới tỷ lệ với global micro-batch. Mức chênh thực tế phụ thuộc kích thước tensors, số rank, interconnect và thuật toán; không mặc định mọi cách shard đều phải truyền cả queue mỗi bước. Không thể dùng softmax độc lập trên mỗi shard rồi lấy trung bình loss mà giữ nguyên objective.

## 3. Hiện trạng và nơi cần sửa

| Nơi sửa | Hiện trạng | Thay đổi cần làm |
| --- | --- | --- |
| [engine.py](../src/trainingph1/engine.py): setup | Dùng một `DEVICE`, chưa init distributed, tạo optimizer trên model thường | Init process group/device/mesh; dựng online và EMA trước shard; shard rồi tạo optimizer |
| `engine.py`: data | Import loader cũ, chưa truyền các tham số của dataset multilevel | Nối [dataloaderph1_multilevel.py](../src/data/dataloaderph1_multilevel.py), truyền tokenizer/max length/distributed và cấu hình sampler |
| `engine.py`: fingerprints/replay | Cache paths cố định `train.sqlite`, `validation.sqlite`; replay chưa gọi epoch helper | Lấy đường dẫn từ `loader.dataset.cache_path`; gọi `set_dataloader_epoch`; quản lý generator/RNG theo rank |
| [model.py](../src/trainingph1/model.py), [vision.py](../src/vision.py), [qformer.py](../src/qformer.py) | Vision đang `mock=True`; tied weights; Q-Former gọi qua nhiều đường | Thiết kế owner và đường forward trước khi shard; cho phép chọn backbone thật để nghiệm thu |
| [ema.py](../src/queue/ema.py) | Deepcopy, cập nhật parameter/buffer trực tiếp | Copy trước shard; validate layout và cập nhật shard tương ứng sau step |
| [lossph1.py](../src/losses/lossph1.py) | Teacher features vừa là queries vừa là keys; trả mean local | Tách queries local/candidates global; trả sum và valid count cho từng objective |
| `engine.py`: loss/step | Gather chưa có; ITM có thể bỏ nhánh; finite/clip local | Lịch collective thống nhất; global counts, global finite/clip; gather một lần/micro-batch |
| [moco.py](../src/queue/moco.py) và engine | FIFO local; pending keys enqueue sau step | Giữ FIFO local đơn giản; engine lo broadcast ban đầu và enqueue global keys giống nhau |
| [checkpoint.py](../src/utils/checkpoint.py), logging | Format 5, file checkpoint và RNG theo một process | Thêm checkpoint distributed, state theo rank; chỉ rank 0 ghi log/metadata |
| [config.py](../src/config.py) | Cấu hình training hiện tại | Thêm cấu hình cần thiết cho runtime, sampler và precision; validate cấu hình trên mọi rank |

Dataloader dùng một `MultiLevelLengthBatchSampler` cho train/validation và cả hai chế độ length-aware/thông thường; không còn sampler validation riêng. `valid_mask` là bool `[B]`. Với distributed và `drop_last=False`, sampler giữ mọi mẫu thật và đệm đủ các batch giữa rank. `drop_last=True` vẫn hỗ trợ bỏ phần dư. `batch_size` là số mẫu mỗi rank, không cần chia hết cho world size. Token lengths dùng int64; distributed chưa init và loader rỗng đều bị từ chối. Engine còn phải nối loader và sử dụng mask; checkpoint/replay để giai đoạn sau theo phạm vi hiện tại.

### Bản đồ sửa theo hàm

| File / hàm | Thay đổi cụ thể |
| --- | --- |
| `engine.py`: `prepare_training`, `run_training` | Init/cleanup process group, device theo local rank; dựng online/EMA, shard, rồi optimizer; chỉ rank 0 tạo run metadata và phân phối run directory |
| `engine.py`: `get_dataloader`, `get_data_fingerprints`, `_training_batches`, `train_one_epoch` | Đổi import sang multilevel; truyền tokenizer/max length/distributed, `LENGTH_MEGA_BATCH_MULT`, `LENGTH_JITTER`, seed đã có; lấy cache path từ dataset; gọi epoch helper trước iterator; lưu/phục hồi generator theo rank |
| `model.py`: `__init__`, `encode_image`, `encode_text`, `forward` | Ownership cho embeddings/LM head tied; bảo đảm mọi đường gọi đi qua hooks của owner; thay hard-code `mock=True` bằng lựa chọn cấu hình |
| `ema.py`: `__init__`, `update` | Copy model chưa shard; đối chiếu online/teacher layouts và cập nhật shard tương ứng; xử lý buffers đúng layout |
| `lossph1.py`: `get_itc_loss` | Tách teacher queries local và candidate keys/IDs global; giữ online queries local; xuất loss sum/count |
| `lossph1.py`: `get_itm_loss`, `get_itg_loss` | Trả sum/count, hỗ trợ valid masks và trường hợp không có phần tử hợp lệ |
| `engine.py`: `hepler_compute_loss`, `_loss_counts` | Gather teacher keys/IDs một lần; truyền inputs loss mới; mọi rank gọi đủ ba lượt ITM; tính valid counts từ masks |
| `engine.py`: `train_one_epoch` | Reduce group counts, global finite/norm/clip, thống nhất step/skip; pending keys đã global; enqueue đúng một lần sau step |
| `engine.py`: `validate`, `_training_events`, `_save_state` | Reduce metrics; mọi rank cùng validation/save; rank 0 log; đồng bộ best metric và lịch checkpoint |
| `checkpoint.py`: `create_run`, `save_checkpoint`, `load_checkpoint`, `load_pretrained` | Phân tách metadata rank 0 và distributed state; lưu/load shards + shared queue + RNG từng rank; giữ đường init weights từ format 5 |
| `dataloaderph1_multilevel.py`: sampler/collator | Dùng chung `valid_mask` cho train/validation; engine/loss/queue cần loại padding khỏi queries, candidates, ITM negatives, counts, metrics và enqueue |
| `moco.py`: `enqueue`, `get` | Giữ FIFO và API đọc local; chỉ thêm validation/empty-input guard cần thiết. Collective nằm ở runtime/engine |

Đề xuất thêm một file `src/utils/distributed.py` nhỏ cho init/cleanup, FSDP shard setup, gather keys, broadcast queue và global reductions. Đây là vị trí dự kiến, chưa tồn tại; không cần tạo class backend tổng quát. `qformer.py` và `vision.py` chỉ sửa nội bộ nếu kiểm chứng cho thấy cần đổi đường forward; việc gắn sharding có thể thực hiện từ setup bên ngoài. `config.py` đã có length jitter/mega-batch và AMP: tái sử dụng các giá trị đó, chỉ thêm lựa chọn thực sự thiếu.

## 4. Shard plan cần kiểm chứng trước khi nối cả pipeline

- Shard từng `QFormerBlock`, rồi Q-Former/ITC encoder và phần parameters còn lại. Kiểm kê parameters theo identity để không bỏ sót hoặc gán trùng owner.
- Shard các block backbone vision thật và module bao ngoài; giữ đường mock cho test nhỏ. Khởi tạo không được đòi full online + teacher + optimizer cùng nằm trên GPU nếu vượt bộ nhớ; dùng CPU/meta initialization thích hợp khi cần.
- Embeddings và LM head cần một owner chung cho tied weight. Hướng ưu tiên là module text frozen sở hữu cả hai, gọi qua forward theo chức năng embedding/projection; phải giữ alias và kiểm chứng trước khi chốt thay đổi cấu trúc.
- LM head frozen vẫn phải truyền gradient từ ITG logits về Q-Former. Không bọc projection ITG bằng `no_grad()` chỉ vì weight frozen.
- Các lệnh `encode_image()`, `encode_text()` và truy cập `qformer_model` phải đi qua module sở hữu weights. Nếu dùng method ngoài `forward`, đăng ký hook thích hợp hoặc đổi đường gọi. Không coi root sharding tự bao phủ mọi helper. Xem [forward hooks của FSDP2](https://docs.pytorch.org/docs/2.12/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.register_fsdp_forward_method).
- Online và teacher là hai bộ trọng số logic riêng. Tạo EMA trước shard; kiểm tra tên, shape, dtype, mesh và placements tương ứng. Sau step: `teacher_shard = m * teacher_shard + (1 - m) * online_shard`.
- Teacher và vision chạy không có backward: đặt reshard sau forward rõ ràng, kể cả root. Kiểm kê buffers, khởi tạo nhất quán và copy EMA buffers theo layout thực tế.

**Cổng kiểm chứng đầu tiên:** model nhỏ chạy ITC, ba lượt ITM, ITG rồi một backward; Q-Former dùng chung nhiều lượt vẫn đúng; tied weights còn nguyên; ITG truyền gradient; teacher EMA khớp tham chiếu. Chỉ nối training đầy đủ sau khi bài kiểm tra này qua trên hai GPU.

## 5. Hợp đồng đồng bộ MoCo queue

Một slot là `(image_features[Q,D], text_features[D], image_id)`. Hai bank và IDs luôn được cập nhật cùng nhau. `K` là capacity logic chung; mỗi rank chứa đủ `K`, không tăng capacity thành `K * world_size`. Tổng storage vật lý là `world_size` bản queue; không dùng `owner_rank/local_slot`, queue payload shards, remap shards khi resume hay distributed softmax do chia queue.

1. Khởi tạo cùng cấu hình queue; broadcast cả năm buffers từ rank 0 khi bắt đầu/resume.
2. Mỗi rank tính teacher keys detached của micro-batch local.
3. All-gather image keys, text keys, IDs và `valid_mask` theo cùng thứ tự rank. Loại keys đệm khỏi candidate bank và enqueue. Sampler đảm bảo cùng batch size giữa rank; sequence length local có thể khác vì keys đã có shape cố định.
4. Dùng global keys cho ITC hiện tại và giữ trong `pending_keys`. Không gather lại khi enqueue.
5. Chỉ khi mọi rank thống nhất optimizer step thành công mới enqueue theo thứ tự `(micro-batch, rank, sample)`. Keys được tính trước lần cập nhật EMA của step này.
6. Nếu skip, mọi rank bỏ pending keys; queue, EMA, scheduler và global step giữ nguyên. Validation không ghi queue.

Trong cả accumulation group, queue bất biến đến khi tất cả backward hoàn tất. Không all-reduce trung bình features, không broadcast full queue mỗi step, không deduplicate theo image ID: nhiều caption/cùng ảnh vẫn là các records hợp lệ.

Với `G` records mới, mọi rank thực hiện cùng một phép cập nhật:

```text
ptr_new   = (ptr_old + G) % K
count_new = min(K, count_old + G)
```

Nếu `G > K`, giữ `K` records cuối nhưng pointer tiến theo toàn bộ `G`; `G == 0` là no-op. Validate `K > 0`. FIFO hiện tại có thể giữ nguyên thuật toán chính; bổ sung guard nếu cần cho input rỗng và shape không hợp lệ.

Điều kiện đúng: `image`, `text`, `image_ids`, `ptr`, `count` bằng nhau giữa rank sau mỗi commit. Test so toàn bộ buffers; debug có thể kiểm tra checksum và metadata định kỳ, lỗi thì dừng để điều tra thay vì âm thầm ghi đè. Queue giống nhau ban đầu và cùng chuỗi cập nhật là cơ chế giữ đồng bộ.

Ví dụ hai rank, hai micro-batch: mỗi bản queue nhận `[m0/r0, m0/r1, m1/r0, m1/r1]` một lần sau optimizer step thành công.

## 6. Loss và optimizer boundary

Đồng bộ queue không bắt buộc phải đổi current-batch candidates. **Kế hoạch này đề xuất global current candidates cho ITC** để mỗi query thấy keys từ mọi rank ngay trong micro-batch hiện tại:

```text
online queries  = features local, có gradient
teacher queries = teacher features local, detached
candidate bank  = teacher keys global của micro-batch hiện tại + queue cũ
positive mask   = local image IDs so với global candidate IDs và queue IDs
```

Phải tách hai vai trò teacher queries/keys trong API criterion; giữ queries local khi tạo pseudo targets. Hard targets chuẩn hóa theo tất cả positives cùng ID. Gather teacher keys không cần autograd. ITC này mở rộng candidates so với code local hiện tại; phép đối chiếu phải dùng cùng global bank.

ITM baseline vẫn lấy negatives trong batch local. Mọi rank chạy đủ ba lượt ITM theo cùng thứ tự. Rank không có negative dùng dummy pairs và loss/count bằng 0 nhưng giữ graph; không bỏ forward/backward. ITG đếm các labels hợp lệ sau shift, bỏ padding và `-100`.

Mỗi objective trả `local_loss_sum` và `valid_count`: ITC đếm samples (sum của trung bình hai chiều); ITM đếm pairs hợp lệ; ITG đếm tokens hợp lệ. Với gradient được backend trung bình trên `W` rank:

```text
global_count = tổng valid_count của objective trên mọi rank, mọi micro-batch trong group
backward_loss = objective_weight * W * local_loss_sum / global_count
```

Không chia thêm accumulation steps. Nếu global count bằng 0, dùng zero loss có graph. Kiểm chứng quy ước reduction thực tế của backend. Nhóm accumulation cuối ngắn hơn vẫn dùng counts thực tế.

```text
Lấy accumulation group → all-reduce counts → zero_grad
  Mỗi micro-batch:
    online/teacher forward → gather keys/IDs
    ITC + ba lượt ITM + ITG → scaled loss → backward
    giữ global keys pending
All-reduce finite flag → global gradient norm/clip → thống nhất step/skip
  Thành công: optimizer → scheduler → EMA → enqueue → global_step += 1
  Skip: bỏ gradients và pending keys
Cập nhật next_batch → validation/checkpoint theo cùng lịch ở mọi rank
```

Baseline đồng bộ gradients mỗi backward. Clip phải tính norm toàn bộ parameters duy nhất trên mọi shard; không lấy norm shard local thay cho norm model. Nếu thêm FP16 sau này, overflow và scaler update phải đồng bộ trước khi bất kỳ rank nào optimizer step.

Accumulation không gộp các micro-batch thành một contrastive batch: mỗi micro-batch dùng global keys hiện tại và queue cũ. Với local batch `B`, `W` rank, `A` micro-batch, một step không có padding thêm `W * B * A` records; step có padding chỉ thêm tổng số records hợp lệ; giữ nguyên `K` khi tăng `W` làm queue thay mới nhanh hơn.

## 7. Data, validation và resume

**Checkpoint và exact resume tạm hoãn.** Nội dung thiết kế bên dưới được giữ cho giai đoạn sau; không thuộc đợt cập nhật dataloader này.

Dựng process group/device trước dataset/loader. Dùng chung sampler seed, lengths, cấu hình và epoch; gọi `set_dataloader_epoch(loader, epoch)` trước iterator, kể cả khi resume. Để giữ mọi mẫu training, truyền `drop_last=False` (config hiện vẫn đặt `DROP_LAST=True`); sampler đệm các slots thiếu để mọi rank có cùng số batch và batch size. Vẫn cho phép `drop_last=True` để bỏ phần dư, từ chối loader rỗng. Seed khởi tạo model giống nhau; RNG augmentation/dropout và generator được quản lý theo rank. Cache cần đường dẫn chung cho baseline một node.

Train và validation distributed dùng chung một sampler. Với `drop_last=False`, sampler phát `(index, valid)`; mẫu thật có `valid=True`, mẫu lặp để đệm có `valid=False`. Dataset/collator đưa cờ này vào `batch["valid_mask"]`; không deduplicate theo image ID. Mọi rank nhận batch đủ `batch_size`, kể cả rank chỉ có padding; dataset rỗng bị từ chối. Single-process giữ batch cuối ngắn khi `drop_last=False`, mask toàn `True`.

Engine/loss phải giữ cùng lịch forward/collective và loại slots đệm khỏi queries, current candidate bank, ITM negatives, loss counts và metrics. Khi gather keys phải gather kèm mask; queue chỉ enqueue keys thật, giữ thứ tự `(micro-batch, rank, sample)`. Rank chỉ có padding vẫn tham gia nhưng loss/count bằng 0. Các tính năng mask trong engine/loss/queue chưa được nối. Validation không cập nhật EMA/queue; reduce sum/count và chỉ rank 0 ghi kết quả. Ví dụ 19 mẫu, 2 GPU, batch 4/GPU: 3 batch/rank, tổng 24 slots = 19 thật + 5 đệm.


Checkpoint mới dùng [Distributed Checkpoint 2.12](https://docs.pytorch.org/docs/2.12/distributed.checkpoint.html) cho online/optimizer/EMA sharded; mọi rank tham gia. Lưu thêm:

| Phạm vi | Nội dung |
| --- | --- |
| Một bản chung | Full queue, scheduler, EMA momentum, epoch/next batch/global step, config, tokenizer, fingerprints, topology |
| Theo rank | Python/NumPy/Torch CPU/CUDA RNG, loader generator và trạng thái replay/worker cần thiết |

Chỉ save sau optimizer boundary hoàn chỉnh, không còn pending keys hoặc gradients. Mọi rank hoàn tất ghi rồi rank 0 mới đánh dấu checkpoint hợp lệ/cập nhật `last`. Load model/optimizer/EMA shards, phân phối shared state, broadcast queue, khôi phục local RNG/data state trước bước tiếp theo.

Baseline exact resume yêu cầu cùng world size, local batch, accumulation, sampler, dữ liệu và môi trường. DCP reshard weights không tự bảo toàn chuỗi batch/RNG. Giữ đường đọc format 5 để khởi tạo weights; không tuyên bố exact distributed resume từ checkpoint single-process. Sửa metadata/fingerprints để dùng tên cache multilevel thực tế; kiểm thử replay với workers riêng.

## 8. Thứ tự triển khai và nghiệm thu

| Bước | Công việc | Điều kiện qua |
| --- | --- | --- |
| 1. Runtime + data | Init distributed, chọn device, nối multilevel loader, epoch/seeds/fingerprints | Hai process nhận partition đúng, cùng số batch, resume sampler đúng |
| 2. Model + EMA | Spike ownership/tied weights/nhiều forward; shard online, frozen modules và teacher; optimizer sau shard | Hai GPU chạy đủ objectives/backward; gradient ITG và EMA khớp tham chiếu |
| 3. Global loss + queue + step | Tách ITC inputs, gather keys, ITM schedule, counts, finite/clip, queue commit | Loss/gradient khớp baseline; mọi rank cùng step/skip; queue hoàn toàn giống nhau |
| 4. Validation + checkpoint | Padding masks, sum/count, rank-0 logging, DCP và RNG theo rank | Validation không sửa training state; resume khớp batch và update kế tiếp |
| 5. Backbone thật + đo memory | Kiểm chứng toàn bộ pipeline trên vision thật, đo peak từng rank/throughput | Đúng số học và nằm trong VRAM mục tiêu; chỉ sau đó tối ưu communication/activation memory |

Các test cần có khi triển khai:

- Queue rỗng/chưa đầy/đầy/wrap-around, `G > K`, `G == 0`, ID lặp xuyên rank; cùng thứ tự record và không enqueue hai lần.
- Hai process thật cho gather/broadcast; ít nhất hai GPU cho FSDP, không chỉ mock rank.
- ITC local queries/global keys có pseudo targets; so loss/gradient với model nhỏ cùng bank và cùng phạm vi ITM negatives.
- Một rank không có ITM negative, số tokens khác nhau, zero valid tokens, accumulation group cuối ngắn.
- NaN/Inf ở một rank: tất cả cùng skip; queue/EMA/scheduler/global step không tiến.
- Tied weights, gradient qua frozen LM head, EMA shard đúng, full weights được thu hồi theo policy.
- Validation size không chia hết world size, rank chỉ có padding; metrics/candidates loại đúng phần đệm.
- Save/resume tại boundary, gồm ngay sau skip: so queue, teacher, optimizer, RNG, next batch và update tiếp theo.

Chốt module boundaries và kích thước chạy qua bước 2/5 dựa trên kiểm chứng thực tế; không giảm phạm vi frozen sharding hoặc chuyển sang sharded queue một cách ngầm định. ZeRO-3 là hướng backend thay thế về sau, dùng cùng hợp đồng queue/loss; không triển khai song song hai backend trong đợt này.
