# Kế hoạch FSDP cho stage 2–3

Cập nhật: 09/10/2026. Đây là kế hoạch thiết kế, chưa xác nhận engine multimodal, distributed checkpoint hoặc QLoRA nhiều GPU đã được triển khai và nghiệm thu.

## 1. Phạm vi và các quyết định chưa chốt

Mục tiêu là training mô hình trong `src/trainingph2/model.py` với sharding parameters, gradients và optimizer states, tương đương phạm vi bộ nhớ của ZeRO-3.

Hướng ưu tiên là FSDP2 với `reshard_after_forward=True`, phù hợp engine stage 1 hiện tại. DeepSpeed ZeRO-3 là backend khác; không bật đồng thời hai backend cho cùng mô hình.

Hai giả định cần xác nhận trước khi chốt cấu hình:

- Stage 2 học căn chỉnh ảnh–ngôn ngữ; stage 3 instruction finetuning ảnh–text.
- Số GPU, loại GPU, VRAM mỗi GPU và kết nối giữa GPU chưa được xác định.

Stage 2 và stage 3 dùng chung một engine multimodal. Khác biệt nằm ở dữ liệu, nhóm tham số được học, learning rate và lịch đánh giá. Không sao chép toàn bộ engine cho từng stage.

## 2. Hiện trạng

| Thành phần | Đã có | Cần bổ sung |
| --- | --- | --- |
| Mô hình | `ModelStage2`: DINOv3 → instruction-aware Q-Former → projector → Qwen | Khởi tạo và freeze policy phù hợp FSDP |
| LLM | Qwen3-4B, LoRA/QLoRA, visual prefix, generation | Nghiệm thu sharding thực của weights 4-bit |
| Dữ liệu | `ChatCollator`, `InstructCollator` | Dataset, cache độ dài, distributed sampler, valid mask và replay |
| Training | `trainingph2/finetune.py` chạy text SFT bằng Trainer | Engine training ảnh–text nhiều GPU |
| Loss | Mean causal LM loss trên batch local | Loss sum và target-token count toàn cục |
| Logging | Console/file logger và TensorBoard của stage 1 | Diagnostics cho Q-Former/projector/LoRA và causal LM |
| Checkpoint | Export Q-Former, adapter, projector và preprocessing | Distributed training checkpoint đầy đủ |
| Inference | `generation(image, prompt)` và tensor API `generate()` | Generation an toàn trong distributed validation |

Có thể dùng lại cơ chế runtime, accumulation boundary, global skip, RNG replay và công bố checkpoint nguyên tử của stage 1. Không dùng nguyên xi checkpoint stage 1 vì schema hiện gắn với `ModelStage1`, EMA và queue.

## 3. Chính sách training đề xuất

| Thành phần | Stage 2: căn chỉnh | Stage 3: instruction finetuning |
| --- | --- | --- |
| DINOv3 | Frozen | Frozen |
| BERT embeddings của Q-Former | Frozen | Frozen |
| Q-Former | Học; có thể warmup projector trước | Học với LR thấp hơn hoặc freeze theo ablation |
| Projector | Học | Tiếp tục học |
| Qwen base weights | Frozen | Frozen |
| LoRA của Qwen | Tạm tắt | Bật QLoRA |
| Dữ liệu | Ảnh–caption/câu trả lời mô tả có prompt | Ảnh–instruction–answer đa dạng |
| Objective | Answer-token cross entropy | Answer-token cross entropy |

Projector mới khởi tạo cần học biểu diễn phù hợp với Qwen. Baseline stage 2 ổn định đầu vào thị giác trước khi stage 3 điều chỉnh khả năng trả lời của LLM. Nếu quyết định bật QLoRA ngay stage 2, engine vẫn hỗ trợ bằng cấu hình; cần xác định khác biệt về dữ liệu hoặc mục tiêu của stage 3.

Freeze Qwen bằng `requires_grad=False`, không bọc toàn bộ forward Qwen trong `no_grad()`: backward qua LLM vẫn cần thiết để học projector và Q-Former. Vision và BERT embeddings frozen giữ ở eval; Q-Former frozen cũng giữ ở eval.

Trainable policy của Q-Former, projector và adapter phải độc lập. API hiện tại cần được điều chỉnh để hỗ trợ stage 2 học projector/Q-Former khi không học adapter.

Chuyển stage 2 → stage 3 tạo run mới từ weights, khởi tạo lại optimizer/scheduler và lưu nguồn checkpoint. Resume trong cùng stage khôi phục toàn bộ training state.

## 4. Nghiệm thu FSDP + QLoRA trước engine

`languagemodel.py` hiện nạp Qwen lên GPU của từng rank trước khi shard và chưa đặt `bnb_4bit_quant_storage`. Vì vậy chưa thể xem cấu hình hiện tại là QLoRA full sharding.

Thiết lập cần kiểm chứng:

- NF4 4-bit và double quantization.
- `bnb_4bit_compute_dtype=torch.bfloat16` khi phần cứng hỗ trợ.
- `bnb_4bit_quant_storage` là floating dtype phù hợp với đường FSDP-QLoRA; dự kiến BF16.
- Dtype của quantized storage, non-quantized weights và mixed-precision policy phải tương thích.
- LoRA gắn trước sharding; optimizer tạo sau sharding.
- Khởi tạo weights và adapter đồng nhất giữa rank; sau đó dùng RNG training theo rank.

Thử nghiệm đầu tiên dùng Qwen nhỏ trên hai GPU:

1. Nạp NF4, gắn LoRA và shard model.
2. Xác minh base Qwen weights thực sự được shard, không chỉ adapter.
3. So sánh loss, gradient và optimizer update với bản không shard trên cùng global batch.
4. Lưu/nạp weights, quantization metadata và optimizer; kiểm tra bước cập nhật kế tiếp.
5. Đo peak VRAM cả lúc load lẫn training.

Không coi việc đưa toàn bộ `Params4bit` vào ignored parameters là đạt yêu cầu full sharding. Cần kiểm tra kết quả số học và trạng thái quantization, không chỉ kiểm tra chương trình không crash.

Nếu FSDP2 + NF4 không vượt qua nghiệm thu trên phiên bản thư viện đang dùng, phải quyết định rõ giữa đường FSDP `FULL_SHARD` hỗ trợ QLoRA và FSDP2 với BF16 LoRA. Không tự đổi backend hoặc bỏ quantization mà không cập nhật quyết định thiết kế.

## 5. Sharding, khởi tạo và bộ nhớ

Đơn vị sharding dự kiến:

- Từng transformer block của DINOv3.
- Từng block của Q-Former.
- Từng decoder layer của Qwen.
- Embeddings, LM head và projector trong các nhóm phù hợp; kiểm tra ownership nếu có shared weights.
- Root model quản lý các tham số còn lại.

Không chỉ wrap toàn bộ mô hình thành một nhóm lớn. Cần kiểm tra các đường gọi embeddings trực tiếp, `encode_image()` và `generate()` để không truy cập tham số sharded ngoài phạm vi unshard hợp lệ. Quy tắc wrap của FSDP1 và FSDP2 không được dùng thay thế lẫn nhau.

Tách dựng kiến trúc/nạp weights khỏi việc đưa toàn model lên GPU. Kiểm chứng chiến lược load tiết kiệm RAM/VRAM riêng cho quantized model; không giả định meta initialization và broadcast hoạt động giống model thông thường.

Optimizer có nhóm riêng cho Q-Former, projector và LoRA. Dùng AdamW chuẩn cho baseline; paged AdamW 8-bit chỉ thêm sau khi kiểm chứng tương thích với tham số sharded và checkpoint.

Biện pháp kiểm soát bộ nhớ:

- BF16 khi phần cứng hỗ trợ, gradient reduction FP32 ở baseline nếu backend cho phép.
- Activation checkpointing theo block, bảo toàn RNG dropout; tránh bật chồng hai cơ chế checkpointing.
- `use_cache=False` khi training.
- Giới hạn tổng visual prefix + prompt + answer trong context Qwen.
- Theo dõi activation, logits và communication buffers, không chỉ parameter memory.
- Cân nhắc loss theo chunk nếu logits `[B, Q+T, vocab_size]` là nút thắt; phải đối chiếu loss/gradient trước khi bật.
- Bắt đầu accumulation với gradient synchronization đơn giản, đúng đắn; chỉ tối ưu lịch communication sau khi có số đo bộ nhớ và throughput.

## 6. Dataset, sampler và loss

Batch gồm:

```text
images
qformer_input_ids, qformer_attention_mask
input_ids, attention_mask, labels
valid_mask
```

Q-Former chỉ nhận prompt. Qwen nhận visual prefix và prompt + answer theo chat template. Labels bỏ qua visual prefix, prompt và padding; chỉ answer và EOS đóng góp vào loss.

Dataset/sampler cần:

- Số micro-batch bằng nhau giữa các rank.
- Padding mẫu bằng `valid_mask` khi không drop phần dư.
- Length-aware batching theo chuỗi Qwen; theo dõi riêng instruction length của Q-Former.
- Kiểm tra dữ liệu hỏng trước training, tránh một rank tự bỏ mẫu và làm lệch lịch collective.
- Fingerprint manifest, tokenizer, chat template, preprocessing và truncation policy.
- Replay sampler và augmentation khi resume giữa epoch, kể cả khi có workers.
- Augmentation phù hợp tác vụ: VQA/OCR không mặc định dùng crop/flip có thể làm mất hoặc đổi thông tin câu trả lời.

Loss chuẩn hóa theo toàn bộ target tokens trong mọi rank và accumulation group:

```text
S[r,m] = tổng cross entropy hợp lệ ở rank r, micro-batch m
N[r,m] = số target tokens hợp lệ sau causal shift
N_global = tổng N[r,m] của toàn accumulation group
L_global = tổng S[r,m] / N_global
```

Với cơ chế average gradient theo world size, mỗi micro-batch backward dùng `world_size * S_local / N_global`. Không lấy trung bình các mean loss local, và không chia thêm accumulation steps sau khi đã chuẩn hóa bằng token count của cả group.

Rank chỉ có mẫu đệm vẫn thực hiện đủ forward/backward với zero loss nối computation graph. `_inputs()` hiện từ chối batch không có supervised target; cần thiết kế lại đường loss để hỗ trợ trường hợp này và tránh NaN từ mean loss trên toàn labels `-100`.

Nếu cả group không có token hợp lệ hoặc bất kỳ rank nào có nonfinite loss/gradient, mọi rank thống nhất bỏ qua update. Scheduler và successful global step chỉ tăng sau update thành công.

Stage 2–3 baseline không dùng EMA, MoCo queue hoặc ba loss ITC/ITM/ITG của stage 1.

## 7. Engine chung

Luồng thực thi:

```text
init runtime/process group
→ kiểm tra config đồng nhất giữa rank
→ dựng model, nạp nguồn weights, đặt trainable policy
→ shard model
→ tạo optimizer/scheduler
→ khôi phục training state nếu resume
→ train theo accumulation group
→ validation/checkpoint theo successful optimizer step
→ export inference khi kết thúc
```

Tổ chức mã dự kiến, giữ số module tối thiểu:

| Vị trí | Trách nhiệm |
| --- | --- |
| `trainingph2/model.py` | Mô hình chung cho stage 2–3, forward và inference |
| `trainingph2/engine_fsdp.py` | Engine multimodal dùng chung, sharding và lịch sự kiện |
| Cấu hình stage 2/3 | Data paths, freeze policy, optimizer groups và lịch training |
| `data/` và `trainingph2/data.py` | Dataset/sampler và collator |
| `utils/` | Distributed checkpoint và logging/diagnostics dùng chung |

Stage 3 chỉ cần entry point/config mỏng nếu cùng objective causal LM. Giữ nguyên engine stage 1; chỉ tách helper chung khi có nhu cầu tái sử dụng thực tế. `trainingph2/finetune.py` hiện tại vẫn là text SFT, không coi đó là engine multimodal.

## 8. Logging và validation

Tái sử dụng console/file logger và TensorBoard writer, thay diagnostics đặc thù stage 1 bằng:

| Nhóm | Metrics |
| --- | --- |
| Chất lượng | Answer CE, perplexity, teacher-forced token accuracy |
| Gradient | Norm Q-Former/projector/LoRA, global norm, clip scale |
| Dữ liệu | Valid samples, target tokens, padding ratio, truncation ratio |
| Hiệu năng | Tokens/s, samples/s, data time, step time, peak VRAM lớn nhất giữa rank |
| Độ ổn định | Nonfinite loss/gradient, skipped steps, LR từng nhóm |
| Generation | Ảnh, prompt, target và câu trả lời sinh thực |

Metrics tổng hợp bằng numerator/denominator toàn cục. Chỉ rank 0 ghi TensorBoard; mỗi rank ghi `logs/rank-N.log`. Resume phải xử lý event vượt checkpoint để tránh lịch sử metrics trùng hoặc sai step.

Validation có hai nhịp:

- Thường xuyên: teacher-forced loss trên validation loader, không làm đổi RNG/model/optimizer state của training.
- Thưa hơn: generation trên tập cố định, đo metric theo tác vụ và quan sát câu trả lời thực.

Teacher-forced accuracy không thay thế chất lượng generation. Theo dõi thêm độ dài câu trả lời, tỷ lệ EOS và mức phụ thuộc vào ảnh qua ablation ảnh đúng/ảnh tráo.

Không gọi `generation()` riêng ở rank 0 trên model vẫn sharded. Hai phương án hợp lệ là generation có tất cả rank tham gia với lịch đồng bộ, hoặc inference bằng model riêng từ export. Phải xử lý EOS và số batch khác nhau để tránh lệch collective. Chỉ rank 0 ghi kết quả sau khi tổng hợp.

## 9. Training checkpoint và inference export

### Training checkpoint

Checkpoint phải lưu hoặc tham chiếu đủ để khôi phục chính xác:

- Weights được cập nhật; nguồn frozen vision/BERT/Qwen bất biến, có revision/digest.
- Quantization state nếu dùng QLoRA; không giả định re-quantize lại base sẽ cho exact resume.
- Sharded optimizer state và scheduler.
- Epoch, next batch, successful global step và số target tokens đã xử lý.
- RNG Python, NumPy, PyTorch và CUDA theo rank.
- Sampler/loader state, augmentation replay và metrics đang tích lũy.
- Trainable parameter names, optimizer parameter groups, LoRA và quantization config.
- Hai tokenizer, chat template, image preprocessing và dataset fingerprints.
- Phiên bản thư viện, backend, world size và checkpoint format version.

Baseline ưu tiên tính đúng đắn của distributed state dict và quantization state trước khi tối ưu checkpoint chỉ lưu phần trainable. Không tự động gom toàn bộ base model lên một GPU để lưu.

Tái sử dụng nguyên tắc của stage 1:

1. Chỉ lưu sau optimizer boundary và `zero_grad`.
2. Mọi rank tham gia thao tác collective cần thiết.
3. Ghi snapshot mới, kiểm tra mọi rank hoàn tất rồi mới công bố `last`/`best`.
4. Lỗi ghi file không thay pointer của checkpoint tốt trước đó.
5. Ban đầu yêu cầu resume cùng world size, backend và cấu hình tương thích. Reshard khi đổi world size là hạng mục riêng; không hứa exact replay khi cách chia dữ liệu thay đổi.

Schema cần độc lập với EMA/queue và kiến trúc stage 1. Phân biệt rõ khởi tạo stage mới từ weights với resume training state.

### Inference export

Export gom đúng Q-Former/projector/adapter từ shards, kèm tokenizer, preprocessing và manifest tham chiếu frozen base. Tất cả rank tham gia gather cần thiết, chỉ rank 0 ghi artifact cuối.

Không gọi trực tiếp `save_pretrained()`/`torch.save(state_dict())` hiện tại trên model đã sharded rồi coi kết quả là export hoàn chỉnh. Phải kiểm tra logits và generation sau export trên model không shard.

## 10. Trình tự triển khai và nghiệm thu

| Bước | Điều kiện hoàn tất |
| --- | --- |
| 1. Backend | NF4 + sharding + backward + checkpoint chạy đúng trên hai GPU |
| 2. Dữ liệu/loss | Masking, token normalization và accumulation khớp reference |
| 3. Engine | Trainable policy, optimizer groups, clipping, global skip và validation đúng |
| 4. Checkpoint | Resume giữa epoch khớp chạy liên tục trong môi trường kiểm thử xác định; lỗi save không mất checkpoint trước |
| 5. Logging/generation | Metrics đúng toàn cục, generation không deadlock |
| 6. Stage 2 pilot | Overfit tập nhỏ, đo VRAM/throughput, kiểm tra ảnh tác động đến câu trả lời |
| 7. Stage 3 pilot | Khởi tạo đúng weights stage 2, reset optimizer, đánh giá instruction following |

Các test trọng tâm:

- Một GPU reference so với hai rank: loss, gradients và optimizer update.
- Độ dài answer khác nhau, batch dư, một rank chỉ có padding và toàn group không có target.
- Frozen weights không đổi; gradients tới đúng Q-Former/projector/LoRA theo policy.
- Prompt thay đổi ảnh hưởng query features; answer không lọt vào Q-Former.
- Activation checkpointing không làm mất gradient hoặc thay đổi semantics dropout ngoài sai số cho phép.
- Nonfinite trên một rank làm mọi rank skip cùng step.
- Interrupted/resumed run so với uninterrupted run, gồm workers và augmentation.
- QLoRA quantization state được giữ qua save/load.
- Generation có EOS khác nhau giữa rank không treo.
- Export rồi nạp inference cho logits/generation tương đương.

Chỉ chốt batch size, accumulation, sequence length, offload và learning rate sau khi có phần cứng đích và số đo pilot. Ưu tiên nghiệm thu backend trước, rồi mới viết hoàn chỉnh engine/logging/checkpoint trên backend đó.

## 11. Tài liệu đối chiếu

- [PyTorch FSDP2: sharding, optimizer và state dict](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html).
- [Transformers FSDP2: wrapping, loading và activation checkpointing](https://huggingface.co/docs/transformers/fsdp).
- [bitsandbytes FSDP-QLoRA: floating quantization storage](https://huggingface.co/docs/bitsandbytes/fsdp_qlora).
- [PEFT với FSDP và QLoRA](https://huggingface.co/docs/peft/accelerate/fsdp); phân biệt hướng dẫn FSDP1 với API FSDP2 của repo.
- [Accelerate issue về FSDP2 và Params4bit storage mặc định](https://github.com/huggingface/accelerate/issues/3983); đây là báo cáo trên phiên bản cụ thể, không thay thế nghiệm thu stack hiện tại.
