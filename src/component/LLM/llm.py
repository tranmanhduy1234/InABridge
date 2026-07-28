from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class LLMTrainMode(str, Enum):
    FROZEN = "frozen"  # Toàn bộ LLM đóng băng — chỉ các module khác (projector, ...) được train
    LORA = "lora"       # LoRA adapters thêm vào attention / MLP, base weights đóng băng
    QLORA = "qlora"     # Base weights quantize 4-bit (NF4), LoRA adapters train bình thường

@dataclass
class LoRAConfig:
    r: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    target_modules: Optional[list[str]] = None
    bias: str = "none"

    def __post_init__(self):
        if self.r <= 0:
            raise ValueError(f"LoRA rank phải > 0, nhận được r={self.r}.")
        if self.alpha <= 0:
            raise ValueError(f"LoRA alpha phải > 0, nhận được alpha={self.alpha}.")
        if self.bias not in ("none", "all", "lora_only"):
            raise ValueError(
                f"bias phải là 'none' | 'all' | 'lora_only', nhận được '{self.bias}'."
            )

    @property
    def scaling(self) -> float:
        return self.alpha / self.r

    def resolve_target_modules(self) -> list[str]:
        if self.target_modules is not None:
            return self.target_modules
        # Default target cho hầu hết LLaMA / Qwen / Mistral architecture
        return [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]


# ---------------------------------------------------------------------------
# LanguageModelDecoder
# ---------------------------------------------------------------------------

class LanguageModelDecoder(nn.Module):
    """
    Wrapper LLM decoder cho pipeline VLM.

    Nhận visual token embeddings (đã được project sang llm_dim) cùng
    với text token embeddings, rồi chạy forward qua causal LM.

    Các chế độ train:
        FROZEN — toàn bộ LLM đóng băng.
        LORA   — inject LoRA adapter (peft), base weights đóng băng.
        QLORA  — load 4-bit NF4 (bitsandbytes) + LoRA adapter.

    Args:
        model_id:       HuggingFace model ID.
        train_mode:     Chế độ training khởi tạo.
        lora_config:    Cấu hình LoRA (bắt buộc với LORA / QLORA).
        torch_dtype:    dtype cho base model (mặc định bfloat16).
        device_map:     Truyền thẳng vào from_pretrained (mặc định "auto").
    """

    def __init__(
        self,
        model_id: str,
        train_mode: LLMTrainMode = LLMTrainMode.FROZEN,
        lora_config: Optional[LoRAConfig] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
    ):
        super().__init__()

        self.model_id = model_id
        self.train_mode = train_mode
        self.torch_dtype = torch_dtype
        self._lora_config = lora_config
        self._lora_applied = False

        self.model, self.tokenizer = self._load_base_model(
            model_id, train_mode, torch_dtype, device_map
        )

        self.set_train_mode(train_mode, lora_config=lora_config)

    # ------------------------------------------------------------------
    # Load helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_bnb_config() -> BitsAndBytesConfig:
        """4-bit NF4 config cho QLoRA."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    def _load_base_model(
        self,
        model_id: str,
        train_mode: LLMTrainMode,
        torch_dtype: torch.dtype,
        device_map: str,
    ):
        kwargs: dict = {
            "pretrained_model_name_or_path": model_id,
            "device_map": device_map,
        }

        if train_mode == LLMTrainMode.QLORA:
            if device_map is None:
                raise ValueError(
                    "QLORA yêu cầu device_map hợp lệ (vd: 'auto') vì "
                    "bitsandbytes 4-bit không hỗ trợ device_map=None."
                )
            kwargs["quantization_config"] = self._build_bnb_config()
        else:
            kwargs["torch_dtype"] = torch_dtype

        model = AutoModelForCausalLM.from_pretrained(**kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    # ------------------------------------------------------------------
    # Layer resolution
    # ------------------------------------------------------------------

    def _resolve_decoder_layers(self) -> nn.ModuleList:
        """
        Tìm danh sách decoder layers theo cấu trúc phổ biến.
        Hỗ trợ: LLaMA, Qwen, Mistral, Phi, Falcon, GPT-NeoX, ...
        """
        for attr in ("layers", "h", "transformer"):
            if hasattr(self.model, attr):
                candidate = getattr(self.model, attr)
                if isinstance(candidate, nn.ModuleList):
                    return candidate
                # transformer.h (GPT-2 style)
                if hasattr(candidate, "h") and isinstance(candidate.h, nn.ModuleList):
                    return candidate.h
                if hasattr(candidate, "layers") and isinstance(candidate.layers, nn.ModuleList):
                    return candidate.layers

        # LLaMA / Qwen2 thường nằm ở model.model.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers

        raise RuntimeError(
            f"Không tìm thấy decoder layers trên {self.model.__class__.__name__}. "
            "Kiểm tra cấu trúc bằng print_full_layers() rồi override _resolve_decoder_layers()."
        )

    @property
    def layers(self) -> nn.ModuleList:
        return self._resolve_decoder_layers()

    # ------------------------------------------------------------------
    # Parameter stats
    # ------------------------------------------------------------------

    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_ratio(self) -> float:
        total = self.total_parameters
        return self.trainable_parameters / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # set_train_mode
    # ------------------------------------------------------------------

    def set_train_mode(
        self,
        mode: LLMTrainMode,
        lora_config: Optional[LoRAConfig] = None,
    ):
        """
        Chuyển đổi chế độ training của LLM.

        Args:
            mode:         Chế độ mới (FROZEN | LORA | QLORA).
            lora_config:  Cấu hình LoRA; bắt buộc khi mode là LORA / QLORA
                          (có thể bỏ qua nếu đã set trước đó với cùng config).
        """
        # Validate + chuẩn hoá mode NGAY TỪ ĐẦU, trước khi mutate bất kỳ state
        # nào (unload adapter, gán self.train_mode, freeze params, ...).
        # Nếu không validate trước, một mode không hợp lệ (vd: "full" — mode
        # cũ đã bị loại bỏ) có thể khiến self.train_mode bị gán giá trị rác
        # ngay trước khi ValueError được raise, làm hỏng invariant của object.
        try:
            mode = LLMTrainMode(mode)
        except ValueError:
            raise ValueError(f"Unsupported train mode: {mode}")

        # Nếu đang chuyển ra khỏi LORA/QLORA sang mode khác, phải gỡ adapter
        # trước — nếu không, các LoRA Linear layer vẫn còn trong graph,
        # _lora_applied vẫn True (sai trạng thái), và _resolve_decoder_layers()
        # có thể trỏ nhầm vào cấu trúc của PeftModel.
        if self._lora_applied and mode not in (LLMTrainMode.LORA, LLMTrainMode.QLORA):
            self._unload_lora()

        # Nếu LoRA đã được inject với CÙNG config và CÙNG mode, không inject
        # lại (tránh double-wrap). Nếu config khác (rank/alpha/target_modules
        # đổi) hoặc chuyển từ LORA <-> QLORA, phải gỡ adapter cũ rồi inject lại.
        if self._lora_applied and mode in (LLMTrainMode.LORA, LLMTrainMode.QLORA):
            cfg = lora_config or self._lora_config
            if cfg is not None and cfg == self._lora_config and mode == self.train_mode:
                self.train_mode = mode
                return
            self._unload_lora()

        self.train_mode = mode

        # Reset tất cả về frozen trước
        for p in self.model.parameters():
            p.requires_grad_(False)

        # ----------------------------------------------------------------
        if mode == LLMTrainMode.FROZEN:
            return

        # ----------------------------------------------------------------
        if mode in (LLMTrainMode.LORA, LLMTrainMode.QLORA):
            cfg = lora_config or self._lora_config
            if cfg is None:
                raise ValueError(
                    "LORA / QLORA yêu cầu truyền vào lora_config (LoRAConfig)."
                )
            self._lora_config = cfg
            self._apply_lora(cfg)
            return

        raise ValueError(f"Unsupported train mode: {mode}")

    # ------------------------------------------------------------------
    # LoRA injection
    # ------------------------------------------------------------------

    def _apply_lora(self, cfg: LoRAConfig):
        """
        Inject LoRA adapter vào model bằng thư viện peft.
        Yêu cầu: pip install peft
        """
        try:
            from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError:
            raise ImportError(
                "peft chưa được cài đặt. "
                "Chạy: pip install peft"
            )

        if self.train_mode == LLMTrainMode.QLORA:
            # Chuẩn bị model 4-bit cho kbit training (cast norm layers sang float32, ...)
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=True,
            )

        peft_cfg = PeftLoraConfig(
            r=cfg.r,
            lora_alpha=cfg.alpha,
            lora_dropout=cfg.dropout,
            target_modules=cfg.resolve_target_modules(),
            bias=cfg.bias,
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, peft_cfg)
        self._lora_applied = True

    def _unload_lora(self, merge: bool = False):
        """
        Gỡ adapter LoRA hiện tại khỏi self.model, trả về base model gốc.

        Args:
            merge: nếu True, merge weight LoRA vào base trước khi gỡ
                   (giữ lại hiệu ứng đã train). Nếu False, vứt bỏ adapter
                   hoàn toàn (dùng khi chuyển sang FROZEN mà
                   không cần giữ lại điều chỉnh từ LoRA).
        """
        if not self._lora_applied:
            return
        if merge:
            self.model = self.model.merge_and_unload()
        else:
            # unload(): bỏ adapter, trả lại base model — không merge weight.
            self.model = self.model.unload()
        self._lora_applied = False

    def merge_lora_weights(self):
        """
        Merge LoRA adapter vào base weights và xoá adapter overhead.
        Chỉ dùng khi inference hoặc tiếp tục fine-tune.

        Sau khi merge, hiệu ứng LoRA đã nằm trong base weights và không còn
        adapter riêng để theo dõi. Vì không còn train_mode FULL, toàn bộ
        base weights sẽ được đóng băng lại (train_mode = FROZEN). Gọi
        set_train_mode(LORA / QLORA, ...) sau đó nếu muốn tiếp tục fine-tune
        bằng adapter mới, hoặc mở khoá thủ công bằng
        `for p in self.model.parameters(): p.requires_grad_(True)` nếu thực
        sự cần train toàn bộ.
        """
        if not self._lora_applied:
            raise RuntimeError(
                "Chưa có LoRA adapter nào được inject. "
                "Gọi set_train_mode(LORA / QLORA) trước."
            )
        self._unload_lora(merge=True)
        self.train_mode = LLMTrainMode.FROZEN
        for p in self.model.parameters():
            p.requires_grad_(False)

    def save_lora_adapter(self, save_dir: str):
        """
        Lưu chỉ LoRA adapter weights (nhẹ hơn nhiều so với lưu full model).
        """
        if not self._lora_applied:
            raise RuntimeError("Không có LoRA adapter nào để lưu.")
        self.model.save_pretrained(save_dir)

    def load_lora_adapter(self, adapter_dir: str, is_trainable: bool = True):
        """
        Load lại LoRA adapter đã lưu vào base model hiện tại.

        Nếu đang có adapter khác được áp dụng, adapter cũ sẽ bị gỡ (không
        merge) trước khi load adapter mới, để tránh wrap PeftModel lồng
        nhau (PeftModel trên PeftModel).
        """
        try:
            from peft import PeftModel
        except ImportError:
            raise ImportError("peft chưa được cài đặt. Chạy: pip install peft")

        if self._lora_applied:
            self._unload_lora(merge=False)

        self.model = PeftModel.from_pretrained(self.model, adapter_dir, is_trainable=is_trainable)
        self._lora_applied = True
        self.train_mode = LLMTrainMode.LORA

    # ------------------------------------------------------------------
    # Gradient checkpointing
    # ------------------------------------------------------------------

    def enable_gradient_checkpointing(self):
        self.model.gradient_checkpointing_enable()

    def disable_gradient_checkpointing(self):
        self.model.gradient_checkpointing_disable()

    # ------------------------------------------------------------------
    # Debug / inspect
    # ------------------------------------------------------------------

    def print_full_layers(self):
        """In toàn bộ leaf modules kèm tên đầy đủ và loại."""
        print("\n[MODEL STRUCTURE]")
        print(f"{'Name':<60} {'Type'}")
        print("-" * 80)
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:
                print(f"{name:<60} {module.__class__.__name__}")

    def print_trainable_params(self):
        """In tên các parameter đang được train kèm shape."""
        print("\n[TRAINABLE PARAMETERS]")
        print(f"{'Name':<60} {'Shape'}")
        print("-" * 80)
        found = False
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                print(f"{name:<60} {list(p.shape)}")
                found = True
        if not found:
            print("  (không có parameter nào đang được train — mode FROZEN)")

    def summary(self):
        """In tóm tắt trạng thái hiện tại của LLMDecoder."""
        total = self.total_parameters
        trainable = self.trainable_parameters

        lora_info = ""
        if self._lora_applied and self._lora_config is not None:
            cfg = self._lora_config
            lora_info = (
                f"\n  LoRA rank       : {cfg.r}"
                f"\n  LoRA alpha      : {cfg.alpha}"
                f"\n  LoRA scaling    : {cfg.scaling:.4f}"
                f"\n  LoRA dropout    : {cfg.dropout}"
                f"\n  Target modules  : {cfg.resolve_target_modules()}"
            )

        print("\n" + "=" * 60)
        print("[SUMMARY — LanguageModelDecoder]")
        print("=" * 60)
        print(f"  Model ID        : {self.model_id}")
        print(f"  Model class     : {self.model.__class__.__name__}")
        print(f"  Train mode      : {self.train_mode.value}")
        print(f"  LoRA applied    : {self._lora_applied}")
        if lora_info:
            print(lora_info, end="")
        print(f"\n  Trainable params: {trainable:,}")
        print(f"  Total params    : {total:,}")
        if total > 0:
            print(f"  Trainable %     : {trainable / total:.4%}")
        else:
            print("  Trainable %     : N/A")
        print("=" * 60)

    def check_grad_flow(self, warn_no_backward: bool = True):
        """
        Kiểm tra gradient sau backward pass.

        Args:
            warn_no_backward: nếu True, in cảnh báo khi không có grad nào cả
                               (thường do chưa gọi loss.backward()).
        """
        print("\n[GRADIENT FLOW CHECK]")
        print(f"{'Name':<60} {'Status'}")
        print("-" * 80)

        ok_count = 0
        no_grad_count = 0

        for name, p in self.named_parameters():
            if p.requires_grad:
                if p.grad is not None:
                    grad_norm = p.grad.norm().item()
                    if grad_norm == 0.0:
                        status = "ZERO GRAD  ⚠"
                    elif grad_norm > 1e3:
                        status = f"EXPLODING  ⚠  norm={grad_norm:.2e}"
                    elif grad_norm < 1e-7:
                        status = f"VANISHING  ⚠  norm={grad_norm:.2e}"
                    else:
                        status = f"OK         norm={grad_norm:.4f}"
                    ok_count += 1
                else:
                    status = "NO GRAD"
                    no_grad_count += 1
                print(f"  {name:<58} {status}")

        print("-" * 80)
        print(f"  OK: {ok_count}   |   NO GRAD: {no_grad_count}")

        if warn_no_backward and ok_count == 0 and no_grad_count > 0:
            print(
                "\n  ⚠  Tất cả parameter đều chưa có gradient.\n"
                "     Hãy chạy loss.backward() trước khi gọi check_grad_flow()."
            )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = True,
    ):
        if input_ids is None and inputs_embeds is None:
            raise ValueError(
                "forward() cần ít nhất một trong hai: input_ids hoặc inputs_embeds."
            )

        return self.model(
            input_ids=input_ids if inputs_embeds is None else None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )