from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from src.config import LLM_MODL_ID
except ImportError:
    LLM_MODL_ID = "Qwen/Qwen3-4B"


class LLMTrainMode(str, Enum):
    FROZEN = "frozen"
    LORA = "lora"
    QLORA = "qlora"


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    target_modules: Optional[list[str]] = None
    bias: str = "none"

    def __post_init__(self):
        if self.r <= 0 or self.alpha <= 0:
            raise ValueError("LoRA r và alpha phải > 0.")
        if self.bias not in ("none", "all", "lora_only"):
            raise ValueError(f"Giá trị bias không hợp lệ: {self.bias}")

    @property
    def scaling(self) -> float:
        return self.alpha / self.r

    def resolve_target_modules(self) -> list[str]:
        if self.target_modules is not None:
            return self.target_modules
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class LanguageModelDecoder(nn.Module):
    """LLM Decoder Wrapper cho Qwen3-4B trong pipeline VLM."""

    def __init__(
        self,
        model_id: str = LLM_MODL_ID,
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

        self.model, self.tokenizer = self._load_base_model(model_id, train_mode, torch_dtype, device_map)
        self.set_train_mode(train_mode, lora_config=lora_config)

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def get_output_embeddings(self) -> nn.Module:
        return self.model.get_output_embeddings()

    def _load_base_model(self, model_id: str, train_mode: LLMTrainMode, torch_dtype: torch.dtype, device_map: str):
        kwargs = {"pretrained_model_name_or_path": model_id, "device_map": device_map, "trust_remote_code": True}
        if train_mode == LLMTrainMode.QLORA:
            if device_map is None:
                raise ValueError("QLORA yêu cầu device_map hợp lệ (như 'auto').")
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            kwargs["torch_dtype"] = torch_dtype

        model = AutoModelForCausalLM.from_pretrained(**kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return model, tokenizer

    def _resolve_decoder_layers(self) -> nn.ModuleList:
        model_obj = getattr(self.model, "base_model", self.model)
        for attr in ("layers", "h", "transformer"):
            if hasattr(model_obj, attr):
                candidate = getattr(model_obj, attr)
                if isinstance(candidate, nn.ModuleList):
                    return candidate
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        raise RuntimeError("Không tìm thấy decoder layers.")

    @property
    def layers(self) -> nn.ModuleList:
        return self._resolve_decoder_layers()

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

    def set_train_mode(self, mode: LLMTrainMode, lora_config: Optional[LoRAConfig] = None):
        mode = LLMTrainMode(mode)
        if self._lora_applied and mode not in (LLMTrainMode.LORA, LLMTrainMode.QLORA):
            self._unload_lora()

        if self._lora_applied and mode in (LLMTrainMode.LORA, LLMTrainMode.QLORA):
            cfg = lora_config or self._lora_config
            if cfg is not None and cfg == self._lora_config and mode == self.train_mode:
                self.train_mode = mode
                return
            self._unload_lora()

        self.train_mode = mode
        for p in self.model.parameters():
            p.requires_grad_(False)

        if mode == LLMTrainMode.FROZEN:
            return

        if mode in (LLMTrainMode.LORA, LLMTrainMode.QLORA):
            cfg = lora_config or self._lora_config
            if cfg is None:
                raise ValueError("Chế độ LoRA yêu cầu truyền lora_config.")
            self._lora_config = cfg
            self._apply_lora(cfg)

    def _apply_lora(self, cfg: LoRAConfig):
        from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training

        if self.train_mode == LLMTrainMode.QLORA:
            self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=True)

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
        if not self._lora_applied:
            return
        self.model = self.model.merge_and_unload() if merge else self.model.unload()
        self._lora_applied = False

    def merge_lora_weights(self):
        if not self._lora_applied:
            raise RuntimeError("Chưa áp dụng LoRA adapter nào.")
        self._unload_lora(merge=True)
        self.train_mode = LLMTrainMode.FROZEN
        for p in self.model.parameters():
            p.requires_grad_(False)

    def save_lora_adapter(self, save_dir: str):
        if not self._lora_applied:
            raise RuntimeError("Không có LoRA adapter để lưu.")
        self.model.save_pretrained(save_dir)

    def load_lora_adapter(self, adapter_dir: str, is_trainable: bool = True):
        from peft import PeftModel
        if self._lora_applied:
            self._unload_lora(merge=False)
        self.model = PeftModel.from_pretrained(self.model, adapter_dir, is_trainable=is_trainable)
        self._lora_applied = True
        self.train_mode = LLMTrainMode.LORA

    def enable_gradient_checkpointing(self):
        self.model.gradient_checkpointing_enable()

    def disable_gradient_checkpointing(self):
        self.model.gradient_checkpointing_disable()

    def summary(self):
        print(f"\n[LLM Summary] Model: {self.model_id} | Mode: {self.train_mode.value} | Params: {self.trainable_parameters:,}/{self.total_parameters:,} ({self.trainable_ratio:.2%})")

    def check_grad_flow(self, warn_no_backward: bool = True):
        print("\n[GRADIENT FLOW CHECK]")
        ok_count, no_grad_count = 0, 0
        for name, p in self.named_parameters():
            if p.requires_grad:
                if p.grad is not None:
                    norm = p.grad.norm().item()
                    status = "ZERO GRAD ⚠" if norm == 0 else f"OK norm={norm:.4f}"
                    ok_count += 1
                else:
                    status = "NO GRAD"
                    no_grad_count += 1
                print(f"  {name:<60} {status}")
        print(f"OK: {ok_count} | NO GRAD: {no_grad_count}")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = True,
        **kwargs,
    ):
        if input_ids is None and inputs_embeds is None:
            raise ValueError("forward() cần input_ids hoặc inputs_embeds.")
        return self.model(
            input_ids=input_ids if inputs_embeds is None else None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        **kwargs,
    ):
        if input_ids is None and inputs_embeds is None:
            raise ValueError("generate() cần input_ids hoặc inputs_embeds.")
        return self.model.generate(
            input_ids=input_ids if inputs_embeds is None else None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )