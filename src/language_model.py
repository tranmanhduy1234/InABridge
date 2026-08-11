"""Qwen causal-language-model wrapper with frozen, LoRA and QLoRA modes.

The wrapper deliberately owns the PEFT lifecycle.  In particular, a quantized
base model is never presented as a full-precision LoRA model and a merged
QLoRA adapter is rejected rather than silently merging into four-bit weights.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config import LLM_MODEL_ID


class LLMTrainMode(str, Enum):
    """Supported parameter-update policies for the language model."""

    FROZEN = "frozen"
    LORA = "lora"
    QLORA = "qlora"


@dataclass
class LoRAConfig:
    """Configuration for a causal-LM LoRA adapter.

    The default targets cover Qwen's attention projections and SwiGLU MLP.
    ``modules_to_save`` may be used for an explicitly trainable output head,
    although InA-Bridge normally leaves it unset to keep the whole Qwen base
    frozen.
    """

    r: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    target_modules: Optional[list[str]] = None
    bias: str = "none"
    use_rslora: bool = False
    modules_to_save: Optional[list[str]] = None
    init_lora_weights: bool | str = True

    def __post_init__(self) -> None:
        if isinstance(self.r, bool) or not isinstance(self.r, int) or self.r <= 0:
            raise ValueError("LoRA r must be a positive integer.")
        if not isinstance(self.alpha, (int, float)) or isinstance(self.alpha, bool):
            raise TypeError("LoRA alpha must be numeric.")
        if not math.isfinite(float(self.alpha)) or self.alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive.")
        if not isinstance(self.dropout, (int, float)) or isinstance(self.dropout, bool):
            raise TypeError("LoRA dropout must be numeric.")
        if not math.isfinite(float(self.dropout)) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1).")
        if self.bias not in {"none", "all", "lora_only"}:
            raise ValueError(f"Unsupported LoRA bias policy: {self.bias!r}.")
        if not isinstance(self.use_rslora, bool):
            raise TypeError("LoRA use_rslora must be a boolean.")
        if not isinstance(self.init_lora_weights, (bool, str)):
            raise TypeError("LoRA init_lora_weights must be a boolean or PEFT strategy name.")
        if isinstance(self.init_lora_weights, str) and not self.init_lora_weights.strip():
            raise ValueError("LoRA init_lora_weights strategy cannot be empty.")
        self._validate_module_names(self.target_modules, "target_modules")
        self._validate_module_names(self.modules_to_save, "modules_to_save")

    @staticmethod
    def _validate_module_names(values: Optional[list[str]], field: str) -> None:
        if values is None:
            return
        if not isinstance(values, list) or not values:
            raise ValueError(f"LoRA {field} must be a non-empty list when provided.")
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(f"LoRA {field} contains an empty or non-string name.")
        if len(set(values)) != len(values):
            raise ValueError(f"LoRA {field} contains duplicate names.")

    @property
    def scaling(self) -> float:
        return float(self.alpha) / self.r

    def resolve_target_modules(self) -> list[str]:
        if self.target_modules is not None:
            return list(self.target_modules)
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]


class LanguageModelDecoder(nn.Module):
    """Production wrapper around a Hugging Face decoder-only language model.

    ``QLORA`` must be selected when loading the model because four-bit
    quantization is a load-time operation.  It cannot be enabled later by
    merely changing ``requires_grad`` flags.
    """

    _FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

    def __init__(
        self,
        model_id: str = LLM_MODEL_ID,
        train_mode: LLMTrainMode = LLMTrainMode.FROZEN,
        lora_config: Optional[LoRAConfig] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: Optional[Any] = None,
        *,
        revision: Optional[str] = None,
        trust_remote_code: bool = True,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string.")
        try:
            requested_mode = LLMTrainMode(train_mode)
        except ValueError as exc:
            raise ValueError(f"Unsupported language-model train mode: {train_mode!r}.") from exc
        if torch_dtype not in self._FLOAT_DTYPES:
            raise ValueError(
                "torch_dtype must be torch.float16, torch.bfloat16, or torch.float32."
            )
        if requested_mode in {LLMTrainMode.LORA, LLMTrainMode.QLORA}:
            if lora_config is None:
                raise ValueError(f"{requested_mode.value} mode requires lora_config.")
            if not isinstance(lora_config, LoRAConfig):
                raise TypeError("lora_config must be an instance of LoRAConfig.")

        self.model_id = model_id
        self.train_mode = LLMTrainMode.FROZEN
        self.torch_dtype = torch_dtype
        self._lora_config = lora_config
        self._lora_applied = False
        self._adapter_parameter_names: set[str] = set()
        self._adapters_trainable = False
        self._active_adapter_name = "default"
        self._gradient_checkpointing_enabled = False
        # Quantization is fixed at load time, even after PEFT wraps the model.
        self._base_loaded_in_4bit = requested_mode == LLMTrainMode.QLORA

        self.model, self.tokenizer = self._load_base_model(
            model_id=model_id,
            train_mode=requested_mode,
            torch_dtype=torch_dtype,
            device_map=device_map,
            revision=revision,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        self._base_loaded_in_4bit = bool(
            self._base_loaded_in_4bit
            or getattr(self.model, "is_loaded_in_4bit", False)
            or getattr(self.model, "is_quantized", False)
        )
        self._original_use_cache = getattr(getattr(self.model, "config", None), "use_cache", None)
        self._configure_padding()
        self.set_train_mode(requested_mode, lora_config=lora_config)

    @property
    def device(self) -> torch.device:
        embedding = self.get_input_embeddings()
        parameter = next(embedding.parameters(), None)
        if parameter is not None:
            return parameter.device
        parameter = next(self.parameters(), None)
        return parameter.device if parameter is not None else torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        embedding = self.get_input_embeddings()
        parameter = next(embedding.parameters(), None)
        if parameter is not None and (parameter.is_floating_point() or parameter.is_complex()):
            return parameter.dtype
        return self.torch_dtype

    @property
    def is_quantized(self) -> bool:
        return self._base_loaded_in_4bit

    @property
    def lora_applied(self) -> bool:
        return self._lora_applied

    def get_input_embeddings(self) -> nn.Module:
        embedding = self.model.get_input_embeddings()
        if embedding is None:
            raise RuntimeError("The language model does not expose input embeddings.")
        return embedding

    def get_output_embeddings(self) -> Optional[nn.Module]:
        getter = getattr(self.model, "get_output_embeddings", None)
        return getter() if callable(getter) else None

    def _load_base_model(
        self,
        model_id: str,
        train_mode: LLMTrainMode,
        torch_dtype: torch.dtype,
        device_map: Optional[Any],
        revision: Optional[str],
        trust_remote_code: bool,
        local_files_only: bool,
    ) -> tuple[nn.Module, Any]:
        common_kwargs: dict[str, Any] = {
            "pretrained_model_name_or_path": model_id,
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        if revision is not None:
            common_kwargs["revision"] = revision

        model_kwargs = dict(common_kwargs)
        # Transformers 5 uses ``dtype``.  This also controls unquantized layers
        # (embeddings/norms/head) in a bitsandbytes model.
        model_kwargs["dtype"] = torch_dtype
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        if train_mode == LLMTrainMode.QLORA:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )

        try:
            model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        except TypeError as exc:
            # Transformers < 4.56 used torch_dtype.  Keeping this narrow
            # fallback makes saved projects usable with PEFT 0.14-era stacks.
            if "dtype" not in str(exc):
                raise
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
            model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

        tokenizer = AutoTokenizer.from_pretrained(
            **common_kwargs,
            use_fast=True,
        )
        return model, tokenizer

    def _configure_padding(self) -> None:
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None or self.tokenizer.eos_token is None:
                raise ValueError(
                    "Tokenizer has neither a pad token nor a usable EOS token. "
                    "Configure one explicitly before batching."
                )
            # Reuse EOS instead of adding a vocabulary row that would require
            # resizing a quantized/tied embedding matrix.
            self.tokenizer.pad_token = self.tokenizer.eos_token

        pad_token_id = int(self.tokenizer.pad_token_id)
        config = getattr(self.model, "config", None)
        if config is not None and getattr(config, "pad_token_id", None) is None:
            config.pad_token_id = pad_token_id
        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is not None and generation_config.pad_token_id is None:
            generation_config.pad_token_id = pad_token_id

    def _validate_lora_targets(self, targets: list[str]) -> None:
        module_names = tuple(name for name, _ in self.model.named_modules())
        missing = [
            target
            for target in targets
            if not any(name == target or name.endswith(f".{target}") for name in module_names)
        ]
        if missing:
            raise ValueError(
                "LoRA target modules were not found in the language model: "
                f"{missing}. Check the architecture-specific projection names."
            )

    def set_train_mode(
        self,
        mode: LLMTrainMode,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        """Apply a complete and internally consistent freezing policy."""

        try:
            mode = LLMTrainMode(mode)
        except ValueError as exc:
            raise ValueError(f"Unsupported language-model train mode: {mode!r}.") from exc

        if mode == LLMTrainMode.QLORA and not self._base_loaded_in_4bit:
            raise RuntimeError(
                "QLoRA cannot be enabled after loading a full-precision base model; "
                "construct LanguageModelDecoder with train_mode='qlora'."
            )
        if mode == LLMTrainMode.LORA and self._base_loaded_in_4bit:
            raise RuntimeError("A four-bit base model must use QLORA mode, not LORA mode.")

        # An adapter loaded from disk owns its PEFT config.  Re-selecting its
        # existing mode should only change trainability, not rebuild randomly
        # initialized weights from a separately supplied local config.
        if self._lora_applied and mode == self.train_mode and lora_config is None:
            self.set_adapters_trainable(True)
            return

        cfg = lora_config or self._lora_config
        if mode in {LLMTrainMode.LORA, LLMTrainMode.QLORA}:
            if cfg is None:
                raise ValueError(f"{mode.value} mode requires lora_config.")
            if not isinstance(cfg, LoRAConfig):
                raise TypeError("lora_config must be an instance of LoRAConfig.")

        if self._lora_applied:
            if mode == self.train_mode and cfg == self._lora_config:
                self.set_adapters_trainable(True)
                return
            self._unload_lora(merge=False)

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.train_mode = mode
        self._adapters_trainable = False

        if mode == LLMTrainMode.FROZEN:
            self.model.eval()
            return

        self._lora_config = copy.deepcopy(cfg)
        try:
            self._apply_lora(self._lora_config)
        except Exception:
            if self._lora_applied:
                try:
                    self._unload_lora(merge=False)
                except Exception:
                    # Preserve the adapter-construction exception; the object
                    # is still forced into a non-trainable fail-safe state.
                    pass
            self.train_mode = LLMTrainMode.FROZEN
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
            self.model.eval()
            raise

    def _apply_lora(self, cfg: LoRAConfig) -> None:
        try:
            from peft import (
                LoraConfig as PeftLoraConfig,
                get_peft_model,
                prepare_model_for_kbit_training,
            )
        except ImportError as exc:
            raise RuntimeError(
                "LoRA/QLoRA requires the optional 'peft' dependency. "
                "Install the project's train extras."
            ) from exc

        targets = cfg.resolve_target_modules()
        self._validate_lora_targets(targets)
        if self.train_mode == LLMTrainMode.QLORA:
            # Checkpointing is enabled separately by the trainer.  Preparing
            # here still freezes/casts all non-quantized base layers correctly.
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=False,
            )

        peft_cfg = PeftLoraConfig(
            r=cfg.r,
            lora_alpha=cfg.alpha,
            lora_dropout=cfg.dropout,
            target_modules=targets,
            bias=cfg.bias,
            use_rslora=cfg.use_rslora,
            modules_to_save=(list(cfg.modules_to_save) if cfg.modules_to_save else None),
            init_lora_weights=cfg.init_lora_weights,
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, peft_cfg)
        self._lora_applied = True
        self._active_adapter_name = "default"
        self._adapter_parameter_names = {
            name for name, parameter in self.model.named_parameters() if parameter.requires_grad
        }
        if not self._adapter_parameter_names:
            raise RuntimeError("PEFT created no trainable adapter parameters.")
        self._adapters_trainable = True
        self.assert_base_model_frozen()

    def _unload_lora(self, merge: bool = False) -> None:
        if not self._lora_applied:
            return
        operation_name = "merge_and_unload" if merge else "unload"
        operation = getattr(self.model, operation_name, None)
        if not callable(operation):
            raise RuntimeError(f"The installed PEFT model does not support {operation_name}().")
        self.model = operation(safe_merge=True) if merge else operation()
        self._lora_applied = False
        self._adapter_parameter_names.clear()
        self._adapters_trainable = False
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def merge_lora_weights(self) -> None:
        """Safely merge a full-precision LoRA adapter into its base model."""

        if not self._lora_applied:
            raise RuntimeError("No LoRA adapter is loaded.")
        if self._base_loaded_in_4bit:
            raise RuntimeError(
                "Refusing to merge QLoRA into a four-bit base. Reload the base "
                "in float16/bfloat16, attach the adapter, then merge it there."
            )
        self._unload_lora(merge=True)
        self.train_mode = LLMTrainMode.FROZEN
        self.model.eval()

    def save_lora_adapter(
        self,
        save_dir: str | Path,
        *,
        safe_serialization: bool = True,
        save_tokenizer: bool = False,
    ) -> None:
        """Save adapter weights/config without duplicating the base LLM."""

        if not self._lora_applied:
            raise RuntimeError("No LoRA adapter is loaded.")
        destination = Path(save_dir).expanduser()
        if destination.exists() and not destination.is_dir():
            raise NotADirectoryError(destination)
        destination.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(
            destination,
            safe_serialization=safe_serialization,
            selected_adapters=[self._active_adapter_name],
        )
        if save_tokenizer:
            self.tokenizer.save_pretrained(destination)

    def load_lora_adapter(
        self,
        adapter_dir: str | Path,
        is_trainable: bool = True,
        *,
        adapter_name: str = "default",
    ) -> None:
        """Attach a PEFT adapter to the already loaded compatible base model."""

        if not isinstance(adapter_name, str) or not adapter_name.strip():
            raise ValueError("adapter_name must be a non-empty string.")
        if not isinstance(is_trainable, bool):
            raise TypeError("is_trainable must be a boolean.")
        if isinstance(adapter_dir, str) and not adapter_dir.strip():
            raise ValueError("adapter_dir must be a non-empty path or Hub model ID.")
        if isinstance(adapter_dir, Path) and not adapter_dir.expanduser().is_dir():
            raise FileNotFoundError(adapter_dir.expanduser())
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Loading an adapter requires the optional 'peft' dependency.") from exc

        if self._lora_applied:
            self._unload_lora(merge=False)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        try:
            self.model = PeftModel.from_pretrained(
                self.model,
                str(adapter_dir),
                adapter_name=adapter_name,
                is_trainable=is_trainable,
            )
        except Exception:
            self.train_mode = LLMTrainMode.FROZEN
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
            self.model.eval()
            raise
        self._lora_applied = True
        self._active_adapter_name = adapter_name
        self.train_mode = (
            LLMTrainMode.QLORA if self._base_loaded_in_4bit else LLMTrainMode.LORA
        )
        self._adapter_parameter_names = {
            name for name, parameter in self.model.named_parameters() if parameter.requires_grad
        }
        # An inference-only load has no trainable parameters.  Discover the
        # standard PEFT-owned tensors so it can later be made trainable safely.
        if not getattr(self, "_adapter_parameter_names", set()):
            self._adapter_parameter_names = {
                name
                for name, _ in self.model.named_parameters()
                if self._looks_like_adapter_parameter(name)
            }
        self._adapters_trainable = False
        self.set_adapters_trainable(is_trainable)
        self.assert_base_model_frozen()

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only PEFT-owned tensors for portable checkpoints."""

        if not self._lora_applied:
            raise RuntimeError("No LoRA adapter is loaded.")
        from peft import get_peft_model_state_dict

        return dict(
            get_peft_model_state_dict(
                self.model,
                adapter_name=self._active_adapter_name,
            )
        )

    def load_adapter_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        strict: bool = True,
    ) -> Any:
        """Load a portable adapter-only state dictionary into an attached adapter."""

        if not self._lora_applied:
            raise RuntimeError("Attach a LoRA adapter before loading its state dictionary.")
        if not isinstance(state_dict, Mapping) or not state_dict:
            raise ValueError("state_dict must be a non-empty mapping.")
        from peft import set_peft_model_state_dict

        result = set_peft_model_state_dict(
            self.model,
            dict(state_dict),
            adapter_name=self._active_adapter_name,
        )
        if strict:
            missing = list(getattr(result, "missing_keys", ()))
            unexpected = list(getattr(result, "unexpected_keys", ()))
            # PEFT may report frozen base keys as missing; adapter keys are the
            # only relevant strictness boundary for an adapter-only checkpoint.
            missing_adapter = [key for key in missing if self._looks_like_adapter_parameter(key)]
            if missing_adapter or unexpected:
                raise RuntimeError(
                    "Adapter state mismatch: "
                    f"missing={missing_adapter[:8]}, unexpected={unexpected[:8]}."
                )
        return result

    def enable_gradient_checkpointing(self) -> None:
        """Enable non-reentrant activation checkpointing and disable KV cache."""

        if self._gradient_checkpointing_enabled:
            return
        enable = getattr(self.model, "gradient_checkpointing_enable", None)
        if not callable(enable):
            raise AttributeError("The language model does not support gradient checkpointing.")
        try:
            enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            enable()
        require_input_grads = getattr(self.model, "enable_input_require_grads", None)
        if callable(require_input_grads) and self._lora_applied:
            require_input_grads()
        config = getattr(self.model, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False
        self._gradient_checkpointing_enabled = True

    def disable_gradient_checkpointing(self) -> None:
        if not self._gradient_checkpointing_enabled:
            return
        disable = getattr(self.model, "gradient_checkpointing_disable", None)
        if not callable(disable):
            raise AttributeError("The language model does not support gradient checkpointing.")
        disable()
        disable_input_grads = getattr(self.model, "disable_input_require_grads", None)
        if callable(disable_input_grads):
            disable_input_grads()
        config = getattr(self.model, "config", None)
        if config is not None and self._original_use_cache is not None:
            config.use_cache = self._original_use_cache
        self._gradient_checkpointing_enabled = False

    @staticmethod
    def _looks_like_adapter_parameter(name: str) -> bool:
        return any(
            marker in name
            for marker in (
                "lora_A",
                "lora_B",
                "lora_embedding_A",
                "lora_embedding_B",
                ".modules_to_save.",
                "trainable_tokens_",
            )
        )

    def set_adapters_trainable(self, enabled: bool) -> None:
        """Toggle PEFT-owned parameters while always freezing the base Qwen."""

        enabled = bool(enabled)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if not enabled or not self._lora_applied:
            self._adapters_trainable = False
            if not enabled:
                self.model.eval()
            return

        set_adapter = getattr(self.model, "set_adapter", None)
        if callable(set_adapter):
            try:
                set_adapter(self._active_adapter_name, inference_mode=False)
            except TypeError:
                set_adapter(self._active_adapter_name)
            peft_trainable = {
                name for name, parameter in self.model.named_parameters() if parameter.requires_grad
            }
            if peft_trainable:
                self._adapter_parameter_names.update(peft_trainable)

        if not getattr(self, "_adapter_parameter_names", set()):
            self._adapter_parameter_names = {
                name
                for name, _ in self.model.named_parameters()
                if self._looks_like_adapter_parameter(name)
            }
        if not self._adapter_parameter_names:
            raise RuntimeError("No PEFT adapter parameters were found to enable.")
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(name in self._adapter_parameter_names)
        self._adapters_trainable = True
        if self.training:
            self.model.train(True)
        self.assert_base_model_frozen()

    def assert_base_model_frozen(self) -> None:
        """Fail fast if the PEFT freezing boundary has been violated."""

        allowed = getattr(self, "_adapter_parameter_names", set())
        leaked = [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
            and name not in allowed
            and not self._looks_like_adapter_parameter(name)
        ]
        if leaked:
            raise RuntimeError(f"Base LLM parameters unexpectedly trainable: {leaked[:8]}.")

    def trainable_parameter_summary(self) -> dict[str, int | float]:
        total = sum(parameter.numel() for parameter in self.model.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad
        )
        return {
            "trainable": trainable,
            "total": total,
            "trainable_fraction": (trainable / total if total else 0.0),
        }

    def train(self, mode: bool = True) -> LanguageModelDecoder:
        super().train(mode)
        # A completely frozen tower and an inference-only adapter must never
        # activate base-model dropout during surrounding bridge training.
        if self.train_mode == LLMTrainMode.FROZEN or not self._adapters_trainable:
            self.model.eval()
        return self

    @staticmethod
    def _validate_model_inputs(
        *,
        input_ids: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        allow_both_inputs: bool = False,
    ) -> tuple[int, int]:
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Exactly one of input_ids or inputs_embeds is required.")
        if input_ids is not None and inputs_embeds is not None and not allow_both_inputs:
            raise ValueError("input_ids and inputs_embeds are mutually exclusive.")
        if input_ids is not None:
            if input_ids.ndim != 2:
                raise ValueError("input_ids must have shape [batch, sequence].")
            if input_ids.dtype not in {torch.int32, torch.int64}:
                raise TypeError("input_ids must use torch.int32 or torch.int64 dtype.")
            batch_size, sequence_length = input_ids.shape
        else:
            assert inputs_embeds is not None
            if inputs_embeds.ndim != 3:
                raise ValueError("inputs_embeds must have shape [batch, sequence, hidden].")
            if not inputs_embeds.is_floating_point():
                raise TypeError("inputs_embeds must use a floating-point dtype.")
            batch_size, sequence_length = inputs_embeds.shape[:2]
        if batch_size <= 0 or sequence_length <= 0:
            raise ValueError("Language-model inputs must have non-empty batch and sequence axes.")

        if input_ids is not None and inputs_embeds is not None:
            if input_ids.shape != inputs_embeds.shape[:2]:
                raise ValueError("input_ids and inputs_embeds must have matching batch/sequence axes.")
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape[0] != batch_size:
                raise ValueError("attention_mask must have shape [batch, attended_sequence].")
            if attention_mask.shape[1] < sequence_length:
                raise ValueError("attention_mask is shorter than the current input sequence.")
        if labels is not None:
            if labels.ndim != 2 or labels.shape != (batch_size, sequence_length):
                raise ValueError("labels must match the current [batch, sequence] input shape.")
            if labels.dtype != torch.long:
                raise TypeError("labels must use torch.long dtype.")
        return batch_size, sequence_length

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Any = None,
        use_cache: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        self._validate_model_inputs(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        if kwargs.pop("return_dict", True) is not True:
            raise ValueError("LanguageModelDecoder.forward always returns a ModelOutput.")
        if use_cache is None:
            use_cache = not (self.training or self._gradient_checkpointing_enabled)
        if self.training and self._gradient_checkpointing_enabled:
            use_cache = False
        return self.model(
            input_ids=input_ids,
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
        **kwargs: Any,
    ) -> Any:
        """Generate from token IDs or a multimodal soft-prompt embedding sequence."""

        self._validate_model_inputs(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise TypeError("max_new_tokens must be an integer.")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")

        generation_kwargs = dict(kwargs)
        generation_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        if self.tokenizer.eos_token_id is not None:
            generation_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        generation_kwargs.setdefault("use_cache", True)

        was_training = self.model.training
        self.model.eval()
        try:
            return self.model.generate(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                **generation_kwargs,
            )
        finally:
            if was_training and self.train_mode != LLMTrainMode.FROZEN:
                self.model.train(True)
