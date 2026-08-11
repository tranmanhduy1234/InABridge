"""Validated configuration objects for production training runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import CENTER_CROP, IMAGE_ENCODER_MODEL_ID, IMAGE_SIZE, LLM_MODEL_ID, QFORMER_TOKENIZER_ID


@dataclass
class DataConfig:
    train_manifest: str = ""
    train_manifests: Optional[List[str]] = None
    validation_manifest: Optional[str] = None
    image_root: Optional[str] = None
    image_column: str = "image"
    text_column: str = "text"
    instruction_column: str = "instruction"
    answer_column: str = "answer"
    num_workers: int = 8
    prefetch_factor: int = 2
    persistent_workers: bool = True
    max_instruction_length: int = 128
    max_text_length: int = 512
    max_samples: Optional[int] = None
    square_root_mixture_sampling: bool = True


@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-4
    qformer_learning_rate: Optional[float] = None
    projector_learning_rate: Optional[float] = None
    lora_learning_rate: Optional[float] = None
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0


@dataclass
class LossConfig:
    stage1_itc_weight: float = 1.0
    stage1_itm_weight: float = 1.0
    stage1_itg_weight: float = 1.0
    stage1_itc_label_smoothing: float = 0.0
    stage1_itm_label_smoothing: float = 0.0
    stage1_itg_label_smoothing: float = 0.0
    stage2_label_smoothing: float = 0.0
    stage2_z_loss_weight: float = 0.0
    reduction: str = "token_mean"


@dataclass
class ModelConfig:
    image_encoder_id: str = IMAGE_ENCODER_MODEL_ID
    image_size: int = IMAGE_SIZE
    center_crop: bool = CENTER_CROP
    qformer_tokenizer_id: str = QFORMER_TOKENIZER_ID
    llm_model_id: str = LLM_MODEL_ID
    llm_train_mode: str = "qlora"
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    tune_qformer_stage2: bool = True
    gradient_checkpointing: bool = True
    bridge_checkpoint: Optional[str] = None
    initialize_qformer_from_pretrained: bool = True


@dataclass
class TrainingConfig:
    stage: int = 2
    output_dir: str = "outputs/ina-bridge"
    seed: int = 42
    epochs: int = 1
    train_batch_size: int = 1
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    mixed_precision: str = "bf16"
    log_every_steps: int = 10
    eval_every_steps: int = 500
    save_every_steps: int = 500
    save_total_limit: int = 3
    resume_from_checkpoint: Optional[str] = None
    system_prompt: Optional[str] = None
    enable_thinking: bool = False
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def validate(self) -> None:
        if self.stage not in (1, 2):
            raise ValueError("stage must be 1 or 2")
        if not self.data.train_manifest and not self.data.train_manifests:
            raise ValueError("data.train_manifest or data.train_manifests is required")
        for name in ("epochs", "train_batch_size", "eval_batch_size", "gradient_accumulation_steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.data.num_workers < 0:
            raise ValueError("data.num_workers cannot be negative")
        if not 0.0 <= self.optimizer.warmup_ratio < 1.0:
            raise ValueError("optimizer.warmup_ratio must be in [0, 1)")
        if self.optimizer.learning_rate <= 0 or self.optimizer.max_grad_norm <= 0:
            raise ValueError("learning rate and max_grad_norm must be positive")
        if self.mixed_precision not in ("no", "fp16", "bf16"):
            raise ValueError("mixed_precision must be one of: no, fp16, bf16")
        if self.stage == 1 and self.train_batch_size < 2:
            raise ValueError("Stage 1 needs train_batch_size >= 2 per process for ITC/ITM")
        if self.stage == 1 and self.eval_batch_size < 2:
            raise ValueError("Stage 1 needs eval_batch_size >= 2 per process")
        if self.stage == 1 and self.model.llm_train_mode != "frozen":
            raise ValueError("Stage 1 must use model.llm_train_mode='frozen'")
        if self.model.llm_train_mode not in ("frozen", "lora", "qlora"):
            raise ValueError("Unsupported LLM training mode")
        if self.stage == 2 and self.model.llm_train_mode != "qlora":
            raise ValueError("Stage 2 production policy requires QLoRA")
        if self.stage == 2 and not self.model.tune_qformer_stage2:
            raise ValueError("InstructBLIP Stage 2 requires a trainable instruction-aware Q-Former")
        if self.stage == 2 and not self.model.bridge_checkpoint:
            raise ValueError("Stage 2 must initialize from a Stage-1 bridge checkpoint")
        if self.model.image_size <= 0 or self.model.image_size % 16 != 0:
            raise ValueError("model.image_size must be positive and divisible by 16")
        if self.save_total_limit <= 0:
            raise ValueError("save_total_limit must be positive")
        loss_weights = (
            self.loss.stage1_itc_weight,
            self.loss.stage1_itm_weight,
            self.loss.stage1_itg_weight,
        )
        if min(loss_weights) < 0 or sum(loss_weights) == 0:
            raise ValueError("Stage-1 loss weights must be non-negative and not all zero")
        smoothing_values = (
            self.loss.stage1_itc_label_smoothing,
            self.loss.stage1_itm_label_smoothing,
            self.loss.stage1_itg_label_smoothing,
            self.loss.stage2_label_smoothing,
        )
        if any(not 0.0 <= value < 1.0 for value in smoothing_values):
            raise ValueError("All label-smoothing values must be in [0, 1)")
        if self.loss.stage2_z_loss_weight < 0.0:
            raise ValueError("loss.stage2_z_loss_weight cannot be negative")
        if self.loss.reduction not in ("token_mean", "sample_mean"):
            raise ValueError("loss.reduction must be 'token_mean' or 'sample_mean'")

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TrainingConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown training config keys: {sorted(unknown)}")
        values = dict(raw)
        values["data"] = DataConfig(**values.get("data", {}))
        values["optimizer"] = OptimizerConfig(**values.get("optimizer", {}))
        values["loss"] = LossConfig(**values.get("loss", {}))
        values["model"] = ModelConfig(**values.get("model", {}))
        config = cls(**values)
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
