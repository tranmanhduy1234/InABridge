"""Stage-aware optimizer parameter grouping."""

from typing import Dict

import torch
from torch.optim import AdamW

from src.model import InABridgeModel
from src.training.config import TrainingConfig


def _component_lr(name: str, config: TrainingConfig) -> float:
    optim = config.optimizer
    if name.startswith("qformer.") and optim.qformer_learning_rate is not None:
        return optim.qformer_learning_rate
    if name.startswith("projector.") and optim.projector_learning_rate is not None:
        return optim.projector_learning_rate
    if name.startswith("llm.") and optim.lora_learning_rate is not None:
        return optim.lora_learning_rate
    return optim.learning_rate


def build_optimizer(model: InABridgeModel, config: TrainingConfig) -> AdamW:
    """Create component-aware AdamW groups with correct norm/bias decay."""
    groups: Dict[tuple[float, float], list[torch.nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        no_decay = parameter.ndim < 2 or name.endswith(".bias") or "norm" in name.lower()
        key = (_component_lr(name, config), 0.0 if no_decay else config.optimizer.weight_decay)
        groups.setdefault(key, []).append(parameter)
    if not groups:
        raise ValueError("No trainable parameters after applying the stage freezing policy")
    return AdamW(
        [
            {"params": params, "lr": learning_rate, "weight_decay": decay}
            for (learning_rate, decay), params in groups.items()
        ],
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        eps=config.optimizer.eps,
    )
