"""Complete Stage-2 instruction-tuning objective for the QLoRA language model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from .common import causal_token_loss


@dataclass
class Stage2LossOutput:
    loss: torch.Tensor
    nll_loss: torch.Tensor
    smooth_loss: torch.Tensor
    z_loss: torch.Tensor
    token_accuracy: torch.Tensor
    num_supervised_tokens: torch.Tensor
    mean_tokens_per_sample: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "loss": self.loss,
            "nll_loss": self.nll_loss,
            "smooth_loss": self.smooth_loss,
            "z_loss": self.z_loss,
            "token_accuracy": self.token_accuracy,
            "num_supervised_tokens": self.num_supervised_tokens,
            "mean_tokens_per_sample": self.mean_tokens_per_sample,
        }


class Stage2CausalLMLoss(nn.Module):
    """Assistant-only shifted causal LM loss over visual-prefix sequences.

    Text labels are supplied by the collator with prompt/padding positions set
    to ``ignore_index``. This module prepends an ignored visual-prefix region,
    applies the causal shift, and computes the objective itself. Optional
    z-loss regularizes the logit normalizer and can improve low-precision
    stability without changing which tokens are supervised.
    """

    def __init__(
        self,
        *,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        z_loss_weight: float = 0.0,
        reduction: str = "token_mean",
    ) -> None:
        super().__init__()
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if z_loss_weight < 0.0:
            raise ValueError("z_loss_weight cannot be negative")
        if reduction not in {"token_mean", "sample_mean"}:
            raise ValueError("reduction must be 'token_mean' or 'sample_mean'")
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.z_loss_weight = z_loss_weight
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        text_labels: torch.Tensor,
        visual_prefix_length: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Stage2LossOutput:
        if text_labels.ndim != 2:
            raise ValueError("text_labels must have shape [batch, text_sequence]")
        if visual_prefix_length < 0:
            raise ValueError("visual_prefix_length cannot be negative")
        expected = text_labels.size(1) + visual_prefix_length
        if logits.ndim != 3 or logits.size(0) != text_labels.size(0) or logits.size(1) != expected:
            raise ValueError(
                "Stage-2 logits must match [batch, visual_prefix_length + text_length, vocab]"
            )
        labels = text_labels.to(device=logits.device, dtype=torch.long).clone()
        if attention_mask is not None:
            if attention_mask.shape != text_labels.shape:
                raise ValueError("attention_mask and text_labels must have the same shape")
            labels.masked_fill_(attention_mask.to(logits.device).eq(0), self.ignore_index)
        prefix = torch.full(
            (labels.size(0), visual_prefix_length),
            self.ignore_index,
            dtype=labels.dtype,
            device=logits.device,
        )
        combined_labels = torch.cat([prefix, labels], dim=1)
        result = causal_token_loss(
            logits,
            combined_labels,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            z_loss_weight=self.z_loss_weight,
            reduction=self.reduction,
        )
        return Stage2LossOutput(
            loss=result.loss,
            nll_loss=result.nll_loss,
            smooth_loss=result.smooth_loss,
            z_loss=result.z_loss,
            token_accuracy=result.token_accuracy,
            num_supervised_tokens=result.num_supervised_tokens,
            mean_tokens_per_sample=result.mean_tokens_per_sample,
        )
