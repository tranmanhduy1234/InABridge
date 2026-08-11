"""Complete BLIP-2/InstructBLIP Stage-1 representation-learning losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .common import causal_token_loss


class _AllGatherWithGradient(torch.autograd.Function):
    """Autograd-safe equal-batch all-gather without deprecated APIs."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        ctx.rank = dist.get_rank()
        ctx.world_size = dist.get_world_size()
        gathered = [torch.empty_like(tensor) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, tensor.contiguous())
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        local_gradient = gradient.chunk(ctx.world_size, dim=0)[ctx.rank].contiguous()
        dist.all_reduce(local_gradient, op=dist.ReduceOp.SUM)
        return (local_gradient,)


@dataclass
class ContrastiveLossOutput:
    loss: torch.Tensor
    loss_image_to_text: torch.Tensor
    loss_text_to_image: torch.Tensor
    accuracy_image_to_text: torch.Tensor
    accuracy_text_to_image: torch.Tensor
    temperature: torch.Tensor
    global_batch_size: torch.Tensor


@dataclass
class Stage1LossOutput:
    loss: torch.Tensor
    loss_itc: torch.Tensor
    loss_itm: torch.Tensor
    loss_itg: torch.Tensor
    loss_itc_image_to_text: torch.Tensor
    loss_itc_text_to_image: torch.Tensor
    itc_accuracy_image_to_text: torch.Tensor
    itc_accuracy_text_to_image: torch.Tensor
    itm_accuracy: torch.Tensor
    itm_positive_accuracy: torch.Tensor
    itm_negative_accuracy: torch.Tensor
    itg_token_accuracy: torch.Tensor
    itg_num_supervised_tokens: torch.Tensor
    temperature: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "loss": self.loss,
            "loss_itc": self.loss_itc,
            "loss_itm": self.loss_itm,
            "loss_itg": self.loss_itg,
            "loss_itc_image_to_text": self.loss_itc_image_to_text,
            "loss_itc_text_to_image": self.loss_itc_text_to_image,
            "itc_accuracy_image_to_text": self.itc_accuracy_image_to_text,
            "itc_accuracy_text_to_image": self.itc_accuracy_text_to_image,
            "itm_accuracy": self.itm_accuracy,
            "itm_positive_accuracy": self.itm_positive_accuracy,
            "itm_negative_accuracy": self.itm_negative_accuracy,
            "itg_token_accuracy": self.itg_token_accuracy,
            "itg_num_supervised_tokens": self.itg_num_supervised_tokens,
            "temperature": self.temperature,
        }


class Stage1Loss(nn.Module):
    """Weighted image-text contrastive, matching and generation objectives.

    ITC follows BLIP-2: every query competes against all texts/images and the
    best query score represents an image. In distributed training, negatives
    from every rank participate while gradients remain connected.
    """

    def __init__(
        self,
        pad_token_id: int,
        itc_weight: float = 1.0,
        itm_weight: float = 1.0,
        itg_weight: float = 1.0,
        *,
        itc_label_smoothing: float = 0.0,
        itm_label_smoothing: float = 0.0,
        itg_label_smoothing: float = 0.0,
        temperature_min: float = 0.01,
        temperature_max: float = 1.0,
        reduction: str = "token_mean",
    ) -> None:
        super().__init__()
        weights = (itc_weight, itm_weight, itg_weight)
        if min(weights) < 0 or sum(weights) == 0:
            raise ValueError("Stage-1 weights must be non-negative and not all zero")
        for name, value in (
            ("itc_label_smoothing", itc_label_smoothing),
            ("itm_label_smoothing", itm_label_smoothing),
            ("itg_label_smoothing", itg_label_smoothing),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if not 0.0 < temperature_min <= temperature_max:
            raise ValueError("temperature bounds must satisfy 0 < min <= max")
        if reduction not in {"token_mean", "sample_mean"}:
            raise ValueError("reduction must be 'token_mean' or 'sample_mean'")
        self.pad_token_id = pad_token_id
        self.itc_weight, self.itm_weight, self.itg_weight = weights
        self.itc_label_smoothing = itc_label_smoothing
        self.itm_label_smoothing = itm_label_smoothing
        self.itg_label_smoothing = itg_label_smoothing
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        self.reduction = reduction

    @staticmethod
    def _distributed_features(features: torch.Tensor) -> tuple[torch.Tensor, int]:
        if not (dist.is_available() and dist.is_initialized()):
            return features, 0
        local_size = torch.tensor([features.size(0)], device=features.device, dtype=torch.long)
        sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
        dist.all_gather(sizes, local_size)
        size_values = [int(item.item()) for item in sizes]
        if len(set(size_values)) != 1:
            raise ValueError(
                "Distributed ITC requires equal per-rank batch sizes; use drop_last=True"
            )
        gathered = _AllGatherWithGradient.apply(features)
        return gathered, dist.get_rank() * features.size(0)

    def contrastive_loss(
        self,
        query_features: torch.Tensor,
        text_features: torch.Tensor,
        temperature: torch.Tensor,
    ) -> ContrastiveLossOutput:
        if query_features.ndim != 3 or text_features.ndim != 2:
            raise ValueError("ITC features must be [batch, queries, dim] and [batch, dim]")
        if query_features.size(0) != text_features.size(0):
            raise ValueError("ITC image and text local batch sizes differ")
        if query_features.size(-1) != text_features.size(-1):
            raise ValueError("ITC image and text embedding dimensions differ")
        if query_features.size(0) < 2:
            raise ValueError("ITC requires at least two image-text pairs per process")
        if temperature.numel() != 1:
            raise ValueError("ITC temperature must be a scalar")

        queries = F.normalize(query_features.float(), dim=-1)
        texts = F.normalize(text_features.float(), dim=-1)
        all_queries, offset = self._distributed_features(queries)
        all_texts, text_offset = self._distributed_features(texts)
        if offset != text_offset:
            raise RuntimeError("Inconsistent distributed ITC rank offsets")
        temp = temperature.float().reshape(()).clamp(self.temperature_min, self.temperature_max)
        targets = offset + torch.arange(queries.size(0), device=queries.device)

        logits_i2t = torch.einsum("bqd,gd->bgq", queries, all_texts).amax(dim=-1) / temp
        logits_t2i = torch.einsum("gqd,bd->bgq", all_queries, texts).amax(dim=-1) / temp
        loss_i2t = F.cross_entropy(
            logits_i2t, targets, label_smoothing=self.itc_label_smoothing
        )
        loss_t2i = F.cross_entropy(
            logits_t2i, targets, label_smoothing=self.itc_label_smoothing
        )
        return ContrastiveLossOutput(
            loss=0.5 * (loss_i2t + loss_t2i),
            loss_image_to_text=loss_i2t,
            loss_text_to_image=loss_t2i,
            accuracy_image_to_text=logits_i2t.argmax(-1).eq(targets).float().mean(),
            accuracy_text_to_image=logits_t2i.argmax(-1).eq(targets).float().mean(),
            temperature=temp.detach(),
            global_batch_size=torch.tensor(
                float(all_texts.size(0)), device=queries.device, dtype=torch.float32
            ),
        )

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        caption_ids: torch.Tensor,
        itm_labels: torch.Tensor,
        temperature: Optional[torch.Tensor] = None,
        caption_mask: Optional[torch.Tensor] = None,
    ) -> Stage1LossOutput:
        required = {"itc_query_proj", "itc_text_proj", "itm_logits", "itg_logits"}
        missing = required.difference(outputs)
        if missing:
            raise KeyError(f"Missing Stage-1 outputs: {sorted(missing)}")
        if caption_ids.ndim != 2:
            raise ValueError("caption_ids must have shape [batch, sequence]")
        if caption_mask is not None and caption_mask.shape != caption_ids.shape:
            raise ValueError("caption_mask and caption_ids must have the same shape")

        temp = temperature if temperature is not None else outputs.get("itc_temperature")
        if temp is None:
            temp = caption_ids.new_tensor(0.07, dtype=torch.float32)
        itc = self.contrastive_loss(outputs["itc_query_proj"], outputs["itc_text_proj"], temp)

        itm_logits = outputs["itm_logits"]
        itm_labels = itm_labels.to(device=itm_logits.device, dtype=torch.long).reshape(-1)
        if itm_logits.ndim != 2 or itm_logits.size(-1) != 2:
            raise ValueError("ITM logits must have shape [pairs, 2]")
        if itm_logits.size(0) != itm_labels.numel():
            raise ValueError("ITM logits and labels have different batch sizes")
        if bool(((itm_labels != 0) & (itm_labels != 1)).any()):
            raise ValueError("ITM labels must contain only 0 (negative) or 1 (positive)")
        loss_itm = F.cross_entropy(
            itm_logits.float(), itm_labels, label_smoothing=self.itm_label_smoothing
        )
        itm_predictions = itm_logits.argmax(dim=-1)
        positives, negatives = itm_labels.eq(1), itm_labels.eq(0)

        def class_accuracy(mask: torch.Tensor) -> torch.Tensor:
            if not bool(mask.any()):
                return torch.zeros((), device=itm_logits.device)
            return itm_predictions[mask].eq(itm_labels[mask]).float().mean()

        itg_labels = caption_ids.to(outputs["itg_logits"].device).clone()
        itg_labels.masked_fill_(itg_labels.eq(self.pad_token_id), -100)
        if caption_mask is not None:
            itg_labels.masked_fill_(caption_mask.to(itg_labels.device).eq(0), -100)
        itg = causal_token_loss(
            outputs["itg_logits"],
            itg_labels,
            label_smoothing=self.itg_label_smoothing,
            reduction=self.reduction,
        )
        total = self.itc_weight * itc.loss + self.itm_weight * loss_itm + self.itg_weight * itg.loss
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("Non-finite Stage-1 aggregate loss")
        return Stage1LossOutput(
            loss=total,
            loss_itc=itc.loss,
            loss_itm=loss_itm,
            loss_itg=itg.loss,
            loss_itc_image_to_text=itc.loss_image_to_text,
            loss_itc_text_to_image=itc.loss_text_to_image,
            itc_accuracy_image_to_text=itc.accuracy_image_to_text,
            itc_accuracy_text_to_image=itc.accuracy_text_to_image,
            itm_accuracy=itm_predictions.eq(itm_labels).float().mean(),
            itm_positive_accuracy=class_accuracy(positives),
            itm_negative_accuracy=class_accuracy(negatives),
            itg_token_accuracy=itg.token_accuracy,
            itg_num_supervised_tokens=itg.num_supervised_tokens,
            temperature=itc.temperature,
        )