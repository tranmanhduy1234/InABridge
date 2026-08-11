"""Numerically stable primitives shared by the two training stages."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CausalTokenLoss:
    """Unreduced statistics for an autoregressive token objective."""

    loss: torch.Tensor
    nll_loss: torch.Tensor
    smooth_loss: torch.Tensor
    z_loss: torch.Tensor
    token_accuracy: torch.Tensor
    num_supervised_tokens: torch.Tensor
    mean_tokens_per_sample: torch.Tensor


def _validate_labels(labels: torch.Tensor, vocab_size: int, ignore_index: int) -> None:
    valid = labels.ne(ignore_index)
    if not bool(valid.any()):
        raise ValueError("Causal LM loss received no supervised target tokens")
    supervised = labels[valid]
    if bool((supervised < 0).any()) or bool((supervised >= vocab_size).any()):
        minimum = int(supervised.min().item())
        maximum = int(supervised.max().item())
        raise ValueError(
            f"Target token IDs must be in [0, {vocab_size}); got [{minimum}, {maximum}]"
        )


def causal_token_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    z_loss_weight: float = 0.0,
    reduction: str = "token_mean",
) -> CausalTokenLoss:
    """Compute shifted causal cross entropy without delegating to Transformers.

    ``labels[:, t]`` supervises ``logits[:, t - 1]``.  Computation is promoted
    to fp32 so that large-vocabulary bf16/fp16 training remains stable.
    ``sample_mean`` gives each sample equal weight; ``token_mean`` gives each
    answer token equal weight.
    """

    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("logits must be [batch, sequence, vocab] and labels [batch, sequence]")
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            f"Logit/label sequence mismatch: {tuple(logits.shape[:2])} != {tuple(labels.shape)}"
        )
    if logits.size(1) < 2 or logits.size(-1) < 2:
        raise ValueError("Causal LM loss needs sequence length >= 2 and vocabulary size >= 2")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    if z_loss_weight < 0.0:
        raise ValueError("z_loss_weight cannot be negative")
    if reduction not in {"token_mean", "sample_mean"}:
        raise ValueError("reduction must be 'token_mean' or 'sample_mean'")

    # Use logsumexp identities instead of materialising a second full
    # [batch, sequence, vocabulary] log-probability tensor. This matters for
    # Qwen's large vocabulary and still keeps the reduction in fp32.
    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:].to(logits.device)
    _validate_labels(shifted_labels, shifted_logits.size(-1), ignore_index)

    valid = shifted_labels.ne(ignore_index)
    safe_labels = shifted_labels.masked_fill(~valid, 0)
    log_normalizer = torch.logsumexp(shifted_logits, dim=-1)
    target_logits = shifted_logits.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    nll_tokens = log_normalizer - target_logits
    smooth_tokens = log_normalizer - shifted_logits.mean(dim=-1)
    ce_tokens = (1.0 - label_smoothing) * nll_tokens + label_smoothing * smooth_tokens
    z_tokens = log_normalizer.square()

    valid_float = valid.to(ce_tokens.dtype)
    token_counts = valid_float.sum(dim=1)
    sample_valid = token_counts.gt(0)

    def reduce(values: torch.Tensor) -> torch.Tensor:
        masked = values * valid_float
        if reduction == "token_mean":
            return masked.sum() / valid_float.sum()
        per_sample = masked.sum(dim=1) / token_counts.clamp_min(1.0)
        return per_sample[sample_valid].mean()

    nll = reduce(nll_tokens)
    smooth = reduce(smooth_tokens)
    z_loss = reduce(z_tokens)
    ce_loss = reduce(ce_tokens)
    loss = ce_loss + z_loss_weight * z_loss
    predictions = shifted_logits.argmax(dim=-1)
    accuracy = ((predictions == shifted_labels) & valid).sum().float() / valid.sum()

    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Non-finite causal language-modeling loss")

    return CausalTokenLoss(
        loss=loss,
        nll_loss=nll,
        smooth_loss=smooth,
        z_loss=z_loss,
        token_accuracy=accuracy,
        num_supervised_tokens=valid.sum().to(dtype=torch.float32),
        mean_tokens_per_sample=token_counts[sample_valid].mean(),
    )
