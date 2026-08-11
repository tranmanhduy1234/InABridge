# -*- coding: utf-8 -*-
"""Instruction-aware Q-Former used by InA-Bridge.

The module keeps the three masking regimes introduced by BLIP-2 (ITC, ITM
and ITG), and the instruction/query interaction used by InstructBLIP.  The
text backbone is DeBERTa-v3 compatible: word embeddings, disentangled
self-attention and feed-forward blocks can be warm-started from a Hugging Face
DeBERTa-v2/v3 checkpoint.  Cross-attention is deliberately ordinary scaled
dot-product attention because one-dimensional DeBERTa relative positions are
not meaningful for DINO patch tokens.

Only learned query tokens cross-attend to the image.  Text and query tokens do
share self-attention, subject to the task-specific mask.  Relative position
terms are applied to the text-to-text sub-matrix only; learned queries have no
ordered one-dimensional positions, matching the position-free query prefix in
BLIP-2.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.config import IMAGE_ENCODER_OUT_DIMENSION, NUM_QUERIES


DEBERTA_CONFIG = {
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout": 0.1,
    "layer_norm_eps": 1e-7,
    "max_relative_positions": 512,
    # DeBERTa-v3-base log-buckets long distances into 256 positions on each
    # side.  A smaller value is selected automatically for tiny test models.
    "position_buckets": 256,
    "vocab_size": 128_100,
    "initializer_range": 0.02,
}


def _require_positive(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _validate_binary_mask(
    mask: torch.Tensor,
    expected_shape: Tuple[int, int],
    name: str,
) -> torch.Tensor:
    if mask.ndim != 2 or tuple(mask.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(mask.shape)}")
    if mask.dtype.is_floating_point and not bool(torch.isfinite(mask).all()):
        raise ValueError(f"{name} contains NaN or infinity")
    if bool(((mask != 0) & (mask != 1)).any()):
        raise ValueError(f"{name} must contain only 0/1 or boolean values")
    if bool(mask.eq(0).all(dim=1).any()):
        raise ValueError(f"every sample in {name} must contain at least one valid token")
    return mask.to(dtype=torch.bool)


def _log_bucket_positions(
    relative_positions: torch.Tensor,
    bucket_size: int,
    max_position: int,
) -> torch.Tensor:
    """DeBERTa-v2/v3 logarithmic relative-position bucketing.

    Near positions remain exact and distant positions are compressed
    logarithmically.  The implementation mirrors Hugging Face DeBERTa but
    keeps constants on the source tensor's device.
    """

    if bucket_size <= 0:
        return relative_positions
    midpoint = bucket_size // 2
    if midpoint < 1 or max_position <= midpoint:
        return relative_positions.clamp(-bucket_size + 1, bucket_size - 1)

    sign = relative_positions.sign()
    absolute = relative_positions.abs()
    safe_absolute = torch.where(
        absolute <= midpoint,
        torch.full_like(absolute, midpoint),
        absolute,
    )
    denominator = math.log((max_position - 1) / midpoint)
    logarithmic = (
        torch.ceil(torch.log(safe_absolute.float() / midpoint) / denominator * (midpoint - 1))
        + midpoint
    ).to(dtype=relative_positions.dtype)
    bucketed = torch.where(absolute <= midpoint, relative_positions, logarithmic * sign)
    return bucketed.clamp(-bucket_size + 1, bucket_size - 1)


class DebertaTextEmbeddings(nn.Module):
    """DeBERTa-v3 word embedding, layer normalization and dropout."""

    def __init__(
        self,
        vocab_size: int = DEBERTA_CONFIG["vocab_size"],
        hidden_size: int = DEBERTA_CONFIG["hidden_size"],
        dropout: float = DEBERTA_CONFIG["hidden_dropout_prob"],
        layer_norm_eps: float = DEBERTA_CONFIG["layer_norm_eps"],
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        _require_positive("vocab_size", vocab_size)
        _require_positive("hidden_size", hidden_size)
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
        # Keep the canonical capitalization for direct DeBERTa state transfer.
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [batch, length], got {tuple(input_ids.shape)}")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must be an int32 or int64 tensor")
        embeddings = self.word_embeddings(input_ids)
        return self.dropout(self.LayerNorm(embeddings))

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def unfreeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(True)


class DisentangledSelfAttn(nn.Module):
    """Memory-conscious DeBERTa disentangled self-attention.

    Instead of materializing relative embeddings with shape ``[L, L, H]``,
    this implementation projects the compact ``[2R, H]`` table once and then
    gathers relative scores.  This avoids a very large intermediate at normal
    caption lengths while retaining content-to-content, content-to-position
    and position-to-content terms.
    """

    def __init__(
        self,
        hidden: int = DEBERTA_CONFIG["hidden_size"],
        num_heads: int = DEBERTA_CONFIG["num_attention_heads"],
        dropout: float = DEBERTA_CONFIG["attention_probs_dropout"],
        max_rel: int = DEBERTA_CONFIG["max_relative_positions"],
        position_buckets: Optional[int] = None,
        *,
        relative_dropout: Optional[float] = None,
    ) -> None:
        super().__init__()
        _require_positive("hidden", hidden)
        _require_positive("num_heads", num_heads)
        _require_positive("max_rel", max_rel)
        if hidden % num_heads != 0:
            raise ValueError("hidden must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if relative_dropout is None:
            relative_dropout = dropout
        if not 0.0 <= relative_dropout < 1.0:
            raise ValueError("relative_dropout must be in [0, 1)")

        if position_buckets is None:
            position_buckets = min(DEBERTA_CONFIG["position_buckets"], max_rel)
        if position_buckets < 0:
            raise ValueError("position_buckets must be non-negative")

        self.num_heads = num_heads
        # Historical aliases retained for checkpoints/code written against the
        # first implementation of this repository.
        self.nh = num_heads
        self.head_dim = hidden // num_heads
        self.hd = self.head_dim
        self.max_relative_positions = max_rel
        self.max_rel = max_rel
        self.position_buckets = position_buckets
        self.relative_span = position_buckets if position_buckets > 0 else max_rel
        self.content_scale = math.sqrt(self.head_dim)
        self.relative_scale = math.sqrt(self.head_dim * 3.0)
        self.scale = self.relative_scale

        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.o_proj = nn.Linear(hidden, hidden)
        self.pos_q = nn.Linear(hidden, hidden)
        self.pos_k = nn.Linear(hidden, hidden)
        self.pos_drop = nn.Dropout(relative_dropout)
        self.attn_drop = nn.Dropout(dropout)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def _relative_indices(self, length: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(length, device=device, dtype=torch.long)
        # DeBERTa convention: query_position - key_position.
        relative = positions[:, None] - positions[None, :]
        if self.position_buckets > 0:
            relative = _log_bucket_positions(
                relative, self.position_buckets, self.max_relative_positions
            )
        return (relative + self.relative_span).clamp(0, 2 * self.relative_span - 1)

    @staticmethod
    def _attention_bias(
        attention_mask: Optional[torch.Tensor],
        batch: int,
        length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        mask = attention_mask.to(device=device)
        blocked = torch.finfo(dtype).min

        if mask.ndim == 2:
            if tuple(mask.shape) != (batch, length):
                raise ValueError(
                    f"2-D attention_mask must have shape {(batch, length)}, got {tuple(mask.shape)}"
                )
            allowed = mask if mask.dtype == torch.bool else mask.ne(0)
            if bool(allowed.logical_not().all(dim=-1).any()):
                raise ValueError("attention_mask leaves a sample without an attendable key")
            return torch.zeros(batch, 1, 1, length, dtype=dtype, device=device).masked_fill(
                ~allowed[:, None, None, :], blocked
            )

        if mask.ndim not in (3, 4):
            raise ValueError("attention_mask must be 2-D, 3-D or 4-D")
        if mask.ndim == 3:
            if tuple(mask.shape) != (batch, length, length):
                raise ValueError(
                    "3-D attention_mask must have shape [batch, query_length, key_length]"
                )
            mask = mask.unsqueeze(1)
        elif mask.size(0) != batch or mask.size(-2) not in (1, length) or mask.size(-1) != length:
            raise ValueError("4-D attention_mask has incompatible batch or sequence dimensions")

        if mask.dtype == torch.bool:
            return torch.zeros_like(mask, dtype=dtype).masked_fill(~mask, blocked)
        if not mask.dtype.is_floating_point:
            allowed = mask.ne(0)
            return torch.zeros_like(mask, dtype=dtype).masked_fill(~allowed, blocked)
        if not bool(torch.isfinite(mask).all()):
            # Negative infinity is a legitimate additive mask. NaN/+inf are not.
            if bool(torch.isnan(mask).any()) or bool(torch.isposinf(mask).any()):
                raise ValueError("attention_mask contains NaN or positive infinity")
        # A float 0/1 matrix is treated as an allow-mask. Any negative value
        # identifies an additive attention bias.
        if bool((mask >= 0).all()) and bool(((mask == 0) | (mask == 1)).all()):
            return torch.zeros_like(mask, dtype=dtype).masked_fill(~mask.bool(), blocked)
        return mask.to(dtype=dtype)

    def _relative_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        relative_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        batch, heads, length, _ = query.shape
        expected = (2 * self.relative_span, heads * self.head_dim)
        if relative_embeddings.ndim != 2 or tuple(relative_embeddings.shape) != expected:
            raise ValueError(
                f"relative_embeddings must have shape {expected}, "
                f"got {tuple(relative_embeddings.shape)}"
            )

        relative_embeddings = self.pos_drop(relative_embeddings)
        pos_key = self.pos_k(relative_embeddings).view(
            2 * self.relative_span, heads, self.head_dim
        ).permute(1, 0, 2)
        pos_query = self.pos_q(relative_embeddings).view(
            2 * self.relative_span, heads, self.head_dim
        ).permute(1, 0, 2)
        indices = self._relative_indices(length, query.device)

        # [B, h, query, 2R] -> gather relative key for each (query, key).
        c2p_all = torch.einsum("bhid,hrd->bhir", query, pos_key)
        c2p = torch.gather(
            c2p_all,
            -1,
            indices[None, None, :, :].expand(batch, heads, -1, -1),
        )

        # [B, h, key, 2R], gathered in [key, query] order then transposed.
        p2c_all = torch.einsum("bhjd,hrd->bhjr", key, pos_query)
        p2c = torch.gather(
            p2c_all,
            -1,
            indices.transpose(0, 1)[None, None, :, :].expand(batch, heads, -1, -1),
        ).transpose(-2, -1)
        return (c2p + p2c) / self.relative_scale

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        relative_embeddings: Optional[torch.Tensor] = None,
        relative_position_start: int = 0,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape [batch, length, hidden], got {tuple(x.shape)}")
        batch, length, hidden = x.shape
        if hidden != self.num_heads * self.head_dim:
            raise ValueError("x hidden dimension does not match the attention module")
        if not 0 <= relative_position_start <= length:
            raise ValueError("relative_position_start must be between 0 and sequence length")

        query = self._split_heads(self.q_proj(x))
        key = self._split_heads(self.k_proj(x))
        value = self._split_heads(self.v_proj(x))
        raw_content_scores = torch.matmul(query, key.transpose(-2, -1))
        # Learned queries do not use relative-position terms, so their content
        # scores retain the ordinary Transformer scale sqrt(d). Text-to-text
        # scores are replaced below by DeBERTa's sqrt(3d) scaling.
        scores = raw_content_scores / self.content_scale

        if relative_embeddings is not None and relative_position_start < length:
            start = relative_position_start
            relative = self._relative_scores(
                query[:, :, start:, :], key[:, :, start:, :], relative_embeddings
            )
            # Avoid an in-place update on a view; it is fragile under activation
            # checkpointing and compiled autograd.
            text_delta = (
                raw_content_scores[:, :, start:, start:] / self.relative_scale
                + relative
                - scores[:, :, start:, start:]
            )
            relative_full = scores.new_zeros(scores.shape)
            relative_full[:, :, start:, start:] = text_delta
            scores = scores + relative_full

        bias = self._attention_bias(attention_mask, batch, length, scores.dtype, scores.device)
        if bias is not None:
            scores = scores + bias

        # fp32 softmax prevents underflow/overflow in bf16/fp16 training while
        # preserving gradients to the original projections.
        probabilities = F.softmax(scores.float(), dim=-1).to(dtype=value.dtype)
        probabilities = self.attn_drop(probabilities)
        context = torch.matmul(probabilities, value)
        context = context.transpose(1, 2).contiguous().view(batch, length, hidden)
        return self.o_proj(context)


class CrossAttnLayer(nn.Module):
    """Standard multi-head cross-attention from learned queries to patches."""

    def __init__(
        self,
        hidden: int = DEBERTA_CONFIG["hidden_size"],
        image_dim: int = IMAGE_ENCODER_OUT_DIMENSION,
        num_heads: int = DEBERTA_CONFIG["num_attention_heads"],
        dropout: float = DEBERTA_CONFIG["attention_probs_dropout"],
        layer_norm_eps: float = DEBERTA_CONFIG["layer_norm_eps"],
    ) -> None:
        super().__init__()
        if hidden % num_heads != 0:
            raise ValueError("hidden must be divisible by num_heads")
        self.hidden_size = hidden
        self.image_dim = image_dim
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.dropout_probability = dropout

        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(image_dim, hidden)
        self.v_proj = nn.Linear(image_dim, hidden)
        self.o_proj = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden, eps=layer_norm_eps)
        self.drop = nn.Dropout(dropout)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query_tokens: torch.Tensor,
        image_features: torch.Tensor,
        image_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if query_tokens.ndim != 3 or query_tokens.size(-1) != self.hidden_size:
            raise ValueError(
                f"query_tokens must have shape [batch, queries, {self.hidden_size}]"
            )
        if image_features.ndim != 3 or image_features.size(-1) != self.image_dim:
            raise ValueError(
                f"image_features must have shape [batch, patches, {self.image_dim}]"
            )
        if query_tokens.size(0) != image_features.size(0):
            raise ValueError("query and image batch sizes must match")
        if image_features.size(1) == 0:
            raise ValueError("image_features must contain at least one patch")

        attention_mask = None
        if image_attention_mask is not None:
            valid = _validate_binary_mask(
                image_attention_mask,
                (image_features.size(0), image_features.size(1)),
                "image_attention_mask",
            ).to(device=image_features.device)
            # Boolean SDPA masks use True for allowed positions.
            attention_mask = valid[:, None, None, :]

        query = self._split_heads(self.q_proj(query_tokens))
        key = self._split_heads(self.k_proj(image_features))
        value = self._split_heads(self.v_proj(image_features))
        context = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout_probability if self.training else 0.0,
            is_causal=False,
        )
        batch, _, query_length, _ = context.shape
        context = context.transpose(1, 2).contiguous().view(batch, query_length, self.hidden_size)
        return self.norm(query_tokens + self.drop(self.o_proj(context)))


class DebertaLayer(nn.Module):
    """Post-normalized DeBERTa self-attention and feed-forward block."""

    def __init__(self, cfg: Mapping[str, object] = DEBERTA_CONFIG) -> None:
        super().__init__()
        hidden = int(cfg["hidden_size"])
        intermediate = int(cfg["intermediate_size"])
        dropout = float(cfg["hidden_dropout_prob"])
        eps = float(cfg["layer_norm_eps"])
        self.attn = DisentangledSelfAttn(
            hidden=hidden,
            num_heads=int(cfg["num_attention_heads"]),
            dropout=float(cfg["attention_probs_dropout"]),
            relative_dropout=dropout,
            max_rel=int(cfg["max_relative_positions"]),
            position_buckets=int(cfg["position_buckets"]),
        )
        self.norm1 = nn.LayerNorm(hidden, eps=eps)
        self.fc1 = nn.Linear(hidden, intermediate)
        self.fc2 = nn.Linear(intermediate, hidden)
        self.norm2 = nn.LayerNorm(hidden, eps=eps)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def self_attention(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        relative_embeddings: torch.Tensor,
        relative_position_start: int,
    ) -> torch.Tensor:
        attention_output = self.attn(
            hidden,
            attention_mask,
            relative_embeddings=relative_embeddings,
            relative_position_start=relative_position_start,
        )
        return self.norm1(hidden + self.drop(attention_output))

    def feed_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        output = self.fc2(self.drop(self.act(self.fc1(hidden))))
        return self.norm2(hidden + self.drop(output))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        *,
        relative_embeddings: Optional[torch.Tensor] = None,
        relative_position_start: int = 0,
    ) -> torch.Tensor:
        if relative_embeddings is None:
            raise ValueError("relative_embeddings are required for DebertaLayer")
        hidden = self.self_attention(x, mask, relative_embeddings, relative_position_start)
        return self.feed_forward(hidden)


class QFormer(nn.Module):
    """BLIP-2/InstructBLIP querying transformer with a DeBERTa text backbone.

    Public methods ``forward``, ``forward_itc``, ``forward_itm`` and
    ``forward_itg`` intentionally retain the API used by :mod:`src.model`.
    """

    _ATTENTION_MODES = frozenset({"instruction", "bidirectional", "unimodal", "causal"})

    def __init__(
        self,
        num_queries: int = NUM_QUERIES,
        image_hidden_dim: int = IMAGE_ENCODER_OUT_DIMENSION,
        cross_attn_every: int = 2,
        freeze_text_embeds: bool = False,
        hidden_size: int = DEBERTA_CONFIG["hidden_size"],
        num_hidden_layers: int = DEBERTA_CONFIG["num_hidden_layers"],
        num_attention_heads: int = DEBERTA_CONFIG["num_attention_heads"],
        intermediate_size: int = DEBERTA_CONFIG["intermediate_size"],
        vocab_size: int = DEBERTA_CONFIG["vocab_size"],
        max_relative_positions: int = DEBERTA_CONFIG["max_relative_positions"],
        position_buckets: Optional[int] = None,
        padding_idx: Optional[int] = 0,
        hidden_dropout_prob: float = DEBERTA_CONFIG["hidden_dropout_prob"],
        attention_dropout_prob: float = DEBERTA_CONFIG["attention_probs_dropout"],
        layer_norm_eps: float = DEBERTA_CONFIG["layer_norm_eps"],
        itc_dimension: int = 256,
        initial_temperature: float = 0.07,
        validate_numerics: bool = False,
    ) -> None:
        super().__init__()
        for name, value in (
            ("num_queries", num_queries),
            ("image_hidden_dim", image_hidden_dim),
            ("cross_attn_every", cross_attn_every),
            ("hidden_size", hidden_size),
            ("num_hidden_layers", num_hidden_layers),
            ("num_attention_heads", num_attention_heads),
            ("intermediate_size", intermediate_size),
            ("vocab_size", vocab_size),
            ("max_relative_positions", max_relative_positions),
            ("itc_dimension", itc_dimension),
        ):
            _require_positive(name, value)
        if hidden_size % num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        for name, value in (
            ("hidden_dropout_prob", hidden_dropout_prob),
            ("attention_dropout_prob", attention_dropout_prob),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if not math.isfinite(initial_temperature) or initial_temperature <= 0:
            raise ValueError("initial_temperature must be finite and positive")
        if padding_idx is not None and not 0 <= padding_idx < vocab_size:
            raise ValueError("padding_idx must be inside the configured vocabulary")

        if position_buckets is None:
            position_buckets = min(DEBERTA_CONFIG["position_buckets"], max_relative_positions)
        if not isinstance(position_buckets, int) or position_buckets < 0:
            raise ValueError("position_buckets must be a non-negative integer")
        relative_span = position_buckets if position_buckets > 0 else max_relative_positions

        self.config = {
            **DEBERTA_CONFIG,
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "intermediate_size": intermediate_size,
            "vocab_size": vocab_size,
            "max_relative_positions": max_relative_positions,
            "position_buckets": position_buckets,
            "padding_idx": padding_idx,
            "hidden_dropout_prob": hidden_dropout_prob,
            "attention_probs_dropout": attention_dropout_prob,
            "layer_norm_eps": layer_norm_eps,
        }
        self.num_queries = num_queries
        self.image_hidden_dim = image_hidden_dim
        self.hidden_dim = hidden_size
        self.num_layers = num_hidden_layers
        self.cross_attn_every = cross_attn_every
        self.freeze_text_embeds = bool(freeze_text_embeds)
        self.validate_numerics = bool(validate_numerics)
        self.gradient_checkpointing = False

        self.query_tokens = nn.Parameter(torch.empty(1, num_queries, hidden_size))
        self.dec_token = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.q_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.text_embeddings = DebertaTextEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            dropout=hidden_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            padding_idx=padding_idx,
        )

        self.relative_embeddings = nn.Embedding(2 * relative_span, hidden_size)
        self.relative_embeddings_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.deb_layers = nn.ModuleList([DebertaLayer(self.config) for _ in range(num_hidden_layers)])
        self.cross_attention_layer_indices = tuple(
            index for index in range(num_hidden_layers) if index % cross_attn_every == 0
        )
        self.cross_layers = nn.ModuleList(
            CrossAttnLayer(
                hidden=hidden_size,
                image_dim=image_hidden_dim,
                num_heads=num_attention_heads,
                dropout=attention_dropout_prob,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in self.cross_attention_layer_indices
        )

        self.itc_query_proj = nn.Linear(hidden_size, itc_dimension)
        self.itc_text_proj = nn.Linear(hidden_size, itc_dimension)
        self.itc_temp = nn.Parameter(torch.tensor(float(initial_temperature)))
        self.itm_head = nn.Linear(hidden_size, 2)

        # BLIP-2 uses a complete language-model prediction transform rather
        # than a bare projection.  The final decoder remains weight-tied with
        # the input word embedding table.
        self.itg_transform_dense = nn.Linear(hidden_size, hidden_size)
        self.itg_transform_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.itg_lm_head = nn.Linear(hidden_size, vocab_size, bias=True)

        self.apply(self._initialize_weights)
        nn.init.normal_(self.query_tokens, mean=0.0, std=self.config["initializer_range"])
        nn.init.normal_(self.dec_token, mean=0.0, std=self.config["initializer_range"])
        self.tie_weights()
        self.enforce_freeze_policy()

    def _initialize_weights(self, module: nn.Module) -> None:
        std = float(self.config["initializer_range"])
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def tie_weights(self) -> None:
        """Tie ITG output weights to DeBERTa input embeddings."""

        if self.itg_lm_head.weight.shape != self.text_embeddings.word_embeddings.weight.shape:
            raise RuntimeError("ITG decoder and text embedding shapes are incompatible")
        self.itg_lm_head.weight = self.text_embeddings.word_embeddings.weight

    @property
    def is_gradient_checkpointing(self) -> bool:
        return self.gradient_checkpointing

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def set_trainable(self, trainable: bool, *, respect_frozen_text: bool = True) -> None:
        """Set Q-Former trainability without accidentally unfreezing text embeddings.

        ``src.model`` should use this method instead of iterating over all
        parameters whenever ``freeze_text_embeds=True`` is configured.
        """

        for parameter in self.parameters():
            parameter.requires_grad_(trainable)
        if trainable and respect_frozen_text:
            self.enforce_freeze_policy()

    def enforce_freeze_policy(self) -> None:
        if self.freeze_text_embeds:
            self.text_embeddings.freeze()
            # Weight tying means this also freezes ``itg_lm_head.weight``;
            # its independent bias and transform remain trainable.

    def _relative_embedding_values(self) -> torch.Tensor:
        return self.relative_embeddings_norm(self.relative_embeddings.weight)

    def _validate_image_features(self, image_features: torch.Tensor) -> None:
        if image_features.ndim != 3 or image_features.size(-1) != self.image_hidden_dim:
            raise ValueError(
                f"image_features must have shape [batch, patches, {self.image_hidden_dim}], "
                f"got {tuple(image_features.shape)}"
            )
        if image_features.size(0) == 0 or image_features.size(1) == 0:
            raise ValueError("image_features must contain a non-empty batch and patch sequence")
        if not image_features.dtype.is_floating_point:
            raise TypeError("image_features must be floating point")
        if image_features.device != self.query_tokens.device:
            raise ValueError(
                f"image_features are on {image_features.device}, Q-Former is on "
                f"{self.query_tokens.device}"
            )
        if self.validate_numerics and not bool(torch.isfinite(image_features).all()):
            raise FloatingPointError("image_features contain NaN or infinity")

    def _validate_ids(self, input_ids: torch.Tensor, batch_size: Optional[int], name: str) -> None:
        if input_ids.ndim != 2:
            raise ValueError(f"{name} must have shape [batch, length], got {tuple(input_ids.shape)}")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} must be an int32 or int64 tensor")
        if batch_size is not None and input_ids.size(0) != batch_size:
            raise ValueError(f"{name} batch size does not match image_features")
        if input_ids.size(1) == 0:
            raise ValueError(f"{name} must contain at least one token")
        if input_ids.device != self.query_tokens.device:
            raise ValueError(f"{name} and Q-Former must be on the same device")
        if self.validate_numerics:
            if bool(input_ids.lt(0).any()) or bool(input_ids.ge(self.config["vocab_size"]).any()):
                raise ValueError(f"{name} contains token IDs outside the configured vocabulary")

    def build_attention_mask(
        self,
        batch_size: int,
        num_queries: int,
        instruction_mask: Optional[torch.Tensor],
        device: torch.device,
        mode: str = "instruction",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Build the additive query/text self-attention mask.

        Modes:
          * ``instruction``: queries read instruction; text remains text-only.
          * ``bidirectional``: queries and text interact in both directions.
          * ``unimodal``: query and text branches are completely separated.
          * ``causal``: query is a visual prefix; text reads query and past text.
        """

        _require_positive("batch_size", batch_size)
        _require_positive("num_queries", num_queries)
        if mode not in self._ATTENTION_MODES:
            raise ValueError(f"unknown attention mode {mode!r}; expected {sorted(self._ATTENTION_MODES)}")
        if not dtype.is_floating_point:
            raise TypeError("attention-mask dtype must be floating point")
        if instruction_mask is None:
            raise ValueError("instruction_mask is required when text tokens are present")
        valid_text = _validate_binary_mask(
            instruction_mask,
            (batch_size, instruction_mask.size(1) if instruction_mask.ndim == 2 else -1),
            "instruction_mask",
        ).to(device=device)
        text_length = valid_text.size(1)
        total_length = num_queries + text_length
        blocked = torch.finfo(dtype).min
        mask = torch.zeros(batch_size, 1, total_length, total_length, device=device, dtype=dtype)

        if mode == "instruction":
            mask[:, :, num_queries:, :num_queries] = blocked
        elif mode == "unimodal":
            mask[:, :, :num_queries, num_queries:] = blocked
            mask[:, :, num_queries:, :num_queries] = blocked
        elif mode == "causal":
            mask[:, :, :num_queries, num_queries:] = blocked
            future = torch.triu(
                torch.ones(text_length, text_length, dtype=torch.bool, device=device), diagonal=1
            )
            mask[:, :, num_queries:, num_queries:].masked_fill_(future, blocked)

        # Padding tokens are never keys. Padding query rows remain defined so
        # attention softmax never receives a fully-masked row; their outputs are
        # subsequently ignored by pooling/loss masks.
        mask[:, :, :, num_queries:].masked_fill_(~valid_text[:, None, None, :], blocked)
        return mask

    def build_instruct_attention_mask(
        self,
        batch_size: int,
        num_queries: int,
        instruction_mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Backward-compatible alias for the instruction mask builder."""

        return self.build_attention_mask(
            batch_size, num_queries, instruction_mask, device, mode="instruction"
        )

    def _run_one_layer(
        self,
        layer_index: int,
        hidden: torch.Tensor,
        image_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        image_attention_mask: Optional[torch.Tensor],
        relative_embeddings: torch.Tensor,
        relative_position_start: int,
    ) -> torch.Tensor:
        layer = self.deb_layers[layer_index]
        hidden = layer.self_attention(
            hidden, attention_mask, relative_embeddings, relative_position_start
        )
        if image_features is not None and layer_index in self._cross_layer_lookup:
            cross_layer = self.cross_layers[self._cross_layer_lookup[layer_index]]
            updated_queries = cross_layer(
                hidden[:, : self.num_queries], image_features, image_attention_mask
            )
            hidden = torch.cat([updated_queries, hidden[:, self.num_queries :]], dim=1)
        return layer.feed_forward(hidden)

    @property
    def _cross_layer_lookup(self) -> Dict[int, int]:
        # At most six entries for the production configuration. Constructing
        # this tiny mapping avoids registering mutable state in checkpoints.
        return {layer_index: index for index, layer_index in enumerate(self.cross_attention_layer_indices)}

    def _run_layers(
        self,
        hidden: torch.Tensor,
        image_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        *,
        image_attention_mask: Optional[torch.Tensor] = None,
        relative_position_start: int = 0,
    ) -> torch.Tensor:
        relative_embeddings = self._relative_embedding_values()
        for layer_index in range(self.num_layers):
            if self.gradient_checkpointing and self.training:
                def custom_forward(current_hidden: torch.Tensor, index: int = layer_index) -> torch.Tensor:
                    return self._run_one_layer(
                        index,
                        current_hidden,
                        image_features,
                        attention_mask,
                        image_attention_mask,
                        relative_embeddings,
                        relative_position_start,
                    )

                hidden = checkpoint(custom_forward, hidden, use_reentrant=False)
            else:
                hidden = self._run_one_layer(
                    layer_index,
                    hidden,
                    image_features,
                    attention_mask,
                    image_attention_mask,
                    relative_embeddings,
                    relative_position_start,
                )
        return hidden

    def forward_instruction_aware(
        self,
        image_features: torch.Tensor,
        instruction_ids: Optional[torch.Tensor] = None,
        instruction_mask: Optional[torch.Tensor] = None,
        instruction_embeds: Optional[torch.Tensor] = None,
        attention_mode: str = "instruction",
        image_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Extract instruction-conditioned visual query representations."""

        self._validate_image_features(image_features)
        batch = image_features.size(0)
        if instruction_ids is not None and instruction_embeds is not None:
            raise ValueError("provide either instruction_ids or instruction_embeds, not both")
        if attention_mode not in self._ATTENTION_MODES:
            raise ValueError(f"unknown attention mode {attention_mode!r}")
        if image_attention_mask is not None:
            image_attention_mask = _validate_binary_mask(
                image_attention_mask,
                (batch, image_features.size(1)),
                "image_attention_mask",
            ).to(device=image_features.device)

        queries = self.q_norm(self.query_tokens.expand(batch, -1, -1))
        text_embeddings: Optional[torch.Tensor] = None
        if instruction_ids is not None:
            self._validate_ids(instruction_ids, batch, "instruction_ids")
            text_embeddings = self.text_embeddings(instruction_ids)
        elif instruction_embeds is not None:
            if instruction_embeds.ndim != 3 or tuple(instruction_embeds.shape[:1]) != (batch,):
                raise ValueError("instruction_embeds must have shape [batch, length, hidden]")
            if instruction_embeds.size(1) == 0 or instruction_embeds.size(2) != self.hidden_dim:
                raise ValueError(
                    f"instruction_embeds must have a non-empty sequence and hidden size {self.hidden_dim}"
                )
            if instruction_embeds.device != queries.device:
                raise ValueError("instruction_embeds and Q-Former must be on the same device")
            if not instruction_embeds.dtype.is_floating_point:
                raise TypeError("instruction_embeds must be floating point")
            if self.validate_numerics and not bool(torch.isfinite(instruction_embeds).all()):
                raise FloatingPointError("instruction_embeds contain NaN or infinity")
            text_embeddings = instruction_embeds.to(dtype=queries.dtype)

        if text_embeddings is None:
            if instruction_mask is not None:
                raise ValueError("instruction_mask was provided without instruction text")
            hidden = queries
            self_attention_mask = None
            relative_position_start = self.num_queries
        else:
            text_length = text_embeddings.size(1)
            if instruction_mask is None:
                instruction_mask = torch.ones(
                    batch, text_length, dtype=torch.bool, device=text_embeddings.device
                )
            else:
                instruction_mask = _validate_binary_mask(
                    instruction_mask, (batch, text_length), "instruction_mask"
                ).to(device=text_embeddings.device)
            hidden = torch.cat([queries, text_embeddings], dim=1)
            self_attention_mask = self.build_attention_mask(
                batch,
                self.num_queries,
                instruction_mask,
                hidden.device,
                mode=attention_mode,
                dtype=hidden.dtype,
            )
            # Queries are an unordered learned set. Relative positions only
            # apply inside the text suffix.
            relative_position_start = self.num_queries

        hidden = self._run_layers(
            hidden,
            image_features,
            self_attention_mask,
            image_attention_mask=image_attention_mask,
            relative_position_start=relative_position_start,
        )
        query_output = hidden[:, : self.num_queries]
        if self.validate_numerics and not bool(torch.isfinite(query_output).all()):
            raise FloatingPointError("Q-Former produced non-finite query representations")
        return {
            "query_output": query_output,
            "hidden_states": hidden,
            "text_output": hidden[:, self.num_queries :],
        }

    def forward_image_only(
        self,
        image_features: torch.Tensor,
        image_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.forward_instruction_aware(
            image_features=image_features,
            image_attention_mask=image_attention_mask,
        )

    def forward(
        self,
        image_features: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.forward_instruction_aware(
            image_features=image_features,
            instruction_ids=input_ids,
            instruction_mask=attention_mask,
            image_attention_mask=image_attention_mask,
        )

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode text independently with the shared DeBERTa blocks."""

        self._validate_ids(input_ids, None, "input_ids")
        batch, length = input_ids.shape
        if attention_mask is None:
            valid = torch.ones(batch, length, dtype=torch.bool, device=input_ids.device)
        else:
            valid = _validate_binary_mask(
                attention_mask, (batch, length), "attention_mask"
            ).to(device=input_ids.device)
        hidden = self.text_embeddings(input_ids)
        hidden = self._run_layers(
            hidden,
            image_features=None,
            attention_mask=valid,
            relative_position_start=0,
        )
        # DeBERTa tokenizers normally place [CLS] first. Selecting the first
        # valid token also handles left-padded or custom tokenized inputs.
        first_valid = valid.to(torch.long).argmax(dim=1)
        pooled = hidden[torch.arange(batch, device=hidden.device), first_valid]
        return {"last_hidden_state": hidden, "pooled_output": pooled}

    def forward_itc(
        self,
        image_features: torch.Tensor,
        caption_ids: torch.Tensor,
        caption_mask: Optional[torch.Tensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return normalized multi-query image and pooled text embeddings."""

        self._validate_image_features(image_features)
        self._validate_ids(caption_ids, image_features.size(0), "caption_ids")
        image_output = self.forward_image_only(image_features, image_attention_mask)
        text_output = self.encode_text(caption_ids, caption_mask)
        query_features = F.normalize(self.itc_query_proj(image_output["query_output"]), dim=-1)
        text_features = F.normalize(self.itc_text_proj(text_output["pooled_output"]), dim=-1)
        return query_features, text_features

    def forward_itm(
        self,
        image_features: torch.Tensor,
        caption_ids: torch.Tensor,
        caption_mask: Optional[torch.Tensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Classify matched/mismatched image-text pairs from fused queries."""

        self._validate_ids(caption_ids, image_features.size(0), "caption_ids")
        output = self.forward_instruction_aware(
            image_features=image_features,
            instruction_ids=caption_ids,
            instruction_mask=caption_mask,
            attention_mode="bidirectional",
            image_attention_mask=image_attention_mask,
        )
        # BLIP-2 predicts with every learned query and averages the evidence.
        return self.itm_head(output["query_output"]).mean(dim=1)

    def forward_itg(
        self,
        image_features: torch.Tensor,
        caption_ids: torch.Tensor,
        caption_mask: Optional[torch.Tensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Produce causal image-grounded next-token logits for Stage 1."""

        self._validate_ids(caption_ids, image_features.size(0), "caption_ids")
        if caption_ids.size(1) < 2:
            raise ValueError("ITG requires at least two caption tokens for next-token prediction")
        caption_embeddings = self.text_embeddings(caption_ids)
        caption_embeddings = torch.cat(
            [self.dec_token.expand(caption_ids.size(0), -1, -1), caption_embeddings[:, 1:]],
            dim=1,
        )
        output = self.forward_instruction_aware(
            image_features=image_features,
            instruction_embeds=caption_embeddings,
            instruction_mask=caption_mask,
            attention_mode="causal",
            image_attention_mask=image_attention_mask,
        )
        text_hidden = output["text_output"]
        prediction_hidden = self.itg_transform_norm(
            F.gelu(self.itg_transform_dense(text_hidden))
        )
        logits = self.itg_lm_head(prediction_hidden)
        if self.validate_numerics and not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("ITG head produced non-finite logits")
        return logits

    @staticmethod
    def _resolve_module(root: nn.Module, path: str) -> Optional[nn.Module]:
        value: object = root
        for component in path.split("."):
            value = getattr(value, component, None)
            if value is None:
                return None
        return value if isinstance(value, nn.Module) else None

    @staticmethod
    def _copy_module_parameters(target: nn.Module, source: nn.Module) -> int:
        """Copy same-named, same-shaped parameters/buffers and return tensor count."""

        target_state = target.state_dict()
        source_state = source.state_dict()
        compatible = {
            name: value
            for name, value in source_state.items()
            if name in target_state and tuple(target_state[name].shape) == tuple(value.shape)
        }
        if not compatible:
            return 0
        target.load_state_dict(compatible, strict=False)
        return len(compatible)

    def initialize_from_pretrained(
        self,
        model_id: Union[str, Path],
        *,
        local_files_only: bool = False,
        revision: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        pretrained_model: Optional[nn.Module] = None,
    ) -> int:
        """Warm-start compatible text parameters from DeBERTa-v2/v3.

        Supplying ``pretrained_model`` makes the method fully offline and is
        useful for controlled initialization/tests.  Otherwise Hugging Face is
        asked to resolve ``model_id``; this method itself is never called during
        ordinary forward passes.
        """

        if pretrained_model is None:
            from transformers import AutoModel

            pretrained_model = AutoModel.from_pretrained(
                str(model_id),
                local_files_only=local_files_only,
                revision=revision,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
            )
        backbone = pretrained_model
        # Also accept a task head exposing the base encoder as ``deberta``.
        if not hasattr(backbone, "embeddings") and hasattr(backbone, "deberta"):
            backbone = backbone.deberta

        embeddings = getattr(backbone, "embeddings", None)
        encoder = getattr(backbone, "encoder", None)
        source_layers = getattr(encoder, "layer", None)
        if not isinstance(embeddings, nn.Module) or source_layers is None:
            raise ValueError(f"{model_id!s} is not a compatible DeBERTa-v2/v3 encoder")
        if len(source_layers) != self.num_layers:
            raise ValueError(
                f"layer mismatch: pretrained={len(source_layers)}, qformer={self.num_layers}"
            )

        source_config = getattr(backbone, "config", getattr(pretrained_model, "config", None))
        if source_config is not None:
            for name, target_value in (
                ("hidden_size", self.hidden_dim),
                ("num_attention_heads", self.config["num_attention_heads"]),
            ):
                source_value = getattr(source_config, name, target_value)
                if source_value != target_value:
                    raise ValueError(
                        f"pretrained {name}={source_value} is incompatible with Q-Former {target_value}"
                    )

        copied = self._copy_module_parameters(self.text_embeddings, embeddings)
        required_embedding = self.text_embeddings.word_embeddings.weight
        source_word_embeddings = getattr(embeddings, "word_embeddings", None)
        if (
            not isinstance(source_word_embeddings, nn.Embedding)
            or source_word_embeddings.weight.shape != required_embedding.shape
        ):
            raise ValueError(
                "pretrained word embedding shape does not match the configured Q-Former vocabulary"
            )

        base_mappings = (
            ("attention.self.query_proj", "attn.q_proj"),
            ("attention.self.key_proj", "attn.k_proj"),
            ("attention.self.value_proj", "attn.v_proj"),
            ("attention.output.dense", "attn.o_proj"),
            ("attention.output.LayerNorm", "norm1"),
            ("intermediate.dense", "fc1"),
            ("output.dense", "fc2"),
            ("output.LayerNorm", "norm2"),
        )
        for layer_number, (source_layer, target_layer) in enumerate(
            zip(source_layers, self.deb_layers)
        ):
            source_self_attention = self._resolve_module(source_layer, "attention.self")
            if source_self_attention is None:
                raise ValueError(
                    f"pretrained layer {layer_number} has no DeBERTa self-attention module"
                )
            if bool(getattr(source_self_attention, "share_att_key", False)):
                # DeBERTa-v3-base reuses its content Q/K projections for
                # positional terms. Q-Former keeps separate modules so they can
                # later specialize, but starts them from those shared values.
                position_mappings = (
                    ("attention.self.query_proj", "attn.pos_q"),
                    ("attention.self.key_proj", "attn.pos_k"),
                )
            else:
                position_mappings = (
                    ("attention.self.pos_query_proj", "attn.pos_q"),
                    ("attention.self.pos_key_proj", "attn.pos_k"),
                )
            for source_path, target_path in base_mappings + position_mappings:
                source_module = self._resolve_module(source_layer, source_path)
                target_module = self._resolve_module(target_layer, target_path)
                if source_module is None or target_module is None:
                    raise ValueError(
                        f"pretrained layer {layer_number} is missing required module {source_path}"
                    )
                count = self._copy_module_parameters(target_module, source_module)
                if count == 0:
                    raise ValueError(
                        f"pretrained module {source_path} has no shape-compatible parameters"
                    )
                copied += count

        source_relative = getattr(encoder, "rel_embeddings", None)
        if not isinstance(source_relative, nn.Embedding):
            raise ValueError("pretrained DeBERTa encoder has no relative-position embeddings")
        if source_relative.weight.shape != self.relative_embeddings.weight.shape:
            raise ValueError(
                "relative-position table mismatch: "
                f"pretrained={tuple(source_relative.weight.shape)}, "
                f"qformer={tuple(self.relative_embeddings.weight.shape)}; check position_buckets"
            )
        with torch.no_grad():
            self.relative_embeddings.weight.copy_(source_relative.weight)
        copied += 1

        source_relative_norm = getattr(encoder, "LayerNorm", None)
        if isinstance(source_relative_norm, nn.Module):
            copied += self._copy_module_parameters(
                self.relative_embeddings_norm, source_relative_norm
            )

        # Re-establish the alias after loading and preserve the explicit freeze
        # contract. Newly introduced queries, cross-attention and task heads
        # intentionally retain their random initialization.
        self.tie_weights()
        self.enforce_freeze_policy()
        return copied
