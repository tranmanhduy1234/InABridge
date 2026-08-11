"""Stable gated SwiGLU projection from Q-Former into Qwen embedding space."""

from __future__ import annotations

import math
from numbers import Integral, Real

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMSNorm with FP32 accumulation for FP16/BF16 activations.

    Keeping this implementation local gives identical behavior across supported
    PyTorch versions and preserves a simple ``norm.weight`` state-dict key.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if not isinstance(dim, Integral) or isinstance(dim, bool) or dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim!r}.")
        if (
            not isinstance(eps, Real)
            or isinstance(eps, bool)
            or not math.isfinite(float(eps))
            or eps <= 0
        ):
            raise ValueError(f"eps must be finite and positive, got {eps!r}.")
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("RMSNorm inputs must be a torch.Tensor.")
        if inputs.shape[-1] != self.dim:
            raise ValueError(
                f"RMSNorm expected last dimension {self.dim}, got {inputs.shape[-1]}."
            )
        accumulation_dtype = (
            torch.float32 if inputs.dtype in (torch.float16, torch.bfloat16) else inputs.dtype
        )
        values = inputs.to(accumulation_dtype)
        inverse_rms = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + self.eps)
        normalized = (values * inverse_rms).to(inputs.dtype)
        return normalized * self.weight

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"


class QwenProjector(nn.Module):
    """Map Q-Former query tokens to the Qwen hidden dimension.

    The projector combines a direct residual projection with a normalized
    SwiGLU branch and a learned per-output-channel gate::

        residual = W_r x
        branch   = W_2 (SiLU(a) * b),  [a, b] = W_1 RMSNorm(x)
        output   = residual + sigmoid(W_g RMSNorm(x)) * branch

    By default ``W_2`` is zero-initialized, so Stage 2 starts from a well-scaled
    linear bridge and introduces nonlinear corrections gradually.  The return
    value remains ``(projected_tokens, gate)`` for the model/trainer interface.
    """

    def __init__(
        self,
        qformer_dim: int = 768,
        hidden_dim: int = 1536,
        llm_dim: int = 2560,
        zero_init_last_layer: bool = True,
        *,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        gate_init_bias: float = -2.0,
        check_finite: bool = False,
    ) -> None:
        super().__init__()
        dimensions = {
            "qformer_dim": qformer_dim,
            "hidden_dim": hidden_dim,
            "llm_dim": llm_dim,
        }
        for name, value in dimensions.items():
            if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if not isinstance(zero_init_last_layer, bool):
            raise TypeError("zero_init_last_layer must be a boolean.")
        if (
            not isinstance(dropout, Real)
            or isinstance(dropout, bool)
            or not math.isfinite(float(dropout))
            or not 0.0 <= float(dropout) < 1.0
        ):
            raise ValueError(f"dropout must be finite and in [0, 1), got {dropout!r}.")
        if (
            not isinstance(gate_init_bias, Real)
            or isinstance(gate_init_bias, bool)
            or not math.isfinite(float(gate_init_bias))
        ):
            raise ValueError("gate_init_bias must be a finite number.")
        if not isinstance(check_finite, bool):
            raise TypeError("check_finite must be a boolean.")

        self.qformer_dim = int(qformer_dim)
        self.hidden_dim = int(hidden_dim)
        self.llm_dim = int(llm_dim)
        self.zero_init_last_layer = zero_init_last_layer
        self.gate_init_bias = float(gate_init_bias)
        self.check_finite = check_finite

        self.norm = RMSNorm(self.qformer_dim, eps=norm_eps)
        self.fc1 = nn.Linear(self.qformer_dim, self.hidden_dim * 2)
        self.fc2 = nn.Linear(self.hidden_dim, self.llm_dim)
        self.residual_proj = nn.Linear(self.qformer_dim, self.llm_dim)
        self.gate = nn.Linear(self.qformer_dim, self.llm_dim)
        self.dropout = nn.Dropout(float(dropout))
        self._reset_parameters()

    @property
    def in_dim(self) -> int:
        return self.qformer_dim

    @property
    def out_dim(self) -> int:
        return self.llm_dim

    @property
    def device(self) -> torch.device:
        return self.fc1.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.fc1.weight.dtype

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @property
    def total_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _reset_parameters(self) -> None:
        """Initialize every branch explicitly and reproducibly.

        A zero gate weight gives all visual channels the same conservative
        initial mixing coefficient.  It does not permanently disable learning:
        ``fc2`` receives gradients immediately and the upstream/gate branches
        begin receiving non-zero gradients once the nonlinear branch departs
        from zero.
        """

        nn.init.ones_(self.norm.weight)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        nn.init.xavier_uniform_(self.residual_proj.weight)
        nn.init.zeros_(self.residual_proj.bias)

        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, self.gate_init_bias)

        if self.zero_init_last_layer:
            nn.init.zeros_(self.fc2.weight)
        else:
            # The residual and nonlinear paths are summed; reduced gain keeps
            # their combined initial variance controlled.
            nn.init.xavier_uniform_(self.fc2.weight, gain=1.0 / math.sqrt(2.0))
        nn.init.zeros_(self.fc2.bias)

    def _validate_input(self, inputs: torch.Tensor) -> None:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("Projector inputs must be a torch.Tensor.")
        if inputs.ndim != 3:
            raise ValueError(
                "QwenProjector expects Q-Former tokens with shape (B, Q, D), "
                f"got {tuple(inputs.shape)}."
            )
        if inputs.shape[0] <= 0 or inputs.shape[1] <= 0:
            raise ValueError("Projector batch and query dimensions must be non-empty.")
        if inputs.shape[-1] != self.qformer_dim:
            raise ValueError(
                f"Projector expected qformer_dim={self.qformer_dim}, got "
                f"{inputs.shape[-1]}."
            )
        if not inputs.is_floating_point():
            raise TypeError(f"Projector inputs must be floating point, got {inputs.dtype}.")
        if self.check_finite and not bool(torch.isfinite(inputs).all().item()):
            raise ValueError("Projector inputs contain NaN or infinite values.")

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_input(inputs)

        # Model.encode_visual normally performs this conversion.  Repeating it
        # here makes the component safe to use independently and keeps linear,
        # normalization, and gate parameters on one device/dtype.
        values = inputs.to(device=self.device, dtype=self.dtype, non_blocking=True)
        normalized = self.norm(values)

        residual = self.residual_proj(values)
        activation, linear = self.fc1(normalized).chunk(2, dim=-1)
        hidden = F.silu(activation) * linear
        branch = self.fc2(self.dropout(hidden))

        # Sigmoid is evaluated in FP32 for stable gates under mixed precision,
        # then restored to the branch dtype before fusion.
        gate_logits = self.gate(normalized)
        gate_values = torch.sigmoid(gate_logits.float()).to(branch.dtype)
        projected = residual + gate_values * branch

        expected_shape = (*inputs.shape[:-1], self.llm_dim)
        if tuple(projected.shape) != expected_shape or tuple(gate_values.shape) != expected_shape:
            raise RuntimeError(
                "Internal projector shape invariant failed: expected "
                f"{expected_shape}, got output={tuple(projected.shape)}, "
                f"gate={tuple(gate_values.shape)}."
            )
        if self.check_finite and not bool(torch.isfinite(projected).all().item()):
            raise FloatingPointError("Projector produced NaN or infinite values.")
        return projected, gate_values

    def extra_repr(self) -> str:
        return (
            f"qformer_dim={self.qformer_dim}, hidden_dim={self.hidden_dim}, "
            f"llm_dim={self.llm_dim}, zero_init_last_layer={self.zero_init_last_layer}"
        )
