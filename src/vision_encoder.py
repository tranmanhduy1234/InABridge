"""Hugging Face vision-tower adapter used by InA-Bridge.

The wrapper deliberately exposes *spatial patch tokens*, not the pooled CLS
representation returned by many ViT checkpoints.  It supports DINOv3/DINOv2
style outputs while keeping the foundation tower frozen by default.
"""

from __future__ import annotations

from contextlib import nullcontext
from enum import Enum
import math
from numbers import Integral, Real
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn
from transformers import AutoModel


class TrainMode(str, Enum):
    """Fine-tuning policy for the vision foundation model."""

    FROZEN = "frozen"
    PARTIAL = "partial"
    FULL = "full"


class ImageEncoder(nn.Module):
    """Return dense patch features from a Hugging Face vision transformer.

    Parameters
    ----------
    model_id:
        Hugging Face model identifier.  It is retained even when ``model`` is
        injected so checkpoints can record which foundation model is expected.
    return_layer:
        Index into ``hidden_states``.  Negative indices follow normal Python
        semantics; ``-1`` is the final transformer block and ``-2`` is the
        penultimate block.
    model:
        Optional already-created model.  This is useful for tests and for
        callers that need full control over checkpoint loading.
    apply_output_norm:
        Apply the model's final LayerNorm to the selected hidden state.  When
        omitted, DINO's ``config.apply_layernorm`` is respected and models
        exposing a post encoder norm default to normalized features.
    check_finite:
        Expensive diagnostic validation for NaN/Inf inputs and outputs.  Keep
        disabled for regular large-scale training to avoid device syncs.
    pretrained_kwargs:
        Keyword arguments forwarded to ``AutoModel.from_pretrained``.  No
        remote-code or dtype policy is silently enabled here.
    """

    _LAYER_PATHS: tuple[tuple[str, ...], ...] = (
        # DINOv3 ViT in current Transformers releases.
        ("model", "layer"),
        ("model", "layers"),
        # DINOv2 and common Hugging Face ViTs.
        ("encoder", "layer"),
        ("encoder", "layers"),
        ("encoder", "block"),
        ("encoder", "blocks"),
        ("vision_model", "encoder", "layer"),
        ("vision_model", "encoder", "layers"),
        # timm/OpenCLIP-compatible injected towers.
        ("transformer", "resblocks"),
        ("transformer", "layers"),
        ("transformer", "blocks"),
        ("blocks",),
        ("layers",),
        ("layer",),
    )
    _NORM_PATHS: tuple[tuple[str, ...], ...] = (
        ("norm",),
        ("layernorm",),
        ("layer_norm",),
        ("ln_f",),
        ("vision_model", "post_layernorm"),
        ("vision_model", "layernorm"),
        ("encoder", "layernorm"),
        ("encoder", "layer_norm"),
        ("encoder", "norm"),
    )
    _PATCH_EMBED_PATHS: tuple[tuple[str, ...], ...] = (
        ("embeddings", "patch_embeddings", "projection"),
        ("embeddings", "patch_embeddings"),
        ("embeddings", "patch_embed"),
        ("patch_embed", "proj"),
        ("patch_embed",),
        ("patch_embeddings",),
        ("patch_embedding",),
        ("conv_proj",),
    )

    def __init__(
        self,
        model_id: str,
        return_layer: int = -2,
        model: Optional[nn.Module] = None,
        *,
        apply_output_norm: Optional[bool] = None,
        check_finite: bool = False,
        pretrained_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string.")
        if not isinstance(return_layer, Integral) or isinstance(return_layer, bool):
            raise TypeError("return_layer must be an integer.")
        if not isinstance(check_finite, bool):
            raise TypeError("check_finite must be a boolean.")
        if pretrained_kwargs is not None and not isinstance(pretrained_kwargs, Mapping):
            raise TypeError("pretrained_kwargs must be a mapping or None.")
        if model is not None and pretrained_kwargs:
            raise ValueError("pretrained_kwargs cannot be used when model is injected.")

        self.model_id = model_id
        self.model = (
            model
            if model is not None
            else AutoModel.from_pretrained(model_id, **dict(pretrained_kwargs or {}))
        )
        if not isinstance(self.model, nn.Module):
            raise TypeError("model must be an instance of torch.nn.Module.")
        if not hasattr(self.model, "config"):
            raise ValueError("The vision model must expose a Hugging Face-style config.")

        self.check_finite = check_finite
        self.train_mode = TrainMode.FROZEN
        self.unfreeze_ratio = 0.0
        self.unfrozen_layer_count = 0
        self.unfrozen_layer_range: tuple[int, int] = (0, 0)
        self._partial_unfreeze_embeddings = False

        num_hidden_states = self._get_num_hidden_states()
        self.return_layer = self._normalize_return_layer(int(return_layer), num_hidden_states)

        register_tokens = self._config_value("num_register_tokens", default=0)
        if not isinstance(register_tokens, Integral) or isinstance(register_tokens, bool):
            raise TypeError("config.num_register_tokens must be an integer.")
        if register_tokens < 0:
            raise ValueError("config.num_register_tokens cannot be negative.")
        self.num_register_tokens = int(register_tokens)

        raw_patch_size = self._config_value("patch_size")
        self._patch_size_hw = self._normalize_pair(raw_patch_size, "patch_size")
        # Preserve the convenient scalar public attribute for square ViT
        # patches, while still supporting rectangular injected encoders.
        self.patch_size: int | tuple[int, int]
        if self._patch_size_hw[0] == self._patch_size_hw[1]:
            self.patch_size = self._patch_size_hw[0]
        else:
            self.patch_size = self._patch_size_hw

        channels = self._config_value("num_channels", "in_chans", default=3)
        if not isinstance(channels, Integral) or isinstance(channels, bool) or channels <= 0:
            raise ValueError("The configured number of input channels must be positive.")
        self.num_channels = int(channels)

        norm_available = self._resolve_output_norm() is not None
        if apply_output_norm is None:
            configured = self._config_value("apply_layernorm", default=None)
            self.apply_output_norm = norm_available and (
                True if configured is None else bool(configured)
            )
        elif not isinstance(apply_output_norm, bool):
            raise TypeError("apply_output_norm must be bool or None.")
        else:
            if apply_output_norm and not norm_available:
                raise ValueError(
                    "apply_output_norm=True but no output normalization module was found."
                )
            self.apply_output_norm = apply_output_norm

        # The foundation tower is frozen unless the caller opts in explicitly.
        self.set_train_mode(TrainMode.FROZEN)

    # ------------------------------------------------------------------
    # Properties and config introspection
    # ------------------------------------------------------------------

    def _reference_tensor(self) -> Optional[torch.Tensor]:
        for parameter in self.model.parameters():
            return parameter
        for buffer in self.model.buffers():
            return buffer
        return None

    @property
    def device(self) -> torch.device:
        reference = self._reference_tensor()
        return reference.device if reference is not None else torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        for parameter in self.model.parameters():
            if parameter.is_floating_point() or parameter.is_complex():
                return parameter.dtype
        for buffer in self.model.buffers():
            if buffer.is_floating_point() or buffer.is_complex():
                return buffer.dtype
        return torch.float32

    @property
    def embed_dim(self) -> int:
        value = self._config_value("hidden_size", "embed_dim", "d_model", default=None)
        if value is None:
            hidden_sizes = self._config_value("hidden_sizes", default=None)
            if isinstance(hidden_sizes, Sequence) and hidden_sizes:
                value = hidden_sizes[-1]
        if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
            raise AttributeError(
                f"Cannot determine a positive embed_dim from "
                f"{self.model.__class__.__name__} config."
            )
        return int(value)

    @property
    def layers(self) -> nn.ModuleList:
        """Transformer block collection used by partial fine-tuning."""

        return self._resolve_encoder_layers()

    @property
    def trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )

    @property
    def total_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    @property
    def trainable_ratio(self) -> float:
        total = self.total_parameters
        return self.trainable_parameters / total if total else 0.0

    def _config_sources(self) -> tuple[Any, ...]:
        config = self.model.config
        nested = getattr(config, "vision_config", None)
        return (config,) if nested is None else (config, nested)

    def _config_value(self, *names: str, default: Any = ...) -> Any:
        for config in self._config_sources():
            for name in names:
                if hasattr(config, name):
                    value = getattr(config, name)
                    if value is not None:
                        return value
        if default is not ...:
            return default
        joined = ", ".join(repr(name) for name in names)
        raise RuntimeError(
            f"None of {joined} was found in {self.model.__class__.__name__} config."
        )

    def _get_num_hidden_states(self) -> int:
        depth = self._config_value("num_hidden_layers", "depth", "n_layers", default=None)
        if not isinstance(depth, Integral) or isinstance(depth, bool) or depth <= 0:
            raise ValueError("Vision config must provide a positive num_hidden_layers/depth.")
        # Hugging Face hidden_states contains the embedding output followed by
        # one tensor for every transformer block.
        return int(depth) + 1

    @staticmethod
    def _normalize_pair(value: Any, name: str) -> tuple[int, int]:
        if isinstance(value, Integral) and not isinstance(value, bool):
            pair = (int(value), int(value))
        elif (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and len(value) == 2
            and all(
                isinstance(item, Integral) and not isinstance(item, bool)
                for item in value
            )
        ):
            pair = (int(value[0]), int(value[1]))
        else:
            raise TypeError(f"config.{name} must be a positive int or a pair of ints.")
        if any(item <= 0 for item in pair):
            raise ValueError(f"config.{name} entries must be positive, got {pair}.")
        return pair

    @staticmethod
    def _normalize_return_layer(return_layer: int, num_hidden: int) -> int:
        if return_layer < -num_hidden or return_layer >= num_hidden:
            raise ValueError(
                f"return_layer={return_layer} is out of range; expected an index in "
                f"[-{num_hidden}, {num_hidden - 1}] for {num_hidden} hidden states."
            )
        return num_hidden + return_layer if return_layer < 0 else return_layer

    @staticmethod
    def _module_at_path(root: nn.Module, path: Sequence[str]) -> Optional[nn.Module]:
        target: Any = root
        for name in path:
            target = getattr(target, name, None)
            if target is None:
                return None
        return target if isinstance(target, nn.Module) else None

    def _resolve_encoder_layers(self) -> nn.ModuleList:
        for path in self._LAYER_PATHS:
            candidate = self._module_at_path(self.model, path)
            if isinstance(candidate, nn.ModuleList) and len(candidate):
                return candidate

        # A conservative fallback for injected/remote architectures.  Prefer a
        # collection whose length equals config.num_hidden_layers so an inner
        # MLP ModuleList cannot accidentally be selected.
        expected = self._get_num_hidden_states() - 1
        matches: list[tuple[int, int, nn.ModuleList]] = []
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.ModuleList) or not len(module):
                continue
            lowered = name.lower()
            if not any(word in lowered for word in ("layer", "block", "resblock")):
                continue
            matches.append((int(len(module) == expected), len(module), module))
        if matches:
            return max(matches, key=lambda item: (item[0], item[1]))[2]
        raise RuntimeError(
            f"Cannot locate transformer layers on {self.model.__class__.__name__}; "
            "partial fine-tuning is unavailable for this architecture."
        )

    def _resolve_output_norm(self) -> Optional[nn.Module]:
        for path in self._NORM_PATHS:
            module = self._module_at_path(self.model, path)
            if module is not None:
                return module
        return None

    def _resolve_patch_embedding(self) -> Optional[nn.Module]:
        for path in self._PATCH_EMBED_PATHS:
            module = self._module_at_path(self.model, path)
            if module is not None:
                return module
        return None

    # ------------------------------------------------------------------
    # Token geometry
    # ------------------------------------------------------------------

    def compute_grid_shape(self, height: int, width: Optional[int] = None) -> tuple[int, int]:
        """Return the patch grid for a possibly rectangular image."""

        width = height if width is None else width
        if any(
            not isinstance(value, Integral) or isinstance(value, bool) or value <= 0
            for value in (height, width)
        ):
            raise ValueError("height and width must be positive integers.")
        patch_height, patch_width = self._patch_size_hw
        if height % patch_height or width % patch_width:
            raise ValueError(
                f"Image size {height}x{width} is not divisible by patch size "
                f"{patch_height}x{patch_width}."
            )
        return height // patch_height, width // patch_width

    def compute_grid_size(self, image_size: int) -> int:
        """Backward-compatible square-image grid size helper."""

        grid_height, grid_width = self.compute_grid_shape(image_size, image_size)
        if grid_height != grid_width:
            raise ValueError(
                "compute_grid_size requires a square patch grid; use compute_grid_shape instead."
            )
        return grid_height

    def compute_num_patches(self, image_size: int, width: Optional[int] = None) -> int:
        grid_height, grid_width = self.compute_grid_shape(image_size, width)
        return grid_height * grid_width

    def compute_total_tokens(
        self,
        image_size: int,
        include_cls: bool = True,
        width: Optional[int] = None,
        include_registers: bool = True,
    ) -> int:
        patches = self.compute_num_patches(image_size, width)
        return patches + int(include_cls) + (
            self.num_register_tokens if include_registers else 0
        )

    def strip_special_tokens(
        self,
        features: torch.Tensor,
        drop_cls: bool = True,
        *,
        expected_num_patches: Optional[int] = None,
    ) -> torch.Tensor:
        """Remove DINO register tokens and, normally, the CLS token.

        DINOv2/DINOv3 lay tokens out as ``[CLS, registers..., patches...]``.
        ``drop_cls=False`` retains CLS but still removes register tokens.
        """

        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a torch.Tensor.")
        if features.ndim != 3:
            raise ValueError(
                f"Expected features with shape (B, L, D), got {tuple(features.shape)}."
            )
        if features.shape[0] <= 0 or features.shape[2] != self.embed_dim:
            raise ValueError(
                f"Invalid feature shape {tuple(features.shape)}; final dimension must "
                f"equal embed_dim={self.embed_dim}."
            )

        special_count = 1 + self.num_register_tokens
        if features.shape[1] < special_count:
            raise ValueError(
                f"Feature sequence has {features.shape[1]} tokens but at least "
                f"{special_count} CLS/register tokens were expected."
            )
        patch_count = features.shape[1] - special_count
        if expected_num_patches is not None and patch_count != expected_num_patches:
            raise ValueError(
                "Vision token geometry mismatch: received "
                f"{features.shape[1]} total tokens ({patch_count} patches after "
                f"1 CLS + {self.num_register_tokens} registers), expected "
                f"{expected_num_patches} patches."
            )

        patches = features[:, special_count:, :]
        if drop_cls:
            return patches
        return torch.cat((features[:, :1, :], patches), dim=1)

    # ------------------------------------------------------------------
    # Train mode control
    # ------------------------------------------------------------------

    def set_train_mode(
        self,
        mode: TrainMode | str,
        ratio: float = 0.1,
        unfreeze_embeddings: bool = False,
    ) -> None:
        """Apply a frozen, last-block, or full fine-tuning policy atomically."""

        try:
            resolved_mode = TrainMode(mode)
        except (TypeError, ValueError) as error:
            valid = ", ".join(item.value for item in TrainMode)
            raise ValueError(f"Unknown train mode {mode!r}; expected one of: {valid}.") from error
        if not isinstance(unfreeze_embeddings, bool):
            raise TypeError("unfreeze_embeddings must be a boolean.")

        encoder_layers: Optional[nn.ModuleList] = None
        patch_embedding: Optional[nn.Module] = None
        num_unfreeze = 0
        active_layer_count = 0
        if resolved_mode == TrainMode.PARTIAL:
            if (
                not isinstance(ratio, Real)
                or isinstance(ratio, bool)
                or not math.isfinite(float(ratio))
                or not 0.0 < float(ratio) < 1.0
            ):
                raise ValueError(f"ratio must be finite and in (0, 1), got {ratio!r}.")
            # Resolve everything before mutating requires_grad so an unsupported
            # architecture cannot leave the module half reconfigured.
            encoder_layers = self._resolve_encoder_layers()
            # hidden_states[0] is the embedding output and hidden_states[k]
            # is the output after block k-1.  A penultimate return layer does
            # not depend on the model's final block, so unfreezing the literal
            # last blocks would silently add parameters that receive no
            # gradient.  Fine-tune the suffix of the *active* block prefix.
            active_layer_count = min(self.return_layer, len(encoder_layers))
            if active_layer_count <= 0:
                raise ValueError(
                    "Partial block fine-tuning requires return_layer to select "
                    "an output after at least one transformer block."
                )
            num_unfreeze = min(
                active_layer_count,
                max(1, math.ceil(active_layer_count * float(ratio))),
            )
            if unfreeze_embeddings:
                patch_embedding = self._resolve_patch_embedding()
                if patch_embedding is None:
                    raise RuntimeError(
                        "unfreeze_embeddings=True but the patch embedding module "
                        "could not be located."
                    )

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        if resolved_mode == TrainMode.FULL:
            for parameter in self.model.parameters():
                parameter.requires_grad_(True)
        elif resolved_mode == TrainMode.PARTIAL:
            assert encoder_layers is not None
            first_unfrozen = active_layer_count - num_unfreeze
            for layer in encoder_layers[first_unfrozen:active_layer_count]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)
            output_norm = self._resolve_output_norm()
            if output_norm is not None:
                for parameter in output_norm.parameters():
                    parameter.requires_grad_(True)
            if patch_embedding is not None:
                for parameter in patch_embedding.parameters():
                    parameter.requires_grad_(True)

        self.train_mode = resolved_mode
        self.unfreeze_ratio = (
            float(ratio)
            if resolved_mode == TrainMode.PARTIAL
            else (1.0 if resolved_mode == TrainMode.FULL else 0.0)
        )
        self.unfrozen_layer_count = num_unfreeze if resolved_mode == TrainMode.PARTIAL else (
            self._get_num_hidden_states() - 1 if resolved_mode == TrainMode.FULL else 0
        )
        self.unfrozen_layer_range = (
            (active_layer_count - num_unfreeze, active_layer_count)
            if resolved_mode == TrainMode.PARTIAL
            else (
                (0, self._get_num_hidden_states() - 1)
                if resolved_mode == TrainMode.FULL
                else (0, 0)
            )
        )
        self._partial_unfreeze_embeddings = (
            resolved_mode == TrainMode.PARTIAL and unfreeze_embeddings
        )

        if resolved_mode == TrainMode.FROZEN:
            self.model.eval()
        else:
            self.model.train(self.training)
        self.assert_train_mode()

    def assert_train_mode(self) -> None:
        """Fail fast if external code violated the selected freeze invariant."""

        parameters = tuple(self.model.parameters())
        trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in parameters)
        if self.train_mode == TrainMode.FROZEN and trainable:
            raise RuntimeError("Frozen vision encoder contains trainable parameters.")
        if self.train_mode == TrainMode.FULL and trainable != total:
            raise RuntimeError("Full vision fine-tuning requires every parameter to be trainable.")
        if self.train_mode == TrainMode.PARTIAL and total and not 0 < trainable < total:
            raise RuntimeError("Partial vision fine-tuning must train a strict parameter subset.")

    def train(self, mode: bool = True) -> "ImageEncoder":
        """Keep a frozen tower deterministic when its parent enters train mode."""

        super().train(mode)
        if self.train_mode == TrainMode.FROZEN:
            self.model.eval()
        return self

    def enable_gradient_checkpointing(self, **kwargs: Any) -> None:
        method = getattr(self.model, "gradient_checkpointing_enable", None)
        if not callable(method):
            raise AttributeError("Model does not support gradient_checkpointing_enable().")
        method(**kwargs)

    def disable_gradient_checkpointing(self) -> None:
        method = getattr(self.model, "gradient_checkpointing_disable", None)
        if not callable(method):
            raise AttributeError("Model does not support gradient_checkpointing_disable().")
        method()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _validate_input_tensor(self, pixel_values: torch.Tensor) -> tuple[int, int]:
        if not isinstance(pixel_values, torch.Tensor):
            raise TypeError("pixel_values must be a torch.Tensor.")
        if pixel_values.ndim != 4:
            raise ValueError(
                "pixel_values must have shape (B, C, H, W), got "
                f"{tuple(pixel_values.shape)}."
            )
        batch, channels, height, width = pixel_values.shape
        if batch <= 0:
            raise ValueError("pixel_values cannot contain an empty batch.")
        if channels != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} image channels, received {channels}."
            )
        if not pixel_values.is_floating_point():
            raise TypeError(
                f"pixel_values must use a floating dtype, got {pixel_values.dtype}."
            )
        self.compute_grid_shape(height, width)
        if self.check_finite and not bool(torch.isfinite(pixel_values).all().item()):
            raise ValueError("pixel_values contains NaN or infinite values.")
        return height, width

    @staticmethod
    def _output_field(outputs: Any, name: str) -> Any:
        if hasattr(outputs, name):
            return getattr(outputs, name)
        if isinstance(outputs, Mapping):
            return outputs.get(name)
        return None

    def _extract_output_tensors(
        self, outputs: Any
    ) -> tuple[Optional[Sequence[torch.Tensor]], Optional[torch.Tensor]]:
        hidden_states = self._output_field(outputs, "hidden_states")
        last_hidden_state = self._output_field(outputs, "last_hidden_state")

        if isinstance(outputs, torch.Tensor):
            last_hidden_state = outputs
        elif isinstance(outputs, (tuple, list)):
            if outputs and isinstance(outputs[0], torch.Tensor):
                last_hidden_state = outputs[0]
            if hidden_states is None:
                for item in outputs:
                    if (
                        isinstance(item, (tuple, list))
                        and item
                        and all(isinstance(tensor, torch.Tensor) for tensor in item)
                    ):
                        hidden_states = item
                        break

        if hidden_states is not None and not isinstance(hidden_states, (tuple, list)):
            raise TypeError("Vision model hidden_states must be a tuple/list of tensors.")
        if last_hidden_state is not None and not isinstance(last_hidden_state, torch.Tensor):
            raise TypeError("Vision model last_hidden_state must be a tensor.")
        return hidden_states, last_hidden_state

    def _select_hidden_state(self, outputs: Any) -> torch.Tensor:
        hidden_states, last_hidden_state = self._extract_output_tensors(outputs)
        if hidden_states is None:
            expected_final = self._get_num_hidden_states() - 1
            if self.return_layer != expected_final or last_hidden_state is None:
                raise RuntimeError(
                    "Vision model did not return hidden_states; an intermediate "
                    "return_layer therefore cannot be selected."
                )
            return last_hidden_state
        if not hidden_states:
            raise RuntimeError("Vision model returned an empty hidden_states collection.")
        if self.return_layer >= len(hidden_states):
            raise IndexError(
                f"return_layer={self.return_layer} exceeds the {len(hidden_states)} "
                "hidden states returned at runtime."
            )
        if not all(isinstance(state, torch.Tensor) for state in hidden_states):
            raise TypeError("Every item in hidden_states must be a tensor.")

        selected = hidden_states[self.return_layer]
        # HF DINO models apply their final normalization only to
        # last_hidden_state.  Reuse that exact result for the final block.
        used_normalized_last = (
            self.apply_output_norm
            and self.return_layer == len(hidden_states) - 1
            and last_hidden_state is not None
            and tuple(last_hidden_state.shape) == tuple(selected.shape)
        )
        if used_normalized_last:
            return last_hidden_state
        if self.apply_output_norm:
            output_norm = self._resolve_output_norm()
            if output_norm is None:  # guarded at construction, defensive here
                raise RuntimeError("Configured output normalization is unavailable.")
            selected = output_norm(selected)
        return selected

    def _to_patch_sequence(
        self,
        features: torch.Tensor,
        height: int,
        width: int,
        drop_special_tokens: bool,
    ) -> torch.Tensor:
        expected_patches = self.compute_num_patches(height, width)
        if features.ndim == 4:
            # Backbone-style output: (B, C, grid_h, grid_w), already stripped.
            if not drop_special_tokens:
                raise ValueError(
                    "The vision model returned an already-stripped 4D feature map; "
                    "CLS/register tokens cannot be reconstructed."
                )
            grid_height, grid_width = self.compute_grid_shape(height, width)
            if features.shape[1] != self.embed_dim:
                raise ValueError(
                    f"4D feature map channel dimension {features.shape[1]} does not "
                    f"match embed_dim={self.embed_dim}."
                )
            if tuple(features.shape[-2:]) != (grid_height, grid_width):
                raise ValueError(
                    f"Feature-map grid {tuple(features.shape[-2:])} does not match "
                    f"expected {(grid_height, grid_width)}."
                )
            return features.flatten(2).transpose(1, 2).contiguous()
        if features.ndim != 3:
            raise ValueError(
                "Selected vision features must be (B, L, D) or (B, D, Gh, Gw), "
                f"got {tuple(features.shape)}."
            )
        if features.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Vision feature width {features.shape[-1]} does not match "
                f"embed_dim={self.embed_dim}."
            )
        if not drop_special_tokens:
            expected_total = expected_patches + 1 + self.num_register_tokens
            if features.shape[1] != expected_total:
                raise ValueError(
                    f"Vision token geometry mismatch: got {features.shape[1]} tokens, "
                    f"expected {expected_total}."
                )
            return features
        return self.strip_special_tokens(
            features, drop_cls=True, expected_num_patches=expected_patches
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        drop_special_tokens: bool = True,
    ) -> torch.Tensor:
        """Encode images as ``(batch, spatial_patches, embed_dim)`` features."""

        if not isinstance(drop_special_tokens, bool):
            raise TypeError("drop_special_tokens must be a boolean.")
        height, width = self._validate_input_tensor(pixel_values)

        # HF DINO casts to its patch-embedding dtype internally, while many
        # compatible injected towers do not.  Doing it here makes both paths
        # deterministic and avoids accidental cross-device calls.
        model_inputs = pixel_values.to(device=self.device, dtype=self.dtype, non_blocking=True)
        grad_context = torch.no_grad() if self.train_mode == TrainMode.FROZEN else nullcontext()
        with grad_context:
            outputs = self.model(
                pixel_values=model_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            features = self._select_hidden_state(outputs)
            features = self._to_patch_sequence(
                features, height, width, drop_special_tokens=drop_special_tokens
            )

        if features.shape[0] != pixel_values.shape[0]:
            raise ValueError(
                f"Vision model changed batch size from {pixel_values.shape[0]} "
                f"to {features.shape[0]}."
            )
        if self.check_finite and not bool(torch.isfinite(features).all().item()):
            raise FloatingPointError("Vision encoder produced NaN or infinite features.")
        return features
