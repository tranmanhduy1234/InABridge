from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel

class TrainMode(str, Enum):
    FROZEN = "frozen"
    PARTIAL = "partial"
    FULL = "full"

class ImageEncoder(nn.Module):
    def __init__(
        self,
        model_id: str,
        return_layer: int = -2,
    ):
        super().__init__()
        self.model_id: str = model_id
        self.model = AutoModel.from_pretrained(model_id)
        self.train_mode: TrainMode = TrainMode.FROZEN
        self.unfreeze_ratio: float = 0.0

        num_hidden = self._get_num_hidden_states()
        self.return_layer: int = self._normalize_return_layer(return_layer, num_hidden)
        self.num_register_tokens: int = getattr(self.model.config, "num_register_tokens", 0)
        self.patch_size: int = self._get_config_attr("patch_size")

        self.set_train_mode(TrainMode.FROZEN)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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

    @property
    def embed_dim(self) -> int:
        for attr in ("hidden_size", "embed_dim", "d_model"):
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        raise AttributeError(
            f"Cannot determine embed_dim from {self.model.__class__.__name__} config."
        )

    @property
    def layers(self) -> nn.ModuleList:
        return self._resolve_encoder_layers()

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

    # ------------------------------------------------------------------
    # Config Introspection
    # ------------------------------------------------------------------

    def _get_config_attr(self, name: str) -> int:
        if hasattr(self.model.config, name):
            return getattr(self.model.config, name)
        raise RuntimeError(
            f"'{name}' not found in {self.model.__class__.__name__} config."
        )

    def _get_num_hidden_states(self) -> int:
        return self._get_config_attr("num_hidden_layers") + 1

    @staticmethod
    def _normalize_return_layer(return_layer: int, num_hidden: int) -> int:
        if return_layer < -num_hidden or return_layer >= num_hidden:
            raise ValueError(
                f"return_layer={return_layer} out of range. "
                f"Valid: [{-num_hidden}, {num_hidden - 1}] "
                f"for {num_hidden} hidden states."
            )
        return num_hidden + return_layer if return_layer < 0 else return_layer

    def _resolve_encoder_layers(self) -> nn.ModuleList:
            candidates = [
                (self.model, "encoder", "layer"),
                (self.model, "encoder", "layers"),
                (self.model, "encoder", "block"),
                (self.model, "encoder", "blocks"),
                (self.model, "transformer", "resblocks"),
                (self.model, "transformer", "layers"),
                (self.model, "transformer", "blocks"),
                (self.model, "layernorm", None),
                (self.model, None, "layers"),
                (self.model, None, "layer"),
                (self.model, None, "blocks"),
                (self.model, None, "block"),
            ]
            for root, sub_attr, layer_attr in candidates:
                target = getattr(root, sub_attr, root) if sub_attr else root
                if target is not None and layer_attr and hasattr(target, layer_attr):
                    val = getattr(target, layer_attr)
                    if isinstance(val, nn.ModuleList):
                        return val

            # Fallback: Tự động quét tìm ModuleList đại diện cho các lớp Transformer
            for name, module in self.model.named_modules():
                if isinstance(module, nn.ModuleList) and len(module) > 0:
                    if any(kw in name.lower() for kw in ["layer", "block", "resblock"]):
                        return module

            raise RuntimeError(
                f"Cannot locate encoder layers on {self.model.__class__.__name__}."
            )

    # ------------------------------------------------------------------
    # Token Geometry
    # ------------------------------------------------------------------

    def compute_grid_size(self, image_size: int) -> int:
        if image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size={image_size} is not divisible by patch_size={self.patch_size}."
            )
        return image_size // self.patch_size

    def compute_num_patches(self, image_size: int) -> int:
        grid = self.compute_grid_size(image_size)
        return grid * grid

    def compute_total_tokens(self, image_size: int, include_cls: bool = True) -> int:
        num_patches = self.compute_num_patches(image_size)
        cls_count = 1 if include_cls else 0
        return cls_count + self.num_register_tokens + num_patches

    def strip_special_tokens(self, feats: torch.Tensor, drop_cls: bool = True) -> torch.Tensor:
        if feats.dim() != 3:
            raise ValueError(f"Expected 3D tensor (B, L, D), got shape {list(feats.shape)}.")

        special_offset = 1 + self.num_register_tokens
        if drop_cls:
            return feats[:, special_offset:, :]

        cls_tok = feats[:, :1, :]
        patch_tok = feats[:, special_offset:, :]
        return torch.cat([cls_tok, patch_tok], dim=1)

    # ------------------------------------------------------------------
    # Train Mode Control
    # ------------------------------------------------------------------

    def set_train_mode(
        self,
        mode: TrainMode,
        ratio: float = 0.1,
        unfreeze_embeddings: bool = False,
    ) -> None:
        self.train_mode = mode
        self.unfreeze_ratio = ratio if mode == TrainMode.PARTIAL else (1.0 if mode == TrainMode.FULL else 0.0)

        for p in self.model.parameters():
            p.requires_grad_(False)

        if mode == TrainMode.FROZEN:
            return

        if mode == TrainMode.FULL:
            for p in self.model.parameters():
                p.requires_grad_(True)
            return

        if mode == TrainMode.PARTIAL:
            if not (0.0 < ratio < 1.0):
                raise ValueError(f"ratio must be in (0.0, 1.0), got {ratio}.")

            enc_layers = self._resolve_encoder_layers()
            num_unfreeze = max(1, round(len(enc_layers) * ratio))

            for layer in enc_layers[-num_unfreeze:]:
                for p in layer.parameters():
                    p.requires_grad_(True)

            self._unfreeze_final_norm()

            if unfreeze_embeddings:
                self._unfreeze_patch_embeddings()
            return

        raise ValueError(f"Unsupported train mode: {mode}")

    def _unfreeze_final_norm(self) -> None:
        for target in (self.model, getattr(self.model, "encoder", None)):
            if target is None:
                continue
            for attr in ("layernorm", "layer_norm", "norm", "ln_f"):
                module = getattr(target, attr, None)
                if isinstance(module, nn.Module):
                    for p in module.parameters():
                        p.requires_grad_(True)
                    return

    def _unfreeze_patch_embeddings(self) -> None:
        for target in (self.model, getattr(self.model, "embeddings", None)):
            if target is None:
                continue
            for attr in ("embeddings", "patch_embed", "patch_embeddings", "patch_embedding"):
                module = getattr(target, attr, None)
                if isinstance(module, nn.Module):
                    for p in module.parameters():
                        p.requires_grad_(True)
                    return

    def enable_gradient_checkpointing(self) -> None:
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        else:
            raise AttributeError("Model does not support gradient_checkpointing_enable().")

    def disable_gradient_checkpointing(self) -> None:
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        else:
            raise AttributeError("Model does not support gradient_checkpointing_disable().")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def print_full_layers(self) -> None:
        print("\n[MODEL STRUCTURE]")
        print(f"{'Name':<60} {'Type'}")
        print("-" * 80)
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:
                print(f"{name:<60} {module.__class__.__name__}")

    def print_trainable_params(self) -> None:
        print("\n[TRAINABLE PARAMETERS]")
        print(f"{'Name':<60} {'Shape'}")
        print("-" * 80)
        found = False
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                print(f"{name:<60} {list(p.shape)}")
                found = True
        if not found:
            print("  (No trainable parameters — FROZEN mode)")

    def summary(self) -> None:
        num_hidden = self._get_num_hidden_states()
        total = self.total_parameters
        trainable = self.trainable_parameters

        print("\n" + "=" * 60)
        print("[SUMMARY]")
        print("=" * 60)
        print(f"  Model ID        : {self.model_id}")
        print(f"  Model class     : {self.model.__class__.__name__}")
        print(f"  Embed dim       : {self.embed_dim}")
        print(f"  Patch size      : {self.patch_size}")
        print(f"  Register tokens : {self.num_register_tokens}")
        print(f"  Train mode      : {self.train_mode.value}")
        print(f"  Unfreeze ratio  : {self.unfreeze_ratio:.2f}")
        print(f"  Return layer    : {self.return_layer} / {num_hidden - 1}")
        print(f"  Trainable params: {trainable:,}")
        print(f"  Total params    : {total:,}")
        print(f"  Trainable %     : {trainable / total:.4%}" if total > 0 else "  Trainable %     : N/A")
        print("=" * 60)

    def check_grad_flow(self, warn_no_backward: bool = True) -> None:
        print("\n[GRADIENT FLOW CHECK]")
        print(f"{'Name':<60} {'Status'}")
        print("-" * 80)

        ok_count = 0
        no_grad_count = 0

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue

            if p.grad is not None:
                grad_norm = p.grad.norm().item()
                if grad_norm == 0.0:
                    status = "ZERO GRAD  ⚠"
                elif grad_norm > 1e3:
                    status = f"EXPLODING  ⚠  norm={grad_norm:.2e}"
                elif grad_norm < 1e-7:
                    status = f"VANISHING  ⚠  norm={grad_norm:.2e}"
                else:
                    status = f"OK         norm={grad_norm:.4f}"
                ok_count += 1
            else:
                status = "NO GRAD"
                no_grad_count += 1
            print(f"  {name:<58} {status}")

        print("-" * 80)
        print(f"  OK: {ok_count}   |   NO GRAD: {no_grad_count}")

        if warn_no_backward and ok_count == 0 and no_grad_count > 0:
            print(
                "\n  ⚠  All trainable parameters have no gradient.\n"
                "     Call loss.backward() before check_grad_flow()."
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _validate_input_tensor(self, pixel_values: torch.Tensor) -> None:
        if pixel_values.dim() != 4:
            raise ValueError(
                f"pixel_values must be 4D (B, C, H, W), got {pixel_values.dim()}D "
                f"with shape {list(pixel_values.shape)}."
            )
        _, _, h, w = pixel_values.shape
        if h != w:
            raise ValueError(f"Input must be square (H == W), got ({h}, {w}).")
        if h % self.patch_size != 0:
            raise ValueError(
                f"Image size {h}x{w} is not divisible by patch_size={self.patch_size}."
            )

    def _extract_hidden_state(
        self,
        pixel_values: torch.Tensor,
        drop_special_tokens: bool,
    ) -> torch.Tensor:
        self._validate_input_tensor(pixel_values)

        outputs = self.model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden_states.")

        if self.return_layer >= len(hidden_states):
            raise IndexError(
                f"return_layer={self.return_layer} exceeds available "
                f"hidden states ({len(hidden_states)})."
            )

        feats = hidden_states[self.return_layer]
        if drop_special_tokens:
            feats = self.strip_special_tokens(feats, drop_cls=True)
        return feats

    def forward(
        self,
        pixel_values: torch.Tensor,
        drop_special_tokens: bool = True,
    ) -> torch.Tensor:
        return self._extract_hidden_state(pixel_values, drop_special_tokens)

    def debug_forward(
        self,
        pixel_values: torch.Tensor,
        verbose: bool = True,
        drop_special_tokens: bool = True,
    ) -> torch.Tensor:
        self._validate_input_tensor(pixel_values)

        outputs = self.model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden_states.")

        image_size = pixel_values.shape[-1]
        expected_total = self.compute_total_tokens(image_size)

        if verbose:
            print(f"\n[DEBUG FORWARD] Input shape: {list(pixel_values.shape)}")
            print(f"  Total hidden states: {len(hidden_states)}")
            print(
                f"  Expected token layout @ {image_size}x{image_size}: "
                f"1 CLS + {self.num_register_tokens} register + "
                f"{self.compute_num_patches(image_size)} patch = {expected_total} total."
            )
            for i, hs in enumerate(hidden_states):
                marker = " <-- return_layer" if i == self.return_layer else ""
                mismatch = "" if hs.shape[1] == expected_total else "  ⚠ MISMATCH"
                print(f"  [{i:>3}]  shape={list(hs.shape)}{marker}{mismatch}")

        if self.return_layer >= len(hidden_states):
            raise IndexError(
                f"return_layer={self.return_layer} exceeds available "
                f"hidden states ({len(hidden_states)})."
            )

        feats = hidden_states[self.return_layer]
        if drop_special_tokens:
            feats = self.strip_special_tokens(feats, drop_cls=True)

        if verbose:
            label = "patch-only" if drop_special_tokens else "raw"
            print(f"\n  Output feature shape ({label}): {list(feats.shape)}")

        return feats