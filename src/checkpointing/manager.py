"""Portable bridge checkpoint loading, metadata writes, and retention."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn


def build_portable_state_dict(
    model: nn.Module,
    state: Optional[Mapping[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Keep bridge weights and trainable adapters, excluding immutable towers."""
    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    source = model.state_dict() if state is None else state
    return {
        name: tensor.detach().cpu()
        for name, tensor in source.items()
        if name.startswith(("qformer.", "projector.")) or name in trainable
    }


def load_bridge_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    allow_llm_adapter_keys: bool = True,
    required_prefixes: tuple[str, ...] = ("qformer.",),
) -> tuple[list[str], list[str]]:
    checkpoint_path = Path(path).expanduser().resolve()
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "bridge_model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint is not a state dictionary: {checkpoint_path}")
    absent = [prefix for prefix in required_prefixes if not any(k.startswith(prefix) for k in state)]
    if absent:
        raise ValueError(f"Checkpoint is missing required components: {absent}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if allow_llm_adapter_keys:
        unexpected = [key for key in unexpected if not key.startswith("llm.")]
    return list(missing), list(unexpected)


def write_json_atomic(path: str | Path, payload: Dict[str, Any]) -> None:
    """Write metadata atomically so interruption cannot leave partial JSON."""
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, destination)


def rotate_checkpoints(
    output_dir: str | Path, limit: int, protected: Optional[Path] = None
) -> None:
    if limit <= 0:
        raise ValueError("Checkpoint retention limit must be positive")
    checkpoints = sorted(Path(output_dir).glob("checkpoint-*"))
    protected_path = None
    if protected is not None and any(path.resolve() == protected.resolve() for path in checkpoints):
        protected_path = protected
    removable = [
        path
        for path in checkpoints
        if protected_path is None or path.resolve() != protected_path.resolve()
    ]
    keep_unprotected = max(0, limit - (1 if protected_path is not None else 0))
    for stale in removable[:-keep_unprotected] if keep_unprotected else removable:
        if stale.is_dir():
            shutil.rmtree(stale)
