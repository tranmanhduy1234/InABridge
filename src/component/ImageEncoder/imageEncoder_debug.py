"""
Debug & test suite for ImageEncoder (DINOv3 as Image Encoder for InA-Bridge).

Usage:
    python -m src.component.ImageEncoder.imageEncoder_debug
    python src/component/ImageEncoder/imageEncoder_debug.py --model_id facebook/dinov3-vit7b16-pretrain-lvd1689m
"""

import os
import sys
import traceback
import argparse
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.component.ImageEncoder.imageEncoder import ImageEncoder, TrainMode

DEFAULT_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
IMG_SIZE = 448
LOG_FILE = os.path.join(CURRENT_DIR, "imageEncoder_debug.log")

SEP = "=" * 70


def make_input(batch: int = BATCH_SIZE, img_size: int = IMG_SIZE, device: str = DEVICE) -> torch.Tensor:
    return torch.randn(batch, 3, img_size, img_size, device=device)


def align_img_size(enc: ImageEncoder, preferred: int = IMG_SIZE) -> int:
    if preferred % enc.patch_size != 0:
        return enc.patch_size * 16
    return preferred

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

class DualLogger:
    def __init__(self, filename: str):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def section(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def ok(msg: str):
    print(f"  ✅  {msg}")

def fail(msg: str):
    print(f"  ❌  {msg}")

def warn(msg: str):
    print(f"  ⚠️   {msg}")


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_init(model_id: str) -> ImageEncoder:
    section("TEST 1 — Model Init & Properties")
    enc = ImageEncoder(model_id).to(DEVICE)

    ok(f"Model loaded: {model_id}")
    ok(f"Device      : {enc.device}")
    ok(f"Dtype       : {enc.dtype}")
    ok(f"Embed dim   : {enc.embed_dim}")
    ok(f"Patch size  : {enc.patch_size}")
    ok(f"Register tok: {enc.num_register_tokens}")
    enc.summary()
    return enc


def test_return_layer(model_id: str):
    section("TEST 2 — return_layer Validation")

    cases = [
        (-2, True, "Valid negative (second-to-last)"),
        (-1, True, "Valid negative (last)"),
        (0, True, "Index 0 (embedding layer)"),
        (3, True, "Valid positive"),
        (-999, False, "Negative out-of-range"),
        (999, False, "Positive out-of-range"),
    ]

    for idx, should_pass, desc in cases:
        try:
            enc = ImageEncoder(model_id, return_layer=idx)
            if should_pass:
                ok(f"return_layer={idx:>5}  →  normalized={enc.return_layer}  ({desc})")
            else:
                fail(f"return_layer={idx:>5}  should have raised ValueError  ({desc})")
        except (ValueError, IndexError) as e:
            if not should_pass:
                ok(f"return_layer={idx:>5}  correctly rejected: {e}")
            else:
                fail(f"return_layer={idx:>5}  unexpected error: {e}")


def test_train_modes(model_id: str):
    section("TEST 3 — TrainMode (FROZEN / PARTIAL / FULL)")

    enc = ImageEncoder(model_id).to(DEVICE)

    for mode, ratio, expect_trainable in [
        (TrainMode.FROZEN, 0.2, False),
        (TrainMode.PARTIAL, 0.25, True),
        (TrainMode.PARTIAL, 0.5, True),
        (TrainMode.FULL, 0.2, True),
    ]:
        enc.set_train_mode(mode, ratio=ratio)
        tp = enc.trainable_parameters
        tr = enc.trainable_ratio
        has_trainable = tp > 0

        status = ok if has_trainable == expect_trainable else fail
        status(
            f"mode={mode.value:<8}  ratio={ratio:.2f}  "
            f"trainable={tp:>10,} params  ({tr:.2%})"
        )

    print()
    for bad_ratio in [0.0, 1.0, -0.1, 1.5]:
        try:
            enc.set_train_mode(TrainMode.PARTIAL, ratio=bad_ratio)
            fail(f"ratio={bad_ratio} should have raised ValueError")
        except ValueError as e:
            ok(f"ratio={bad_ratio} correctly rejected: {e}")


def test_token_geometry(model_id: str):
    section("TEST 4 — Token Geometry & strip_special_tokens")

    enc = ImageEncoder(model_id).to(DEVICE)
    img_size = align_img_size(enc)

    grid = enc.compute_grid_size(img_size)
    num_patches = enc.compute_num_patches(img_size)
    total_raw = enc.compute_total_tokens(img_size)

    ok(f"image_size={img_size} → grid={grid}x{grid} | patches={num_patches} | raw_tokens={total_raw}")

    invalid_size = img_size + 1
    try:
        enc.compute_num_patches(invalid_size)
        fail(f"image_size={invalid_size} should have raised ValueError")
    except ValueError as e:
        ok(f"image_size={invalid_size} correctly rejected: {e}")

    dummy_raw = torch.randn(1, total_raw, enc.embed_dim, device=DEVICE)

    stripped = enc.strip_special_tokens(dummy_raw, drop_cls=True)
    assert stripped.shape[1] == num_patches, f"strip_special_tokens error: got {stripped.shape[1]}, expected {num_patches}"
    ok(f"strip_special_tokens(drop_cls=True)  → shape: {list(stripped.shape)}")

    stripped_cls = enc.strip_special_tokens(dummy_raw, drop_cls=False)
    assert stripped_cls.shape[1] == num_patches + 1, "strip_special_tokens(drop_cls=False) error"
    ok(f"strip_special_tokens(drop_cls=False) → shape: {list(stripped_cls.shape)}")


def test_forward(model_id: str):
    section("TEST 5 — Forward & Debug Forward")

    enc = ImageEncoder(model_id).to(DEVICE)
    img_size = align_img_size(enc)
    x = make_input(batch=BATCH_SIZE, img_size=img_size)

    feats_debug = enc.debug_forward(x, verbose=True, drop_special_tokens=True)
    ok(f"debug_forward → shape = {list(feats_debug.shape)}")

    feats = enc(x, drop_special_tokens=True)
    assert feats.shape == feats_debug.shape, "Shape mismatch between debug_forward and forward!"
    ok(f"forward → shape = {list(feats.shape)}")

    expected_patches = enc.compute_num_patches(img_size)
    assert feats.shape[1] == expected_patches, (
        f"Patch count mismatch: got {feats.shape[1]}, expected {expected_patches}"
    )
    ok(f"Patch token count matches geometry: {expected_patches}")
    ok(f"Device: {feats.device}  |  Dtype: {feats.dtype}")


def test_invalid_inputs(model_id: str):
    section("TEST 6 — Input Tensor Validation")

    enc = ImageEncoder(model_id).to(DEVICE)

    try:
        enc(torch.randn(3, 224, 224, device=DEVICE))
        fail("3D tensor should have raised ValueError")
    except ValueError as e:
        ok(f"3D tensor rejected: {e}")

    try:
        enc(torch.randn(2, 3, 224, 256, device=DEVICE))
        fail("Non-square input should have raised ValueError")
    except ValueError as e:
        ok(f"Non-square input rejected: {e}")


def test_grad_flow(model_id: str):
    section("TEST 7 — Gradient Flow Audit")

    enc = ImageEncoder(model_id, return_layer=-2).to(DEVICE)
    enc.set_train_mode(TrainMode.PARTIAL, ratio=0.25)
    enc.train()

    img_size = align_img_size(enc)
    x = make_input(img_size=img_size)
    feats = enc(x)
    loss = feats.mean()
    loss.backward()

    enc.check_grad_flow(warn_no_backward=False)

    max_active_layer_idx = enc.return_layer - 1
    num_hidden = enc._get_num_hidden_states()
    is_last_state = (enc.return_layer == num_hidden - 1)

    missing_grad = []
    unexpected_grad = []
    layer_path_markers = [
        "encoder.layer.",
        "encoder.layers.",
        "encoder.block.",
        "encoder.blocks.",
        "model.layers.",
        "model.blocks.",
        "model.layer.",
        "model.block.",
    ]

    for name, p in enc.named_parameters():
        if not p.requires_grad:
            continue

        is_downstream = False
        matched_marker = next((m for m in layer_path_markers if m in name), None)

        if matched_marker is not None:
            layer_idx = int(name.split(matched_marker)[1].split(".")[0])
            if layer_idx > max_active_layer_idx:
                is_downstream = True
        elif any(kw in name.lower() for kw in ["layernorm", "layer_norm", "norm", "ln_f"]) and not is_last_state:
            is_downstream = True

        if is_downstream:
            if p.grad is not None:
                unexpected_grad.append(name)
        else:
            if p.grad is None:
                missing_grad.append(name)

    if missing_grad:
        fail(f"Graph break detected! {len(missing_grad)} params missing grad: {missing_grad[:3]}...")
        raise AssertionError("Computational graph broke early.")
    elif unexpected_grad:
        fail(f"Graph leak detected! {len(unexpected_grad)} downstream params have grad: {unexpected_grad[:3]}...")
        raise AssertionError("Computational graph leaked downstream.")
    else:
        ok("Gradient flow verified — 100% correct.")


def test_grad_flow_no_backward(model_id: str):
    section("TEST 8 — check_grad_flow Warning (no backward)")

    enc = ImageEncoder(model_id).to(DEVICE)
    enc.set_train_mode(TrainMode.PARTIAL, ratio=0.25)
    enc.check_grad_flow(warn_no_backward=True)
    ok("Warning displayed correctly.")


def test_grad_checkpointing(model_id: str):
    section("TEST 9 — Gradient Checkpointing")

    enc = ImageEncoder(model_id).to(DEVICE)
    enc.set_train_mode(TrainMode.PARTIAL, ratio=0.25)

    try:
        enc.enable_gradient_checkpointing()
        ok("enable_gradient_checkpointing() succeeded.")

        img_size = align_img_size(enc)
        x = make_input(img_size=img_size)
        feats = enc(x)
        loss = feats.mean()
        loss.backward()
        ok("Backward pass with gradient checkpointing succeeded.")

        enc.disable_gradient_checkpointing()
        ok("disable_gradient_checkpointing() succeeded.")
    except Exception as e:
        warn(f"Gradient checkpointing issue: {e}")


def test_print_helpers(model_id: str):
    section("TEST 10 — Print & Display Helpers")

    enc = ImageEncoder(model_id).to(DEVICE)

    for mode in [TrainMode.FROZEN, TrainMode.PARTIAL, TrainMode.FULL]:
        enc.set_train_mode(mode, ratio=0.2)
        print(f"\n--- TrainMode: {mode.value} ---")
        enc.print_trainable_params()

    enc.print_full_layers()
    enc.summary()
    ok("All print helpers executed without errors.")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ImageEncoder Debug & Test Suite")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID, help="HuggingFace model ID")
    args = parser.parse_args()

    logger = DualLogger(LOG_FILE)
    sys.stdout = logger

    print(f"\n{'#' * 70}")
    print(f"  IMAGE ENCODER DEBUG & TEST SUITE")
    print(f"  Model ID : {args.model_id}")
    print(f"  Device   : {DEVICE}")
    print(f"  Log File : {LOG_FILE}")
    print(f"{'#' * 70}")

    tests = [
        ("Init & Properties", lambda: test_init(args.model_id)),
        ("return_layer Validation", lambda: test_return_layer(args.model_id)),
        ("TrainMode", lambda: test_train_modes(args.model_id)),
        ("Token Geometry", lambda: test_token_geometry(args.model_id)),
        ("Forward Pass", lambda: test_forward(args.model_id)),
        ("Input Validation", lambda: test_invalid_inputs(args.model_id)),
        ("Gradient Flow", lambda: test_grad_flow(args.model_id)),
        ("Grad Flow Warning", lambda: test_grad_flow_no_backward(args.model_id)),
        ("Gradient Checkpointing", lambda: test_grad_checkpointing(args.model_id)),
        ("Print Helpers", lambda: test_print_helpers(args.model_id)),
    ]

    passed, failed_count = 0, 0

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception:
            fail(f"[{name}] Failed:")
            traceback.print_exc(file=sys.stdout)
            failed_count += 1

    print(f"\n{SEP}")
    print(f"  RESULTS:  ✅ {passed} Passed   |   ❌ {failed_count} Failed")
    print(f"{SEP}\n")

    sys.stdout = logger.terminal
    logger.close()

    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()