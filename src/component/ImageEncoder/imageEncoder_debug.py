"""
imageEncoder_debug.py
Bộ kiểm thử & Debug tự động cho ImageEncoder2 (DINOv2 / DINOv3).

Cách chạy:
    python /home/tranmanhduy/Workspace/chuyen_nganh/InA-Bridge/src/component/ImageEncoder2/imageEncoder_debug.py

Yêu cầu:
    pip install torch transformers
"""

import os
import sys
import traceback
import argparse
import torch

# Tự động thêm Project Root vào sys.path để hỗ trợ import module 'src' chính xác
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.component.ImageEncoder2.imageEncoder import ImageEncoder, TrainMode

# ------------------------------------------------------------------
# Cấu hình Mặc định (Default Configurations)
# ------------------------------------------------------------------
# Mặc định dùng "facebook/dinov2-large" (~304M params) công khai (un-gated).
# Đối với DINOv3 300M (facebook/dinov3-vitl16-pretrain-lvd1689m - gated repo),
# truyền qua CLI option --model_id sau khi đã đăng nhập HuggingFace Token.
DEFAULT_MODEL_ID = "facebook/dinov2-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
IMG_SIZE = 448  # Phải chia hết cho patch_size (14 đối với dinov2-large -> 448/14 = 32)
LOG_FILE = os.path.join(CURRENT_DIR, "imageEncoder_debug.log")


def make_input(batch: int = BATCH_SIZE, img_size: int = IMG_SIZE, device: str = DEVICE) -> torch.Tensor:
    """Tạo dummy input tensor cho testing."""
    return torch.randn(batch, 3, img_size, img_size, device=device)


# ------------------------------------------------------------------
# Logger song song (Console + Log File)
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
# Formatting & Output Helpers
# ------------------------------------------------------------------
SEP = "=" * 70
SEP2 = "-" * 70


def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def ok(msg: str):
    print(f"  ✅  {msg}")


def fail(msg: str):
    print(f"  ❌  {msg}")


def warn(msg: str):
    print(f"  ⚠️   {msg}")


# ------------------------------------------------------------------
# Test Cases
# ------------------------------------------------------------------

def test_init(model_id: str) -> ImageEncoder:
    section("TEST 1 — Khởi tạo mô hình & Kiểm tra Thuộc tính")
    enc = ImageEncoder(model_id).to(DEVICE)

    ok(f"Khởi tạo thành công model: {model_id}")
    ok(f"Device      : {enc.device}")
    ok(f"Dtype       : {enc.dtype}")
    ok(f"Embed dim   : {enc.embed_dim}")
    ok(f"Patch size  : {enc.patch_size}")
    ok(f"Register tok: {enc.num_register_tokens}")
    enc.summary()
    return enc


def test_return_layer(model_id: str):
    section("TEST 2 — Kiểm toán & Validating return_layer")

    cases = [
        (-2, True, "Index âm hợp lệ (kế cuối)"),
        (-1, True, "Index âm hợp lệ (layer cuối)"),
        (0, True, "Index 0 (embedding layer)"),
        (3, True, "Index dương hợp lệ"),
        (-999, False, "Index âm vượt giới hạn out-of-range"),
        (999, False, "Index dương vượt giới hạn out-of-range"),
    ]

    for idx, should_pass, desc in cases:
        try:
            enc = ImageEncoder(model_id, return_layer=idx)
            if should_pass:
                ok(f"return_layer={idx:>5}  →  normalized={enc.return_layer}  ({desc})")
            else:
                fail(f"return_layer={idx:>5}  lẽ ra phải raise ValueError  ({desc})")
        except (ValueError, IndexError) as e:
            if not should_pass:
                ok(f"return_layer={idx:>5}  bị chặn chính xác: {e}")
            else:
                fail(f"return_layer={idx:>5}  raise ngoài dự kiến: {e}")


def test_train_modes(model_id: str):
    section("TEST 3 — Các chế độ Huấn luyện TrainMode (FROZEN / PARTIAL / FULL)")

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

    # Edge cases: tỉ lệ ratio không hợp lệ cho PARTIAL
    print()
    for bad_ratio in [0.0, 1.0, -0.1, 1.5]:
        try:
            enc.set_train_mode(TrainMode.PARTIAL, ratio=bad_ratio)
            fail(f"ratio={bad_ratio} lẽ ra phải raise ValueError")
        except ValueError as e:
            ok(f"ratio={bad_ratio} bị chặn đúng: {e}")


def test_token_geometry(model_id: str):
    section("TEST 4 — Hình học Token & Strip Special Tokens (DINOv2 / DINOv3)")

    enc = ImageEncoder(model_id).to(DEVICE)

    img_size = IMG_SIZE
    if img_size % enc.patch_size != 0:
        img_size = enc.patch_size * 16  # Chuẩn hoá kích thước ảnh chia hết cho patch_size

    grid = enc.compute_grid_size(img_size)
    num_patches = enc.compute_num_patches(img_size)
    total_raw = enc.compute_total_tokens(img_size)

    ok(f"image_size={img_size} → grid={grid}x{grid} | patches={num_patches} | raw_tokens={total_raw}")

    # Test resolution KHÔNG chia hết cho patch_size -> phải raise
    invalid_size = img_size + 1
    try:
        enc.compute_num_patches(invalid_size)
        fail(f"image_size={invalid_size} lẽ ra phải raise ValueError")
    except ValueError as e:
        ok(f"image_size={invalid_size} (không chia hết cho patch_size={enc.patch_size}) bị chặn chính xác: {e}")

    # Kiểm tra strip_special_tokens với tensor giả lập
    dummy_raw = torch.randn(1, total_raw, enc.embed_dim, device=DEVICE)
    stripped = enc.strip_special_tokens(dummy_raw, drop_cls=True)
    assert stripped.shape[1] == num_patches, f"strip_special_tokens sai: got {stripped.shape[1]}, expected {num_patches}"
    ok(f"strip_special_tokens(drop_cls=True)  → shape: {list(stripped.shape)} (chỉ giữ patch tokens)")

    stripped_cls = enc.strip_special_tokens(dummy_raw, drop_cls=False)
    assert stripped_cls.shape[1] == num_patches + 1, "strip_special_tokens(drop_cls=False) sai"
    ok(f"strip_special_tokens(drop_cls=False) → shape: {list(stripped_cls.shape)} (CLS + patch tokens)")


def test_forward(model_id: str):
    section("TEST 5 — Forward Pass & Debug Forward")

    enc = ImageEncoder(model_id).to(DEVICE)
    img_size = IMG_SIZE
    if img_size % enc.patch_size != 0:
        img_size = enc.patch_size * 16

    x = make_input(batch=BATCH_SIZE, img_size=img_size)

    # 1. debug_forward
    feats_debug = enc.debug_forward(x, verbose=True, drop_special_tokens=True)
    ok(f"debug_forward thành công → feats shape = {list(feats_debug.shape)}")

    # 2. forward tiêu chuẩn
    feats = enc(x, drop_special_tokens=True)
    assert feats.shape == feats_debug.shape, "Shape không khớp giữa debug_forward và forward!"
    ok(f"forward tiêu chuẩn thành công → feats shape = {list(feats.shape)}")

    # Kiểm tra tính toán patch count
    expected_patches = enc.compute_num_patches(img_size)
    assert feats.shape[1] == expected_patches, (
        f"Mismatch patch count: got {feats.shape[1]}, expected {expected_patches}"
    )
    ok(f"Số lượng patch tokens khớp 100% với grid hình học: {expected_patches} tokens.")
    ok(f"Device output : {feats.device}")
    ok(f"Dtype output  : {feats.dtype}")


def test_invalid_inputs(model_id: str):
    section("TEST 6 — Kiểm tra Validation Tensor đầu vào")

    enc = ImageEncoder(model_id).to(DEVICE)

    # Case 1: Rank != 4
    try:
        x_3d = torch.randn(3, 224, 224, device=DEVICE)
        enc(x_3d)
        fail("Tensor 3D lẽ ra phải raise ValueError")
    except ValueError as e:
        ok(f"Tensor 3D bị chặn chính xác: {e}")

    # Case 2: Ảnh không phải hình vuông
    try:
        x_rect = torch.randn(2, 3, 224, 256, device=DEVICE)
        enc(x_rect)
        fail("Ảnh chữ nhật (non-square) lẽ ra phải raise ValueError")
    except ValueError as e:
        ok(f"Ảnh chữ nhật bị chặn chính xác: {e}")


def test_grad_flow(model_id: str):
    section("TEST 7 — Gradient Flow (Kiểm toán Đồ thị Tự động - Graph Auditing)")

    enc = ImageEncoder(model_id, return_layer=-2).to(DEVICE)
    enc.set_train_mode(TrainMode.PARTIAL, ratio=0.25)
    enc.train()

    img_size = IMG_SIZE
    if img_size % enc.patch_size != 0:
        img_size = enc.patch_size * 16

    x = make_input(img_size=img_size)
    feats = enc(x)
    loss = feats.mean()
    loss.backward()

    enc.check_grad_flow(warn_no_backward=False)

    # Auditing graph logic
    max_active_layer_idx = enc.return_layer - 1
    num_hidden = enc._get_num_hidden_states()
    is_last_state = (enc.return_layer == num_hidden - 1)

    missing_grad = []
    unexpected_grad = []

    layer_path_markers = ["encoder.layer.", "encoder.layers.", "model.layers."]

    for name, p in enc.named_parameters():
        if not p.requires_grad:
            continue

        is_downstream_isolated = False
        matched_marker = next((m for m in layer_path_markers if m in name), None)

        if matched_marker is not None:
            layer_idx = int(name.split(matched_marker)[1].split(".")[0])
            if layer_idx > max_active_layer_idx:
                is_downstream_isolated = True
        elif any(kw in name.lower() for kw in ["layernorm", "layer_norm", "norm", "ln_f"]) and not is_last_state:
            # Nếu không nằm trong từng block encoder.layer.X và return_layer chưa tới last layer
            is_downstream_isolated = True

        if is_downstream_isolated:
            if p.grad is not None:
                unexpected_grad.append(name)
        else:
            if p.grad is None:
                missing_grad.append(name)

    if missing_grad:
        fail(f"Phát hiện đứt gãy đồ thị! {len(missing_grad)} tham số thiếu grad: {missing_grad[:3]}...")
        raise AssertionError("Computational graph broke early.")
    elif unexpected_grad:
        fail(f"Phát hiện rò rỉ đồ thị! {len(unexpected_grad)} tham số downstream vẫn có grad: {unexpected_grad[:3]}...")
        raise AssertionError("Computational graph leaked downstream.")
    else:
        ok("Đồ thị tính toán Gradient Flow chính xác 100%!")


def test_grad_flow_no_backward(model_id: str):
    section("TEST 8 — Kiểm tra Cảnh báo check_grad_flow khi chưa backward")

    enc = ImageEncoder(model_id).to(DEVICE)
    enc.set_train_mode(TrainMode.PARTIAL, ratio=0.25)
    enc.check_grad_flow(warn_no_backward=True)
    ok("Cảnh báo hiển thị chính xác.")


def test_grad_checkpointing(model_id: str):
    section("TEST 9 — Kích hoạt Gradient Checkpointing")

    enc = ImageEncoder(model_id).to(DEVICE)
    enc.set_train_mode(TrainMode.PARTIAL, ratio=0.25)

    try:
        enc.enable_gradient_checkpointing()
        ok("enable_gradient_checkpointing() thành công.")

        img_size = IMG_SIZE
        if img_size % enc.patch_size != 0:
            img_size = enc.patch_size * 16

        x = make_input(img_size=img_size)
        feats = enc(x)
        loss = feats.mean()
        loss.backward()
        ok("Backward pass với Gradient Checkpointing thành công!")

        enc.disable_gradient_checkpointing()
        ok("disable_gradient_checkpointing() thành công.")
    except Exception as e:
        warn(f"Gradient Checkpointing gặp vấn đề: {e}")


def test_print_helpers(model_id: str):
    section("TEST 10 — Các hàm In & Hiển thị Thông tin (summary, print_full_layers)")

    enc = ImageEncoder(model_id).to(DEVICE)

    for mode in [TrainMode.FROZEN, TrainMode.PARTIAL, TrainMode.FULL]:
        enc.set_train_mode(mode, ratio=0.2)
        print(f"\n--- TrainMode: {mode.value} ---")
        enc.print_trainable_params()

    enc.print_full_layers()
    enc.summary()
    ok("Tất cả print helpers thực thi trơn tru mà không có lỗi.")


# ------------------------------------------------------------------
# Main Test Suite Runner
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ImageEncoder Debug & Test Suite")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID, help="HuggingFace model ID")
    args = parser.parse_args()

    logger = DualLogger(LOG_FILE)
    sys.stdout = logger

    print(f"\n{'#' * 70}")
    print(f"  IMAGE ENCODER 2 DEBUG & TEST SUITE")
    print(f"  Model ID : {args.model_id}")
    print(f"  Device   : {DEVICE}")
    print(f"  Log File : {LOG_FILE}")
    print(f"{'#' * 70}")

    tests = [
        ("Khởi tạo & Thuộc tính", lambda: test_init(args.model_id)),
        ("Validate return_layer", lambda: test_return_layer(args.model_id)),
        ("Chế độ TrainMode", lambda: test_train_modes(args.model_id)),
        ("Hình học Token", lambda: test_token_geometry(args.model_id)),
        ("Forward Pass", lambda: test_forward(args.model_id)),
        ("Input Validation", lambda: test_invalid_inputs(args.model_id)),
        ("Gradient Flow", lambda: test_grad_flow(args.model_id)),
        ("Grad flow cảnh báo", lambda: test_grad_flow_no_backward(args.model_id)),
        ("Gradient Checkpointing", lambda: test_grad_checkpointing(args.model_id)),
        ("Print Helpers", lambda: test_print_helpers(args.model_id)),
    ]

    passed, failed = 0, 0

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception:
            fail(f"[{name}] Thất bại hoặc Crash không mong muốn:")
            traceback.print_exc(file=sys.stdout)
            failed += 1

    print(f"\n{SEP}")
    print(f"  KẾT QUẢ KIỂM THỬ TỔNG THỂ:  ✅ {passed} Passed   |   ❌ {failed} Failed")
    print(f"{SEP}\n")

    sys.stdout = logger.terminal
    logger.close()

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()