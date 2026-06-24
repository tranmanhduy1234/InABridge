"""
debug_encoder.py
Chạy: python debug_encoder.py
Yêu cầu: pip install torch transformers
"""

import sys
import torch
import traceback
from src.component.ImageEncoder.imageEncoder import ImageEncoder, TrainMode

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MODEL_ID = "facebook/dinov2-large"   # Đổi thành model bạn dùng
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
BATCH    = 2
IMG_SIZE = 448
LOG_FILE = "/home/tranmanhduy/Workspace/chuyen_nganh/InA-Bridge/src/component/ImageEncoder/imageEncoder_debug.log"  # Tên file xuất log mặc định

# Dummy input
def make_input():
    return torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)


# ------------------------------------------------------------------
# Bộ điều hướng luồng (Tee Logger) — Tự động ghi song song Console + File
# ------------------------------------------------------------------
class DecoderLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        # Ép định dạng utf-8 để hiển thị chuẩn emoji ✅ ❌ ⚠️ trên mọi hệ điều hành
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
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
SEP  = "=" * 70
SEP2 = "-" * 70

def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def ok(msg):  print(f"  ✅  {msg}")
def fail(msg): print(f"  ❌  {msg}")
def warn(msg): print(f"  ⚠️   {msg}")


# ------------------------------------------------------------------
# Test 1 — Khởi tạo cơ bản
# ------------------------------------------------------------------
def test_init():
    section("TEST 1 — Khởi tạo cơ bản")
    enc = ImageEncoder(MODEL_ID).to(DEVICE)
    enc.summary()
    ok("Khởi tạo thành công")
    return enc

# ------------------------------------------------------------------
# Test 2 — Validate return_layer (âm, dương, out-of-range)
# ------------------------------------------------------------------
def test_return_layer():
    section("TEST 2 — Validate return_layer")

    cases = [
        (-2,   True,  "index âm hợp lệ"),
        (-1,   True,  "index âm = last layer"),
        (0,    True,  "index 0 = embedding layer"),
        (3,    True,  "index dương hợp lệ"),
        (-999, False, "index âm out-of-range"),
        (999,  False, "index dương out-of-range"),
    ]

    for idx, should_pass, desc in cases:
        try:
            enc = ImageEncoder(MODEL_ID, return_layer=idx)
            if should_pass:
                ok(f"return_layer={idx:>5}  →  normalized={enc.return_layer}  ({desc})")
            else:
                fail(f"return_layer={idx:>5}  lẽ ra phải raise ValueError  ({desc})")
        except (ValueError, IndexError) as e:
            if not should_pass:
                ok(f"return_layer={idx:>5}  bị chặn đúng: {e}")
            else:
                fail(f"return_layer={idx:>5}  raise không mong muốn: {e}")

# ------------------------------------------------------------------
# Test 3 — Tất cả TrainMode
# ------------------------------------------------------------------
def test_train_modes():
    section("TEST 3 — TrainMode: FROZEN / PARTIAL / FULL")

    enc = ImageEncoder(MODEL_ID).to(DEVICE)

    for mode, ratio, expect_trainable in [
        (TrainMode.FROZEN,  0.2,  False),
        (TrainMode.PARTIAL, 0.25, True),
        (TrainMode.PARTIAL, 0.5,  True),
        (TrainMode.FULL,    0.2,  True),
    ]:
        enc.set_train_mode(mode, ratio=ratio)
        tp = enc.trainable_parameters
        tr = enc.trainable_ratio
        has_trainable = tp > 0

        status = ok if has_trainable == expect_trainable else fail
        status(
            f"mode={mode.value:<8}  ratio={ratio:.2f}  "
            f"trainable={tp:>10,}  ({tr:.2%})"
        )

    # Edge case: ratio = 0 hoặc 1 phải raise
    print()
    for bad_ratio in [0.0, 1.0, -0.1, 1.5]:
        try:
            enc.set_train_mode(TrainMode.PARTIAL, ratio=bad_ratio)
            fail(f"ratio={bad_ratio} lẽ ra phải raise ValueError")
        except ValueError as e:
            ok(f"ratio={bad_ratio} bị chặn đúng: {e}")


# ------------------------------------------------------------------
# Test 4 — Forward pass + debug_forward
# ------------------------------------------------------------------
def test_forward():
    section("TEST 4 — Forward pass")

    enc = ImageEncoder(MODEL_ID).to(DEVICE)
    x   = make_input()

    # debug_forward in đầy đủ hidden states
    feats = enc.debug_forward(x, verbose=True)
    ok(f"debug_forward OK  →  feats.shape={list(feats.shape)}")

    # forward thông thường
    feats2 = enc(x)
    assert feats.shape == feats2.shape, "Shape không khớp giữa debug_forward và forward!"
    ok(f"forward OK  →  feats.shape={list(feats2.shape)}")

    # Kiểm tra dtype và device
    assert feats2.device.type == DEVICE.split(":")[0], "Device mismatch!"
    ok(f"Device OK: {feats2.device}")
    ok(f"Dtype : {feats2.dtype}")

# ------------------------------------------------------------------
# Test 5 — Gradient flow (backward pass) - BẢN CẬP NHẬT CHÍNH XÁC
# ------------------------------------------------------------------
def test_grad_flow():
    section("TEST 5 — Gradient flow (Computational Graph Verification)")

    # Khởi tạo encoder lên Device
    enc = ImageEncoder(MODEL_ID).to(DEVICE)
    
    # Thiết lập chế độ PARTIAL để kiểm tra luồng gradient hỗn hợp
    enc.set_train_mode(TrainMode.PARTIAL, ratio=0.25)
    enc.train()
    x      = make_input()
    feats  = enc(x)                          
    loss   = feats.mean()
    loss.backward()

    # Gọi hàm hiển thị chi tiết trực quan của bạn
    enc.check_grad_flow()

    # -----------------------------------------------------------
    # LOGIC KIỂM TOÁN ĐỒ THỊ TỰ ĐỘNG (AUTOMATED GRAPH AUDITING)
    # -----------------------------------------------------------
    # Do enc.return_layer đã được tự động chuẩn hóa thành index dương (ví dụ: 23)
    # Layer lớn nhất đóng góp vào feats này sẽ là: return_layer - 1 (ví dụ: layer 22)
    max_active_layer_idx = enc.return_layer - 1
    num_hidden = enc._get_num_hidden_states()
    is_last_state = (enc.return_layer == num_hidden - 1)

    missing_grad = []      # Lẽ ra phải có grad nhưng lại bị None (Lỗi đứt gãy đồ thị)
    unexpected_grad = []   # Lẽ ra phải là None nhưng lại có grad (Lỗi rò rỉ đồ thị)

    for name, p in enc.named_parameters():
        if not p.requires_grad:
            continue  # Bỏ qua các lớp đã bị đóng băng hoàn toàn (FROZEN)
        
        # Mặc định coi lớp đó nằm trong luồng tính toán (Upstream)
        is_downstream_isolated = False
        
        # 1. Kiểm tra nếu thuộc các khối Encoder Block
        if "encoder.layer." in name:
            # Trích xuất số thứ tự layer từ chuỗi "model.encoder.layer.X. ..."
            layer_idx = int(name.split("encoder.layer.")[1].split(".")[0])
            if layer_idx > max_active_layer_idx:
                is_downstream_isolated = True
                
        # 2. Kiểm tra nếu thuộc final layernorm
        elif "layernorm" in name and not is_last_state:
            is_downstream_isolated = True

        # ---- Tiến hành đối chiếu hệ thống ----
        if is_downstream_isolated:
            # Các lớp nằm sau điểm cắt đồ thị BẮT BUỘC grad phải bằng None
            if p.grad is not None:
                unexpected_grad.append(name)
        else:
            # Các lớp nằm trước hoặc bằng điểm cắt đồ thị BẮT BUỘC phải sinh grad
            if p.grad is None:
                missing_grad.append(name)

    # -----------------------------------------------------------
    # ĐÁNH GIÁ VÀ NGHIỆM THU
    # -----------------------------------------------------------
    if missing_grad:
        fail(f"Phát hiện lỗi đứt gãy đồ thị! {len(missing_grad)} tham số thiếu grad: {missing_grad[:3]}...")
        raise AssertionError("Computational graph broke early.")
        
    elif unexpected_grad:
        fail(f"Phát hiện lỗi rò rỉ đồ thị! {len(unexpected_grad)} tham số tính toán thừa: {unexpected_grad[:3]}...")
        raise AssertionError("Computational graph leaked downstream.")
        
    else:
        ok("Hệ thống Đồ thị tính toán đạt độ chính xác 100%!")
        ok(f"  -> Luồng upstream (<= layer.{max_active_layer_idx}) sinh gradient và cập nhật tốt.")
        if max_active_layer_idx < len(enc.layers) - 1:
            ok(f"  -> Luồng downstream (>= layer.{max_active_layer_idx+1}) ngắt graph tự động thành công (Tiết kiệm bộ nhớ).")


# ------------------------------------------------------------------
# Test 6 — Kiểm tra warn khi chưa backward
# ------------------------------------------------------------------
def test_grad_flow_no_backward():
    section("TEST 6 — check_grad_flow khi chưa backward (expect warning)")

    enc = ImageEncoder(MODEL_ID).to(DEVICE)
    enc.set_train_mode(TrainMode.PARTIAL, ratio=0.25)
    # KHÔNG gọi backward
    enc.check_grad_flow(warn_no_backward=True)
    ok("Warning hiển thị đúng (xem output ở trên)")


# ------------------------------------------------------------------
# Test 7 — Gradient checkpointing
# ------------------------------------------------------------------
def test_grad_checkpointing():
    section("TEST 7 — Gradient checkpointing")

    enc = ImageEncoder(MODEL_ID).to(DEVICE)
    enc.set_train_mode(TrainMode.PARTIAL, ratio=0.25)

    try:
        enc.enable_gradient_checkpointing()
        ok("enable_gradient_checkpointing OK")

        x     = make_input().requires_grad_(False)
        feats = enc(x)
        loss  = feats.mean()
        loss.backward()
        ok("Backward với gradient checkpointing OK")

        enc.disable_gradient_checkpointing()
        ok("disable_gradient_checkpointing OK")
    except Exception as e:
        warn(f"Gradient checkpointing không được hỗ trợ bởi model này: {e}")

# ------------------------------------------------------------------
# Test 8 — print helpers không crash
# ------------------------------------------------------------------
def test_print_helpers():
    section("TEST 8 — print_full_layers / print_trainable_params / summary")

    enc = ImageEncoder(MODEL_ID).to(DEVICE)

    for mode in [TrainMode.FROZEN, TrainMode.PARTIAL, TrainMode.FULL]:
        enc.set_train_mode(mode, ratio=0.2)
        print(f"\n--- {mode.value} ---")
        enc.print_trainable_params()

    enc.print_full_layers()
    enc.summary()
    ok("Tất cả print helpers OK")

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    # 1. Kích hoạt bộ chuyển hướng luồng sang File Log song song với Console
    logger = DecoderLogger(LOG_FILE)
    sys.stdout = logger

    print(f"\n{'#' * 70}")
    print(f"  ImageEncoder Debug Suite")
    print(f"  Model  : {MODEL_ID}")
    print(f"  Device : {DEVICE}")
    print(f"  Log    : Ghi đồng thời ra file '{LOG_FILE}'")
    print(f"{'#' * 70}")

    tests = [
        ("Init",                   test_init),
        ("Return layer validation", test_return_layer),
        ("Train modes",            test_train_modes),
        ("Forward pass",           test_forward),
        ("Gradient flow",          test_grad_flow),
        ("Grad flow no backward",  test_grad_flow_no_backward),
        ("Gradient checkpointing", test_grad_checkpointing),
        ("Print helpers",          test_print_helpers),
    ]

    passed, failed = 0, 0

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            fail(f"[{name}] crash không mong muốn:")
            # Định hướng traceback in thẳng vào sys.stdout (để logger bắt được vào file)
            traceback.print_exc(file=sys.stdout)
            failed += 1

    print(f"\n{SEP}")
    print(f"  KẾT QUẢ:  ✅ {passed} passed   ❌ {failed} failed")
    print(f"{SEP}\n")
    
    # 2. Khôi phục lại luồng hệ thống cũ và đóng file một cách an toàn
    sys.stdout = logger.terminal
    logger.close()

if __name__ == "__main__":
    main()