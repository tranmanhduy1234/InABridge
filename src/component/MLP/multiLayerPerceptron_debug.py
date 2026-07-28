"""
debug_qwen_projector.py
Kiểm tra shape, gradient flow, gate saturation, và param count
cho QwenProjector (DINOv2-L + DeBERTa Q-Former → Qwen2.5-7B).

Chạy: python debug_qwen_projector.py
"""

import logging
import copy
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# Đổi sang class đã định nghĩa hoặc import từ module dự án của bạn
from qwen_projector import QwenProjector

# ─── Logger setup ─────────────────────────────────────────────────────────────

# Tự động lưu log cùng thư mục với script hiện tại (tránh lỗi đường dẫn tuyệt đối)
LOG_PATH = Path(__file__).parent / "debug_qwen_projector.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler()  # In ra cả terminal để tiện theo dõi
    ],
)
log = logging.getLogger("projector_debug")

# ─── Helpers ──────────────────────────────────────────────────────────────────

def sep(title: str = ""):
    w = 60
    if title:
        pad = max(0, (w - len(title) - 2) // 2)
        line = "─" * pad + f" {title} " + "─" * (w - pad - len(title) - 2)
    else:
        line = "─" * w
    log.info(line)


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    by_layer = {name: p.numel() for name, p in model.named_parameters()}
    return {"total": total, "trainable": trainable, "by_layer": by_layer}

# ─── Test 1: Shape check ──────────────────────────────────────────────────────

def test_shapes(model, B=2, N=64, qformer_dim=768):
    sep("1. Shape check")
    x = torch.randn(B, N, qformer_dim)
    log.info(f"  Input  : {tuple(x.shape)}")
    with torch.no_grad():
        out, gate = model(x)
    log.info(f"  Output : {tuple(out.shape)}")
    log.info(f"  Gate   : {tuple(gate.shape)}")
    expected = (B, N, model.fc2.out_features)
    
    assert out.shape == expected, f"Shape mismatch! Expected {expected}, got {out.shape}"
    log.info("  ✓ Shape OK")

# ─── Test 2: Param count ──────────────────────────────────────────────────────

def test_param_count(model):
    sep("2. Parameter count")
    info = count_params(model)
    log.info(f"  Total params     : {info['total']:,}")
    log.info(f"  Trainable params : {info['trainable']:,}")
    log.info("")
    for name, n in info["by_layer"].items():
        log.info(f"  {name:<30s}  {n:>12,}")

# ─── Test 3: Gradient flow ────────────────────────────────────────────────────

def test_gradient_flow(model, B=2, N=64, qformer_dim=768):
    sep("3. Gradient flow")
    model.zero_grad()  # Reset gradient trước khi backward
    
    x = torch.randn(B, N, qformer_dim, requires_grad=True)
    out, _ = model(x)

    target = torch.randn_like(out)
    loss = F.mse_loss(out, target)
    loss.backward()

    dead, total = 0, 0
    for name, p in model.named_parameters():
        if p.grad is None:
            log.error(f"  ✗ NO GRAD: {name}")
            dead += 1
        else:
            gnorm = p.grad.norm().item()
            if gnorm > 1e-8:
                log.info(f"  ✓ {name:<30s}  grad_norm={gnorm:.4e}")
            else:
                log.warning(f"  ⚠ {name:<30s}  grad_norm={gnorm:.4e}  (near-zero)")
        total += 1

    if dead == 0:
        log.info(f"\n  ✓ All {total} param tensors have gradients")
    else:
        log.error(f"\n  ✗ {dead}/{total} param tensors missing gradients!")

    if x.grad is not None:
        log.info(f"  Input grad norm  : {x.grad.norm().item():.4e}")

# ─── Test 4: Gate saturation ─────────────────────────────────────────────────

def test_gate_saturation(model, B=4, N=64, qformer_dim=768, n_samples=8):
    sep("4. Gate saturation (sigmoid outputs)")
    gates = []
    with torch.no_grad():
        for _ in range(n_samples):
            x = torch.randn(B, N, qformer_dim)
            _, gate = model(x)
            gates.append(gate)

    g = torch.stack(gates)
    mean_g  = g.mean().item()
    std_g   = g.std().item()
    frac_lo = (g < 0.05).float().mean().item()
    frac_hi = (g > 0.95).float().mean().item()

    log.info(f"  Mean gate value   : {mean_g:.4f}  (ideal ≈ 0.5 at init)")
    log.info(f"  Std  gate value   : {std_g:.4f}")
    log.info(f"  Fraction < 0.05   : {frac_lo:.2%}  (saturated closed)")
    log.info(f"  Fraction > 0.95   : {frac_hi:.2%}  (saturated open)")

    if frac_lo + frac_hi > 0.3:
        log.warning("  ⚠ High saturation — consider bias init or lower lr for gate")
    else:
        log.info("  ✓ Gate not saturated at init")

# ─── Test 5: Numerical stability ─────────────────────────────────────────────

def test_numerical_stability(model, qformer_dim=768):
    sep("5. Numerical stability")
    cases = {
        "normal"     : torch.randn(1, 32, qformer_dim),
        "large_scale": torch.randn(1, 32, qformer_dim) * 10,
        "near_zero"  : torch.randn(1, 32, qformer_dim) * 1e-4,
    }
    with torch.no_grad():
        for label, x in cases.items():
            out, _ = model(x)
            has_nan = torch.isnan(out).any().item()
            has_inf = torch.isinf(out).any().item()
            out_std = out.std().item()
            if has_nan or has_inf:
                log.error(f"  ✗ [{label:<12s}]  out_std={out_std:.4e}  nan={has_nan}  inf={has_inf}")
            else:
                log.info(f"  ✓ [{label:<12s}]  out_std={out_std:.4e}  nan={has_nan}  inf={has_inf}")

# ─── Test 6: FP16 / BF16 (Đã sửa lỗi khởi tạo lại mô hình) ────────────────────

def test_half_precision(model, qformer_dim=768):
    sep("6. Half-precision (bfloat16)")
    try:
        # Sử dụng deepcopy đối tượng `model` hiện tại thay vì khởi tạo mới
        model_bf16 = copy.deepcopy(model).bfloat16()
        x = torch.randn(1, 32, qformer_dim, dtype=torch.bfloat16)
        with torch.no_grad():
            out, _ = model_bf16(x)
        has_nan = torch.isnan(out).any().item()
        log.info(f"  ✓ bfloat16 OK — out dtype={out.dtype}, nan={has_nan}, shape={tuple(out.shape)}")
    except Exception as e:
        log.error(f"  ✗ bfloat16 failed: {e}")

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    QFORMER_DIM = 768
    HIDDEN_DIM  = 1536
    LLM_DIM     = 3584   # Qwen2.5-7B hidden size

    log.info("")
    sep("QwenProjector Debug")
    log.info(f"  qformer_dim={QFORMER_DIM}, hidden_dim={HIDDEN_DIM}, llm_dim={LLM_DIM}")
    sep()

    model = QwenProjector(QFORMER_DIM, HIDDEN_DIM, LLM_DIM)

    test_shapes(model, qformer_dim=QFORMER_DIM)
    log.info("")
    test_param_count(model)
    log.info("")
    test_gradient_flow(model, qformer_dim=QFORMER_DIM)
    log.info("")
    test_gate_saturation(model, qformer_dim=QFORMER_DIM)
    log.info("")
    test_numerical_stability(model, qformer_dim=QFORMER_DIM)
    log.info("")
    test_half_precision(model, qformer_dim=QFORMER_DIM)
    sep()
    log.info("  Debug complete.")

    print(f"\nLog saved → {LOG_PATH.resolve()}")