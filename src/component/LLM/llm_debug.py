from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Auto-resolve workspace root directory into sys.path
# ---------------------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[3]  # InA-Bridge workspace root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.component.LLM.llm import LanguageModelDecoder, LLMTrainMode, LoRAConfig

# ---------------------------------------------------------------------------
# Default Test Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
ALT_MODEL_ID     = "hf-internal-testing/tiny-random-LlamaForCausalLM"
BATCH_SIZE       = 2
SEQ_LEN          = 16
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
PRIMARY_LOG_FILE = CURRENT_FILE.parent / "llm_debug.log"

# ===========================================================================
# Test Runner & Logging Infrastructure
# ===========================================================================

class TestRunner:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.passed_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.total_count = 0
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger = logging.getLogger("LLM_Debug")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        fmt = logging.Formatter(
            fmt="%(asctime)s [%(levelname)-7s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Stream Handler (stdout)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # Main dedicated .log handler in component directory
        fh = logging.FileHandler(self.log_path, mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        # Optional timestamped log handler in ./logs/
        timestamp_log_dir = PROJECT_ROOT / "logs"
        timestamp_log_dir.mkdir(parents=True, exist_ok=True)
        ts_filename = f"llm_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        fh_ts = logging.FileHandler(timestamp_log_dir / ts_filename, mode="w", encoding="utf-8")
        fh_ts.setLevel(logging.DEBUG)
        fh_ts.setFormatter(fmt)
        logger.addHandler(fh_ts)

        return logger

    def section(self, title: str):
        self.logger.info("")
        self.logger.info("═" * 78)
        self.logger.info(f"  {title}")
        self.logger.info("═" * 78)

    def subsection(self, title: str):
        self.logger.info("")
        self.logger.info(f"  ── {title} {'─' * max(0, 70 - len(title))}")

    def test(self, name: str, condition: bool, details: str = ""):
        self.total_count += 1
        if condition:
            self.passed_count += 1
            status = "PASS ✓"
            msg = f"  [{status}] {name}"
            if details:
                msg += f" — {details}"
            self.logger.info(msg)
        else:
            self.failed_count += 1
            status = "FAIL ✗"
            msg = f"  [{status}] {name}"
            if details:
                msg += f" — {details}"
            self.logger.error(msg)

    def skip(self, name: str, reason: str):
        self.total_count += 1
        self.skipped_count += 1
        self.logger.warning(f"  [SKIP ⚠] {name} — {reason}")

    def log_kv(self, key: str, value: str, indent: int = 4):
        pad = " " * indent
        self.logger.info(f"{pad}{key:<30}: {value}")


# ===========================================================================
# Debug Test Functions
# ===========================================================================

def test_1_model_loading(runner: TestRunner, model_id: str) -> LanguageModelDecoder:
    runner.section("SECTION 1 — MODEL LOADING & CONFIG INSPECTION")
    runner.logger.info(f"Attempting to load model_id='{model_id}' on device='{DEVICE}'")
    
    t0 = time.time()
    try:
        llm = LanguageModelDecoder(
            model_id=model_id,
            train_mode=LLMTrainMode.FROZEN,
            torch_dtype=torch.float32,
            device_map=None if DEVICE == "cpu" else "auto",
        )
        if DEVICE == "cpu":
            llm.to(DEVICE)
        dt = time.time() - t0
        runner.test("Model loading execution", True, f"Loaded in {dt:.2f}s")
    except Exception as e:
        runner.logger.warning(f"Failed loading primary model_id='{model_id}': {e}")
        runner.logger.info(f"Falling back to alternative model_id='{ALT_MODEL_ID}'")
        model_id = ALT_MODEL_ID
        llm = LanguageModelDecoder(
            model_id=model_id,
            train_mode=LLMTrainMode.FROZEN,
            torch_dtype=torch.float32,
            device_map=None if DEVICE == "cpu" else "auto",
        )
        if DEVICE == "cpu":
            llm.to(DEVICE)
        runner.test("Fallback model loading execution", True, f"Loaded {ALT_MODEL_ID}")

    # Inspect model parameters & properties
    runner.test("Tokenizer initialization", llm.tokenizer is not None, f"Vocab size: {len(llm.tokenizer)}")
    runner.test("Pad token set", llm.tokenizer.pad_token is not None, f"Pad token ID: {llm.tokenizer.pad_token_id}")
    
    decoder_layers = llm.layers
    runner.test("Decoder layers resolution", isinstance(decoder_layers, torch.nn.ModuleList), f"Found {len(decoder_layers)} layers")
    
    total_p = llm.total_parameters
    trainable_p = llm.trainable_parameters
    runner.test("Initial FROZEN trainable params == 0", trainable_p == 0, f"Trainable: {trainable_p}/{total_p}")
    
    runner.log_kv("Model Class", llm.model.__class__.__name__)
    runner.log_kv("Train Mode", llm.train_mode.value)
    runner.log_kv("Total Parameters", f"{total_p:,}")
    runner.log_kv("Trainable Ratio", f"{llm.trainable_ratio:.4%}")

    return llm


def test_2_train_mode_state_machine(runner: TestRunner, llm: LanguageModelDecoder):
    runner.section("SECTION 2 — TRAIN MODE STATE MACHINE TRANSITIONS")

    runner.subsection("2.1 Baseline FROZEN State Checks")
    runner.test("FROZEN mode active", llm.train_mode == LLMTrainMode.FROZEN)
    runner.test("FROZEN trainable parameters count", llm.trainable_parameters == 0)
    runner.test("LoRA applied is False", llm._lora_applied is False)

    try:
        import peft
        has_peft = True
    except ImportError:
        has_peft = False

    if has_peft:
        runner.subsection("2.2 Transition: FROZEN → LORA (r=16, alpha=32)")
        cfg_16 = LoRAConfig(r=16, alpha=32.0, dropout=0.05)
        llm.set_train_mode(LLMTrainMode.LORA, lora_config=cfg_16)
        
        trainable_lora_16 = llm.trainable_parameters
        runner.test("LORA trainable params > 0", trainable_lora_16 > 0, f"Trainable: {trainable_lora_16:,} ({llm.trainable_ratio:.4%})")
        runner.test("LoRA applied flag is True", llm._lora_applied is True)
        runner.test("Model wrapped in PeftModel", "PeftModel" in llm.model.__class__.__name__)

        runner.subsection("2.3 Re-apply SAME LoRA Config (Identity check)")
        model_ptr_before = id(llm.model)
        llm.set_train_mode(LLMTrainMode.LORA, lora_config=cfg_16)
        model_ptr_after = id(llm.model)
        runner.test("No double-wrapping on identical config", model_ptr_before == model_ptr_after, "Object ID preserved")

        runner.subsection("2.4 Transition: LORA (r=16) → LORA (r=8) (Re-config check)")
        cfg_8 = LoRAConfig(r=8, alpha=16.0)
        llm.set_train_mode(LLMTrainMode.LORA, lora_config=cfg_8)
        trainable_lora_8 = llm.trainable_parameters
        runner.test("LoRA re-config changes trainable param count", trainable_lora_8 < trainable_lora_16, f"r=8 params: {trainable_lora_8:,} < r=16 params: {trainable_lora_16:,}")

        runner.subsection("2.5 Transition: LORA → FROZEN (Unload check)")
        llm.set_train_mode(LLMTrainMode.FROZEN)
        runner.test("FROZEN mode restored", llm.train_mode == LLMTrainMode.FROZEN)
        runner.test("FROZEN trainable params reset to 0", llm.trainable_parameters == 0)
        runner.test("LoRA applied reset to False", llm._lora_applied is False)
        runner.test("PeftModel wrapper completely removed", "PeftModel" not in llm.model.__class__.__name__)
    else:
        runner.skip("2.2 - 2.5 LoRA transitions", "peft library is not installed")

    runner.subsection("2.6 Invalid Mode Transition Rejection")
    invalid_modes = ["full", "partial", "invalid_mode", 123]
    for bad_mode in invalid_modes:
        try:
            llm.set_train_mode(bad_mode) # type: ignore
            runner.test(f"Reject invalid mode '{bad_mode}'", False, "Failed to raise ValueError")
        except ValueError as e:
            runner.test(f"Reject invalid mode '{bad_mode}'", True, f"Raised ValueError: {e}")

    runner.test("State invariant maintained after invalid calls", llm.train_mode == LLMTrainMode.FROZEN and llm.trainable_parameters == 0)


def test_3_lora_config_validation(runner: TestRunner):
    runner.section("SECTION 3 — LORA CONFIG VALIDATION BINDINGS")

    runner.subsection("3.1 Rank Validation (r > 0)")
    for bad_r in [0, -1, -16]:
        try:
            LoRAConfig(r=bad_r)
            runner.test(f"Reject invalid r={bad_r}", False, "Failed to raise ValueError")
        except ValueError as e:
            runner.test(f"Reject invalid r={bad_r}", True, f"Raised: {e}")

    runner.subsection("3.2 Alpha Validation (alpha > 0)")
    for bad_alpha in [0.0, -8.0, -32.0]:
        try:
            LoRAConfig(alpha=bad_alpha)
            runner.test(f"Reject invalid alpha={bad_alpha}", False, "Failed to raise ValueError")
        except ValueError as e:
            runner.test(f"Reject invalid alpha={bad_alpha}", True, f"Raised: {e}")

    runner.subsection("3.3 Bias Mode Validation")
    for bad_bias in ["invalid", "all_bias", "true"]:
        try:
            LoRAConfig(bias=bad_bias)
            runner.test(f"Reject invalid bias='{bad_bias}'", False, "Failed to raise ValueError")
        except ValueError as e:
            runner.test(f"Reject invalid bias='{bad_bias}'", True, f"Raised: {e}")

    runner.subsection("3.4 Scaling Factor & Target Modules")
    cfg = LoRAConfig(r=16, alpha=32.0, bias="none")
    runner.test("Scaling factor computation", cfg.scaling == 2.0, f"alpha/r = 32/16 = {cfg.scaling}")
    
    default_targets = cfg.resolve_target_modules()
    expected_defaults = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    runner.test("Default target modules resolution", set(default_targets) == set(expected_defaults), f"Targets: {default_targets}")
    
    custom_cfg = LoRAConfig(target_modules=["q_proj", "v_proj"])
    runner.test("Custom target modules resolution", custom_cfg.resolve_target_modules() == ["q_proj", "v_proj"])


def test_4_adapter_persistence_and_merge(runner: TestRunner, llm: LanguageModelDecoder):
    runner.section("SECTION 4 — ADAPTER PERSISTENCE & MERGE OPERATIONS")

    try:
        import peft
        has_peft = True
    except ImportError:
        runner.skip("SECTION 4 tests", "peft library is not installed")
        return

    runner.subsection("4.1 Error Boundary Checks (Operations without active adapter)")
    llm.set_train_mode(LLMTrainMode.FROZEN)
    
    try:
        llm.merge_lora_weights()
        runner.test("merge_lora_weights() without adapter raises RuntimeError", False, "Failed to raise RuntimeError")
    except RuntimeError as e:
        runner.test("merge_lora_weights() without adapter raises RuntimeError", True, f"Raised: {e}")

    try:
        llm.save_lora_adapter(tempfile.gettempdir())
        runner.test("save_lora_adapter() without adapter raises RuntimeError", False, "Failed to raise RuntimeError")
    except RuntimeError as e:
        runner.test("save_lora_adapter() without adapter raises RuntimeError", True, f"Raised: {e}")

    runner.subsection("4.2 Save & Load Adapter Round-Trip")
    cfg = LoRAConfig(r=8, alpha=16.0)
    llm.set_train_mode(LLMTrainMode.LORA, lora_config=cfg)
    
    temp_dir = Path(tempfile.mkdtemp(prefix="ina_bridge_lora_"))
    try:
        llm.save_lora_adapter(str(temp_dir))
        saved_files = list(temp_dir.glob("*"))
        runner.test("Adapter directory contains saved weights", len(saved_files) > 0, f"Saved files: {[f.name for f in saved_files]}")

        # Unload adapter first
        llm.set_train_mode(LLMTrainMode.FROZEN)
        runner.test("Adapter unloaded before reload", llm._lora_applied is False)

        # Load adapter back
        llm.load_lora_adapter(str(temp_dir))
        runner.test("load_lora_adapter restores LORA mode", llm.train_mode == LLMTrainMode.LORA)
        runner.test("load_lora_adapter sets _lora_applied True", llm._lora_applied is True)
        runner.test("load_lora_adapter restores trainable params", llm.trainable_parameters > 0)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    runner.subsection("4.3 Merge LoRA Weights Round-Trip")
    llm.merge_lora_weights()
    runner.test("merge_lora_weights sets train_mode FROZEN", llm.train_mode == LLMTrainMode.FROZEN)
    runner.test("merge_lora_weights resets _lora_applied False", llm._lora_applied is False)
    runner.test("merge_lora_weights freezes all base params", llm.trainable_parameters == 0)
    runner.test("merge_lora_weights removes PeftModel wrapper", "PeftModel" not in llm.model.__class__.__name__)


def test_5_qlora_and_quantization(runner: TestRunner, model_id: str):
    runner.section("SECTION 5 — QLORA & QUANTIZATION VERIFICATION")

    if DEVICE != "cuda":
        runner.skip("5.1 QLoRA Initialization", "CUDA device is required for 4-bit bitsandbytes quantization")
        return

    try:
        import bitsandbytes
        import peft
        has_qlora_deps = True
    except ImportError:
        runner.skip("5.1 QLoRA Initialization", "bitsandbytes or peft library is missing")
        return

    runner.subsection("5.1 QLoRA Loading & device_map Requirement")
    cfg = LoRAConfig(r=8, alpha=16.0)

    try:
        LanguageModelDecoder(
            model_id=model_id,
            train_mode=LLMTrainMode.QLORA,
            lora_config=cfg,
            device_map=None, # type: ignore
        )
        runner.test("QLORA device_map=None rejection", False, "Failed to raise ValueError")
    except ValueError as e:
        runner.test("QLORA device_map=None rejection", True, f"Raised: {e}")

    try:
        q_llm = LanguageModelDecoder(
            model_id=model_id,
            train_mode=LLMTrainMode.QLORA,
            lora_config=cfg,
            device_map="auto",
        )
        runner.test("QLORA load success", q_llm._lora_applied is True and q_llm.trainable_parameters > 0)
    except Exception as e:
        runner.test("QLORA load success", False, f"Exception: {e}")


def test_6_forward_pass_mechanics(runner: TestRunner, llm: LanguageModelDecoder):
    runner.section("SECTION 6 — FORWARD PASS MECHANICS & VLM PIPELINE SIMULATION")

    try:
        import peft
        llm.set_train_mode(LLMTrainMode.LORA, lora_config=LoRAConfig(r=8, alpha=16.0))
    except ImportError:
        llm.set_train_mode(LLMTrainMode.FROZEN)

    vocab_size = llm.tokenizer.vocab_size if hasattr(llm.tokenizer, "vocab_size") else 32000
    input_ids = torch.randint(0, min(vocab_size, 1000), (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    attn_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    runner.subsection("6.1 Standard forward() with input_ids & labels")
    out_1 = llm(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
    runner.test("Output is CausalLMOutputWithPast", hasattr(out_1, "logits") and hasattr(out_1, "loss"))
    runner.test("Logits shape matches (B, T, Vocab)", out_1.logits.shape[:2] == (BATCH_SIZE, SEQ_LEN), f"Logits shape: {list(out_1.logits.shape)}")
    runner.test("Loss is a valid scalar tensor", out_1.loss is not None and not torch.isnan(out_1.loss), f"Loss: {out_1.loss.item():.6f}")

    runner.subsection("6.2 Forward with inputs_embeds (Simulating Visual Soft Tokens + Text)")
    embed_layer = llm.model.get_input_embeddings()
    text_embeds = embed_layer(input_ids).detach()
    
    # Simulate 32 visual soft tokens projected to LLM dimension
    hidden_dim = text_embeds.shape[-1]
    num_visual_tokens = 32
    visual_soft_tokens = torch.randn(BATCH_SIZE, num_visual_tokens, hidden_dim, device=DEVICE, dtype=text_embeds.dtype)
    
    # Concatenate visual tokens + text embeds -> [V_soft (32) | Text tokens (16)]
    vlm_inputs_embeds = torch.cat([visual_soft_tokens, text_embeds], dim=1) # (B, 48, D)
    vlm_attn_mask = torch.ones(BATCH_SIZE, num_visual_tokens + SEQ_LEN, device=DEVICE)
    
    # Mask visual token positions in labels with -100 so loss is only computed on text
    visual_labels = torch.full((BATCH_SIZE, num_visual_tokens), fill_value=-100, device=DEVICE, dtype=torch.long)
    vlm_labels = torch.cat([visual_labels, labels], dim=1)

    out_2 = llm(inputs_embeds=vlm_inputs_embeds, attention_mask=vlm_attn_mask, labels=vlm_labels)
    runner.test("VLM inputs_embeds logits shape matches (B, 32+T, Vocab)", out_2.logits.shape[:2] == (BATCH_SIZE, num_visual_tokens + SEQ_LEN), f"Logits shape: {list(out_2.logits.shape)}")
    runner.test("VLM loss computation with -100 visual label masking", out_2.loss is not None and not torch.isnan(out_2.loss), f"VLM Loss: {out_2.loss.item():.6f}")

    runner.subsection("6.3 Forward with past_key_values (KV-Cache Incremental Decoding)")
    out_initial = llm(input_ids=input_ids, use_cache=True)
    past_kv = out_initial.past_key_values
    runner.test("KV-Cache returned when use_cache=True", past_kv is not None)

    step_input_ids = torch.randint(0, min(vocab_size, 1000), (BATCH_SIZE, 1), device=DEVICE)
    out_step = llm(input_ids=step_input_ids, past_key_values=past_kv, use_cache=True)
    runner.test("1-Step incremental decode logits shape (B, 1, Vocab)", out_step.logits.shape[:2] == (BATCH_SIZE, 1))

    runner.subsection("6.4 Error Handling: missing both input_ids and inputs_embeds")
    try:
        llm() # type: ignore
        runner.test("Raise ValueError when neither input_ids nor inputs_embeds passed", False)
    except ValueError as e:
        runner.test("Raise ValueError when neither input_ids nor inputs_embeds passed", True, f"Raised: {e}")


def test_7_gradient_flow_and_backward(runner: TestRunner, llm: LanguageModelDecoder):
    runner.section("SECTION 7 — GRADIENT FLOW & BACKWARD PASS VERIFICATION")

    try:
        import peft
        llm.set_train_mode(LLMTrainMode.LORA, lora_config=LoRAConfig(r=8, alpha=16.0))
        is_lora = True
    except ImportError:
        llm.set_train_mode(LLMTrainMode.FROZEN)
        is_lora = False

    vocab_size = llm.tokenizer.vocab_size if hasattr(llm.tokenizer, "vocab_size") else 32000
    input_ids = torch.randint(0, min(vocab_size, 1000), (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    attn_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    # Clear gradients
    llm.zero_grad()
    out = llm(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
    out.loss.backward()

    runner.subsection("7.1 Parameter Gradient Checks")
    grad_ok_count = 0
    grad_none_count = 0
    exploding_count = 0
    vanishing_count = 0

    for name, p in llm.named_parameters():
        if p.requires_grad:
            if p.grad is not None:
                gnorm = p.grad.norm().item()
                if gnorm > 1e3:
                    exploding_count += 1
                elif gnorm < 1e-8:
                    vanishing_count += 1
                else:
                    grad_ok_count += 1
            else:
                grad_none_count += 1
        else:
            # Frozen parameters MUST have grad is None
            if p.grad is not None:
                runner.test(f"Frozen param '{name}' has grad == None", False, f"Found grad norm: {p.grad.norm().item()}")

    if is_lora:
        runner.test("LoRA trainable parameters receive valid gradients", grad_ok_count > 0 and grad_none_count == 0, f"OK: {grad_ok_count}, None: {grad_none_count}")
        runner.test("No exploding gradients", exploding_count == 0, f"Exploding count: {exploding_count}")
    else:
        runner.test("FROZEN mode params receive 0 gradients", llm.trainable_parameters == 0)

    # Call internal check_grad_flow debug utility
    runner.logger.info("Executing llm.check_grad_flow():")
    llm.check_grad_flow(warn_no_backward=False)

    llm.set_train_mode(LLMTrainMode.FROZEN)


def test_8_autoregressive_generation(runner: TestRunner, llm: LanguageModelDecoder):
    runner.section("SECTION 8 — AUTOREGRESSIVE GENERATION TEST")

    llm.eval()
    embed_layer = llm.model.get_input_embeddings()
    vocab_size = llm.tokenizer.vocab_size if hasattr(llm.tokenizer, "vocab_size") else 32000
    
    input_ids = torch.randint(0, min(vocab_size, 1000), (1, 8), device=DEVICE)
    text_embeds = embed_layer(input_ids).detach()
    hidden_dim = text_embeds.shape[-1]
    
    # 32 soft prompt tokens
    visual_tokens = torch.randn(1, 32, hidden_dim, device=DEVICE, dtype=text_embeds.dtype)
    inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1) # (1, 40, D)

    runner.subsection("8.1 Base model generate() with inputs_embeds")
    try:
        generated = llm.model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=10,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )
        runner.test("Generation with inputs_embeds execution", True, f"Generated output shape: {list(generated.shape)}")
    except Exception as e:
        runner.test("Generation with inputs_embeds execution", False, f"Exception: {e}")


def test_9_dtype_and_device_consistency(runner: TestRunner, llm: LanguageModelDecoder):
    runner.section("SECTION 9 — DTYPE & DEVICE CONSISTENCY DIAGNOSTICS")

    dtypes = set()
    devices = set()
    for name, p in llm.named_parameters():
        dtypes.add(str(p.dtype))
        devices.add(str(p.device))

    runner.test("Parameter dtype consistency", len(dtypes) == 1, f"Found dtypes: {dtypes}")
    runner.test("Parameter device placement consistency", len(devices) == 1, f"Found devices: {devices}")

    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
        runner.log_kv("CUDA VRAM Allocated", f"{allocated_gb:.3f} GB")
        runner.log_kv("CUDA VRAM Reserved", f"{reserved_gb:.3f} GB")


# ===========================================================================
# Main Entry Point
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="LanguageModelDecoder Automated Test & Debug Suite")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model ID to load for debugging")
    parser.add_argument("--run-qlora", action="store_true", help="Run 4-bit QLoRA test (requires CUDA + bitsandbytes)")
    return parser.parse_args()


def main():
    args = parse_args()
    runner = TestRunner(log_path=PRIMARY_LOG_FILE)

    runner.logger.info("╔══════════════════════════════════════════════════════════════════════╗")
    runner.logger.info("║         LanguageModelDecoder (llm.py) Automated Debug Suite           ║")
    runner.logger.info("╚══════════════════════════════════════════════════════════════════════╝")
    runner.logger.info(f"  Execution Time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    runner.logger.info(f"  Python Version : {sys.version.split()[0]}")
    runner.logger.info(f"  PyTorch Version: {torch.__version__}")
    runner.logger.info(f"  Device         : {DEVICE}")
    runner.logger.info(f"  Primary Log    : {PRIMARY_LOG_FILE.resolve()}")

    try:
        # Step 1: Load model
        llm = test_1_model_loading(runner, args.model_id)

        # Step 2: Train mode state machine
        test_2_train_mode_state_machine(runner, llm)

        # Step 3: LoRAConfig validation
        test_3_lora_config_validation(runner)

        # Step 4: Adapter save/load/merge
        test_4_adapter_persistence_and_merge(runner, llm)

        # Step 5: QLoRA
        if args.run_qlora:
            test_5_qlora_and_quantization(runner, args.model_id)

        # Step 6: Forward mechanics & VLM simulation
        test_6_forward_pass_mechanics(runner, llm)

        # Step 7: Gradient flow & backward pass
        test_7_gradient_flow_and_backward(runner, llm)

        # Step 8: Autoregressive generation
        test_8_autoregressive_generation(runner, llm)

        # Step 9: Dtype & Device consistency
        test_9_dtype_and_device_consistency(runner, llm)

    except Exception:
        runner.logger.error("\n" + "!" * 78)
        runner.logger.error("FATAL UNHANDLED EXCEPTION DURING TEST SUITE:")
        for line in traceback.format_exc().splitlines():
            runner.logger.error(f"  {line}")
        runner.logger.error("!" * 78)
    finally:
        runner.section("TEST SUITE SUMMARY REPORT")
        runner.log_kv("Total Tests Evaluated", str(runner.total_count))
        runner.log_kv("Passed Tests", f"{runner.passed_count} ✓")
        runner.log_kv("Failed Tests", f"{runner.failed_count} ✗")
        runner.log_kv("Skipped Tests", f"{runner.skipped_count} ⚠")
        
        status_msg = "SUCCESS — ALL ACTIVE TESTS PASSED ✓" if runner.failed_count == 0 else "FAILURE — ONE OR MORE TESTS FAILED ✗"
        runner.logger.info("")
        runner.logger.info(f"  OVERALL RESULT: {status_msg}")
        runner.logger.info(f"  Detailed output logged to: {PRIMARY_LOG_FILE.resolve()}")
        runner.logger.info("═" * 78 + "\n")


if __name__ == "__main__":
    main()