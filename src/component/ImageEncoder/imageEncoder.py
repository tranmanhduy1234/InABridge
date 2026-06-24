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
        
        self.model = AutoModel.from_pretrained(model_id)
        self.model_id = model_id
        self.train_mode = TrainMode.FROZEN
        self.unfreeze_ratio = 0.0

        num_hidden = self._get_num_hidden_states()
        self.return_layer = self._validate_return_layer(return_layer, num_hidden)

        self.set_train_mode(TrainMode.FROZEN)

    def _get_num_hidden_states(self) -> int:
        """Trả về số lượng hidden states mà model này sinh ra."""
        cfg = self.model.config
        # Hầu hết ViT-family: num_hidden_layers + 1 (embedding + N encoder layers)
        if hasattr(cfg, "num_hidden_layers"):
            return cfg.num_hidden_layers + 1
        raise RuntimeError(
            f"Không thể xác định số hidden states của {self.model.__class__.__name__}. "
            "Hãy set return_layer thủ công sau khi kiểm tra model.config."
        )

    @staticmethod
    def _validate_return_layer(return_layer: int, num_hidden: int) -> int:
        """
        Chuẩn hoá index (hỗ trợ âm) và raise rõ ràng nếu out-of-range.
        Trả về index dương tương ứng để tránh lỗi im lặng khi num_hidden thay đổi.
        """
        if return_layer < -num_hidden or return_layer >= num_hidden:
            raise ValueError(
                f"return_layer={return_layer} out-of-range. "
                f"Model có {num_hidden} hidden states, "
                f"index hợp lệ: [{-num_hidden}, {num_hidden - 1}]."
            )
            
        if return_layer < 0:
            return num_hidden + return_layer
        return return_layer

    def _resolve_encoder_layers(self) -> nn.ModuleList:
        if hasattr(self.model, "encoder"):
            enc = self.model.encoder
            if hasattr(enc, "layer"):
                return enc.layer
            if hasattr(enc, "layers"):
                return enc.layers
        if hasattr(self.model, "layers"):
            return self.model.layers
        raise RuntimeError(
            f"Không tìm thấy encoder layers trên {self.model.__class__.__name__}. "
            "Kiểm tra cấu trúc bằng print_full_layers() rồi override _resolve_encoder_layers()."
        )

    @property
    def layers(self) -> nn.ModuleList:
        return self._resolve_encoder_layers()

    @property
    def trainable_parameters(self) -> int:
        """Tính trên toàn bộ ImageEncoder (self), không chỉ self.model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_ratio(self) -> float:
        total = self.total_parameters
        return self.trainable_parameters / total if total > 0 else 0.0

    def set_train_mode(
        self,
        mode: TrainMode,
        ratio: float = 0.1,
        unfreeze_embeddings: bool = False,
    ):
        self.train_mode = mode
        self.unfreeze_ratio = ratio if mode == TrainMode.PARTIAL else 0.0

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
                raise ValueError(f"ratio phải nằm trong khoảng (0, 1), nhận được {ratio}.")
            
            enc_layers = self._resolve_encoder_layers()
            num_layers = len(enc_layers)
            num_unfreeze = max(1, round(num_layers * ratio))

            for layer in enc_layers[-num_unfreeze:]:
                for p in layer.parameters():
                    p.requires_grad_(True)

            for attr in ("layernorm", "layer_norm", "norm", "ln_f"):
                if hasattr(self.model, attr):
                    for p in getattr(self.model, attr).parameters():
                        p.requires_grad_(True)
                    break  # Chỉ unfreeze cái đầu tiên tìm thấy

            # Tuỳ chọn: unfreeze embeddings
            if unfreeze_embeddings:
                for attr in ("embeddings", "patch_embed", "patch_embedding"):
                    if hasattr(self.model, attr):
                        for p in getattr(self.model, attr).parameters():
                            p.requires_grad_(True)
                        break
            return

        raise ValueError(f"Unsupported train mode: {mode}")

    def enable_gradient_checkpointing(self):
        self.model.gradient_checkpointing_enable()

    def disable_gradient_checkpointing(self):
        self.model.gradient_checkpointing_disable()

    # ------------------------------------------------------------------
    # Debug / inspect
    # ------------------------------------------------------------------

    def print_full_layers(self):
        """In toàn bộ leaf modules kèm tên đầy đủ và loại."""
        print("\n[MODEL STRUCTURE]")
        print(f"{'Name':<60} {'Type'}")
        print("-" * 80)
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:
                print(f"{name:<60} {module.__class__.__name__}")

    def print_trainable_params(self):
        """In tên các parameter đang được train kèm shape."""
        print("\n[TRAINABLE PARAMETERS]")
        print(f"{'Name':<60} {'Shape'}")
        print("-" * 80)
        found = False
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                print(f"{name:<60} {list(p.shape)}")
                found = True
        if not found:
            print("  (không có parameter nào đang được train — mode FROZEN)")

    def summary(self):
        """
        In tóm tắt nhất quán, tính trên self (toàn ImageEncoder)
        thay vì chỉ self.model để phản ánh đúng trainable_ratio.
        """
        num_hidden = self._get_num_hidden_states()
        total = self.total_parameters
        trainable = self.trainable_parameters

        print("\n" + "=" * 60)
        print("[SUMMARY]")
        print("=" * 60)
        print(f"  Model ID        : {self.model_id}")
        print(f"  Model class     : {self.model.__class__.__name__}")
        print(f"  Train mode      : {self.train_mode.value}")
        print(f"  Unfreeze ratio  : {self.unfreeze_ratio:.2f}")
        print(f"  Return layer    : {self.return_layer} / {num_hidden - 1}")
        print(f"  Trainable params: {trainable:,}")
        print(f"  Total params    : {total:,}")
        print(f"  Trainable %     : {trainable / total:.4%}" if total > 0 else "  Trainable %     : N/A")
        print("=" * 60)

    def check_grad_flow(self, warn_no_backward: bool = True):
        """
        Kiểm tra gradient sau backward pass.

        Args:
            warn_no_backward: nếu True, in cảnh báo khi không có grad nào cả
                               (thường do chưa gọi loss.backward()).
        """
        print("\n[GRADIENT FLOW CHECK]")
        print(f"{'Name':<60} {'Status'}")
        print("-" * 80)

        ok_count = 0
        no_grad_count = 0

        for name, p in self.named_parameters():
            if p.requires_grad:
                if p.grad is not None:
                    # Phát hiện vanishing / exploding gradient
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
                "\n  ⚠  Tất cả parameter đều chưa có gradient.\n"
                "     Hãy chạy loss.backward() trước khi gọi check_grad_flow()."
            )

    def debug_forward(
        self,
        pixel_values: torch.Tensor,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Chạy forward và in thông tin từng hidden state để debug return_layer.

        Args:
            pixel_values: tensor đầu vào.
            verbose: nếu True, in shape của tất cả hidden states.

        Returns:
            Feature tensor ở layer self.return_layer.
        """
        outputs = self.model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states

        if hidden_states is None:
            raise RuntimeError(
                "Model không trả về hidden_states. "
                "Kiểm tra lại output_hidden_states=True có được hỗ trợ không."
            )

        if verbose:
            print(f"\n[DEBUG FORWARD]  input shape: {list(pixel_values.shape)}")
            print(f"  Tổng hidden states: {len(hidden_states)}")
            for i, hs in enumerate(hidden_states):
                marker = " <-- return_layer" if i == self.return_layer - 1 else ""
                print(f"  [{i:>3}]  shape={list(hs.shape)}{marker}")

        if self.return_layer >= len(hidden_states):
            raise IndexError(
                f"return_layer={self.return_layer} vượt quá số hidden states"
                f"thực tế ({len(hidden_states)}). "
                "Khởi tạo lại encoder với return_layer hợp lệ."
            )

        feats = hidden_states[self.return_layer]

        if verbose:
            print(f"\n  Output feats shape: {list(feats.shape)}")

        return feats

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states

        if hidden_states is None:
            raise RuntimeError(
                "Model không trả về hidden_states."
                "Kiểm tra lại output_hidden_states=True có được hỗ trợ không."
            )

        if self.return_layer >= len(hidden_states):
            raise IndexError(
                f"return_layer={self.return_layer} vượt quá số hidden states "
                f"thực tế ({len(hidden_states)})."
            )

        return hidden_states[self.return_layer]