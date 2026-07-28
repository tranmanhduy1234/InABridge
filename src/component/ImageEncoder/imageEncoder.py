from enum import Enum
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoModel

class TrainMode(str, Enum):
    """Các chế độ huấn luyện hỗ trợ cho Vision Encoder."""
    FROZEN = "frozen"
    PARTIAL = "partial"
    FULL = "full"

class ImageEncoder(nn.Module):
    """
    Wrapper chuyên dụng cho các dòng Vision Encoder họ Transformer (ViT / DINOv2 / DINOv3).
    Tương thích hoàn toàn với các mô hình HuggingFace AutoModel.

    Đặc điểm nổi bật & Xử lý DINOv3/DINOv2:
      1. Register Tokens (DINOv3): DINOv3 chèn `num_register_tokens` (mặc định 4)
         ngay sau CLS token. Các token này không mang thông tin không gian hữu ích
         và PHẢI được loại bỏ trước khi đưa vào Q-Former cross-attention.
      2. Dynamic Patch Geometry: Tính toán linh hoạt `grid_size` và `num_patches`
         dựa theo resolution thực tế và `patch_size` thay vì hardcode.
      3. Partial Fine-tuning: Cho phép unfreeze một tỉ lệ (`unfreeze_ratio`) các transformer blocks
         cuối cùng kèm theo LayerNorm hệ thống.
      4. Computational Graph Inspection: Tích hợp công cụ tự động kiểm tra luồng gradient flow.
    """

    def __init__(
        self,
        model_id: str,
        return_layer: int = -2,
    ):
        """
        Khởi tạo ImageEncoder từ mô hình pre-trained HuggingFace.

        Args:
            model_id: Identifier tên hoặc đường dẫn checkpoint trên HuggingFace Hub.
            return_layer: Layer index cần trích xuất feature. Hỗ trợ index âm (vd: -1 là layer cuối, -2 là kế cuối).
        """
        super().__init__()

        self.model_id: str = model_id
        self.model = AutoModel.from_pretrained(model_id)

        self.train_mode: TrainMode = TrainMode.FROZEN
        self.unfreeze_ratio: float = 0.0

        # Kiểm tra và chuẩn hoá return_layer
        num_hidden = self._get_num_hidden_states()
        self.return_layer: int = self._validate_return_layer(return_layer, num_hidden)

        # Trích xuất cấu hình đặc thù DINOv2 / DINOv3
        self.num_register_tokens: int = getattr(self.model.config, "num_register_tokens", 0)
        self.patch_size: int = self._get_patch_size()

        # Mặc định ở chế độ FROZEN khi khởi tạo
        self.set_train_mode(TrainMode.FROZEN)

    # ------------------------------------------------------------------
    # Properties & Device / Dtype Introspection
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """Thiết bị (device) hiện tại của mô hình."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        """Kiểu dữ liệu (dtype) hiện tại của mô hình."""
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32

    @property
    def embed_dim(self) -> int:
        """Chiều của không gian feature ẩn (hidden dimension / embedding dimension)."""
        cfg = self.model.config
        for attr in ("hidden_size", "embed_dim", "d_model"):
            if hasattr(cfg, attr):
                return getattr(cfg, attr)
        raise AttributeError(f"Không thể xác định embed_dim từ config của {self.model.__class__.__name__}.")

    @property
    def layers(self) -> nn.ModuleList:
        """ModuleList chứa các lớp Transformer Encoder block."""
        return self._resolve_encoder_layers()

    @property
    def trainable_parameters(self) -> int:
        """Tổng số tham số có requires_grad=True trên toàn bộ ImageEncoder."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_parameters(self) -> int:
        """Tổng số tham số của toàn bộ ImageEncoder."""
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_ratio(self) -> float:
        """Tỉ lệ tham số được huấn luyện (trainable / total)."""
        total = self.total_parameters
        return self.trainable_parameters / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Config Introspection & Helpers
    # ------------------------------------------------------------------

    def _get_num_hidden_states(self) -> int:
        """Trả về tổng số hidden states sinh ra (bao gồm embedding layer + N encoder layers)."""
        cfg = self.model.config
        if hasattr(cfg, "num_hidden_layers"):
            return cfg.num_hidden_layers + 1
        raise RuntimeError(
            f"Không thể xác định số hidden states của {self.model.__class__.__name__}. "
            "Vui lòng kiểm tra thuộc tính model.config."
        )

    def _get_patch_size(self) -> int:
        """Lấy giá trị patch_size từ config của mô hình."""
        cfg = self.model.config
        if hasattr(cfg, "patch_size"):
            return cfg.patch_size
        raise RuntimeError(
            f"Không tìm thấy patch_size trong config của {self.model.__class__.__name__}."
        )

    @staticmethod
    def _validate_return_layer(return_layer: int, num_hidden: int) -> int:
        """
        Chuẩn hoá index (hỗ trợ âm) và kiểm tra out-of-range.
        Trả về index dương tương ứng trong dải [0, num_hidden - 1].
        """
        if return_layer < -num_hidden or return_layer >= num_hidden:
            raise ValueError(
                f"return_layer={return_layer} out-of-range. "
                f"Mô hình có {num_hidden} hidden states, "
                f"index hợp lệ: [{-num_hidden}, {num_hidden - 1}]."
            )
        if return_layer < 0:
            return num_hidden + return_layer
        return return_layer

    def _resolve_encoder_layers(self) -> nn.ModuleList:
        """
        Duyệt tìm ModuleList chứa các encoder layer tương thích với nhiều kiến trúc HF ViT.
        """
        if hasattr(self.model, "encoder"):
            enc = self.model.encoder
            if hasattr(enc, "layer"):
                return enc.layer
            if hasattr(enc, "layers"):
                return enc.layers
        if hasattr(self.model, "layers"):
            return self.model.layers
        if hasattr(self.model, "layer"):
            return self.model.layer
        raise RuntimeError(
            f"Không tìm thấy encoder layers trên {self.model.__class__.__name__}."
        )

    # ------------------------------------------------------------------
    # Patch & Grid Geometry
    # ------------------------------------------------------------------

    def compute_grid_size(self, image_size: int) -> int:
        """Tính số patch mỗi cạnh (grid size = image_size / patch_size). Raise nếu không chia hết."""
        if image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size={image_size} không chia hết cho patch_size={self.patch_size}. "
                f"Vui lòng chọn resolution là bội số của {self.patch_size}."
            )
        return image_size // self.patch_size

    def compute_num_patches(self, image_size: int) -> int:
        """Tính tổng số patch tokens (grid * grid) ở resolution tương ứng."""
        grid = self.compute_grid_size(image_size)
        return grid * grid

    def compute_total_tokens(self, image_size: int, include_cls: bool = True) -> int:
        """
        Tính tổng số token trong raw hidden state (CLS + register tokens + patch tokens).
        """
        num_patches = self.compute_num_patches(image_size)
        cls_count = 1 if include_cls else 0
        return cls_count + self.num_register_tokens + num_patches

    def strip_special_tokens(self, feats: torch.Tensor, drop_cls: bool = True) -> torch.Tensor:
        """
        Loại bỏ CLS token và Register tokens khỏi chuỗi feature tensor.
        Layout mặc định của DINOv2/v3: [CLS, reg_1, ..., reg_R, patch_1, ..., patch_N].

        Args:
            feats: Tensor hình dạng (B, L, D).
            drop_cls: Nếu True, loại bỏ luôn CLS token. Nếu False, giữ lại CLS token ở đầu.

        Returns:
            Tensor (B, num_patches, D) nếu drop_cls=True, hoặc (B, 1 + num_patches, D) nếu drop_cls=False.
        """
        if feats.dim() != 3:
            raise ValueError(f"Dự kiến tensor 3D (B, L, D), nhận được tensor shape {list(feats.shape)}.")

        start = 1 + self.num_register_tokens if drop_cls else self.num_register_tokens
        if drop_cls:
            return feats[:, start:, :]

        cls_tok = feats[:, :1, :]
        patch_tok = feats[:, 1 + self.num_register_tokens:, :]
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
        """
        Cấu hình chế độ huấn luyện (FROZEN, PARTIAL, FULL).

        Args:
            mode: Giá trị TrainMode enum.
            ratio: Tỉ lệ phần trăm các Transformer blocks cuối cùng cần unfreeze (chỉ dùng cho PARTIAL).
            unfreeze_embeddings: Có unfreeze phần patch embedding hay không (dùng cho PARTIAL).
        """
        self.train_mode = mode
        self.unfreeze_ratio = ratio if mode == TrainMode.PARTIAL else (1.0 if mode == TrainMode.FULL else 0.0)

        # Đóng băng toàn bộ trước
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
                raise ValueError(f"ratio phải nằm trong khoảng (0.0, 1.0), nhận được {ratio}.")

            enc_layers = self._resolve_encoder_layers()
            num_layers = len(enc_layers)
            num_unfreeze = max(1, round(num_layers * ratio))

            # Unfreeze N blocks cuối cùng
            for layer in enc_layers[-num_unfreeze:]:
                for p in layer.parameters():
                    p.requires_grad_(True)

            # Unfreeze LayerNorm cuối (tìm trong self.model và self.model.encoder)
            norm_found = False
            for target_obj in (self.model, getattr(self.model, "encoder", None)):
                if target_obj is None:
                    continue
                for attr in ("layernorm", "layer_norm", "norm", "ln_f"):
                    if hasattr(target_obj, attr):
                        norm_module = getattr(target_obj, attr)
                        if isinstance(norm_module, nn.Module):
                            for p in norm_module.parameters():
                                p.requires_grad_(True)
                            norm_found = True
                            break
                if norm_found:
                    break

            # Tùy chọn unfreeze patch embeddings
            if unfreeze_embeddings:
                for target_obj in (self.model, getattr(self.model, "embeddings", None)):
                    if target_obj is None:
                        continue
                    for attr in ("embeddings", "patch_embed", "patch_embeddings", "patch_embedding"):
                        if hasattr(target_obj, attr):
                            embed_module = getattr(target_obj, attr)
                            if isinstance(embed_module, nn.Module):
                                for p in embed_module.parameters():
                                    p.requires_grad_(True)
                                break
            return

        raise ValueError(f"Train mode không được hỗ trợ: {mode}")

    def enable_gradient_checkpointing(self) -> None:
        """Kích hoạt Gradient Checkpointing để tiết kiệm bộ nhớ GPU."""
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        else:
            raise AttributeError("Mô hình không hỗ trợ gradient_checkpointing_enable().")

    def disable_gradient_checkpointing(self) -> None:
        """Tắt Gradient Checkpointing."""
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        else:
            raise AttributeError("Mô hình không hỗ trợ gradient_checkpointing_disable().")

    # ------------------------------------------------------------------
    # Inspection & Debug Helpers
    # ------------------------------------------------------------------

    def print_full_layers(self) -> None:
        """In cấu trúc chi tiết tất cả các leaf modules."""
        print("\n[MODEL STRUCTURE]")
        print(f"{'Name':<60} {'Type'}")
        print("-" * 80)
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:
                print(f"{name:<60} {module.__class__.__name__}")

    def print_trainable_params(self) -> None:
        """In danh sách các tham số đang được phép tính gradient."""
        print("\n[TRAINABLE PARAMETERS]")
        print(f"{'Name':<60} {'Shape'}")
        print("-" * 80)
        found = False
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                print(f"{name:<60} {list(p.shape)}")
                found = True
        if not found:
            print("  (Không có tham số nào đang trainable - Chế độ FROZEN)")

    def summary(self) -> None:
        """In tóm tắt thông tin tổng quan của ImageEncoder."""
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
        """
        Kiểm tra trạng thái gradient của các tham số trainable sau lan truyền ngược.

        Args:
            warn_no_backward: In cảnh báo nếu chưa thực hiện loss.backward().
        """
        print("\n[GRADIENT FLOW CHECK]")
        print(f"{'Name':<60} {'Status'}")
        print("-" * 80)

        ok_count = 0
        no_grad_count = 0

        for name, p in self.named_parameters():
            if p.requires_grad:
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
                "\n  ⚠  Cảnh báo: Tất cả parameter trainable đều chưa có gradient.\n"
                "     Vui lòng gọi loss.backward() trước khi chạy check_grad_flow()."
            )

    # ------------------------------------------------------------------
    # Forward Pass & Debug Forward
    # ------------------------------------------------------------------

    def _validate_input_tensor(self, pixel_values: torch.Tensor) -> None:
        """Kiểm tra tính hợp lệ của tensor đầu vào."""
        if pixel_values.dim() != 4:
            raise ValueError(
                f"Đầu vào pixel_values phải là 4D tensor (B, C, H, W), "
                f"nhận được tensor {pixel_values.dim()}D với shape {list(pixel_values.shape)}."
            )
        _, _, h, w = pixel_values.shape
        if h != w:
            raise ValueError(f"Đầu vào ảnh phải có dạng hình vuông (H == W), nhận được ({h}, {w}).")
        if h % self.patch_size != 0:
            raise ValueError(
                f"Kích thước ảnh {h}x{w} không chia hết cho patch_size={self.patch_size}."
            )

    def debug_forward(
        self,
        pixel_values: torch.Tensor,
        verbose: bool = True,
        drop_special_tokens: bool = True,
    ) -> torch.Tensor:
        """
        Thực hiện forward pass kèm in log chi tiết từng hidden state và token breakdown để debug.
        """
        self._validate_input_tensor(pixel_values)

        outputs = self.model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Mô hình không trả về hidden_states. Kiểm tra lại output_hidden_states.")

        image_size = pixel_values.shape[-1]
        expected_total = self.compute_total_tokens(image_size)

        if verbose:
            print(f"\n[DEBUG FORWARD] Input shape: {list(pixel_values.shape)}")
            print(f"  Tổng số hidden states: {len(hidden_states)}")
            print(
                f"  Cấu trúc Token kỳ vọng @ {image_size}x{image_size}: "
                f"1 CLS + {self.num_register_tokens} register + "
                f"{self.compute_num_patches(image_size)} patch = {expected_total} tokens total."
            )
            for i, hs in enumerate(hidden_states):
                marker = " <-- return_layer" if i == self.return_layer else ""
                mismatch = "" if hs.shape[1] == expected_total else "  ⚠ MISMATCH"
                print(f"  [{i:>3}]  shape={list(hs.shape)}{marker}{mismatch}")

        if self.return_layer >= len(hidden_states):
            raise IndexError(
                f"return_layer={self.return_layer} vượt quá số hidden states ({len(hidden_states)})."
            )

        feats = hidden_states[self.return_layer]
        if drop_special_tokens:
            feats = self.strip_special_tokens(feats, drop_cls=True)

        if verbose:
            print(f"\n  Output feature shape ({'patch-only' if drop_special_tokens else 'raw'}): {list(feats.shape)}")

        return feats

    def forward(
        self,
        pixel_values: torch.Tensor,
        drop_special_tokens: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass tiêu chuẩn trích xuất features từ Vision Encoder.

        Args:
            pixel_values: Tensor ảnh đầu vào (B, C, H, W).
            drop_special_tokens: Nếu True, loại bỏ CLS và register tokens, chỉ giữ lại patch tokens.

        Returns:
            Tensor đại diện feature ảnh (B, num_patches, D) hoặc (B, L_raw, D).
        """
        self._validate_input_tensor(pixel_values)

        outputs = self.model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Mô hình không trả về hidden_states.")

        if self.return_layer >= len(hidden_states):
            raise IndexError(
                f"return_layer={self.return_layer} vượt quá số hidden states ({len(hidden_states)})."
            )

        feats = hidden_states[self.return_layer]
        if drop_special_tokens:
            feats = self.strip_special_tokens(feats, drop_cls=True)

        return feats