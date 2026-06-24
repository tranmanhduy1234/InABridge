"""
ITG (Image-grounded Text Generation) Loss
==========================================
Từ paper BLIP-2 (Li et al., 2023).

Ý tưởng chính:
  - Q-Former nhận queries + image features + text tokens làm input.
  - Multimodal Causal Attention Mask:
      * Query tokens: tự attention với nhau, KHÔNG thấy text tokens.
      * Text tokens:  thấy TẤT CẢ query tokens + các text tokens trước đó (causal).
  - Loss = cross-entropy trên text tokens (language modeling loss).
  - Điều này buộc queries phải nén đủ thông tin ảnh để model sinh ra đúng text.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────────────────────────────────────
# 1. Tạo Multimodal Causal Attention Mask
# ─────────────────────────────────────────────────────────────────────────────

def build_multimodal_causal_mask(num_queries: int, seq_len: int) -> torch.Tensor:
    """
    Tạo attention mask cho ITG loss theo BLIP-2.

    Args:
        num_queries: số lượng learnable query tokens (Q).
        seq_len:     độ dài chuỗi text (T).

    Returns:
        mask: BoolTensor [Q+T, Q+T], True = được attend, False = bị mask.

    Layout:
        Rows/Cols = [Q tokens | T tokens]

        Query tokens (Q×Q): bidirectional → tất cả True
        Query→Text  (Q×T):  False  (queries không thấy text)
        Text→Query  (T×Q):  True   (text thấy tất cả queries)
        Text→Text   (T×T):  causal lower-triangular
    """
    total = num_queries + seq_len
    mask = torch.zeros(total, total, dtype=torch.bool)

    # [1] Q↔Q: bidirectional (queries thấy nhau)
    mask[:num_queries, :num_queries] = True

    # [2] Q→T: False (queries KHÔNG thấy text — để tránh information leak)
    # mask[:num_queries, num_queries:] = False  # đã là False

    # [3] T→Q: True (mỗi text token thấy toàn bộ queries)
    mask[num_queries:, :num_queries] = True

    # [4] T→T: causal lower-triangular
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    mask[num_queries:, num_queries:] = causal

    return mask


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tính ITG Loss (cross-entropy trên text tokens)
# ─────────────────────────────────────────────────────────────────────────────

def compute_itg_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_queries: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Tính ITG (Image-grounded Text Generation) loss.

    Args:
        logits:      [B, Q+T, vocab_size]  — output logits của Q-Former.
        labels:      [B, T]                — ground-truth text token IDs.
                     Dùng ignore_index để bỏ padding / token không cần tính loss.
        num_queries: số lượng query tokens (Q) để bỏ qua phần đầu của logits.
        ignore_index: token ID bị bỏ qua (mặc định -100).

    Returns:
        loss: scalar tensor — cross-entropy loss trên text tokens.

    Lưu ý:
        - Logits tại vị trí t dự đoán token t+1 (shift bởi 1).
        - Chỉ tính loss trên text tokens (bỏ query tokens).
    """
    B, total, vocab_size = logits.shape
    T = total - num_queries
    assert labels.shape == (B, T), (
        f"labels phải có shape [B, T] = [{B}, {T}], nhận được {labels.shape}"
    )

    # Lấy logits phần text: [B, T, vocab_size]
    text_logits = logits[:, num_queries:, :]  # [B, T, V]

    # Shift: logit tại t dự đoán label t+1
    # → dự đoán: text_logits[:, :-1, :]  (T-1 positions)
    # → target:  labels[:, 1:]           (T-1 tokens)
    shift_logits = text_logits[:, :-1, :].contiguous()  # [B, T-1, V]
    shift_labels = labels[:, 1:].contiguous()            # [B, T-1]

    # Cross-entropy (bỏ padding qua ignore_index)
    loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),  # [B*(T-1), V]
        shift_labels.view(-1),              # [B*(T-1)]
        ignore_index=ignore_index,
        reduction="mean",
    )
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# 3. Mini Q-Former để demo
# ─────────────────────────────────────────────────────────────────────────────

class MiniQFormer(nn.Module):
    """
    Q-Former rút gọn để minh họa ITG loss.
    Dùng một Transformer encoder với custom attention mask.
    """

    def __init__(
        self,
        num_queries: int = 4,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        vocab_size: int = 100,
        img_feat_dim: int = 128,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Learnable query embeddings
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, d_model))

        # Image projection: img_feat_dim → d_model
        self.img_proj = nn.Linear(img_feat_dim, d_model)

        # Token embedding cho text
        self.token_embed = nn.Embedding(vocab_size, d_model)

        # Positional encoding đơn giản
        self.pos_enc = nn.Embedding(512, d_model)

        # Cross-attention: queries attend to image features
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_norm  = nn.LayerNorm(d_model)

        # Self-attention với causal mask
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection → vocab
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        img_features: torch.Tensor,   # [B, num_img_tokens, img_feat_dim]
        text_tokens: torch.Tensor,     # [B, T]
    ) -> dict:
        B, T = text_tokens.shape
        Q = self.num_queries
        device = img_features.device

        # 1. Project image features
        img_feat = self.img_proj(img_features)  # [B, num_img, d_model]

        # 2. Cross-attention: queries → image features
        queries = self.query_embed.expand(B, -1, -1)  # [B, Q, d_model]
        queries, _ = self.cross_attn(queries, img_feat, img_feat)
        queries = self.cross_norm(queries)

        # 3. Embed text tokens + positional encoding
        pos = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
        text_emb = self.token_embed(text_tokens) + self.pos_enc(pos)  # [B, T, d]

        # 4. Ghép queries + text: [B, Q+T, d_model]
        x = torch.cat([queries, text_emb], dim=1)

        # 5. Tạo multimodal causal mask và chuyển sang additive mask
        bool_mask = build_multimodal_causal_mask(Q, T).to(device)  # [Q+T, Q+T]
        # True = attend, False = blocked → đổi về additive (0 / -inf)
        attn_mask = torch.zeros_like(bool_mask, dtype=x.dtype)
        attn_mask[~bool_mask] = float("-inf")

        # 6. Self-attention qua Transformer
        out = self.transformer(x, mask=attn_mask)  # [B, Q+T, d_model]

        # 7. LM head
        logits = self.lm_head(out)  # [B, Q+T, vocab_size]

        return {"logits": logits, "hidden_states": out}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Demo tính toán
# ─────────────────────────────────────────────────────────────────────────────

def demo():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Hyperparameters
    BATCH        = 2
    NUM_QUERIES  = 4
    SEQ_LEN      = 8        # độ dài text (bao gồm [BOS] và [EOS])
    IMG_TOKENS   = 16       # số patch tokens của image encoder
    IMG_FEAT_DIM = 128
    VOCAB_SIZE   = 100
    IGNORE_IDX   = -100
    PAD_TOKEN    = 0

    print("=" * 60)
    print("DEMO: ITG Loss (Image-grounded Text Generation)")
    print("=" * 60)

    # ── Bước 1: Hiển thị Attention Mask ─────────────────────────────────────
    print(f"\n[1] Multimodal Causal Attention Mask (Q={NUM_QUERIES}, T={SEQ_LEN})")
    mask = build_multimodal_causal_mask(NUM_QUERIES, SEQ_LEN)
    total = NUM_QUERIES + SEQ_LEN

    # In đẹp
    header = "    " + " ".join(f"{'Q'+str(i):>2}" for i in range(NUM_QUERIES))
    header += " " + " ".join(f"{'t'+str(i):>2}" for i in range(SEQ_LEN))
    print(header)
    print("    " + "─" * (total * 3))
    for r in range(total):
        row_label = f"Q{r}  " if r < NUM_QUERIES else f"t{r-NUM_QUERIES}  "
        row = " ".join("█" if mask[r, c] else "·" for c in range(total))
        print(f"{row_label}│ {row}")
    print()
    print("  █ = có thể attend (thấy token đó)")
    print("  · = bị mask (không thấy token đó)")
    print()
    print("  Nhận xét:")
    print("  • Queries (Q) ↔ Queries: bidirectional (hàng đầu)")
    print("  • Queries → Text: BLOCKED (queries không thấy text)")
    print("  • Text → Queries: mở hoàn toàn (text thấy tất cả queries)")
    print("  • Text → Text:   causal (chỉ thấy token trước đó)")

    # ── Bước 2: Forward pass qua MiniQFormer ────────────────────────────────
    print(f"\n[2] Forward pass qua MiniQFormer")
    model = MiniQFormer(
        num_queries=NUM_QUERIES, d_model=64, nhead=4,
        vocab_size=VOCAB_SIZE, img_feat_dim=IMG_FEAT_DIM,
    ).to(device)

    img_features  = torch.randn(BATCH, IMG_TOKENS, IMG_FEAT_DIM, device=device)
    # Text: [BOS]=1, nội dung, [EOS]=2, padding dùng IGNORE_IDX
    text_tokens   = torch.tensor([
        [1, 10, 20, 30, 40, 50,  2, PAD_TOKEN],
        [1, 15, 25, 35, 45,  2, PAD_TOKEN, PAD_TOKEN],
    ], device=device)

    # Labels: giống text nhưng padding → IGNORE_IDX
    labels = text_tokens.clone()
    labels[text_tokens == PAD_TOKEN] = IGNORE_IDX

    print(f"  img_features shape : {list(img_features.shape)}")
    print(f"  text_tokens shape  : {list(text_tokens.shape)}")
    print(f"  labels shape       : {list(labels.shape)}")

    outputs = model(img_features, text_tokens)
    logits  = outputs["logits"]
    print(f"  logits shape       : {list(logits.shape)}  (B, Q+T, vocab)")

    # ── Bước 3: Tính ITG Loss ───────────────────────────────────────────────
    print(f"\n[3] Tính ITG Loss")
    loss = compute_itg_loss(logits, labels, NUM_QUERIES, ignore_index=IGNORE_IDX)
    print(f"  ITG Loss           : {loss.item():.4f}")

    # Tính perplexity
    perplexity = math.exp(loss.item())
    print(f"  Perplexity         : {perplexity:.2f}")

    # ── Bước 4: Backward pass (kiểm tra gradient) ───────────────────────────
    print(f"\n[4] Backward pass")
    loss.backward()
    total_params  = sum(p.numel() for p in model.parameters())
    grad_params   = sum(p.numel() for p in model.parameters() if p.grad is not None)
    print(f"  Tổng parameters    : {total_params:,}")
    print(f"  Params có gradient : {grad_params:,}")

    # ── Bước 5: Kiểm tra query_embed gradient ───────────────────────────────
    print(f"\n[5] Query embedding gradient")
    qgrad = model.query_embed.grad
    if qgrad is not None:
        print(f"  query_embed.grad.norm() = {qgrad.norm().item():.6f}")
        print("  ✓ Queries nhận gradient từ ITG loss → học cách trích xuất")
        print("    thông tin ảnh phù hợp với text cần sinh ra.")
    else:
        print("  ✗ Không tìm thấy gradient cho query_embed!")

    # ── Bước 6: So sánh nhanh với/không có image features ───────────────────
    print(f"\n[6] So sánh loss: có vs không có image features")
    model.eval()
    with torch.no_grad():
        # Có image features (giả lập ảnh có nội dung)
        real_img  = torch.randn(BATCH, IMG_TOKENS, IMG_FEAT_DIM, device=device)
        out_real  = model(real_img, text_tokens)
        loss_real = compute_itg_loss(out_real["logits"], labels, NUM_QUERIES, IGNORE_IDX)

        # Không có image features (zeros = ảnh trống/noise)
        zero_img  = torch.zeros(BATCH, IMG_TOKENS, IMG_FEAT_DIM, device=device)
        out_zero  = model(zero_img, text_tokens)
        loss_zero = compute_itg_loss(out_zero["logits"], labels, NUM_QUERIES, IGNORE_IDX)

    print(f"  Loss với image features thật  : {loss_real.item():.4f}")
    print(f"  Loss với image features zeros : {loss_zero.item():.4f}")
    print()
    print("  (Sau khi train, loss_real << loss_zero —")
    print("   model học dùng thông tin ảnh để sinh text tốt hơn.)")

    print("\n" + "=" * 60)
    print("Tóm tắt ITG Loss:")
    print("  loss = CrossEntropy(shift_logits, shift_labels)")
    print("  trong đó:")
    print("    - shift_logits = text logits[:, :-1, :]  (bỏ token cuối)")
    print("    - shift_labels = labels[:, 1:]           (bỏ token đầu)")
    print("    - Chỉ tính trên text tokens (bỏ Q query tokens)")
    print("    - Mask padding bằng ignore_index = -100")
    print("=" * 60)

if __name__ == "__main__":
    demo()