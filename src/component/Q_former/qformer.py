import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ══════════════════════════════════════════════════════════════
#  Cấu hình (trung thực với checkpoints thực tế)
# ══════════════════════════════════════════════════════════════
DEBERTA_CONFIG = dict(
    hidden_size              = 768,
    num_hidden_layers        = 12,
    num_attention_heads      = 12,
    intermediate_size        = 3072,
    hidden_dropout_prob      = 0.1,
    attention_probs_dropout  = 0.1,
    layer_norm_eps           = 1e-7,
    max_relative_positions   = 512,
    vocab_size               = 128_100,  # DeBERTa-v3-base
)

QWEN_CONFIG = dict(
    vocab_size       = 152_064,  # Qwen2.5-7B-Instruct
    hidden_size      = 3_584,
    num_hidden_layers= 28,
    num_attention_heads = 28,
    intermediate_size   = 18_944,
)

# ══════════════════════════════════════════════════════════════
#  1.  DeBERTa Disentangled Self-Attention
#      (content-to-position + position-to-content bias)
# ══════════════════════════════════════════════════════════════
class DisentangledSelfAttn(nn.Module):
    """
    DeBERTa-v3 disentangled attention:
      score = (c2c) + (c2p) + (p2c)
    Relative position được biểu diễn riêng biệt với nội dung token.
    """

    def __init__(self, hidden: int, num_heads: int,
                 dropout: float = 0.1, max_rel: int = 512):
        super().__init__()
        assert hidden % num_heads == 0
        self.nh    = num_heads
        self.hd    = hidden // num_heads
        self.scale = math.sqrt(self.hd)
        self.max_rel = max_rel

        # Content projections
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.o_proj = nn.Linear(hidden, hidden)

        # Relative position embeddings (disentangled)
        self.pos_emb = nn.Embedding(2 * max_rel, hidden)
        self.pos_q   = nn.Linear(hidden, hidden)   # position → query side
        self.pos_k   = nn.Linear(hidden, hidden)   # position → key   side

        self.attn_drop = nn.Dropout(dropout)

    def _rel_idx(self, L: int, device) -> torch.Tensor:
        """Tính chỉ số vị trí tương đối i→j, shape (L, L)."""
        i = torch.arange(L, device=device).unsqueeze(1)
        j = torch.arange(L, device=device).unsqueeze(0)
        rel = (j - i).clamp(-self.max_rel + 1, self.max_rel - 1)
        return rel + self.max_rel  # shift về [0, 2*max_rel)

    def forward(self, x: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L, H = x.shape
        nh, hd  = self.nh, self.hd

        def split_heads(t):
            return t.view(B, L, nh, hd).transpose(1, 2)  # (B, nh, L, hd)

        q = split_heads(self.q_proj(x))
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))

        # ── Content-to-content ────────────────────────────────────
        c2c = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B,nh,L,L)

        # ── Relative position embeddings ──────────────────────────
        ridx = self._rel_idx(L, x.device)           # (L, L)
        pe   = self.pos_emb(ridx)                    # (L, L, H)

        pk = self.pos_k(pe).view(L, L, nh, hd)      # (L, L, nh, hd)
        pq = self.pos_q(pe).view(L, L, nh, hd)

        q_h  = q.permute(1, 0, 2, 3)                # (nh, B, L, hd)
        pk_h = pk.permute(2, 0, 1, 3)               # (nh, L, L, hd)
        pq_h = pq.permute(2, 0, 1, 3)
        k_h  = k.permute(1, 0, 2, 3)

        # ── Content-to-position: query attends to relative pos ─────
        c2p = torch.einsum('hblx,hljx->hblj', q_h, pk_h
                           ).permute(1, 0, 2, 3) / self.scale

        # ── Position-to-content: rel-pos attends to keys ──────────
        p2c = torch.einsum('hblx,hljx->hblj', k_h, pq_h
                           ).permute(1, 0, 2, 3).transpose(-2, -1) / self.scale

        attn = c2c + c2p + p2c

        if attention_mask is not None:
            # mask: (B, L) → (B, 1, 1, L) — che padding tokens
            bias = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * -1e4
            attn = attn + bias

        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out  = torch.matmul(attn, v)                 # (B, nh, L, hd)
        out  = out.transpose(1, 2).contiguous().view(B, L, H)
        return self.o_proj(out)


# ══════════════════════════════════════════════════════════════
#  2.  DeBERTa Transformer Block  (Attn + LayerNorm + FFN)
# ══════════════════════════════════════════════════════════════
class DebertaLayer(nn.Module):
    """Pre-norm DeBERTa block với GELU activation."""

    def __init__(self, cfg: dict):
        super().__init__()
        H    = cfg['hidden_size']
        mid  = cfg['intermediate_size']
        drop = cfg['hidden_dropout_prob']
        eps  = cfg['layer_norm_eps']

        self.attn  = DisentangledSelfAttn(
            H, cfg['num_attention_heads'], drop,
            cfg['max_relative_positions'])
        self.norm1 = nn.LayerNorm(H, eps=eps)
        self.fc1   = nn.Linear(H, mid)
        self.fc2   = nn.Linear(mid, H)
        self.norm2 = nn.LayerNorm(H, eps=eps)
        self.drop  = nn.Dropout(drop)
        self.act   = nn.GELU()

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        # Self-attention sub-layer (post-norm)
        x = self.norm1(x + self.drop(self.attn(x, mask)))
        # FFN sub-layer
        ff = self.fc2(self.drop(self.act(self.fc1(x))))
        return self.norm2(x + self.drop(ff))


# ══════════════════════════════════════════════════════════════
#  3.  Cross-Attention  (learnable queries → image patches)
#      Được chèn vào mỗi `cross_every` DeBERTa layers
# ══════════════════════════════════════════════════════════════
class CrossAttnLayer(nn.Module):
    """
    Queries attend to frozen image encoder features.
    Image features được project từ image_dim → hidden_dim trước.
    """

    def __init__(self, hidden: int, image_dim: int,
                 num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.img_proj   = nn.Linear(image_dim, hidden)
        self.cross_attn = nn.MultiheadAttention(
            hidden, num_heads, dropout=dropout, batch_first=True)
        self.norm       = nn.LayerNorm(hidden)
        self.drop       = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor,
                image_features: torch.Tensor) -> torch.Tensor:
        """
        queries        : (B, Q, hidden)
        image_features : (B, P, image_dim)
        """
        img = self.img_proj(image_features)           # (B, P, hidden)
        out, _ = self.cross_attn(queries, img, img)   # (B, Q, hidden)
        return self.norm(queries + self.drop(out))


# ══════════════════════════════════════════════════════════════
#  4.  Qwen2.5-7B Word Embedding  (frozen)
#      vocab = 152,064  |  dim = 3,584
# ══════════════════════════════════════════════════════════════
class QwenEmbedding(nn.Module):
    """
    Embedding table với kích thước THỰC của Qwen2.5-7B-Instruct.
    Toàn bộ trọng số bị đóng băng (frozen) trong Q-Former.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embed      = nn.Embedding(vocab_size, embed_dim)
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim
        nn.init.normal_(self.embed.weight, std=0.02)
        # Đóng băng toàn bộ
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)


# ══════════════════════════════════════════════════════════════
#  5.  Q-Former  (BLIP-2 Core Module)
# ══════════════════════════════════════════════════════════════
class QFormer(nn.Module):
    """
    Lightweight Querying Transformer — cầu nối giữa frozen image encoder
    và frozen LLM theo thiết kế BLIP-2.

    Kiến trúc
    ──────────
    ┌─────────────────────────────────────────────────────────┐
    │  Image features (B, 257, 1024) ← ViT-L/14              │
    │         ↓  [CrossAttnLayer × 6]                         │
    │  Learnable Queries (B, 32, 768)                         │
    │  ←→ DeBERTa-v3-base 12 layers (self-attn + disentangled)│
    │  ←→ Text tokens via shared self-attention (optional)    │
    │         ↓  [out_proj: 768 → 3584]                       │
    │  Soft Visual Prompts (B, 32, 3584) → frozen LLM        │
    └─────────────────────────────────────────────────────────┘

    Nguồn weights
    ─────────────
    • DeBERTa backbone : microsoft/deberta-v3-base
    • Qwen embedding   : Qwen/Qwen2.5-7B-Instruct (frozen)
    """

    def __init__(
        self,
        num_queries      : int = 32,
        image_hidden_dim : int = 1024,   # ViT-L/14 output dim
        cross_attn_every : int = 2,      # chèn cross-attn mỗi N DeBERTa layers
    ):
        super().__init__()
        cfg  = DEBERTA_CONFIG
        qcfg = QWEN_CONFIG

        H = cfg['hidden_size']    # 768
        E = qcfg['hidden_size']   # 3584

        self.num_queries      = num_queries
        self.image_hidden_dim = image_hidden_dim
        self.hidden_dim       = H
        self.qwen_dim         = E
        self.num_layers       = cfg['num_hidden_layers']
        self.cross_attn_every = cross_attn_every

        # ── Learnable query vectors ────────────────────────────────
        self.queries  = nn.Parameter(torch.zeros(1, num_queries, H))
        nn.init.normal_(self.queries, std=0.02)
        self.q_norm   = nn.LayerNorm(H)

        # ── DeBERTa-v3-base transformer layers ─────────────────────
        self.deb_layers = nn.ModuleList(
            [DebertaLayer(cfg) for _ in range(self.num_layers)])

        # ── Cross-attention: queries ↔ image features ─────────────
        n_cross = self.num_layers // cross_attn_every
        self.cross_layers = nn.ModuleList([
            CrossAttnLayer(H, image_hidden_dim, cfg['num_attention_heads'])
            for _ in range(n_cross)
        ])

        # ── Qwen2.5-7B embedding table (frozen) ───────────────────
        self.tok_emb  = QwenEmbedding(qcfg['vocab_size'], E)
        # Project Qwen dim → DeBERTa hidden dim
        self.tok_proj = nn.Linear(E, H)

        # ── Output FC: DeBERTa hidden → Qwen dim (soft prompts) ───
        self.out_proj = nn.Linear(H, E)

    # ─── Forward pass ─────────────────────────────────────────────
    def forward(
        self,
        image_features : torch.Tensor,               # (B, P, 1024)
        input_ids      : torch.Tensor | None = None, # (B, L)  optional text
    ) -> dict[str, torch.Tensor]:
        """
        Trả về
        -------
        query_output  : (B, 32, 3584)  – soft visual prompts cho LLM
        query_hidden  : (B, 32, 768)   – biểu diễn trung gian
        """
        B = image_features.size(0)

        # Khởi tạo queries cho batch
        q = self.q_norm(self.queries.expand(B, -1, -1))  # (B, Q, H)

        # Ghép queries với text tokens (nếu có)
        if input_ids is not None:
            tok_emb = self.tok_proj(self.tok_emb(input_ids))  # (B, L, H)
            hidden  = torch.cat([q, tok_emb], dim=1)           # (B, Q+L, H)
        else:
            hidden = q                                          # (B, Q, H)

        attn_mask = hidden.new_ones(B, hidden.size(1))  # attend to all
        ci = 0

        for i, layer in enumerate(self.deb_layers):
            hidden = layer(hidden, attn_mask)

            # Chèn cross-attention mỗi `cross_attn_every` blocks
            if (i + 1) % self.cross_attn_every == 0:
                q_h    = hidden[:, :self.num_queries, :]
                q_h    = self.cross_layers[ci](q_h, image_features)
                hidden = torch.cat([q_h, hidden[:, self.num_queries:, :]], dim=1)
                ci    += 1

        query_hidden = hidden[:, :self.num_queries, :]   # (B, Q, H)
        query_output = self.out_proj(query_hidden)        # (B, Q, E)

        return {"query_output": query_output,
                "query_hidden": query_hidden}

    # ─── Tiện ích ─────────────────────────────────────────────────
    def count_params(self, trainable: bool = True) -> int:
        return sum(p.numel() for p in self.parameters()
                   if (p.requires_grad if trainable else True))

    def print_summary(self):
        T  = self.count_params(False)
        tr = self.count_params(True)
        fr = T - tr
        W  = 64

        def fmt(n): return f"{n:>14,}  ({n/1e6:7.2f} M)"

        print(f"\n{'═'*W}")
        print("  Q-Former — Model Summary  (BLIP-2 Architecture)")
        print(f"{'═'*W}")

        rows = [
            ("Nguồn backbone weights",  "microsoft/deberta-v3-base"),
            ("Nguồn tokenizer/embed",   "Qwen/Qwen2.5-7B-Instruct"),
            ("Hidden dim (DeBERTa)",    str(self.hidden_dim)),
            ("Số DeBERTa layers",       str(self.num_layers)),
            ("Số attention heads",      str(DEBERTA_CONFIG['num_attention_heads'])),
            ("FFN intermediate size",   str(DEBERTA_CONFIG['intermediate_size'])),
            ("Số learnable queries",    str(self.num_queries)),
            ("Image encoder dim",       f"{self.image_hidden_dim}  (ViT-L/14)"),
            ("Số cross-attn layers",    f"{len(self.cross_layers)}  "
                                        f"(mỗi {self.cross_attn_every} blocks)"),
            ("Vocab size (Qwen2.5)",    f"{QWEN_CONFIG['vocab_size']:,}"),
            ("Qwen embed dim",          str(self.qwen_dim)),
            ("Output shape → LLM",      f"(B, {self.num_queries}, {self.qwen_dim})"),
        ]
        for k, v in rows:
            print(f"  {k:<32}: {v}")

        print(f"{'─'*W}")
        print(f"  {'Tổng tham số':<32}: {fmt(T)}")
        print(f"  {'Trainable':<32}: {fmt(tr)}")
        print(f"  {'Frozen (Qwen embed)':<32}: {fmt(fr)}")
        print(f"{'─'*W}")

        mods = {
            "DeBERTa backbone (12 layers)" : self.deb_layers,
            "Cross-attn layers (6×)"       : self.cross_layers,
            "Learnable queries"            : self.queries,
            "Qwen embedding [frozen]"      : self.tok_emb,
            "Token projection (E→H)"       : self.tok_proj,
            "Output projection (H→E)"      : self.out_proj,
        }
        for nm, m in mods.items():
            n  = (m.numel() if isinstance(m, nn.Parameter)
                  else sum(p.numel() for p in m.parameters()))
            rg = (m.requires_grad if isinstance(m, nn.Parameter)
                  else any(p.requires_grad for p in m.parameters()))
            print(f"  {nm:<38}: {n:>11,}  ({n/1e6:.2f}M)"
                  f"  [{'train ✓' if rg else 'frozen ✗'}]")
        print(f"{'═'*W}\n")


# ══════════════════════════════════════════════════════════════
#  Demo
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    print("\n" + "━"*64)
    print("  BLIP-2  Q-Former  Demo")
    print("━"*64)
    print(f"  Device : {DEVICE}")

    # ── 1. Khởi tạo ─────────────────────────────────────────────
    print("\n[1] Khởi tạo Q-Former …")
    model = QFormer(
        num_queries=32,
        image_hidden_dim=1024,
        cross_attn_every=2,
    ).to(DEVICE)
    model.print_summary()

    # ── 2. Dữ liệu giả lập ──────────────────────────────────────
    B, P, L = 2, 257, 12       # batch=2, patches=257, seq_len=12
    img = torch.randn(B, P, 1024, device=DEVICE)
    ids = torch.randint(0, QWEN_CONFIG['vocab_size'], (B, L), device=DEVICE)

    print("[2] Dữ liệu đầu vào (giả lập ViT-L/14 + Qwen tokenizer)")
    print(f"    image_features : {tuple(img.shape)}"
          f"  ← (Batch, Patches=257, ImageDim=1024)")
    print(f"    input_ids      : {tuple(ids.shape)}"
          f"       ← (Batch, SeqLen)")

    # ── 3. Forward: image + text ─────────────────────────────────
    print("\n[3] Forward pass — chế độ image + text conditioning …")
    with torch.no_grad():
        out = model(img, ids)
    print(f"    query_output  (→ LLM soft prompts) : {tuple(out['query_output'].shape)}")
    print(f"    query_hidden  (trung gian)          : {tuple(out['query_hidden'].shape)}")

    # ── 4. Forward: image only ────────────────────────────────────
    print("\n[4] Forward pass — chế độ image-only …")
    with torch.no_grad():
        out2 = model(img)
    print(f"    query_output  (no text) : {tuple(out2['query_output'].shape)}")

    # ── 5. Bottleneck ─────────────────────────────────────────────
    img_tok = P * 1024
    q_tok   = 32 * model.hidden_dim
    print(f"\n[5] Thông tin nén (Bottleneck)")
    print(f"    Image features (raw)  : {img_tok:>9,}  ({P} patches × 1024 dim)")
    print(f"    Q-Former query output : {q_tok:>9,}  (32 queries × {model.hidden_dim} dim)")
    print(f"    Tỉ lệ nén             : {img_tok/q_tok:.1f}×")

    print("\n" + "━"*64)
    print("  Demo hoàn tất ✓")
    print("━"*64 + "\n")