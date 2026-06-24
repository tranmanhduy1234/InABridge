"""
BLIP-2 Architecture Simulation (Corrected)
============================================
Mô phỏng chi tiết cấu trúc BLIP-2 dựa theo Figure 2 của paper và hình diagram:

Kiến trúc Q-Former gồm HAI NHÁNH SONG SONG dùng chung Self-Attention:
  ┌─────────────────────────────────────────────────────────────┐
  │  Nhánh TRÁI  (Image Transformer):                           │
  │    Self-Attention → Cross-Attention → FFN  (×N blocks)      │
  │    Dùng cho: ITM (bidirectional) và ITC (unimodal)          │
  │                                                             │
  │  Nhánh PHẢI  (Text Transformer):                            │
  │    Self-Attention → FFN  (×N blocks, KHÔNG cross-attn)      │
  │    Dùng cho: ITG (multimodal causal)                        │
  └─────────────────────────────────────────────────────────────┘

ITC lấy đầu ra từ CẢ HAI NHÁNH:
  - Nhánh trái  → query representation Z  (B, 32, 768)
  - Nhánh phải  → text [CLS] embedding t  (B, 768)
  → Tính similarity giữa Z (max over queries) và t

3 Objectives + 3 Attention Masks:
  ITM (Image-Text Matching)         → Bidirectional mask   → nhánh TRÁI
  ITC (Image-Text Contrastive)      → Unimodal mask        → nhánh TRÁI (query) + PHẢI (text)
  ITG (Image-Grounded Text Gen)     → Multimodal Causal    → nhánh PHẢI

ViT output shape: [4, 729, 1152]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════
# Hằng số cấu hình (theo paper)
# ══════════════════════════════════════════════════════════════
BATCH_SIZE     = 4
VIT_SEQ_LEN    = 729     # số patch tokens từ ViT-g/14 (e.g. 27×27)
VIT_HIDDEN     = 1152    # chiều hidden của ViT-g/14

NUM_QUERIES    = 32      # số Learned Query tokens
QFORMER_HIDDEN = 768     # chiều hidden Q-Former (= BERTbase hidden)
QFORMER_HEADS  = 12      # số attention heads
QFORMER_LAYERS = 12      # số transformer blocks
FFN_DIM        = 3072    # = 4 × QFORMER_HIDDEN
ITC_PROJ_DIM   = 256     # chiều projection cho ITC loss
DROPOUT        = 0.1

LLM_HIDDEN     = 2048    # chiều embedding đầu vào LLM (OPT-2.7B / FlanT5-XL)
BERT_VOCAB     = 30522   # BERTbase vocab size
LLM_VOCAB      = 50272   # OPT vocab size


# ══════════════════════════════════════════════════════════════
# PHẦN 1: CÁC BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════

class MultiHeadSelfAttention(nn.Module):
    """
    Self-Attention CHIA SẺ giữa image transformer và text transformer.
    Attention mask kiểm soát query ↔ text interaction theo từng objective.
    """
    def __init__(self, hidden=QFORMER_HIDDEN, num_heads=QFORMER_HEADS, dropout=DROPOUT):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = hidden // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj   = nn.Linear(hidden, hidden)
        self.k_proj   = nn.Linear(hidden, hidden)
        self.v_proj   = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x        : (B, L, 768)  — L = Nq hoặc Nq+Nt tùy objective
            attn_mask: (1, 1, L, L) — 0=attend, -inf=blocked
        Returns:
            (B, L, 768)
        """
        B, L, _ = x.shape
        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, L, L)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = self.dropout(F.softmax(scores, dim=-1))

        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_proj(out)


class CrossAttention(nn.Module):
    """
    Cross-Attention: Query tokens → Image features (frozen ViT).
    Chỉ có trong nhánh TRÁI, chèn vào mỗi 2 block.
    Q = query tokens (B, 32, 768)
    K = V = image features (B, 729, 1152) được project về 768
    """
    def __init__(self, q_dim=QFORMER_HIDDEN, kv_dim=VIT_HIDDEN,
                 num_heads=QFORMER_HEADS, dropout=DROPOUT):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = q_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj   = nn.Linear(q_dim,  q_dim)
        self.k_proj   = nn.Linear(kv_dim, q_dim)   # 1152 → 768
        self.v_proj   = nn.Linear(kv_dim, q_dim)   # 1152 → 768
        self.out_proj = nn.Linear(q_dim,  q_dim)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, query_tokens, image_features):
        """
        Args:
            query_tokens  : (B, 32,  768)
            image_features: (B, 729, 1152) — frozen, không gradient
        Returns:
            (B, 32, 768)
        """
        B, Nq, _ = query_tokens.shape
        Ni       = image_features.size(1)

        Q = self.q_proj(query_tokens).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(image_features).view(B, Ni, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(image_features).view(B, Ni, self.num_heads, self.head_dim).transpose(1, 2)

        attn = self.dropout(F.softmax(
            torch.matmul(Q, K.transpose(-2, -1)) * self.scale, dim=-1
        ))  # (B, H, 32, 729)

        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, Nq, -1)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, hidden=QFORMER_HIDDEN, ffn_dim=FFN_DIM, dropout=DROPOUT):
        super().__init__()
        self.fc1     = nn.Linear(hidden, ffn_dim)
        self.fc2     = nn.Linear(ffn_dim, hidden)
        self.act     = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


# ══════════════════════════════════════════════════════════════
# PHẦN 2: HAI LOẠI BLOCK
# ══════════════════════════════════════════════════════════════

class ImageTransformerBlock(nn.Module):
    """
    NHÁNH TRÁI — Image Transformer Block:
      Self-Attention (shared) → [Cross-Attention]* → FFN
    (* Cross-Attention có mặt ở every other block)

    Dùng cho:
      - ITM: bidirectional mask, nhận [query + text]
      - ITC: unimodal mask, nhận [query + text], chỉ lấy query output
    """
    def __init__(self, shared_self_attn: MultiHeadSelfAttention, has_cross_attn: bool):
        super().__init__()
        # Self-attention CHIA SẺ với text transformer block cùng layer
        self.self_attn     = shared_self_attn
        self.norm1         = nn.LayerNorm(QFORMER_HIDDEN)

        self.has_cross_attn = has_cross_attn
        if has_cross_attn:
            self.cross_attn = CrossAttention()
            self.norm_cross = nn.LayerNorm(QFORMER_HIDDEN)

        self.ffn   = FeedForward()
        self.norm2 = nn.LayerNorm(QFORMER_HIDDEN)

    def forward(self, seq, image_features, attn_mask=None):
        """
        Args:
            seq           : (B, Nq+Nt, 768) — [query_tokens; text_tokens]
            image_features: (B, 729, 1152)
            attn_mask     : (1, 1, Nq+Nt, Nq+Nt)
        Returns:
            (B, Nq+Nt, 768)
        """
        # ── Self-Attention (shared, với mask) ──────────────────
        seq = self.norm1(seq + self.self_attn(seq, attn_mask=attn_mask))

        # ── Cross-Attention chỉ cho QUERY PART ─────────────────
        if self.has_cross_attn:
            q   = seq[:, :NUM_QUERIES, :]                      # (B, 32, 768)
            q   = self.norm_cross(q + self.cross_attn(q, image_features))
            seq = torch.cat([q, seq[:, NUM_QUERIES:, :]], dim=1)

        # ── Feed-Forward ────────────────────────────────────────
        seq = self.norm2(seq + self.ffn(seq))
        return seq


class TextTransformerBlock(nn.Module):
    """
    NHÁNH PHẢI — Text Transformer Block:
      Self-Attention (shared) → FFN
    KHÔNG có Cross-Attention.

    Dùng cho:
      - ITG: multimodal causal mask, nhận [query + text]
    """
    def __init__(self, shared_self_attn: MultiHeadSelfAttention):
        super().__init__()
        self.self_attn = shared_self_attn
        self.norm1     = nn.LayerNorm(QFORMER_HIDDEN)
        self.ffn       = FeedForward()
        self.norm2     = nn.LayerNorm(QFORMER_HIDDEN)

    def forward(self, seq, attn_mask=None):
        """
        Args:
            seq      : (B, Nq+Nt, 768)
            attn_mask: (1, 1, Nq+Nt, Nq+Nt)
        Returns:
            (B, Nq+Nt, 768)
        """
        seq = self.norm1(seq + self.self_attn(seq, attn_mask=attn_mask))
        seq = self.norm2(seq + self.ffn(seq))
        return seq


# ══════════════════════════════════════════════════════════════
# PHẦN 3: ATTENTION MASK BUILDERS
# ══════════════════════════════════════════════════════════════

def build_bidirectional_mask(nq: int, nt: int) -> torch.Tensor:
    """
    ITM — Bidirectional: tất cả tokens attend to tất cả tokens.
    ┌──────┬──────┐
    │ Q→Q  │ Q→T  │  tất cả = 0 (unmasked)
    ├──────┼──────┤
    │ T→Q  │ T→T  │
    └──────┴──────┘
    """
    return torch.zeros(nq + nt, nq + nt)


def build_unimodal_mask(nq: int, nt: int) -> torch.Tensor:
    """
    ITC — Unimodal: query attend query, text attend text; cross = BLOCKED.
    ┌──────┬──────┐
    │  0   │ -inf │  Q chỉ attend Q
    ├──────┼──────┤
    │ -inf │  0   │  T chỉ attend T
    └──────┴──────┘
    """
    L    = nq + nt
    mask = torch.full((L, L), float('-inf'))
    mask[:nq, :nq] = 0.0   # Q→Q
    mask[nq:, nq:] = 0.0   # T→T
    return mask


def build_multimodal_causal_mask(nq: int, nt: int) -> torch.Tensor:
    """
    ITG — Multimodal Causal:
    ┌──────┬──────┐
    │  0   │ -inf │  Q attend Q (bidirectional), Q CANNOT attend T
    ├──────┼──────┤
    │  0   │causal│  T attend ALL Q + causal T
    └──────┴──────┘
    """
    L    = nq + nt
    mask = torch.full((L, L), float('-inf'))
    mask[:nq, :nq] = 0.0                                 # Q→Q full
    mask[nq:, :nq] = 0.0                                 # T→Q full
    # T→T causal (lower triangular)
    causal = torch.tril(torch.zeros(nt, nt))
    causal[causal == 0] = float('-inf')
    causal.fill_diagonal_(0.0)
    # tril trả về 0 ở lower-tri và 0 ở diagonal — cần upper-tri = -inf
    causal = torch.full((nt, nt), float('-inf'))
    causal = torch.tril(torch.zeros(nt, nt)).masked_fill(
        torch.tril(torch.ones(nt, nt)) == 0, float('-inf')
    )
    mask[nq:, nq:] = causal
    return mask


def _to_4d(mask: torch.Tensor, device) -> torch.Tensor:
    """(L, L) → (1, 1, L, L) để broadcast với (B, H, L, L)."""
    return mask.to(device).unsqueeze(0).unsqueeze(0)


# ══════════════════════════════════════════════════════════════
# PHẦN 4: Q-FORMER (12 LAYERS, MỖI LAYER CÓ 2 NHÁNH)
# ══════════════════════════════════════════════════════════════

class QFormer(nn.Module):
    """
    Q-Former đúng theo Figure 2 của paper:

    Mỗi layer i có:
      - 1 shared MultiHeadSelfAttention
      - ImageTransformerBlock  (nhánh trái, cross-attn ở block chẵn)
      - TextTransformerBlock   (nhánh phải, không cross-attn)

    ITC  → image_block với unimodal mask  (query part)
         + text_block với unimodal mask   (text [CLS] part)
    ITM  → image_block với bidirectional mask (query + text)
    ITG  → text_block với multimodal causal mask (query + text)
    """
    def __init__(self, num_layers=QFORMER_LAYERS):
        super().__init__()

        # Learned Query tokens: 32 × 768 (model parameters)
        self.query_tokens = nn.Parameter(
            torch.nn.init.normal_(torch.empty(1, NUM_QUERIES, QFORMER_HIDDEN), std=0.02)
        )

        # Text embeddings (BERT-style)
        self.text_embedding  = nn.Embedding(BERT_VOCAB, QFORMER_HIDDEN)
        self.pos_embedding   = nn.Embedding(512, QFORMER_HIDDEN)
        self.embed_dropout   = nn.Dropout(DROPOUT)
        self.embed_norm      = nn.LayerNorm(QFORMER_HIDDEN)

        # 12 layers, mỗi layer: 1 shared SA + 1 image block + 1 text block
        # Cross-attn có ở image block của layer chẵn (0,2,4,6,8,10)
        self.image_blocks = nn.ModuleList()
        self.text_blocks  = nn.ModuleList()
        for i in range(num_layers):
            shared_sa = MultiHeadSelfAttention()
            self.image_blocks.append(
                ImageTransformerBlock(shared_sa, has_cross_attn=(i % 2 == 0))
            )
            self.text_blocks.append(
                TextTransformerBlock(shared_sa)
            )

        # Final LayerNorm
        self.norm = nn.LayerNorm(QFORMER_HIDDEN)

        # ── Projection heads ────────────────────────────────────
        # ITC: project query output và text [CLS] lên ITC_PROJ_DIM=256
        self.itc_query_proj = nn.Linear(QFORMER_HIDDEN, ITC_PROJ_DIM)
        self.itc_text_proj  = nn.Linear(QFORMER_HIDDEN, ITC_PROJ_DIM)
        self.itc_temp       = nn.Parameter(torch.tensor(0.07))  # learnable temperature

        # ITM: binary classifier trên query outputs
        self.itm_head = nn.Linear(QFORMER_HIDDEN, 2)

        # ITG: language model head
        self.itg_lm_head = nn.Linear(QFORMER_HIDDEN, BERT_VOCAB, bias=False)

    # ── Helpers ─────────────────────────────────────────────────

    def _embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids (B, Nt) → (B, Nt, 768)."""
        B, Nt = input_ids.shape
        pos   = torch.arange(Nt, device=input_ids.device).unsqueeze(0)
        emb   = self.text_embedding(input_ids) + self.pos_embedding(pos)
        return self.embed_norm(self.embed_dropout(emb))

    def _run_image_branch(self, seq, image_features, mask_2d):
        """Chạy toàn bộ 12 image (left) blocks."""
        mask = _to_4d(mask_2d, image_features.device)
        for blk in self.image_blocks:
            seq = blk(seq, image_features, attn_mask=mask)
        return self.norm(seq)

    def _run_text_branch(self, seq, image_features, mask_2d):
        """Chạy toàn bộ 12 text (right) blocks."""
        mask = _to_4d(mask_2d, image_features.device)
        for blk in self.text_blocks:
            seq = blk(seq, attn_mask=mask)
        return self.norm(seq)

    # ── Objective Forwards ───────────────────────────────────────

    def forward_itc(self, image_features, input_ids):
        """
        ITC — Image-Text Contrastive Learning
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Lấy đầu ra từ CẢ HAI NHÁNH với Unimodal mask:
          • Nhánh TRÁI  → Z_q : (B, 32, 768) query representations
          • Nhánh PHẢI  → t   : (B, 768)     text [CLS] embedding
        Loss: contrastive similarity giữa max(Z_q · t) và ground truth.

        Args:
            image_features: (B, 729, 1152)
            input_ids     : (B, Nt)
        Returns:
            Z_q_proj: (B, 32, 256) — projected query features
            t_proj  : (B, 256)     — projected text [CLS] feature
        """
        B  = image_features.size(0)
        Nt = input_ids.size(1)

        query    = self.query_tokens.expand(B, -1, -1)  # (B, 32, 768)
        text_emb = self._embed_text(input_ids)           # (B, Nt, 768)
        seq      = torch.cat([query, text_emb], dim=1)   # (B, 32+Nt, 768)

        mask = build_unimodal_mask(NUM_QUERIES, Nt)

        # ── Nhánh TRÁI → query output ──────────────────────────
        img_out = self._run_image_branch(seq, image_features, mask)
        Z_q     = img_out[:, :NUM_QUERIES, :]            # (B, 32, 768)

        # ── Nhánh PHẢI → text [CLS] output ────────────────────
        txt_out = self._run_text_branch(seq, image_features, mask)
        t_cls   = txt_out[:, NUM_QUERIES, :]             # (B, 768) — lấy first text token ([CLS])

        # ── Project lên ITC space ──────────────────────────────
        Z_q_proj = F.normalize(self.itc_query_proj(Z_q), dim=-1)  # (B, 32, 256)
        t_proj   = F.normalize(self.itc_text_proj(t_cls),  dim=-1)  # (B, 256)

        return Z_q_proj, t_proj

    def forward_itm(self, image_features, input_ids):
        """
        ITM — Image-Text Matching
        ━━━━━━━━━━━━━━━━━━━━━━━━━
        CHỈ dùng Nhánh TRÁI với Bidirectional mask.
        Query outputs qua binary classifier → match score.

        Returns:
            itm_logits: (B, 2) — averaged over 32 query outputs
        """
        B  = image_features.size(0)
        Nt = input_ids.size(1)

        query    = self.query_tokens.expand(B, -1, -1)
        text_emb = self._embed_text(input_ids)
        seq      = torch.cat([query, text_emb], dim=1)   # (B, 32+Nt, 768)

        mask    = build_bidirectional_mask(NUM_QUERIES, Nt)
        img_out = self._run_image_branch(seq, image_features, mask)

        Z = img_out[:, :NUM_QUERIES, :]                  # (B, 32, 768)
        logits_per_query = self.itm_head(Z)              # (B, 32, 2)
        return logits_per_query.mean(dim=1)              # (B, 2)

    def forward_itg(self, image_features, input_ids):
        """
        ITG — Image-Grounded Text Generation
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        CHỈ dùng Nhánh PHẢI với Multimodal Causal mask.
        Text tokens predict next token conditioned on query tokens.

        Returns:
            logits: (B, Nt, vocab_size)
        """
        B  = image_features.size(0)
        Nt = input_ids.size(1)

        query    = self.query_tokens.expand(B, -1, -1)
        text_emb = self._embed_text(input_ids)
        seq      = torch.cat([query, text_emb], dim=1)   # (B, 32+Nt, 768)

        mask    = build_multimodal_causal_mask(NUM_QUERIES, Nt)
        txt_out = self._run_text_branch(seq, image_features, mask)

        text_out = txt_out[:, NUM_QUERIES:, :]           # (B, Nt, 768)
        return self.itg_lm_head(text_out)                # (B, Nt, vocab_size)

    def extract_query_features(self, image_features):
        """
        Stage 2: Trích xuất query features để đưa vào LLM.
        Dùng nhánh TRÁI, không có text input (chỉ query tokens).

        Returns:
            Z: (B, 32, 768)
        """
        B     = image_features.size(0)
        query = self.query_tokens.expand(B, -1, -1)      # (B, 32, 768)
        mask  = build_bidirectional_mask(NUM_QUERIES, 0)
        out   = self._run_image_branch(query, image_features, mask)
        return out  # (B, 32, 768)


# ══════════════════════════════════════════════════════════════
# PHẦN 5: STAGE 2 — FC PROJECTION → LLM INPUT
# ══════════════════════════════════════════════════════════════

class QFormerToLLMProjection(nn.Module):
    """
    Linear projection: Q-Former output (768) → LLM embedding dim (2048).
    Projected query embeddings được prepend vào text embeddings
    như "soft visual prompts" cho frozen LLM.
    """
    def __init__(self, q_dim=QFORMER_HIDDEN, llm_dim=LLM_HIDDEN):
        super().__init__()
        self.fc = nn.Linear(q_dim, llm_dim)

    def forward(self, Z):
        """Z: (B, 32, 768) → (B, 32, llm_dim)"""
        return self.fc(Z)


# ══════════════════════════════════════════════════════════════
# PHẦN 6: BLIP-2 FULL PIPELINE
# ══════════════════════════════════════════════════════════════

class BLIP2(nn.Module):
    """
    BLIP-2 Pipeline:
      Stage 1 → Train Q-Former với frozen ViT (3 objectives)
      Stage 2 → Q-Former + FC → frozen LLM input
    """
    def __init__(self):
        super().__init__()
        self.qformer    = QFormer()
        self.projection = QFormerToLLMProjection()

        # LLM text embedding (giả lập OPT vocab)
        self.llm_text_emb = nn.Embedding(LLM_VOCAB, LLM_HIDDEN)

    def stage1_forward(self, image_features, input_ids):
        """
        Returns dict chứa outputs của cả 3 objectives.
        """
        z_q_proj, t_proj = self.qformer.forward_itc(image_features, input_ids)
        itg_logits        = self.qformer.forward_itg(image_features, input_ids)
        itm_logits        = self.qformer.forward_itm(image_features, input_ids)
        return {
            "itc_query_proj" : z_q_proj,    # (B, 32, 256) — image side
            "itc_text_proj"  : t_proj,       # (B, 256)     — text side
            "itg_logits"     : itg_logits,   # (B, Nt, vocab)
            "itm_logits"     : itm_logits,   # (B, 2)
        }

    def stage2_forward(self, image_features, llm_input_ids=None):
        """
        Tạo input cho frozen LLM.
        Returns:
            llm_input: (B, 32[+Nt], llm_dim) — visual prompts [+ text]
        """
        Z              = self.qformer.extract_query_features(image_features)  # (B, 32, 768)
        visual_prompts = self.projection(Z)                                    # (B, 32, 2048)

        if llm_input_ids is not None:
            text_emb  = self.llm_text_emb(llm_input_ids)               # (B, Nt, 2048)
            llm_input = torch.cat([visual_prompts, text_emb], dim=1)    # (B, 32+Nt, 2048)
        else:
            llm_input = visual_prompts

        return llm_input


# ══════════════════════════════════════════════════════════════
# PHẦN 7: DEMO / SANITY CHECK
# ══════════════════════════════════════════════════════════════

def ps(name, tensor):  # print shape helper
    print(f"    {name:<42s}: {tuple(tensor.shape)}")


def verify_masks(nq, nt):
    """Xác nhận các mask đúng theo paper."""
    bi  = build_bidirectional_mask(nq, nt)
    uni = build_unimodal_mask(nq, nt)
    cau = build_multimodal_causal_mask(nq, nt)

    # Bidirectional: tất cả = 0
    assert (bi == 0).all(), "Bidirectional: phải toàn 0"

    # Unimodal: cross block = -inf
    assert torch.isinf(uni[:nq, nq:]).all(), "Unimodal: Q→T phải -inf"
    assert torch.isinf(uni[nq:, :nq]).all(), "Unimodal: T→Q phải -inf"
    assert (uni[:nq, :nq] == 0).all(),       "Unimodal: Q→Q phải 0"
    assert (uni[nq:, nq:] == 0).all(),       "Unimodal: T→T phải 0"

    # Multimodal Causal: Q→T blocked, T→Q open, T→T causal
    assert torch.isinf(cau[:nq, nq:]).all(),  "Causal: Q→T phải -inf"
    assert (cau[nq:, :nq] == 0).all(),         "Causal: T→Q phải 0"
    assert (cau[nq, nq] == 0).all(),           "Causal: T[0]→T[0] phải 0 (diagonal)"
    assert torch.isinf(cau[nq, nq+1]).all(),   "Causal: T[0]→T[1] phải -inf"

    return bi, uni, cau


def main():
    SEP = "─" * 65
    print("=" * 65)
    print("   BLIP-2 Architecture Simulation  (Corrected)")
    print("=" * 65)

    torch.manual_seed(42)
    B = BATCH_SIZE

    # ── Inputs ───────────────────────────────────────────────────
    image_features = torch.randn(B, VIT_SEQ_LEN, VIT_HIDDEN)   # (4, 729, 1152)
    input_ids      = torch.randint(0, BERT_VOCAB, (B, 20))      # (4, 20)
    llm_input_ids  = torch.randint(0, LLM_VOCAB,  (B, 20))      # (4, 20)

    print(f"\n{SEP}\n  INPUTS\n{SEP}")
    ps("ViT output (frozen)", image_features)
    ps("Text input_ids (BERT vocab)", input_ids)

    # ── Model & params ───────────────────────────────────────────
    model  = BLIP2()
    total  = sum(p.numel() for p in model.parameters())
    print(f"\n{SEP}\n  MODEL PARAMETERS\n{SEP}")
    print(f"    {'Q-Former + Projection (trainable)':<42s}: {total:,}")
    print(f"    {'Frozen ViT & LLM':<42s}: not counted")

    # ── Attention Masks ──────────────────────────────────────────
    Nq, Nt = NUM_QUERIES, input_ids.size(1)
    print(f"\n{SEP}\n  ATTENTION MASKS  [Nq={Nq}, Nt={Nt}]\n{SEP}")
    bi, uni, cau = verify_masks(Nq, Nt)
    ps("ITM  Bidirectional mask  (Nq+Nt, Nq+Nt)", bi)
    ps("ITC  Unimodal mask       (Nq+Nt, Nq+Nt)", uni)
    ps("ITG  Multimodal Causal   (Nq+Nt, Nq+Nt)", cau)
    print("    ✓ Tất cả masks đúng theo paper")

    # ── Stage 1 Forward ─────────────────────────────────────────
    print(f"\n{SEP}\n  STAGE 1 — Ba Objectives Song Song\n{SEP}")
    with torch.no_grad():
        s1 = model.stage1_forward(image_features, input_ids)

    print("\n  [ITC] Image-Text Contrastive  — CẢ 2 NHÁNH + Unimodal mask:")
    ps("    Nhánh TRÁI → query proj Z_q", s1["itc_query_proj"])
    ps("    Nhánh PHẢI → text [CLS] t  ", s1["itc_text_proj"])
    print(f"    → Compute pairwise similarity: max(Z_q·t) per sample")

    print("\n  [ITM] Image-Text Matching     — Nhánh TRÁI + Bidirectional mask:")
    ps("    → Match logits (B, 2)", s1["itm_logits"])
    print(f"    → Averaged over {NUM_QUERIES} query outputs → BCE loss")

    print("\n  [ITG] Image-Grounded Text Gen — Nhánh PHẢI + Multimodal Causal:")
    ps("    → LM logits (B, Nt, vocab)", s1["itg_logits"])
    print(f"    → Language modeling loss trên text tokens")

    # ── Stage 2 Forward ─────────────────────────────────────────
    print(f"\n{SEP}\n  STAGE 2 — Q-Former → FC → LLM Input\n{SEP}")
    with torch.no_grad():
        Z              = model.qformer.extract_query_features(image_features)
        visual_prompts = model.projection(Z)
        llm_input      = model.stage2_forward(image_features, llm_input_ids)

    ps("Q-Former output Z", Z)
    ps("FC Projection (soft visual prompts)", visual_prompts)
    ps("LLM Input [visual_prompts ++ text]", llm_input)

    # ── Data Flow Summary ────────────────────────────────────────
    print(f"\n{SEP}\n  DATA FLOW SUMMARY\n{SEP}")
    rows = [
        ("Input Image",                      f"→ frozen ViT"),
        ("ViT output (frozen)",               f"(B=4,  729, 1152)"),
        ("",                                  ""),
        ("══ STAGE 1: Q-Former Training ══",  ""),
        ("Learned Query tokens",              f"(B=4,   32,  768)"),
        ("+ Text embeddings (Nt=20)",          f"(B=4,   20,  768)"),
        ("→ concat seq",                      f"(B=4,   52,  768)"),
        ("",                                  ""),
        ("[ITC] Left branch, unimodal mask",  ""),
        ("   query output Z_q",               f"(B=4,   32,  768)  → proj → (B, 32, 256)"),
        ("[ITC] Right branch, unimodal mask", ""),
        ("   text [CLS] output t",            f"(B=4,        768)  → proj → (B, 256)"),
        ("[ITM] Left branch, bidir mask",     f"→ logits (B=4, 2)"),
        ("[ITG] Right branch, causal mask",   f"→ logits (B=4, 20, {BERT_VOCAB})"),
        ("",                                  ""),
        ("══ STAGE 2: Connect to LLM ══",     ""),
        ("Q-Former query output",             f"(B=4,   32,  768)"),
        ("FC Projection (768→2048)",           f"(B=4,   32, 2048)  ← soft visual prompts"),
        ("+ LLM text embedding (Nt=20)",       f"(B=4,   20, 2048)"),
        ("→ LLM Input (prepend visual)",       f"(B=4,   52, 2048)"),
        ("→ Frozen LLM (OPT / FlanT5)",        "→ Output Text"),
    ]
    for name, shape in rows:
        if name == "":
            print()
        elif shape == "":
            print(f"  {name}")
        else:
            print(f"    {name:<42s} {shape}")

    print(f"\n{SEP}")
    print("  ✓ Simulation hoàn thành!")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()