import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_mask_qformer(attn_mask, num_queries: int, objective: str):
    objective = objective.lower()
    if objective not in {"itc", "itm", "itg"}:
        raise ValueError("Objective must be one of 'itc', 'itm', 'itg'")

    attn_mask = attn_mask.bool()
    B, T = attn_mask.shape
    L, device = num_queries + T, attn_mask.device

    if objective == "itm":
        structural = torch.ones(L, L, device=device, dtype=torch.bool)
    elif objective == "itc":
        structural = torch.zeros(L, L, device=device, dtype=torch.bool)
        structural[:num_queries, :num_queries] = True
        structural[num_queries:, num_queries:] = True
    elif objective == "itg":
        structural = torch.zeros(L, L, dtype=torch.bool, device=device)
        structural[:num_queries, :num_queries] = True
        structural[num_queries:, :num_queries] = True
        structural[num_queries:, num_queries:] = torch.ones(T, T, dtype=torch.bool, device=device).tril()

    valid = torch.cat((torch.ones(B, num_queries, dtype=torch.bool, device=device), attn_mask), dim=1)
    mask = structural[None] & valid[:, None, :]

    return mask[:, None]

def to_additive_mask(mask, dtype):
    return torch.zeros_like(mask, dtype=dtype).masked_fill(~mask, torch.finfo(dtype).min)

class CrossAttention(nn.Module):
    def __init__(self, hidden, image_dim, heads, dropout, eps, init_std):
        super().__init__()
        if hidden % heads:
            raise ValueError("Hidden dimension must be divisible by number of heads")

        self.heads, self.head_dim = heads, hidden // heads
        self.Wq = nn.Linear(hidden, hidden)
        self.Wk = nn.Linear(image_dim, hidden)
        self.Wv = nn.Linear(image_dim, hidden)
        self.Wo = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden, eps=eps)
        self.dropout = nn.Dropout(dropout)
        self.apply(lambda m: self._init(m, init_std))

    @staticmethod
    def _init(m, std):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _split(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, queries, image):
        q = self._split(self.Wq(queries)).contiguous()
        k = self._split(self.Wk(image)).contiguous()
        v = self._split(self.Wv(image)).contiguous()

        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout.p if self.training else 0.0, is_causal=False
        )
        x = x.transpose(1, 2).reshape_as(queries)
        return self.norm(queries + self.dropout(self.Wo(x)))

class QFormerBlock(nn.Module):
    def __init__(self, bert_layer, image_dim, use_cross, cfg):
        super().__init__()
        self.self_attn = bert_layer.attention
        self.text_intermediate = bert_layer.intermediate
        self.text_output = bert_layer.output
        self.query_intermediate = copy.deepcopy(bert_layer.intermediate)
        self.query_output = copy.deepcopy(bert_layer.output)

        self.cross = CrossAttention(
            cfg.hidden_size, image_dim, cfg.num_attention_heads,
            cfg.attention_probs_dropout_prob, cfg.layer_norm_eps, cfg.initializer_range
        ) if use_cross else None

    def forward(self, x, mask, image, num_queries):
        x = self.self_attn(x, attention_mask=mask)[0]
        q, text = x[:, :num_queries], x[:, num_queries:]
        if self.cross is not None:
            q = self.cross(q, image)
        q = self.query_output(self.query_intermediate(q), q)
        if text.size(1):
            text = self.text_output(self.text_intermediate(text), text)
        return torch.cat((q, text), dim=1)

class QFormer(nn.Module):
    def __init__(self, bert, num_queries, image_dim, cross_attn_every, hidden_dim):
        super().__init__()
        self.cfg = bert.config
        if hidden_dim != self.cfg.hidden_size:
            raise ValueError("hidden_dim must match BERT hidden_size")

        self.image_dim = image_dim
        self.num_queries = num_queries
        self.cross_attn_every = cross_attn_every
        self.hidden_dim = hidden_dim

        self.query_tokens = nn.Parameter(torch.empty(1, self.num_queries, self.hidden_dim))
        self.layers = nn.ModuleList([
            QFormerBlock(layer, image_dim, i % cross_attn_every == 0, self.cfg)
            for i, layer in enumerate(bert.encoder.layer)
        ])

        nn.init.normal_(self.query_tokens, std=self.cfg.initializer_range)

    def forward(self, image_features, text_embedding, attn_mask, objective):
        B = image_features.size(0)
        q = self.query_tokens.expand(B, -1, -1)
        if text_embedding is None:
            if attn_mask is not None:
                raise ValueError("attn_mask requires text_embedding")
            attn_mask = torch.empty(B, 0, dtype=torch.bool, device=image_features.device)
            x = q
        else:
            if text_embedding.ndim != 3 or text_embedding.size(-1) != self.hidden_dim:
                raise ValueError("text_embedding must have shape [batch, length, hidden_dim]")
            if text_embedding.size(0) != B:
                raise ValueError("batch size mismatch")
            if attn_mask is None:
                attn_mask = torch.ones(text_embedding.shape[:2], dtype=torch.bool,
                                       device=text_embedding.device)
            elif attn_mask.shape != text_embedding.shape[:2]:
                raise ValueError("attn mask shape mismatch")
            x = torch.cat((q, text_embedding), dim=1)
        mask = generate_mask_qformer(
            attn_mask, self.num_queries, objective
        )
        mask = to_additive_mask(mask, x.dtype)
        for layer in self.layers:
            x = layer(x, mask, image_features, self.num_queries)
        return {
            "query_output": x[:, :self.num_queries],
            "text_output": x[:, self.num_queries:],
            "hidden_states": x
        }

def main():
    from transformers import AutoTokenizer, BertModel
    from src.utils.seed import seed_everything
    seed_everything()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    bert = BertModel.from_pretrained("bert-base-uncased", add_pooling_layer=False)
    embeddings = bert.embeddings.requires_grad_(False).to(device).eval()
    model = QFormer(
        bert=bert,
        num_queries=128,
        image_dim=1024,
        cross_attn_every=2,
        hidden_dim=768,
    ).to(device)
    texts = [
        "A dog is running on the grass.",
        "A car is parked beside the road.",
        "Two people are sitting at a table.",
    ]
    tokens = tokenizer(
        texts,
        padding=True, truncation=True, max_length=128,
        return_tensors="pt"
    ).to(device)
    B = len(texts)
    image = torch.randn(B, 256, 1024, device=device)
    print("device:", device)
    print("input_ids:", tuple(tokens.input_ids.shape))
    print("image:", tuple(image.shape))
    print()
    model.eval()
    with torch.no_grad():
        text_embeddings = embeddings(input_ids=tokens.input_ids)
        for objective in ("itc", "itm", "itg"):
            out = model(
                image,
                text_embeddings,
                attn_mask=tokens.attention_mask.bool(),
                objective=objective
            )
            print(f"[{objective.upper()}]")
            print("query :", tuple(out["query_output"].shape))
            print("text  :", tuple(out["text_output"].shape))
            print("hidden:", tuple(out["hidden_states"].shape))
            print()
    print(
        "params:",
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
if __name__ == "__main__":
    main()
