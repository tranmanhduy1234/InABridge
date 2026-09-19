import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from src_src import config_model as cfg

def generate_mask_qformer(padding_mask, queries_num=cfg.NUM_QUERIES,
                          objective=cfg.QFORMER_DEFAULT_OBJECTIVE,
                          padding_only_key=cfg.QFORMER_PADDING_ONLY_KEY):
    objective = objective.lower()
    if objective not in {"itc", "itm", "itg"}:
        raise ValueError("objective must be itc, itm or itg")

    padding_mask = padding_mask.bool()
    B, T = padding_mask.shape
    L, device = queries_num + T, padding_mask.device

    if objective == "itm":
        structural = torch.ones(L, L, dtype=torch.bool, device=device)
    elif objective == "itc":
        structural = torch.zeros(L, L, dtype=torch.bool, device=device)
        structural[:queries_num, :queries_num] = True
        structural[queries_num:, queries_num:] = True
    else:
        structural = torch.zeros(L, L, dtype=torch.bool, device=device)
        structural[:queries_num, :queries_num] = True
        structural[queries_num:, :queries_num] = True
        structural[queries_num:, queries_num:] = torch.ones(T, T, dtype=torch.bool, device=device).tril()

    valid = torch.cat((torch.ones(B, queries_num, dtype=torch.bool, device=device), ~padding_mask), dim=1)
    mask = structural[None] & valid[:, None, :]

    if not padding_only_key:
        mask &= valid[:, :, None]
        b, q = (~valid).nonzero(as_tuple=True)
        mask[b, q, q] = True

    return mask[:, None]

def to_additive_mask(mask, dtype):
    return torch.zeros_like(mask, dtype=dtype).masked_fill(~mask, torch.finfo(dtype).min)

class CrossAttention(nn.Module):
    def __init__(self, hidden, image_dim, heads, dropout, eps, init_std):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")

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
        return x.view(B, L, self.heads, self.head_dim).transpose(1, 2)

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
    def __init__(self, bert, num_queries=cfg.NUM_QUERIES,
                 *, image_dim, cross_attn_every=cfg.CROSS_ATTN_EVERY,
                 padding_only_key=cfg.QFORMER_PADDING_ONLY_KEY):
        super().__init__()
        cfg = bert.config #type: ignore

        self.image_dim = image_dim
        self.num_queries = num_queries
        self.padding_only_key = padding_only_key
        self.hidden_size = cfg.hidden_size
        self.query_tokens = nn.Parameter(torch.empty(1, num_queries, cfg.hidden_size))

        self.layers = nn.ModuleList([
            QFormerBlock(layer, image_dim, i % cross_attn_every == 0, cfg)
            for i, layer in enumerate(bert.encoder.layer) #type: ignore
        ])

        nn.init.normal_(self.query_tokens, std=cfg.initializer_range)

    def forward(self, image_features, text_embeddings=None, padding_mask=None,
                objective=cfg.QFORMER_DEFAULT_OBJECTIVE, padding_only_key=None):
        B = image_features.size(0)
        q = self.query_tokens.expand(B, -1, -1)
        if text_embeddings is None:
            if padding_mask is not None:
                raise ValueError("padding_mask requires text_embeddings")
            padding_mask = torch.empty(B, 0, dtype=torch.bool, device=image_features.device)
            x = q
        else:
            if text_embeddings.ndim != 3 or text_embeddings.size(-1) != self.hidden_size:
                raise ValueError("text_embeddings must have shape [batch, length, hidden_size]")
            if text_embeddings.size(0) != B:
                raise ValueError("batch size mismatch")
            if padding_mask is None:
                padding_mask = torch.zeros(text_embeddings.shape[:2], dtype=torch.bool,
                                          device=text_embeddings.device)
            elif padding_mask.shape != text_embeddings.shape[:2]:
                raise ValueError("padding_mask shape mismatch")
            x = torch.cat((q, text_embeddings), dim=1)
        mask = generate_mask_qformer(
            padding_mask, self.num_queries, objective,
            self.padding_only_key if padding_only_key is None else padding_only_key,
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
    from transformers import AutoConfig, AutoTokenizer, BertModel

    torch.manual_seed(cfg.SEED)
    device = cfg.DEVICE
    tokenizer = AutoTokenizer.from_pretrained(cfg.QFORMER_MODEL_ID, use_fast=cfg.TOKENIZER_USE_FAST)
    vision_config = AutoConfig.from_pretrained(cfg.IMAGE_ENCODER_MODEL_ID)
    bert = BertModel.from_pretrained(cfg.QFORMER_MODEL_ID, add_pooling_layer=False)
    embeddings = bert.embeddings.requires_grad_(False).to(device).eval()
    model = QFormer(bert, image_dim=vision_config.hidden_size).to(device)

    texts = [
        "A dog is running on the grass.",
        "A car is parked beside the road.",
        "Two people are sitting at a table.",
    ]

    tokens = tokenizer(
        texts,
        padding=cfg.TOKENIZER_PADDING, truncation=cfg.TOKENIZER_TRUNCATION,
        max_length=cfg.MAX_INSTRUCTION_LENGTH,
        return_tensors="pt"
    ).to(device)

    B = len(texts)
    image = torch.randn(
        B, (cfg.IMAGE_SIZE // vision_config.patch_size) ** 2, vision_config.hidden_size, device=device
    )

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
                padding_mask=~tokens.attention_mask.bool(),
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
