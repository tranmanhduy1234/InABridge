import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from src.vision import ImageEncoder
from src.qformer import QFormer

class ITCEncoder(nn.Module):
    def __init__(self, qformer:QFormer, itc_dim, normalize_eps):
        super().__init__()
        self.qformer = qformer
        self.normalize_eps = normalize_eps
        self.query_proj = nn.Linear(qformer.hidden_dim, itc_dim, bias=False)
        self.text_proj = nn.Linear(qformer.hidden_dim, itc_dim, bias=False)
        self._init_head()

    def _init_head(self):
        for head in (self.query_proj, self.text_proj):
            nn.init.normal_(head.weight, mean=0.0, std=self.qformer.cfg.initializer_range)

    def forward(self, image_features, text_embeddings, attn_mask):
        out = self.qformer(image_features, text_embeddings, attn_mask.bool(), "itc")
        return {
            "image_features": F.normalize(self.query_proj(out["query_output"]), dim=-1, eps=self.normalize_eps),
            "text_features": F.normalize(self.text_proj(out["text_output"][:, 0]), dim=-1, eps=self.normalize_eps),
        }

class ModelStage1(nn.Module):
    def __init__(self, bert_name, return_layer, model_vision_id, num_queries,
                 cross_attn_every, hidden_dim, itc_dim, normalize_eps,
                 bert_config=None, vision_config=None):
        super().__init__()
        self.bert_name = bert_name
        self.vision_encoder = ImageEncoder(return_layer, model_vision_id, vision_config)
        bert = (BertModel.from_pretrained(bert_name, add_pooling_layer=False) if bert_config is None
                else BertModel(bert_config, add_pooling_layer=False))
        self.embeddings = bert.embeddings.requires_grad_(False).eval()
        qformer = QFormer(
            bert=bert,
            num_queries=num_queries,
            image_dim=self.vision_encoder.hidden_size,
            cross_attn_every=cross_attn_every,
            hidden_dim=hidden_dim,
        )
        self.itc_encoder = ITCEncoder(qformer, itc_dim, normalize_eps)
        self.itm_logit = nn.Linear(hidden_dim, 2)
        nn.init.normal_(self.itm_logit.weight, std=bert.config.initializer_range)
        nn.init.zeros_(self.itm_logit.bias)
        self.lm_head = nn.Linear(hidden_dim, self.embeddings.word_embeddings.num_embeddings, bias=False)
        self.lm_head.weight = self.embeddings.word_embeddings.weight
        self.lm_head.requires_grad_(False).eval()

    @property
    def qformer_model(self):
        return self.itc_encoder.qformer

    def encode_image(self, images):
        return self.vision_encoder(images)

    def encode_text(self, input_ids):
        return self.embeddings(input_ids=input_ids)

    def train(self, mode=True):
        super().train(mode)
        self.embeddings.eval()
        self.lm_head.eval()
        return self

    def forward(self, image_features, input_ids, attn_mask, objective):
        objective = objective.lower()
        if objective not in {"itc", "itm", "itg"}:
            raise ValueError("objective must be itc, itm or itg")
        text_embeddings = self.encode_text(input_ids)
        if objective == "itc":
            return self.itc_encoder(image_features, text_embeddings, attn_mask)
        out = self.qformer_model(image_features, text_embeddings, attn_mask, objective)
        if objective == "itm":
            return {"itm_logits": self.itm_logit(out["query_output"]).mean(1)}
        return {"itg_logits": self.lm_head(out["text_output"][:, :-1])}

@torch.inference_mode()
def main():
    from transformers import AutoTokenizer

    from src.utils.seed import seed_everything

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelStage1(
        bert_name="bert-base-uncased",
        return_layer=-1,
        model_vision_id="facebook/dinov3-vitl16-pretrain-lvd1689m",
        num_queries=128,
        cross_attn_every=2,
        hidden_dim=768,
        itc_dim=256,
        normalize_eps=1e-12,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    texts = [
        "A dog is running on the grass.",
        "Two people are sitting at a table.",
        "A red car is parked beside a house.",
    ]
    # Synthetic images demonstrate tensor flow; use preprocessed images for real data.
    images = torch.randn(len(texts), 3, 224, 224, device=device)
    tokens = tokenizer(
        texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
    ).to(device)
    attn_mask = tokens.attention_mask.bool()  # True = valid token, as in the dataloader.
    image_features = model.encode_image(images)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,} total, {trainable:,} trainable")
    print(f"Device: {device}")
    print(f"Images: {tuple(images.shape)}")
    print(f"Image features: {tuple(image_features.shape)}")
    print(f"Input IDs: {tuple(tokens.input_ids.shape)}")
    print(f"Attention mask: {tuple(attn_mask.shape)}")
    print(f"Texts: {texts}")
    for objective in ("itc", "itm", "itg"):
        outputs = model(
            image_features=image_features,
            input_ids=tokens.input_ids,
            attn_mask=attn_mask,
            objective=objective,
        )
        print(f"\n[{objective.upper()}]")
        for name, tensor in outputs.items():
            print(f"{name}: {tuple(tensor.shape)}")

if __name__ == "__main__":
    main()
