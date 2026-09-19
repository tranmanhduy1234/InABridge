import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BertModel
from src_src.vision_encoder import ImageEncoder
from src_src.qformer import QFormer
from src_src import config_model as cfg

class ITCEncoder(nn.Module):
    def __init__(self, qformer, itc_dim=cfg.ITC_DIM, normalize_eps=cfg.ITC_NORMALIZE_EPS):
        super().__init__()
        self.qformer = qformer
        self.normalize_eps = normalize_eps
        self.query_proj = nn.Linear(qformer.hidden_size, itc_dim, bias=False)
        self.text_proj = nn.Linear(qformer.hidden_size, itc_dim, bias=False)
    def forward(self, image_features, text_embeddings, attention_mask):
        out = self.qformer(image_features, text_embeddings=text_embeddings,
                           padding_mask=~attention_mask.bool(), objective="itc")
        return {
            "image_features": F.normalize(self.query_proj(out["query_output"]), dim=-1, eps=self.normalize_eps),
            "text_features": F.normalize(self.text_proj(out["text_output"][:, 0]), dim=-1, eps=self.normalize_eps),
        }

class ModelStage1(nn.Module):
    def __init__(self, bert_name=cfg.QFORMER_MODEL_ID, *, vision_encoder=None, bert=None,
                 itc_encoder=None):
        super().__init__()
        self.vision_encoder = vision_encoder if vision_encoder is not None else ImageEncoder()
        if bert is None:
            bert = BertModel.from_pretrained(bert_name, add_pooling_layer=False)
        self.embeddings = bert.embeddings.requires_grad_(False).eval() #type: ignore
        if itc_encoder is None:
            itc_encoder = ITCEncoder(QFormer(bert, image_dim=self.vision_encoder.hidden_size))
        self.itc_encoder = itc_encoder
        qformer = itc_encoder.qformer
        self.itm_logit = nn.Linear(qformer.hidden_size, 2)
        self.lm_head = nn.Linear(qformer.hidden_size, self.embeddings.word_embeddings.num_embeddings, bias=False)
        self.lm_head.weight = self.embeddings.word_embeddings.weight
        self.lm_head.requires_grad_(False)

    @property
    def counter_parameter(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable = total - trainable
        return (
            f"Total: {total:,} | "
            f"Trainable: {trainable:,} | "
            f"Non-trainable: {non_trainable:,}"
        )
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
        return self

    def forward(self, image_features, input_ids, attention_mask, objective):
        objective = objective.lower()
        if objective not in {"itc", "itm", "itg"}:
            raise ValueError(f"Unknown objective: {objective}")
        text_embeddings = self.encode_text(input_ids)
        if objective == "itc":
            return self.itc_encoder(image_features, text_embeddings, attention_mask)
        out = self.qformer_model(image_features, text_embeddings=text_embeddings,
                                 padding_mask=~attention_mask.bool(), objective=objective)
        if objective == "itm":
            return {"itm_logits": self.itm_logit(out["query_output"]).mean(1)}
        return {"itg_logits": self.lm_head(out["text_output"][:, :-1])}

@torch.inference_mode()
def main():
    torch.manual_seed(cfg.SEED)
    device = torch.device(cfg.DEVICE)
    model = ModelStage1().to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.QFORMER_MODEL_ID, use_fast=cfg.TOKENIZER_USE_FAST)
    texts = [
        "A dog is running on the grass.",
        "Two people are sitting at a table.",
        "A red car is parked beside a house.",
    ]
    images = torch.randn(len(texts), 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, device=device)
    tokens = tokenizer(texts, padding=cfg.TOKENIZER_PADDING, truncation=cfg.TOKENIZER_TRUNCATION,
                       max_length=cfg.MAX_INSTRUCTION_LENGTH, return_tensors="pt").to(device)
    image_features = model.encode_image(images)
    print(f"Counter parameter: {model.counter_parameter}")
    print(f"Device: {device}")
    print(f"Images: {tuple(images.shape)}")
    print(f"Texts: {texts}")
    for objective in ("itc", "itm", "itg"):
        outputs = model(image_features, tokens.input_ids, tokens.attention_mask, objective)
        print(f"\n[{objective.upper()}]")
        for name, tensor in outputs.items():
            print(f"{name}: {tuple(tensor.shape)}")

if __name__ == "__main__":
    main()
