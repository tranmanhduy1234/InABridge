# -*- coding: utf-8 -*-
"""
DeBERTa-v3 Weight Loader and Tokenizer Utilities for InstructBLIP Q-Former
==========================================================================
Module providing pretrained weight extraction from `microsoft/deberta-v3-base`
into the custom InstructBLIP Q-Former architecture.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

DEFAULT_DEBERTA_MODEL_ID = "microsoft/deberta-v3-base"

def get_deberta_tokenizer(model_id: str = DEFAULT_DEBERTA_MODEL_ID) -> AutoTokenizer:
    """
    Tải bộ mã hóa ngôn ngữ (Tokenizer) DeBERTa-v3 cho InstructBLIP Q-Former.
    
    Args:
        model_id: HuggingFace model ID (default: microsoft/deberta-v3-base).
    
    Returns:
        AutoTokenizer tương ứng.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is None else "[PAD]"
    return tokenizer


def load_deberta_weights(
    qformer_model: nn.Module,
    model_id: str = DEFAULT_DEBERTA_MODEL_ID,
    verbose: bool = True,
) -> nn.Module:
    """
    Tải trọng số đã huấn luyện trước từ DeBERTa-v3-base của HuggingFace
    nạp vào các lớp Self-Attention, FFN và Embedding của InstructBLIP Q-Former.
    
    Args:
        qformer_model: Đối tượng QFormer (InstructBLIP) trong qformer.py.
        model_id: Model ID trên HuggingFace.
        verbose: In thông báo chi tiết trong quá trình gán trọng số.
        
    Returns:
        qformer_model đã được cập nhật trọng số pretrained.
    """
    if verbose:
        print(f"[DeBERTa Loader] Đang tải trọng số gốc từ '{model_id}' ...")

    try:
        hf_deberta = AutoModel.from_pretrained(model_id)
        hf_sd = hf_deberta.state_dict()
    except Exception as e:
        print(f"[DeBERTa Loader] ⚠ Lỗi khi tải mô hình từ HuggingFace: {e}")
        return qformer_model

    loaded_count = 0
    skipped_count = 0

    with torch.no_grad():
        # 1. Nạp Word Embeddings & LayerNorm
        if hasattr(qformer_model, "text_embeddings"):
            emb_module = qformer_model.text_embeddings
            if "embeddings.word_embeddings.weight" in hf_sd and hasattr(emb_module, "word_embeddings"):
                emb_module.word_embeddings.weight.copy_(hf_sd["embeddings.word_embeddings.weight"])
                loaded_count += 1
            if "embeddings.LayerNorm.weight" in hf_sd and hasattr(emb_module, "LayerNorm"):
                emb_module.LayerNorm.weight.copy_(hf_sd["embeddings.LayerNorm.weight"])
                emb_module.LayerNorm.bias.copy_(hf_sd["embeddings.LayerNorm.bias"])
                loaded_count += 2

        # 2. Nạp Relative Position Embeddings
        rel_pos_weight = hf_sd.get("encoder.rel_embeddings.weight", None)

        # 3. Nạp 12 Transformer Layers (DeBERTa Self-Attention + FFN)
        if hasattr(qformer_model, "deb_layers"):
            for i, layer in enumerate(qformer_model.deb_layers):
                prefix = f"encoder.layer.{i}."

                # Disentangled Self-Attention
                attn = layer.attn
                q_w = hf_sd.get(f"{prefix}attention.self.q_proj.weight")
                q_b = hf_sd.get(f"{prefix}attention.self.q_proj.bias")
                k_w = hf_sd.get(f"{prefix}attention.self.k_proj.weight")
                k_b = hf_sd.get(f"{prefix}attention.self.k_proj.bias")
                v_w = hf_sd.get(f"{prefix}attention.self.v_proj.weight")
                v_b = hf_sd.get(f"{prefix}attention.self.v_proj.bias")
                o_w = hf_sd.get(f"{prefix}attention.output.dense.weight")
                o_b = hf_sd.get(f"{prefix}attention.output.dense.bias")

                pos_q_w = hf_sd.get(f"{prefix}attention.self.pos_q.weight")
                pos_q_b = hf_sd.get(f"{prefix}attention.self.pos_q.bias")
                pos_k_w = hf_sd.get(f"{prefix}attention.self.pos_k.weight")
                pos_k_b = hf_sd.get(f"{prefix}attention.self.pos_k.bias")

                if q_w is not None and hasattr(attn, "q_proj"):
                    attn.q_proj.weight.copy_(q_w)
                    if q_b is not None and attn.q_proj.bias is not None:
                        attn.q_proj.bias.copy_(q_b)
                    loaded_count += 1

                if k_w is not None and hasattr(attn, "k_proj"):
                    attn.k_proj.weight.copy_(k_w)
                    if k_b is not None and attn.k_proj.bias is not None:
                        attn.k_proj.bias.copy_(k_b)
                    loaded_count += 1

                if v_w is not None and hasattr(attn, "v_proj"):
                    attn.v_proj.weight.copy_(v_w)
                    if v_b is not None and attn.v_proj.bias is not None:
                        attn.v_proj.bias.copy_(v_b)
                    loaded_count += 1

                if o_w is not None and hasattr(attn, "o_proj"):
                    attn.o_proj.weight.copy_(o_w)
                    if o_b is not None and attn.o_proj.bias is not None:
                        attn.o_proj.bias.copy_(o_b)
                    loaded_count += 1

                if pos_q_w is not None and hasattr(attn, "pos_q"):
                    attn.pos_q.weight.copy_(pos_q_w)
                    if pos_q_b is not None and attn.pos_q.bias is not None:
                        attn.pos_q.bias.copy_(pos_q_b)
                    loaded_count += 1

                if pos_k_w is not None and hasattr(attn, "pos_k"):
                    attn.pos_k.weight.copy_(pos_k_w)
                    if pos_k_b is not None and attn.pos_k.bias is not None:
                        attn.pos_k.bias.copy_(pos_k_b)
                    loaded_count += 1

                if rel_pos_weight is not None and hasattr(attn, "pos_emb"):
                    attn.pos_emb.weight.copy_(rel_pos_weight)
                    loaded_count += 1

                # LayerNorm 1
                ln1_w = hf_sd.get(f"{prefix}attention.output.LayerNorm.weight")
                ln1_b = hf_sd.get(f"{prefix}attention.output.LayerNorm.bias")
                if ln1_w is not None and hasattr(layer, "norm1"):
                    layer.norm1.weight.copy_(ln1_w)
                    if ln1_b is not None:
                        layer.norm1.bias.copy_(ln1_b)
                    loaded_count += 1

                # Feed Forward Network (FFN)
                fc1_w = hf_sd.get(f"{prefix}intermediate.dense.weight")
                fc1_b = hf_sd.get(f"{prefix}intermediate.dense.bias")
                fc2_w = hf_sd.get(f"{prefix}output.dense.weight")
                fc2_b = hf_sd.get(f"{prefix}output.dense.bias")

                if fc1_w is not None and hasattr(layer, "fc1"):
                    layer.fc1.weight.copy_(fc1_w)
                    if fc1_b is not None:
                        layer.fc1.bias.copy_(fc1_b)
                    loaded_count += 1

                if fc2_w is not None and hasattr(layer, "fc2"):
                    layer.fc2.weight.copy_(fc2_w)
                    if fc2_b is not None:
                        layer.fc2.bias.copy_(fc2_b)
                    loaded_count += 1

                # LayerNorm 2
                ln2_w = hf_sd.get(f"{prefix}output.LayerNorm.weight")
                ln2_b = hf_sd.get(f"{prefix}output.LayerNorm.bias")
                if ln2_w is not None and hasattr(layer, "norm2"):
                    layer.norm2.weight.copy_(ln2_w)
                    if ln2_b is not None:
                        layer.norm2.bias.copy_(ln2_b)
                    loaded_count += 1

    if verbose:
        print(f"[DeBERTa Loader] ✓ Nạp thành công {loaded_count} nhóm trọng số vào Q-Former!")

    return qformer_model


if __name__ == "__main__":
    print("=" * 60)
    print("  DeBERTa-v3 Utility Test")
    print("=" * 60)
    tokenizer = get_deberta_tokenizer()
    sample_text = "Instruction: Describe the image in detail."
    tokens = tokenizer(sample_text, return_tensors="pt")
    print(f"Sample Text : '{sample_text}'")
    print(f"Token IDs   : {tokens['input_ids'].shape} -> {tokens['input_ids'][0].tolist()[:10]}...")
    print("=" * 60)