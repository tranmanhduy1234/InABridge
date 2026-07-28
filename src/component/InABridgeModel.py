from __future__ import annotations

from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from src.component.ImageEncoder.imageEncoder import ImageEncoder
from src.component.Q_former.qformer import QFormer
from src.component.MLP.multiLayerPerceptron import QwenProjector
from src.component.LLM.llm import LanguageModelDecoder, LLMTrainMode, LoRAConfig
from src.config import (
    IMAGE_ENCODER_MODEL_ID,
    RETURN_LAYER,
    IMAGE_ENCODER_OUT_DIMENSION,
    NUM_QUERIES,
    MLP_IN_DIMENSION,
    MLP_HIDDEN_DIM,
    MLP_OUT_DIMENSION,
    LLM_MODL_ID,
    LLM_IN_DIMENSION,
    COMPUTE_TYPE,
)


class InABridgeModel(nn.Module):
    """
    InA-Bridge: Instruction-Aware Bridge Vision-Language Model.

    Kết hợp 4 thành phần chính:
        1. ImageEncoder  — DINOv3-L (frozen)
        2. QFormer       — DeBERTa-v3 Q-Former (trainable)
        3. QwenProjector — SwiGLU MLP bridge (trainable)
        4. LLM           — Qwen3-4B decoder (frozen / LoRA)

    Training Stages:
        Stage 1: Train Q-Former (ITC + ITM + ITG). LLM chưa được sử dụng.
        Stage 2: Train QwenProjector + LoRA trên LLM. Q-Former frozen hoặc fine-tune.
    """

    def __init__(
        self,
        # Image Encoder
        image_encoder_id: str = IMAGE_ENCODER_MODEL_ID,
        return_layer: int = RETURN_LAYER,
        # Q-Former
        num_queries: int = NUM_QUERIES,
        image_hidden_dim: int = IMAGE_ENCODER_OUT_DIMENSION,
        cross_attn_every: int = 2,
        freeze_text_embeds: bool = True,
        # Projector
        projector_in_dim: int = MLP_IN_DIMENSION,
        projector_hidden_dim: int = MLP_HIDDEN_DIM,
        projector_out_dim: int = MLP_OUT_DIMENSION,
        # LLM
        llm_model_id: str = LLM_MODL_ID,
        llm_train_mode: LLMTrainMode = LLMTrainMode.FROZEN,
        lora_config: Optional[LoRAConfig] = None,
        torch_dtype: torch.dtype = COMPUTE_TYPE,
    ):
        super().__init__()

        # ── 1. Image Encoder (Frozen) ─────────────────────────────
        self.image_encoder = ImageEncoder(
            model_id=image_encoder_id,
            return_layer=return_layer,
        )

        # ── 2. Q-Former Bridge (Trainable) ────────────────────────
        self.qformer = QFormer(
            num_queries=num_queries,
            image_hidden_dim=image_hidden_dim,
            cross_attn_every=cross_attn_every,
            freeze_text_embeds=freeze_text_embeds,
        )

        # ── 3. QwenProjector MLP (Trainable) ──────────────────────
        self.projector = QwenProjector(
            qformer_dim=projector_in_dim,
            hidden_dim=projector_hidden_dim,
            llm_dim=projector_out_dim,
        )

        # ── 4. LLM Decoder (Frozen / LoRA) ────────────────────────
        self.llm = LanguageModelDecoder(
            model_id=llm_model_id,
            train_mode=llm_train_mode,
            lora_config=lora_config,
            torch_dtype=torch_dtype,
        )

    # ------------------------------------------------------------------
    # Forward — Stage 2 Generative Training
    # ------------------------------------------------------------------

    def forward(
        self,
        pixel_values: torch.Tensor,
        instruction_ids: Optional[torch.Tensor] = None,
        instruction_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass cho Stage 2 generative training.

        Args:
            pixel_values:     (B, 3, H, W) — ảnh đầu vào.
            instruction_ids:  (B, L_inst) — token IDs instruction cho Q-Former (DeBERTa tokenizer).
            instruction_mask: (B, L_inst) — attention mask cho instruction.
            input_ids:        (B, L_text) — token IDs text cho LLM (Qwen tokenizer).
            attention_mask:   (B, L_total) — attention mask cho LLM (visual + text).
            labels:           (B, L_text) — target labels cho LM loss.

        Returns:
            CausalLMOutputWithPast chứa loss và logits.
        """
        # Step 1: Extract visual features
        image_features = self.image_encoder(pixel_values)  # (B, N_patches, 1024)

        # Step 2: Q-Former instruction-aware feature extraction
        qformer_out = self.qformer(
            image_features=image_features,
            input_ids=instruction_ids,
            attention_mask=instruction_mask,
        )
        query_output = qformer_out["query_output"]  # (B, 32, 768)

        # Step 3: Project to LLM embedding space
        visual_tokens, gate = self.projector(query_output)  # (B, 32, 2560)

        # Step 4: Get LLM text embeddings
        text_embeds = self.llm.model.get_input_embeddings()(input_ids)  # (B, L_text, 2560)

        # Step 5: Concatenate visual tokens + text embeddings
        # Layout: [V_soft (32 tokens) | instruction + answer text]
        inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)  # (B, 32+L_text, 2560)

        # Step 6: Build attention mask cho combined sequence
        if attention_mask is not None:
            visual_mask = torch.ones(
                visual_tokens.size(0), visual_tokens.size(1),
                dtype=attention_mask.dtype, device=attention_mask.device,
            )
            combined_mask = torch.cat([visual_mask, attention_mask], dim=1)
        else:
            combined_mask = None

        # Step 7: Build labels (mask visual token positions with -100)
        if labels is not None:
            visual_labels = torch.full(
                (labels.size(0), visual_tokens.size(1)),
                fill_value=-100,
                dtype=labels.dtype,
                device=labels.device,
            )
            combined_labels = torch.cat([visual_labels, labels], dim=1)
        else:
            combined_labels = None

        # Step 8: Forward through LLM
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            labels=combined_labels,
        )

    # ------------------------------------------------------------------
    # Generate — Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        instruction_ids: Optional[torch.Tensor] = None,
        instruction_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Autoregressive generation cho inference.

        Args:
            pixel_values:      (B, 3, H, W) — ảnh đầu vào.
            instruction_ids:   (B, L_inst) — instruction token IDs (DeBERTa tokenizer).
            instruction_mask:  (B, L_inst) — instruction attention mask.
            input_ids:         (B, L_prompt) — prompt token IDs (Qwen tokenizer).
            attention_mask:    (B, L_prompt) — prompt attention mask.
            max_new_tokens:    Số token tối đa được sinh.
            temperature:       Sampling temperature.
            top_p:             Top-p (nucleus) sampling.
            repetition_penalty: Penalty cho token lặp lại.
            do_sample:         Có sampling hay greedy decoding.

        Returns:
            generated_ids: (B, max_new_tokens) — token IDs đã sinh.
        """
        # Step 1-3: Visual feature extraction + projection
        image_features = self.image_encoder(pixel_values)
        qformer_out = self.qformer(
            image_features=image_features,
            input_ids=instruction_ids,
            attention_mask=instruction_mask,
        )
        query_output = qformer_out["query_output"]
        visual_tokens, _ = self.projector(query_output)

        # Step 4: Build inputs_embeds
        if input_ids is not None:
            text_embeds = self.llm.model.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
        else:
            inputs_embeds = visual_tokens

        # Step 5: Build attention mask
        if attention_mask is not None:
            visual_mask = torch.ones(
                visual_tokens.size(0), visual_tokens.size(1),
                dtype=attention_mask.dtype, device=attention_mask.device,
            )
            combined_mask = torch.cat([visual_mask, attention_mask], dim=1)
        else:
            combined_mask = torch.ones(
                inputs_embeds.size(0), inputs_embeds.size(1),
                dtype=torch.long, device=inputs_embeds.device,
            )

        # Step 6: Generate
        return self.llm.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Stage 1 — Pre-training losses (ITC / ITM / ITG)
    # ------------------------------------------------------------------

    def forward_stage1(
        self,
        pixel_values: torch.Tensor,
        caption_ids: torch.Tensor,
        caption_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass cho Stage 1 representation learning.
        Trả về output cần thiết cho 3 loss functions: ITC, ITM, ITG.

        Args:
            pixel_values: (B, 3, H, W) — ảnh đầu vào.
            caption_ids:  (B, L_cap) — caption token IDs (DeBERTa tokenizer).
            caption_mask: (B, L_cap) — caption attention mask.

        Returns:
            Dict chứa:
                - itc_query_proj: (B, 32, 256) — query projections cho ITC
                - itc_text_proj:  (B, 256) — text projections cho ITC
                - itm_logits:     (B, 2) — matching logits cho ITM
                - itg_logits:     (B, L_cap, vocab_size) — LM logits cho ITG
        """
        # Extract visual features (LLM không được sử dụng trong Stage 1)
        image_features = self.image_encoder(pixel_values)

        # ITC
        z_q_proj, t_proj = self.qformer.forward_itc(
            image_features=image_features,
            caption_ids=caption_ids,
            caption_mask=caption_mask,
        )

        # ITM
        itm_logits = self.qformer.forward_itm(
            image_features=image_features,
            caption_ids=caption_ids,
            caption_mask=caption_mask,
        )

        # ITG
        itg_logits = self.qformer.forward_itg(
            image_features=image_features,
            caption_ids=caption_ids,
            caption_mask=caption_mask,
        )

        return {
            "itc_query_proj": z_q_proj,
            "itc_text_proj": t_proj,
            "itm_logits": itm_logits,
            "itg_logits": itg_logits,
        }

    # ------------------------------------------------------------------
    # Trainable parameter management
    # ------------------------------------------------------------------

    def get_trainable_params_stage1(self) -> list:
        """Trả về danh sách parameter trainable cho Stage 1 (Q-Former only)."""
        params = []
        for p in self.qformer.parameters():
            if p.requires_grad:
                params.append(p)
        return params

    def get_trainable_params_stage2(self) -> list:
        """Trả về danh sách parameter trainable cho Stage 2 (Projector + LoRA)."""
        params = []
        for p in self.projector.parameters():
            if p.requires_grad:
                params.append(p)
        for p in self.llm.parameters():
            if p.requires_grad:
                params.append(p)
        return params

    # ------------------------------------------------------------------
    # Summary & Debug
    # ------------------------------------------------------------------

    def summary(self):
        """In tóm tắt toàn bộ model."""
        total_p = sum(p.numel() for p in self.parameters())
        train_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_p = total_p - train_p

        print("\n" + "═" * 70)
        print("  InA-Bridge Model — Summary")
        print("═" * 70)
        print(f"  Image Encoder    : {IMAGE_ENCODER_MODEL_ID}")
        print(f"  Q-Former         : DeBERTa-v3-base ({NUM_QUERIES} queries)")
        print(f"  Projector        : QwenProjector ({MLP_IN_DIMENSION}→{MLP_HIDDEN_DIM}→{MLP_OUT_DIMENSION})")
        print(f"  LLM Decoder      : {LLM_MODL_ID}")
        print("─" * 70)
        print(f"  Total params     : {total_p:>14,} ({total_p/1e9:.2f}B)")
        print(f"  Trainable        : {train_p:>14,} ({train_p/1e6:.2f}M)")
        print(f"  Frozen           : {frozen_p:>14,} ({frozen_p/1e6:.2f}M)")
        print("═" * 70 + "\n")

        # Sub-component summaries
        self.image_encoder.summary()
        self.qformer.print_summary()
        self.llm.summary()