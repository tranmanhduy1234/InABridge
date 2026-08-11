"""End-to-end orchestration for the two-stage InA-Bridge model."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Any, Dict, Mapping, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    COMPUTE_TYPE,
    IMAGE_ENCODER_MODEL_ID,
    IMAGE_ENCODER_OUT_DIMENSION,
    LLM_MODEL_ID,
    NUM_QUERIES,
    PROJECTOR_HIDDEN_DIMENSION,
    PROJECTOR_IN_DIMENSION,
    PROJECTOR_OUT_DIMENSION,
    RETURN_LAYER,
)
from src.language_model import LLMTrainMode, LanguageModelDecoder, LoRAConfig
from src.projector import QwenProjector
from src.qformer import QFormer
from src.vision_encoder import ImageEncoder


@dataclass
class VisualEncoding:
    """Intermediate bridge activations useful for diagnostics and regularizers."""

    query_tokens: torch.Tensor
    visual_tokens: torch.Tensor
    projector_gate: Optional[torch.Tensor]


@dataclass
class MultimodalInputs:
    """Inputs passed to the decoder after visual-prefix construction."""

    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    labels: Optional[torch.Tensor]
    visual_prefix_length: int
    visual_encoding: VisualEncoding


class InABridgeModel(nn.Module):
    """Instruction-aware DINO -> Q-Former -> Qwen vision-language bridge.

    Stage 1 trains Q-Former with ITC, ITM, and ITG while DINO is frozen and
    Qwen need not be loaded. Stage 2 keeps DINO and base Qwen frozen, trains the
    instruction-aware Q-Former and projector, and enables only LoRA/QLoRA
    parameters in the decoder.
    """

    _STAGE1_ONLY_QFORMER_PREFIXES = (
        "itc_query_proj.",
        "itc_text_proj.",
        "itm_head.",
        "itg_transform_dense.",
        "itg_transform_norm.",
        "itg_lm_head.",
        "dec_token",
        "itc_temp",
    )

    def __init__(
        self,
        image_encoder_id: str = IMAGE_ENCODER_MODEL_ID,
        return_layer: int = RETURN_LAYER,
        num_queries: int = NUM_QUERIES,
        image_hidden_dim: int = IMAGE_ENCODER_OUT_DIMENSION,
        cross_attn_every: int = 2,
        freeze_text_embeds: bool = False,
        projector_in_dim: int = PROJECTOR_IN_DIMENSION,
        projector_hidden_dim: int = PROJECTOR_HIDDEN_DIMENSION,
        projector_out_dim: int = PROJECTOR_OUT_DIMENSION,
        llm_model_id: str = LLM_MODEL_ID,
        llm_train_mode: LLMTrainMode = LLMTrainMode.FROZEN,
        lora_config: Optional[LoRAConfig] = None,
        torch_dtype: torch.dtype = COMPUTE_TYPE,
        llm_device_map: Optional[Any] = None,
        image_encoder: Optional[ImageEncoder] = None,
        qformer: Optional[QFormer] = None,
        projector: Optional[QwenProjector] = None,
        llm: Optional[LanguageModelDecoder] = None,
        load_llm: bool = True,
    ) -> None:
        super().__init__()
        if num_queries <= 0:
            raise ValueError("num_queries must be positive")
        if not load_llm and llm is not None:
            raise ValueError("Cannot supply llm when load_llm=False")

        self.image_encoder = image_encoder or ImageEncoder(
            model_id=image_encoder_id,
            return_layer=return_layer,
        )
        resolved_image_dim = (
            image_hidden_dim if qformer is not None else int(self.image_encoder.embed_dim)
        )
        self.qformer = qformer or QFormer(
            num_queries=num_queries,
            image_hidden_dim=resolved_image_dim,
            cross_attn_every=cross_attn_every,
            freeze_text_embeds=freeze_text_embeds,
        )
        self.projector = projector or QwenProjector(
            qformer_dim=projector_in_dim,
            hidden_dim=projector_hidden_dim,
            llm_dim=projector_out_dim,
        )
        self.llm: Optional[nn.Module]
        self.llm = None
        if load_llm:
            self.llm = llm or LanguageModelDecoder(
                model_id=llm_model_id,
                train_mode=llm_train_mode,
                lora_config=lora_config,
                torch_dtype=torch_dtype,
                device_map=llm_device_map,
            )

        # Preserve intentionally frozen Q-Former parameters, such as frozen
        # text embeddings, when switching between stages later.
        self._permanently_frozen_qformer = {
            name for name, parameter in self.qformer.named_parameters() if not parameter.requires_grad
        }
        self.training_stage: Optional[int] = None
        self._validate_component_dimensions()

    # ------------------------------------------------------------------
    # Component and tensor validation
    # ------------------------------------------------------------------

    @staticmethod
    def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
        for tensor in chain(module.parameters(), module.buffers()):
            dtype = tensor.dtype if tensor.is_floating_point() else torch.float32
            return tensor.device, dtype
        return torch.device("cpu"), torch.float32

    @staticmethod
    def _embedding_dimension(embedding: nn.Module) -> Optional[int]:
        dimension = getattr(embedding, "embedding_dim", None)
        if dimension is not None:
            return int(dimension)
        weight = getattr(embedding, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            return int(weight.size(1))
        return None

    def _validate_component_dimensions(self) -> None:
        vision_dim = getattr(self.image_encoder, "embed_dim", None)
        qformer_image_dim = getattr(self.qformer, "image_hidden_dim", None)
        qformer_dim = getattr(self.qformer, "hidden_dim", None)
        projector_in = getattr(self.projector, "in_dim", None)
        if projector_in is None:
            projector_in = getattr(getattr(self.projector, "fc1", None), "in_features", None)
        projector_out = getattr(self.projector, "out_dim", None)
        if projector_out is None:
            projector_out = getattr(getattr(self.projector, "fc2", None), "out_features", None)
        llm_dim = None
        if self.llm is not None:
            llm_dim = self._embedding_dimension(self.llm.get_input_embeddings())
        for source, target, name in (
            (vision_dim, qformer_image_dim, "vision encoder -> Q-Former"),
            (qformer_dim, projector_in, "Q-Former -> projector"),
            (projector_out, llm_dim, "projector -> LLM"),
        ):
            if source is not None and target is not None and int(source) != int(target):
                raise ValueError(f"Dimension mismatch ({name}): {source} != {target}")

    @staticmethod
    def _validate_ids_and_mask(
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        *,
        name: str,
        batch_size: int,
        required: bool,
    ) -> None:
        if input_ids is None:
            if required:
                raise ValueError(f"{name}_ids is required")
            if attention_mask is not None:
                raise ValueError(f"{name}_mask cannot be supplied without {name}_ids")
            return
        if input_ids.ndim != 2 or input_ids.size(0) != batch_size:
            raise ValueError(f"{name}_ids must have shape [batch, sequence]")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name}_ids must contain integer token IDs")
        if input_ids.size(1) == 0:
            raise ValueError(f"{name}_ids cannot have an empty sequence")
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError(f"{name}_mask and {name}_ids must have the same shape")
            if bool(((attention_mask != 0) & (attention_mask != 1)).any()):
                raise ValueError(f"{name}_mask must contain only 0 and 1")
            if bool(attention_mask.sum(dim=1).eq(0).any()):
                raise ValueError(f"Every {name} sequence needs at least one valid token")

    @staticmethod
    def _validate_pixels(pixel_values: torch.Tensor) -> None:
        if pixel_values.ndim != 4 or pixel_values.size(0) == 0:
            raise ValueError("pixel_values must have shape [non-empty batch, channels, height, width]")
        if pixel_values.size(1) != 3:
            raise ValueError("pixel_values must contain three RGB channels")
        if not pixel_values.is_floating_point():
            raise TypeError("pixel_values must be a floating-point tensor")

    # ------------------------------------------------------------------
    # Visual and multimodal encoding
    # ------------------------------------------------------------------

    def encode_visual(
        self,
        pixel_values: torch.Tensor,
        instruction_ids: Optional[torch.Tensor] = None,
        instruction_mask: Optional[torch.Tensor] = None,
        *,
        return_details: bool = False,
    ) -> torch.Tensor | VisualEncoding:
        """Encode images into soft prompt tokens in the decoder embedding space."""

        self._validate_pixels(pixel_values)
        batch_size = pixel_values.size(0)
        self._validate_ids_and_mask(
            instruction_ids,
            instruction_mask,
            name="instruction",
            batch_size=batch_size,
            required=False,
        )
        vision_device, vision_dtype = self._module_device_dtype(self.image_encoder)
        image_features = self.image_encoder(
            pixel_values.to(device=vision_device, dtype=vision_dtype, non_blocking=True)
        )
        if image_features.ndim != 3 or image_features.size(0) != batch_size:
            raise RuntimeError("Image encoder must return [batch, patches, hidden]")

        q_device, q_dtype = self._module_device_dtype(self.qformer)
        image_features = image_features.to(device=q_device, dtype=q_dtype, non_blocking=True)
        if instruction_ids is not None:
            instruction_ids = instruction_ids.to(q_device, non_blocking=True)
        if instruction_mask is not None:
            instruction_mask = instruction_mask.to(q_device, non_blocking=True)
        qformer_output = self.qformer(
            image_features=image_features,
            input_ids=instruction_ids,
            attention_mask=instruction_mask,
        )
        if not isinstance(qformer_output, Mapping) or "query_output" not in qformer_output:
            raise RuntimeError("Q-Former output must be a dict containing 'query_output'")
        query_tokens = qformer_output["query_output"]
        if query_tokens.ndim != 3 or query_tokens.size(0) != batch_size:
            raise RuntimeError("Q-Former query_output must have shape [batch, queries, hidden]")

        projector_device, projector_dtype = self._module_device_dtype(self.projector)
        projected = self.projector(
            query_tokens.to(
                device=projector_device,
                dtype=projector_dtype,
                non_blocking=True,
            )
        )
        if isinstance(projected, tuple):
            visual_tokens, gate = projected
        else:
            visual_tokens, gate = projected, None
        if visual_tokens.ndim != 3 or visual_tokens.shape[:2] != query_tokens.shape[:2]:
            raise RuntimeError("Projector must preserve [batch, query] dimensions")
        details = VisualEncoding(query_tokens, visual_tokens, gate)
        return details if return_details else visual_tokens

    def prepare_multimodal_inputs(
        self,
        pixel_values: torch.Tensor,
        instruction_ids: Optional[torch.Tensor] = None,
        instruction_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        *,
        include_labels: bool = True,
    ) -> MultimodalInputs:
        """Build one canonical ``[visual prefix | text]`` decoder sequence."""

        if self.llm is None:
            raise RuntimeError("The language model is not loaded")
        self._validate_pixels(pixel_values)
        batch_size = pixel_values.size(0)
        self._validate_ids_and_mask(
            input_ids,
            attention_mask,
            name="text",
            batch_size=batch_size,
            required=False,
        )
        if labels is not None:
            if input_ids is None or labels.shape != input_ids.shape:
                raise ValueError("labels and input_ids must have the same shape")
            if labels.dtype not in (torch.int32, torch.int64):
                raise TypeError("labels must contain integer token IDs")

        visual = self.encode_visual(
            pixel_values,
            instruction_ids,
            instruction_mask,
            return_details=True,
        )
        assert isinstance(visual, VisualEncoding)
        embedding = self.llm.get_input_embeddings()
        embedding_device, embedding_dtype = self._module_device_dtype(embedding)
        visual_tokens = visual.visual_tokens.to(
            device=embedding_device,
            dtype=embedding_dtype,
            non_blocking=True,
        )
        if input_ids is None:
            text_embeds = visual_tokens.new_empty(
                (batch_size, 0, visual_tokens.size(-1))
            )
            text_mask = torch.empty(
                (batch_size, 0), dtype=torch.long, device=embedding_device
            )
        else:
            input_ids = input_ids.to(embedding_device, non_blocking=True)
            text_embeds = embedding(input_ids)
            if attention_mask is None:
                text_mask = torch.ones_like(input_ids, dtype=torch.long)
            else:
                text_mask = attention_mask.to(embedding_device, non_blocking=True)
        if text_embeds.size(-1) != visual_tokens.size(-1):
            raise RuntimeError("Visual tokens and text embeddings have different hidden sizes")

        inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
        visual_mask = torch.ones(
            (batch_size, visual_tokens.size(1)),
            dtype=text_mask.dtype,
            device=embedding_device,
        )
        combined_mask = torch.cat([visual_mask, text_mask], dim=1)
        combined_labels = None
        if labels is not None and include_labels:
            text_labels = labels.to(device=embedding_device, dtype=torch.long).clone()
            if attention_mask is not None:
                text_labels.masked_fill_(text_mask.eq(0), -100)
            visual_labels = torch.full(
                (batch_size, visual_tokens.size(1)),
                -100,
                dtype=torch.long,
                device=embedding_device,
            )
            combined_labels = torch.cat([visual_labels, text_labels], dim=1)
        return MultimodalInputs(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            labels=combined_labels,
            visual_prefix_length=visual_tokens.size(1),
            visual_encoding=VisualEncoding(
                query_tokens=visual.query_tokens,
                visual_tokens=visual_tokens,
                projector_gate=visual.projector_gate,
            ),
        )

    # ------------------------------------------------------------------
    # Stage 2 forward and generation
    # ------------------------------------------------------------------

    def forward(
        self,
        pixel_values: torch.Tensor,
        instruction_ids: Optional[torch.Tensor] = None,
        instruction_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        compute_loss_with_llm: bool = True,
        caption_ids: Optional[torch.Tensor] = None,
        caption_mask: Optional[torch.Tensor] = None,
        itm_caption_ids: Optional[torch.Tensor] = None,
        itm_caption_mask: Optional[torch.Tensor] = None,
        itm_image_indices: Optional[torch.Tensor] = None,
        **llm_kwargs: Any,
    ):
        # Stage 1 must enter through ``forward`` so DistributedDataParallel and
        # Accelerate install their normal reducer hooks. Calling a custom method
        # directly on a wrapped module can bypass DDP's forward lifecycle.
        if caption_ids is not None:
            if any(
                value is not None
                for value in (instruction_ids, instruction_mask, input_ids, attention_mask, labels)
            ):
                raise ValueError("Stage-1 caption inputs cannot be mixed with Stage-2 inputs")
            return self.forward_stage1(
                pixel_values=pixel_values,
                caption_ids=caption_ids,
                caption_mask=caption_mask,
                itm_caption_ids=itm_caption_ids,
                itm_caption_mask=itm_caption_mask,
                itm_image_indices=itm_image_indices,
            )
        if any(
            value is not None
            for value in (caption_mask, itm_caption_ids, itm_caption_mask, itm_image_indices)
        ):
            raise ValueError("Stage-1 auxiliary inputs require caption_ids")
        if input_ids is None:
            raise ValueError("input_ids is required for generative training")
        batch = self.prepare_multimodal_inputs(
            pixel_values=pixel_values,
            instruction_ids=instruction_ids,
            instruction_mask=instruction_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            include_labels=compute_loss_with_llm,
        )
        if "use_cache" not in llm_kwargs and self.training:
            llm_kwargs["use_cache"] = False
        assert self.llm is not None
        return self.llm(
            inputs_embeds=batch.inputs_embeds,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
            **llm_kwargs,
        )

    @torch.inference_mode()
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
        **generation_kwargs: Any,
    ) -> torch.Tensor:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if do_sample and temperature <= 0:
            raise ValueError("temperature must be positive when sampling")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        batch = self.prepare_multimodal_inputs(
            pixel_values=pixel_values,
            instruction_ids=instruction_ids,
            instruction_mask=instruction_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            include_labels=False,
        )
        kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            **generation_kwargs,
        }
        if do_sample:
            kwargs.update({"temperature": temperature, "top_p": top_p})
        assert self.llm is not None
        generator = getattr(self.llm, "generate", None)
        if not callable(generator):
            generator = self.llm.model.generate  # type: ignore[attr-defined]
        return generator(
            inputs_embeds=batch.inputs_embeds,
            attention_mask=batch.attention_mask,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Stage 1 outputs and hard-negative mining
    # ------------------------------------------------------------------

    @staticmethod
    def _distributed_active() -> bool:
        return dist.is_available() and dist.is_initialized()

    @staticmethod
    def _all_gather_detached(tensor: torch.Tensor) -> torch.Tensor:
        if not InABridgeModel._distributed_active():
            return tensor.detach()
        gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor.detach().contiguous())
        return torch.cat(gathered, dim=0)

    def _stage1_negative_bank(
        self,
        image_features: torch.Tensor,
        query_features: torch.Tensor,
        text_features: torch.Tensor,
        caption_ids: torch.Tensor,
        caption_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        if not self._distributed_active():
            return (
                image_features.detach(),
                query_features.detach(),
                text_features.detach(),
                caption_ids,
                caption_mask,
                0,
            )

        local_batch = torch.tensor(
            [caption_ids.size(0)], device=caption_ids.device, dtype=torch.long
        )
        batch_sizes = [torch.zeros_like(local_batch) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_sizes, local_batch)
        if len({int(size.item()) for size in batch_sizes}) != 1:
            raise ValueError("Distributed Stage-1 mining requires equal per-rank batch sizes")

        local_length = torch.tensor(
            [caption_ids.size(1)], device=caption_ids.device, dtype=torch.long
        )
        dist.all_reduce(local_length, op=dist.ReduceOp.MAX)
        max_length = int(local_length.item())
        pad_length = max_length - caption_ids.size(1)
        if pad_length:
            caption_ids = F.pad(caption_ids, (0, pad_length), value=0)
            caption_mask = F.pad(caption_mask, (0, pad_length), value=0)

        all_images = self._all_gather_detached(image_features)
        all_queries = self._all_gather_detached(query_features)
        all_texts = self._all_gather_detached(text_features)
        all_ids = self._all_gather_detached(caption_ids)
        all_masks = self._all_gather_detached(caption_mask)
        offset = dist.get_rank() * caption_ids.size(0)
        return all_images, all_queries, all_texts, all_ids, all_masks, offset

    def _choose_negative(self, similarities: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
        scores = similarities.float().clone()
        scores.scatter_(1, positives.unsqueeze(1), torch.finfo(scores.dtype).min)
        if scores.size(1) < 2:
            raise ValueError("ITM hard-negative mining needs at least two global pairs")
        if not self.training:
            return scores.argmax(dim=1)
        weights = torch.softmax(scores, dim=1)
        # Softmax can underflow for a pathological row. Fall back to uniform
        # non-positive sampling instead of returning an invalid multinomial.
        invalid = ~torch.isfinite(weights).all(dim=1) | weights.sum(dim=1).le(0)
        if bool(invalid.any()):
            weights[invalid] = 1.0
            weights[invalid, positives[invalid]] = 0.0
        return torch.multinomial(weights, num_samples=1).squeeze(1)

    def forward_stage1(
        self,
        pixel_values: torch.Tensor,
        caption_ids: torch.Tensor,
        caption_mask: Optional[torch.Tensor] = None,
        itm_caption_ids: Optional[torch.Tensor] = None,
        itm_caption_mask: Optional[torch.Tensor] = None,
        itm_image_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        self._validate_pixels(pixel_values)
        batch_size = pixel_values.size(0)
        self._validate_ids_and_mask(
            caption_ids,
            caption_mask,
            name="caption",
            batch_size=batch_size,
            required=True,
        )
        if batch_size < 2:
            raise ValueError("Stage 1 requires at least two image-caption pairs")

        vision_device, vision_dtype = self._module_device_dtype(self.image_encoder)
        image_features = self.image_encoder(
            pixel_values.to(device=vision_device, dtype=vision_dtype, non_blocking=True)
        )
        q_device, q_dtype = self._module_device_dtype(self.qformer)
        image_features = image_features.to(device=q_device, dtype=q_dtype, non_blocking=True)
        caption_ids = caption_ids.to(q_device, non_blocking=True)
        caption_mask = (
            torch.ones_like(caption_ids)
            if caption_mask is None
            else caption_mask.to(q_device, non_blocking=True)
        )

        query_projection, text_projection = self.qformer.forward_itc(
            image_features=image_features,
            caption_ids=caption_ids,
            caption_mask=caption_mask,
        )

        if itm_caption_ids is not None:
            if itm_caption_ids.ndim != 2 or itm_caption_ids.size(0) < batch_size:
                raise ValueError("itm_caption_ids must contain at least the positive batch")
            pair_count = itm_caption_ids.size(0)
            itm_caption_ids = itm_caption_ids.to(q_device, non_blocking=True)
            if itm_caption_mask is None:
                itm_caption_mask = torch.ones_like(itm_caption_ids)
            elif itm_caption_mask.shape != itm_caption_ids.shape:
                raise ValueError("itm_caption_mask and itm_caption_ids must have the same shape")
            else:
                itm_caption_mask = itm_caption_mask.to(q_device, non_blocking=True)
            if itm_image_indices is None:
                if pair_count % batch_size:
                    raise ValueError(
                        "ITM pair count must be a batch multiple unless itm_image_indices is given"
                    )
                itm_images = image_features.repeat(pair_count // batch_size, 1, 1)
            else:
                indices = itm_image_indices.to(q_device, dtype=torch.long).reshape(-1)
                if indices.numel() != pair_count:
                    raise ValueError("itm_image_indices must provide one image index per ITM pair")
                if bool((indices < 0).any()) or bool((indices >= batch_size).any()):
                    raise ValueError("itm_image_indices contains an out-of-range image index")
                itm_images = image_features.index_select(0, indices)
            itm_labels = torch.zeros(pair_count, dtype=torch.long, device=q_device)
            itm_labels[:batch_size] = 1
        else:
            (
                all_images,
                all_queries,
                all_texts,
                all_caption_ids,
                all_caption_masks,
                offset,
            ) = self._stage1_negative_bank(
                image_features,
                query_projection,
                text_projection,
                caption_ids,
                caption_mask,
            )
            positives = offset + torch.arange(batch_size, device=q_device)
            mining_temperature = self.qformer.itc_temp.detach().float().clamp(0.01, 1.0)
            similarity_i2t = torch.einsum(
                "bqd,gd->bgq", query_projection.detach(), all_texts
            ).amax(dim=-1) / mining_temperature
            similarity_t2i = torch.einsum(
                "gqd,bd->bgq", all_queries, text_projection.detach()
            ).amax(dim=-1) / mining_temperature
            negative_text = self._choose_negative(similarity_i2t, positives)
            negative_image = self._choose_negative(similarity_t2i, positives)

            local_length = all_caption_ids.size(1)
            if caption_ids.size(1) < local_length:
                local_ids = F.pad(caption_ids, (0, local_length - caption_ids.size(1)), value=0)
                local_masks = F.pad(
                    caption_mask, (0, local_length - caption_mask.size(1)), value=0
                )
            else:
                local_ids, local_masks = caption_ids, caption_mask
            itm_caption_ids = torch.cat(
                [local_ids, all_caption_ids.index_select(0, negative_text), local_ids], dim=0
            )
            itm_caption_mask = torch.cat(
                [local_masks, all_caption_masks.index_select(0, negative_text), local_masks], dim=0
            )
            itm_images = torch.cat(
                [image_features, image_features, all_images.index_select(0, negative_image)],
                dim=0,
            )
            itm_labels = torch.cat(
                [
                    torch.ones(batch_size, dtype=torch.long, device=q_device),
                    torch.zeros(2 * batch_size, dtype=torch.long, device=q_device),
                ]
            )

        itm_logits = self.qformer.forward_itm(
            image_features=itm_images,
            caption_ids=itm_caption_ids,
            caption_mask=itm_caption_mask,
        )
        itg_logits = self.qformer.forward_itg(
            image_features=image_features,
            caption_ids=caption_ids,
            caption_mask=caption_mask,
        )
        return {
            "itc_query_proj": query_projection,
            "itc_text_proj": text_projection,
            "itm_logits": itm_logits,
            "itm_labels": itm_labels,
            "itg_logits": itg_logits,
            "itc_temperature": self.qformer.itc_temp,
        }

    # ------------------------------------------------------------------
    # Stage-aware trainability
    # ------------------------------------------------------------------

    def _set_qformer_trainability(self, stage: int, tune_stage2: bool) -> None:
        enabled = stage == 1 or tune_stage2
        setter = getattr(self.qformer, "set_trainable", None)
        if callable(setter):
            setter(enabled, respect_frozen_text=True)
        else:
            for parameter in self.qformer.parameters():
                parameter.requires_grad_(enabled)
        for name, parameter in self.qformer.named_parameters():
            trainable = enabled and name not in self._permanently_frozen_qformer
            if stage == 2 and name.startswith(self._STAGE1_ONLY_QFORMER_PREFIXES):
                trainable = False
            parameter.requires_grad_(trainable)
        enforce = getattr(self.qformer, "enforce_freeze_policy", None)
        if callable(enforce):
            enforce()

    def _set_llm_trainability(self, stage: int) -> None:
        if self.llm is None:
            if stage == 2:
                raise RuntimeError("Stage 2 requires a loaded language model")
            return
        if stage == 1:
            for parameter in self.llm.parameters():
                parameter.requires_grad_(False)
            if hasattr(self.llm, "set_adapters_trainable"):
                self.llm.set_adapters_trainable(False)
            self.llm.eval()
            return

        if isinstance(self.llm, LanguageModelDecoder) and not self.llm.lora_applied:
            raise RuntimeError(
                "Stage 2 requires an attached LoRA/QLoRA adapter; base Qwen fine-tuning is forbidden"
            )
        if hasattr(self.llm, "set_adapters_trainable"):
            self.llm.set_adapters_trainable(True)
            if hasattr(self.llm, "assert_base_model_frozen"):
                self.llm.assert_base_model_frozen()
        else:
            # Injected/test decoders have no adapter contract. Never interpret
            # this as permission to fine-tune their full foundation weights.
            for name, parameter in self.llm.named_parameters():
                parameter.requires_grad_("lora_" in name.lower())

    def set_training_stage(self, stage: int, tune_qformer_stage2: bool = True) -> None:
        if stage not in (1, 2):
            raise ValueError("stage must be 1 or 2")
        self.image_encoder.set_train_mode("frozen")
        self._set_qformer_trainability(stage, tune_qformer_stage2)
        for parameter in self.projector.parameters():
            parameter.requires_grad_(stage == 2)
        self._set_llm_trainability(stage)
        self.training_stage = stage
        self.assert_training_policy(stage, tune_qformer_stage2)

    def get_trainable_params_stage1(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.qformer.parameters() if parameter.requires_grad]

    def get_trainable_params_stage2(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def trainable_parameter_summary(self) -> Dict[str, Dict[str, int]]:
        summary: Dict[str, Dict[str, int]] = {}
        for name in ("image_encoder", "qformer", "projector", "llm"):
            module = getattr(self, name)
            parameters = [] if module is None else list(module.parameters())
            summary[name] = {
                "trainable": sum(p.numel() for p in parameters if p.requires_grad),
                "total": sum(p.numel() for p in parameters),
            }
        return summary

    def assert_training_policy(self, stage: int, tune_qformer_stage2: bool = True) -> None:
        summary = self.trainable_parameter_summary()
        if summary["image_encoder"]["trainable"]:
            raise RuntimeError("The vision encoder must remain frozen in both stages")
        qformer_should_train = stage == 1 or tune_qformer_stage2
        if bool(summary["qformer"]["trainable"]) != qformer_should_train:
            raise RuntimeError("Q-Former freeze policy does not match the selected stage")
        if bool(summary["projector"]["trainable"]) != (stage == 2):
            raise RuntimeError("Projector freeze policy does not match the selected stage")
        if stage == 1 and summary["llm"]["trainable"]:
            raise RuntimeError("The language model must be frozen during Stage 1")
        if stage == 2 and self.llm is not None:
            leaked = [
                name
                for name, parameter in self.llm.named_parameters()
                if parameter.requires_grad and "lora_" not in name.lower()
            ]
            if leaked:
                raise RuntimeError(f"Non-adapter LLM parameters are trainable: {leaked[:5]}")

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen components must not reactivate dropout/statistics when the
        # containing model enters training mode.
        self.image_encoder.eval()
        if self.training_stage == 1:
            self.projector.eval()
            if self.llm is not None:
                self.llm.eval()
        return self