"""Image preprocessing, prompt construction, checkpoint loading, and generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoTokenizer

from src.checkpointing import load_bridge_checkpoint
from src.language_model import LLMTrainMode, LoRAConfig
from src.model import InABridgeModel
from src.training.config import TrainingConfig


class InABridgeInferenceEngine:
    def __init__(
        self,
        model: InABridgeModel,
        image_processor: Any,
        qformer_tokenizer: Any,
        system_prompt: Optional[str] = None,
        enable_thinking: bool = False,
        max_instruction_length: int = 128,
        max_text_length: int = 512,
    ) -> None:
        if model.llm is None:
            raise ValueError("Inference requires a loaded language model")
        self.model = model.eval()
        self.image_processor = image_processor
        self.qformer_tokenizer = qformer_tokenizer
        self.tokenizer = model.llm.tokenizer
        self.system_prompt = system_prompt
        self.enable_thinking = enable_thinking
        self.max_instruction_length = max_instruction_length
        self.max_text_length = max_text_length

    @staticmethod
    def _load_image(image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return ImageOps.exif_transpose(image).convert("RGB")
        with Image.open(Path(image).expanduser()) as loaded:
            return ImageOps.exif_transpose(loaded).convert("RGB").copy()

    def _prompt(self, instruction: str) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": instruction})
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            return self.tokenizer.apply_chat_template(
                messages, enable_thinking=self.enable_thinking, **kwargs
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    @torch.inference_mode()
    def predict(
        self,
        image: Any,
        instruction: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
    ) -> str:
        if not instruction.strip():
            raise ValueError("instruction cannot be empty")
        pixels = self.image_processor(
            images=[self._load_image(image)], return_tensors="pt"
        )["pixel_values"]
        q_tokens = self.qformer_tokenizer(
            [instruction],
            padding=True,
            truncation=True,
            max_length=self.max_instruction_length,
            return_tensors="pt",
        )
        prompt_tokens = self.tokenizer(
            [self._prompt(instruction)],
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        generated = self.model.generate(
            pixel_values=pixels.to(self.model.image_encoder.device),
            instruction_ids=q_tokens["input_ids"],
            instruction_mask=q_tokens["attention_mask"],
            input_ids=prompt_tokens["input_ids"],
            attention_mask=prompt_tokens["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
        )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


def load_engine_from_checkpoint(
    checkpoint: str | Path,
    *,
    device_map: Any = "auto",
) -> InABridgeInferenceEngine:
    """Reconstruct an engine from a trainer checkpoint directory."""
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    directory = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
    bridge_path = directory / "bridge_model.pt" if checkpoint_path.is_dir() else checkpoint_path
    config_path = directory / "training_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing training_config.json next to {bridge_path}")
    config = TrainingConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
    if config.stage != 2:
        raise ValueError("Inference requires a Stage-2 checkpoint")
    mode = LLMTrainMode(config.model.llm_train_mode)
    lora = None
    if mode in (LLMTrainMode.LORA, LLMTrainMode.QLORA):
        lora = LoRAConfig(
            r=config.model.lora_rank,
            alpha=config.model.lora_alpha,
            dropout=config.model.lora_dropout,
        )
    model = InABridgeModel(
        image_encoder_id=config.model.image_encoder_id,
        llm_model_id=config.model.llm_model_id,
        llm_train_mode=mode,
        lora_config=lora,
        llm_device_map=device_map,
    )
    _, unexpected = load_bridge_checkpoint(
        model,
        bridge_path,
        allow_llm_adapter_keys=False,
        required_prefixes=("qformer.", "projector."),
    )
    if unexpected:
        raise ValueError(f"Unexpected checkpoint keys: {unexpected}")
    embedding_parameter = next(model.llm.get_input_embeddings().parameters())
    bridge_device = embedding_parameter.device
    if bridge_device.type == "cuda":
        bridge_dtype = embedding_parameter.dtype
        model.image_encoder.to(device=bridge_device, dtype=bridge_dtype)
        model.qformer.to(device=bridge_device, dtype=bridge_dtype)
        model.projector.to(device=bridge_device, dtype=bridge_dtype)
    processor = AutoImageProcessor.from_pretrained(
        config.model.image_encoder_id,
        size={"height": config.model.image_size, "width": config.model.image_size},
        do_center_crop=config.model.center_crop,
    )
    q_tokenizer = AutoTokenizer.from_pretrained(config.model.qformer_tokenizer_id, use_fast=True)
    return InABridgeInferenceEngine(
        model,
        processor,
        q_tokenizer,
        system_prompt=config.system_prompt,
        enable_thinking=config.enable_thinking,
        max_instruction_length=config.data.max_instruction_length,
        max_text_length=config.data.max_text_length,
    )
