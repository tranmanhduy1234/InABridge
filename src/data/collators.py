"""Stage-specific batch collation and token masking."""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import torch


@dataclass
class InABridgeCollator:
    """Create DINO pixels and independent Q-Former/Qwen token streams."""

    image_processor: Any
    qformer_tokenizer: Any
    llm_tokenizer: Any
    max_instruction_length: int = 128
    max_text_length: int = 512
    system_prompt: Optional[str] = None
    enable_thinking: bool = False

    def __post_init__(self) -> None:
        self.llm_tokenizer.padding_side = "right"
        self.qformer_tokenizer.padding_side = "right"
        if self.llm_tokenizer.pad_token_id is None:
            if self.llm_tokenizer.eos_token_id is None:
                raise ValueError("The LLM tokenizer needs a pad or EOS token")
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

    def _render(self, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
        kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
        try:
            return self.llm_tokenizer.apply_chat_template(
                messages, enable_thinking=self.enable_thinking, **kwargs
            )
        except TypeError:
            return self.llm_tokenizer.apply_chat_template(messages, **kwargs)

    def __call__(self, examples: List[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        required = {"image", "instruction", "answer"}
        for index, example in enumerate(examples):
            missing = required.difference(example)
            if missing:
                raise KeyError(f"Example {index} is missing fields: {sorted(missing)}")

        pixels = self.image_processor(
            images=[example["image"] for example in examples], return_tensors="pt"
        )["pixel_values"]
        instructions = [str(example["instruction"]) for example in examples]
        q_tokens = self.qformer_tokenizer(
            instructions,
            padding=True,
            truncation=True,
            max_length=self.max_instruction_length,
            return_tensors="pt",
        )

        prompts, full_texts = [], []
        for instruction, example in zip(instructions, examples):
            prefix = []
            if self.system_prompt:
                prefix.append({"role": "system", "content": self.system_prompt})
            prefix.append({"role": "user", "content": instruction})
            prompts.append(self._render(prefix, add_generation_prompt=True))
            full_texts.append(
                self._render(
                    prefix + [{"role": "assistant", "content": str(example["answer"])}],
                    add_generation_prompt=False,
                )
            )

        llm_tokens = self.llm_tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        prompt_ids = self.llm_tokenizer(
            prompts,
            padding=False,
            truncation=True,
            max_length=self.max_text_length,
            add_special_tokens=False,
        )["input_ids"]
        labels = llm_tokens["input_ids"].clone()
        labels[llm_tokens["attention_mask"] == 0] = -100
        for row, ids in enumerate(prompt_ids):
            labels[row, : min(len(ids), labels.size(1))] = -100
            if not torch.any(labels[row] != -100):
                raise ValueError(
                    f"Example {row} has no answer tokens after truncation; "
                    "increase max_text_length or shorten the instruction"
                )
        return {
            "pixel_values": pixels,
            "instruction_ids": q_tokens["input_ids"],
            "instruction_mask": q_tokens["attention_mask"],
            "input_ids": llm_tokens["input_ids"],
            "attention_mask": llm_tokens["attention_mask"],
            "labels": labels,
        }


class Stage1Collator:
    def __init__(self, image_processor: Any, tokenizer: Any, max_length: int = 128) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        if tokenizer.pad_token_id is None:
            raise ValueError("Stage-1 tokenizer must define pad_token_id")

    def __call__(self, examples: List[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        if len(examples) < 2:
            raise ValueError("Stage 1 requires at least two samples per batch")
        pixels = self.image_processor(
            images=[sample["image"] for sample in examples], return_tensors="pt"
        )["pixel_values"]
        tokens = self.tokenizer(
            [str(sample["text"]) for sample in examples],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixels,
            "caption_ids": tokens["input_ids"],
            "caption_mask": tokens["attention_mask"],
        }
