"""Command-line entry point for reproducible InA-Bridge training."""

from __future__ import annotations

import argparse
import logging
import os

import torch
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoImageProcessor, AutoTokenizer

from src.data import (
    InABridgeCollator,
    SquareRootMixtureSampler,
    Stage1Collator,
    VisionLanguageManifestDataset,
)
from src.checkpointing import load_bridge_checkpoint
from src.language_model import LLMTrainMode, LoRAConfig
from src.model import InABridgeModel
from src.training.config import TrainingConfig
from src.training.trainer import InABridgeTrainer


def _dataset(config: TrainingConfig, manifest: str) -> VisionLanguageManifestDataset:
    data = config.data
    return VisionLanguageManifestDataset(
        manifest=manifest,
        image_root=data.image_root,
        image_column=data.image_column,
        text_column=data.text_column,
        instruction_column=data.instruction_column,
        answer_column=data.answer_column,
        stage=config.stage,
        max_samples=data.max_samples,
    )


def _loader(
    dataset,
    collator,
    config: TrainingConfig,
    training: bool,
    sampler=None,
) -> DataLoader:
    workers = config.data.num_workers
    kwargs = {
        "dataset": dataset,
        "batch_size": config.train_batch_size if training else config.eval_batch_size,
        "shuffle": training and sampler is None,
        "sampler": sampler,
        "num_workers": workers,
        "collate_fn": collator,
        "pin_memory": True,
        "drop_last": config.stage == 1,
        "persistent_workers": config.data.persistent_workers and workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = config.data.prefetch_factor
    return DataLoader(**kwargs)


def run(config: TrainingConfig) -> None:
    image_processor = AutoImageProcessor.from_pretrained(
        config.model.image_encoder_id,
        size={"height": config.model.image_size, "width": config.model.image_size},
        do_center_crop=config.model.center_crop,
    )
    qformer_tokenizer = AutoTokenizer.from_pretrained(
        config.model.qformer_tokenizer_id, use_fast=True
    )
    lora = None
    llm_mode = LLMTrainMode(config.model.llm_train_mode)
    if llm_mode in (LLMTrainMode.LORA, LLMTrainMode.QLORA):
        lora = LoRAConfig(
            r=config.model.lora_rank,
            alpha=config.model.lora_alpha,
            dropout=config.model.lora_dropout,
        )
    qlora_device_map = None
    if llm_mode == LLMTrainMode.QLORA:
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA requires a CUDA GPU with bitsandbytes support")
        qlora_device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}
    model = InABridgeModel(
        image_encoder_id=config.model.image_encoder_id,
        llm_model_id=config.model.llm_model_id,
        llm_train_mode=llm_mode,
        lora_config=lora,
        # Stage 1 never executes Qwen, so avoid allocating four billion frozen
        # parameters. Accelerate owns placement for ordinary Stage-2 runs.
        llm_device_map=qlora_device_map,
        load_llm=config.stage == 2,
    )
    if config.model.initialize_qformer_from_pretrained and not config.model.bridge_checkpoint:
        copied = model.qformer.initialize_from_pretrained(config.model.qformer_tokenizer_id)
        logging.getLogger(__name__).info("Initialized %d Q-Former tensors from DeBERTa", copied)
    if config.model.bridge_checkpoint:
        _, unexpected = load_bridge_checkpoint(model, config.model.bridge_checkpoint)
        if unexpected:
            raise ValueError(f"Unexpected keys in bridge checkpoint: {unexpected}")

    if config.stage == 1:
        collator = Stage1Collator(
            image_processor, qformer_tokenizer, config.data.max_instruction_length
        )
        pad_token_id = qformer_tokenizer.pad_token_id
    else:
        collator = InABridgeCollator(
            image_processor=image_processor,
            qformer_tokenizer=qformer_tokenizer,
            llm_tokenizer=model.llm.tokenizer,
            max_instruction_length=config.data.max_instruction_length,
            max_text_length=config.data.max_text_length,
            system_prompt=config.system_prompt,
            enable_thinking=config.enable_thinking,
        )
        pad_token_id = None

    manifest_paths = config.data.train_manifests or [config.data.train_manifest]
    train_datasets = [_dataset(config, path) for path in manifest_paths]
    train_dataset = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)
    sampler = None
    if len(train_datasets) > 1 and config.data.square_root_mixture_sampling:
        sampler = SquareRootMixtureSampler(
            [len(dataset) for dataset in train_datasets], seed=config.seed
        )
    train_loader = _loader(train_dataset, collator, config, True, sampler=sampler)
    validation_loader = None
    if config.data.validation_manifest:
        validation_loader = _loader(
            _dataset(config, config.data.validation_manifest), collator, config, False
        )
    trainer = InABridgeTrainer(
        config,
        model,
        train_loader,
        validation_loader,
        stage1_pad_token_id=pad_token_id,
    )
    trainer.train()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a JSON training config")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run(TrainingConfig.from_json(args.config))


if __name__ == "__main__":
    main()
