"""Accelerate-based, resumable trainer for both InA-Bridge stages."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from accelerate import Accelerator, skip_first_batches
from accelerate.utils import set_seed
from transformers import get_cosine_schedule_with_warmup

from src.model import InABridgeModel
from src.checkpointing import build_portable_state_dict, rotate_checkpoints, write_json_atomic
from src.training.config import TrainingConfig
from src.training.optimizer import build_optimizer
from src.losses import Stage1Loss, Stage2CausalLMLoss

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainerState:
    epoch: int = 0
    batch_in_epoch: int = 0
    global_step: int = 0
    best_validation_loss: float = math.inf


class InABridgeTrainer:
    def __init__(
        self,
        config: TrainingConfig,
        model: InABridgeModel,
        train_dataloader: Iterable[Dict[str, torch.Tensor]],
        validation_dataloader: Optional[Iterable[Dict[str, torch.Tensor]]] = None,
        accelerator: Optional[Accelerator] = None,
        stage1_pad_token_id: Optional[int] = None,
    ) -> None:
        config.validate()
        self.config = config
        self.accelerator = accelerator or Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
            log_with="tensorboard",
            project_dir=config.output_dir,
        )
        set_seed(config.seed, device_specific=True)
        model.set_training_stage(config.stage, config.model.tune_qformer_stage2)
        LOGGER.info("Stage %d parameter policy: %s", config.stage, model.trainable_parameter_summary())
        if config.model.gradient_checkpointing:
            if hasattr(model.qformer, "gradient_checkpointing_enable"):
                model.qformer.gradient_checkpointing_enable()
            if config.stage == 2 and hasattr(model.llm, "enable_gradient_checkpointing"):
                model.llm.enable_gradient_checkpointing()
        self.criterion = None
        if config.stage == 1:
            if stage1_pad_token_id is None:
                raise ValueError("stage1_pad_token_id is required for Stage 1")
            self.criterion = Stage1Loss(
                stage1_pad_token_id,
                itc_weight=config.loss.stage1_itc_weight,
                itm_weight=config.loss.stage1_itm_weight,
                itg_weight=config.loss.stage1_itg_weight,
                itc_label_smoothing=config.loss.stage1_itc_label_smoothing,
                itm_label_smoothing=config.loss.stage1_itm_label_smoothing,
                itg_label_smoothing=config.loss.stage1_itg_label_smoothing,
                reduction=config.loss.reduction,
            )
        else:
            self.criterion = Stage2CausalLMLoss(
                label_smoothing=config.loss.stage2_label_smoothing,
                z_loss_weight=config.loss.stage2_z_loss_weight,
                reduction=config.loss.reduction,
            )

        self.optimizer = build_optimizer(model, config)
        update_steps_per_epoch = math.ceil(
            len(train_dataloader) / config.gradient_accumulation_steps  # type: ignore[arg-type]
        )
        self.total_steps = update_steps_per_epoch * config.epochs
        warmup_steps = int(self.total_steps * config.optimizer.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, self.total_steps
        )

        prepared = [model, self.optimizer, train_dataloader, self.scheduler]
        if validation_dataloader is not None:
            prepared.append(validation_dataloader)
        prepared = list(self.accelerator.prepare(*prepared))
        self.model, self.optimizer, self.train_dataloader, self.scheduler = prepared[:4]
        self.validation_dataloader = prepared[4] if len(prepared) == 5 else None
        self.state = TrainerState()
        self.last_loss_metrics: Dict[str, torch.Tensor] = {}
        self.output_dir = Path(config.output_dir)
        if self.accelerator.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.accelerator.wait_for_everyone()
        self.accelerator.register_save_state_pre_hook(self._save_model_hook)
        self.accelerator.register_load_state_pre_hook(self._load_model_hook)
        self.accelerator.init_trackers("ina-bridge", config=config.to_dict())

    def _save_model_hook(self, models, weights, output_dir: str) -> None:
        if self.accelerator.is_main_process:
            model = self.accelerator.unwrap_model(models[0])
            gathered_state = weights[0] if weights else None
            torch.save(
                build_portable_state_dict(model, gathered_state),
                Path(output_dir) / "bridge_model.pt",
            )
        # Prevent Accelerate from serializing frozen DINO/Qwen foundation weights.
        weights.clear()

    def _load_model_hook(self, models, input_dir: str) -> None:
        path = Path(input_dir) / "bridge_model.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = self.accelerator.unwrap_model(models[0])
        model.load_state_dict(state, strict=False)
        models.clear()

    def _stage1_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        caption_ids = batch["caption_ids"]
        caption_mask = batch["caption_mask"]
        outputs = self.model(
            pixel_values=batch["pixel_values"],
            caption_ids=caption_ids,
            caption_mask=caption_mask,
        )
        labels = outputs["itm_labels"]
        assert self.criterion is not None
        result = self.criterion(outputs, caption_ids, labels, caption_mask=caption_mask)
        self.last_loss_metrics = result.as_dict()
        return result.loss

    def _loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.config.stage == 1:
            return self._stage1_loss(batch)
        output = self.model(**batch, compute_loss_with_llm=False)
        assert isinstance(self.criterion, Stage2CausalLMLoss)
        visual_length = output.logits.size(1) - batch["labels"].size(1)
        result = self.criterion(
            logits=output.logits,
            text_labels=batch["labels"],
            visual_prefix_length=visual_length,
            attention_mask=batch.get("attention_mask"),
        )
        self.last_loss_metrics = result.as_dict()
        return result.loss

    def evaluate(self) -> Optional[float]:
        if self.validation_dataloader is None:
            return None
        self.model.eval()
        metric_values: Dict[str, list[torch.Tensor]] = {}
        with torch.no_grad():
            for batch in self.validation_dataloader:
                self._loss(batch)
                for name, value in self.last_loss_metrics.items():
                    gathered = self.accelerator.gather_for_metrics(
                        value.detach().float().reshape(1)
                    )
                    metric_values.setdefault(name, []).append(gathered)
        self.model.train()
        if not metric_values:
            raise ValueError("Validation dataloader produced no batches")
        metrics = {
            f"validation/{name}": torch.cat(values).mean().item()
            for name, values in metric_values.items()
        }
        value = metrics["validation/loss"]
        self.accelerator.log(metrics, step=self.state.global_step)
        return value

    def _checkpoint_dir(self, step: int) -> Path:
        return self.output_dir / f"checkpoint-{step:08d}"

    def save_checkpoint(self, validation_loss: Optional[float] = None) -> None:
        target = self._checkpoint_dir(self.state.global_step)
        self.accelerator.save_state(str(target), safe_serialization=True)
        if self.accelerator.is_main_process:
            is_best = (
                validation_loss is not None
                and validation_loss < self.state.best_validation_loss
            )
            if validation_loss is not None:
                self.state.best_validation_loss = min(
                    self.state.best_validation_loss, validation_loss
                )
            state_payload = {
                "epoch": self.state.epoch,
                "batch_in_epoch": self.state.batch_in_epoch,
                "global_step": self.state.global_step,
                "best_validation_loss": (
                    self.state.best_validation_loss
                    if math.isfinite(self.state.best_validation_loss)
                    else None
                ),
            }
            write_json_atomic(target / "trainer_state.json", state_payload)
            write_json_atomic(target / "training_config.json", self.config.to_dict())
            best_path = None
            best_metadata = self.output_dir / "best_checkpoint.json"
            if is_best:
                write_json_atomic(
                    best_metadata,
                    {"checkpoint": str(target.resolve()), "validation_loss": validation_loss},
                )
                best_path = target
            elif best_metadata.exists():
                payload = json.loads(best_metadata.read_text(encoding="utf-8"))
                best_path = Path(payload["checkpoint"])
            rotate_checkpoints(
                self.output_dir, self.config.save_total_limit, protected=best_path
            )
        self.accelerator.wait_for_everyone()

    def resume(self, checkpoint: str) -> None:
        path = Path(checkpoint)
        self.accelerator.load_state(str(path))
        state_path = path / "trainer_state.json"
        if state_path.exists():
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            if payload.get("best_validation_loss") is None:
                payload["best_validation_loss"] = math.inf
            self.state = TrainerState(**payload)
        LOGGER.info("Resumed from %s at optimizer step %d", path, self.state.global_step)

    def train(self) -> TrainerState:
        if self.config.resume_from_checkpoint:
            self.resume(self.config.resume_from_checkpoint)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(self.state.epoch, self.config.epochs):
            self.state.epoch = epoch
            if hasattr(self.train_dataloader, "set_epoch"):
                self.train_dataloader.set_epoch(epoch)
            epoch_loader = self.train_dataloader
            skipped = self.state.batch_in_epoch if epoch == self.state.epoch else 0
            if skipped:
                epoch_loader = skip_first_batches(epoch_loader, skipped)
            for batch_index, batch in enumerate(epoch_loader, start=skipped):
                with self.accelerator.accumulate(self.model):
                    loss = self._loss(batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite loss at step {self.state.global_step}: {loss.item()}"
                        )
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.config.optimizer.max_grad_norm
                        )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                if not self.accelerator.sync_gradients:
                    continue
                self.state.global_step += 1
                self.state.batch_in_epoch = batch_index + 1
                if self.state.global_step % self.config.log_every_steps == 0:
                    metrics = {
                        f"train/{name}": self.accelerator.gather(
                            value.detach().float().reshape(1)
                        ).mean().item()
                        for name, value in self.last_loss_metrics.items()
                    }
                    metrics.update(
                        {
                            "train/learning_rate": self.scheduler.get_last_lr()[0],
                            "train/epoch": epoch,
                        }
                    )
                    self.accelerator.log(metrics, step=self.state.global_step)
                validation_loss = None
                if (
                    self.validation_dataloader is not None
                    and self.state.global_step % self.config.eval_every_steps == 0
                ):
                    validation_loss = self.evaluate()
                if self.state.global_step % self.config.save_every_steps == 0:
                    self.save_checkpoint(validation_loss)
            self.state.epoch = epoch + 1
            self.state.batch_in_epoch = 0
            validation_loss = self.evaluate()
            self.save_checkpoint(validation_loss)
        self.accelerator.end_training()
        return self.state
