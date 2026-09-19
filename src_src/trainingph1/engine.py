import argparse
from pathlib import Path

import torch
from transformers import AutoImageProcessor

from src_src import config_model as cfg
from src_src.data.dataloaderStage1 import VLMDatasetStage1, build_dataloader, build_transform
from src_src.losses.lossStage1 import calculate_loss_ph1
from src_src.queueMoco.ema import EMA
from src_src.queueMoco.moco import MoCoQueue
from src_src.trainingph1.model import ModelStage1

def train_epoch(model, momentum_model, queue, dataloader, optimizer, scaler, device, epoch):
    model.train()
    momentum_model.eval()
    totals = dict.fromkeys(("loss", "loss_itc", "loss_itm", "loss_itg"), 0.0)
    samples = 0
    amp = device.type == "cuda" and cfg.COMPUTE_TYPE in (torch.float16, torch.bfloat16)
    for step, batch in enumerate(dataloader, 1):
        images = batch["images"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        distill_weight = cfg.DISTILL_WEIGHT
        if epoch == 1:
            distill_weight *= (step - 1) / max(1, len(dataloader) - 1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=cfg.COMPUTE_TYPE, enabled=amp):
            losses = calculate_loss_ph1(
                model, momentum_model.model, queue, images, input_ids, attention_mask,
                distill_weight=distill_weight, update_queue=False,
            )
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {step}")
        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], cfg.MAX_GRAD_NORM,
            error_if_nonfinite=not scaler.is_enabled(),
        )
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < previous_scale:
            print(f"Epoch {epoch}, batch {step}: skipped optimizer step (gradient overflow)")
            continue
        momentum_model.update(model)
        queue.enqueue(losses["image_keys"], losses["text_keys"])
        samples += images.size(0)
        for name in totals:
            totals[name] += losses[name].detach().item() * images.size(0)
        if step % cfg.LOG_EVERY == 0:
            metrics = " ".join(f"{name}={value / samples:.4f}" for name, value in totals.items())
            print(f"Epoch {epoch}, batch {step}/{len(dataloader)}: {metrics} distill_weight={distill_weight:.4f}")
    if samples == 0:
        raise RuntimeError("No successful training batches; check dataset size, batch size, and gradients")
    return {name: value / samples for name, value in totals.items()}


def train(json_path, image_dir, *, epochs=cfg.TRAIN_EPOCHS, batch_size=cfg.BATCH_SIZE,
          num_workers=cfg.NUM_WORKERS, output_dir=cfg.TRAIN_OUTPUT_DIR):
    if epochs < 1 or batch_size < 2 or num_workers < 0:
        raise ValueError("epochs must be positive, batch_size >= 2, and num_workers >= 0")
    if not Path(json_path).is_file() or not Path(image_dir).is_dir():
        raise FileNotFoundError("Expected a JSON manifest and an existing image directory")
    torch.manual_seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelStage1().to(device)
    model.vision_encoder.compute_grid_shape(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
    processor = AutoImageProcessor.from_pretrained(model.vision_encoder.model_id)
    dataset = VLMDatasetStage1(
        json_path, image_dir,
        transform=build_transform(normalization=(processor.image_mean, processor.image_std)),
    )
    dataloader = build_dataloader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        drop_last=True,
    )
    if len(dataloader) == 0:
        raise ValueError("Dataset must contain at least batch_size samples")
    momentum_model = EMA(model).to(device)
    queue = MoCoQueue(cfg.QUEUE_SIZE, model.qformer_model.num_queries,
                      model.itc_encoder.query_proj.out_features).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg.COMPUTE_TYPE == torch.float16)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Stage 1: device={device}, samples={len(dataloader.dataset)}, batches={len(dataloader)}")
    for epoch in range(1, epochs + 1):
        metrics = train_epoch(model, momentum_model, queue, dataloader, optimizer, scaler, device, epoch)
        print(f"Epoch {epoch}/{epochs}: {metrics}")
        # Frozen vision weights can be reloaded from model_id.
        checkpoint = {
            "epoch": epoch, "metrics": metrics,
            "model": {k: v for k, v in model.state_dict().items() if not k.startswith("vision_encoder.")},
            "ema": {k: v for k, v in momentum_model.state_dict().items() if not k.startswith("model.vision_encoder.")},
            "queue": queue.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "model_config": {
                "model_id": model.vision_encoder.model_id, "return_layer": model.vision_encoder.return_layer,
                "bert_name": cfg.QFORMER_MODEL_ID, "num_queries": model.qformer_model.num_queries,
                "cross_every": cfg.CROSS_ATTN_EVERY, "itc_dim": model.itc_encoder.query_proj.out_features,
            },
            "training_config": {
                "image_size": cfg.IMAGE_SIZE, "max_length": cfg.MAX_INSTRUCTION_LENGTH,
                "batch_size": batch_size, "momentum": cfg.MOMENTUM,
                "temperature": cfg.ITC_TEMPERATURE, "distill_weight": cfg.DISTILL_WEIGHT,
                "distill_warmup_epochs": 1,
                "seed": cfg.SEED, "learning_rate": cfg.LEARNING_RATE,
                "weight_decay": cfg.WEIGHT_DECAY, "max_grad_norm": cfg.MAX_GRAD_NORM,
                "compute_type": str(cfg.COMPUTE_TYPE),
            },
        }
        temporary = output_dir / "last.pt.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(output_dir / "last.pt")
    return model

def main():
    parser = argparse.ArgumentParser(description="Train the stage-1 VLM with ITC, ITM and ITG")
    parser.add_argument("--json-path", default=cfg.TRAIN_JSON_PATH, required=cfg.TRAIN_JSON_PATH is None)
    parser.add_argument("--image-dir", default=cfg.TRAIN_IMAGE_DIR, required=cfg.TRAIN_IMAGE_DIR is None)
    parser.add_argument("--output-dir", default=argparse.SUPPRESS)
    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num-workers", type=int, default=argparse.SUPPRESS)
    train(**vars(parser.parse_args()))

if __name__ == "__main__":
    main()
