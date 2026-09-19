import io
import json
import math
import os
import sys
import tarfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import zstandard as zstd
from PIL import Image
from transformers import DINOv3ViTImageProcessor

if not __package__: sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src_src import config_model as cfg
from src_src.vision_encoder import ImageEncoder

def prepare_dino_input(
    image_path: str | Path,
    encoder: ImageEncoder,
    processor: DINOv3ViTImageProcessor,
    max_long_side: int,
) -> tuple[torch.Tensor, Image.Image, dict]:
    if max_long_side <= 0:
        raise ValueError("max_long_side must be positive.")
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    original_width, original_height = image.size
    scale = max_long_side / max(original_width, original_height)
    resized_width, resized_height = max(1, round(original_width * scale)), max(1, round(original_height * scale))
    resized = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
    patch_size = encoder.model.config.patch_size
    padded_width = math.ceil(resized_width / patch_size) * patch_size
    padded_height = math.ceil(resized_height / patch_size) * patch_size
    pad_left, pad_top = (padded_width - resized_width) // 2, (padded_height - resized_height) // 2
    pad_right, pad_bottom = padded_width - resized_width - pad_left, padded_height - resized_height - pad_top
    canvas = np.empty((padded_height, padded_width, 3), dtype=np.float32)
    canvas[:] = np.asarray(processor.image_mean, dtype=np.float32)
    canvas[pad_top:pad_top + resized_height, pad_left:pad_left + resized_width] = np.asarray(resized, dtype=np.float32) / 255.0
    pixel_values = torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(0).contiguous()
    mean = pixel_values.new_tensor(processor.image_mean).view(1, 3, 1, 1)
    std = pixel_values.new_tensor(processor.image_std).view(1, 3, 1, 1)
    pixel_values = ((pixel_values - mean) / std).to(next(encoder.model.parameters()))
    metadata = {
        "source": str(image_path),
        "original_hw": [original_height, original_width],
        "resized_hw": [resized_height, resized_width],
        "input_hw": [padded_height, padded_width],
        "padding": {"top": pad_top, "bottom": pad_bottom, "left": pad_left, "right": pad_right},
    }
    return pixel_values, image, metadata

@torch.no_grad()
def extract_patch_features(encoder: ImageEncoder, pixel_values: torch.Tensor) -> torch.Tensor:
    # Dense visualization/storage needs all patches, not ImageEncoder's single token.
    features = encoder.model(pixel_values).last_hidden_state
    return features[:, 1 + encoder.model.config.num_register_tokens:, :].float()

def find_images(folder_path: str | Path) -> list[Path]:
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")
    images = sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in cfg.IMAGE_EXTENSIONS)
    if not images:
        raise RuntimeError(f"No images found in {folder}")
    return images

def add_tar_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mtime = len(payload), 0
    archive.addfile(info, io.BytesIO(payload))


def benchmark_folder_encode_storage(
    encoder: ImageEncoder,
    processor: DINOv3ViTImageProcessor,
    folder_path: str | Path,
    output_path: str | Path,
    num_images: int,
    max_long_side: int,
    compression_level: int,
) -> dict:
    if num_images <= 0:
        raise ValueError("num_images must be positive.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_paths = find_images(folder_path)
    metadata = {
        "format": "DINOv3 dense feature archive",
        "dtype": "float32",
        "model_id": encoder.model_id,
        "return_layer": -1,
        "max_long_side": max_long_side,
        "compression": {"algorithm": "zstandard", "level": compression_level},
        "images": [],
        "failed_images": [],
    }
    raw_bytes, encoded_count = 0, 0
    patch_counts: list[int] = []
    started_at = time.perf_counter()
    temporary_path = Path(f"{output_path}.tmp")
    temporary_path.unlink(missing_ok=True)
    compressor = zstd.ZstdCompressor(level=compression_level, threads=-1)
    try:
        with temporary_path.open("wb") as file, compressor.stream_writer(file, closefd=False) as stream, tarfile.open(fileobj=stream, mode="w|") as archive:
            for image_path in image_paths:
                if encoded_count >= num_images:
                    break
                try:
                    pixel_values, _, image_metadata = prepare_dino_input(image_path, encoder, processor, max_long_side)
                    feature = extract_patch_features(encoder, pixel_values)[0].cpu().numpy()
                    buffer = io.BytesIO()
                    np.save(buffer, feature, allow_pickle=False)
                    feature_name = f"features/{encoded_count:06d}.npy"
                    add_tar_bytes(archive, feature_name, buffer.getvalue())
                    feature_bytes = feature.nbytes
                    image_metadata.update({"index": encoded_count, "archive_feature": feature_name, "feature_shape": list(feature.shape), "feature_raw_bytes": feature_bytes})
                    metadata["images"].append(image_metadata)
                    raw_bytes += feature_bytes
                    patch_counts.append(feature.shape[0])
                    encoded_count += 1
                    print(f"\rEncoded {encoded_count:3d}/{num_images} | {image_path.name}", end="", flush=True)
                except Exception as error:
                    metadata["failed_images"].append({"source": str(image_path), "error": repr(error)})
                    print(f"\n[SKIP] {image_path}: {error}")
            metadata["num_encoded"] = encoded_count
            metadata["num_failed"] = len(metadata["failed_images"])
            add_tar_bytes(archive, "metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False).encode())
        if encoded_count < num_images:
            raise RuntimeError(f"Only {encoded_count} valid images were encoded, but {num_images} were requested.")
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    elapsed = time.perf_counter() - started_at
    compressed_bytes = output_path.stat().st_size
    raw_per_image, compressed_per_image = raw_bytes / encoded_count, compressed_bytes / encoded_count
    stats = {
        "num_images": encoded_count,
        "failed_images": len(metadata["failed_images"]),
        "raw_bytes": raw_bytes,
        "compressed_bytes": compressed_bytes,
        "compression_ratio": raw_bytes / compressed_bytes if compressed_bytes else float("inf"),
        "raw_bytes_per_image": raw_per_image,
        "compressed_bytes_per_image": compressed_per_image,
        "estimated_1m_bytes": compressed_per_image * 1_000_000,
        "elapsed_seconds": elapsed,
        "images_per_second": encoded_count / elapsed,
        "output_path": str(output_path),
    }
    mib, gib = 1024**2, 1024**3
    print(
        f"\n{'=' * 70}\nStorage result\n{'=' * 70}\n"
        f"Images encoded      : {encoded_count}\n"
        f"Failed images       : {stats['failed_images']}\n"
        f"Raw Float32 size    : {raw_bytes / mib:.3f} MiB\n"
        f"Compressed size     : {compressed_bytes / mib:.3f} MiB\n"
        f"Compression ratio   : {stats['compression_ratio']:.2f}x\n"
        f"Raw / image         : {raw_per_image / mib:.3f} MiB\n"
        f"Compressed / image  : {compressed_per_image / mib:.3f} MiB\n"
        f"Patches min/max/avg : {min(patch_counts)} / {max(patch_counts)} / {np.mean(patch_counts):.2f}\n"
        f"Elapsed             : {elapsed:.2f} s\n"
        f"Throughput          : {stats['images_per_second']:.2f} images/s\n"
        f"Estimated 1M images : {stats['estimated_1m_bytes'] / gib:.2f} GiB\n"
        f"Output              : {output_path}"
    )
    return stats

def save_pca_visualization(
    encoder: ImageEncoder,
    processor: DINOv3ViTImageProcessor,
    image_path: str | Path,
    output_path: str | Path,
    max_long_side: int,
) -> None:
    pixel_values, image, metadata = prepare_dino_input(image_path, encoder, processor, max_long_side)
    features = extract_patch_features(encoder, pixel_values)[0]
    original_width, original_height = image.size
    resized_height, resized_width = metadata["resized_hw"]
    padded_height, padded_width = metadata["input_hw"]
    pad_top, pad_left = metadata["padding"]["top"], metadata["padding"]["left"]
    grid_height, grid_width = encoder.compute_grid_shape(padded_height, padded_width)
    centered = features - features.mean(dim=0, keepdim=True)
    _, _, vectors = torch.pca_lowrank(centered, q=3, center=False)
    rgb = centered @ vectors[:, :3]
    low, high = torch.quantile(rgb, 0.02, dim=0, keepdim=True), torch.quantile(rgb, 0.98, dim=0, keepdim=True)
    rgb = ((rgb - low) / (high - low + 1e-6)).clamp(0, 1)
    rgb = rgb.reshape(grid_height, grid_width, 3).permute(2, 0, 1).unsqueeze(0)
    rgb = F.interpolate(rgb, size=(padded_height, padded_width), mode="bilinear", align_corners=False)
    rgb = rgb[:, :, pad_top:pad_top + resized_height, pad_left:pad_left + resized_width]
    rgb = F.interpolate(rgb, size=(original_height, original_width), mode="bilinear", align_corners=False)[0].permute(1, 2, 0).cpu().numpy()
    original = np.asarray(image, dtype=np.float32) / 255.0
    overlay = np.clip(0.35 * original + 0.65 * rgb, 0, 1)
    figure, axes = plt.subplots(1, 3, figsize=(15, 6))
    for axis, data, title in zip(axes, (original, rgb, overlay), ("Input", "DINO PCA Features", "Overlay")):
        axis.imshow(data)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    # figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(
        f"\n{'=' * 70}\nDINOv3 demo image\n{'=' * 70}\n"
        f"Image              : {image_path}\n"
        f"Original           : {(original_height, original_width)}\n"
        f"Resized            : {(resized_height, resized_width)}\n"
        f"Model input        : {(padded_height, padded_width)}\n"
        f"Patch grid         : {(grid_height, grid_width)}\n"
        f"Num patches        : {grid_height * grid_width}\n"
        f"Feature shape      : {tuple(features.shape)}\n"
        f"Feature raw size   : {features.numel() * 4 / 1024**2:.3f} MiB"
    )

def main() -> None:
    image_folder = cfg.FEATURE_IMAGE_FOLDER
    max_long_side = cfg.FEATURE_MAX_LONG_SIDE
    pca_output_path = cfg.FEATURE_PCA_OUTPUT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ImageEncoder().to(device).eval()
    processor = DINOv3ViTImageProcessor.from_pretrained(cfg.IMAGE_ENCODER_MODEL_ID)
    demo_image = find_images(image_folder)[0]
    print(f"Demo image: {demo_image}")
    save_pca_visualization(encoder, processor, demo_image, pca_output_path, max_long_side)
    print(f"Total parameters: {encoder.count_parameters() / 1e6:.2f}M")   

if __name__ == "__main__":
    main()
