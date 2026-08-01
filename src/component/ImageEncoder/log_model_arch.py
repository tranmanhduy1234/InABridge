import os
import argparse
import torch
from transformers import AutoModel

def export_model_log(model_id: str, output_log: str):
    print(f"Đang tải mô hình '{model_id}'...")
    model = AutoModel.from_pretrained(model_id)

    print(f"Đang xuất cấu trúc ra file '{output_log}'...")
    with open(output_log, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write(f"DETAILED ARCHITECTURE LOG FOR: {model_id}\n")
        f.write(f"Model Class: {model.__class__.__name__}\n")
        f.write("=" * 90 + "\n\n")

        # 1. Cấu trúc cây
        f.write("[1. FULL PYTORCH REPRESENTATION]\n")
        f.write("-" * 90 + "\n")
        f.write(str(model) + "\n\n")

        # 2. Danh sách Named Modules
        f.write("[2. NAMED MODULES LIST]\n")
        f.write(f"{'Path':<65} | {'Module Type'}\n")
        f.write("-" * 90 + "\n")
        for name, module in model.named_modules():
            path_str = name if name else "(root)"
            f.write(f"{path_str:<65} | {module.__class__.__name__}\n")
        f.write("\n")

        # 3. Danh sách Tham số
        f.write("[3. PARAMETERS DETAILS]\n")
        f.write(f"{'Parameter Name':<65} | {'Shape'}\n")
        f.write("-" * 90 + "\n")
        total_p = 0
        for name, p in model.named_parameters():
            f.write(f"{name:<65} | {list(p.shape)}\n")
            total_p += p.numel()

        f.write("-" * 90 + "\n")
        f.write(f"Total Parameters: {total_p:,}\n")
        f.write("=" * 90 + "\n")

    print(f"✅ Hoàn tất! Bạn có thể mở file '{output_log}' để kiểm tra chính xác tên các layer.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--output", type=str, default="dinov3_arch.log")
    args = parser.parse_args()

    export_model_log(args.model_id, args.output)