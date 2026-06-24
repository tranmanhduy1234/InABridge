# -*- coding: utf-8 -*-
"""
HUẤN LUYỆN FSDP / ZeRO-3 — ToyVLMBridge (Đã cập nhật PyTorch DCP chuẩn 2.x+)
========================================================================
Triển khai Fully Sharded Data Parallel (ZeRO-3) kết hợp Distributed Checkpoint.
"""

import os
import time
import argparse
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,          
    BackwardPrefetch,          
    MixedPrecision,            
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

# --- Import gói Distributed Checkpoint (DCP) mới ---
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict


# ─────────────────────────────────────────────
# 0. THIẾT LẬP PROCESS GROUP
# ─────────────────────────────────────────────

def setup_distributed():
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    return rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


# ─────────────────────────────────────────────
# 1. MÔ HÌNH
# ─────────────────────────────────────────────

class ToyVLMBridge(nn.Module):
    def __init__(self, in_features: int = 1152, hidden_dim: int = 768, vocab_size: int = 32000):
        super().__init__()
        self.projection         = nn.Linear(in_features, hidden_dim)
        self.cross_attention_sim = nn.Linear(hidden_dim,  hidden_dim)
        self.llm_head           = nn.Linear(hidden_dim,  vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.projection(x))           
        x = F.gelu(self.cross_attention_sim(x))  
        return self.llm_head(x)                  


# ─────────────────────────────────────────────
# 2. WRAP FSDP (ZeRO-3)
# ─────────────────────────────────────────────

def wrap_fsdp(model: nn.Module, rank: int) -> FSDP:
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")

    mp_policy = None
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,   
            reduce_dtype=torch.float32,   
            buffer_dtype=torch.bfloat16,
        )

    min_params = 100_000
    auto_wrap = functools.partial(size_based_auto_wrap_policy, min_num_params=min_params)

    wrapped = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,   
        auto_wrap_policy=auto_wrap,
        mixed_precision=mp_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  
        device_id=device if torch.cuda.is_available() else None,
        sync_module_states=True,
        forward_prefetch=True,
    )
    return wrapped


# ─────────────────────────────────────────────
# 3. DỮ LIỆU GIẢ LẬP
# ─────────────────────────────────────────────

def make_dummy_dataset(n_samples: int = 1024, in_features: int = 1152, vocab_size: int = 32000) -> TensorDataset:
    X = torch.randn(n_samples, in_features)
    y = torch.randint(0, vocab_size, (n_samples,))
    return TensorDataset(X, y)

def make_dataloader(dataset: TensorDataset, batch_size: int, rank: int, world_size: int, num_workers: int = 2) -> DataLoader:
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

# ─────────────────────────────────────────────
# 4. OPTIMIZER & LR SCHEDULER
# ─────────────────────────────────────────────

def build_optimizer(model: nn.Module, lr: float = 1e-4, weight_decay: float = 0.01):
    decay_params     = [p for n, p in model.named_parameters() if "bias" not in n and p.requires_grad]
    no_decay_params  = [p for n, p in model.named_parameters() if "bias"     in n and p.requires_grad]

    param_groups = [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))


def build_scheduler(optimizer, num_warmup_steps: int, num_total_steps: int):
    import math
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_total_steps - num_warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────
# 5. NEW: SAVE / LOAD CHECKPOINT DÙNG PYTORCH DCP
# ─────────────────────────────────────────────

def save_checkpoint(model: nn.Module, optimizer, epoch: int, rank: int, is_fsdp: bool, save_dir: str = "./checkpoints"):
    """
    Sử dụng Distributed Checkpoint (DCP) để ghi trạng thái phân mảnh 
    của cả Model và Optimizer trực tiếp và song song xuống đĩa cứng.
    """
    # Tạo một thư mục riêng cho epoch này (DCP ghi nhiều file phân mảnh vào 1 folder)
    ckpt_dir = os.path.join(save_dir, f"epoch_{epoch:03d}")
    if is_main(rank):
        os.makedirs(ckpt_dir, exist_ok=True)
    
    if is_fsdp:
        dist.barrier() # Đảm bảo thư mục đã được tạo xong trên mọi rank

    # 1. Thu thập state_dict dạng phân mảnh (Sharded) thông qua hàm thống nhất toàn cục của PyTorch 2.x
    state_dict = {"model": model, "optimizer": optimizer}
    
    if is_fsdp:
        # get_state_dict sẽ tự động bóc tách cấu trúc song song / phân mảnh của FSDP
        model_state, optim_state = get_state_dict(model, optimizer)
        state_dict = {"model": model_state, "optimizer": optim_state}
        
        # 2. Tiến hành ghi song song từ tất cả các rank vào thư mục chỉ định
        dcp.save(state_dict=state_dict, checkpoint_id=ckpt_dir)
        dist.barrier()
    else:
        # Nếu chạy single process truyền thống, lưu file đơn chuẩn PyTorch
        torch.save({
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict()
        }, os.path.join(ckpt_dir, "single_node.pt"))

    if is_main(rank):
        print(f"[Rank 0] Checkpoint (DCP) saved successfully to → {ckpt_dir}")


def load_checkpoint(model: nn.Module, optimizer, ckpt_dir: str, rank: int, is_fsdp: bool):
    """
    Đọc dữ liệu phân mảnh từ thư mục Checkpoint và phân phối lại vào Model/Optimizer hiện tại.
    """
    if not os.path.exists(ckpt_dir):
        if is_main(rank):
            print(f"[Rank 0] Thư mục Checkpoint không tồn tại: {ckpt_dir}")
        return 0

    if is_fsdp:
        # 1. Tạo cấu trúc trạng thái rỗng từ mô hình hiện tại để làm khuôn nạp dữ liệu
        model_state, optim_state = get_state_dict(model, optimizer)
        state_dict = {"model": model_state, "optimizer": optim_state}
        
        # 2. Đọc trực tiếp và map các mảnh trọng số từ đĩa cứng vào cấu trúc tương ứng
        dcp.load(state_dict=state_dict, checkpoint_id=ckpt_dir)
        
        # 3. Đẩy ngược trạng thái vừa nạp trở lại các module của FSDP
        set_state_dict(model, optimizer, model_state_dict=state_dict["model"], optim_state_dict=state_dict["optimizer"])
    else:
        # Chế độ Single node nạp file đơn
        single_pt = os.path.join(ckpt_dir, "single_node.pt")
        if os.path.exists(single_pt):
            ckpt = torch.load(single_pt, map_location="cpu")
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optim_state"])
            
    # Lấy thông tin Epoch từ tên thư mục (ví dụ: 'epoch_002' -> epoch = 2, start_epoch = 3)
    try:
        epoch_num = int(os.path.basename(ckpt_dir).split("_")[1])
        start_epoch = epoch_num + 1
    except Exception:
        start_epoch = 0

    if is_main(rank):
        print(f"[Rank 0] Resumed thành công từ DCP folder: {ckpt_dir} → bắt đầu Epoch {start_epoch}")
    
    return start_epoch


# ─────────────────────────────────────────────
# 6. VÒNG LẶP TRAIN / EVAL
# ─────────────────────────────────────────────

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    device:    torch.device,
    rank:      int,
    epoch:     int,
    grad_clip: float = 1.0,
    log_every: int   = 20,
) -> float:
    model.train()
    total_loss = 0.0
    t0 = time.time()

    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        logits = model(x)                  
        loss   = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()

        if isinstance(model, FSDP):
            model.clip_grad_norm_(grad_clip)
        else:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if is_main(rank) and (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            avg_loss = total_loss / (step + 1)
            lr_now   = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch:3d} | Step {step+1:4d}/{len(loader)} "
                f"| Loss {avg_loss:.4f} | LR {lr_now:.2e} | {elapsed:.1f}s"
            )

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    is_fsdp:   bool,
) -> float:
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for x, y in loader:
            x, y   = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item()

    avg_local_loss = total_loss / max(1, len(loader))

    if is_fsdp:
        loss_tensor = torch.tensor(avg_local_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        return loss_tensor.item()
    
    return avg_local_loss


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",       type=int,   default=5)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int,   default=50)
    p.add_argument("--n_samples",    type=int,   default=2048)
    p.add_argument("--in_features",  type=int,   default=1152)
    p.add_argument("--hidden_dim",   type=int,   default=768)
    p.add_argument("--vocab_size",   type=int,   default=32000)
    p.add_argument("--save_dir",     type=str,   default="./checkpoints")
    p.add_argument("--resume",       type=str,   default=None, help="Đường dẫn tới FOLDER epoch_xxx")
    p.add_argument("--log_every",    type=int,   default=10)
    p.add_argument("--no_dist",      action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    use_fsdp = not args.no_dist

    if not use_fsdp:
        rank       = 0
        world_size = 1
        device     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[Single process] device={device}")
    else:
        rank, world_size = setup_distributed()
        device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")

    if is_main(rank):
        print("=" * 60)
        print(f"  FSDP ZeRO-3 + Distributed Checkpoint (DCP) Training")
        print(f"  World size : {world_size}")
        print(f"  Device     : {device}")
        print("=" * 60)

    dataset = make_dummy_dataset(args.n_samples, args.in_features, args.vocab_size)
    if not use_fsdp:
        loader_train = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        loader_val   = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    else:
        loader_train = make_dataloader(dataset, args.batch_size, rank, world_size)
        loader_val   = make_dataloader(dataset, args.batch_size, rank, world_size)

    model = ToyVLMBridge(args.in_features, args.hidden_dim, args.vocab_size)

    if not use_fsdp:
        model = model.to(device)
    else:
        model = wrap_fsdp(model, rank)

    optimizer   = build_optimizer(model, args.lr, args.weight_decay)
    total_steps = args.epochs * len(loader_train)
    scheduler   = build_scheduler(optimizer, args.warmup_steps, total_steps)
    criterion   = nn.CrossEntropyLoss()

    # ── Nạp Checkpoint từ Thư mục DCP ──
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume, rank, use_fsdp)

    best_val_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        if use_fsdp:
            loader_train.sampler.set_epoch(epoch)  

        t_epoch = time.time()

        train_loss = train_one_epoch(
            model, loader_train, optimizer, scheduler, criterion,
            device, rank, epoch,
            grad_clip=args.grad_clip,
            log_every=args.log_every,
        )

        val_loss = evaluate(model, loader_val, criterion, device, use_fsdp)

        if is_main(rank):
            epoch_time = time.time() - t_epoch
            print(
                f"\n>>> Epoch {epoch:3d}/{args.epochs - 1} done | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Time: {epoch_time:.1f}s\n"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, rank, use_fsdp, args.save_dir)
        
        if use_fsdp:
            dist.barrier()

    if is_main(rank):
        print(f"\n✓ Training hoàn tất. Best Val Loss: {best_val_loss:.4f}")

    if use_fsdp:
        cleanup_distributed()

if __name__ == "__main__":
    main()