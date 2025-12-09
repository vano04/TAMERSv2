import argparse
import contextlib
import os
import traceback

import torch
import torch.cuda.nvtx as nvtx
import torch.distributed as dist
import torch.profiler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.LAION.dataset import build_dataloader
from clap_model import CLAP

def setup_ddp():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. No CPU fallback.")

    # torchrun sets these for us
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ or "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Must be launched with torchrun. Example: "
                           "torchrun --standalone --nproc_per_node=1 train.py")

    backend = "nccl"
    dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"[DDP] backend={backend}, world_size={world_size}, "
              f"rank={rank}, local_rank={local_rank}, device={device}")

    return rank, local_rank, world_size, device


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def concat_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """Gradient-friendly all_gather. Identity if world_size == 1."""
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor
    tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors, tensor)
    return torch.cat(tensors, dim=0)

def train():
    rank, local_rank, world_size, device = setup_ddp()

    parser = argparse.ArgumentParser(description="Train CLAP projection")
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to load')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    args = parser.parse_args()

    # ---------------- config ----------------
    data_root = "datasets/LAION"
    batch_size = 4096
    num_workers = 8
    prefetch_factor = 4
    num_epochs = 20
    log_every = 10

    lr = 3e-6 
    weight_decay = 0.01
    max_logit_scale = 5.0       # clamp logit_scale to avoid explosion
    log_dir = "runs/clap_proj"
    # ----------------------------------------

    if not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"Sharded dataset not found at '{data_root}'. Run reshard_dataset.py before training."
        )

    dataloader, sampler = build_dataloader(
        root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        in_memory=True,
    )
    steps_per_epoch = len(dataloader)

    # CLAP only: train audio_proj, text_proj, logit_scale
    model = CLAP(dim=768).cuda(device)
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    clap = model.module

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Load checkpoint if provided
    start_epoch = 0
    global_step = 0
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"], strict=True)
        else:
            clap.load_state_dict(ckpt, strict=True)
        if args.resume:
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt.get("epoch", 0) + 1
            global_step = ckpt.get("global_step", 0)
        # For DDP, weights are loaded on all ranks

    # TensorBoard only on rank 0
    writer = SummaryWriter(log_dir=log_dir) if rank == 0 else None
    profiler_context = contextlib.nullcontext()
    trace_path = os.getenv("PROFILER_TRACE_PATH", "trace.json")
    if rank == 0 and os.getenv("ENABLE_PROFILER"):
        os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
        profiler_context = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=5, warmup=5, active=20, repeat=1),
            on_trace_ready=lambda prof: prof.export_chrome_trace(trace_path),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )

    with profiler_context as profiler:
        for epoch in range(start_epoch, num_epochs):
            sampler.set_epoch(epoch)
            model.train()

            if rank == 0:
                data_iter = tqdm(dataloader, desc=f"Epoch {epoch}", dynamic_ncols=True)
            else:
                data_iter = dataloader

            for step, batch in enumerate(data_iter):
                nvtx.range_push("batch_load")
                a_emb = batch["audio_embedding"].cuda(local_rank, non_blocking=True)
                t_emb = batch["text_embedding"].cuda(local_rank, non_blocking=True)
                nvtx.range_pop()

                nvtx.range_push("forward")
                _, z_a_local, z_t_local = model(a_emb, t_emb)
                nvtx.range_pop()

                nvtx.range_push("all_gather")
                z_a = concat_all_gather(z_a_local)
                z_t = concat_all_gather(z_t_local)
                nvtx.range_pop()

                nvtx.range_push("logits")
                scale = clap.logit_scale.exp()
                logits = scale * (z_t @ z_a.t())
                nvtx.range_pop()

                nvtx.range_push("loss")
                loss = clap.info_nce(logits)
                nvtx.range_pop()

                nvtx.range_push("backward")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nvtx.range_pop()

                nvtx.range_push("step")
                optimizer.step()
                nvtx.range_pop()

                with torch.no_grad():
                    clap.logit_scale.clamp_(max=max_logit_scale)

                if profiler is not None:
                    profiler.step()

                if writer is not None and step % log_every == 0:
                    global_step = epoch * steps_per_epoch + step
                    loss_value = float(loss.detach().cpu())
                    logit_value = float(clap.logit_scale.detach().cpu())
                    writer.add_scalar("train/loss_clap", loss_value, global_step)
                    writer.add_scalar("train/logit_scale", logit_value, global_step)

    # Save final checkpoint (rank 0)
    if rank == 0:
        os.makedirs("checkpoints", exist_ok=True)
        ckpt = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": num_epochs - 1,
            "global_step": num_epochs * steps_per_epoch,
            "config": {
                "data_root": data_root,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "prefetch_factor": prefetch_factor,
                "num_epochs": num_epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "max_logit_scale": max_logit_scale,
                "log_dir": log_dir,
            },
        }
        torch.save(ckpt, "checkpoints/clap_proj_last.pt")
        torch.save(clap.state_dict(), "checkpoints/clap_proj_heads.pt")
        if writer is not None:
            writer.close()

    cleanup_ddp()


if __name__ == "__main__":
    try:
        train()
    except Exception:
        # Make sure elastic/torchrun shows the real cause
        print("Unhandled exception in train():")
        traceback.print_exc()
        raise