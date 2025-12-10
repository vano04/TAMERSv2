import argparse
import contextlib
import math
import os
import time
from typing import Tuple

import torch
import torch.distributed as dist
import torch.profiler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.LAION.dataset import EmotionEmbeddingDataset, InMemoryEmotionEmbeddingDataset
from slm_model import ProsodySLM

def setup_ddp() -> Tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required on all ranks.")
    required = {"LOCAL_RANK", "RANK", "WORLD_SIZE"}
    if not required.issubset(os.environ):
        raise RuntimeError(
            "Must be launched with torchrun or in an environment that sets LOCAL_RANK, RANK, WORLD_SIZE",
        )

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(
            f"[DDP] backend=nccl world_size={world_size} rank={rank} "
            f"local_rank={local_rank} device={device}"
        )

    return rank, local_rank, world_size, device


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if dist.get_world_size() == 1:
        return value
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value /= dist.get_world_size()
    return value


def main() -> None:
    rank, local_rank, world_size, device = setup_ddp()

    parser = argparse.ArgumentParser(description="Stage-3 Prosody-Aware SLM training")
    parser.add_argument("--data-root", type=str, default="datasets/LAION", help="Path to sharded embeddings")
    parser.add_argument("--clap-ckpt", type=str, default="checkpoints/clap_projectors.pt", help="Frozen CLAP projectors")
    parser.add_argument("--num-emotions", type=int, required=True, help="Number of emotion classes")
    parser.add_argument("--in-dim", type=int, default=768, help="Input embedding dimension")
    parser.add_argument("--proj-dim", type=int, default=768, help="CLAP projection dimension")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="Hidden width for classifier")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout in classifier MLP")
    parser.add_argument("--batch-size", type=int, default=512, help="Global batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers per rank")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Prefetch factor per worker")
    parser.add_argument("--log-dir", type=str, default="runs/slm_stage3", help="TensorBoard log directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to load")
    parser.add_argument("--resume", action="store_true", help="Resume optimizer and epoch from checkpoint")
    parser.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision")
    parser.add_argument("--in-memory", action="store_true", default=True, help="Load dataset shards into RAM")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Optional gradient clipping norm")
    args = parser.parse_args()

    if rank == 0:
        print("Training configuration:\n" + "\n".join(f"  {k}={v}" for k, v in sorted(vars(args).items())))

    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(f"Dataset root not found: {args.data_root}")
    if not os.path.isfile(args.clap_ckpt):
        raise FileNotFoundError(f"CLAP projector checkpoint not found: {args.clap_ckpt}")
    if args.checkpoint is not None and not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    per_rank_batch = args.batch_size // world_size
    if per_rank_batch * world_size != args.batch_size:
        raise ValueError("batch_size must be divisible by world_size")

    dataset_cls = InMemoryEmotionEmbeddingDataset if args.in_memory else EmotionEmbeddingDataset
    train_ds = dataset_cls(args.data_root, split="train")
    val_ds = dataset_cls(args.data_root, split="val")

    if len(train_ds) == 0:
        raise ValueError("Training split is empty; verify dataset shards and split ratio.")
    if len(val_ds) == 0:
        raise ValueError("Validation split is empty; consider adjusting split ratio or dataset size.")

    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)

    def _make_loader(dataset, sampler, drop_last: bool) -> DataLoader:
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": per_rank_batch,
            "sampler": sampler,
            "num_workers": args.num_workers,
            "pin_memory": True,
            "drop_last": drop_last,
        }
        if args.num_workers > 0:
            loader_kwargs["prefetch_factor"] = args.prefetch_factor
            loader_kwargs["persistent_workers"] = True
        else:
            loader_kwargs["persistent_workers"] = False
        return DataLoader(**loader_kwargs)

    train_loader = _make_loader(train_ds, train_sampler, drop_last=True)
    val_loader = _make_loader(val_ds, val_sampler, drop_last=False)

    steps_per_epoch = len(train_loader)

    model = ProsodySLM(
        clap_ckpt_path=args.clap_ckpt,
        in_dim=args.in_dim,
        proj_dim=args.proj_dim,
        num_emotions=args.num_emotions,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch = 0
    global_step = 0
    best_val_acc = 0.0
    best_ckpt_path = os.path.join("checkpoints", "slm_stage3_best.pt")
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=f"cuda:{local_rank}")
        state_key = "model_state" if "model_state" in ckpt else "state_dict"
        if state_key in ckpt:
            model.load_state_dict(ckpt[state_key])
        else:
            model.module.load_state_dict(ckpt, strict=True)
        model.module.clap.eval()
        if args.resume:
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt.get("epoch", 0) + 1
            global_step = ckpt.get("global_step", 0)
        best_val_acc = ckpt.get("best_val_acc", best_val_acc)
        if rank == 0:
            print(f"Loaded checkpoint from {args.checkpoint}")

    writer = SummaryWriter(log_dir=args.log_dir) if rank == 0 else None
    profiler_context = contextlib.nullcontext()
    trace_path = os.getenv("PROFILER_TRACE_PATH")
    enable_profiler = trace_path is not None and rank == 0
    if enable_profiler:
        assert trace_path is not None
        trace_dir = os.path.dirname(trace_path) or "."
        os.makedirs(trace_dir, exist_ok=True)
        profiler_context = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=4, repeat=1),
            on_trace_ready=lambda prof: prof.export_chrome_trace(trace_path),
        )

    with profiler_context as profiler:
        for epoch in range(start_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            model.train()

            iterator = tqdm(train_loader, desc=f"Epoch {epoch}", disable=rank != 0, dynamic_ncols=True)
            epoch_loss = torch.zeros(1, device=device)
            epoch_correct = torch.zeros(1, device=device)
            epoch_count = torch.zeros(1, device=device)
            start_time = time.time()

            for step, batch in enumerate(iterator):
                x_a = batch["audio_embedding"].to(device, non_blocking=True)
                x_t = batch["text_embedding"].to(device, non_blocking=True)
                y = batch["label_id"].to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=args.amp):
                    logits = model(x_a, x_t)
                    loss = criterion(logits, y)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if args.grad_clip > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                preds = logits.argmax(dim=-1)
                batch_size = y.size(0)
                epoch_loss += loss.detach() * batch_size
                correct = (preds == y).sum().detach().to(epoch_correct.dtype)
                epoch_correct += correct
                epoch_count += torch.tensor(batch_size, device=device, dtype=epoch_count.dtype)

                if profiler is not None:
                    profiler.step()

                if writer is not None and step % 10 == 0:
                    global_step = epoch * steps_per_epoch + step
                    writer.add_scalar("train/loss", float(loss.detach()), global_step)
                    acc_value = (preds == y).float().mean().detach()
                    writer.add_scalar("train/acc", float(acc_value), global_step)

            metrics = torch.stack((epoch_loss, epoch_correct, epoch_count)).squeeze(-1)
            metrics = all_reduce_mean(metrics)
            loss_epoch = (metrics[0] / metrics[2]).item()
            acc_epoch = (metrics[1] / metrics[2]).item()
            epoch_time = time.time() - start_time

            val_sampler.set_epoch(epoch)
            model.eval()
            val_correct = torch.zeros(1, device=device)
            val_total = torch.zeros(1, device=device)
            with torch.no_grad():
                for val_batch in val_loader:
                    x_a_val = val_batch["audio_embedding"].to(device, non_blocking=True)
                    x_t_val = val_batch["text_embedding"].to(device, non_blocking=True)
                    y_val = val_batch["label_id"].to(device, non_blocking=True)

                    with torch.cuda.amp.autocast(enabled=args.amp):
                        val_logits = model(x_a_val, x_t_val)
                    val_preds = val_logits.argmax(dim=-1)
                    val_correct += (val_preds == y_val).sum().detach().to(val_correct.dtype)
                    val_total += torch.tensor(y_val.numel(), device=device, dtype=val_total.dtype)

            val_metrics = torch.stack((val_correct, val_total)).squeeze(-1)
            val_metrics = all_reduce_mean(val_metrics)
            total_samples = val_metrics[1].item()
            val_acc = val_metrics[0].item() / total_samples if total_samples > 0 else float("nan")
            model.train()

            if rank == 0:
                print(
                    f"Epoch {epoch}: loss={loss_epoch:.4f} acc={acc_epoch:.4f} "
                    f"val_acc={val_acc:.4f} time={epoch_time:.1f}s"
                )
                if writer is not None:
                    writer.add_scalar("epoch/loss", loss_epoch, epoch)
                    writer.add_scalar("epoch/acc", acc_epoch, epoch)
                    writer.add_scalar("epoch/time_s", epoch_time, epoch)
                    writer.add_scalar("val/acc", val_acc, epoch)

                if not math.isnan(val_acc) and val_acc > best_val_acc:
                    best_val_acc = val_acc
                    os.makedirs("checkpoints", exist_ok=True)
                    best_ckpt = {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch,
                        "global_step": (epoch + 1) * steps_per_epoch,
                        "val_acc": val_acc,
                        "best_val_acc": best_val_acc,
                        "config": vars(args),
                    }
                    torch.save(best_ckpt, best_ckpt_path)
                    print(f"  Saved new best checkpoint with val_acc={val_acc:.4f}")

                if writer is not None:
                    writer.add_scalar("val/best_acc", best_val_acc, epoch)

            global_step = (epoch + 1) * steps_per_epoch

            if rank == 0:
                os.makedirs("checkpoints", exist_ok=True)
                ckpt = {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val_acc": best_val_acc,
                    "val_acc": val_acc,
                    "config": vars(args),
                }
                torch.save(ckpt, f"checkpoints/slm_stage3_epoch{epoch:03d}.pt")

    if writer is not None:
        writer.close()
    cleanup_ddp()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        cleanup_ddp()
        raise
