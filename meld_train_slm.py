import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple, cast

import torch
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from datasets.MELD.dataloader import build_dataloader
from models.meld_slm_model import MELDBaseModel, MELDModelOptionA, MELDModelOptionB

NUM_MELD_LABELS = 7

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDP training for MELD using frozen Stage-3 SLM backbone")
    parser.add_argument("--data-root", type=str, default="datasets/MELD", help="Root directory containing train/eval/test splits")
    parser.add_argument(
        "--stage3-ckpt",
        type=str,
        default="models/slm_stage3_epoch149_best.pt",
        help="Path to the frozen Stage-3 SLM checkpoint",
    )
    parser.add_argument(
        "--clap-ckpt",
        type=str,
        default="models/clap_projectors.pt",
        help="Path to the frozen CLAP projector weights",
    )
    parser.add_argument("--mode", choices=["A", "B"], default="B", help="Adapter strategy: A=feature head, B=36->7 logits")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Global batch size across all ranks")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader worker processes per rank")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Prefetch factor per worker")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")
    parser.add_argument("--log-interval", type=int, default=10, help="Steps between training log prints")
    parser.add_argument("--eval-interval", type=int, default=1, help="Epoch interval for validation")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory for model checkpoints")
    parser.add_argument("--resume", type=str, help="Path to a checkpoint created by this script")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Optional gradient norm clipping")
    parser.add_argument("--in-dim", type=int, default=768, help="Stage-3 input embedding dimension")
    parser.add_argument("--proj-dim", type=int, default=768, help="Projection dimension inside Stage-3")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="Hidden dimension for Stage-3 MLP")
    parser.add_argument("--seed", type=int, default=17, help="Base random seed")
    parser.add_argument("--backend", type=str, choices=["nccl", "gloo"], help="Torch distributed backend override")
    return parser.parse_args()


def setup_ddp(backend: str | None) -> Tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DDP training.")

    required_env = {"LOCAL_RANK", "RANK", "WORLD_SIZE"}
    if not required_env.issubset(os.environ):
        raise RuntimeError("Expected torchrun launch (LOCAL_RANK, RANK, WORLD_SIZE env vars missing).")

    backend = backend or "nccl"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if rank == 0:
        print(f"[DDP] backend={backend} world_size={world_size} rank={rank} local_rank={local_rank}")

    return rank, world_size, local_rank, device


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    common_kwargs = {
        "stage3_ckpt": args.stage3_ckpt,
        "clap_ckpt_path": args.clap_ckpt,
        "num_meld_labels": NUM_MELD_LABELS,
        "in_dim": args.in_dim,
        "proj_dim": args.proj_dim,
        "hidden_dim": args.hidden_dim,
        "device": device,
    }
    if args.mode == "A":
        model = MELDModelOptionA(**common_kwargs)
    else:
        model = MELDModelOptionB(**common_kwargs)
    return model


def all_reduce_sum(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def evaluate(model: torch.nn.Module, loader, criterion, device: torch.device, amp: bool) -> Tuple[float, float]:
    model.eval()
    total = torch.zeros(3, device=device)

    with torch.no_grad():
        for batch in loader:
            labels = batch["label_id"].to(device, non_blocking=True)
            with autocast(device_type='cuda', enabled=amp):
                logits = model(batch)
                loss = criterion(logits, labels)
            preds = logits.argmax(dim=-1)
            correct = (preds == labels).sum()
            batch_size = torch.tensor(labels.size(0), device=device, dtype=torch.float32)

            stats = torch.stack(
                (
                    loss.detach() * batch_size,
                    correct.to(dtype=torch.float32),
                    batch_size,
                )
            )
            total += stats

    total = all_reduce_sum(total)
    total_loss, total_correct, total_count = total.tolist()
    mean_loss = total_loss / max(total_count, 1.0)
    accuracy = total_correct / max(total_count, 1.0)
    model.train()
    return float(mean_loss), float(accuracy)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_val_acc: float,
    args: argparse.Namespace,
) -> None:
    module = cast(MELDBaseModel, model.module if isinstance(model, DDP) else model)
    payload = {
        "trainable_state": module.trainable_state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_acc": best_val_acc,
        "config": vars(args),
        "mode": args.mode,
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None) -> Tuple[int, int, float]:
    payload = torch.load(path, map_location="cpu")
    module = cast(MELDBaseModel, model.module if isinstance(model, DDP) else model)
    module.load_trainable_state_dict(payload["trainable_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    epoch = int(payload.get("epoch", -1)) + 1
    global_step = int(payload.get("global_step", 0))
    best_val = float(payload.get("best_val_acc", 0.0))
    return epoch, global_step, best_val


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    rank = 0
    writer: SummaryWriter | None = None

    try:
        rank, world_size, local_rank, device = setup_ddp(args.backend)
        set_seed(args.seed + rank)

        if not Path(args.stage3_ckpt).is_file():
            raise FileNotFoundError(f"Stage-3 checkpoint not found: {args.stage3_ckpt}")
        if not Path(args.clap_ckpt).is_file():
            raise FileNotFoundError(
                f"CLAP projector checkpoint not found: {args.clap_ckpt}. Provide --clap-ckpt pointing to the correct file."
            )

        if rank == 0:
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)

        per_rank_batch = args.batch_size // world_size
        if per_rank_batch * world_size != args.batch_size:
            raise ValueError("Global batch size must be divisible by world size")
        if rank == 0:
            print(f"Using per-rank batch size {per_rank_batch}")

        train_loader, train_sampler_obj = build_dataloader(
            root=args.data_root,
            split="train",
            batch_size=per_rank_batch,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
        )
        eval_loader, eval_sampler_obj = build_dataloader(
            root=args.data_root,
            split="eval",
            batch_size=per_rank_batch,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
        )

        train_sampler: Any = train_sampler_obj
        eval_sampler: Any = eval_sampler_obj

        model = build_model(args, device)
        model.to(device)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

        for n, p in model.named_parameters():
            if p.requires_grad and rank == 0:
                print("trainable:", n, p.shape)

        if rank == 0:
            model.eval()
            with torch.no_grad():
                batch = next(iter(train_loader))
                labels = batch["label_id"].to(device)
                logits = model(batch)
                print("logits shape:", logits.shape)
                print("labels[0:8]:", labels[:8].tolist())
                print("preds[0:8]:", logits.argmax(-1)[:8].tolist())
                print("sample probs[0]:", torch.softmax(logits[0], -1))
            model.train()

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found in the model.")

        class_weights = torch.tensor(
            [2.06, 4.17, 4.19, 1.64, 1.00, 2.62, 1.98], # computed from train split distribution
            device=device
        )
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1).to(device)
        scaler = GradScaler(device='cuda', enabled=args.amp)

        start_epoch = 0
        global_step = 0
        best_val_acc = 0.0

        if args.resume:
            resume_path = Path(args.resume)
            if not resume_path.is_file():
                raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
            start_epoch, global_step, best_val_acc = load_checkpoint(resume_path, model, optimizer)
            if rank == 0:
                print(f"Resumed from {resume_path} at epoch {start_epoch}, global_step {global_step}")

        if rank == 0:
            log_dir = Path(args.save_dir) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(log_dir))

        total_steps = len(train_loader)

        for epoch in range(start_epoch, args.epochs):
            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)
            if eval_sampler is not None and hasattr(eval_sampler, "set_epoch"):
                eval_sampler.set_epoch(epoch)

            model.train()
            epoch_stats = torch.zeros(3, device=device)
            epoch_start = time.time()

            for step, batch in enumerate(train_loader):
                labels = batch["label_id"].to(device, non_blocking=True)

                with autocast(device_type='cuda', enabled=args.amp):
                    logits = model(batch)
                    loss = criterion(logits, labels)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if args.grad_clip > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                preds = logits.argmax(dim=-1)
                correct = (preds == labels).sum()
                batch_size = torch.tensor(labels.size(0), device=device, dtype=torch.float32)

                batch_stats = torch.stack(
                    (
                        loss.detach() * batch_size,
                        correct.to(dtype=torch.float32),
                        batch_size,
                    )
                )
                epoch_stats += batch_stats

                if writer is not None and (global_step % args.log_interval == 0):
                    writer.add_scalar("train/loss", float(loss.detach()), global_step)
                    writer.add_scalar("train/acc", float((preds == labels).float().mean().item()), global_step)

                if rank == 0 and (step % args.log_interval == 0):
                    print(
                        f"Epoch {epoch} [{step}/{total_steps}] loss={loss.item():.4f} "
                        f"acc={(preds == labels).float().mean().item():.4f}"
                    )

                global_step += 1

            epoch_stats = all_reduce_sum(epoch_stats)
            epoch_loss = (epoch_stats[0] / epoch_stats[2]).item()
            epoch_acc = (epoch_stats[1] / epoch_stats[2]).item()
            epoch_time = time.time() - epoch_start

            if rank == 0:
                print(
                    f"Epoch {epoch} done: loss={epoch_loss:.4f} acc={epoch_acc:.4f} time={epoch_time:.1f}s"
                )
                if writer is not None:
                    writer.add_scalar("epoch/loss", epoch_loss, epoch)
                    writer.add_scalar("epoch/acc", epoch_acc, epoch)
                    writer.add_scalar("epoch/time", epoch_time, epoch)

            if (epoch + 1) % args.eval_interval == 0:
                val_loss, val_acc = evaluate(model, eval_loader, criterion, device, args.amp)
                if rank == 0:
                    print(f"  Eval @ epoch {epoch}: loss={val_loss:.4f} acc={val_acc:.4f}")
                    if writer is not None:
                        writer.add_scalar("val/loss", val_loss, epoch)
                        writer.add_scalar("val/acc", val_acc, epoch)

                    is_best = val_acc > best_val_acc
                    best_val_acc = max(best_val_acc, val_acc)
                    ckpt_path = Path(args.save_dir) / f"meld_slm_{args.mode}_epoch{epoch:03d}.pt"
                    save_checkpoint(ckpt_path, model, optimizer, epoch, global_step, best_val_acc, args)
                    print(f"  Saved checkpoint to {ckpt_path}")
                    if is_best:
                        best_path = Path(args.save_dir) / f"meld_slm_{args.mode}_best.pt"
                        save_checkpoint(best_path, model, optimizer, epoch, global_step, best_val_acc, args)
                        print(f"  New best model (acc={val_acc:.4f}) saved to {best_path}")

        if rank == 0:
            print("Training complete.")

    finally:
        if writer is not None:
            writer.close()
        cleanup_ddp()


if __name__ == "__main__":
    main()
