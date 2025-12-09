#!/usr/bin/env python3
"""
Stage 4 Evaluation: MELD Emotion Classification

Evaluates the fine-tuned MELD model on the official MELD test split.
Computes accuracy, per-class metrics, and weighted F1 (MELD benchmark).
"""

import argparse
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import numpy as np

from datasets.MELD.dataloader import AudioDataset, meld_collate, MELD_LABELS
from models.meld_slm_model import MELDModelOptionA, MELDModelOptionB


NUM_MELD_LABELS = 7


def load_model(args, device: torch.device) -> torch.nn.Module:
    """Load the appropriate MELD model variant."""
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
    
    # Load fine-tuned weights
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        
        # Handle different checkpoint formats
        if "trainable_state" in ckpt:
            model.load_trainable_state_dict(ckpt["trainable_state"])
        elif "meld_head" in ckpt:
            model.load_trainable_state_dict({"meld_head": ckpt["meld_head"]})
        elif "adapter" in ckpt:
            model.load_trainable_state_dict({"adapter": ckpt["adapter"]})
        else:
            # Try to load directly
            model.load_trainable_state_dict(ckpt)
    
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 4 MELD emotion classification")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/meld_slm_A_best.pt",
        help="Path to fine-tuned MELD checkpoint",
    )
    parser.add_argument(
        "--stage3-ckpt",
        type=str,
        default="models/slm_stage3_epoch149_best.pt",
        help="Path to frozen Stage 3 backbone checkpoint",
    )
    parser.add_argument(
        "--clap-ckpt",
        type=str,
        default="models/clap_projectors.pt",
        help="Path to CLAP projector checkpoint",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="datasets/MELD",
        help="Path to MELD dataset",
    )
    parser.add_argument(
        "--mode",
        choices=["A", "B"],
        default="A",
        help="Model variant: A=feature head, B=logit adapter",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers",
    )
    parser.add_argument(
        "--in-dim",
        type=int,
        default=768,
        help="Input embedding dimension",
    )
    parser.add_argument(
        "--proj-dim",
        type=int,
        default=768,
        help="Projection dimension",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=1024,
        help="Hidden dimension for classifier",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Load model
    print(f"Loading MELD model (mode={args.mode})")
    model = load_model(args, device)

    # Load test dataset
    print(f"Loading MELD test split from {args.data_root}")
    test_ds = AudioDataset(args.data_root, split="test")
    print(f"Test set size: {len(test_ds)}")

    # Create dataloader (no DDP, single process)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=meld_collate,
        pin_memory=True,
        drop_last=False,
    )

    # Run inference
    all_preds = []
    all_labels = []
    all_logits = []

    print("\nRunning inference on MELD test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # Move batch to device
            batch_gpu = {
                "audio": batch["audio"].to(device),
                "sr": batch["sr"],
                "text": batch["text"],
            }
            labels = batch["label_id"]
            
            logits = model(batch_gpu)
            preds = logits.argmax(dim=-1)
            
            all_logits.append(logits.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_logits = torch.cat(all_logits, dim=0)

    # Compute metrics
    y_true = all_labels.numpy()
    y_pred = all_preds.numpy()
    
    correct = (all_preds == all_labels).sum().item()
    total = len(all_labels)
    accuracy = correct / total * 100

    # Classification report
    report = classification_report(
        y_true, y_pred,
        labels=list(range(NUM_MELD_LABELS)),
        target_names=MELD_LABELS,
        output_dict=True,
        zero_division=0,
    )
    
    # Weighted F1 is the standard MELD benchmark metric
    weighted_f1 = report["weighted avg"]["f1-score"] * 100

    # Print results
    print("\n" + "=" * 60)
    print("Stage 4 MELD Emotion Classification - Test Set Results")
    print("=" * 60)
    print(f"\nDataset: {args.data_root}")
    print(f"Test samples: {total}")
    print(f"Model mode: {args.mode}")
    print(f"Checkpoint: {args.checkpoint}")
    
    print("\n--- Overall Metrics ---")
    print(f"  Accuracy:           {accuracy:.2f}%")
    print(f"  Weighted F1 (main): {weighted_f1:.2f}%")
    print(f"  Macro F1:           {report['macro avg']['f1-score']*100:.2f}%")
    print(f"  Weighted Precision: {report['weighted avg']['precision']*100:.2f}%")
    print(f"  Weighted Recall:    {report['weighted avg']['recall']*100:.2f}%")
    
    print(f"\n--- Per-Class Metrics ---")
    print(f"  {'Emotion':12s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for label in MELD_LABELS:
        if label in report:
            m = report[label]
            print(f"  {label:12s} {m['precision']*100:>9.1f}% {m['recall']*100:>9.1f}% {m['f1-score']*100:>9.1f}% {int(m['support']):>10d}")
    
    # Confusion matrix
    print(f"\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_MELD_LABELS)))
    
    # Header
    header = "          " + " ".join(f"{l[:4]:>6s}" for l in MELD_LABELS)
    print(header)
    for i, label in enumerate(MELD_LABELS):
        row = f"{label[:9]:9s} " + " ".join(f"{cm[i, j]:>6d}" for j in range(NUM_MELD_LABELS))
        print(row)
    
    # Most confused pairs
    print(f"\n--- Top Confusion Pairs ---")
    confusion_pairs = []
    for i in range(NUM_MELD_LABELS):
        for j in range(NUM_MELD_LABELS):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append({
                    "true": MELD_LABELS[i],
                    "pred": MELD_LABELS[j],
                    "count": cm[i, j],
                    "pct": cm[i, j] / cm[i].sum() * 100 if cm[i].sum() > 0 else 0,
                })
    confusion_pairs.sort(key=lambda x: x["count"], reverse=True)
    for cp in confusion_pairs[:10]:
        print(f"  {cp['true']:10s} -> {cp['pred']:10s}: {cp['count']:4d} ({cp['pct']:5.1f}% of true class)")
    
    print("\n" + "=" * 60)
    print(f"MELD Benchmark Result: Weighted F1 = {weighted_f1:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
