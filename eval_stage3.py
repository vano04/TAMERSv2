#!/usr/bin/env python3
"""
Stage 3 Evaluation: 36-Emotion Classification on LAION

Evaluates the model on the 2% test split of LAION.
Computes accuracy, per-class metrics, and confusion analysis.
"""

import argparse
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import numpy as np

from datasets.LAION.dataset import InMemoryEmotionEmbeddingDataset
from slm_model import ProsodySLM


# 36 emotion labels used in training (from LAION's got talent enhanced)
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust",
    "embarrassment", "excitement", "fear", "gratitude", "grief", "joy",
    "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral",
    # Extended emotions (if present in dataset)
    "anticipation", "boredom", "contempt", "contentment", "envy", "guilt",
    "horror", "interest"
]


def load_checkpoint(model: ProsodySLM, ckpt_path: str, device: torch.device) -> None:
    """Load model checkpoint, handling DDP and various formats."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    if "model_state" in ckpt:
        state = ckpt["model_state"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    
    # Remove 'module.' prefix from DDP checkpoints
    cleaned = OrderedDict()
    for k, v in state.items():
        new_key = k[7:] if k.startswith("module.") else k
        cleaned[new_key] = v
    
    model.load_state_dict(cleaned, strict=True)
    model.to(device)
    model.eval()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 3 emotion classification on LAION test split")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/slm_stage3_epoch149_best.pt",
        help="Path to Stage 3 SLM checkpoint",
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
        default="datasets/LAION",
        help="Path to LAION dataset with parquet shards",
    )
    parser.add_argument(
        "--num-emotions",
        type=int,
        default=36,
        help="Number of emotion classes",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="K for top-K accuracy",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Running on device: {device}")

    # Build model
    print(f"Loading model from {args.checkpoint}")
    model = ProsodySLM(
        clap_ckpt_path=args.clap_ckpt,
        in_dim=768,
        proj_dim=768,
        num_emotions=args.num_emotions,
        hidden_dim=1024,
        dropout=0.1,
    )
    load_checkpoint(model, args.checkpoint, device)

    # Load test split (2% of data)
    print(f"Loading test split from {args.data_root}")
    test_ds = InMemoryEmotionEmbeddingDataset(args.data_root, split="test")
    print(f"Test set size: {len(test_ds)}")

    # Run inference
    all_preds = []
    all_labels = []
    all_logits = []

    print("\nRunning inference...")
    with torch.no_grad():
        for i in tqdm(range(0, len(test_ds), args.batch_size)):
            batch_indices = range(i, min(i + args.batch_size, len(test_ds)))
            
            audio_batch = torch.stack([test_ds[j]["audio_embedding"] for j in batch_indices]).to(device)
            text_batch = torch.stack([test_ds[j]["text_embedding"] for j in batch_indices]).to(device)
            labels_batch = torch.stack([test_ds[j]["label_id"] for j in batch_indices])
            
            logits = model(audio_batch, text_batch)
            preds = logits.argmax(dim=-1)
            
            all_logits.append(logits.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels_batch)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_logits = torch.cat(all_logits, dim=0)

    # Compute metrics
    correct = (all_preds == all_labels).sum().item()
    total = len(all_labels)
    accuracy = correct / total * 100

    # Top-K accuracy
    top_k_preds = all_logits.topk(args.top_k, dim=-1).indices
    top_k_correct = (top_k_preds == all_labels.unsqueeze(1)).any(dim=1).sum().item()
    top_k_accuracy = top_k_correct / total * 100

    # Per-class analysis
    unique_labels = torch.unique(all_labels).numpy()
    num_classes_present = len(unique_labels)
    
    # Use sklearn for detailed report
    y_true = all_labels.numpy()
    y_pred = all_preds.numpy()
    
    # Get label names for present classes
    label_names = [EMOTION_LABELS[i] if i < len(EMOTION_LABELS) else f"class_{i}" for i in unique_labels]

    # Print results
    print("\n" + "=" * 60)
    print("Stage 3 Emotion Classification - LAION Test Set Results")
    print("=" * 60)
    print(f"\nDataset: {args.data_root}")
    print(f"Test samples: {total}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Classes present in test set: {num_classes_present}/{args.num_emotions}")
    print("\n--- Overall Metrics ---")
    print(f"  Top-1 Accuracy: {accuracy:.2f}%")
    print(f"  Top-{args.top_k} Accuracy: {top_k_accuracy:.2f}%")
    
    # Macro/Weighted averages
    report = classification_report(
        y_true, y_pred,
        labels=unique_labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    
    print(f"\n--- Aggregate Metrics ---")
    print(f"  Macro Precision:    {report['macro avg']['precision']*100:.2f}%")
    print(f"  Macro Recall:       {report['macro avg']['recall']*100:.2f}%")
    print(f"  Macro F1:           {report['macro avg']['f1-score']*100:.2f}%")
    print(f"  Weighted Precision: {report['weighted avg']['precision']*100:.2f}%")
    print(f"  Weighted Recall:    {report['weighted avg']['recall']*100:.2f}%")
    print(f"  Weighted F1:        {report['weighted avg']['f1-score']*100:.2f}%")
    
    # Per-class breakdown (top 10 most frequent)
    print(f"\n--- Per-Class F1 Scores (Top 10 by support) ---")
    class_metrics = []
    for i, label in enumerate(unique_labels):
        name = label_names[i]
        if name in report:
            class_metrics.append({
                "name": name,
                "f1": report[name]["f1-score"],
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "support": report[name]["support"],
            })
    
    class_metrics.sort(key=lambda x: x["support"], reverse=True)
    for cm in class_metrics[:10]:
        print(f"  {cm['name']:20s}: F1={cm['f1']*100:5.1f}%  P={cm['precision']*100:5.1f}%  R={cm['recall']*100:5.1f}%  (n={int(cm['support'])})")
    
    # Confusion analysis: most confused pairs
    print(f"\n--- Top Confusion Pairs ---")
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    confusion_pairs = []
    for i in range(len(unique_labels)):
        for j in range(len(unique_labels)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append({
                    "true": label_names[i],
                    "pred": label_names[j],
                    "count": cm[i, j],
                })
    confusion_pairs.sort(key=lambda x: x["count"], reverse=True)
    for cp in confusion_pairs[:5]:
        print(f"  {cp['true']:15s} -> {cp['pred']:15s}: {cp['count']} samples")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
