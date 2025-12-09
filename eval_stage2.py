#!/usr/bin/env python3
"""
Stage 2 Evaluation: CLAP Alignment

Evaluates the CLAP projector alignment on the 2% test split of LAION.
Computes retrieval metrics (Recall@K) for audio-to-text and text-to-audio.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from clap_model import CLAP
from datasets.LAION.dataset import InMemoryEmotionEmbeddingDataset


def compute_retrieval_metrics(
    audio_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
    batch_size: int = 256,
) -> dict[str, float]:
    """
    Compute retrieval metrics for audio-text alignment.
    Uses batched computation to avoid OOM on large test sets.
    
    Args:
        audio_embeds: (N, D) L2-normalized audio embeddings
        text_embeds: (N, D) L2-normalized text embeddings
        ks: tuple of K values for Recall@K
        batch_size: batch size for similarity computation
    
    Returns:
        Dictionary with R@K for both directions
    """
    # Move to CPU for large matrix operations to avoid OOM
    audio_embeds = audio_embeds.cpu()
    text_embeds = text_embeds.cpu()
    N = audio_embeds.size(0)
    
    # Compute ranks in batches
    a2t_ranks = torch.zeros(N, dtype=torch.long)
    t2a_ranks = torch.zeros(N, dtype=torch.long)
    
    print(f"  Computing A2T ranks ({N} queries)...")
    for i in tqdm(range(0, N, batch_size), desc="  A2T"):
        end = min(i + batch_size, N)
        # Similarity of audio[i:end] with all texts
        sim_batch = audio_embeds[i:end] @ text_embeds.T  # (batch, N)
        # For each query, find rank of the correct match (diagonal)
        for j, row in enumerate(range(i, end)):
            scores = sim_batch[j]
            # Rank = how many items have higher similarity than the correct one
            correct_score = scores[row]
            rank = (scores > correct_score).sum().item()
            a2t_ranks[row] = rank
    
    print(f"  Computing T2A ranks ({N} queries)...")
    for i in tqdm(range(0, N, batch_size), desc="  T2A"):
        end = min(i + batch_size, N)
        # Similarity of text[i:end] with all audios
        sim_batch = text_embeds[i:end] @ audio_embeds.T  # (batch, N)
        for j, row in enumerate(range(i, end)):
            scores = sim_batch[j]
            correct_score = scores[row]
            rank = (scores > correct_score).sum().item()
            t2a_ranks[row] = rank
    
    metrics = {}
    for k in ks:
        metrics[f"A2T_R@{k}"] = (a2t_ranks < k).float().mean().item() * 100
        metrics[f"T2A_R@{k}"] = (t2a_ranks < k).float().mean().item() * 100
    
    # Mean Reciprocal Rank
    metrics["A2T_MRR"] = (1.0 / (a2t_ranks.float() + 1)).mean().item() * 100
    metrics["T2A_MRR"] = (1.0 / (t2a_ranks.float() + 1)).mean().item() * 100
    
    # Median rank
    metrics["A2T_MedianRank"] = a2t_ranks.float().median().item() + 1
    metrics["T2A_MedianRank"] = t2a_ranks.float().median().item() + 1
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 CLAP alignment")
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
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for embedding computation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Running on device: {device}")

    # Load CLAP model
    print(f"Loading CLAP projectors from {args.clap_ckpt}")
    clap = CLAP(dim=768)
    ckpt = torch.load(args.clap_ckpt, map_location="cpu")
    
    # Handle different checkpoint formats
    if "audio_proj" in ckpt and "text_proj" in ckpt:
        # Direct projector weights
        clap.audio_proj.fc.load_state_dict(_extract_linear(ckpt["audio_proj"]))
        clap.text_proj.fc.load_state_dict(_extract_linear(ckpt["text_proj"]))
        if "logit_scale" in ckpt:
            clap.logit_scale.data = ckpt["logit_scale"]
    elif "model_state" in ckpt:
        # Full model checkpoint
        state = ckpt["model_state"]
        # Remove 'module.' prefix if present (from DDP)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        clap.load_state_dict(state, strict=False)
    else:
        clap.load_state_dict(ckpt, strict=False)
    
    clap.to(device)
    clap.eval()

    # Load test split (2% of data)
    print(f"Loading test split from {args.data_root}")
    test_ds = InMemoryEmotionEmbeddingDataset(args.data_root, split="test")
    print(f"Test set size: {len(test_ds)}")

    # Collect all embeddings
    all_audio_embeds = []
    all_text_embeds = []

    print("Computing projected embeddings...")
    with torch.no_grad():
        for i in tqdm(range(0, len(test_ds), args.batch_size)):
            batch_indices = range(i, min(i + args.batch_size, len(test_ds)))
            
            audio_batch = torch.stack([test_ds[j]["audio_embedding"] for j in batch_indices]).to(device)
            text_batch = torch.stack([test_ds[j]["text_embedding"] for j in batch_indices]).to(device)
            
            # Project through CLAP
            _, z_a, z_t = clap(audio_batch, text_batch)
            
            all_audio_embeds.append(z_a.cpu())
            all_text_embeds.append(z_t.cpu())

    audio_embeds = torch.cat(all_audio_embeds, dim=0)
    text_embeds = torch.cat(all_text_embeds, dim=0)

    print(f"\nEmbedding shapes: audio={audio_embeds.shape}, text={text_embeds.shape}")

    # Move to device for metric computation
    audio_embeds = audio_embeds.to(device)
    text_embeds = text_embeds.to(device)

    # Compute metrics
    print("\nComputing retrieval metrics...")
    metrics = compute_retrieval_metrics(audio_embeds, text_embeds)

    # Print results
    print("\n" + "=" * 50)
    print("Stage 2 CLAP Alignment - Test Set Results")
    print("=" * 50)
    print(f"\nDataset: {args.data_root}")
    print(f"Test samples: {len(test_ds)}")
    print(f"Checkpoint: {args.clap_ckpt}")
    print("\n--- Audio-to-Text Retrieval ---")
    print(f"  R@1:  {metrics['A2T_R@1']:.2f}%")
    print(f"  R@5:  {metrics['A2T_R@5']:.2f}%")
    print(f"  R@10: {metrics['A2T_R@10']:.2f}%")
    print(f"  MRR:  {metrics['A2T_MRR']:.2f}%")
    print(f"  Median Rank: {metrics['A2T_MedianRank']:.1f}")
    print("\n--- Text-to-Audio Retrieval ---")
    print(f"  R@1:  {metrics['T2A_R@1']:.2f}%")
    print(f"  R@5:  {metrics['T2A_R@5']:.2f}%")
    print(f"  R@10: {metrics['T2A_R@10']:.2f}%")
    print(f"  MRR:  {metrics['T2A_MRR']:.2f}%")
    print(f"  Median Rank: {metrics['T2A_MedianRank']:.1f}")
    print("=" * 50)


def _extract_linear(state_dict: dict) -> dict:
    """Extract linear layer weights from projector state dict."""
    if "weight" in state_dict and "bias" in state_dict:
        return {"weight": state_dict["weight"], "bias": state_dict["bias"]}
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("fc."):
            remapped[key.split(".", 1)[1]] = value
    return remapped


if __name__ == "__main__":
    main()
