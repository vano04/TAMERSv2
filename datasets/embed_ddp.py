# GPT Generated code:

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import inspect
import torch
import torch.distributed as dist
import torchaudio
from sentence_transformers import SentenceTransformer
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel

try:
    from torch.amp.autocast_mode import autocast as torch_autocast
except (ImportError, AttributeError):
    from torch.cuda.amp import autocast as torch_autocast

try:
    from torchcodec.decoders import AudioDecoder

    TORCHCODEC_AVAILABLE = True
except ImportError:
    AudioDecoder = None
    TORCHCODEC_AVAILABLE = False

try:  # Optional performance boost for JSON loading
    import orjson
except ImportError:
    orjson = None

PROJECT_ROOT = Path(__file__).resolve().parent
LAION_DIR = PROJECT_ROOT / "Datasets" / "LAION"
if str(LAION_DIR) not in sys.path:
    sys.path.insert(0, str(LAION_DIR))

from audio_dataset import (
    RAW_TO_TARGET_LABEL,
    TARGET_LABELS,
    TARGET_LABEL_SET,
    _normalize_label,
)

MANIFEST_NAME = "manifest.jsonl"
ENGLISH_BRANCH = Path("audio/en")
EXPECTED_EXTENSION = ".wav"
LABEL_TO_ID = {label: idx for idx, label in enumerate(TARGET_LABELS)}
DEFAULT_LOADER_WORKERS = max(1, min(16, os.cpu_count() or 1))

StateDict = Dict[str, torch.Tensor]


@dataclass
class SampleMetadata:
    path: Path
    transcript: str
    canonical_label: str
    raw_label: str
    directory_label: str
    label_id: int


@dataclass
class ModelBundle:
    quantizer: torch.nn.Module
    acoustic: torch.nn.Module
    projection: torch.nn.Linear
    text: SentenceTransformer


@dataclass
class TokenPayload:
    input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor]


@dataclass
class TokenCacheConfig:
    directory: Path
    dataset_root: Path
    read: bool
    write: bool


@dataclass
class DistState:
    distributed: bool
    rank: int
    world_size: int
    local_rank: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDP embedding pre-compute for English splits (manifest streaming).")
    parser.add_argument("--laion-root", type=Path, default=LAION_DIR, help="Root directory for the LAION dataset.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where parquet shards are stored.")
    parser.add_argument("--batch-size", type=int, default=8, help="Samples per batch for audio encoding.")
    parser.add_argument("--text-batch-size", type=int, default=32, help="Batch size for SentenceTransformer encode calls.")
    parser.add_argument(
        "--loader-workers",
        type=int,
        default=DEFAULT_LOADER_WORKERS,
        help="Background threads dedicated to audio loading and resampling (0 disables the pool).",
    )
    parser.add_argument("--strict", action="store_true", help="Enable strict manifest validation (abort on bad rows).")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing shard files.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip manifests whose shard already exists in the output directory.",
    )
    parser.add_argument("--seed", type=int, default=17, help="Base random seed used for projection init.")
    parser.add_argument("--projection-checkpoint", type=Path, help="Path to a torch checkpoint with projection weights.")
    parser.add_argument("--compression", default="snappy", help="Parquet compression codec.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Computation device override.")
    parser.add_argument("--backend", choices=["nccl", "gloo"], help="Distributed backend override.")
    parser.add_argument("--log-every", type=int, default=1, help="Log progress every N manifests per rank.")
    parser.add_argument("--target-sr", type=int, default=16000, help="Target sampling rate for audio decoding.")
    parser.add_argument(
        "--decode-backend",
        choices=["auto", "torchaudio", "torchcodec"],
        default="auto",
        help="Audio decoding backend (torchcodec can be faster if installed).",
    )
    parser.add_argument(
        "--token-cache-dir",
        type=Path,
        help="Optional directory where quantizer token tensors are stored for reuse.",
    )
    parser.add_argument(
        "--token-cache-mode",
        choices=["read", "write", "readwrite"],
        default="readwrite",
        help="Token cache usage policy: read existing entries, write new ones, or both.",
    )
    return parser.parse_args()


def infer_hidden_dim(model: torch.nn.Module) -> int:
    attr_candidates = (
        "hidden_size",
        "hidden_dim",
        "d_model",
        "model_dim",
        "encoder_embed_dim",
        "audio_hidden_dim",
        "model_hidden_size",
    )
    for attr in attr_candidates:
        value = getattr(model.config, attr, None)
        if isinstance(value, (int, float)) and int(value) > 0:
            return int(value)

    hidden_dims = getattr(model.config, "hidden_dims", None)
    if isinstance(hidden_dims, (list, tuple)) and hidden_dims:
        candidate = int(hidden_dims[-1])
        if candidate > 0:
            return candidate

    param = next(model.parameters(), None)
    device = param.device if param is not None else torch.device("cpu")
    seq_len = int(getattr(model.config, "seq_len", 16) or 16)
    dummy_len = max(1, min(seq_len, 16))
    dummy_tokens = torch.zeros((1, dummy_len), dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(seq=dummy_tokens, output_hidden_states=True)

    hidden_states = outputs.get("hidden_states") if isinstance(outputs, dict) else getattr(outputs, "hidden_states", None)
    if not hidden_states:
        raise AttributeError("Acoustic model did not return hidden states for dimension inference.")

    value = int(hidden_states[-1].shape[-1])
    if value <= 0:
        raise ValueError("Acoustic model hidden dimension must be positive.")
    return value


def resolve_amp_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def autocast_context(device: torch.device, amp_enabled: bool, amp_dtype: torch.dtype):
    if amp_enabled and device.type == "cuda":
        params = inspect.signature(torch_autocast).parameters
        if "device_type" in params:
            return torch_autocast(device_type="cuda", dtype=amp_dtype)
        return torch_autocast(dtype=amp_dtype)
    return nullcontext()


def init_distributed(backend: str | None, requested_device: str) -> DistState:
    available = dist.is_available()
    env_configured = all(key in os.environ for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if not available or not env_configured:
        return DistState(distributed=False, rank=0, world_size=1, local_rank=0)
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return DistState(distributed=True, rank=rank, world_size=world_size, local_rank=local_rank)
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return DistState(distributed=True, rank=rank, world_size=world_size, local_rank=local_rank)


def resolve_device(arg_device: str, distributed: bool, local_rank: int) -> torch.device:
    if arg_device == "cpu":
        return torch.device("cpu")
    if arg_device == "cuda" or (arg_device == "auto" and torch.cuda.is_available()):
        if distributed:
            torch.cuda.set_device(local_rank)
        return torch.device("cuda")
    return torch.device("cpu")


def load_models(device: torch.device) -> ModelBundle:
    torch.set_grad_enabled(False)
    quantizer = AutoModel.from_pretrained("TuKoResearch/WavCochV8192", trust_remote_code=True).to(device).eval()
    acoustic = AutoModel.from_pretrained(
        "TuKoResearch/AuriStream100M_RoPE_librilight", trust_remote_code=True
    ).to(device).eval()
    text_model = SentenceTransformer("google/embeddinggemma-300m").to(device)
    text_model.eval()

    projection_in = infer_hidden_dim(acoustic)
    text_embed_dim = text_model.get_sentence_embedding_dimension()
    if text_embed_dim is None or text_embed_dim <= 0:
        raise ValueError("SentenceTransformer reports invalid embedding dimension.")
    projection = torch.nn.Linear(projection_in, int(text_embed_dim)).to(device).eval()

    return ModelBundle(
        quantizer=quantizer,
        acoustic=acoustic,
        projection=projection,
        text=text_model,
    )


def maybe_load_projection(path: Path | None, projection: torch.nn.Linear, rank: int) -> None:
    if path is None:
        return
    if not path.is_file():
        raise FileNotFoundError(f"Projection checkpoint not found: {path}")
    state: Optional[StateDict]
    if rank == 0:
        loaded = torch.load(path, map_location=projection.weight.device)
        if not isinstance(loaded, dict):
            raise TypeError("Projection checkpoint is not a state_dict")
        candidate = {str(k): v for k, v in loaded.items()}
        for value in candidate.values():
            if not isinstance(value, torch.Tensor):
                raise TypeError("Projection checkpoint contains non-tensor values")
        state = cast(StateDict, candidate)
    else:
        state = None
    if dist.is_available() and dist.is_initialized():
        container: List[Optional[StateDict]] = [state]
        dist.broadcast_object_list(container, src=0)
        state = container[0]
    if state is None:
        raise RuntimeError("Failed to broadcast projection state.")
    projection.load_state_dict(state)


def broadcast_projection(projection: torch.nn.Linear, distributed: bool) -> None:
    if not distributed:
        return
    weight = projection.weight.data
    bias = projection.bias.data if projection.bias is not None else None
    dist.broadcast(weight, src=0)
    if bias is not None:
        dist.broadcast(bias, src=0)


def loads_json(line: str) -> dict:
    if orjson is not None:
        return orjson.loads(line)
    return json.loads(line)


def list_manifest_paths(dataset_root: Path) -> List[Path]:
    branch_root = (dataset_root / ENGLISH_BRANCH).resolve()
    if not branch_root.is_dir():
        raise FileNotFoundError(f"Missing English audio branch at {branch_root}")
    manifests: List[Path] = []
    for emotion_dir in sorted(branch_root.iterdir()):
        if not emotion_dir.is_dir():
            continue
        manifest = emotion_dir / MANIFEST_NAME
        if manifest.is_file():
            manifests.append(manifest)
    if not manifests:
        raise FileNotFoundError(f"No manifests discovered under {branch_root}")
    return manifests


def iter_manifest_entries(
    manifest_path: Path,
    dataset_root: Path,
    *,
    strict: bool,
) -> Iterator[SampleMetadata]:
    directory_label = manifest_path.parent.name
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = loads_json(line)
            except Exception as exc:
                if strict:
                    raise ValueError(f"Invalid JSON in {manifest_path} (line {line_number})") from exc
                continue

            file_value = payload.get("file")
            transcript = payload.get("transcript")
            raw_label_value = payload.get("label") or payload.get("raw_label")

            if not isinstance(file_value, str) or not file_value:
                if strict:
                    raise ValueError(f"Invalid 'file' value in {manifest_path}")
                continue
            if not isinstance(transcript, str) or not transcript:
                if strict:
                    raise ValueError(f"Missing transcript in {manifest_path}")
                continue
            if not isinstance(raw_label_value, str) or not raw_label_value:
                if strict:
                    raise ValueError(f"Missing label in {manifest_path}")
                continue

            raw_label = raw_label_value.strip()
            normalized = _normalize_label(raw_label)
            canonical = RAW_TO_TARGET_LABEL.get(normalized, normalized)
            if canonical not in TARGET_LABEL_SET or canonical not in LABEL_TO_ID:
                if strict:
                    raise ValueError(
                        f"Label '{raw_label}' (normalized '{normalized}') not covered by TARGET_LABELS"
                    )
                continue

            audio_path = Path(file_value)
            if not audio_path.is_absolute():
                audio_path = (dataset_root / audio_path).resolve()

            if audio_path.suffix.lower() != EXPECTED_EXTENSION:
                if strict:
                    raise ValueError(f"Unexpected extension for {audio_path}")
                continue
            if not audio_path.is_file():
                if strict:
                    raise FileNotFoundError(audio_path)
                continue

            yield SampleMetadata(
                path=audio_path,
                transcript=transcript,
                canonical_label=canonical,
                raw_label=raw_label,
                directory_label=directory_label,
                label_id=LABEL_TO_ID[canonical],
            )


def count_manifest_entries(
    manifest_path: Path,
    dataset_root: Path,
    *,
    strict: bool,
) -> int:
    return sum(1 for _ in iter_manifest_entries(manifest_path, dataset_root, strict=strict))


def chunked(iterator: Iterable[SampleMetadata], chunk_size: int) -> Iterator[List[SampleMetadata]]:
    batch: List[SampleMetadata] = []
    for item in iterator:
        batch.append(item)
        if len(batch) == chunk_size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_waveform(path: Path, target_sr: int, backend: str) -> torch.Tensor:
    if backend == "torchcodec" and TORCHCODEC_AVAILABLE:
        assert AudioDecoder is not None  # for type checkers
        decoded = AudioDecoder(str(path), sample_rate=target_sr, num_channels=1).get_all_samples()
        waveform = decoded.data.squeeze(0).contiguous()
    else:
        waveform, sr = torchaudio.load(str(path))
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)
        waveform = waveform.float().squeeze(0)

    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak
    return waveform.clamp(-1.0, 1.0).contiguous()


def load_waveform_batch(
    samples: List[SampleMetadata],
    target_sr: int,
    executor: Optional[ThreadPoolExecutor],
    backend: str,
) -> List[torch.Tensor]:
    if not samples:
        return []
    if executor is None:
        return [load_waveform(sample.path, target_sr, backend) for sample in samples]

    def _load(sample: SampleMetadata) -> torch.Tensor:
        return load_waveform(sample.path, target_sr, backend)

    return list(executor.map(_load, samples))


def quantize_waveforms(
    waveforms: List[torch.Tensor],
    quantizer: torch.nn.Module,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> List[TokenPayload]:
    if not waveforms:
        return []

    padded = pad_sequence(waveforms, batch_first=True).contiguous()
    if device.type == "cuda":
        padded = padded.pin_memory()
    wav = padded.to(device, non_blocking=True).unsqueeze(1)

    autocast_ctx = autocast_context(device, amp_enabled, amp_dtype)

    with torch.no_grad():
        with autocast_ctx:
            token_payload = quantizer(wav)
        if "input_ids" not in token_payload:
            raise KeyError("Quantizer output missing 'input_ids'")
        token_ids = token_payload["input_ids"]
        attention_mask = token_payload.get("attention_mask")

    payloads: List[TokenPayload] = []
    for index in range(token_ids.size(0)):
        ids = token_ids[index].detach().to(device="cpu")
        if ids.dtype != torch.long:
            ids = ids.to(torch.long)
        mask_tensor = None
        if attention_mask is not None:
            mask_tensor = attention_mask[index].detach().to(device="cpu")
            mask_tensor = mask_tensor.to(torch.float32)
        payloads.append(TokenPayload(input_ids=ids, attention_mask=mask_tensor))

    return payloads


def encode_text_batch(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    if not texts:
        dim = model.get_sentence_embedding_dimension()
        if dim is None or dim <= 0:
            raise ValueError("SentenceTransformer reports invalid embedding dimension.")
        return np.zeros((0, int(dim)), dtype=np.float32)
    embeddings = model.encode(
        texts,
        batch_size=max(1, batch_size),
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    return embeddings


def encode_audio_from_tokens(
    token_payloads: List[TokenPayload],
    acoustic: torch.nn.Module,
    projection: torch.nn.Linear,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    if not token_payloads:
        return torch.empty((0, projection.out_features), dtype=torch.float32)

    token_tensors = [payload.input_ids.to(torch.long) for payload in token_payloads]
    padded_tokens = pad_sequence(token_tensors, batch_first=True, padding_value=0)

    if any(payload.attention_mask is not None for payload in token_payloads):
        mask_components = []
        for payload in token_payloads:
            if payload.attention_mask is not None:
                mask_components.append(payload.attention_mask)
            else:
                mask_components.append(torch.ones_like(payload.input_ids, dtype=torch.float32))
        padded_mask = pad_sequence(mask_components, batch_first=True, padding_value=0.0)
    else:
        padded_mask = torch.zeros_like(padded_tokens, dtype=torch.float32)
        for row, payload in enumerate(token_payloads):
            length = payload.input_ids.size(0)
            padded_mask[row, :length] = 1.0

    token_tensor = padded_tokens.to(device, non_blocking=True)
    mask_tensor = padded_mask.to(device, non_blocking=True)

    vocab_size = getattr(acoustic.config, "vocab_size", None)
    if vocab_size is not None:
        token_tensor = token_tensor.clamp(min=0, max=int(vocab_size) - 1)

    batch_size = token_tensor.size(0)
    feature_dim = projection.in_features
    summed_hidden = torch.zeros((batch_size, feature_dim), device=device, dtype=torch.float32)
    frame_counts = torch.zeros((batch_size, 1), device=device, dtype=torch.float32)

    max_seq_len = int(getattr(acoustic.config, "seq_len", token_tensor.size(1)))
    autocast_ctx = autocast_context(device, amp_enabled, amp_dtype)

    with torch.no_grad():
        for start in range(0, token_tensor.size(1), max_seq_len):
            window = token_tensor[:, start : start + max_seq_len]
            if window.numel() == 0:
                continue
            with autocast_ctx:
                outputs = acoustic(seq=window, output_hidden_states=True)
            hidden = outputs["hidden_states"][-1].float()
            mask_slice = mask_tensor[:, start : start + hidden.size(1)].unsqueeze(-1).to(hidden.dtype)
            summed_hidden = summed_hidden + (hidden * mask_slice).sum(dim=1)
            frame_counts = frame_counts + mask_slice.sum(dim=1)

    if torch.any(frame_counts == 0):
        raise RuntimeError("No frames processed in audio encoder")

    hidden_mean = summed_hidden / frame_counts
    projected = projection(hidden_mean)
    normalized = torch.nn.functional.normalize(projected, p=2, dim=-1)
    return normalized.to(device="cpu", dtype=torch.float32)


def token_cache_path(sample: SampleMetadata, cache: TokenCacheConfig) -> Path:
    try:
        relative = sample.path.relative_to(cache.dataset_root)
    except ValueError:
        relative = Path(sample.path.name)
    return (cache.directory / relative).with_suffix(".pt")


def load_cached_tokens(sample: SampleMetadata, cache: TokenCacheConfig) -> Optional[TokenPayload]:
    if not cache.read:
        return None
    path = token_cache_path(sample, cache)
    if not path.is_file():
        return None
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        return None
    input_ids = data.get("input_ids")
    attention_mask = data.get("attention_mask")
    if not isinstance(input_ids, torch.Tensor):
        return None
    if input_ids.dtype != torch.long:
        input_ids = input_ids.to(torch.long)
    mask_tensor: Optional[torch.Tensor] = None
    if isinstance(attention_mask, torch.Tensor):
        mask_tensor = attention_mask.to(torch.float32)
    return TokenPayload(input_ids=input_ids, attention_mask=mask_tensor)


def store_cached_tokens(sample: SampleMetadata, payload: TokenPayload, cache: TokenCacheConfig) -> None:
    if not cache.write:
        return
    path = token_cache_path(sample, cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_store = {
        "input_ids": payload.input_ids.to(torch.long),
        "attention_mask": payload.attention_mask.to(torch.float32) if payload.attention_mask is not None else None,
    }
    torch.save(to_store, path)


def convert_embeddings(np_array: np.ndarray, embed_type: pa.FixedSizeListType) -> pa.Array:
    if np_array.dtype != np.float32:
        np_array = np_array.astype(np.float32)
    vector_dim = embed_type.list_size
    flat = pa.array(np_array.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, vector_dim).cast(embed_type)


def write_batch(
    writer: pq.ParquetWriter,
    samples: List[SampleMetadata],
    text_embeddings: np.ndarray,
    audio_embeddings: np.ndarray,
    text_embed_type: pa.FixedSizeListType,
    audio_embed_type: pa.FixedSizeListType,
) -> None:
    paths = [str(sample.path) for sample in samples]
    transcripts = [sample.transcript for sample in samples]
    text_array = convert_embeddings(text_embeddings, text_embed_type)
    audio_array = convert_embeddings(audio_embeddings, audio_embed_type)
    table = pa.Table.from_arrays(
        [
            pa.array(paths, type=pa.string()),
            pa.array(transcripts, type=pa.string()),
            pa.array([sample.label_id for sample in samples], type=pa.int32()),
            audio_array,
            text_array,
        ],
        names=[
            "path",
            "transcript",
            "label_id",
            "audio_embedding",
            "text_embedding",
        ],
    )
    writer.write_table(table)


def build_writer(
    path: Path,
    compression: str,
    metadata: Dict[str, str],
    text_embed_type: pa.FixedSizeListType,
    audio_embed_type: pa.FixedSizeListType,
) -> pq.ParquetWriter:
    schema = pa.schema(
        [
            ("path", pa.string()),
            ("transcript", pa.string()),
            ("label_id", pa.int32()),
            ("audio_embedding", audio_embed_type),
            ("text_embedding", text_embed_type),
        ],
        metadata={key.encode("utf-8"): value.encode("utf-8") for key, value in metadata.items()},
    )
    return pq.ParquetWriter(path, schema=schema, compression=compression)


def process_manifest(
    manifest_path: Path,
    dataset_root: Path,
    output_dir: Path,
    models: ModelBundle,
    device: torch.device,
    args: argparse.Namespace,
    rank: int,
    text_embed_type: pa.FixedSizeListType,
    audio_embed_type: pa.FixedSizeListType,
    base_metadata: Dict[str, str],
    cache_config: Optional[TokenCacheConfig],
) -> Tuple[int, bool]:
    try:
        relative_manifest = str(manifest_path.relative_to(dataset_root))
    except ValueError:
        relative_manifest = str(manifest_path)

    emotion_name = manifest_path.parent.name
    shard_path = output_dir / f"{emotion_name}.rank{rank:02d}.parquet"
    if shard_path.exists():
        if args.overwrite:
            shard_path.unlink()
        elif args.skip_existing:
            expected_count = count_manifest_entries(manifest_path, dataset_root, strict=args.strict)
            existing_rows = -1
            existing_manifest: Optional[str] = None
            try:
                parquet_file = pq.ParquetFile(shard_path)
                metadata = parquet_file.metadata
                if metadata is not None:
                    existing_rows = metadata.num_rows
                    if metadata.metadata is not None:
                        manifest_entry = metadata.metadata.get(b"manifest")
                        if manifest_entry is not None:
                            existing_manifest = manifest_entry.decode("utf-8", errors="ignore")
            except Exception as exc:  # pragma: no cover - corrupted parquet or unreadable file
                print(f"[rank {rank}] {emotion_name}: failed to inspect existing shard ({exc!r}), will rebuild.")

            if existing_rows == expected_count and (
                existing_manifest is None or existing_manifest == relative_manifest
            ):
                print(
                    f"[rank {rank}] {emotion_name}: shard already exists with {existing_rows} rows, skipping."
                )
                return 0, True

            reason_parts = []
            if existing_rows != expected_count:
                reason_parts.append(f"row count {existing_rows} != expected {expected_count}")
            if existing_manifest is not None and existing_manifest != relative_manifest:
                reason_parts.append("manifest metadata mismatch")
            reason = ", ".join(reason_parts) or "unknown mismatch"
            print(
                f"[rank {rank}] {emotion_name}: existing shard at {shard_path} is outdated ({reason}); rebuilding."
            )
            shard_path.unlink()
        else:
            raise FileExistsError(f"Shard already exists: {shard_path}. Use --overwrite to replace it.")

    entries = iter_manifest_entries(manifest_path, dataset_root, strict=args.strict)
    audio_batch_size = max(1, args.batch_size)
    text_batch_size = max(1, args.text_batch_size)
    loader_workers = max(0, args.loader_workers)
    executor: Optional[ThreadPoolExecutor] = None
    if loader_workers > 1:
        executor = ThreadPoolExecutor(max_workers=loader_workers)

    writer: Optional[pq.ParquetWriter] = None
    total_written = 0
    amp_enabled = device.type == "cuda"
    amp_dtype = resolve_amp_dtype()

    try:
        for batch in chunked(entries, audio_batch_size):
            if not batch:
                continue
            texts = [sample.transcript for sample in batch]
            text_embeddings = encode_text_batch(models.text, texts, text_batch_size)

            payloads: List[Optional[TokenPayload]] = []
            missing_indices: List[int] = []

            if cache_config is not None and cache_config.read:
                for idx, sample in enumerate(batch):
                    cached = load_cached_tokens(sample, cache_config)
                    if cached is not None:
                        payloads.append(cached)
                    else:
                        payloads.append(None)
                        missing_indices.append(idx)
            else:
                payloads = [None] * len(batch)
                missing_indices = list(range(len(batch)))

            if missing_indices:
                missing_samples = [batch[index] for index in missing_indices]
                waveforms = load_waveform_batch(missing_samples, args.target_sr, executor, args.decode_backend)
                quantized = quantize_waveforms(
                    waveforms,
                    models.quantizer,
                    device,
                    amp_enabled,
                    amp_dtype,
                )
                if len(quantized) != len(missing_indices):
                    raise RuntimeError("Quantizer returned unexpected number of payloads")
                for local_idx, sample_idx in enumerate(missing_indices):
                    payload = quantized[local_idx]
                    payloads[sample_idx] = payload
                    if cache_config is not None and cache_config.write:
                        store_cached_tokens(batch[sample_idx], payload, cache_config)

            complete_payloads = [cast(TokenPayload, payload) for payload in payloads]

            audio_tensor = encode_audio_from_tokens(
                complete_payloads,
                models.acoustic,
                models.projection,
                device,
                amp_enabled,
                amp_dtype,
            )
            audio_np = audio_tensor.numpy()

            if writer is None:
                writer_metadata = dict(base_metadata)
                writer_metadata.update(
                    {
                        "manifest": relative_manifest,
                        "emotion": emotion_name,
                        "rank": str(rank),
                    }
                )
                writer = build_writer(
                    shard_path,
                    args.compression,
                    writer_metadata,
                    text_embed_type,
                    audio_embed_type,
                )

            write_batch(writer, batch, text_embeddings, audio_np, text_embed_type, audio_embed_type)
            total_written += len(batch)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        if writer is not None:
            writer.close()
        elif shard_path.exists():
            shard_path.unlink()

    return total_written, False


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def main() -> None:
    args = parse_args()

    if args.decode_backend == "auto":
        args.decode_backend = "torchcodec" if TORCHCODEC_AVAILABLE else "torchaudio"
    elif args.decode_backend == "torchcodec" and not TORCHCODEC_AVAILABLE:
        raise RuntimeError("torchcodec backend requested but torchcodec is not installed.")

    cache_config: Optional[TokenCacheConfig] = None
    if args.token_cache_dir is not None:
        mode = args.token_cache_mode
        read = mode in ("read", "readwrite")
        write = mode in ("write", "readwrite")
        if write:
            args.token_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_config = TokenCacheConfig(
            directory=args.token_cache_dir,
            dataset_root=args.laion_root,
            read=read,
            write=write,
        )

    dist_state = init_distributed(args.backend, args.device)
    rank = dist_state.rank
    world_size = dist_state.world_size

    device = resolve_device(args.device, dist_state.distributed, dist_state.local_rank)
    if rank == 0:
        print(f"Running on device {device} | world_size={world_size}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    models = load_models(device)
    maybe_load_projection(args.projection_checkpoint, models.projection, rank)
    broadcast_projection(models.projection, dist_state.distributed)

    audio_embed_dim = int(models.projection.out_features)
    if audio_embed_dim <= 0:
        raise ValueError("Projection layer must output a positive number of features.")

    text_embed_dim = models.text.get_sentence_embedding_dimension()
    if text_embed_dim is None or text_embed_dim <= 0:
        raise ValueError("SentenceTransformer reports invalid embedding dimension.")
    text_embed_dim = int(text_embed_dim)
    audio_embed_type = pa.list_(pa.float32(), audio_embed_dim)
    text_embed_type = pa.list_(pa.float32(), text_embed_dim)

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    manifests = list_manifest_paths(args.laion_root)
    manifests_per_rank = manifests[rank::world_size]

    if rank == 0:
        print(f"Discovered {len(manifests)} manifests. Rank 0 handles {len(manifests_per_rank)} of them.")

    base_metadata = {
        "text_model": "google/embeddinggemma-300m",
        "audio_quantizer": "TuKoResearch/WavCochV8192",
        "audio_model": "TuKoResearch/AuriStream100M_RoPE_librilight",
        "projection_shape": f"{models.projection.in_features}->{audio_embed_dim}",
        "audio_embedding_dim": str(audio_embed_dim),
        "text_embedding_dim": str(text_embed_dim),
        "target_sr": str(args.target_sr),
        "world_size": str(world_size),
        "decode_backend": args.decode_backend,
        "token_cache_mode": args.token_cache_mode if cache_config is not None else "none",
    }

    total_written = 0
    skipped_manifests = 0
    for manifest_index, manifest_path in enumerate(manifests_per_rank, start=1):
        count, skipped = process_manifest(
            manifest_path,
            args.laion_root,
            args.output_dir,
            models,
            device,
            args,
            rank,
            text_embed_type,
            audio_embed_type,
            base_metadata,
            cache_config,
        )
        if skipped:
            skipped_manifests += 1
            continue

        total_written += count
        log_interval = max(args.log_every, 1)
        if count > 0 and (manifest_index % log_interval == 0):
            print(f"[rank {rank}] {manifest_path.parent.name}: +{count} samples (total {total_written})")
        elif count == 0:
            print(f"[rank {rank}] {manifest_path.parent.name}: no valid samples")

    barrier()
    processed_manifests = len(manifests_per_rank) - skipped_manifests
    summary = f"[rank {rank}] Finished with {total_written} samples across {processed_manifests} manifests."
    if skipped_manifests:
        summary += f" Skipped {skipped_manifests} manifest(s) with existing shards."
    print(summary)

    if rank == 0:
        print(f"Parquet shards stored under {args.output_dir}")


if __name__ == "__main__":
    main()
