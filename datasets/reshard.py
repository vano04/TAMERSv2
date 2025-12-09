"""Utility for reshaping per-emotion parquet files into mixed shuffled shards."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _discover_parquets(input_root: Path) -> Sequence[Path]:
    files = sorted(input_root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No '*.parquet' files found under {input_root}")
    return files


def _read_tables(files: Iterable[Path]) -> list[pa.Table]:
    tables: list[pa.Table] = []
    for path in files:
        table = pq.read_table(path)
        if table.num_rows == 0:
            continue
        if "label_id" not in table.column_names:
            raise ValueError(f"Missing 'label_id' column in {path}")
        if table.column("label_id").null_count > 0:
            raise ValueError(f"Null entries detected in 'label_id' column for {path}")
        tables.append(table)
    if not tables:
        raise RuntimeError("All input tables are empty.")
    return tables


def _shuffle_table(table: pa.Table, *, seed: int) -> pa.Table:
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(table.num_rows)
    indices = pa.array(permutation, type=pa.int64())
    return table.take(indices)


def _write_shards(
    table: pa.Table,
    output_root: Path,
    *,
    shard_size: int,
    row_group_size: int,
    compression: str,
) -> int:
    output_root.mkdir(parents=True, exist_ok=True)
    total_rows = table.num_rows
    num_shards = math.ceil(total_rows / shard_size)

    written = 0
    for shard_idx in range(num_shards):
        start = shard_idx * shard_size
        end = min(start + shard_size, total_rows)
        shard_rows = end - start
        if shard_rows <= 0:
            break

        shard_table = table.slice(start, shard_rows)
        effective_row_group_size = max(1, min(row_group_size, shard_rows))
        shard_path = output_root / f"shard_{shard_idx:04d}.parquet"
        pq.write_table(
            shard_table,
            shard_path,
            compression=compression,
            row_group_size=effective_row_group_size,
        )
        written += 1

    return written


def reshard_dataset(
    *,
    input_dir: Path,
    output_dir: Path,
    shard_size: int,
    row_group_size: int,
    compression: str,
    seed: int,
) -> None:
    if row_group_size <= 0:
        raise ValueError("row_group_size must be positive")
    if row_group_size >= shard_size:
        raise ValueError("row_group_size must be much smaller than shard_size")

    files = _discover_parquets(input_dir)
    tables = _read_tables(files)
    combined = pa.concat_tables(tables, promote=True)
    shuffled = _shuffle_table(combined, seed=seed)
    num_shards = _write_shards(
        shuffled,
        output_dir,
        shard_size=shard_size,
        row_group_size=row_group_size,
        compression=compression,
    )
    print(
        f"Wrote {num_shards} shards (≈{shard_size} rows each) to {output_dir} "
        f"from {combined.num_rows} input rows."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("Data"), help="Directory containing per-emotion parquet files.")
    parser.add_argument("--output-dir", type=Path, default=Path("data_sharded"), help="Destination directory for mixed shards.")
    parser.add_argument("--shard-size", type=int, default=100_000, help="Target number of rows per shard.")
    parser.add_argument("--row-group-size", type=int, default=4_096, help="Row group size for parquet writer (much smaller than shard size).")
    parser.add_argument("--compression", type=str, default="zstd", help="Compression codec for parquet shards.")
    parser.add_argument("--seed", type=int, default=1337, help="Seed for global shuffle.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    reshard_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        row_group_size=args.row_group_size,
        compression=args.compression,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()