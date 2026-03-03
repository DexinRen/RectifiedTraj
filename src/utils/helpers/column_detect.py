#!/usr/bin/env python3
"""
Inspect one parquet file quickly:
- file size, row groups, metadata row count
- schema
- first N rows (sorted by timestamp if present)

Use `--parquet-path` for an exact file, or `--parquet-dir` to auto-pick one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


DEFAULT_PARQUET_DIR = Path("./dataset/raw/BlogWatcher")


def _pick_parquet_file(parquet_dir: Path) -> Path:
    if not parquet_dir.exists() or not parquet_dir.is_dir():
        raise FileNotFoundError(f"Parquet directory not found: {parquet_dir}")
    candidates = sorted(parquet_dir.glob("*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No parquet file found under: {parquet_dir}")
    return candidates[0]


def _resolve_parquet_path(parquet_path: str | None, parquet_dir: str | None) -> Path:
    if parquet_path:
        path = Path(parquet_path)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() != ".parquet":
            raise RuntimeError(f"Expected .parquet file, got: {path}")
        return path
    if parquet_dir:
        return _pick_parquet_file(Path(parquet_dir))
    return _pick_parquet_file(DEFAULT_PARQUET_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a parquet schema/sample quickly")
    parser.add_argument(
        "--parquet-path",
        type=str,
        default="",
        help="Exact parquet file path to inspect.",
    )
    parser.add_argument(
        "--parquet-dir",
        type=str,
        default="",
        help="Directory to auto-pick one parquet file from (sorted order).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=25,
        help="Number of rows to print.",
    )
    parser.add_argument(
        "--cap-rows",
        type=int,
        default=200000,
        help="Max rows loaded from first row-group before sorting/printing.",
    )
    parser.add_argument(
        "--full-row-count",
        action="store_true",
        help="Also sum rows across all row groups.",
    )
    args = parser.parse_args()

    path = _resolve_parquet_path(
        parquet_path=str(args.parquet_path).strip() or None,
        parquet_dir=str(args.parquet_dir).strip() or None,
    )

    n = max(1, int(args.rows))
    k_rows = max(1, int(args.cap_rows))
    full_row_count = bool(args.full_row_count)

    size_bytes = path.stat().st_size
    size_mib = size_bytes / (1024**2)
    size_gib = size_bytes / (1024**3)

    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    num_row_groups = pf.num_row_groups
    meta_rows = pf.metadata.num_rows
    num_cols = len(schema)

    print("=== Dataset size ===")
    print(f"Path: {path}")
    print(f"File size: {size_bytes:,} bytes ({size_mib:.2f} MiB, {size_gib:.3f} GiB)")
    print(f"Columns: {num_cols}")
    print(f"Row groups: {num_row_groups}")
    print(f"Total rows (metadata): {meta_rows:,}")

    if full_row_count:
        rg_rows = sum(pf.metadata.row_group(i).num_rows for i in range(num_row_groups))
        print(f"Total rows (sum row groups): {rg_rows:,}")

    print("\n=== Schema ===")
    print(schema)

    table = pf.read_row_groups([0])
    if table.num_rows > k_rows:
        table = table.slice(0, k_rows)
    df = table.to_pandas()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp", kind="mergesort")
    else:
        print("\n[WARN] No 'timestamp' column found; printing without sorting.")

    head = df.head(n)
    mode = "sorted by timestamp" if "timestamp" in df.columns else "unsorted"
    print(f"\n=== First {len(head)} rows ({mode}) ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(head.to_string(index=False))


if __name__ == "__main__":
    main()
