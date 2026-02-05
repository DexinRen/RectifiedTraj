#!/usr/bin/env python3
"""
Print parquet dataset "size" (file size, row groups, total rows from metadata),
schema, and the first N rows sorted by timestamp.

Notes:
- timestamp is stored as string in your parquet; we parse to datetime before sorting.
- We only read the first row group (and cap rows) to avoid loading the full dataset.
"""

from pathlib import Path

synthetic = "dataset/raw/synthetic/part-00013-05c97ce6-7509-443a-bf4a-76418b8b4cd9.c000.zstd.parquet"
utokyo = "dataset/raw/real/utokyo_one_agent_one_month.parquet"
def main() -> None:
    path = Path(synthetic)
    if not path.exists():
        raise FileNotFoundError(path)

    n = 25
    k_rows = 200_000          # max rows to load from the first row group for sorting
    full_row_count = False    # set True if you want to sum rows across all row groups

    # File size on disk
    size_bytes = path.stat().st_size
    size_mib = size_bytes / (1024 ** 2)
    size_gib = size_bytes / (1024 ** 3)

    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow

    num_row_groups = pf.num_row_groups
    meta_rows = pf.metadata.num_rows  # total rows from parquet metadata (cheap)
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

    # Read a manageable chunk (first row group), then cap rows
    table = pf.read_row_groups([0])
    if table.num_rows > k_rows:
        table = table.slice(0, k_rows)

    df = table.to_pandas()

    # Parse + sort by timestamp (stored as string)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp", kind="mergesort")
    else:
        print("\n[WARN] No 'timestamp' column found; printing without sorting.")

    head = df.head(n)

    print(f"\n=== First {len(head)} rows ({'sorted by timestamp' if 'timestamp' in df.columns else 'unsorted'}) ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(head.to_string(index=False))


if __name__ == "__main__":
    main()
