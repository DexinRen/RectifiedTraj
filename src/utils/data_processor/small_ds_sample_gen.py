import polars as pl
from pathlib import Path

src = Path(
    "/home/culprit/Projects/RectifiedTraj/dataset/raw/"
    "part-00013-05c97ce6-7509-443a-bf4a-76418b8b4cd9.c000.zstd.parquet"
)

dst = Path("~/Desktop/part-00013.sample10k.parquet").expanduser()

# lazy scan (same as parquet_processor)
df = pl.scan_parquet(src)

# take first 10,000 physical rows
sample = df.head(10_000).collect()

# write parquet (keep compression)
sample.write_parquet(dst, compression="zstd")

print(f"[OK] Saved sample parquet to: {dst}")
print(f"Rows: {sample.height}")
print(f"Columns: {sample.columns}")

# print first 5 rows (Polars only)
print("\n=== First 5 rows (raw parquet data) ===")
print(sample.head(5))
