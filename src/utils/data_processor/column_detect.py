import polars as pl
from pathlib import Path

RAW_DIR = Path("./dataset/raw")
SAMPLE_N = 1000

modalities = set()

for p in RAW_DIR.glob("*.parquet"):
    df = pl.scan_parquet(p)
    sample = df.select("modality").head(SAMPLE_N).collect()
    modalities.update(sample["modality"].unique().to_list())

print(sorted(modalities))
