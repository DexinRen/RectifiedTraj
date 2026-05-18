#!/usr/bin/env python3
"""Survey point-dense lon/lat hotspots from raw parquet files."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

_CPU_TOTAL = os.cpu_count() or 1
_THREAD_BUDGET = max(1, min(4, int(_CPU_TOTAL) - 2))
for _var in (
    "POLARS_MAX_THREADS",
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, str(_THREAD_BUDGET))

import polars as pl


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compute point-density hotspots from raw parquet lon/lat coordinates "
            "using a grid histogram."
        )
    )
    p.add_argument(
        "--input",
        required=True,
        help="Raw parquet file or directory of parquet files.",
    )
    p.add_argument(
        "--coord-field",
        choices=["clean", "noisy"],
        default="clean",
        help="Use clean longitude/latitude or noisy longitude_n/latitude_n.",
    )
    p.add_argument(
        "--cell-sizes",
        nargs="+",
        type=float,
        default=[0.05, 0.02, 0.01, 0.005, 0.002, 0.001],
        help="Grid sizes in degrees.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top bins to keep for each grid size.",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional limit on number of parquet files when input is a directory. 0 means all files.",
    )
    p.add_argument(
        "--row-stride",
        type=int,
        default=10,
        help=(
            "Keep every Nth row before histogramming. "
            "Default 10 reduces preview cost; set 1 for full resolution."
        ),
    )
    p.add_argument(
        "--output-dir",
        default="./bin/density_survey",
        help="Directory for JSON and CSV outputs.",
    )
    return p


def _resolve_parquet_files(path_value: str, max_files: int) -> list[Path]:
    path = Path(path_value)
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")
    files = sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {path}")
    if int(max_files) > 0:
        files = files[: int(max_files)]
    return files


def _coordinate_columns(coord_field: str) -> tuple[str, str]:
    if coord_field == "noisy":
        return "longitude_n", "latitude_n"
    return "longitude", "latitude"


def _extent(files: list[Path], lon_col: str, lat_col: str) -> dict[str, float]:
    df = (
        pl.scan_parquet([str(p) for p in files])
        .filter(
            pl.col(lon_col).is_not_null()
            & pl.col(lat_col).is_not_null()
            & pl.col(lon_col).is_finite()
            & pl.col(lat_col).is_finite()
        )
        .select(
            [
                pl.col(lon_col).min().alias("min_lon"),
                pl.col(lon_col).max().alias("max_lon"),
                pl.col(lat_col).min().alias("min_lat"),
                pl.col(lat_col).max().alias("max_lat"),
                pl.len().alias("n_points"),
            ]
        )
        .collect()
        .to_dicts()[0]
    )
    return {
        "min_lon": float(df["min_lon"]),
        "max_lon": float(df["max_lon"]),
        "min_lat": float(df["min_lat"]),
        "max_lat": float(df["max_lat"]),
        "n_points": int(df["n_points"]),
    }


def _survey_one_grid(
    files: list[Path],
    lon_col: str,
    lat_col: str,
    cell: float,
    top_k: int,
    row_stride: int,
) -> list[dict]:
    cell_f = float(cell)
    lf = pl.scan_parquet([str(p) for p in files]).filter(
        pl.col(lon_col).is_not_null()
        & pl.col(lat_col).is_not_null()
        & pl.col(lon_col).is_finite()
        & pl.col(lat_col).is_finite()
    )
    stride = max(1, int(row_stride))
    if stride > 1:
        lf = lf.with_row_index("rid").filter((pl.col("rid") % stride) == 0).drop("rid")
    df = (
        lf
        .select(
            [
                ((pl.col(lon_col) / cell_f).floor() * cell_f).alias("lon_bin"),
                ((pl.col(lat_col) / cell_f).floor() * cell_f).alias("lat_bin"),
            ]
        )
        .group_by(["lon_bin", "lat_bin"])
        .agg(pl.len().alias("points"))
        .sort("points", descending=True)
        .limit(int(top_k))
        .collect()
    )

    out: list[dict] = []
    for row in df.to_dicts():
        lon_bin = float(row["lon_bin"])
        lat_bin = float(row["lat_bin"])
        out.append(
            {
                "cell_size_deg": cell_f,
                "lon_bin": lon_bin,
                "lat_bin": lat_bin,
                "points": int(row["points"]),
                "bbox": {
                    "min_lon": lon_bin,
                    "max_lon": lon_bin + cell_f,
                    "min_lat": lat_bin,
                    "max_lat": lat_bin + cell_f,
                },
                "center": {
                    "lon": lon_bin + (cell_f * 0.5),
                    "lat": lat_bin + (cell_f * 0.5),
                },
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "cell_size_deg",
                "points",
                "lon_bin",
                "lat_bin",
                "min_lon",
                "max_lon",
                "min_lat",
                "max_lat",
                "center_lon",
                "center_lat",
            ]
        )
        for idx, row in enumerate(rows, start=1):
            bbox = row["bbox"]
            center = row["center"]
            writer.writerow(
                [
                    idx,
                    row["cell_size_deg"],
                    row["points"],
                    row["lon_bin"],
                    row["lat_bin"],
                    bbox["min_lon"],
                    bbox["max_lon"],
                    bbox["min_lat"],
                    bbox["max_lat"],
                    center["lon"],
                    center["lat"],
                ]
            )


def main() -> None:
    args = _build_parser().parse_args()
    parquet_files = _resolve_parquet_files(args.input, max_files=int(args.max_files))
    lon_col, lat_col = _coordinate_columns(args.coord_field)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extent = _extent(parquet_files, lon_col, lat_col)
    summary: dict[str, object] = {
        "input": str(args.input),
        "coord_field": str(args.coord_field),
        "row_stride": int(args.row_stride),
        "files": [str(p) for p in parquet_files],
        "n_files": int(len(parquet_files)),
        "extent": extent,
        "top_bins": {},
    }

    for cell in [float(x) for x in args.cell_sizes]:
        rows = _survey_one_grid(
            parquet_files,
            lon_col,
            lat_col,
            cell,
            int(args.top_k),
            int(args.row_stride),
        )
        key = f"{cell:.6f}"
        summary["top_bins"][key] = rows
        csv_path = output_dir / f"density_top_cell_{key}.csv"
        _write_csv(csv_path, rows)

    json_path = output_dir / "density_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
