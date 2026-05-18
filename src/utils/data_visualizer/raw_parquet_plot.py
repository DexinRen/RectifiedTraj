#!/usr/bin/env python3
"""Render raw parquet lon/lat data as a density heatmap."""

from __future__ import annotations

import argparse
import json
import math
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

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot raw parquet lon/lat data as a density heatmap."
    )
    p.add_argument("--input", required=True, help="Raw parquet file or directory of parquet files.")
    p.add_argument(
        "--coord-field",
        choices=["clean", "noisy"],
        default="clean",
        help="Use clean longitude/latitude or noisy longitude_n/latitude_n.",
    )
    p.add_argument(
        "--cell-size",
        type=float,
        default=0.005,
        help="Grid size in degrees for the density heatmap.",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional limit on number of parquet files when input is a directory. 0 means all files.",
    )
    p.add_argument(
        "--output-dir",
        default="./bin/raw_parquet_plots",
        help="Directory for the PNG and JSON summary.",
    )
    p.add_argument(
        "--title",
        default="",
        help="Optional figure title.",
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
    p.add_argument("--min-lon", type=float, default=None, help="Optional bbox filter min longitude.")
    p.add_argument("--max-lon", type=float, default=None, help="Optional bbox filter max longitude.")
    p.add_argument("--min-lat", type=float, default=None, help="Optional bbox filter min latitude.")
    p.add_argument("--max-lat", type=float, default=None, help="Optional bbox filter max latitude.")
    p.add_argument("--center-lon", type=float, default=None, help="Optional center longitude.")
    p.add_argument("--center-lat", type=float, default=None, help="Optional center latitude.")
    p.add_argument(
        "--radius-miles",
        type=float,
        default=None,
        help="Optional radius in miles around --center-lon/--center-lat.",
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


def _miles_radius_to_bbox(center_lon: float, center_lat: float, radius_miles: float) -> dict[str, float]:
    radius_km = float(radius_miles) * 1.609344
    lat_delta = float(radius_km) / 110.574
    cos_lat = math.cos(math.radians(float(center_lat)))
    lon_delta = float(radius_km) / max(111.320 * abs(cos_lat), 1e-9)
    return {
        "min_lon": float(center_lon) - lon_delta,
        "max_lon": float(center_lon) + lon_delta,
        "min_lat": float(center_lat) - lat_delta,
        "max_lat": float(center_lat) + lat_delta,
    }


def _resolve_bbox(args: argparse.Namespace) -> dict[str, float] | None:
    corners = [args.min_lon, args.max_lon, args.min_lat, args.max_lat]
    if any(v is not None for v in corners):
        if not all(v is not None for v in corners):
            raise ValueError("If any bbox corner is set, all of min/max lon/lat must be set.")
        if float(args.min_lon) >= float(args.max_lon):
            raise ValueError("min-lon must be smaller than max-lon.")
        if float(args.min_lat) >= float(args.max_lat):
            raise ValueError("min-lat must be smaller than max-lat.")
        return {
            "min_lon": float(args.min_lon),
            "max_lon": float(args.max_lon),
            "min_lat": float(args.min_lat),
            "max_lat": float(args.max_lat),
        }

    center_vals = [args.center_lon, args.center_lat, args.radius_miles]
    if any(v is not None for v in center_vals):
        if not all(v is not None for v in center_vals):
            raise ValueError("center-lon, center-lat, and radius-miles must be set together.")
        if float(args.radius_miles) <= 0.0:
            raise ValueError("radius-miles must be positive.")
        bbox = _miles_radius_to_bbox(
            center_lon=float(args.center_lon),
            center_lat=float(args.center_lat),
            radius_miles=float(args.radius_miles),
        )
        bbox["center_lon"] = float(args.center_lon)
        bbox["center_lat"] = float(args.center_lat)
        bbox["radius_miles"] = float(args.radius_miles)
        return bbox

    return None


def _density_histogram(
    files: list[Path],
    lon_col: str,
    lat_col: str,
    cell_size: float,
    row_stride: int,
    bbox: dict[str, float] | None = None,
) -> tuple[pl.DataFrame, dict[str, float]]:
    cell = float(cell_size)
    lf = pl.scan_parquet([str(p) for p in files]).filter(
        pl.col(lon_col).is_not_null()
        & pl.col(lat_col).is_not_null()
        & pl.col(lon_col).is_finite()
        & pl.col(lat_col).is_finite()
    )
    stride = max(1, int(row_stride))
    if stride > 1:
        lf = lf.with_row_index("rid").filter((pl.col("rid") % stride) == 0).drop("rid")
    if bbox is not None:
        lf = lf.filter(
            (pl.col(lon_col) >= float(bbox["min_lon"]))
            & (pl.col(lon_col) <= float(bbox["max_lon"]))
            & (pl.col(lat_col) >= float(bbox["min_lat"]))
            & (pl.col(lat_col) <= float(bbox["max_lat"]))
        )
    hist = (
        lf.select(
            [
                ((pl.col(lon_col) / cell).floor() * cell).alias("lon_bin"),
                ((pl.col(lat_col) / cell).floor() * cell).alias("lat_bin"),
            ]
        )
        .group_by(["lon_bin", "lat_bin"])
        .agg(pl.len().alias("points"))
        .sort(["lat_bin", "lon_bin"])
        .collect()
    )
    if hist.height <= 0:
        raise RuntimeError("No points found for the requested raw parquet plot region.")
    extent = {
        "min_lon": float(hist["lon_bin"].min()),
        "max_lon": float(hist["lon_bin"].max() + cell),
        "min_lat": float(hist["lat_bin"].min()),
        "max_lat": float(hist["lat_bin"].max() + cell),
        "n_bins": int(hist.height),
        "max_points_in_bin": int(hist["points"].max()),
    }
    return hist, extent


def _dense_matrix(hist: pl.DataFrame, cell_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cell = float(cell_size)
    lon_bins = np.sort(hist["lon_bin"].unique().to_numpy())
    lat_bins = np.sort(hist["lat_bin"].unique().to_numpy())
    lon_index = {float(v): i for i, v in enumerate(lon_bins.tolist())}
    lat_index = {float(v): i for i, v in enumerate(lat_bins.tolist())}
    matrix = np.zeros((len(lat_bins), len(lon_bins)), dtype=np.float64)

    for row in hist.iter_rows(named=True):
        x = lon_index[float(row["lon_bin"])]
        y = lat_index[float(row["lat_bin"])]
        matrix[y, x] = float(row["points"])

    lon_edges = np.concatenate([lon_bins, [float(lon_bins[-1] + cell)]])
    lat_edges = np.concatenate([lat_bins, [float(lat_bins[-1] + cell)]])
    return matrix, lon_edges, lat_edges


def _plot(matrix: np.ndarray, lon_edges: np.ndarray, lat_edges: np.ndarray, output_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 10), facecolor="#f7f3ea")
    ax.set_facecolor("#faf7f0")
    display = np.log1p(matrix)
    im = ax.imshow(
        display,
        origin="lower",
        cmap="inferno",
        extent=(float(lon_edges[0]), float(lon_edges[-1]), float(lat_edges[0]), float(lat_edges[-1])),
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("log(1 + points per cell)")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _build_parser().parse_args()
    files = _resolve_parquet_files(args.input, int(args.max_files))
    lon_col, lat_col = _coordinate_columns(args.coord_field)
    cell_size = float(args.cell_size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bbox = _resolve_bbox(args)

    hist, extent = _density_histogram(
        files,
        lon_col,
        lat_col,
        cell_size,
        row_stride=int(args.row_stride),
        bbox=bbox,
    )
    matrix, lon_edges, lat_edges = _dense_matrix(hist, cell_size)

    base_name = Path(args.input).stem if Path(args.input).is_file() else Path(args.input).name
    slug = f"{base_name}_{args.coord_field}_cell_{cell_size:.6f}".replace(".", "p")
    if bbox is not None:
        slug += "_region"
    png_path = output_dir / f"{slug}.png"
    json_path = output_dir / f"{slug}.json"

    if args.title.strip():
        title = args.title.strip()
    elif bbox is not None and "radius_miles" in bbox:
        title = (
            f"{base_name} raw density ({args.coord_field}, cell={cell_size}, "
            f"radius={bbox['radius_miles']}mi)"
        )
    elif bbox is not None:
        title = f"{base_name} raw density ({args.coord_field}, regional crop, cell={cell_size})"
    else:
        title = f"{base_name} raw density ({args.coord_field}, cell={cell_size})"
    _plot(matrix, lon_edges, lat_edges, png_path, title)

    top = hist.sort("points", descending=True).head(20).to_dicts()
    summary = {
        "input": str(args.input),
        "coord_field": args.coord_field,
        "cell_size": cell_size,
        "row_stride": int(args.row_stride),
        "files": [str(p) for p in files],
        "n_files": int(len(files)),
        "extent": extent,
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "requested_bbox": bbox,
        "plot_file": str(png_path),
        "top_bins": [
            {
                "lon_bin": float(row["lon_bin"]),
                "lat_bin": float(row["lat_bin"]),
                "points": int(row["points"]),
                "bbox": {
                    "min_lon": float(row["lon_bin"]),
                    "max_lon": float(row["lon_bin"] + cell_size),
                    "min_lat": float(row["lat_bin"]),
                    "max_lat": float(row["lat_bin"] + cell_size),
                },
            }
            for row in top
        ],
    }
    json_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
