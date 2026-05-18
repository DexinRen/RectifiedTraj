#!/usr/bin/env python3
"""Render case-study trajectory plots with a shared percent-to-style anchor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch


DEFAULT_OUTPUT_DIR = Path("bin/plots/case_by_case")
DEFAULT_PLOT_KIND = "full_noise"
DEFAULT_FIGSIZE = 8.4
DEFAULT_DPI = 150
DEFAULT_ANCHOR_WIDTH = 6.0
DEFAULT_MIN_WIDTH = 0.35
DEFAULT_MAX_WIDTH = 10.0
DEFAULT_PERCENT_ANCHOR = 30.0
DEFAULT_PERCENT_CAP = 120.0
DEFAULT_POINT_COLOR_CAP = 300.0
DEFAULT_DRAW_BINS = 48
DEFAULT_HALO_WIDTH = 0.45
DEFAULT_POINT_CMAP_MIN = 0.18
DEFAULT_POINT_CMAP_MAX = 0.90
DEFAULT_POINT_COLOR_SHIFT = 0.04

LOW_COLOR = np.array([44, 123, 182], dtype=np.float64) / 255.0
ANCHOR_COLOR = np.array([215, 48, 39], dtype=np.float64) / 255.0
HIGH_COLOR = np.array([103, 0, 13], dtype=np.float64) / 255.0
HALO_COLOR = np.array([1.0, 1.0, 1.0], dtype=np.float64)
FIXED_GREY_BLUE = np.array([31, 119, 180], dtype=np.float64) / 255.0
TEST_MODEL_LABELS = {
    "ResidualReg_transformer_10M_20260303_103644",
    "savgol",
    "RectifiedTraj_hybrid_10M_20251205_011946",
}
POINT_CMAP = matplotlib.colormaps["BuPu"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render case-study trajectory plots from processed PT files."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Trajectory PT file or a directory containing case-study export subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for rendered PNGs.",
    )
    parser.add_argument(
        "--plot-kind",
        choices=["full_noise", "denoised_error_percentage"],
        default=DEFAULT_PLOT_KIND,
        help="Case-study plot variant to render.",
    )
    parser.add_argument("--min-lon", type=float, default=None, help="Optional bbox minimum longitude override.")
    parser.add_argument("--max-lon", type=float, default=None, help="Optional bbox maximum longitude override.")
    parser.add_argument("--min-lat", type=float, default=None, help="Optional bbox minimum latitude override.")
    parser.add_argument("--max-lat", type=float, default=None, help="Optional bbox maximum latitude override.")
    parser.add_argument("--figsize", type=float, default=DEFAULT_FIGSIZE, help="Square figure size in inches.")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Output DPI.")
    parser.add_argument(
        "--anchor-width",
        type=float,
        default=DEFAULT_ANCHOR_WIDTH,
        help="Line width for the 100%% anchor.",
    )
    parser.add_argument(
        "--min-width",
        type=float,
        default=DEFAULT_MIN_WIDTH,
        help="Line width at 0%%.",
    )
    parser.add_argument(
        "--max-width",
        type=float,
        default=DEFAULT_MAX_WIDTH,
        help="Line width at or above --percent-cap.",
    )
    parser.add_argument(
        "--percent-cap",
        type=float,
        default=DEFAULT_PERCENT_CAP,
        help="Upper percent cap used for linewidth clipping in denoised plots.",
    )
    parser.add_argument(
        "--percent-anchor",
        type=float,
        default=DEFAULT_PERCENT_ANCHOR,
        help="Percent value mapped to --anchor-width for denoised plots.",
    )
    parser.add_argument(
        "--point-color-cap",
        type=float,
        default=DEFAULT_POINT_COLOR_CAP,
        help="Upper percent cap used only for pointwise color mapping in denoised plots.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.85,
        help="Global line alpha.",
    )
    parser.add_argument(
        "--draw-bins",
        type=int,
        default=DEFAULT_DRAW_BINS,
        help="Number of low-to-high error bins used for draw ordering.",
    )
    parser.add_argument(
        "--halo-width",
        type=float,
        default=DEFAULT_HALO_WIDTH,
        help="Extra white under-stroke width used to separate dense segments in denoised plots.",
    )
    parser.add_argument(
        "--width-mode",
        choices=["trajectory", "overall"],
        default="trajectory",
        help="Use trajectory-wise or overall-model error percentage to determine linewidth.",
    )
    parser.add_argument(
        "--width-offset",
        type=float,
        default=0.0,
        help="Additive offset applied to rendered denoised linewidths after mapping.",
    )
    parser.add_argument(
        "--color-mode",
        choices=["pointwise", "trajectory", "fixed"],
        default="pointwise",
        help="Use pointwise error colors, trajectory-wise mean error colors, or one fixed color for all denoised segments.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Only render the three test models used for fast style iteration.",
    )
    return parser


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_payload(input_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    blob = torch.load(input_path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        raise ValueError(f"Unsupported PT payload in {input_path}: {type(blob)}")
    rows = blob.get("trajectories")
    if not isinstance(rows, list):
        raise ValueError(f"No trajectories list in {input_path}")
    metadata = blob.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return rows, metadata


def _load_summary_payload(input_path: Path) -> dict[str, Any]:
    summary_path = input_path.parent / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        data = json.loads(summary_path.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _iter_input_pt_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    candidates = sorted(input_path.rglob("trajectories.pt"))
    if not candidates:
        raise FileNotFoundError(f"No trajectories.pt files found under {input_path}")
    return candidates


def _resolve_bbox(metadata: dict[str, Any], args: argparse.Namespace) -> tuple[float, float, float, float]:
    region_bbox = metadata.get("region_bbox")
    if isinstance(region_bbox, dict):
        min_lon = float(region_bbox["min_lon"])
        max_lon = float(region_bbox["max_lon"])
        min_lat = float(region_bbox["min_lat"])
        max_lat = float(region_bbox["max_lat"])
    else:
        min_lon = max_lon = min_lat = max_lat = None
    if args.min_lon is not None:
        min_lon = float(args.min_lon)
    if args.max_lon is not None:
        max_lon = float(args.max_lon)
    if args.min_lat is not None:
        min_lat = float(args.min_lat)
    if args.max_lat is not None:
        max_lat = float(args.max_lat)
    if None in {min_lon, max_lon, min_lat, max_lat}:
        raise ValueError("Bounding box is missing. Provide --min/--max lon/lat or use a PT file with metadata.region_bbox.")
    return float(min_lon), float(max_lon), float(min_lat), float(max_lat)


def _inside_bbox(xy: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    min_lon, max_lon, min_lat, max_lat = bbox
    return (
        np.isfinite(xy[:, 0])
        & np.isfinite(xy[:, 1])
        & (xy[:, 0] >= min_lon)
        & (xy[:, 0] <= max_lon)
        & (xy[:, 1] >= min_lat)
        & (xy[:, 1] <= max_lat)
    )


def _color_for_percent(percent: float, *, percent_cap: float) -> np.ndarray:
    p = max(0.0, float(percent))
    cap = max(float(percent_cap), 100.0)
    if p <= 100.0:
        t = p / 100.0
        return (1.0 - t) * LOW_COLOR + t * ANCHOR_COLOR
    if cap <= 100.0:
        return HIGH_COLOR.copy()
    t = min((p - 100.0) / (cap - 100.0), 1.0)
    return (1.0 - t) * ANCHOR_COLOR + t * HIGH_COLOR


def _colors_for_percent_array(percent: np.ndarray, *, percent_cap: float) -> np.ndarray:
    p = np.asarray(percent, dtype=np.float64)
    p = np.where(np.isfinite(p), p, float(percent_cap))
    p = np.clip(p, 0.0, max(float(percent_cap), 100.0))
    cap = max(float(percent_cap), 100.0)

    colors = np.empty((p.shape[0], 3), dtype=np.float64)
    low_mask = p <= 100.0
    if np.any(low_mask):
        t = (p[low_mask] / 100.0)[:, None]
        colors[low_mask] = (1.0 - t) * LOW_COLOR[None, :] + t * ANCHOR_COLOR[None, :]
    if np.any(~low_mask):
        if cap <= 100.0:
            colors[~low_mask] = HIGH_COLOR[None, :]
        else:
            t = ((p[~low_mask] - 100.0) / (cap - 100.0))[:, None]
            t = np.clip(t, 0.0, 1.0)
            colors[~low_mask] = (1.0 - t) * ANCHOR_COLOR[None, :] + t * HIGH_COLOR[None, :]
    return colors


def _width_for_percent(
    percent: float,
    *,
    anchor_width: float,
    min_width: float,
    max_width: float,
    percent_cap: float,
) -> float:
    p = max(0.0, float(percent))
    cap = max(float(percent_cap), 100.0)
    if p <= 100.0:
        return float(min_width) + (float(anchor_width) - float(min_width)) * (p / 100.0)
    if cap <= 100.0:
        return float(max_width)
    t = min((p - 100.0) / (cap - 100.0), 1.0)
    return float(anchor_width) + (float(max_width) - float(anchor_width)) * t


def _widths_for_percent_array(
    percent: np.ndarray,
    *,
    anchor_width: float,
    min_width: float,
    max_width: float,
    percent_cap: float,
) -> np.ndarray:
    p = np.asarray(percent, dtype=np.float64)
    p = np.where(np.isfinite(p), p, float(percent_cap))
    p = np.clip(p, 0.0, max(float(percent_cap), 100.0))
    cap = max(float(percent_cap), 100.0)

    widths = np.empty(p.shape[0], dtype=np.float64)
    low_mask = p <= 100.0
    if np.any(low_mask):
        widths[low_mask] = float(min_width) + (float(anchor_width) - float(min_width)) * (
            p[low_mask] / 100.0
        )
    if np.any(~low_mask):
        if cap <= 100.0:
            widths[~low_mask] = float(max_width)
        else:
            t = np.clip((p[~low_mask] - 100.0) / (cap - 100.0), 0.0, 1.0)
            widths[~low_mask] = float(anchor_width) + (float(max_width) - float(anchor_width)) * t
    return widths


def _segment_metric_array(
    percent: np.ndarray,
    *,
    percent_cap: float,
) -> np.ndarray:
    p = np.asarray(percent, dtype=np.float64)
    p = np.where(np.isfinite(p), p, float(percent_cap))
    p = np.clip(p, 0.0, max(float(percent_cap), 1.0))
    return p


def _widths_for_denoised_segments(
    percent: np.ndarray,
    *,
    anchor_width: float,
    min_width: float,
    max_width: float,
    percent_anchor: float,
    percent_cap: float,
    width_offset: float = 0.0,
) -> np.ndarray:
    p = _segment_metric_array(percent, percent_cap=percent_cap)
    anchor = max(float(percent_anchor), 1e-6)
    cap = max(float(percent_cap), anchor)
    widths = np.empty(p.shape[0], dtype=np.float64)

    low_mask = p <= anchor
    if np.any(low_mask):
        t = np.sqrt(np.clip(p[low_mask] / anchor, 0.0, 1.0))
        widths[low_mask] = float(min_width) + (float(anchor_width) - float(min_width)) * t

    if np.any(~low_mask):
        if cap <= anchor:
            widths[~low_mask] = float(max_width)
        else:
            t = np.sqrt(np.clip((p[~low_mask] - anchor) / (cap - anchor), 0.0, 1.0))
            widths[~low_mask] = float(anchor_width) + (float(max_width) - float(anchor_width)) * t

    widths = widths + float(width_offset)
    return np.clip(widths, 0.05, None)


def _colors_for_point_error_array(
    percent: np.ndarray,
    *,
    percent_anchor: float,
    percent_cap: float,
) -> np.ndarray:
    del percent_anchor
    p = _segment_metric_array(percent, percent_cap=percent_cap)
    cap = max(float(percent_cap), 1e-6)
    # Use one global sequential map for all models; sqrt spacing expands the
    # low-error region where most strong models lie without changing ordering.
    t = np.sqrt(np.clip(p / cap, 0.0, 1.0))
    t = DEFAULT_POINT_COLOR_SHIFT + (1.0 - DEFAULT_POINT_COLOR_SHIFT) * t
    t = DEFAULT_POINT_CMAP_MIN + (DEFAULT_POINT_CMAP_MAX - DEFAULT_POINT_CMAP_MIN) * t
    rgba = POINT_CMAP(np.clip(t, 0.0, 1.0))
    return np.asarray(rgba[:, :3], dtype=np.float64)


def _model_label_from_metadata(metadata: dict[str, Any], input_path: Path) -> str:
    for key in ["model_label", "model_name", "display_name"]:
        value = metadata.get(key)
        if str(value or "").strip():
            return str(value).strip()
    return input_path.parent.name or input_path.stem


def _finite_mean(values: np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(np.mean(finite, dtype=np.float64))


def _overall_percent_for_plot(
    rows: list[dict[str, Any]],
    input_path: Path,
) -> float:
    summary = _load_summary_payload(input_path)
    summary_value = summary.get("mean_error_percentage_finite")
    try:
        if summary_value is not None and np.isfinite(float(summary_value)):
            return float(summary_value)
    except Exception:
        pass

    means: list[float] = []
    for row in rows:
        pct = _to_numpy(row.get("error_percentage"))
        value = _finite_mean(pct)
        if value is not None:
            means.append(value)
    if means:
        return float(np.mean(np.asarray(means, dtype=np.float64), dtype=np.float64))
    return 0.0


def _segment_colors_for_plot(
    seg_pct: np.ndarray,
    *,
    color_mode: str,
    percent_anchor: float,
    percent_cap: float,
    point_color_cap: float,
    trajectory_percent: float | None = None,
) -> np.ndarray:
    n = int(np.asarray(seg_pct).shape[0])
    if str(color_mode) == "fixed":
        return np.repeat(FIXED_GREY_BLUE[None, :], n, axis=0).astype(np.float32, copy=False)
    if str(color_mode) == "trajectory":
        traj_pct = 0.0 if trajectory_percent is None else float(trajectory_percent)
        return _colors_for_point_error_array(
            np.full(n, traj_pct, dtype=np.float64),
            percent_anchor=percent_anchor,
            percent_cap=point_color_cap,
        ).astype(np.float32, copy=False)
    return _colors_for_point_error_array(
        np.asarray(seg_pct, dtype=np.float64),
        percent_anchor=percent_anchor,
        percent_cap=point_color_cap,
    ).astype(np.float32, copy=False)


def _safe_filename(value: str) -> str:
    text = str(value or "").strip()
    text = text.replace(" ", "_")
    text = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in text)
    text = text.strip("._") or "plot"
    return text


def _contiguous_segments(mask: np.ndarray) -> list[np.ndarray]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    return [segment for segment in np.split(idx, splits) if segment.size >= 2]


def _plot_denoised_error_percentage(
    rows: list[dict[str, Any]],
    *,
    input_path: Path,
    bbox: tuple[float, float, float, float],
    output_path: Path,
    figsize: float,
    dpi: int,
    anchor_width: float,
    min_width: float,
    max_width: float,
    percent_anchor: float,
    percent_cap: float,
    alpha: float,
    draw_bins: int,
    halo_width: float,
    width_mode: str,
    color_mode: str,
    width_offset: float,
    point_color_cap: float,
) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(float(figsize), float(figsize)), dpi=int(dpi))
    visible_trajectories = 0
    visible_points = 0
    visible_segments = 0
    all_segments: list[np.ndarray] = []
    all_segment_colors: list[np.ndarray] = []
    all_segment_widths: list[np.ndarray] = []
    all_draw_metrics: list[np.ndarray] = []
    overall_percent = _overall_percent_for_plot(rows, input_path) if str(width_mode) == "overall" else None
    overall_width = None
    if overall_percent is not None:
        overall_width = float(
            _widths_for_denoised_segments(
                np.asarray([overall_percent], dtype=np.float64),
                anchor_width=anchor_width,
                min_width=min_width,
                max_width=max_width,
                percent_anchor=percent_anchor,
                percent_cap=percent_cap,
                width_offset=width_offset,
            )[0]
        )

    for row in rows:
        denoised = _to_numpy(row.get("denoised"))
        error_pct = _to_numpy(row.get("error_percentage"))
        if denoised.ndim != 2 or denoised.shape[1] < 2 or denoised.shape[0] <= 1:
            continue
        pct = np.asarray(error_pct, dtype=np.float64).reshape(-1)
        xy = np.asarray(denoised[:, :2], dtype=np.float64)
        n = min(int(xy.shape[0]), int(pct.shape[0]))
        if n <= 1:
            continue
        xy = xy[:n]
        pct = _segment_metric_array(pct[:n], percent_cap=percent_cap)

        point_mask = _inside_bbox(xy, bbox)
        seg_mask = point_mask[:-1] & point_mask[1:]
        if not np.any(seg_mask):
            continue

        seg_xy = np.stack([xy[:-1][seg_mask], xy[1:][seg_mask]], axis=1).astype(np.float32, copy=False)
        seg_pct = 0.5 * (pct[:-1][seg_mask] + pct[1:][seg_mask])
        traj_pct = float(np.mean(pct, dtype=np.float64))
        if overall_width is not None:
            traj_width = float(overall_width)
            draw_metric = traj_pct if str(color_mode) == "trajectory" else float(overall_percent)
        else:
            traj_width = float(
                _widths_for_denoised_segments(
                    np.asarray([traj_pct], dtype=np.float64),
                    anchor_width=anchor_width,
                    min_width=min_width,
                    max_width=max_width,
                    percent_anchor=percent_anchor,
                    percent_cap=percent_cap,
                    width_offset=width_offset,
                )[0]
            )
            draw_metric = traj_pct
        all_segments.append(seg_xy)
        all_segment_colors.append(
            _segment_colors_for_plot(
                np.asarray(seg_pct, dtype=np.float64),
                color_mode=str(color_mode),
                percent_anchor=percent_anchor,
                percent_cap=percent_cap,
                point_color_cap=point_color_cap,
                trajectory_percent=traj_pct,
            )
        )
        all_segment_widths.append(np.full(seg_xy.shape[0], traj_width, dtype=np.float32))
        all_draw_metrics.append(np.full(seg_xy.shape[0], draw_metric, dtype=np.float32))
        visible_trajectories += 1
        visible_points += int(np.sum(point_mask))
        visible_segments += int(seg_xy.shape[0])

    if all_segments:
        seg_xy_all = np.concatenate(all_segments, axis=0)
        seg_colors_all = np.concatenate(all_segment_colors, axis=0)
        seg_widths_all = np.concatenate(all_segment_widths, axis=0)
        seg_draw_metric_all = _segment_metric_array(np.concatenate(all_draw_metrics, axis=0), percent_cap=percent_cap)
        n_bins = max(1, int(draw_bins))
        cap = max(float(percent_cap), float(percent_anchor), 1.0)
        if cap <= 0.0:
            bin_ids = np.zeros(seg_draw_metric_all.shape[0], dtype=np.int32)
        else:
            bin_ids = np.floor((seg_draw_metric_all / cap) * n_bins).astype(np.int32)
            bin_ids = np.clip(bin_ids, 0, n_bins - 1)

        for bin_idx in range(n_bins):
            sel = bin_ids == bin_idx
            if not np.any(sel):
                continue
            if float(halo_width) > 0.0:
                halo = LineCollection(
                    seg_xy_all[sel],
                    colors=HALO_COLOR[None, :],
                    linewidths=seg_widths_all[sel] + float(halo_width),
                    alpha=1.0,
                    capstyle="projecting",
                    joinstyle="miter",
                )
                ax.add_collection(halo)
            collection = LineCollection(
                seg_xy_all[sel],
                colors=seg_colors_all[sel],
                linewidths=seg_widths_all[sel],
                alpha=float(alpha),
                capstyle="projecting",
                joinstyle="miter",
            )
            ax.add_collection(collection)

    min_lon, max_lon, min_lat, max_lat = bbox
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    return {
        "visible_trajectories": int(visible_trajectories),
        "visible_points": int(visible_points),
        "visible_segments": int(visible_segments),
        "width_mode": str(width_mode),
        "color_mode": str(color_mode),
        "overall_percent": None if overall_percent is None else float(overall_percent),
        "overall_width": None if overall_width is None else float(overall_width),
        "width_offset": float(width_offset),
    }


def _plot_full_noise(
    rows: list[dict[str, Any]],
    *,
    bbox: tuple[float, float, float, float],
    output_path: Path,
    figsize: float,
    dpi: int,
    anchor_width: float,
    min_width: float,
    max_width: float,
    percent_cap: float,
    alpha: float,
) -> dict[str, Any]:
    percent = 100.0
    color = _color_for_percent(percent, percent_cap=percent_cap)
    width = _width_for_percent(
        percent,
        anchor_width=anchor_width,
        min_width=min_width,
        max_width=max_width,
        percent_cap=percent_cap,
    )

    fig, ax = plt.subplots(figsize=(float(figsize), float(figsize)), dpi=int(dpi))
    visible_trajectories = 0
    visible_points = 0

    for row in rows:
        noisy = _to_numpy(row.get("data"))
        if noisy.ndim != 2 or noisy.shape[1] < 2 or noisy.shape[0] <= 1:
            continue
        xy = np.asarray(noisy[:, :2], dtype=np.float64)
        mask = _inside_bbox(xy, bbox)
        segments = _contiguous_segments(mask)
        if not segments:
            continue
        visible_trajectories += 1
        visible_points += int(np.sum(mask))
        for segment in segments:
            seg_xy = xy[segment]
            ax.plot(seg_xy[:, 0], seg_xy[:, 1], color=color, linewidth=width, alpha=float(alpha))

    min_lon, max_lon, min_lat, max_lat = bbox
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    return {
        "visible_trajectories": int(visible_trajectories),
        "visible_points": int(visible_points),
        "anchor_percent": percent,
        "anchor_width": float(width),
        "anchor_color_rgb": [float(x) for x in color],
    }


def main() -> None:
    args = _build_parser().parse_args()
    input_path = Path(str(args.input))
    output_dir = Path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for pt_path in _iter_input_pt_files(input_path):
        rows, metadata = _load_payload(pt_path)
        bbox = _resolve_bbox(metadata, args)
        model_label = _model_label_from_metadata(metadata, pt_path)

        if bool(args.test) and str(args.plot_kind) == "denoised_error_percentage":
            if model_label not in TEST_MODEL_LABELS:
                continue

        if str(args.plot_kind) == "full_noise":
            output_path = output_dir / f"{str(args.plot_kind)}_{pt_path.stem}.png"
            result = _plot_full_noise(
                rows,
                bbox=bbox,
                output_path=output_path,
                figsize=float(args.figsize),
                dpi=int(args.dpi),
                anchor_width=float(args.anchor_width),
                min_width=float(args.min_width),
                max_width=float(args.max_width),
                percent_cap=float(args.percent_cap),
                alpha=float(args.alpha),
            )
        elif str(args.plot_kind) == "denoised_error_percentage":
            output_path = output_dir / f"{_safe_filename(model_label)}.png"
            result = _plot_denoised_error_percentage(
                rows,
                input_path=pt_path,
                bbox=bbox,
                output_path=output_path,
                figsize=float(args.figsize),
                dpi=int(args.dpi),
                anchor_width=float(args.anchor_width),
                min_width=float(args.min_width),
                max_width=float(args.max_width),
                percent_anchor=float(args.percent_anchor),
                percent_cap=float(args.percent_cap),
                alpha=float(args.alpha),
                draw_bins=int(args.draw_bins),
                halo_width=float(args.halo_width),
                width_mode=str(args.width_mode),
                color_mode=str(args.color_mode),
                width_offset=float(args.width_offset),
                point_color_cap=float(args.point_color_cap),
            )
            result["model_label"] = model_label
        else:
            raise ValueError(f"Unsupported plot kind: {args.plot_kind}")

        results.append(
            {
                "input_path": str(pt_path),
                "output_path": str(output_path),
                "plot_kind": str(args.plot_kind),
                "bbox": {
                    "min_lon": bbox[0],
                    "max_lon": bbox[1],
                    "min_lat": bbox[2],
                    "max_lat": bbox[3],
                },
                **result,
            }
        )
    print(results if len(results) > 1 else results[0])


if __name__ == "__main__":
    main()
