#!/usr/bin/env python3
"""Render multiple case-study visualization variants for side-by-side comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch


DEFAULT_INPUT_DIR = Path("bin/plots/raw_data")
DEFAULT_OUTPUT_DIR = Path("bin/plots/case_by_case_compare")
DEFAULT_FIGSIZE = 8.4
DEFAULT_DPI = 150
DEFAULT_MIN_WIDTH = 0.45
DEFAULT_ANCHOR_WIDTH = 6.0
DEFAULT_MAX_WIDTH = 8.5
DEFAULT_ALPHA = 0.55
DEFAULT_DRAW_BINS = 48
DEFAULT_SAMPLE_PER_FILE = 100_000

PALETTES: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
    "teal_amber_brown": (
        np.array([44, 162, 150], dtype=np.float64) / 255.0,
        np.array([230, 159, 0], dtype=np.float64) / 255.0,
        np.array([112, 66, 20], dtype=np.float64) / 255.0,
    ),
}

PRESETS: list[dict[str, Any]] = [
    {
        "name": "global_rank_sqrt_teal",
        "mode": "global_rank",
        "palette": "teal_amber_brown",
        "anchor_percent": 100.0,
        "cap_percent": 200.0,
        "curve": "sqrt",
    },
    {
        "name": "anchor30_sqrt_teal",
        "mode": "absolute",
        "palette": "teal_amber_brown",
        "anchor_percent": 30.0,
        "cap_percent": 120.0,
        "curve": "sqrt",
    },
    {
        "name": "anchor30_quadratic_teal",
        "mode": "absolute",
        "palette": "teal_amber_brown",
        "anchor_percent": 30.0,
        "cap_percent": 120.0,
        "curve": "quadratic",
    },
    {
        "name": "anchor30_tangent_teal",
        "mode": "absolute",
        "palette": "teal_amber_brown",
        "anchor_percent": 30.0,
        "cap_percent": 120.0,
        "curve": "tangent",
    },
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render multiple case-study visualization variants from saved raw_data "
            "trajectory exports."
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT_DIR), help="Root directory containing case-study raw_data exports.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Root directory for rendered comparison folders.")
    parser.add_argument("--figsize", type=float, default=DEFAULT_FIGSIZE, help="Square figure size in inches.")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Output DPI.")
    parser.add_argument("--min-width", type=float, default=DEFAULT_MIN_WIDTH, help="Minimum line width.")
    parser.add_argument("--anchor-width", type=float, default=DEFAULT_ANCHOR_WIDTH, help="Anchor line width.")
    parser.add_argument("--max-width", type=float, default=DEFAULT_MAX_WIDTH, help="Maximum line width.")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Line alpha.")
    parser.add_argument("--draw-bins", type=int, default=DEFAULT_DRAW_BINS, help="Number of low-to-high error bins used for draw ordering.")
    parser.add_argument("--sample-per-file", type=int, default=DEFAULT_SAMPLE_PER_FILE, help="Approximate samples per file for global-rank reference.")
    return parser


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _safe_filename(value: str) -> str:
    text = str(value or "").strip()
    text = text.replace(" ", "_")
    text = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in text)
    return text.strip("._") or "plot"


def _iter_payloads(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    files = sorted(root.rglob("trajectories.pt"))
    if not files:
        raise FileNotFoundError(f"No trajectories.pt files found under {root}")
    return files


def _load_payload(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        raise ValueError(f"Unsupported payload at {path}: {type(blob)}")
    rows = blob.get("trajectories")
    metadata = blob.get("metadata", {})
    if not isinstance(rows, list):
        raise ValueError(f"No trajectories list in {path}")
    if not isinstance(metadata, dict):
        metadata = {}
    return rows, metadata


def _resolve_bbox(metadata: dict[str, Any]) -> tuple[float, float, float, float]:
    region_bbox = metadata.get("region_bbox")
    if not isinstance(region_bbox, dict):
        raise ValueError("metadata.region_bbox is required for comparison sweep.")
    return (
        float(region_bbox["min_lon"]),
        float(region_bbox["max_lon"]),
        float(region_bbox["min_lat"]),
        float(region_bbox["max_lat"]),
    )


def _model_label(metadata: dict[str, Any], path: Path) -> str:
    for key in ["model_label", "model_name", "display_name"]:
        value = metadata.get(key)
        if str(value or "").strip():
            return str(value).strip()
    return path.parent.name or path.stem


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


def _curve_unit(values: np.ndarray, curve: str) -> np.ndarray:
    x = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    if curve == "linear":
        return x
    if curve == "sqrt":
        return np.sqrt(x)
    if curve == "quadratic":
        return x * x
    if curve == "tangent":
        scale = 0.85 * (math.pi / 2.0)
        return np.tan(scale * x) / math.tan(scale)
    raise ValueError(f"Unsupported curve: {curve}")


def _global_rank_values(sample_sorted: np.ndarray, values: np.ndarray) -> np.ndarray:
    if sample_sorted.size == 0:
        return np.zeros_like(values, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    finite = np.where(np.isfinite(v), v, float(np.max(sample_sorted)))
    ranks = np.searchsorted(sample_sorted, finite, side="left").astype(np.float64)
    denom = max(float(sample_sorted.size - 1), 1.0)
    pct = 200.0 * (ranks / denom)
    return np.clip(pct, 0.0, 200.0)


def _absolute_values(values: np.ndarray, cap_percent: float) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    v = np.where(np.isfinite(v), v, float(cap_percent))
    return np.clip(v, 0.0, float(cap_percent))


def _colors_for_values(
    values: np.ndarray,
    *,
    palette_name: str,
    anchor_percent: float,
    cap_percent: float,
) -> np.ndarray:
    low, anchor, high = PALETTES[palette_name]
    v = np.clip(np.asarray(values, dtype=np.float64), 0.0, float(cap_percent))
    colors = np.empty((v.shape[0], 3), dtype=np.float64)
    low_mask = v <= float(anchor_percent)
    if np.any(low_mask):
        t = np.clip(v[low_mask] / float(anchor_percent), 0.0, 1.0)[:, None]
        colors[low_mask] = (1.0 - t) * low[None, :] + t * anchor[None, :]
    if np.any(~low_mask):
        if float(cap_percent) <= float(anchor_percent):
            colors[~low_mask] = high[None, :]
        else:
            t = ((v[~low_mask] - float(anchor_percent)) / (float(cap_percent) - float(anchor_percent)))[:, None]
            t = np.clip(t, 0.0, 1.0)
            colors[~low_mask] = (1.0 - t) * anchor[None, :] + t * high[None, :]
    return colors


def _widths_for_values(
    values: np.ndarray,
    *,
    anchor_percent: float,
    cap_percent: float,
    min_width: float,
    anchor_width: float,
    max_width: float,
    curve: str,
) -> np.ndarray:
    v = np.clip(np.asarray(values, dtype=np.float64), 0.0, float(cap_percent))
    widths = np.empty(v.shape[0], dtype=np.float64)
    low_mask = v <= float(anchor_percent)
    if np.any(low_mask):
        t = _curve_unit(v[low_mask] / float(anchor_percent), curve)
        widths[low_mask] = float(min_width) + (float(anchor_width) - float(min_width)) * t
    if np.any(~low_mask):
        if float(cap_percent) <= float(anchor_percent):
            widths[~low_mask] = float(max_width)
        else:
            t = _curve_unit(
                (v[~low_mask] - float(anchor_percent)) / (float(cap_percent) - float(anchor_percent)),
                curve,
            )
            widths[~low_mask] = float(anchor_width) + (float(max_width) - float(anchor_width)) * t
    return widths


def _sample_global_reference(payload_paths: list[Path], sample_per_file: int) -> np.ndarray:
    collected: list[np.ndarray] = []
    for idx, pt_path in enumerate(payload_paths, start=1):
        rows, _ = _load_payload(pt_path)
        one_file: list[np.ndarray] = []
        total_points = 0
        for row in rows:
            pct = _to_numpy(row.get("error_percentage")).reshape(-1)
            finite = pct[np.isfinite(pct)]
            if finite.size == 0:
                continue
            total_points += int(finite.size)
            one_file.append(finite)
        if not one_file:
            continue
        merged = np.concatenate(one_file, axis=0)
        stride = max(1, int(math.ceil(float(merged.size) / float(max(int(sample_per_file), 1)))))
        collected.append(np.asarray(merged[::stride], dtype=np.float64))
        print(f"[variant_sweep] global-sample {idx}/{len(payload_paths)} {pt_path.parent.name} sampled={collected[-1].size}")
    if not collected:
        return np.empty((0,), dtype=np.float64)
    return np.sort(np.concatenate(collected, axis=0))


def _render_one(
    *,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    preset: dict[str, Any],
    output_path: Path,
    figsize: float,
    dpi: int,
    min_width: float,
    anchor_width: float,
    max_width: float,
    alpha: float,
    draw_bins: int,
    global_sample_sorted: np.ndarray | None,
) -> dict[str, Any]:
    bbox = _resolve_bbox(metadata)
    fig, ax = plt.subplots(figsize=(float(figsize), float(figsize)), dpi=int(dpi))
    visible_trajectories = 0
    visible_points = 0
    visible_segments = 0
    segments_all: list[np.ndarray] = []
    values_all: list[np.ndarray] = []

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
        pct = pct[:n]
        point_mask = _inside_bbox(xy, bbox)
        seg_mask = point_mask[:-1] & point_mask[1:]
        if not np.any(seg_mask):
            continue

        seg_xy = np.stack([xy[:-1][seg_mask], xy[1:][seg_mask]], axis=1).astype(np.float32, copy=False)
        seg_pct = np.maximum(pct[:-1][seg_mask], pct[1:][seg_mask])
        if preset["mode"] == "global_rank":
            if global_sample_sorted is None:
                raise RuntimeError("global_sample_sorted is required for global_rank preset.")
            seg_values = _global_rank_values(global_sample_sorted, seg_pct)
        else:
            seg_values = _absolute_values(seg_pct, preset["cap_percent"])

        segments_all.append(seg_xy)
        values_all.append(np.asarray(seg_values, dtype=np.float64))
        visible_trajectories += 1
        visible_points += int(np.sum(point_mask))
        visible_segments += int(seg_xy.shape[0])

    if segments_all:
        seg_xy_all = np.concatenate(segments_all, axis=0)
        seg_values_all = np.concatenate(values_all, axis=0)
        colors = _colors_for_values(
            seg_values_all,
            palette_name=str(preset["palette"]),
            anchor_percent=float(preset["anchor_percent"]),
            cap_percent=float(preset["cap_percent"]),
        ).astype(np.float32, copy=False)
        widths = _widths_for_values(
            seg_values_all,
            anchor_percent=float(preset["anchor_percent"]),
            cap_percent=float(preset["cap_percent"]),
            min_width=float(min_width),
            anchor_width=float(anchor_width),
            max_width=float(max_width),
            curve=str(preset["curve"]),
        ).astype(np.float32, copy=False)
        n_bins = max(1, int(draw_bins))
        cap = float(preset["cap_percent"])
        bin_ids = np.floor((np.clip(seg_values_all, 0.0, cap) / cap) * n_bins).astype(np.int32)
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)
        for bin_idx in range(n_bins):
            sel = bin_ids == bin_idx
            if not np.any(sel):
                continue
            collection = LineCollection(
                seg_xy_all[sel],
                colors=colors[sel],
                linewidths=widths[sel],
                alpha=float(alpha),
                capstyle="round",
                joinstyle="round",
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
    }


def main() -> None:
    args = _build_parser().parse_args()
    input_root = Path(str(args.input))
    output_root = Path(str(args.output_dir))
    payload_paths = _iter_payloads(input_root)

    output_root.mkdir(parents=True, exist_ok=True)
    global_sample_sorted = _sample_global_reference(payload_paths, int(args.sample_per_file))
    manifest: list[dict[str, Any]] = []

    for preset in PRESETS:
        preset_dir = output_root / str(preset["name"])
        preset_dir.mkdir(parents=True, exist_ok=True)
        print(f"[variant_sweep] preset={preset['name']} start")
        for idx, pt_path in enumerate(payload_paths, start=1):
            rows, metadata = _load_payload(pt_path)
            label = _model_label(metadata, pt_path)
            output_path = preset_dir / f"{_safe_filename(label)}.png"
            result = _render_one(
                rows=rows,
                metadata=metadata,
                preset=preset,
                output_path=output_path,
                figsize=float(args.figsize),
                dpi=int(args.dpi),
                min_width=float(args.min_width),
                anchor_width=float(args.anchor_width),
                max_width=float(args.max_width),
                alpha=float(args.alpha),
                draw_bins=int(args.draw_bins),
                global_sample_sorted=global_sample_sorted,
            )
            print(
                f"[variant_sweep] preset={preset['name']} file={idx}/{len(payload_paths)} "
                f"model={label} visible_segments={result['visible_segments']}"
            )
            manifest.append(
                {
                    "preset": str(preset["name"]),
                    "input_path": str(pt_path),
                    "output_path": str(output_path),
                    "model_label": str(label),
                    **result,
                }
            )

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "manifest": str(manifest_path), "files": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
