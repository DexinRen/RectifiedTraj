#!/usr/bin/env python3
"""Extract bbox-matching trajectories, optionally denoise them, and plot a map-style figure."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("RECTIFIEDTRAJ_DEVICE", "cpu")

from encoder_decoder import EncoderDecoder, set_runtime_device


LOGGER = logging.getLogger("region_traj_mapper")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Filter trajectory PT files by a lon/lat bounding box, save matching subsets, "
            "optionally denoise them, and render a map-style PNG."
        )
    )
    p.add_argument(
        "--input",
        required=True,
        help="Trajectory .pt file or directory containing trajectory .pt files.",
    )
    p.add_argument("--min-lon", type=float, required=True)
    p.add_argument("--max-lon", type=float, required=True)
    p.add_argument("--min-lat", type=float, required=True)
    p.add_argument("--max-lat", type=float, required=True)
    p.add_argument(
        "--intersect-field",
        choices=["label", "data", "denoised"],
        default="label",
        help="Trajectory field used to decide whether a trajectory passes through the region.",
    )
    p.add_argument(
        "--pad-lon",
        type=float,
        default=0.01,
        help="Extra longitude padding used for the plotted local window and clipped PT output.",
    )
    p.add_argument(
        "--pad-lat",
        type=float,
        default=0.01,
        help="Extra latitude padding used for the plotted local window and clipped PT output.",
    )
    p.add_argument(
        "--max-trajectories",
        type=int,
        default=0,
        help="Optional cap on kept trajectories after sorting by points inside the region. 0 means no cap.",
    )
    p.add_argument(
        "--max-plot-trajectories",
        type=int,
        default=40,
        help="Maximum number of trajectories drawn in the PNG.",
    )
    p.add_argument(
        "--output-dir",
        default="./bin/region_maps",
        help="Output directory for PT files, PNG, and JSON summary.",
    )
    p.add_argument(
        "--checkpoint",
        default="",
        help="Optional checkpoint path for denoising matching trajectories before plotting.",
    )
    p.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Runtime device for denoising.",
    )
    p.add_argument("--q1", type=int, default=None, help="Optional buckle head bytes override.")
    p.add_argument("--q2", type=int, default=None, help="Optional buckle tail bytes override.")
    p.add_argument(
        "--title",
        default="",
        help="Optional custom figure title. Default is derived from the input and bbox.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable info-level logging.",
    )
    return p


def _resolve_input_pt(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")

    preferred_patterns = ("traj_*.pt", "fulltraj_*.pt", "*.pt")
    seen: set[Path] = set()
    candidates: list[Path] = []
    for pattern in preferred_patterns:
        for pt_path in sorted(path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True):
            if not pt_path.is_file():
                continue
            if pt_path in seen:
                continue
            seen.add(pt_path)
            candidates.append(pt_path)
    if not candidates:
        raise FileNotFoundError(f"No trajectory .pt files found under {path}")
    return candidates[0]


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize_row(row: dict[str, Any], *, allow_denoised_field: bool) -> dict[str, Any] | None:
    data = _to_numpy(row.get("data"))
    label = _to_numpy(row.get("label"))
    if data is None or label is None:
        return None
    if data.ndim != 2 or label.ndim != 2 or data.shape[1] < 2 or label.shape[1] < 2:
        return None

    n = min(int(data.shape[0]), int(label.shape[0]))
    if n <= 0:
        return None

    out: dict[str, Any] = {
        "agent_id": row.get("agent_id"),
        "data": np.asarray(data[:n, :2], dtype=np.float64),
        "label": np.asarray(label[:n, :2], dtype=np.float64),
    }

    ts = _to_numpy(row.get("timestamp"))
    if ts is not None:
        ts = np.asarray(ts).reshape(-1)
        n = min(n, int(ts.shape[0]))
    err = _to_numpy(row.get("error_range"))
    if err is None:
        err = _to_numpy(row.get("accuracy"))
    if err is not None:
        err = np.asarray(err).reshape(-1)
        n = min(n, int(err.shape[0]))
    lat_sigma = _to_numpy(row.get("latitude_sigma"))
    if lat_sigma is not None:
        lat_sigma = np.asarray(lat_sigma).reshape(-1)
        n = min(n, int(lat_sigma.shape[0]))
    lon_sigma = _to_numpy(row.get("longitude_sigma"))
    if lon_sigma is not None:
        lon_sigma = np.asarray(lon_sigma).reshape(-1)
        n = min(n, int(lon_sigma.shape[0]))
    denoised = _to_numpy(row.get("denoised")) if allow_denoised_field else None
    if denoised is not None:
        if denoised.ndim != 2 or denoised.shape[1] < 2:
            denoised = None
        else:
            n = min(n, int(denoised.shape[0]))

    if n <= 0:
        return None

    out["data"] = out["data"][:n]
    out["label"] = out["label"][:n]
    out["n_points"] = int(n)
    if ts is not None:
        out["timestamp"] = np.asarray(ts[:n], dtype=np.float64)
    if err is not None:
        err_arr = np.asarray(err[:n], dtype=np.float64)
        out["error_range"] = err_arr
        out["accuracy"] = err_arr
    if lat_sigma is not None:
        out["latitude_sigma"] = np.asarray(lat_sigma[:n], dtype=np.float64)
    if lon_sigma is not None:
        out["longitude_sigma"] = np.asarray(lon_sigma[:n], dtype=np.float64)
    if denoised is not None:
        out["denoised"] = np.asarray(denoised[:n, :2], dtype=np.float64)

    valid = np.isfinite(out["data"]).all(axis=1) & np.isfinite(out["label"]).all(axis=1)
    if "timestamp" in out:
        valid &= np.isfinite(out["timestamp"])
    if "error_range" in out:
        valid &= np.isfinite(out["error_range"])
    if "latitude_sigma" in out:
        valid &= np.isfinite(out["latitude_sigma"])
    if "longitude_sigma" in out:
        valid &= np.isfinite(out["longitude_sigma"])
    if "denoised" in out:
        valid &= np.isfinite(out["denoised"]).all(axis=1)

    if not np.any(valid):
        return None

    for key in ("data", "label", "denoised"):
        if key in out:
            out[key] = out[key][valid]
    for key in ("timestamp", "error_range", "accuracy", "latitude_sigma", "longitude_sigma"):
        if key in out:
            out[key] = out[key][valid]
    out["n_points"] = int(out["data"].shape[0])
    if out["n_points"] <= 0:
        return None
    return out


def _inside_bbox(xy: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    min_lon, max_lon, min_lat, max_lat = bbox
    lon = np.asarray(xy[:, 0], dtype=np.float64)
    lat = np.asarray(xy[:, 1], dtype=np.float64)
    return (
        (lon >= min_lon)
        & (lon <= max_lon)
        & (lat >= min_lat)
        & (lat <= max_lat)
    )


def _expand_bbox(
    bbox: tuple[float, float, float, float],
    *,
    pad_lon: float,
    pad_lat: float,
) -> tuple[float, float, float, float]:
    min_lon, max_lon, min_lat, max_lat = bbox
    return (
        min_lon - float(pad_lon),
        max_lon + float(pad_lon),
        min_lat - float(pad_lat),
        max_lat + float(pad_lat),
    )


def _load_rows(input_pt: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    blob = torch.load(input_pt, map_location="cpu")
    if not isinstance(blob, dict):
        raise ValueError(f"Unsupported PT payload type in {input_pt}: {type(blob)}")
    rows = blob.get("trajectories")
    if not isinstance(rows, list):
        raise ValueError(f"No 'trajectories' list in {input_pt}")
    metadata = blob.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        norm = _normalize_row(row, allow_denoised_field=True)
        if norm is not None:
            normalized.append(norm)
    return normalized, metadata


def _build_manual_config(args: argparse.Namespace) -> dict[str, Any] | None:
    manual: dict[str, Any] = {}
    if args.q1 is not None:
        manual["Q1"] = int(args.q1)
    if args.q2 is not None:
        manual["Q2"] = int(args.q2)
    return manual or None


def _denoise_rows(
    rows: list[dict[str, Any]],
    *,
    checkpoint: str,
    device: str,
    manual_config: dict[str, Any] | None,
) -> dict[str, Any]:
    set_runtime_device(device)
    decoder = EncoderDecoder(checkpoint, manual_config=manual_config)
    denoised_count = 0
    for row in rows:
        noisy = np.asarray(row["data"], dtype=np.float64)
        if noisy.shape[0] <= 0:
            continue
        denoised = decoder.denoise_traj_DF(noisy)
        t_len = min(int(noisy.shape[0]), int(denoised.shape[0]), int(row["label"].shape[0]))
        if t_len <= 0:
            continue
        row["data"] = row["data"][:t_len]
        row["label"] = row["label"][:t_len]
        row["denoised"] = np.asarray(denoised[:t_len, :2], dtype=np.float64)
        for key in ("timestamp", "error_range", "accuracy", "latitude_sigma", "longitude_sigma"):
            if key in row:
                row[key] = np.asarray(row[key])[:t_len]
        row["n_points"] = int(t_len)
        denoised_count += 1
    return {
        "checkpoint": str(checkpoint),
        "device": str(device),
        "count": int(denoised_count),
        "manual_config": manual_config or {},
    }


def _tensorize_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "agent_id": row.get("agent_id"),
        "n_points": int(row["n_points"]),
        "data": torch.as_tensor(row["data"], dtype=torch.float32),
        "label": torch.as_tensor(row["label"], dtype=torch.float32),
    }
    for key in ("timestamp", "error_range", "accuracy", "latitude_sigma", "longitude_sigma"):
        if key in row:
            out[key] = torch.as_tensor(row[key], dtype=torch.float32)
    if "denoised" in row:
        out["denoised"] = torch.as_tensor(row["denoised"], dtype=torch.float32)
    return out


def _save_rows(
    rows: list[dict[str, Any]],
    *,
    output_path: Path,
    metadata: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trajectories": [_tensorize_row(row) for row in rows],
        "metadata": metadata,
    }
    torch.save(payload, output_path)


def _format_bbox_slug(bbox: tuple[float, float, float, float]) -> str:
    min_lon, max_lon, min_lat, max_lat = bbox
    return (
        f"lon_{min_lon:.5f}_{max_lon:.5f}_lat_{min_lat:.5f}_{max_lat:.5f}"
        .replace("-", "m")
        .replace(".", "p")
    )


def _clip_rows_to_window(
    rows: list[dict[str, Any]],
    *,
    bbox: tuple[float, float, float, float],
    field_name: str,
) -> list[dict[str, Any]]:
    clipped: list[dict[str, Any]] = []
    for row in rows:
        field = row.get(field_name)
        if field is None:
            continue
        mask = _inside_bbox(np.asarray(field, dtype=np.float64), bbox)
        if not np.any(mask):
            continue
        clipped_row: dict[str, Any] = {
            "agent_id": row.get("agent_id"),
            "n_points": int(np.sum(mask)),
            "data": row["data"][mask],
            "label": row["label"][mask],
        }
        for key in ("timestamp", "error_range", "accuracy", "latitude_sigma", "longitude_sigma", "denoised"):
            if key in row:
                clipped_row[key] = np.asarray(row[key])[mask]
        clipped.append(clipped_row)
    return clipped


def _aspect_for_lat(mid_lat: float) -> float:
    cos_lat = math.cos(math.radians(float(mid_lat)))
    return 1.0 / max(abs(cos_lat), 1e-6)


def _plot_rows(
    rows: list[dict[str, Any]],
    *,
    region_bbox: tuple[float, float, float, float],
    view_bbox: tuple[float, float, float, float],
    output_path: Path,
    title: str,
    max_plot_trajectories: int,
) -> None:
    plot_rows = rows[: max(1, int(max_plot_trajectories))] if rows else []
    min_lon, max_lon, min_lat, max_lat = region_bbox
    view_min_lon, view_max_lon, view_min_lat, view_max_lat = view_bbox
    mid_lat = 0.5 * (view_min_lat + view_max_lat)

    fig = plt.figure(figsize=(16, 9), facecolor="#f6f1e8")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2], wspace=0.08)
    ax_overview = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])

    for ax in (ax_overview, ax_zoom):
        ax.set_facecolor("#fbf7ef")
        ax.grid(color="#d8cfbf", linestyle="--", linewidth=0.6, alpha=0.6)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect(_aspect_for_lat(mid_lat))

    region_patch_left = mpatches.Rectangle(
        (min_lon, min_lat),
        max_lon - min_lon,
        max_lat - min_lat,
        facecolor="#e2a53b",
        edgecolor="#8b5e00",
        linewidth=1.8,
        alpha=0.18,
    )
    region_patch_right = mpatches.Rectangle(
        (min_lon, min_lat),
        max_lon - min_lon,
        max_lat - min_lat,
        facecolor="#e2a53b",
        edgecolor="#8b5e00",
        linewidth=1.8,
        alpha=0.18,
    )
    ax_overview.add_patch(region_patch_left)
    ax_zoom.add_patch(region_patch_right)

    for row in plot_rows:
        clean = np.asarray(row["label"], dtype=np.float64)
        noisy = np.asarray(row["data"], dtype=np.float64)
        ax_overview.plot(clean[:, 0], clean[:, 1], color="#57616d", linewidth=0.9, alpha=0.38)

        view_mask = _inside_bbox(clean, view_bbox) | _inside_bbox(noisy, view_bbox)
        if "denoised" in row:
            view_mask |= _inside_bbox(np.asarray(row["denoised"], dtype=np.float64), view_bbox)
        if not np.any(view_mask):
            continue

        clean_view = clean[view_mask]
        noisy_view = noisy[view_mask]
        ax_zoom.plot(clean_view[:, 0], clean_view[:, 1], color="#4a4f57", linewidth=1.4, alpha=0.72)
        ax_zoom.plot(noisy_view[:, 0], noisy_view[:, 1], color="#cb6d36", linewidth=1.0, alpha=0.34)
        if "denoised" in row:
            denoised_view = np.asarray(row["denoised"], dtype=np.float64)[view_mask]
            ax_zoom.plot(denoised_view[:, 0], denoised_view[:, 1], color="#0f7c82", linewidth=1.5, alpha=0.82)

    ax_overview.set_title("Matched Trajectories Overview", fontsize=13, color="#2a3038")
    ax_zoom.set_title("Local Window Detail", fontsize=13, color="#2a3038")
    ax_overview.set_xlim(view_min_lon, view_max_lon)
    ax_overview.set_ylim(view_min_lat, view_max_lat)
    ax_zoom.set_xlim(view_min_lon, view_max_lon)
    ax_zoom.set_ylim(view_min_lat, view_max_lat)

    handles = [
        plt.Line2D([0], [0], color="#4a4f57", lw=2, label="clean/reference"),
        plt.Line2D([0], [0], color="#cb6d36", lw=2, alpha=0.6, label="noisy"),
    ]
    if any("denoised" in row for row in plot_rows):
        handles.append(plt.Line2D([0], [0], color="#0f7c82", lw=2, label="denoised"))
    handles.append(
        mpatches.Patch(facecolor="#e2a53b", edgecolor="#8b5e00", alpha=0.2, label="target region")
    )
    ax_zoom.legend(handles=handles, loc="upper right", frameon=True, facecolor="#fbf7ef", edgecolor="#c8bca6")

    fig.suptitle(title, fontsize=18, color="#1f2730", y=0.98)
    fig.text(
        0.5,
        0.02,
        "Saved subsets keep the native trajectory PT schema. If denoising is enabled, a 'denoised' field is added per trajectory.",
        ha="center",
        fontsize=10,
        color="#4f5965",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    bbox = (float(args.min_lon), float(args.max_lon), float(args.min_lat), float(args.max_lat))
    if bbox[0] >= bbox[1]:
        raise ValueError("min-lon must be smaller than max-lon")
    if bbox[2] >= bbox[3]:
        raise ValueError("min-lat must be smaller than max-lat")
    view_bbox = _expand_bbox(bbox, pad_lon=float(args.pad_lon), pad_lat=float(args.pad_lat))

    input_pt = _resolve_input_pt(args.input)
    rows, source_metadata = _load_rows(input_pt)
    if not rows:
        raise RuntimeError(f"No trajectory rows found in {input_pt}")

    denoise_meta = None
    manual_config = _build_manual_config(args)
    intersect_field_name = "label" if args.intersect_field == "denoised" else str(args.intersect_field)
    if args.checkpoint and args.intersect_field == "denoised":
        LOGGER.info(
            "Running denoising on %d trajectories from %s before region filtering",
            len(rows),
            input_pt.name,
        )
        denoise_meta = _denoise_rows(
            rows,
            checkpoint=args.checkpoint,
            device=args.device,
            manual_config=manual_config,
        )
        intersect_field_name = "denoised"

    scored_rows: list[tuple[int, dict[str, Any]]] = []
    for row in rows:
        field = row.get(intersect_field_name)
        if field is None:
            continue
        mask = _inside_bbox(np.asarray(field, dtype=np.float64), bbox)
        hits = int(np.sum(mask))
        if hits <= 0:
            continue
        row["region_point_count"] = hits
        scored_rows.append((hits, row))

    if not scored_rows:
        raise RuntimeError(
            f"No trajectories intersect bbox {bbox} in {input_pt} using field={intersect_field_name}"
        )

    scored_rows.sort(key=lambda item: item[0], reverse=True)
    if int(args.max_trajectories) > 0:
        scored_rows = scored_rows[: int(args.max_trajectories)]
    matched_rows = [row for _, row in scored_rows]

    if args.checkpoint and args.intersect_field != "denoised":
        LOGGER.info(
            "Running denoising on %d matched trajectories from %s",
            len(matched_rows),
            input_pt.name,
        )
        denoise_meta = _denoise_rows(
            matched_rows,
            checkpoint=args.checkpoint,
            device=args.device,
            manual_config=manual_config,
        )

    clipped_rows = _clip_rows_to_window(
        matched_rows,
        bbox=view_bbox,
        field_name=intersect_field_name,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = _format_bbox_slug(bbox)
    full_out = output_dir / f"region_full_{slug}.pt"
    clipped_out = output_dir / f"region_window_{slug}.pt"
    plot_out = output_dir / f"region_map_{slug}.png"
    summary_out = output_dir / f"region_summary_{slug}.json"

    common_meta = {
        "source_file": str(input_pt),
        "source_metadata": source_metadata,
        "region_bbox": {
            "min_lon": bbox[0],
            "max_lon": bbox[1],
            "min_lat": bbox[2],
            "max_lat": bbox[3],
        },
        "view_bbox": {
            "min_lon": view_bbox[0],
            "max_lon": view_bbox[1],
            "min_lat": view_bbox[2],
            "max_lat": view_bbox[3],
        },
        "intersect_field": intersect_field_name,
        "n_source_trajectories": int(len(rows)),
        "n_matched_trajectories": int(len(matched_rows)),
        "n_window_trajectories": int(len(clipped_rows)),
        "region_point_count_total": int(sum(int(row["region_point_count"]) for row in matched_rows)),
        "contains_denoised": bool(any("denoised" in row for row in matched_rows)),
    }
    if denoise_meta is not None:
        common_meta["denoise"] = denoise_meta

    _save_rows(
        matched_rows,
        output_path=full_out,
        metadata={**common_meta, "subset_kind": "full_pass_through"},
    )
    _save_rows(
        clipped_rows,
        output_path=clipped_out,
        metadata={**common_meta, "subset_kind": "window_clipped"},
    )

    title = args.title.strip() if args.title else (
        f"{input_pt.stem}: {len(matched_rows)} pass-through trajectories in lon/lat bbox"
    )
    _plot_rows(
        matched_rows,
        region_bbox=bbox,
        view_bbox=view_bbox,
        output_path=plot_out,
        title=title,
        max_plot_trajectories=int(args.max_plot_trajectories),
    )

    summary = {
        **common_meta,
        "full_output_file": str(full_out),
        "window_output_file": str(clipped_out),
        "plot_file": str(plot_out),
    }
    summary_out.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
