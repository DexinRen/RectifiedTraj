#!/usr/bin/env python3
"""
Extract fixed-length RobotCar trajectories without time overlap.

Behavior:
- Treat each gps.csv file as an independent run.
- Estimate native sample time from positive timestamp diffs inside that file.
- Split each file into continuous spans when timestamps stop increasing or a gap
  exceeds gap_multiplier * native_dt.
- For a target sample time, downsample each span with one fixed stride.
- Slice non-overlapping raw blocks and keep exactly one trajectory per block.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class GpsRow:
    timestamp_us: int
    longitude: float
    latitude: float
    latitude_sigma: float
    longitude_sigma: float
    source_file: str
    source_row: int
    run_id: str


def _sample_label(seconds_value: float) -> str:
    val = float(seconds_value)
    if abs(val - round(val)) < 1e-9:
        return f"{int(round(val))}s"
    return f"{val:g}s".replace(".", "p")


def _find_col_idx(header: list[str], col_name: str) -> int:
    target = str(col_name).strip().lower()
    for idx, name in enumerate(header):
        if str(name).strip().lower() == target:
            return idx
    raise ValueError(f"Missing required column: {col_name}")


def _load_one_gps_csv(path: Path) -> list[GpsRow]:
    rows: list[GpsRow] = []
    run_id = path.parent.parent.name
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return rows

        ts_idx = _find_col_idx(header, "timestamp")
        lat_idx = _find_col_idx(header, "latitude")
        lon_idx = _find_col_idx(header, "longitude")
        lat_sigma_idx = _find_col_idx(header, "latitude_sigma")
        lon_sigma_idx = _find_col_idx(header, "longitude_sigma")

        for ridx, row in enumerate(reader, start=2):
            if not row:
                continue
            try:
                ts_us = int(float(row[ts_idx]))
                lat = float(row[lat_idx])
                lon = float(row[lon_idx])
                lat_sigma = float(row[lat_sigma_idx])
                lon_sigma = float(row[lon_sigma_idx])
            except Exception:
                continue
            if not np.isfinite(lat) or not np.isfinite(lon):
                continue
            rows.append(
                GpsRow(
                    timestamp_us=ts_us,
                    longitude=lon,
                    latitude=lat,
                    latitude_sigma=lat_sigma,
                    longitude_sigma=lon_sigma,
                    source_file=str(path.resolve()),
                    source_row=int(ridx),
                    run_id=str(run_id),
                )
            )
    return rows


def _discover_gps_files(input_root: Path, input_glob: str) -> list[Path]:
    files = sorted(p for p in input_root.glob(input_glob) if p.is_file())
    if not files:
        raise FileNotFoundError(
            f"No gps.csv files found under {input_root} using glob '{input_glob}'."
        )
    return files


def _compute_native_dt_sec(rows: list[GpsRow]) -> float:
    diffs_us: list[int] = []
    ordered = sorted(rows, key=lambda r: (r.timestamp_us, r.source_row))
    for i in range(1, len(ordered)):
        d = ordered[i].timestamp_us - ordered[i - 1].timestamp_us
        if d > 0:
            diffs_us.append(int(d))
    if not diffs_us:
        raise ValueError("Cannot estimate native dt (no positive timestamp diffs).")
    return float(np.mean(np.asarray(diffs_us, dtype=np.float64)) / 1_000_000.0)


def _split_continuous_spans(
    rows_sorted: list[GpsRow],
    *,
    native_dt_sec: float,
    gap_multiplier: float,
) -> list[list[GpsRow]]:
    if not rows_sorted:
        return []
    if native_dt_sec <= 0:
        raise ValueError(f"native_dt_sec must be > 0, got {native_dt_sec}")

    max_gap_us = float(native_dt_sec * float(gap_multiplier) * 1_000_000.0)
    spans: list[list[GpsRow]] = []
    start = 0
    for i in range(1, len(rows_sorted)):
        d = rows_sorted[i].timestamp_us - rows_sorted[i - 1].timestamp_us
        if d <= 0 or d > max_gap_us:
            spans.append(rows_sorted[start:i])
            start = i
    spans.append(rows_sorted[start:])
    return [s for s in spans if s]


def _safe_acc_m(lat_sigma: np.ndarray, lon_sigma: np.ndarray) -> np.ndarray:
    acc = np.sqrt(np.square(lat_sigma) + np.square(lon_sigma))
    valid = np.isfinite(acc) & (acc >= 0.0)
    if int(np.sum(valid)) == 0:
        return np.zeros_like(acc, dtype=np.float32)
    fill = float(np.median(acc[valid]))
    acc = np.where(valid, acc, fill)
    acc = np.maximum(acc, 0.0)
    return acc.astype(np.float32, copy=False)


def extract_robotcar_nooverlap(
    *,
    input_root: str,
    input_glob: str,
    output_pt: str,
    sample_sec: float = 2.0,
    window_len: int = 500,
    gap_multiplier: float = 5.0,
    tolerance_ratio: float = 0.10,
) -> dict[str, Any]:
    root = Path(input_root).resolve()
    out_pt = Path(output_pt).resolve()
    files = _discover_gps_files(root, input_glob)

    trajectories: list[dict[str, Any]] = []
    trajectory_summaries: list[dict[str, Any]] = []
    file_summaries: list[dict[str, Any]] = []

    for path in files:
        rows = _load_one_gps_csv(path)
        if not rows:
            file_summaries.append(
                {
                    "file": str(path.resolve()),
                    "run_id": path.parent.parent.name,
                    "status": "empty",
                    "n_rows": 0,
                    "n_spans": 0,
                    "n_blocks": 0,
                    "n_trajectories": 0,
                }
            )
            continue

        ordered = sorted(rows, key=lambda r: (r.timestamp_us, r.source_row))
        try:
            native_dt_sec = _compute_native_dt_sec(ordered)
        except ValueError:
            file_summaries.append(
                {
                    "file": str(path.resolve()),
                    "run_id": path.parent.parent.name,
                    "status": "no_positive_diffs",
                    "n_rows": int(len(ordered)),
                    "n_spans": 0,
                    "n_blocks": 0,
                    "n_trajectories": 0,
                }
            )
            continue

        ratio = float(sample_sec / native_dt_sec)
        stride = max(1, int(round(ratio)))
        mismatch = abs(ratio - float(stride)) / max(abs(ratio), 1e-12)
        if mismatch > float(tolerance_ratio):
            file_summaries.append(
                {
                    "file": str(path.resolve()),
                    "run_id": path.parent.parent.name,
                    "status": "ratio_mismatch",
                    "n_rows": int(len(ordered)),
                    "native_dt_sec": float(native_dt_sec),
                    "ratio": float(ratio),
                    "stride": int(stride),
                    "mismatch": float(mismatch),
                    "n_spans": 0,
                    "n_blocks": 0,
                    "n_trajectories": 0,
                }
            )
            continue

        spans = _split_continuous_spans(
            ordered,
            native_dt_sec=float(native_dt_sec),
            gap_multiplier=float(gap_multiplier),
        )
        required_native_len = int((window_len - 1) * stride + 1)
        if required_native_len <= 0:
            raise ValueError(
                f"Invalid required_native_len={required_native_len} "
                f"(window_len={window_len}, stride={stride})"
            )

        n_blocks_total = 0
        for span_id, span_rows in enumerate(spans):
            n_span = len(span_rows)
            if n_span < required_native_len:
                continue
            n_blocks = int((n_span - required_native_len) // required_native_len + 1)
            for block_id in range(n_blocks):
                block_start = int(block_id * required_native_len)
                block_end = int(block_start + required_native_len)
                if block_end > n_span:
                    continue
                picked_rows = span_rows[block_start:block_end:stride]
                if len(picked_rows) != window_len:
                    continue

                lon = np.asarray([r.longitude for r in picked_rows], dtype=np.float64)
                lat = np.asarray([r.latitude for r in picked_rows], dtype=np.float64)
                ts = np.asarray([r.timestamp_us for r in picked_rows], dtype=np.float64) / 1_000_000.0
                lat_sigma = np.asarray([r.latitude_sigma for r in picked_rows], dtype=np.float64)
                lon_sigma = np.asarray([r.longitude_sigma for r in picked_rows], dtype=np.float64)
                acc = _safe_acc_m(lat_sigma, lon_sigma)
                source_indices = np.asarray(
                    [r.source_row for r in picked_rows],
                    dtype=np.int32,
                )

                lonlat = np.column_stack([lon, lat]).astype(np.float32, copy=False)
                agent_id = (
                    f"robotcar_{_sample_label(sample_sec)}_{picked_rows[0].run_id}"
                    f"_span{span_id:04d}_block{block_id:08d}"
                )
                traj = {
                    "agent_id": str(agent_id),
                    "sample_time_sec": float(sample_sec),
                    "stride": int(stride),
                    "run_id": str(picked_rows[0].run_id),
                    "source_file": str(path.resolve()),
                    "span_id": int(span_id),
                    "block_id": int(block_id),
                    "window_start_idx": int(block_start),
                    "required_native_len": int(required_native_len),
                    "n_points": int(window_len),
                    "source_row_indices": torch.from_numpy(source_indices).to(torch.int32),
                    "data": torch.from_numpy(lonlat).to(torch.float32),
                    "label": torch.from_numpy(lonlat.copy()).to(torch.float32),
                    "timestamp": torch.from_numpy(ts.astype(np.float64, copy=False)).to(torch.float64),
                    "error_range": torch.from_numpy(acc).to(torch.float32),
                    "accuracy": torch.from_numpy(acc.copy()).to(torch.float32),
                    "latitude_sigma": torch.from_numpy(lat_sigma.astype(np.float32, copy=False)).to(torch.float32),
                    "longitude_sigma": torch.from_numpy(lon_sigma.astype(np.float32, copy=False)).to(torch.float32),
                }
                trajectories.append(traj)
                trajectory_summaries.append(
                    {
                        "agent_id": str(agent_id),
                        "run_id": str(picked_rows[0].run_id),
                        "source_file": str(path.resolve()),
                        "span_id": int(span_id),
                        "block_id": int(block_id),
                        "window_start_idx": int(block_start),
                        "n_points": int(window_len),
                        "first_ts_us": int(picked_rows[0].timestamp_us),
                        "last_ts_us": int(picked_rows[-1].timestamp_us),
                    }
                )
                n_blocks_total += 1

        file_summaries.append(
            {
                "file": str(path.resolve()),
                "run_id": path.parent.parent.name,
                "status": "completed",
                "n_rows": int(len(ordered)),
                "native_dt_sec": float(native_dt_sec),
                "ratio": float(ratio),
                "stride": int(stride),
                "mismatch": float(mismatch),
                "n_spans": int(len(spans)),
                "required_native_len": int(required_native_len),
                "n_blocks": int(n_blocks_total),
                "n_trajectories": int(n_blocks_total),
            }
        )

    payload = {
        "status": "completed",
        "output_pt": str(out_pt),
        "input_root": str(root),
        "input_glob": str(input_glob),
        "sample_time_sec": float(sample_sec),
        "sample_label": _sample_label(sample_sec),
        "window_len": int(window_len),
        "gap_multiplier": float(gap_multiplier),
        "tolerance_ratio": float(tolerance_ratio),
        "dicing_mode": "per_file_non_overlapping_blocks_no_phase_shift",
        "time_overlap_allowed": False,
        "time_id_rule": "one trajectory per non-overlapping native block",
        "columns": ["longitude", "latitude"],
        "side_channels": ["latitude_sigma", "longitude_sigma"],
        "timestamp_unit": "unix_seconds",
        "error_range_unit": "meters",
        "error_range_formula": "sqrt(latitude_sigma^2 + longitude_sigma^2)",
        "n_input_files": int(len(file_summaries)),
        "n_trajectories": int(len(trajectories)),
        "n_points_total": int(len(trajectories) * int(window_len)),
        "metadata": {
            "n_trajectories": int(len(trajectories)),
            "median_length": int(window_len),
        },
        "file_summaries": file_summaries,
        "trajectory_summaries": trajectory_summaries,
        "trajectories": trajectories,
    }

    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_pt)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract RobotCar trajectories without time overlap."
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="./dataset/raw/RobotCar",
        help="Root directory to search for gps.csv files.",
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default="**/gps.csv",
        help="Glob relative to input-root for gps files.",
    )
    parser.add_argument(
        "--output-pt",
        type=str,
        default="./dataset/raw/RobotCar/extracted/RobotCar_2s_500_nooverlap.pt",
        help="Output .pt file path.",
    )
    parser.add_argument(
        "--sample-sec",
        type=float,
        default=2.0,
        help="Target sample time in seconds.",
    )
    parser.add_argument(
        "--window-len",
        type=int,
        default=500,
        help="Maximum trajectory length in points.",
    )
    parser.add_argument(
        "--gap-multiplier",
        type=float,
        default=5.0,
        help="Split spans when timestamp gap exceeds gap-multiplier * native_dt.",
    )
    parser.add_argument(
        "--tolerance-ratio",
        type=float,
        default=0.10,
        help="Allowed mismatch between desired/native ratio and rounded stride.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    payload = extract_robotcar_nooverlap(
        input_root=str(args.input_root),
        input_glob=str(args.input_glob),
        output_pt=str(args.output_pt),
        sample_sec=float(args.sample_sec),
        window_len=int(args.window_len),
        gap_multiplier=float(args.gap_multiplier),
        tolerance_ratio=float(args.tolerance_ratio),
    )
    summary = {
        "status": payload["status"],
        "output_pt": payload["output_pt"],
        "sample_time_sec": payload["sample_time_sec"],
        "window_len": payload["window_len"],
        "n_input_files": payload["n_input_files"],
        "n_trajectories": payload["n_trajectories"],
        "n_points_total": payload["n_points_total"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
