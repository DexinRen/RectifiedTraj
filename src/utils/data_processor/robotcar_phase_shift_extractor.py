#!/usr/bin/env python3
"""
Build phase-shifted RobotCar trajectory datasets with fixed window length.

Key behavior:
- Discover multiple gps.csv files.
- Group by UTC date and merge rows by timestamp within each date.
- Split continuity spans when timestamp gap > gap_multiplier * native_dt.
- For each target sample time, slice non-overlapping native blocks.
- For each block, generate phase-shifted trajectories by modulo indexing.
- Assign one shared time_id per block (across all phases).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
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
    date_id: str


def _parse_sample_seconds(raw: str) -> list[float]:
    vals: list[float] = []
    for tok in str(raw).split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    if not vals:
        raise ValueError("No sample seconds provided.")
    return vals


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


def _find_optional_col_idx(header: list[str], col_name: str) -> int | None:
    try:
        return _find_col_idx(header, col_name)
    except ValueError:
        return None


def _utc_date_from_us(ts_us: int) -> str:
    dt = datetime.fromtimestamp(float(ts_us) / 1_000_000.0, tz=timezone.utc)
    return dt.date().isoformat()


def _load_one_gps_csv(path: Path) -> list[GpsRow]:
    rows: list[GpsRow] = []
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
            date_id = _utc_date_from_us(ts_us)
            rows.append(
                GpsRow(
                    timestamp_us=ts_us,
                    longitude=lon,
                    latitude=lat,
                    latitude_sigma=lat_sigma,
                    longitude_sigma=lon_sigma,
                    source_file=str(path.resolve()),
                    source_row=int(ridx),
                    date_id=date_id,
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


def _group_rows_by_date(rows: list[GpsRow]) -> dict[str, list[GpsRow]]:
    out: dict[str, list[GpsRow]] = {}
    for row in rows:
        out.setdefault(row.date_id, []).append(row)
    return out


def _compute_native_dt_sec_for_date(rows: list[GpsRow]) -> float:
    by_file: dict[str, list[GpsRow]] = {}
    for row in rows:
        by_file.setdefault(row.source_file, []).append(row)

    diffs_us: list[int] = []
    for file_rows in by_file.values():
        ordered = sorted(file_rows, key=lambda r: (r.timestamp_us, r.source_row))
        for i in range(1, len(ordered)):
            d = ordered[i].timestamp_us - ordered[i - 1].timestamp_us
            if d > 0:
                diffs_us.append(int(d))

    if not diffs_us:
        ordered = sorted(rows, key=lambda r: (r.timestamp_us, r.source_file, r.source_row))
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


def _build_one_sample_dataset(
    rows_by_date: dict[str, list[GpsRow]],
    *,
    sample_sec: float,
    window_len: int,
    window_stride: int,
    gap_multiplier: float,
    tolerance_ratio: float,
    output_pt: Path,
) -> dict[str, Any]:
    trajectories: list[dict[str, Any]] = []
    trajectory_summaries: list[dict[str, Any]] = []
    date_summaries: list[dict[str, Any]] = []
    global_file_count: set[str] = set()

    for date_id in sorted(rows_by_date.keys()):
        date_rows = rows_by_date[date_id]
        native_dt_sec = _compute_native_dt_sec_for_date(date_rows)
        ratio = float(sample_sec / native_dt_sec)
        stride = max(1, int(round(ratio)))
        mismatch = abs(ratio - float(stride)) / max(abs(ratio), 1e-12)
        if mismatch > float(tolerance_ratio):
            raise ValueError(
                f"sample={sample_sec}s date={date_id} ratio mismatch too large: "
                f"ratio={ratio:.6f} stride={stride} mismatch={mismatch:.6f} "
                f"tolerance={tolerance_ratio:.6f}"
            )

        merged = sorted(date_rows, key=lambda r: (r.timestamp_us, r.source_file, r.source_row))
        date_row_indices = {id(row): i for i, row in enumerate(merged)}
        spans = _split_continuous_spans(
            merged,
            native_dt_sec=native_dt_sec,
            gap_multiplier=gap_multiplier,
        )

        # Per user dicing rule:
        # - one time_id corresponds to one native block
        # - each block yields `stride` phase-shifted trajectories via modulo slicing
        # - blocks do not overlap; next block starts after previous block length
        block_native_len = int(window_len * stride + stride)
        if block_native_len <= 0:
            raise ValueError(
                f"Invalid block_native_len={block_native_len} (window_len={window_len}, stride={stride})"
            )
        block_hop_native = int(max(1, window_stride) * block_native_len)

        date_summary = {
            "date_id": date_id,
            "native_dt_sec": float(native_dt_sec),
            "sample_sec": float(sample_sec),
            "stride": int(stride),
            "ratio": float(ratio),
            "mismatch": float(mismatch),
            "n_rows_merged": int(len(merged)),
            "n_spans": int(len(spans)),
            "block_native_len": int(block_native_len),
            "block_hop_native": int(block_hop_native),
            "n_time_ids": 0,
            "n_trajectories": 0,
        }

        for span_id, span_rows in enumerate(spans):
            for row in span_rows:
                global_file_count.add(row.source_file)
            n_span = len(span_rows)
            if n_span < block_native_len:
                continue
            n_blocks = int((n_span - block_native_len) // block_hop_native + 1)
            for block_id in range(n_blocks):
                block_start = int(block_id * block_hop_native)
                block_end = int(block_start + block_native_len)
                if block_end > n_span:
                    continue
                block_rows = span_rows[block_start:block_end]
                time_id = (
                    f"{date_id}|span{span_id:04d}|sample{_sample_label(sample_sec)}"
                    f"|block{block_id:08d}|native_start{block_start:08d}"
                )

                # Keep only complete phase trajectories with fixed point count.
                phase_rows: list[tuple[int, list[GpsRow]]] = []
                for phase in range(stride):
                    picked_rows = block_rows[phase::stride]
                    if len(picked_rows) < window_len:
                        phase_rows = []
                        break
                    phase_rows.append((phase, picked_rows[:window_len]))
                if not phase_rows:
                    continue

                for phase, picked_rows in phase_rows:
                    lon = np.asarray([r.longitude for r in picked_rows], dtype=np.float64)
                    lat = np.asarray([r.latitude for r in picked_rows], dtype=np.float64)
                    ts = np.asarray([r.timestamp_us for r in picked_rows], dtype=np.float64) / 1_000_000.0
                    lat_sigma = np.asarray([r.latitude_sigma for r in picked_rows], dtype=np.float64)
                    lon_sigma = np.asarray([r.longitude_sigma for r in picked_rows], dtype=np.float64)
                    acc = _safe_acc_m(lat_sigma, lon_sigma)
                    source_indices = np.asarray(
                        [date_row_indices[id(r)] for r in picked_rows],
                        dtype=np.int32,
                    )

                    lonlat = np.column_stack([lon, lat]).astype(np.float32, copy=False)
                    traj = {
                        "agent_id": (
                            f"robotcar_{_sample_label(sample_sec)}_{date_id}"
                            f"_span{span_id:04d}_block{block_id:08d}_phase{phase:03d}"
                        ),
                        "phase": int(phase),
                        "sample_time_sec": float(sample_sec),
                        "stride": int(stride),
                        "date_id": str(date_id),
                        "span_id": int(span_id),
                        "window_start_idx": int(block_start),
                        "block_id": int(block_id),
                        "block_start_idx": int(block_start),
                        "block_end_idx": int(block_end),
                        "block_native_len": int(block_native_len),
                        "time_id": str(time_id),
                        "source_row_indices": torch.from_numpy(source_indices).to(torch.int32),
                        "n_points": int(window_len),
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
                            "agent_id": traj["agent_id"],
                            "phase": int(phase),
                            "date_id": str(date_id),
                            "span_id": int(span_id),
                            "window_start_idx": int(block_start),
                            "block_id": int(block_id),
                            "time_id": str(time_id),
                            "n_points": int(window_len),
                            "first_ts_us": int(round(ts[0] * 1_000_000.0)),
                            "last_ts_us": int(round(ts[-1] * 1_000_000.0)),
                        }
                    )
                    date_summary["n_trajectories"] += 1
                date_summary["n_time_ids"] += 1

        date_summaries.append(date_summary)

    payload = {
        "status": "completed",
        "output_pt": str(output_pt.resolve()),
        "sample_time_sec": float(sample_sec),
        "sample_label": _sample_label(sample_sec),
        "window_len": int(window_len),
        "window_stride": int(window_stride),
        "gap_multiplier": float(gap_multiplier),
        "tolerance_ratio": float(tolerance_ratio),
        "columns": ["longitude", "latitude"],
        "side_channels": ["latitude_sigma", "longitude_sigma"],
        "timestamp_unit": "unix_seconds",
        "error_range_unit": "meters",
        "error_range_formula": "sqrt(latitude_sigma^2 + longitude_sigma^2)",
        "dicing_mode": "phase_shifted_non_overlapping_time_blocks",
        "time_id_rule": "shared per non-overlapping native block across all phases",
        "n_trajectories": int(len(trajectories)),
        "n_points_total": int(len(trajectories) * int(window_len)),
        "n_dates": int(len(date_summaries)),
        "n_source_files": int(len(global_file_count)),
        "metadata": {
            "n_trajectories": int(len(trajectories)),
            "median_length": int(window_len),
        },
        "date_summaries": date_summaries,
        "trajectory_summaries": trajectory_summaries,
        "trajectories": trajectories,
    }
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_pt)
    return payload


def extract_phase_shift_robotcar(
    *,
    input_root: str,
    input_glob: str,
    output_dir: str,
    sample_seconds: list[float],
    window_len: int = 500,
    window_stride: int = 1,
    gap_multiplier: float = 5.0,
    tolerance_ratio: float = 0.10,
) -> dict[str, Any]:
    root = Path(input_root).resolve()
    out_root = Path(output_dir).resolve()
    files = _discover_gps_files(root, input_glob)

    all_rows: list[GpsRow] = []
    file_summaries: list[dict[str, Any]] = []
    for path in files:
        rows = _load_one_gps_csv(path)
        if not rows:
            continue
        all_rows.extend(rows)
        file_summaries.append(
            {
                "file": str(path.resolve()),
                "n_rows": int(len(rows)),
                "date_min": min(r.date_id for r in rows),
                "date_max": max(r.date_id for r in rows),
            }
        )
    if not all_rows:
        raise ValueError("No valid rows loaded from discovered gps.csv files.")

    rows_by_date = _group_rows_by_date(all_rows)
    sample_summaries: list[dict[str, Any]] = []
    outputs: dict[str, str] = {}
    for sample_sec in sample_seconds:
        label = _sample_label(sample_sec)
        out_pt = out_root / f"RobotCar_{label}.pt"
        payload = _build_one_sample_dataset(
            rows_by_date,
            sample_sec=float(sample_sec),
            window_len=int(window_len),
            window_stride=int(window_stride),
            gap_multiplier=float(gap_multiplier),
            tolerance_ratio=float(tolerance_ratio),
            output_pt=out_pt,
        )
        sample_summaries.append(
            {
                "sample_time_sec": float(sample_sec),
                "sample_label": label,
                "output_pt": str(out_pt.resolve()),
                "n_trajectories": int(payload["n_trajectories"]),
                "n_points_total": int(payload["n_points_total"]),
                "n_dates": int(payload["n_dates"]),
            }
        )
        outputs[label] = str(out_pt.resolve())

    summary = {
        "status": "completed",
        "input_root": str(root),
        "input_glob": str(input_glob),
        "n_input_files": int(len(file_summaries)),
        "n_input_rows": int(len(all_rows)),
        "window_len": int(window_len),
        "window_stride": int(window_stride),
        "gap_multiplier": float(gap_multiplier),
        "tolerance_ratio": float(tolerance_ratio),
        "file_summaries": file_summaries,
        "sample_summaries": sample_summaries,
        "outputs": outputs,
    }
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract phase-shifted RobotCar trajectories with 500-point windows."
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
        "--output-dir",
        type=str,
        default="./dataset/raw/RobotCar/extracted",
        help="Output directory for RobotCar_<sample>.pt files.",
    )
    parser.add_argument(
        "--sample-seconds",
        type=str,
        default="1,2",
        help="Comma-separated sample intervals in seconds.",
    )
    parser.add_argument(
        "--window-len",
        type=int,
        default=500,
        help="Trajectory window length.",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=1,
        help="Block-hop multiplier (1 means adjacent non-overlapping time_id blocks).",
    )
    parser.add_argument(
        "--gap-multiplier",
        type=float,
        default=5.0,
        help="Split spans when timestamp gap > gap-multiplier * native_dt.",
    )
    parser.add_argument(
        "--tolerance-ratio",
        type=float,
        default=0.10,
        help="Allowed relative mismatch between desired/native ratio and rounded stride.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    sample_seconds = _parse_sample_seconds(args.sample_seconds)
    summary = extract_phase_shift_robotcar(
        input_root=str(args.input_root),
        input_glob=str(args.input_glob),
        output_dir=str(args.output_dir),
        sample_seconds=sample_seconds,
        window_len=int(args.window_len),
        window_stride=int(args.window_stride),
        gap_multiplier=float(args.gap_multiplier),
        tolerance_ratio=float(args.tolerance_ratio),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
