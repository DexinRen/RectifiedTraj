#!/usr/bin/env python3
"""
Split Oxford RobotCar gps.csv into multiple trajectories for a desired sample time.

Method:
- Detect native sample time from timestamp diffs (median).
- Compute stride = round(desired_sample_sec / native_sample_sec).
- Split points by modulo phase using source row index k:
  phase p gets rows where k % stride == p.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split RobotCar gps.csv into modulo trajectories at a target sample time."
    )
    parser.add_argument(
        "--gps-csv",
        type=str,
        default="./dataset/raw/RobotCar/sample/gps.csv",
        help="Path to RobotCar gps.csv input.",
    )
    parser.add_argument(
        "--desired-sample-sec",
        type=float,
        required=True,
        help="Target sample time in seconds (for example 1, 2, 5).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./dataset/raw/RobotCar/extracted",
        help="Directory to write output .pt file.",
    )
    parser.add_argument(
        "--output-pt",
        type=str,
        default=None,
        help="Optional explicit output .pt path. Overrides --output-dir naming.",
    )
    parser.add_argument(
        "--base-sample-sec",
        type=float,
        default=None,
        help="Optional native sample time override in seconds. If omitted, auto-detected.",
    )
    parser.add_argument(
        "--tolerance-ratio",
        type=float,
        default=0.10,
        help="Allowed relative mismatch between desired/base and rounded stride.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Optional cap of output points per split trajectory.",
    )
    parser.add_argument(
        "--save-stripped-csv",
        action="store_true",
        help="Also write stripped GPS-only CSV (timestamp, latitude, longitude).",
    )
    return parser


def _load_csv_rows(gps_csv: Path) -> tuple[list[str], list[list[str]], list[int], int]:
    if not gps_csv.exists():
        raise FileNotFoundError(f"gps.csv not found: {gps_csv}")

    with gps_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError(f"Empty CSV: {gps_csv}")

        timestamp_idx = -1
        for i, name in enumerate(header):
            if str(name).strip().lower() == "timestamp":
                timestamp_idx = i
                break
        if timestamp_idx < 0:
            raise ValueError("CSV header does not contain 'timestamp' column.")

        rows: list[list[str]] = []
        timestamps_us: list[int] = []
        for row in reader:
            if not row:
                continue
            if len(row) <= timestamp_idx:
                continue
            try:
                ts = int(float(row[timestamp_idx]))
            except Exception:
                continue
            rows.append(row)
            timestamps_us.append(ts)

    if not rows:
        raise ValueError(f"No valid data rows found in: {gps_csv}")

    return header, rows, timestamps_us, timestamp_idx


def _infer_base_sample_sec(timestamps_us: list[int]) -> float:
    diffs = []
    for i in range(1, len(timestamps_us)):
        d = timestamps_us[i] - timestamps_us[i - 1]
        if d > 0:
            diffs.append(d / 1_000_000.0)
    if not diffs:
        raise ValueError("Cannot infer native sample time (no positive timestamp diffs).")
    return float(statistics.median(diffs))


def _compute_stride(desired_sample_sec: float, base_sample_sec: float, tolerance_ratio: float) -> tuple[int, float]:
    if desired_sample_sec <= 0:
        raise ValueError("--desired-sample-sec must be > 0.")
    if base_sample_sec <= 0:
        raise ValueError("base sample time must be > 0.")

    ratio = float(desired_sample_sec / base_sample_sec)
    stride = max(1, int(round(ratio)))
    mismatch = abs(ratio - float(stride)) / max(ratio, 1e-12)

    if mismatch > float(tolerance_ratio):
        raise ValueError(
            "Desired sample time is not close to an integer multiple of native sample time: "
            f"desired={desired_sample_sec:.6f}s base={base_sample_sec:.6f}s "
            f"ratio={ratio:.6f} rounded_stride={stride} mismatch={mismatch:.6f} "
            f"(tolerance={tolerance_ratio:.6f})."
        )
    return stride, ratio


def _find_required_idx(header: list[str], col: str) -> int:
    target = col.strip().lower()
    for i, name in enumerate(header):
        if str(name).strip().lower() == target:
            return i
    raise ValueError(f"CSV header does not contain required column: {col}")


def _sample_label(seconds_value: float) -> str:
    val = float(seconds_value)
    if abs(val - round(val)) < 1e-9:
        return f"{int(round(val))}s"
    return f"{val:g}s".replace(".", "p")


def _write_stripped_csv(
    path: Path,
    rows: list[list[str]],
    timestamp_idx: int,
    lat_idx: int,
    lon_idx: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "latitude", "longitude"])
        for row in rows:
            writer.writerow([row[timestamp_idx], row[lat_idx], row[lon_idx]])


def split_robotcar_gps(
    gps_csv: str,
    desired_sample_sec: float,
    output_dir: str,
    *,
    output_pt: str | None = None,
    base_sample_sec: float | None = None,
    tolerance_ratio: float = 0.10,
    max_points: int | None = None,
    save_stripped_csv: bool = False,
) -> dict[str, Any]:
    gps_path = Path(gps_csv).resolve()
    out_root = Path(output_dir).resolve()

    header, rows, timestamps_us, timestamp_idx = _load_csv_rows(gps_path)
    lat_idx = _find_required_idx(header, "latitude")
    lon_idx = _find_required_idx(header, "longitude")
    lat_sigma_idx = _find_required_idx(header, "latitude_sigma")
    lon_sigma_idx = _find_required_idx(header, "longitude_sigma")
    kept_columns = ["timestamp", "latitude", "longitude"]
    dropped_columns = [
        str(name)
        for name in header
        if str(name).strip().lower() not in {"timestamp", "latitude", "longitude"}
    ]

    native_sample_sec = (
        float(base_sample_sec) if base_sample_sec is not None else _infer_base_sample_sec(timestamps_us)
    )
    stride, ratio = _compute_stride(float(desired_sample_sec), native_sample_sec, float(tolerance_ratio))
    effective_sample_sec = native_sample_sec * float(stride)

    n = len(rows)
    per_traj_summary: list[dict[str, Any]] = []
    per_traj_records: list[dict[str, Any]] = []

    for phase in range(stride):
        idxs = list(range(phase, n, stride))
        if max_points is not None and max_points > 0:
            idxs = idxs[: int(max_points)]

        if idxs:
            one_ts = np.asarray([float(rows[i][timestamp_idx]) for i in idxs], dtype=np.float64)
            one_lat = np.asarray([float(rows[i][lat_idx]) for i in idxs], dtype=np.float64)
            one_lon = np.asarray([float(rows[i][lon_idx]) for i in idxs], dtype=np.float64)
            one_lat_sigma = np.asarray([float(rows[i][lat_sigma_idx]) for i in idxs], dtype=np.float64)
            one_lon_sigma = np.asarray([float(rows[i][lon_sigma_idx]) for i in idxs], dtype=np.float64)
            one_acc_m = np.sqrt(np.square(one_lat_sigma) + np.square(one_lon_sigma))
            finite_acc = one_acc_m[np.isfinite(one_acc_m) & (one_acc_m >= 0.0)]
            if finite_acc.size <= 0:
                raise ValueError("No valid accuracy values from latitude_sigma/longitude_sigma.")
            acc_fill = float(np.median(finite_acc))
            one_acc_m = np.where(np.isfinite(one_acc_m) & (one_acc_m >= 0.0), one_acc_m, acc_fill)
            one_acc_m = np.maximum(one_acc_m, 0.0)

            # Project schema expects lon-lat in both data (noisy) and label (reference).
            lonlat = np.column_stack([one_lon, one_lat]).astype(np.float32, copy=False)
            ts_sec = (one_ts / 1_000_000.0).astype(np.float64, copy=False)
            err = one_acc_m.astype(np.float32, copy=False)
            lat_sigma = one_lat_sigma.astype(np.float32, copy=False)
            lon_sigma = one_lon_sigma.astype(np.float32, copy=False)

            data_tensor = torch.from_numpy(lonlat).to(torch.float32)
            label_tensor = torch.from_numpy(lonlat.copy()).to(torch.float32)
            ts_tensor = torch.from_numpy(ts_sec).to(torch.float64)
            err_tensor = torch.from_numpy(err).to(torch.float32)
            lat_sigma_tensor = torch.from_numpy(lat_sigma).to(torch.float32)
            lon_sigma_tensor = torch.from_numpy(lon_sigma).to(torch.float32)
            first_ts = int(one_ts[0])
            last_ts = int(one_ts[-1])
            duration_sec = max(0.0, (last_ts - first_ts) / 1_000_000.0)
        else:
            data_tensor = torch.empty((0, 2), dtype=torch.float32)
            label_tensor = torch.empty((0, 2), dtype=torch.float32)
            ts_tensor = torch.empty((0,), dtype=torch.float64)
            err_tensor = torch.empty((0,), dtype=torch.float32)
            lat_sigma_tensor = torch.empty((0,), dtype=torch.float32)
            lon_sigma_tensor = torch.empty((0,), dtype=torch.float32)
            first_ts = None
            last_ts = None
            duration_sec = 0.0

        one_summary = {
            "phase": int(phase),
            "n_points": int(len(idxs)),
            "first_timestamp_us": first_ts,
            "last_timestamp_us": last_ts,
            "duration_sec": float(duration_sec),
            "accuracy_median_m": float(torch.median(err_tensor).item()) if err_tensor.numel() > 0 else 0.0,
        }
        per_traj_summary.append(one_summary)

        per_traj_records.append(
            {
                "agent_id": f"robotcar_phase_{phase:03d}",
                "phase": int(phase),
                "n_points": int(len(idxs)),
                "source_row_indices": idxs,
                "data": data_tensor,
                "label": label_tensor,
                "timestamp": ts_tensor,
                "error_range": err_tensor,
                "accuracy": err_tensor.clone(),
                "latitude_sigma": lat_sigma_tensor,
                "longitude_sigma": lon_sigma_tensor,
            }
        )

    if output_pt:
        out_pt = Path(output_pt).resolve()
    else:
        out_pt = out_root / f"robotcar_{_sample_label(desired_sample_sec)}.pt"
    stripped_csv = out_root / "gps_stripped.csv"

    payload = {
        "status": "completed",
        "input_csv": str(gps_path),
        "output_pt": str(out_pt),
        "n_input_points": int(n),
        "n_trajectories": int(stride),
        "native_sample_sec": float(native_sample_sec),
        "desired_sample_sec": float(desired_sample_sec),
        "effective_sample_sec": float(effective_sample_sec),
        "stride": int(stride),
        "ratio_desired_over_native": float(ratio),
        "tolerance_ratio": float(tolerance_ratio),
        "max_points_per_trajectory": int(max_points) if max_points is not None else None,
        "columns": ["longitude", "latitude"],
        "timestamp_unit": "unix_seconds",
        "error_range_unit": "meters",
        "error_range_formula": "sqrt(latitude_sigma^2 + longitude_sigma^2)",
        "trajectory_summaries": per_traj_summary,
        "trajectories": per_traj_records,
        "dropped_columns": dropped_columns,
    }

    out_pt.parent.mkdir(parents=True, exist_ok=True)
    if bool(save_stripped_csv):
        _write_stripped_csv(
            stripped_csv,
            rows,
            timestamp_idx=timestamp_idx,
            lat_idx=lat_idx,
            lon_idx=lon_idx,
        )
    torch.save(payload, out_pt)

    summary = {
        "status": payload["status"],
        "input_csv": payload["input_csv"],
        "output_pt": payload["output_pt"],
        "n_input_points": payload["n_input_points"],
        "n_trajectories": payload["n_trajectories"],
        "native_sample_sec": payload["native_sample_sec"],
        "desired_sample_sec": payload["desired_sample_sec"],
        "effective_sample_sec": payload["effective_sample_sec"],
        "stride": payload["stride"],
        "ratio_desired_over_native": payload["ratio_desired_over_native"],
        "tolerance_ratio": payload["tolerance_ratio"],
        "max_points_per_trajectory": payload["max_points_per_trajectory"],
        "kept_columns": kept_columns,
        "dropped_columns": dropped_columns,
        "stripped_csv": str(stripped_csv) if bool(save_stripped_csv) else None,
        "trajectory_summaries": per_traj_summary,
    }
    return summary


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    result = split_robotcar_gps(
        gps_csv=str(args.gps_csv),
        desired_sample_sec=float(args.desired_sample_sec),
        output_dir=str(args.output_dir),
        output_pt=str(args.output_pt) if args.output_pt else None,
        base_sample_sec=float(args.base_sample_sec) if args.base_sample_sec is not None else None,
        tolerance_ratio=float(args.tolerance_ratio),
        max_points=int(args.max_points) if args.max_points is not None else None,
        save_stripped_csv=bool(args.save_stripped_csv),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
