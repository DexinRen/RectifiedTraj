#!/usr/bin/env python3
"""
Build RobotCar phase-shifted datasets and inject 3 noise variants.

Outputs:
- ./dataset/raw/RobotCar/extracted/RobotCar_1s.pt
- ./dataset/raw/RobotCar/extracted/RobotCar_2s.pt
- ./dataset/processed/RobotCar/RobotCar_<sample>_<noise>.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from add_noise.inject_robotcar_noise_pt import inject_noise_robotcar_pt
from utils.data_loader_standalone import StandaloneDataLoader
from utils.data_processor.robotcar_phase_shift_extractor import (
    _parse_sample_seconds,
    _sample_label,
    extract_phase_shift_robotcar,
)


def _load_payload(path: Path) -> dict[str, Any]:
    blob = torch.load(path, map_location="cpu")
    if not isinstance(blob, dict):
        raise ValueError(f"PT payload must be dict: {path}")
    if "trajectories" not in blob or not isinstance(blob["trajectories"], list):
        raise ValueError(f"PT payload missing trajectories list: {path}")
    return blob


def _validate_window_len(payload: dict[str, Any], window_len: int) -> dict[str, Any]:
    lengths = []
    for row in payload["trajectories"]:
        data = row.get("data")
        arr = data.detach().cpu().numpy() if torch.is_tensor(data) else np.asarray(data)
        lengths.append(int(arr.shape[0]))
    bad = [i for i, n in enumerate(lengths) if n != int(window_len)]
    if bad:
        raise ValueError(f"Found non-{window_len} trajectories: first_bad_indices={bad[:8]}")
    return {
        "n_trajectories": int(len(lengths)),
        "window_len": int(window_len),
    }


def _validate_time_id_rules(payload: dict[str, Any]) -> dict[str, Any]:
    same_start_to_time: dict[tuple[str, int, int], set[str]] = {}
    time_groups: dict[str, list[dict[str, Any]]] = {}
    for row in payload["trajectories"]:
        date_id = str(row.get("date_id"))
        span_id = int(row.get("span_id"))
        wstart = int(row.get("block_start_idx", row.get("window_start_idx", -1)))
        time_id = str(row.get("time_id"))
        key = (date_id, span_id, wstart)
        same_start_to_time.setdefault(key, set()).add(time_id)
        time_groups.setdefault(time_id, []).append(row)

    bad_same_start = [k for k, v in same_start_to_time.items() if len(v) != 1]
    if bad_same_start:
        raise ValueError(
            "time_id mismatch for same date/span/window_start across phases: "
            f"{bad_same_start[:5]}"
        )

    bad_phase_groups = []
    for time_id, rows in time_groups.items():
        phases = sorted({int(r.get("phase")) for r in rows})
        strides = {int(r.get("stride")) for r in rows}
        starts = {int(r.get("block_start_idx", r.get("window_start_idx", -1))) for r in rows}
        if len(strides) != 1 or len(starts) != 1:
            bad_phase_groups.append((time_id, sorted(strides), sorted(starts)))
            if len(bad_phase_groups) >= 5:
                break
            continue
        expected = next(iter(strides))
        if len(phases) != expected:
            bad_phase_groups.append((time_id, phases, expected))
            if len(bad_phase_groups) >= 5:
                break
    if bad_phase_groups:
        raise ValueError(
            "time_id group has invalid phase/stride composition: "
            f"{bad_phase_groups}"
        )

    return {
        "n_time_groups": int(len(same_start_to_time)),
        "n_phase_groups": int(len(time_groups)),
    }


def _validate_no_cross_date(payload: dict[str, Any]) -> dict[str, Any]:
    bad = 0
    for row in payload["trajectories"]:
        ts = row.get("timestamp")
        ts_arr = ts.detach().cpu().numpy() if torch.is_tensor(ts) else np.asarray(ts)
        if ts_arr.size <= 0:
            continue
        dt_days = np.asarray(ts_arr, dtype=np.float64) // 86400.0
        if np.max(dt_days) != np.min(dt_days):
            bad += 1
    if bad > 0:
        raise ValueError(f"Found trajectories crossing UTC date boundary: {bad}")
    return {"cross_date_violations": int(bad)}


def _validate_span_continuity(payload: dict[str, Any]) -> dict[str, Any]:
    violations = 0
    for row in payload["trajectories"]:
        idxs = row.get("source_row_indices")
        arr = idxs.detach().cpu().numpy() if torch.is_tensor(idxs) else np.asarray(idxs)
        if arr.size <= 1:
            continue
        stride = int(row.get("stride"))
        diffs = np.diff(arr.astype(np.int64, copy=False))
        if not np.all(diffs == stride):
            violations += 1
    if violations > 0:
        raise ValueError(f"Found trajectories with broken source continuity: {violations}")
    return {"continuity_violations": int(violations)}


def _validate_loader_compat(path: Path) -> dict[str, Any]:
    loader = StandaloneDataLoader(
        mode="test",
        data_dir=str(path.parent),
        file_pattern=path.name,
    )
    count = 0
    first_shape = None
    for rec in loader.iter_trajectory_sequences():
        noisy = np.asarray(rec["noisy_lonlat"], dtype=float)
        if noisy.ndim != 2 or noisy.shape[1] != 2:
            raise ValueError(f"Invalid noisy_lonlat shape in loader output: {noisy.shape}")
        err = rec.get("error_range")
        if err is None:
            raise ValueError("error_range missing in loader trajectory sequence.")
        if first_shape is None:
            first_shape = [int(noisy.shape[0]), int(noisy.shape[1])]
        count += 1
        if count >= 10:
            break
    if count <= 0:
        raise ValueError(f"No records read by StandaloneDataLoader from {path}")
    return {"preview_records": int(count), "first_shape": first_shape}


def _noise_center_ok(stats: dict[str, Any]) -> bool:
    mean = float(stats.get("mean", 0.0))
    med = float(stats.get("median", 0.0))
    return (10.0 <= mean <= 20.0) and (10.0 <= med <= 20.0)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build RobotCar phase-shifted 500-point datasets with three noise variants."
    )
    parser.add_argument("--input-root", type=str, default="./dataset/raw/RobotCar")
    parser.add_argument("--input-glob", type=str, default="**/gps.csv")
    parser.add_argument("--raw-output-dir", type=str, default="./dataset/raw/RobotCar/extracted")
    parser.add_argument("--processed-output-dir", type=str, default="./dataset/processed/RobotCar")
    parser.add_argument("--sample-seconds", type=str, default="1,2")
    parser.add_argument("--window-len", type=int, default=500)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--gap-multiplier", type=float, default=5.0)
    parser.add_argument("--tolerance-ratio", type=float, default=0.10)

    parser.add_argument("--seed-base", type=int, default=11)
    parser.add_argument("--target-mean-m", type=float, default=15.0)
    parser.add_argument("--target-median-m", type=float, default=15.0)
    parser.add_argument("--target-std-m", type=float, default=9.0)
    parser.add_argument("--scale-mode", type=str, default="weighted", choices=["mean", "median", "weighted"])
    parser.add_argument("--weight-mean", type=float, default=1.0)
    parser.add_argument("--weight-median", type=float, default=4.0)
    parser.add_argument("--weight-std", type=float, default=1.0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    sample_seconds = _parse_sample_seconds(args.sample_seconds)

    # 1) Extraction
    extraction_summary = extract_phase_shift_robotcar(
        input_root=str(args.input_root),
        input_glob=str(args.input_glob),
        output_dir=str(args.raw_output_dir),
        sample_seconds=sample_seconds,
        window_len=int(args.window_len),
        window_stride=int(args.window_stride),
        gap_multiplier=float(args.gap_multiplier),
        tolerance_ratio=float(args.tolerance_ratio),
    )

    raw_dir = Path(args.raw_output_dir).resolve()
    processed_dir = Path(args.processed_output_dir).resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)

    extraction_checks: dict[str, Any] = {}
    extracted_paths: dict[str, Path] = {}
    for sec in sample_seconds:
        label = _sample_label(sec)
        path = raw_dir / f"RobotCar_{label}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Expected extracted file not found: {path}")
        payload = _load_payload(path)
        checks = {
            "window_len": _validate_window_len(payload, int(args.window_len)),
            "time_id_rules": _validate_time_id_rules(payload),
            "no_cross_date": _validate_no_cross_date(payload),
            "span_continuity": _validate_span_continuity(payload),
            "loader_compat": _validate_loader_compat(path),
            "n_trajectories": int(len(payload["trajectories"])),
        }
        extraction_checks[label] = checks
        extracted_paths[label] = path

    # 2) Noise variants
    noise_specs = [
        ("HeteroGaussian", 0),
        ("PiecewiseBiasJitter", 1),
        ("OU", 2),
    ]
    noise_outputs: list[dict[str, Any]] = []
    for sec in sample_seconds:
        label = _sample_label(sec)
        source_path = extracted_paths[label]
        source_payload = _load_payload(source_path)
        source_time_ids = [str(t.get("time_id")) for t in source_payload["trajectories"]]
        for noise_name, noise_offset in noise_specs:
            out_path = processed_dir / f"RobotCar_{label}_{noise_name}.pt"
            inject_summary = inject_noise_robotcar_pt(
                input_pt=str(source_path),
                output_pt=str(out_path),
                noise_type=noise_name,
                seed=int(args.seed_base + noise_offset),
                target_mean_m=float(args.target_mean_m),
                target_median_m=float(args.target_median_m),
                target_std_m=float(args.target_std_m),
                scale_mode=str(args.scale_mode),
                weight_mean=float(args.weight_mean),
                weight_median=float(args.weight_median),
                weight_std=float(args.weight_std),
            )
            out_payload = _load_payload(out_path)
            out_time_ids = [str(t.get("time_id")) for t in out_payload["trajectories"]]
            if source_time_ids != out_time_ids:
                raise ValueError(f"time_id was not preserved in noise output: {out_path}")

            realized = ((out_payload.get("noise") or {}).get("realized_error_stats_m") or {})
            if not _noise_center_ok(realized):
                raise ValueError(
                    f"Noise center out of expected range (10..20 m mean/median): "
                    f"{out_path} stats={realized}"
                )

            noise_outputs.append(
                {
                    "sample_label": label,
                    "noise_name": noise_name,
                    "output_pt": str(out_path),
                    "inject_summary": inject_summary,
                    "loader_compat": _validate_loader_compat(out_path),
                    "realized_error_stats_m": realized,
                    "time_id_preserved": True,
                }
            )

    report = {
        "status": "completed",
        "config": {
            "input_root": str(Path(args.input_root).resolve()),
            "input_glob": str(args.input_glob),
            "raw_output_dir": str(raw_dir),
            "processed_output_dir": str(processed_dir),
            "sample_seconds": sample_seconds,
            "window_len": int(args.window_len),
            "window_stride": int(args.window_stride),
            "gap_multiplier": float(args.gap_multiplier),
            "tolerance_ratio": float(args.tolerance_ratio),
            "seed_base": int(args.seed_base),
            "target_mean_m": float(args.target_mean_m),
            "target_median_m": float(args.target_median_m),
            "target_std_m": float(args.target_std_m),
            "scale_mode": str(args.scale_mode),
            "weight_mean": float(args.weight_mean),
            "weight_median": float(args.weight_median),
            "weight_std": float(args.weight_std),
        },
        "extraction_summary": extraction_summary,
        "extraction_checks": extraction_checks,
        "noise_outputs": noise_outputs,
    }

    report_path = processed_dir / "robotcar_phase_shift_build_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
