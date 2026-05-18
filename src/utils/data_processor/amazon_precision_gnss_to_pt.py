#!/usr/bin/env python3
"""Convert Amazon Precision GNSS CSV trajectories into test trajectory PT files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


RAW_ROOT = Path("dataset/raw/AmazonPrecisionGNSS/trajectories")
OUTPUT_ROOT = Path("dataset/processed/AmazonPrecisionGNSS/test/traj_test")
SAMPLE_SPECS = (
    ("0p1s", 0.1, 1),
    ("1s", 1.0, 10),
    ("2s", 2.0, 20),
    ("5s", 5.0, 50),
    ("10s", 10.0, 100),
)


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {"n": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}
    return {
        "n": float(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
    }


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _timestamps_from_df(df: pd.DataFrame) -> np.ndarray:
    return (
        pd.to_numeric(df["GPS Week"], errors="raise").to_numpy(dtype=np.float64) * 604800.0
        + pd.to_numeric(df["GPS TOW [s]"], errors="raise").to_numpy(dtype=np.float64)
    )


def _assert_strictly_increasing(ts: np.ndarray, *, label: str) -> None:
    diffs = np.diff(ts)
    if diffs.size == 0:
        raise ValueError(f"{label}: empty timestamps")
    if not np.all(diffs > 0.0):
        bad = int(np.where(diffs <= 0.0)[0][0])
        raise ValueError(f"{label}: timestamps are not strictly increasing at index {bad}")


def _downsample_pair(
    precision_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    *,
    stride: int,
    agent_id: int,
) -> dict[str, Any]:
    precision_ts = _timestamps_from_df(precision_df)
    reference_ts = _timestamps_from_df(reference_df)
    _assert_strictly_increasing(precision_ts, label=f"agent {agent_id} precision")
    _assert_strictly_increasing(reference_ts, label=f"agent {agent_id} reference")
    if precision_ts.shape != reference_ts.shape or not np.array_equal(precision_ts, reference_ts):
        raise ValueError(f"agent {agent_id}: precision/reference timestamps are not exactly aligned")

    idx = np.arange(0, precision_ts.shape[0], int(stride), dtype=np.int64)
    data = np.column_stack(
        [
            pd.to_numeric(precision_df["Lon [deg]"], errors="raise").to_numpy(dtype=np.float32)[idx],
            pd.to_numeric(precision_df["Lat [deg]"], errors="raise").to_numpy(dtype=np.float32)[idx],
        ]
    )
    label = np.column_stack(
        [
            pd.to_numeric(reference_df["Lon [deg]"], errors="raise").to_numpy(dtype=np.float32)[idx],
            pd.to_numeric(reference_df["Lat [deg]"], errors="raise").to_numpy(dtype=np.float32)[idx],
        ]
    )
    ts_abs = precision_ts[idx]
    ts = (ts_abs - ts_abs[0]).astype(np.float32, copy=False)
    return {
        "agent_id": int(agent_id),
        "n_points": int(idx.shape[0]),
        "data": torch.tensor(data, dtype=torch.float32),
        "label": torch.tensor(label, dtype=torch.float32),
        "timestamp": torch.tensor(ts, dtype=torch.float32),
    }


def _collect_trajectory_pairs(raw_root: Path) -> list[tuple[int, Path, Path]]:
    pairs: list[tuple[int, Path, Path]] = []
    for traj_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        name = traj_dir.name
        try:
            agent_id = int(name.split("_")[-1])
        except Exception as exc:
            raise ValueError(f"Unexpected trajectory directory name: {name}") from exc
        precision_path = traj_dir / "precision_gnss.csv"
        reference_path = traj_dir / "reference_gnss.csv"
        if not precision_path.exists() or not reference_path.exists():
            raise FileNotFoundError(f"Missing CSV pair under {traj_dir}")
        pairs.append((agent_id, precision_path, reference_path))
    if len(pairs) != 4:
        raise ValueError(f"Expected 4 trajectory folders, found {len(pairs)}")
    return pairs


def _build_metadata(sample_label: str, sample_time_sec: float, trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = np.asarray([int(t["n_points"]) for t in trajectories], dtype=np.int64)
    interval_batches = []
    for traj in trajectories:
        ts = traj["timestamp"].detach().cpu().numpy().astype(np.float64, copy=False)
        if ts.size >= 2:
            interval_batches.append(np.diff(ts))
    intervals = np.concatenate(interval_batches, axis=0) if interval_batches else np.array([], dtype=np.float64)
    total_points = int(lengths.sum()) if lengths.size else 0
    avg_length = int(round(float(lengths.mean()))) if lengths.size else 0
    median_length = int(np.median(lengths)) if lengths.size else 0
    return {
        "n_trajectories": int(len(trajectories)),
        "total_points": total_points,
        "median_length": median_length,
        "target_M": int(len(trajectories)),
        "target_N": avg_length,
        "sampler": f"constant_{sample_label}",
        "sample_interval_stats_sec": _summary_stats(intervals),
        "interval_sec": float(sample_time_sec),
        "interval_label": str(sample_label),
        "sample_time_label": str(sample_label),
        "avg_length": avg_length,
        "source_dataset": "AmazonPrecisionGNSS",
        "source_format": "paired_csv",
        "ordering_verified": True,
        "timestamp_alignment_verified": True,
    }


def convert_amazon_precision_gnss(
    *,
    raw_root: Path = RAW_ROOT,
    output_root: Path = OUTPUT_ROOT,
) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    pairs = _collect_trajectory_pairs(raw_root)
    written: list[Path] = []

    loaded_pairs = []
    for agent_id, precision_path, reference_path in pairs:
        loaded_pairs.append((agent_id, _load_csv(precision_path), _load_csv(reference_path)))

    for sample_label, sample_time_sec, stride in SAMPLE_SPECS:
        trajectories = [
            _downsample_pair(
                precision_df=precision_df,
                reference_df=reference_df,
                stride=stride,
                agent_id=agent_id,
            )
            for agent_id, precision_df, reference_df in loaded_pairs
        ]
        metadata = _build_metadata(sample_label, sample_time_sec, trajectories)
        filename = f"traj_{sample_label}_{len(trajectories)}_{metadata['avg_length']}.pt"
        output_path = output_root / filename
        torch.save({"trajectories": trajectories, "metadata": metadata}, output_path)
        written.append(output_path)
    return written


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Amazon Precision GNSS CSV trajectories into RectifiedTraj test PT files."
    )
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT, help="Root directory containing trajectory_*/ CSV pairs.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT, help="Output directory for traj_*.pt files.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    written = convert_amazon_precision_gnss(raw_root=args.raw_root, output_root=args.output_root)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
