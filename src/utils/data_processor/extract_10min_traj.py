#!/usr/bin/env python3
"""
Standalone extractor for fixed-interval-sampled trajectories.

Generates regular datasets for each configured interval.
Default intervals:
  - 2 minutes
  - 14 minutes
  - 20 minutes

Uses the same core logic as traj_extractor (test split scan, random agents).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
import logging
import sys
import numpy as np

# Reserve CPU for other workloads: use at most (total_cores - 2) threads by default.
# Users can still override by pre-setting these environment variables.
_cpu_total = os.cpu_count() or 1
_cpu_budget = max(1, _cpu_total - 2)
for _var in (
    "POLARS_MAX_THREADS",
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, str(_cpu_budget))

import polars as pl
import torch

# Ensure src is on sys.path when running as a script
SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
    
from src.utils.data_processor.traj_extractor import (
    scan_parquet_metadata,
    find_agent_files,
    data_processor,
    _detect_column_map,
)

logger = logging.getLogger(__name__)
torch.set_num_threads(_cpu_budget)
if hasattr(torch, "set_num_interop_threads"):
    torch.set_num_interop_threads(max(1, min(2, _cpu_budget)))

# ================================================================
# CONFIG
# ================================================================
PARQUET_DIR = "./dataset/raw"
OUTPUT_DIR_TMPL = "./dataset/processed/full_traj_{interval}min"

M = 200
POINTS_N = 5000
INTERVALS_MIN = [2, 14, 20]

# Tolerance for matching target timestamps (seconds)
TIME_TOLERANCE_SEC = 60


# ================================================================
# HELPERS
# ================================================================

def _load_agent_df(
    agent_id: int,
    parquet_paths: list[str],
    window_start: Optional[float] = None,
    window_end: Optional[float] = None,
    column_map: Optional[dict] = None,
) -> Optional[pl.DataFrame]:
    if parquet_paths:
        column_map = _detect_column_map(str(Path(parquet_paths[0]).parent), column_map)
    else:
        column_map = _detect_column_map(PARQUET_DIR, column_map)

    agent_col = column_map["agent"]
    ts_col = column_map["timestamp"]
    lon_n_col = column_map["longitude_n"]
    lat_n_col = column_map["latitude_n"]
    lon_col = column_map["longitude"]
    lat_col = column_map["latitude"]

    frames = []
    for pq_path in parquet_paths:
        path = Path(pq_path)
        if not path.exists():
            continue

        scan = pl.scan_parquet(path).filter(
            (pl.col(agent_col) == agent_id)
            & pl.col(lon_n_col).is_not_null()
            & pl.col(lat_n_col).is_not_null()
            & pl.col(lon_col).is_not_null()
            & pl.col(lat_col).is_not_null()
            & pl.col(lon_n_col).is_finite()
            & pl.col(lat_n_col).is_finite()
            & pl.col(lon_col).is_finite()
            & pl.col(lat_col).is_finite()
        )

        if window_start is not None and window_end is not None:
            start_sec = int(window_start)
            end_sec = int(window_end)
            ts_sec = pl.col(ts_col).cast(pl.Utf8).str.strptime(pl.Datetime, strict=False).dt.epoch("s")
            scan = scan.filter((ts_sec >= start_sec) & (ts_sec <= end_sec))

        df = (
            scan.select([
                ts_col,
                lon_n_col,
                lat_n_col,
                lon_col,
                lat_col,
            ])
            .sort(ts_col)
            .collect()
        )

        if len(df) > 0:
            frames.append(df)

    if not frames:
        return None

    return pl.concat(frames).sort(ts_col)


def _timestamps_to_unix_seconds(ts_col: np.ndarray) -> np.ndarray:
    if np.issubdtype(ts_col.dtype, np.datetime64):
        return ts_col.astype("datetime64[s]").astype(float)
    if ts_col.dtype == object:
        return ts_col.astype("datetime64[s]").astype(float)
    return ts_col.astype(float)


def _sample_fixed_interval(
    df: pl.DataFrame,
    target_points: int,
    interval_min: int,
    tolerance_sec: int,
) -> Optional[dict]:
    ts_col = df.columns[0]
    lon_n_col = df.columns[1]
    lat_n_col = df.columns[2]
    lon_col = df.columns[3]
    lat_col = df.columns[4]

    ts = _timestamps_to_unix_seconds(df[ts_col].to_numpy())
    if ts.size == 0:
        return None

    interval_sec = interval_min * 60

    # try multiple start indices until we can fill all points
    max_start_idx = max(0, len(ts) - target_points)
    for start_idx in range(0, min(max_start_idx + 1, 2000)):
        start_time = ts[start_idx]

        idxs = []
        current_time = start_time
        last_idx = start_idx

        for _ in range(target_points):
            target_time = current_time
            idx = np.searchsorted(ts, target_time, side="left")
            if idx >= len(ts):
                break
            if idx < last_idx:
                idx = last_idx
            idxs.append(idx)
            last_idx = idx
            current_time = ts[idx] + interval_sec

        if len(idxs) != target_points:
            continue

        idxs = np.array(idxs, dtype=int)
        sampled = df[idxs]
        return {
            "timestamp": ts[idxs],
            "longitude_n": sampled[lon_n_col].to_numpy(),
            "latitude_n": sampled[lat_n_col].to_numpy(),
            "longitude": sampled[lon_col].to_numpy(),
            "latitude": sampled[lat_col].to_numpy(),
        }

    return None


def _save_dataset(
    processed_trajs: list[dict],
    output_dir: str,
    prefix: str,
    target_m: int,
    n_points: int,
    interval_min: int,
) -> str:
    n_trajectories = len(processed_trajs)
    lengths = [t["n_points"] for t in processed_trajs]
    median_length = int(np.median(lengths)) if lengths else 0
    total_points = sum(lengths)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    filename = f"{prefix}_{n_trajectories}_{median_length}.pt"
    output_file = out_path / filename

    save_data = {
        "trajectories": [
            {
                "agent_id": t["agent_id"],
                "n_points": t["n_points"],
                "data": torch.tensor(t["data"], dtype=torch.float32),
                "label": torch.tensor(t["label"], dtype=torch.float32),
            }
            for t in processed_trajs
        ],
        "metadata": {
            "n_trajectories": n_trajectories,
            "total_points": total_points,
            "median_length": median_length,
            "target_M": target_m,
            "n_points": n_points,
            "interval_min": interval_min,
        },
    }

    torch.save(save_data, output_file)
    return str(output_file)


# ================================================================
# MAIN
# ================================================================

def _extract_single_interval(
    *,
    metadata: dict,
    all_agents: list[int],
    output_dir: str,
    m: int,
    n_points: int,
    interval_min: int,
    time_tolerance_sec: int,
) -> dict:
    logger.info(
        "Extracting fixed-interval trajectories: M=%d, N=%d, interval=%d min",
        m,
        n_points,
        interval_min,
    )

    agents = list(all_agents)
    np.random.shuffle(agents)

    processed_regular = []
    agent_idx = 0
    extraction_failures = 0
    for agent_id in agents:
        if len(processed_regular) >= m:
            break

        agent_idx += 1
        agent_files = find_agent_files(agent_id, metadata)
        if not agent_files:
            extraction_failures += 1
            continue

        required_span = (n_points - 1) * interval_min * 60
        candidate_files = [(p, start, end) for p, start, end in agent_files if (end - start) >= required_span]
        if candidate_files:
            pq_path, _, _ = candidate_files[0]
            parquet_paths = [pq_path]
        else:
            parquet_paths = [p for p, _, _ in agent_files]

        window_start = None
        window_end = None

        df = _load_agent_df(agent_id, parquet_paths, window_start, window_end)
        if df is None or len(df) == 0:
            extraction_failures += 1
            continue

        sampled = _sample_fixed_interval(
            df,
            target_points=n_points,
            interval_min=interval_min,
            tolerance_sec=time_tolerance_sec,
        )
        if sampled is None:
            extraction_failures += 1
            continue

        extracted = {
            "agent_id": agent_id,
            "n_points": len(sampled["timestamp"]),
            "longitude_n": sampled["longitude_n"],
            "latitude_n": sampled["latitude_n"],
            "longitude": sampled["longitude"],
            "latitude": sampled["latitude"],
            "timestamp": sampled["timestamp"],
        }
        processed_regular.append(data_processor(extracted))

        if agent_idx % 10 == 0:
            logger.info(
                "Interval %d min | processed %d agents, kept %d",
                interval_min,
                agent_idx,
                len(processed_regular),
            )

    if not processed_regular:
        raise RuntimeError(f"No valid trajectories extracted for interval={interval_min} min")

    reg_path = _save_dataset(
        processed_regular,
        output_dir,
        f"fulltraj{interval_min}min",
        m,
        n_points,
        interval_min,
    )

    logger.info("Interval %d min regular set: %s", interval_min, reg_path)
    logger.info("Interval %d min extraction failures: %d", interval_min, extraction_failures)

    return {
        "interval_min": interval_min,
        "regular_path": reg_path,
        "n_trajectories": len(processed_regular),
        "extraction_failures": extraction_failures,
    }


def extract_10min_traj(
    parquet_dir: str = PARQUET_DIR,
    m: int = M,
    n_points: int = POINTS_N,
    intervals_min: Optional[list[int]] = None,
    time_tolerance_sec: int = TIME_TOLERANCE_SEC,
) -> dict:
    metadata = scan_parquet_metadata(parquet_dir)
    # Guardrail: traj_extractor.scan_parquet_metadata is expected to use TEST SPLIT only
    # (the last 3 parquet files). Keep a runtime check so this script is explicit/safe.
    all_parquet_paths = sorted(Path(parquet_dir).glob("*.parquet"))
    expected_test_files = {str(p) for p in all_parquet_paths[-3:]}
    actual_files = set(metadata.keys())
    if not actual_files:
        raise ValueError("No metadata scanned from parquet files")
    if not actual_files.issubset(expected_test_files):
        raise RuntimeError(
            "Detected non-test parquet files in metadata scan. "
            "Expected only last 3 parquet files (test split). "
            f"Expected subset={sorted(expected_test_files)}, actual={sorted(actual_files)}"
        )
    logger.info(
        "Using TEST SPLIT parquet files (%d): %s",
        len(actual_files),
        ", ".join(Path(p).name for p in sorted(actual_files)),
    )

    all_agents = set()
    for agent_ranges in metadata.values():
        all_agents.update(agent_ranges.keys())
    all_agents = list(all_agents)

    if not all_agents:
        raise ValueError("No agents found in parquet directory")

    target_intervals = intervals_min if intervals_min is not None else list(INTERVALS_MIN)
    if not target_intervals:
        raise ValueError("intervals_min must not be empty")

    interval_results = {}
    for interval_min in target_intervals:
        output_dir = OUTPUT_DIR_TMPL.format(interval=interval_min)
        interval_results[int(interval_min)] = _extract_single_interval(
            metadata=metadata,
            all_agents=all_agents,
            output_dir=output_dir,
            m=m,
            n_points=n_points,
            interval_min=int(interval_min),
            time_tolerance_sec=time_tolerance_sec,
        )

    return {
        "interval_results": interval_results,
        "intervals_min": [int(x) for x in target_intervals],
        "m": int(m),
        "n_points": int(n_points),
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    extract_10min_traj()


if __name__ == "__main__":
    main()
