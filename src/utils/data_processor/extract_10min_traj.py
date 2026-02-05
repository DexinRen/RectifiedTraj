#!/usr/bin/env python3
"""
Standalone extractor for 10-minute-sampled trajectories.

Generates two datasets:
  1) regular 10-min sampling
  2) turbulent sampling (random point drops)

Uses the same core logic as traj_extractor (test split scan, random agents).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import logging
import sys
import numpy as np
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

# ================================================================
# CONFIG
# ================================================================
PARQUET_DIR = "./dataset/raw"
OUTPUT_DIR = "./dataset/processed/full_traj_10min"
OUTPUT_DIR_TURB = "./dataset/processed/full_traj_10min_turb"

M = 1
POINTS_N = 1440
INTERVAL_MIN = 10
DROP_PROB = 0.05

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


def _apply_turbulence(sampled: dict, drop_prob: float, interval_min: int) -> dict:
    ts = sampled["timestamp"]
    n = len(ts)
    if n == 0:
        return sampled

    keep = np.random.rand(n) >= drop_prob
    if keep.sum() < 2:
        keep[:] = True

    ts_kept = ts[keep]
    if ts_kept.size > 1:
        diffs = np.diff(ts_kept)
        if np.all(diffs <= interval_min * 60):
            # force a gap by dropping one interior point
            mid = ts_kept.size // 2
            mask = np.ones(ts_kept.size, dtype=bool)
            if 0 < mid < ts_kept.size - 1:
                mask[mid] = False
            ts_kept = ts_kept[mask]
            keep_idx = np.where(keep)[0]
            keep = np.zeros_like(keep)
            keep[keep_idx[mask]] = True

    out = {
        "timestamp": sampled["timestamp"][keep],
        "longitude_n": sampled["longitude_n"][keep],
        "latitude_n": sampled["latitude_n"][keep],
        "longitude": sampled["longitude"][keep],
        "latitude": sampled["latitude"][keep],
    }
    return out


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

def extract_10min_traj(
    parquet_dir: str = PARQUET_DIR,
    output_dir: str = OUTPUT_DIR,
    output_dir_turb: str = OUTPUT_DIR_TURB,
    m: int = M,
    n_points: int = POINTS_N,
    interval_min: int = INTERVAL_MIN,
    drop_prob: float = DROP_PROB,
    time_tolerance_sec: int = TIME_TOLERANCE_SEC,
) -> dict:
    logger.info(
        "Extracting 10-min trajectories: M=%d, N=%d, interval=%d min",
        m,
        n_points,
        interval_min,
    )

    metadata = scan_parquet_metadata(parquet_dir)
    all_agents = set()
    for agent_ranges in metadata.values():
        all_agents.update(agent_ranges.keys())
    all_agents = list(all_agents)

    if not all_agents:
        raise ValueError("No agents found in parquet directory")

    np.random.shuffle(all_agents)

    processed_regular = []
    processed_turb = []

    agent_idx = 0
    extraction_failures = 0
    for agent_id in all_agents:
        if len(processed_regular) >= m:
            break

        agent_idx += 1
        agent_files = find_agent_files(agent_id, metadata)
        if not agent_files:
            extraction_failures += 1
            continue

        # Prefer a single file that already spans the full duration
        required_span = (n_points - 1) * interval_min * 60
        candidate_files = [(p, start, end) for p, start, end in agent_files if (end - start) >= required_span]
        if candidate_files:
            pq_path, _, _ = candidate_files[0]
            parquet_paths = [pq_path]
        else:
            # Fallback: use all files for that agent (slower)
            parquet_paths = [p for p, _, _ in agent_files]

        # Use full file(s). Do not window, to allow sampling the next available point.
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

        turb = _apply_turbulence(sampled, drop_prob, interval_min)
        extracted_turb = {
            "agent_id": agent_id,
            "n_points": len(turb["timestamp"]),
            "longitude_n": turb["longitude_n"],
            "latitude_n": turb["latitude_n"],
            "longitude": turb["longitude"],
            "latitude": turb["latitude"],
            "timestamp": turb["timestamp"],
        }
        processed_turb.append(data_processor(extracted_turb))

        if agent_idx % 10 == 0:
            logger.info("Processed %d agents, kept %d", agent_idx, len(processed_regular))

    if not processed_regular:
        raise RuntimeError("No valid trajectories extracted")

    reg_path = _save_dataset(processed_regular, output_dir, "fulltraj10min", m, n_points, interval_min)
    turb_path = _save_dataset(processed_turb, output_dir_turb, "fulltraj10min_turb", m, n_points, interval_min)

    logger.info("Saved regular set: %s", reg_path)
    logger.info("Saved turbulent set: %s", turb_path)
    logger.info("Extraction failures: %d", extraction_failures)

    return {
        "regular_path": reg_path,
        "turb_path": turb_path,
        "n_trajectories": len(processed_regular),
        "extraction_failures": extraction_failures,
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
