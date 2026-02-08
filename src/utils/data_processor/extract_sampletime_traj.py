#!/usr/bin/env python3
"""
Extractor for trajectories sampled by a learned/interpolated sample-time distribution.

Selection rule per step:
1) draw one gap (seconds) from distribution
2) find the earliest next point whose timestamp is at least that gap from
   the previous selected point
3) append it and continue
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
from src.utils.data_processor.sample_time_distribution import (
    build_default_sample_time_generator,
    summarize_samples,
)


logger = logging.getLogger(__name__)


PARQUET_DIR = "./dataset/raw"
OUTPUT_DIR = "./dataset/processed/full_traj_sampletime"

M = 100
POINTS_N = 5000

# Target distribution stats (seconds)
TARGET_MEAN_SEC = 867.618530
TARGET_MEDIAN_SEC = 128.0
TARGET_STD_SEC = 4773.800293


def _to_unix_seconds(ts_col: np.ndarray) -> np.ndarray:
    if np.issubdtype(ts_col.dtype, np.datetime64):
        return ts_col.astype("datetime64[s]").astype(np.float64)
    if ts_col.dtype == object:
        return ts_col.astype("datetime64[s]").astype(np.float64)
    return ts_col.astype(np.float64)


def _load_agent_df(
    agent_id: int,
    parquet_paths: list[str],
    column_map: Optional[dict] = None,
) -> Optional[pl.DataFrame]:
    # Keep same filtering pattern used in parquet processing paths:
    # agent filter + null/finite checks + sorted by timestamp.
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

        scan = (
            pl.scan_parquet(path)
            .filter(
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
            .select([ts_col, lon_n_col, lat_n_col, lon_col, lat_col])
            .sort(ts_col)
        )

        try:
            df = scan.collect()
        except Exception as exc:
            logger.warning("Skipping unreadable parquet %s: %s", path.name, exc)
            continue

        if len(df) > 0:
            frames.append(df)

    if not frames:
        return None

    return pl.concat(frames).sort(ts_col)


def _sample_with_distribution(
    df: pl.DataFrame,
    target_points: int,
    *,
    rng: np.random.Generator,
    sample_gap_fn,
    max_start_tries: int = 64,
    retries_per_start: int = 3,
    min_gap_seconds: float = 1.0,
    max_gap_seconds: float | None = None,
) -> Optional[dict]:
    ts_col, lon_n_col, lat_n_col, lon_col, lat_col = df.columns[:5]
    ts = _to_unix_seconds(df[ts_col].to_numpy())
    if ts.size < target_points:
        return None

    n = ts.size
    max_start_idx = n - target_points
    if max_start_idx <= 0:
        start_candidates = np.array([0], dtype=np.int64)
    else:
        count = min(max_start_tries, max_start_idx + 1)
        start_candidates = rng.choice(np.arange(max_start_idx + 1), size=count, replace=False)

    for start_idx in start_candidates:
        for _ in range(retries_per_start):
            idxs = [int(start_idx)]
            ok = True

            sampled_gaps = np.asarray(
                sample_gap_fn(target_points - 1, min_seconds=min_gap_seconds, round_to_int=False),
                dtype=np.float64,
            )
            if max_gap_seconds is not None:
                sampled_gaps = np.minimum(sampled_gaps, float(max_gap_seconds))

            for gap in sampled_gaps:
                prev_idx = idxs[-1]
                target_time = ts[prev_idx] + gap
                nxt_idx = int(np.searchsorted(ts, target_time, side="left"))
                if nxt_idx <= prev_idx:
                    nxt_idx = prev_idx + 1
                if nxt_idx >= n:
                    ok = False
                    break
                idxs.append(nxt_idx)

            if not ok or len(idxs) != target_points:
                continue

            idxs_arr = np.asarray(idxs, dtype=np.int64)
            sampled = df[idxs_arr]
            sampled_ts = ts[idxs_arr]
            return {
                "timestamp": sampled_ts,
                "longitude_n": sampled[lon_n_col].to_numpy(),
                "latitude_n": sampled[lat_n_col].to_numpy(),
                "longitude": sampled[lon_col].to_numpy(),
                "latitude": sampled[lat_col].to_numpy(),
                "intervals_sec": np.diff(sampled_ts),
            }

    return None


def _save_dataset(
    processed_trajs: list[dict],
    output_dir: str,
    prefix: str,
    target_m: int,
    n_points: int,
    interval_stats: dict,
    fit_info: dict,
) -> str:
    n_trajectories = len(processed_trajs)
    lengths = [t["n_points"] for t in processed_trajs]
    median_length = int(np.median(lengths)) if lengths else 0
    total_points = int(sum(lengths))

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
            "target_N": n_points,
            "sample_interval_stats_sec": interval_stats,
            "fit": fit_info,
        },
    }

    torch.save(save_data, output_file)
    return str(output_file)


def extract_sampletime_traj(
    parquet_dir: str = PARQUET_DIR,
    output_dir: str = OUTPUT_DIR,
    m: int = M,
    n_points: int = POINTS_N,
    target_mean_sec: float = TARGET_MEAN_SEC,
    target_median_sec: float = TARGET_MEDIAN_SEC,
    target_std_sec: float = TARGET_STD_SEC,
    fit_seed: int = 42,
    sample_seed: int = 7,
    shuffle_seed: int = 101,
    fit_iter: int = 20000,
    max_gap_seconds: float | None = None,
) -> dict:
    logger.info(
        "Extracting distribution-sampled trajectories: M=%d N=%d target(mean=%.3f med=%.3f std=%.3f)",
        m,
        n_points,
        target_mean_sec,
        target_median_sec,
        target_std_sec,
    )

    fit, generator = build_default_sample_time_generator(
        target_mean=target_mean_sec,
        target_median=target_median_sec,
        target_std=target_std_sec,
        fit_seed=fit_seed,
        sample_seed=sample_seed,
        n_iter=fit_iter,
    )

    metadata = scan_parquet_metadata(parquet_dir)
    all_agents = set()
    for agent_ranges in metadata.values():
        all_agents.update(agent_ranges.keys())
    all_agents = list(all_agents)
    if not all_agents:
        raise ValueError("No agents found in parquet directory")

    rng = np.random.default_rng(shuffle_seed)
    rng.shuffle(all_agents)

    processed = []
    all_intervals = []
    extraction_failures = 0
    sampled_agents = 0

    for agent_id in all_agents:
        if len(processed) >= m:
            break

        sampled_agents += 1
        agent_files = find_agent_files(agent_id, metadata)
        if not agent_files:
            extraction_failures += 1
            continue

        parquet_paths = [p for p, _, _ in agent_files]
        df = _load_agent_df(agent_id, parquet_paths)
        if df is None or len(df) == 0:
            extraction_failures += 1
            continue

        sampled = _sample_with_distribution(
            df,
            target_points=n_points,
            rng=rng,
            sample_gap_fn=generator.sample,
            min_gap_seconds=1.0,
            max_gap_seconds=max_gap_seconds,
        )
        if sampled is None:
            extraction_failures += 1
            continue

        extracted = {
            "agent_id": int(agent_id),
            "n_points": int(len(sampled["timestamp"])),
            "longitude_n": sampled["longitude_n"],
            "latitude_n": sampled["latitude_n"],
            "longitude": sampled["longitude"],
            "latitude": sampled["latitude"],
            "timestamp": sampled["timestamp"],
        }
        processed.append(data_processor(extracted))
        all_intervals.append(sampled["intervals_sec"])

        if sampled_agents % 10 == 0:
            logger.info("Processed agents=%d kept=%d/%d", sampled_agents, len(processed), m)

    if not processed:
        raise RuntimeError("No valid trajectories extracted")

    intervals_flat = np.concatenate(all_intervals, axis=0) if all_intervals else np.array([], dtype=np.float64)
    interval_stats = summarize_samples(intervals_flat)
    fit_info = fit.as_dict()

    output_path = _save_dataset(
        processed_trajs=processed,
        output_dir=output_dir,
        prefix="fulltraj_sampletime",
        target_m=m,
        n_points=n_points,
        interval_stats=interval_stats,
        fit_info=fit_info,
    )

    logger.info("Saved: %s", output_path)
    logger.info(
        "Actual interval stats (sec): mean=%.3f median=%.3f std=%.3f",
        interval_stats["mean"],
        interval_stats["median"],
        interval_stats["std"],
    )

    return {
        "output_path": output_path,
        "n_trajectories": len(processed),
        "extraction_failures": extraction_failures,
        "agents_sampled": sampled_agents,
        "interval_stats_sec": interval_stats,
        "fit": fit_info,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    out = extract_sampletime_traj()
    logger.info("Done: %s", out)


if __name__ == "__main__":
    main()
