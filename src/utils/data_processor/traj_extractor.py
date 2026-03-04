#!/usr/bin/env python3
"""
Extract M full trajectories of target length N from parquet directory.

Handles multi-file trajectory stitching with time continuity checks.
Processes and saves trajectories for BF/DF testing.
"""

import os
import gc
import sys
import threading
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Callable
import logging
from datetime import datetime, timedelta

# Reserve CPU for system responsiveness: by default use at most (total_cores - 2).
_CPU_TOTAL = os.cpu_count() or 1
_CPU_BUDGET = max(1, int(_CPU_TOTAL) - 2)
_THREAD_BUDGET_ENV_VARS = (
    "POLARS_MAX_THREADS",
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _var in _THREAD_BUDGET_ENV_VARS:
    os.environ.setdefault(_var, str(_CPU_BUDGET))

import polars as pl
import numpy as np
import torch


logger = logging.getLogger(__name__)
_THREAD_BUDGET_LOGGED = False
TEST_SPLIT_PARQUET_COUNT = 1


def _log_thread_budget(context: str) -> None:
    global _THREAD_BUDGET_LOGGED
    if _THREAD_BUDGET_LOGGED:
        return
    env_view = {k: os.environ.get(k) for k in _THREAD_BUDGET_ENV_VARS}
    try:
        torch_threads = int(torch.get_num_threads())
    except Exception:
        torch_threads = -1
    try:
        interop_threads = int(torch.get_num_interop_threads())
    except Exception:
        interop_threads = -1
    msg = (
        f"[{context}] Thread budget active: "
        f"cpu_total={int(_CPU_TOTAL)} "
        f"cpu_budget={int(_CPU_BUDGET)} "
        f"torch_threads={int(torch_threads)} "
        f"torch_interop_threads={int(interop_threads)} "
        f"env={env_view} "
        "torch_runtime_thread_mutation=disabled"
    )
    logger.info(msg)
    print(msg, flush=True)
    _THREAD_BUDGET_LOGGED = True


def _progress_print(msg: str) -> None:
    with _STDOUT_LOCK:
        print(msg, flush=True)


_STDOUT_LOCK = threading.Lock()


def _bar(done: int, total: int, width: int = 32) -> str:
    t = max(1, int(total))
    d = max(0, min(int(done), t))
    filled = int(round((d / t) * int(width)))
    return "[" + ("#" * filled) + ("-" * (int(width) - filled)) + "]"


class _LiveThreeLineProgress:
    def __init__(self, sampler_name: str, target_m: int, total_agents: int, stall_limit: int) -> None:
        self.sampler_name = str(sampler_name)
        self.target_m = int(target_m)
        self.total_agents = int(total_agents)
        self.stall_limit = int(stall_limit)
        self._started = False

    def update(
        self,
        *,
        queue_size: int,
        agents_scanned: int,
        stalled_users: int,
        min_len: int,
        max_len: int,
        last_enqueued_len: int,
        avg_sample_sec: float,
    ) -> None:
        q = int(queue_size)
        a = int(agents_scanned)
        stalled = int(max(0, stalled_users))
        stalled_cap = max(1, int(self.stall_limit))
        stalled_for_bar = min(stalled, stalled_cap)
        stalled_remain = max(0, int(self.stall_limit) - stalled)
        min_l = int(min_len)
        max_l = int(max_len)
        last_l = int(last_enqueued_len)
        avg_s = float(avg_sample_sec)

        line1 = (
            f"[traj:{self.sampler_name}] queue     {_bar(q, self.target_m)} "
            f"{q}/{int(self.target_m)}"
        )
        line2 = (
            f"[traj:{self.sampler_name}] halt cnt  {_bar(stalled_for_bar, stalled_cap)} "
            f"{stalled}/{int(self.stall_limit)} (remain={stalled_remain})"
        )
        line3 = (
            f"[traj:{self.sampler_name}] agent: {a}/{int(self.total_agents)} | "
            f"traj len (max: {max_l}  min:{min_l}  last:{last_l}) | "
            f"avg_sample_time={avg_s:.2f}s"
        )

        with _STDOUT_LOCK:
            if not self._started:
                block = line1 + "\n" + line2 + "\n" + line3 + "\n"
                self._started = True
            else:
                # Move cursor up 3 lines and overwrite all in one atomic write.
                block = (
                    "\x1b[3A"
                    "\r\x1b[2K" + line1 + "\n"
                    "\r\x1b[2K" + line2 + "\n"
                    "\r\x1b[2K" + line3 + "\n"
                )
            sys.stdout.write(block)
            sys.stdout.flush()


class _LiveEstimateProgress:
    def __init__(self, total_agents: int, target_agents: int) -> None:
        self.total_agents = int(max(1, total_agents))
        self.target_agents = int(max(1, target_agents))
        self._started = False

    def update(
        self,
        *,
        scanned: int,
        sampled: int,
        parse_fallback_agents: int,
    ) -> None:
        scanned_i = int(max(0, scanned))
        sampled_i = int(max(0, sampled))
        fallback_i = int(max(0, parse_fallback_agents))
        line1 = (
            f"[traj:sampletime] scanned  {_bar(scanned_i, self.total_agents)} "
            f"{scanned_i}/{self.total_agents}"
        )
        line2 = (
            f"[traj:sampletime] sampled  {_bar(sampled_i, self.target_agents)} "
            f"{sampled_i}/{self.target_agents}"
        )
        line3 = f"[traj:sampletime] parse_fallback_agents={fallback_i}"

        with _STDOUT_LOCK:
            if not self._started:
                block = line1 + "\n" + line2 + "\n" + line3 + "\n"
                self._started = True
            else:
                block = (
                    "\x1b[3A"
                    "\r\x1b[2K" + line1 + "\n"
                    "\r\x1b[2K" + line2 + "\n"
                    "\r\x1b[2K" + line3 + "\n"
                )
            sys.stdout.write(block)
            sys.stdout.flush()

# Fixed trajectory extraction policy (no longer configurable through M/N).
FIXED_TRAJ_COUNT = 200
FIXED_TRAJ_POINTS = 5000
MAX_TRAJ_PER_AGENT = 3
MAX_STALLED_USERS_AFTER_POOL_FULL = 50
MAX_REALIZED_GAP_MULTIPLIER = 8.0
MAX_AVG_INTERVAL_MULTIPLIER = 2.0
BLOGWATCH_TARGET_MEAN_SEC = 867.618530
BLOGWATCH_TARGET_MEDIAN_SEC = 128.0
BLOGWATCH_TARGET_STD_SEC = 4773.800293

DEFAULT_COLUMN_MAP = {
    "agent": "agent",
    "timestamp": "timestamp",
    "longitude_n": "longitude_n",
    "latitude_n": "latitude_n",
    "longitude": "longitude",
    "latitude": "latitude",
    "error_range": "error_range",
}

UTOKYO_COLUMN_MAP = {
    "agent": "uuid",
    "timestamp": "datetime",
    "longitude_n": "longitude_noisy",
    "latitude_n": "latitude_noisy",
    "longitude": "longitude_anonymous",
    "latitude": "latitude_anonymous",
    "error_range": "accuracy",
}


def _assert_not_calibration_or_processed_source(parquet_dir: str) -> None:
    """
    Guardrail: trajectory extraction must read raw parquet only.
    """
    path = Path(parquet_dir).resolve()
    parts = {p.lower() for p in path.parts}
    if "processed" in parts:
        raise ValueError(
            f"Trajectory extractor expects raw parquet directory, got processed path: {path}"
        )
    if "calibration" in parts:
        raise ValueError(
            f"Trajectory extractor must not read calibration split path: {path}"
        )


def _detect_column_map(parquet_dir: str, column_map: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    if column_map is not None:
        return column_map

    parquet_paths = sorted(Path(parquet_dir).glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

    sample_path = parquet_paths[0]
    sample_cols = set(pl.read_parquet(sample_path, n_rows=1).columns)

    if {"uuid", "datetime", "latitude_noisy", "longitude_noisy", "latitude_anonymous", "longitude_anonymous"}.issubset(sample_cols):
        return UTOKYO_COLUMN_MAP
    if {"agent", "timestamp", "latitude_n", "longitude_n", "latitude", "longitude"}.issubset(sample_cols):
        return DEFAULT_COLUMN_MAP

    raise ValueError(
        "Unable to detect parquet schema. Provide column_map with keys: "
        "agent, timestamp, longitude_n, latitude_n, longitude, latitude, error_range."
    )


def _timestamp_expr(ts_col: str) -> pl.Expr:
    # Align with mixed timestamp sources (naive/tz-aware strings or datetime columns).
    # Prefer robust parsing to avoid Polars errors on timezone-bearing strings.
    return pl.coalesce(
        pl.col(ts_col).cast(pl.Datetime(time_zone="UTC"), strict=False),
        pl.col(ts_col).cast(pl.Utf8).str.to_datetime(strict=False, time_zone="UTC"),
    ).alias("_timestamp")


def _to_unix_seconds_scalar(value: Any) -> Optional[float]:
    """Convert scalar timestamp-like values to Unix seconds."""
    if value is None:
        return None
    if hasattr(value, "timestamp"):
        try:
            return float(value.timestamp())
        except Exception:
            pass
    if isinstance(value, np.datetime64):
        try:
            return float(value.astype("datetime64[s]").astype(np.int64))
        except Exception:
            pass
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        if not np.isfinite(v):
            return None
        av = abs(v)
        # Heuristic normalization for common epoch units.
        if av > 1e14:
            return v / 1e9  # ns -> s
        if av > 1e11:
            return v / 1e3  # ms -> s
        return v

    text = str(value).strip()
    if not text:
        return None
    text_norm = f"{text[:-1]}+00:00" if text.endswith("Z") else text
    try:
        return float(datetime.fromisoformat(text_norm).timestamp())
    except Exception:
        pass
    try:
        return float(np.datetime64(text).astype("datetime64[s]").astype(np.int64))
    except Exception:
        return None


def _slice_raw_trajectory(
    raw_traj: dict,
    start: int,
    end: int,
    include_error_range: bool = False,
) -> dict:
    out = {
        "agent_id": raw_traj["agent_id"],
        "n_points": int(end - start),
        "longitude_n": raw_traj["longitude_n"][start:end],
        "latitude_n": raw_traj["latitude_n"][start:end],
        "longitude": raw_traj["longitude"][start:end],
        "latitude": raw_traj["latitude"][start:end],
        "timestamp": raw_traj["timestamp"][start:end],
    }
    if include_error_range:
        out["error_range"] = raw_traj["error_range"][start:end]
    return out


def data_processor(extracted_traj: dict) -> dict:
    """
    Purpose:
        Convert raw GPS trajectory to encoder-decoder format.
        Strips timestamp, returns (noisy, clean) pair.
    
    Parameters:
        extracted_traj (dict): {
            "agent_id": int,
            "n_points": int,
            "longitude_n": np.ndarray,  # (N,)
            "latitude_n": np.ndarray,   # (N,)
            "longitude": np.ndarray,    # (N,)
            "latitude": np.ndarray,     # (N,)
            "timestamp": np.ndarray     # (N,)
        }
    
    Return:
        processed_traj (dict): {
            "agent_id": int,
            "n_points": int,
            "data": np.ndarray,   # (N, 2) [longitude_n, latitude_n]
            "label": np.ndarray   # (N, 2) [longitude, latitude]
        }
    
    Notes:
        - Time information removed for encoder-decoder input
        - CRITICAL: [longitude, latitude] order matches training data and encoder-decoder
        - Model was trained with X1 = [longitude_n, latitude_n, timestamp, is_start]
    """
    
    data = np.stack([
        extracted_traj['longitude_n'],
        extracted_traj['latitude_n']
    ], axis=1)
    
    label = np.stack([
        extracted_traj['longitude'],
        extracted_traj['latitude']
    ], axis=1)
    
    return {
        "agent_id": extracted_traj['agent_id'],
        "n_points": extracted_traj['n_points'],
        "data": data,
        "label": label
    }


def data_processor_with_error_range(extracted_traj: dict) -> dict:
    """
    Purpose:
        Convert raw GPS trajectory to encoder-decoder format and keep error_range.

    Parameters:
        extracted_traj (dict): {
            "agent_id": int,
            "n_points": int,
            "longitude_n": np.ndarray,
            "latitude_n": np.ndarray,
            "longitude": np.ndarray,
            "latitude": np.ndarray,
            "timestamp": np.ndarray,
            "error_range": np.ndarray
        }

    Return:
        processed_traj (dict): {
            "agent_id": int,
            "n_points": int,
            "data": np.ndarray,        # (N, 2) noisy [lon_n, lat_n]
            "label": np.ndarray,       # (N, 2) reference center [lon, lat]
            "error_range": np.ndarray, # (N,)   per-point error radius
            "timestamp": np.ndarray    # (N,)   Unix timestamp (float)
        }
    """
    data = np.stack([
        extracted_traj['longitude_n'],
        extracted_traj['latitude_n']
    ], axis=1)

    label = np.stack([
        extracted_traj['longitude'],
        extracted_traj['latitude']
    ], axis=1)

    return {
        "agent_id": extracted_traj['agent_id'],
        "n_points": extracted_traj['n_points'],
        "data": data,
        "label": label,
        "error_range": extracted_traj['error_range'],
        "timestamp": extracted_traj['timestamp'],
    }


def scan_parquet_metadata(
    parquet_dir: str,
    column_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[Any, Tuple[float, float]]]:
    """
    Purpose:
        Scan TEST SPLIT parquet files and build agent time range index.
        Returns mapping: {parquet_path: {agent_id: (start_time, end_time)}}
    
    Parameters:
        parquet_dir (str): Directory containing parquet files
    
    Return:
        metadata (dict): {
            parquet_path (str): {
                agent_id (int): (start_timestamp (float), end_timestamp (float)),
                ...
            },
            ...
        }
    
    Notes:
        - Only uses TEST SPLIT (last file when sorted), matching parquet_processor_test_only
        - Fast scanning: only queries unique agent IDs + min/max timestamp per agent
        - Skips corrupted files
    """
    
    logger.debug(f"Scanning parquet directory: {parquet_dir}")
    _progress_print(f"[traj:metadata] scan start parquet_dir={parquet_dir}")
    _assert_not_calibration_or_processed_source(parquet_dir)
    
    column_map = _detect_column_map(parquet_dir, column_map)
    agent_col = column_map["agent"]
    ts_col = column_map["timestamp"]

    # Get sorted parquet files and select TEST SPLIT ONLY (last file).
    all_parquet_paths = sorted(Path(parquet_dir).glob("*.parquet"))
    logger.debug(f"Found {len(all_parquet_paths)} total parquet files")

    test_files = all_parquet_paths[-int(TEST_SPLIT_PARQUET_COUNT):]
    for p in test_files:
        p_parts = {x.lower() for x in p.resolve().parts}
        if "calibration" in p_parts:
            raise RuntimeError(f"Refusing to scan calibration parquet file: {p}")
    logger.debug(f"Using TEST SPLIT: last {len(test_files)} file(s)")
    file_names = ", ".join(p.name for p in test_files)
    _progress_print(f"[traj:metadata] using test split files={len(test_files)}: {file_names}")
    
    metadata = {}
    
    total_files = int(len(test_files))
    for file_idx, pq_path in enumerate(test_files, start=1):
        logger.debug(f"Scanning {pq_path.name}...")
        _progress_print(
            f"[traj:metadata] file {int(file_idx)}/{int(total_files)} start: {pq_path.name}"
        )
        
        try:
            # Low-memory metadata scan:
            # - read only agent + timestamp columns
            # - avoid expensive per-row datetime parsing in Polars
            _progress_print(f"[traj:metadata] file {pq_path.name}: collect start")
            lazy_meta = (
                pl.scan_parquet(pq_path)
                .select([agent_col, ts_col])
                .filter(
                    pl.col(agent_col).is_not_null() & pl.col(ts_col).is_not_null()
                )
                .group_by(agent_col)
                .agg([
                    pl.col(ts_col).min().alias("start_time"),
                    pl.col(ts_col).max().alias("end_time"),
                ])
            )
            try:
                df = lazy_meta.collect(streaming=True)
            except TypeError:
                # Older Polars versions may not support the streaming flag.
                df = lazy_meta.collect()
            except Exception as stream_exc:
                _progress_print(
                    f"[traj:metadata] file {pq_path.name}: streaming collect failed ({stream_exc}); retry non-streaming"
                )
                df = lazy_meta.collect()
            _progress_print(f"[traj:metadata] file {pq_path.name}: collect done groups={int(len(df))}")
            
            if len(df) == 0:
                logger.warning(f"No valid data in {pq_path.name}")
                _progress_print(f"[traj:metadata] file {pq_path.name}: no valid rows")
                continue
            
            # Build agent_ranges dict
            agent_ranges = {}
            skipped_bad_ts = 0
            for row in df.iter_rows(named=True):
                agent_id = row[agent_col]
                start_ts = _to_unix_seconds_scalar(row["start_time"])
                end_ts = _to_unix_seconds_scalar(row["end_time"])
                if start_ts is None or end_ts is None:
                    skipped_bad_ts += 1
                    continue
                if end_ts < start_ts:
                    start_ts, end_ts = end_ts, start_ts
                agent_ranges[agent_id] = (float(start_ts), float(end_ts))
            if skipped_bad_ts > 0:
                _progress_print(
                    f"[traj:metadata] file {pq_path.name}: skipped_bad_timestamp_agents={int(skipped_bad_ts)}"
                )
            if not agent_ranges:
                _progress_print(
                    f"[traj:metadata] file {pq_path.name}: no parsable timestamp ranges"
                )
                continue
            
            metadata[str(pq_path)] = agent_ranges
            logger.debug(f"Found {len(agent_ranges)} agents in {pq_path.name}")
            _progress_print(
                f"[traj:metadata] file {pq_path.name}: indexed_agents={int(len(agent_ranges))}"
            )
            
        except Exception as e:
            logger.error(f"Failed to scan {pq_path.name}: {e}")
            logger.warning(f"Skipping corrupted file: {pq_path.name}")
            _progress_print(f"[traj:metadata] file {pq_path.name}: failed ({e})")
            continue

    unique_agents = set()
    for agent_ranges in metadata.values():
        unique_agents.update(agent_ranges.keys())
    _progress_print(
        f"[traj:metadata] scan complete files={int(len(metadata))}/{int(total_files)} "
        f"unique_agents={int(len(unique_agents))}"
    )
    return metadata


def find_agent_files(agent_id: Any, metadata: dict) -> List[Tuple[str, float, float]]:
    """
    Purpose:
        Find all parquet files containing given agent, sorted by start time.
    
    Parameters:
        agent_id (int): Target agent ID
        metadata (dict): Output from scan_parquet_metadata
    
    Return:
        file_list (list): [
            (parquet_path (str), start_time (float), end_time (float)),
            ...
        ]
        Sorted by start_time ascending.
    
    Notes:
        - Returns empty list if agent not found in any file
    """
    
    agent_files = []
    
    for pq_path, agent_ranges in metadata.items():
        if agent_id in agent_ranges:
            start_time, end_time = agent_ranges[agent_id]
            agent_files.append((pq_path, start_time, end_time))
    
    # Sort by start time
    agent_files.sort(key=lambda x: x[1])
    
    return agent_files


def build_agent_file_index(metadata: dict) -> Dict[Any, List[Tuple[str, float, float]]]:
    """
    Build once, reuse many times:
      agent_id -> [(parquet_path, start_time, end_time), ...] sorted by start_time.
    """
    agent_to_files: Dict[Any, List[Tuple[str, float, float]]] = {}

    for pq_path, agent_ranges in metadata.items():
        for agent_id, (start_time, end_time) in agent_ranges.items():
            agent_to_files.setdefault(agent_id, []).append((pq_path, start_time, end_time))

    for agent_id in agent_to_files:
        agent_to_files[agent_id].sort(key=lambda x: x[1])

    return agent_to_files


def build_agent_entry_count_index(
    parquet_dir: str,
    *,
    column_map: Optional[Dict[str, str]] = None,
    metadata: Optional[dict] = None,
) -> Dict[Any, int]:
    """
    Count rows per agent on the same TEST-SPLIT files used by metadata scan.
    """
    column_map = _detect_column_map(parquet_dir, column_map)
    agent_col = column_map["agent"]
    if metadata is None:
        metadata = scan_parquet_metadata(parquet_dir, column_map=column_map)
    test_paths = [Path(p) for p in metadata.keys()]
    if not test_paths:
        return {}

    counts: Dict[Any, int] = {}
    for pq_path in test_paths:
        try:
            lazy = (
                pl.scan_parquet(str(pq_path))
                .select([agent_col])
                .filter(pl.col(agent_col).is_not_null())
                .group_by(agent_col)
                .agg(pl.len().alias("n"))
            )
            try:
                df = lazy.collect(streaming=True)
            except TypeError:
                df = lazy.collect()
        except Exception as exc:
            logger.warning("Agent entry count scan failed for %s: %s", pq_path.name, exc)
            continue

        for row in df.iter_rows(named=True):
            a = row[agent_col]
            n = int(row["n"])
            counts[a] = int(counts.get(a, 0)) + int(n)
    return counts


def build_traj_extraction_context(
    parquet_dir: str,
    *,
    column_map: Optional[Dict[str, str]] = None,
    shuffle_seed: int = 101,
    sort_users_by_entries: bool = True,
) -> dict:
    """
    One-time context for all trajectory generation calls in a run.
    """
    col_map = _detect_column_map(parquet_dir, column_map)
    metadata = scan_parquet_metadata(parquet_dir, column_map=col_map)
    agent_file_index = build_agent_file_index(metadata)
    agents = list(agent_file_index.keys())
    if not agents:
        raise ValueError("No agents found while building extraction context")

    agent_counts: Dict[Any, int] = {}
    if sort_users_by_entries:
        agent_counts = build_agent_entry_count_index(
            parquet_dir,
            column_map=col_map,
            metadata=metadata,
        )
        # Descending by count; deterministic tie-break by stringified id.
        agents = sorted(
            agents,
            key=lambda a: (-int(agent_counts.get(a, 0)), str(a)),
        )
    else:
        rng = np.random.default_rng(int(shuffle_seed))
        rng.shuffle(agents)

    return {
        "column_map": col_map,
        "metadata": metadata,
        "agent_file_index": agent_file_index,
        "ordered_agents": agents,
        "agent_entry_counts": agent_counts,
        "sort_users_by_entries": bool(sort_users_by_entries),
    }


def _load_agent_timestamps_for_sampletime(
    agent_id: Any,
    parquet_paths: List[str],
    *,
    max_points: int,
    column_map: Optional[Dict[str, str]] = None,
    verbose_progress: bool = True,
    stats_out: Optional[dict] = None,
) -> np.ndarray:
    """
    Lightweight timestamp-only loader for sample-time estimation.
    Limits rows aggressively to avoid high-memory scans.
    """
    if max_points < 2 or not parquet_paths:
        return np.array([], dtype=np.float64)

    column_map = _detect_column_map(str(Path(parquet_paths[0]).parent), column_map)
    agent_col = column_map["agent"]
    ts_col = column_map["timestamp"]

    valid_paths = [str(Path(p)) for p in parquet_paths if Path(p).exists()]
    if not valid_paths:
        return np.array([], dtype=np.float64)

    batches: List[np.ndarray] = []
    total = 0
    parse_fallback_count = 0
    parse_fail_count = 0
    for pq_path in valid_paths:
        remaining = int(max_points) - int(total)
        if remaining <= 0:
            break
        try:
            lazy = (
                pl.scan_parquet(pq_path)
                .select([agent_col, ts_col])
                .filter((pl.col(agent_col) == agent_id) & pl.col(ts_col).is_not_null())
                .with_columns(_timestamp_expr(ts_col))
                .filter(pl.col("_timestamp").is_not_null())
                .select(["_timestamp"])
                .limit(int(remaining))
            )
            try:
                df = lazy.collect(streaming=True)
            except TypeError:
                df = lazy.collect()
        except Exception:
            # Fallback for mixed/unexpected timestamp strings that Polars cannot parse.
            parse_fallback_count += 1
            try:
                raw_lazy = (
                    pl.scan_parquet(pq_path)
                    .select([agent_col, ts_col])
                    .filter((pl.col(agent_col) == agent_id) & pl.col(ts_col).is_not_null())
                    .select([ts_col])
                    .limit(int(remaining))
                )
                try:
                    raw_df = raw_lazy.collect(streaming=True)
                except TypeError:
                    raw_df = raw_lazy.collect()
            except Exception:
                parse_fail_count += 1
                continue

            if len(raw_df) == 0:
                continue
            raw_vals = raw_df[ts_col].to_list()
            parsed = np.fromiter(
                (
                    v
                    for v in (_to_unix_seconds_scalar(x) for x in raw_vals)
                    if v is not None and np.isfinite(v)
                ),
                dtype=np.float64,
            )
            if parsed.size == 0:
                parse_fail_count += 1
                continue
            batches.append(parsed)
            total += int(parsed.size)
            continue
        if len(df) == 0:
            continue
        ts = df["_timestamp"].to_numpy().astype("datetime64[s]").astype(np.float64)
        if ts.size == 0:
            continue
        batches.append(ts)
        total += int(ts.size)

    if not batches:
        return np.array([], dtype=np.float64)

    merged = np.concatenate(batches, axis=0)
    if parse_fallback_count > 0:
        if isinstance(stats_out, dict):
            stats_out["parse_fallback_agents"] = int(stats_out.get("parse_fallback_agents", 0)) + 1
            stats_out["parse_fallback_uses"] = int(stats_out.get("parse_fallback_uses", 0)) + int(parse_fallback_count)
            stats_out["parse_fallback_failures"] = int(stats_out.get("parse_fallback_failures", 0)) + int(parse_fail_count)
        if verbose_progress:
            _progress_print(
                f"[traj:sampletime] agent={agent_id} parse_fallback_used={int(parse_fallback_count)} "
                f"fallback_failed={int(parse_fail_count)}"
            )
    if merged.size > int(max_points):
        merged = merged[: int(max_points)]
    # Ensure monotonicity for stable dt statistics.
    if merged.size >= 2:
        merged = np.sort(merged, kind="stable")
    return merged


def estimate_dataset_sample_time_seconds(
    parquet_dir: str,
    *,
    max_agents: int = 64,
    points_per_agent: int = 2000,
    seed: int = 42,
    column_map: Optional[Dict[str, str]] = None,
    precomputed_metadata: Optional[dict] = None,
    precomputed_agent_file_index: Optional[Dict[Any, List[Tuple[str, float, float]]]] = None,
    ordered_agents: Optional[List[Any]] = None,
) -> dict:
    """
    Estimate dataset-native sample-time statistics (seconds) on TEST SPLIT.

    Returns:
        {
            "sampled_agents": int,
            "n_intervals": int,
            "mean_sec": float,
            "median_sec": float,
            "std_sec": float,
        }
    """
    if max_agents <= 0:
        raise ValueError("max_agents must be > 0")
    if points_per_agent < 2:
        raise ValueError("points_per_agent must be >= 2")

    column_map = _detect_column_map(parquet_dir, column_map)
    metadata = precomputed_metadata if precomputed_metadata is not None else scan_parquet_metadata(
        parquet_dir,
        column_map=column_map,
    )
    agent_file_index = (
        precomputed_agent_file_index
        if precomputed_agent_file_index is not None
        else build_agent_file_index(metadata)
    )
    if ordered_agents is not None:
        all_agents = [a for a in ordered_agents if a in agent_file_index]
    else:
        all_agents = list(agent_file_index.keys())
    if not all_agents:
        raise ValueError("No agents found while estimating dataset sample time")

    rng = np.random.default_rng(seed)
    rng.shuffle(all_agents)
    total_candidates = int(len(all_agents))
    log_every = _progress_log_interval(total_candidates, target_logs=10)
    msg = (
        "Estimating sample-time stats: "
        f"candidates={int(total_candidates)} target_agents={int(max_agents)} "
        f"points_per_agent={int(points_per_agent)}"
    )
    logger.info(msg)
    _progress_print(msg)
    live = _LiveEstimateProgress(total_agents=total_candidates, target_agents=int(max_agents))
    sampletime_stats = {
        "parse_fallback_agents": 0,
        "parse_fallback_uses": 0,
        "parse_fallback_failures": 0,
    }
    live.update(scanned=0, sampled=0, parse_fallback_agents=0)

    interval_batches = []
    sampled_agents = 0
    for idx, agent_id in enumerate(all_agents, start=1):
        if sampled_agents >= int(max_agents):
            break
        agent_files = agent_file_index.get(agent_id, [])
        if not agent_files:
            live.update(
                scanned=idx,
                sampled=sampled_agents,
                parse_fallback_agents=int(sampletime_stats.get("parse_fallback_agents", 0)),
            )
            continue
        parquet_paths = [p for p, _, _ in agent_files]
        ts = _load_agent_timestamps_for_sampletime(
            agent_id=agent_id,
            parquet_paths=parquet_paths,
            max_points=int(points_per_agent),
            column_map=column_map,
            verbose_progress=False,
            stats_out=sampletime_stats,
        )
        if ts.size < 2:
            live.update(
                scanned=idx,
                sampled=sampled_agents,
                parse_fallback_agents=int(sampletime_stats.get("parse_fallback_agents", 0)),
            )
            continue
        dt = np.diff(ts)
        dt = dt[np.isfinite(dt) & (dt > 0.0)]
        if dt.size == 0:
            live.update(
                scanned=idx,
                sampled=sampled_agents,
                parse_fallback_agents=int(sampletime_stats.get("parse_fallback_agents", 0)),
            )
            continue
        interval_batches.append(dt)
        sampled_agents += 1
        if idx % log_every == 0 or sampled_agents >= int(max_agents) or idx == total_candidates:
            msg = (
                "Sample-time estimation progress: "
                f"scanned={int(idx)}/{int(total_candidates)} "
                f"sampled={int(sampled_agents)}/{int(max_agents)}"
            )
            logger.info(msg)
        live.update(
            scanned=idx,
            sampled=sampled_agents,
            parse_fallback_agents=int(sampletime_stats.get("parse_fallback_agents", 0)),
        )

    if not interval_batches:
        raise ValueError("Failed to estimate sample time: no valid timestamp intervals found")

    with _STDOUT_LOCK:
        sys.stdout.write("\n")
        sys.stdout.flush()

    intervals = np.concatenate(interval_batches)
    return {
        "sampled_agents": int(sampled_agents),
        "n_intervals": int(intervals.size),
        "mean_sec": float(np.mean(intervals)),
        "median_sec": float(np.median(intervals)),
        "std_sec": float(np.std(intervals)),
    }


def _summarize_intervals(intervals: np.ndarray) -> dict:
    values = np.asarray(intervals, dtype=np.float64)
    if values.size == 0:
        return {"n": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}
    return {
        "n": float(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
    }


def _progress_log_interval(total: int, target_logs: int = 20) -> int:
    total_i = max(1, int(total))
    return max(1, total_i // max(1, int(target_logs)))


def _resolve_target_sizes(
    m: Optional[int],
    n_points: Optional[int],
    *,
    allow_size_override: bool = False,
    default_m: int = FIXED_TRAJ_COUNT,
    default_n: int = FIXED_TRAJ_POINTS,
) -> tuple[int, int]:
    target_m = int(default_m)
    target_n = int(default_n)
    if allow_size_override:
        if m is not None:
            target_m = int(m)
        if n_points is not None:
            target_n = int(n_points)
    else:
        if m is not None and int(m) != target_m:
            logger.warning("Ignoring requested m=%s; fixed m=%d is used.", m, target_m)
        if n_points is not None and int(n_points) != target_n:
            logger.warning("Ignoring requested n_points=%s; fixed n_points=%d is used.", n_points, target_n)
    if target_m <= 0:
        raise ValueError("Target m must be > 0")
    if target_n < 2:
        raise ValueError("Target n_points must be >= 2")
    return target_m, target_n


def _load_agent_arrays_for_sampling(
    agent_id: Any,
    parquet_paths: List[str],
    max_rows: Optional[int] = None,
    column_map: Optional[Dict[str, str]] = None,
    include_error_range: bool = False,
    verbose_progress: bool = True,
) -> Optional[dict]:
    if not parquet_paths:
        return None

    column_map = _detect_column_map(str(Path(parquet_paths[0]).parent), column_map)
    agent_col = column_map["agent"]
    ts_col = column_map["timestamp"]
    lon_n_col = column_map["longitude_n"]
    lat_n_col = column_map["latitude_n"]
    lon_col = column_map["longitude"]
    lat_col = column_map["latitude"]
    err_col = column_map.get("error_range")

    valid_paths = [str(Path(p)) for p in parquet_paths if Path(p).exists()]
    if not valid_paths:
        return None
    if verbose_progress:
        _progress_print(
            f"[traj:loader] agent={agent_id} load_start files={int(len(valid_paths))} "
            f"max_rows={int(max_rows) if max_rows is not None else -1}"
        )

    ts_batches: List[np.ndarray] = []
    lon_n_batches: List[np.ndarray] = []
    lat_n_batches: List[np.ndarray] = []
    lon_batches: List[np.ndarray] = []
    lat_batches: List[np.ndarray] = []
    err_batches: List[np.ndarray] = []
    total_rows = 0
    total_files = int(len(valid_paths))
    for file_idx, pq_path in enumerate(valid_paths, start=1):
        remaining = None
        if max_rows is not None:
            remaining = int(max_rows) - int(total_rows)
            if remaining <= 0:
                break
        if verbose_progress:
            _progress_print(
                f"[traj:loader] agent={agent_id} file_start {int(file_idx)}/{int(total_files)} "
                f"name={Path(pq_path).name} remaining={int(remaining) if remaining is not None else -1}"
            )
        try:
            lazy = (
                pl.scan_parquet(pq_path)
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
                    & pl.col(ts_col).is_not_null()
                )
                .select([ts_col, lon_n_col, lat_n_col, lon_col, lat_col])
            )
            if include_error_range and err_col is not None:
                lazy = (
                    pl.scan_parquet(pq_path)
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
                        & pl.col(err_col).is_not_null()
                        & pl.col(err_col).is_finite()
                        & pl.col(ts_col).is_not_null()
                    )
                    .select([ts_col, lon_n_col, lat_n_col, lon_col, lat_col, err_col])
                )
            if remaining is not None:
                lazy = lazy.limit(int(remaining))
            try:
                one_df = lazy.collect(streaming=True)
            except TypeError:
                one_df = lazy.collect()
        except Exception as one_exc:
            logger.warning("Skipping unreadable parquet %s: %s", Path(pq_path).name, one_exc)
            if verbose_progress:
                _progress_print(
                    f"[traj:loader] agent={agent_id} file_failed "
                    f"name={Path(pq_path).name} err={one_exc}"
                )
            continue

        rows = int(len(one_df))
        if verbose_progress:
            _progress_print(
                f"[traj:loader] agent={agent_id} file_done name={Path(pq_path).name} rows={rows}"
            )
        if rows == 0:
            continue
        ts_vals = one_df[ts_col].to_list()
        ts_parsed = np.fromiter(
            (
                v if (v is not None and np.isfinite(v)) else np.nan
                for v in (_to_unix_seconds_scalar(x) for x in ts_vals)
            ),
            dtype=np.float64,
        )
        keep = np.isfinite(ts_parsed)
        if not np.any(keep):
            continue
        ts_batches.append(ts_parsed[keep])
        lon_n_batches.append(one_df[lon_n_col].to_numpy()[keep])
        lat_n_batches.append(one_df[lat_n_col].to_numpy()[keep])
        lon_batches.append(one_df[lon_col].to_numpy()[keep])
        lat_batches.append(one_df[lat_col].to_numpy()[keep])
        if include_error_range and err_col is not None:
            err_batches.append(one_df[err_col].to_numpy()[keep])
        total_rows += rows

    if not ts_batches:
        if verbose_progress:
            _progress_print(f"[traj:loader] agent={agent_id} no_valid_rows")
        return None

    ts = np.concatenate(ts_batches)
    lon_n = np.concatenate(lon_n_batches)
    lat_n = np.concatenate(lat_n_batches)
    lon = np.concatenate(lon_batches)
    lat = np.concatenate(lat_batches)
    err = np.concatenate(err_batches) if err_batches else None
    if ts_batches and len(ts_batches) > 1:
        order = np.argsort(ts, kind="stable")
        ts = ts[order]
        lon_n = lon_n[order]
        lat_n = lat_n[order]
        lon = lon[order]
        lat = lat[order]
        if err is not None:
            err = err[order]
    if verbose_progress:
        _progress_print(
            f"[traj:loader] agent={agent_id} load_done points={int(ts.size)} files_used={int(len(ts_batches))}"
        )
    out = {
        "timestamp": ts,
        "longitude_n": lon_n,
        "latitude_n": lat_n,
        "longitude": lon,
        "latitude": lat,
    }
    if err is not None:
        out["error_range"] = err
    return out


def _sample_indices_with_gap_distribution(
    ts: np.ndarray,
    target_points: int,
    *,
    rng: np.random.Generator,
    sample_gap_fn: Callable[[int], np.ndarray],
    max_start_tries: int = 64,
    retries_per_start: int = 3,
    min_gap_seconds: float = 1.0,
    max_gap_seconds: float | None = None,
    max_realized_gap_multiplier: float | None = float(MAX_REALIZED_GAP_MULTIPLIER),
) -> Optional[dict]:
    if target_points < 2 or ts.size < target_points:
        return None

    n = int(ts.size)
    max_start_idx = n - target_points
    if max_start_idx <= 0:
        start_candidates = np.array([0], dtype=np.int64)
    else:
        count = min(int(max_start_tries), max_start_idx + 1)
        # Sample directly from the integer domain to avoid materializing a large arange buffer.
        start_candidates = rng.choice(max_start_idx + 1, size=count, replace=False)

    for start_idx in start_candidates:
        for _ in range(int(retries_per_start)):
            idxs = [int(start_idx)]
            accepted_gap_sum = 0.0
            accepted_gap_count = 0
            sampled_gaps = np.asarray(sample_gap_fn(target_points - 1), dtype=np.float64)
            if sampled_gaps.shape[0] != target_points - 1:
                raise ValueError(
                    f"sample_gap_fn returned size={sampled_gaps.shape[0]}, expected={target_points - 1}"
                )
            sampled_gaps = np.maximum(sampled_gaps, float(min_gap_seconds))
            if max_gap_seconds is not None:
                sampled_gaps = np.minimum(sampled_gaps, float(max_gap_seconds))

            ok = True
            for gap in sampled_gaps:
                prev_idx = idxs[-1]
                target_time = ts[prev_idx] + float(gap)
                nxt_idx = int(np.searchsorted(ts, target_time, side="left"))
                if nxt_idx <= prev_idx:
                    nxt_idx = prev_idx + 1
                if nxt_idx >= n:
                    ok = False
                    break
                realized_gap = float(ts[nxt_idx] - ts[prev_idx])
                if (
                    max_realized_gap_multiplier is not None
                    and np.isfinite(realized_gap)
                    and np.isfinite(gap)
                    and float(gap) > 0.0
                    and realized_gap > float(max_realized_gap_multiplier) * float(gap)
                ):
                    fallback_idx = int(prev_idx) + 1
                    if fallback_idx >= n:
                        ok = False
                        break
                    fallback_gap = float(ts[fallback_idx] - ts[prev_idx])
                    if not np.isfinite(fallback_gap) or fallback_gap <= 0.0:
                        ok = False
                        break
                    projected_avg = float(accepted_gap_sum + fallback_gap) / float(accepted_gap_count + 1)
                    if projected_avg < (2.0 * float(gap)):
                        nxt_idx = int(fallback_idx)
                        realized_gap = float(fallback_gap)
                    else:
                        return {
                            "gap_break": True,
                            "break_idx": int(nxt_idx),
                            "intervals_sec": None,
                        }
                idxs.append(nxt_idx)
                accepted_gap_sum += float(realized_gap)
                accepted_gap_count += 1

            if not ok or len(idxs) != target_points:
                continue

            idxs_arr = np.asarray(idxs, dtype=np.int64)
            return {
                "idxs": idxs_arr,
                "intervals_sec": np.diff(ts[idxs_arr]),
                "last_idx": int(idxs_arr[-1]),
            }
    return None


def _save_processed_trajectory_dataset(
    processed_trajs: List[dict],
    output_dir: str,
    prefix: str,
    target_m: int,
    n_points: int,
    metadata_extra: Optional[dict] = None,
    filename_override: Optional[str] = None,
) -> str:
    n_trajectories = len(processed_trajs)
    lengths = [int(t["n_points"]) for t in processed_trajs]
    median_length = int(np.median(lengths)) if lengths else 0
    total_points = int(sum(lengths))

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    filename = str(filename_override) if filename_override else f"{prefix}_{n_trajectories}_{median_length}.pt"
    output_file = out_path / filename

    meta = {
        "n_trajectories": n_trajectories,
        "total_points": total_points,
        "median_length": median_length,
        "target_M": int(target_m),
        "target_N": int(n_points),
    }
    if metadata_extra:
        meta.update(metadata_extra)

    trajectories = []
    for t in processed_trajs:
        row = {
            "agent_id": t["agent_id"],
            "n_points": t["n_points"],
            "data": torch.tensor(t["data"], dtype=torch.float32),
            "label": torch.tensor(t["label"], dtype=torch.float32),
        }
        if "error_range" in t:
            acc = torch.tensor(t["error_range"], dtype=torch.float32)
            row["error_range"] = acc
            row["accuracy"] = acc
        if "timestamp" in t:
            row["timestamp"] = torch.tensor(t["timestamp"], dtype=torch.float32)
        trajectories.append(row)

    save_data = {
        "trajectories": trajectories,
        "metadata": meta,
    }
    torch.save(save_data, output_file)
    return str(output_file)


def _sample_time_label_from_seconds(seconds_value: float, *, native: bool = False) -> str:
    if native:
        return "native"
    sec = int(round(float(seconds_value)))
    if sec <= 0:
        sec = 1
    if sec > 60:
        minute = max(1, int(round(float(sec) / 60.0)))
        return f"{minute}min"
    return f"{sec}s"


def _summary_stats(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"avg": 0.0, "median": 0.0, "std": 0.0, "count": 0}
    return {
        "avg": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "count": int(arr.size),
    }


def _haversine_distance_m(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    # Vectorized geodesic approximation for pointwise noisy-vs-reference distance in meters.
    r = 6371008.8
    lat1r = np.radians(lat1.astype(np.float64))
    lat2r = np.radians(lat2.astype(np.float64))
    dlat = lat2r - lat1r
    dlon = np.radians(lon2.astype(np.float64) - lon1.astype(np.float64))
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))


def _build_quality_stats(processed_trajs: List[dict], *, include_error_range: bool) -> dict:
    if not processed_trajs:
        return {"dist2ref": _summary_stats(np.array([], dtype=np.float64))}

    dist_batches: List[np.ndarray] = []
    acc_batches: List[np.ndarray] = []
    for t in processed_trajs:
        data = np.asarray(t["data"], dtype=np.float64)
        label = np.asarray(t["label"], dtype=np.float64)
        dist = _haversine_distance_m(
            data[:, 0],
            data[:, 1],
            label[:, 0],
            label[:, 1],
        )
        dist_batches.append(dist)
        if include_error_range and "error_range" in t:
            acc_batches.append(np.asarray(t["error_range"], dtype=np.float64))

    all_dist = np.concatenate(dist_batches) if dist_batches else np.array([], dtype=np.float64)
    out = {"dist2ref": _summary_stats(all_dist)}

    if include_error_range and acc_batches:
        all_acc = np.concatenate(acc_batches)
        out["accuracy"] = _summary_stats(all_acc)
        tier_defs = [
            ("tier0_acc_leq_5", 5.0),
            ("tier1_acc_leq_10", 10.0),
            ("tier2_acc_leq_15", 15.0),
            ("tier3_acc_leq_30", 30.0),
            ("tier4_all", None),
        ]
        tiers = {}
        for name, thr in tier_defs:
            if thr is None:
                mask = np.ones_like(all_acc, dtype=bool)
            else:
                mask = all_acc <= float(thr)
            tiers[name] = {
                "dist2ref": _summary_stats(all_dist[mask] if np.any(mask) else np.array([], dtype=np.float64)),
                "accuracy": _summary_stats(all_acc[mask] if np.any(mask) else np.array([], dtype=np.float64)),
            }
        out["tiers"] = tiers
    return out


def _infer_target_gap_seconds(
    *,
    is_native_sampler: bool,
    sampler_metadata: Optional[dict],
) -> Optional[float]:
    if bool(is_native_sampler):
        return None
    meta = sampler_metadata if isinstance(sampler_metadata, dict) else {}
    # Prefer explicit fixed-interval targets; for mimic runs prefer median as stable anchor.
    for key in ("interval_sec", "target_median_sec", "target_mean_sec"):
        value = meta.get(key)
        try:
            sec = float(value)
        except Exception:
            continue
        if np.isfinite(sec) and sec > 0.0:
            return float(sec)
    return None


def _required_raw_rows_for_target(
    *,
    target_n: int,
    target_gap_sec: Optional[float],
    native_sample_sec_hint: Optional[float],
    safety_multiplier: float = 1.25,
) -> Optional[int]:
    if target_gap_sec is None or native_sample_sec_hint is None:
        return None
    try:
        gap = float(target_gap_sec)
        native = float(native_sample_sec_hint)
        safety = float(safety_multiplier)
    except Exception:
        return None
    if not (np.isfinite(gap) and np.isfinite(native) and np.isfinite(safety)):
        return None
    if gap <= 0.0 or native <= 0.0 or safety <= 0.0:
        return None
    ratio = max(1.0, float(gap) / float(native))
    need = int(np.ceil(float(target_n) * ratio * safety))
    return max(int(target_n), int(need))


def _extract_trajectories_with_gap_sampler(
    parquet_dir: str,
    output_dir: str,
    *,
    target_m: int,
    target_n: int,
    prefix: str,
    sample_gap_fn: Callable[[int], np.ndarray],
    sampler_name: str,
    sampler_metadata: Optional[dict] = None,
    allow_shorter: bool = False,
    max_gap_seconds: float | None = None,
    shuffle_seed: int = 101,
    metadata: Optional[dict] = None,
    column_map: Optional[Dict[str, str]] = None,
    ordered_agents: Optional[List[Any]] = None,
    agent_entry_counts: Optional[Dict[Any, int]] = None,
    include_error_range: bool = False,
    is_native_sampler: bool = False,
    sample_time_label: Optional[str] = None,
    native_sample_sec_hint: Optional[float] = None,
) -> dict:
    _assert_not_calibration_or_processed_source(parquet_dir)
    column_map = _detect_column_map(parquet_dir, column_map)
    metadata = metadata if metadata is not None else scan_parquet_metadata(parquet_dir, column_map=column_map)
    agent_file_index = build_agent_file_index(metadata)
    if ordered_agents is not None:
        all_agents = [a for a in ordered_agents if a in agent_file_index]
    else:
        all_agents = list(agent_file_index.keys())
    if not all_agents:
        raise ValueError("No agents found in parquet directory")

    target_gap_sec = _infer_target_gap_seconds(
        is_native_sampler=bool(is_native_sampler),
        sampler_metadata=sampler_metadata,
    )
    rows_needed_hint = _required_raw_rows_for_target(
        target_n=int(target_n),
        target_gap_sec=target_gap_sec,
        native_sample_sec_hint=native_sample_sec_hint,
        safety_multiplier=1.25,
    )
    if (
        not bool(is_native_sampler)
        and isinstance(agent_entry_counts, dict)
        and rows_needed_hint is not None
    ):
        def _rows_for_agent(agent_key: Any) -> int:
            if agent_key in agent_entry_counts:
                return int(agent_entry_counts.get(agent_key, 0))
            agent_text = str(agent_key)
            if agent_text in agent_entry_counts:
                return int(agent_entry_counts.get(agent_text, 0))
            return 0

        min_candidate_rows = max(int(target_n), int(np.ceil(float(rows_needed_hint) * 0.80)))
        filtered_agents = [a for a in all_agents if _rows_for_agent(a) >= int(min_candidate_rows)]
        if len(filtered_agents) >= int(target_m):
            msg = (
                f"[traj:{sampler_name}] candidate_filter: "
                f"rows>={int(min_candidate_rows)} "
                f"kept={int(len(filtered_agents))}/{int(len(all_agents))} "
                f"(rows_needed_hint={int(rows_needed_hint)})"
            )
            logger.info(msg)
            _progress_print(msg)
            all_agents = filtered_agents

    rng = np.random.default_rng(shuffle_seed)
    if ordered_agents is None:
        rng.shuffle(all_agents)

    trajectory_pool: List[Tuple[int, dict]] = []
    total_points = 0
    target_total_points = int(target_m) * int(target_n)
    stalled_users_after_pool_full = 0
    interval_sum_sec = 0.0
    interval_count = 0
    last_enqueued_len = 0
    all_intervals = []
    extraction_failures = 0
    sampled_agents = 0
    total_agents = int(len(all_agents))
    log_every_trajs = max(1, int(target_m) // 10)
    last_logged_traj_bucket = -1
    # Keep per-agent buffers bounded to reduce memory spikes under strict caps.
    # Native mode is intentionally conservative (1 trajectory per agent) to avoid allocator bursts.
    per_agent_traj_cap = 1 if bool(is_native_sampler) else int(MAX_TRAJ_PER_AGENT)
    if bool(is_native_sampler):
        max_rows_per_agent = int(target_n)
    else:
        base_rows = max(int(target_n) * 3, 20000)
        if rows_needed_hint is not None:
            max_rows_per_agent = min(80000, max(int(base_rows), int(rows_needed_hint)))
        else:
            max_rows_per_agent = int(base_rows)
    msg = (
        f"[traj:{sampler_name}] Start extraction: "
        f"agents={int(total_agents)} target_traj={int(target_m)} target_points={int(target_n)} "
        f"max_rows_per_agent={int(max_rows_per_agent)} "
        f"target_gap_sec={float(target_gap_sec):.3f} "
        f"native_hint_sec={float(native_sample_sec_hint):.3f}"
        if target_gap_sec is not None and native_sample_sec_hint is not None
        else (
            f"[traj:{sampler_name}] Start extraction: "
            f"agents={int(total_agents)} target_traj={int(target_m)} target_points={int(target_n)} "
            f"max_rows_per_agent={int(max_rows_per_agent)}"
        )
    )
    logger.info(msg)
    _progress_print(msg)
    live = _LiveThreeLineProgress(
        sampler_name=sampler_name,
        target_m=int(target_m),
        total_agents=int(total_agents),
        stall_limit=int(MAX_STALLED_USERS_AFTER_POOL_FULL),
    )
    live.update(
        queue_size=0,
        agents_scanned=0,
        stalled_users=0,
        min_len=0,
        max_len=0,
        last_enqueued_len=0,
        avg_sample_sec=0.0,
    )
    last_rendered_queue = 0
    last_rendered_stalled = 0
    last_rendered_min_len = 0
    last_rendered_max_len = 0
    last_rendered_last_enqueued_len = 0
    render_every_scanned_agents = 10

    def _pool_len_bounds(pool: List[Tuple[int, dict]]) -> Tuple[int, int]:
        if not pool:
            return 0, 0
        # Pool is maintained sorted by n_points descending.
        return int(pool[-1][0]), int(pool[0][0])

    def _maybe_render_progress(current_idx: int, *, force: bool = False) -> None:
        nonlocal last_rendered_queue
        nonlocal last_rendered_stalled
        nonlocal last_rendered_min_len
        nonlocal last_rendered_max_len
        nonlocal last_rendered_last_enqueued_len

        min_len, max_len = _pool_len_bounds(trajectory_pool)
        should_render = (
            bool(force)
            or len(trajectory_pool) != int(last_rendered_queue)
            or int(stalled_users_after_pool_full) != int(last_rendered_stalled)
            or int(min_len) != int(last_rendered_min_len)
            or int(max_len) != int(last_rendered_max_len)
            or int(last_enqueued_len) != int(last_rendered_last_enqueued_len)
            or (int(current_idx) % int(render_every_scanned_agents) == 0)
            or (int(current_idx) == int(total_agents))
        )
        if not should_render:
            return

        live.update(
            queue_size=len(trajectory_pool),
            agents_scanned=int(current_idx),
            stalled_users=stalled_users_after_pool_full,
            min_len=min_len,
            max_len=max_len,
            last_enqueued_len=last_enqueued_len,
            avg_sample_sec=(interval_sum_sec / interval_count) if interval_count > 0 else 0.0,
        )
        last_rendered_queue = int(len(trajectory_pool))
        last_rendered_stalled = int(stalled_users_after_pool_full)
        last_rendered_min_len = int(min_len)
        last_rendered_max_len = int(max_len)
        last_rendered_last_enqueued_len = int(last_enqueued_len)

    for idx, agent_id in enumerate(all_agents, start=1):
        if _pool_reached_target(trajectory_pool, int(target_m), int(target_n)):
            break

        sampled_agents += 1
        pool_improved_for_user = False
        agent_files = agent_file_index.get(agent_id, [])
        if not agent_files:
            extraction_failures += 1
            _maybe_render_progress(idx)
            continue

        parquet_paths = [p for p, _, _ in agent_files]
        agent_data = _load_agent_arrays_for_sampling(
            agent_id,
            parquet_paths,
            max_rows=int(max_rows_per_agent),
            column_map=column_map,
            include_error_range=include_error_range,
            verbose_progress=False,
        )
        if agent_data is None:
            extraction_failures += 1
            _maybe_render_progress(idx)
            continue

        cursor = 0
        per_agent_kept = 0
        while per_agent_kept < int(per_agent_traj_cap):
            ts_all = agent_data["timestamp"]
            if cursor >= len(ts_all):
                break
            rem_len = int(len(ts_all) - cursor)
            if rem_len < 2:
                break

            current_target_n = int(target_n)
            if allow_shorter:
                current_target_n = min(current_target_n, rem_len)
            if current_target_n < 2:
                break

            sampled = None
            if bool(is_native_sampler):
                if rem_len >= current_target_n:
                    idxs_native = np.arange(cursor, cursor + int(current_target_n), dtype=np.int64)
                    sampled = {
                        "idxs": (idxs_native - int(cursor)),
                        "intervals_sec": np.diff(ts_all[idxs_native]),
                        "last_idx": int(current_target_n - 1),
                    }
            else:
                sampled = _sample_indices_with_gap_distribution(
                    ts_all[cursor:],
                    target_points=current_target_n,
                    rng=rng,
                    sample_gap_fn=sample_gap_fn,
                    min_gap_seconds=1.0,
                    max_gap_seconds=max_gap_seconds,
                    max_realized_gap_multiplier=float(MAX_REALIZED_GAP_MULTIPLIER),
                )
            if sampled is not None and bool(sampled.get("gap_break", False)):
                break_idx = int(sampled.get("break_idx", 0))
                cursor += max(1, break_idx)
                continue
            if sampled is None and allow_shorter and current_target_n > 2:
                fallback_targets = [max(2, current_target_n // 2), max(2, current_target_n // 4)]
                for fallback_n in fallback_targets:
                    if fallback_n >= current_target_n:
                        continue
                    sampled = _sample_indices_with_gap_distribution(
                        ts_all[cursor:],
                        target_points=fallback_n,
                        rng=rng,
                        sample_gap_fn=sample_gap_fn,
                        min_gap_seconds=1.0,
                        max_gap_seconds=max_gap_seconds,
                        max_realized_gap_multiplier=float(MAX_REALIZED_GAP_MULTIPLIER),
                    )
                    if sampled is not None and bool(sampled.get("gap_break", False)):
                        break
                    if sampled is not None:
                        break
            if sampled is not None and bool(sampled.get("gap_break", False)):
                break_idx = int(sampled.get("break_idx", 0))
                cursor += max(1, break_idx)
                continue
            if sampled is None:
                break
            if (not bool(is_native_sampler)) and (target_gap_sec is not None):
                sampled_intervals = sampled.get("intervals_sec")
                if sampled_intervals is not None:
                    one_intervals = np.asarray(sampled_intervals, dtype=np.float64)
                    if one_intervals.size > 0:
                        mean_interval_sec = float(np.mean(one_intervals))
                        max_allowed_mean_sec = float(target_gap_sec) * float(MAX_AVG_INTERVAL_MULTIPLIER)
                        if (
                            np.isfinite(mean_interval_sec)
                            and np.isfinite(max_allowed_mean_sec)
                            and mean_interval_sec > max_allowed_mean_sec
                        ):
                            cursor += int(sampled["last_idx"]) + 1
                            continue

            idxs = sampled["idxs"] + cursor
            extracted = {
                "agent_id": int(agent_id),
                "n_points": int(idxs.shape[0]),
                "longitude_n": agent_data["longitude_n"][idxs],
                "latitude_n": agent_data["latitude_n"][idxs],
                "longitude": agent_data["longitude"][idxs],
                "latitude": agent_data["latitude"][idxs],
                "timestamp": agent_data["timestamp"][idxs],
            }
            if include_error_range:
                extracted["error_range"] = agent_data["error_range"][idxs]
                processed_one = data_processor_with_error_range(extracted)
            else:
                processed_one = data_processor(extracted)
            inserted, total_points = _insert_into_pool(
                trajectory_pool,
                total_points,
                processed_one,
                int(target_total_points),
                int(target_m),
            )
            if not inserted:
                # Skip cursor region even when not inserted; prevents cycling on short tails.
                cursor += int(sampled["last_idx"]) + 1
                continue
            pool_improved_for_user = True
            last_enqueued_len = int(processed_one.get("n_points", 0))
            all_intervals.append(sampled["intervals_sec"])
            if sampled["intervals_sec"] is not None:
                one_intervals = np.asarray(sampled["intervals_sec"], dtype=np.float64)
                if one_intervals.size > 0:
                    interval_sum_sec += float(np.sum(one_intervals))
                    interval_count += int(one_intervals.size)
            per_agent_kept += 1
            cursor += int(sampled["last_idx"]) + 1
            traj_bucket = int(len(trajectory_pool) // log_every_trajs)
            if traj_bucket > last_logged_traj_bucket:
                last_logged_traj_bucket = traj_bucket
                pass

        if per_agent_kept == 0:
            extraction_failures += 1
        del agent_data
        if idx % 20 == 0:
            gc.collect()

        # Halt counter starts after scanning first M users, regardless of pool fill.
        if int(idx) >= int(target_m):
            if pool_improved_for_user:
                stalled_users_after_pool_full = 0
            else:
                stalled_users_after_pool_full += 1
                if stalled_users_after_pool_full >= int(MAX_STALLED_USERS_AFTER_POOL_FULL):
                    msg = (
                        f"[traj:{sampler_name}] early_stop_stalled_after_seed_window "
                        f"stalled_users={int(stalled_users_after_pool_full)} "
                        f"seed_window={int(target_m)} pool_size={int(len(trajectory_pool))}"
                    )
                    logger.info(msg)
                    _progress_print(msg)
                    break

        _maybe_render_progress(idx)

    # Move cursor below the live 3-line display.
    sys.stdout.write("\n")
    sys.stdout.flush()

    if not trajectory_pool:
        raise RuntimeError(f"No valid trajectories extracted for sampler={sampler_name}")

    processed = [t[1] for t in trajectory_pool]
    intervals_flat = (
        np.concatenate(all_intervals, axis=0) if all_intervals else np.array([], dtype=np.float64)
    )
    interval_stats = _summarize_intervals(intervals_flat)
    lengths = [int(t["n_points"]) for t in processed]
    total_points = int(sum(lengths))
    median_length = int(np.median(lengths)) if lengths else 0
    avg_length = int(round(float(np.mean(lengths)))) if lengths else 0
    sample_label = str(sample_time_label) if sample_time_label else _sample_time_label_from_seconds(
        interval_stats.get("mean", 0.0),
        native=bool(is_native_sampler),
    )
    out_filename = f"traj_{sample_label}_{int(target_m)}_{int(avg_length)}.pt"

    meta_extra = {"sampler": sampler_name, "sample_interval_stats_sec": interval_stats}
    if sampler_metadata:
        meta_extra.update(sampler_metadata)
    meta_extra["sample_time_label"] = str(sample_label)
    meta_extra["avg_length"] = int(avg_length)
    output_path = _save_processed_trajectory_dataset(
        processed_trajs=processed,
        output_dir=output_dir,
        prefix=prefix,
        target_m=int(target_m),
        n_points=int(target_n),
        metadata_extra=meta_extra,
        filename_override=out_filename,
    )
    quality_stats = _build_quality_stats(processed, include_error_range=include_error_range)

    return {
        "output_path": output_path,
        "n_trajectories": len(processed),
        "total_points": total_points,
        "median_length": median_length,
        "avg_length": int(avg_length),
        "min_length": int(min(lengths)) if lengths else 0,
        "max_length": int(max(lengths)) if lengths else 0,
        "extraction_failures": int(extraction_failures),
        "agents_sampled": int(sampled_agents),
        "interval_stats_sec": interval_stats,
        "sample_time_label": str(sample_label),
        "quality_stats": quality_stats,
        "sampler": sampler_name,
    }


def _format_interval_label(interval_sec: int) -> str:
    if int(interval_sec) % 60 == 0:
        return f"{int(interval_sec) // 60}min"
    return f"{int(interval_sec)}s"


def extract_10min_traj(
    parquet_dir: str = "./dataset/raw/NUMOSIM_Kanto",
    m: int | None = None,
    n_points: int | None = None,
    intervals_min: Optional[list[int]] = None,
    intervals_sec: Optional[list[int]] = None,
    output_dir_tmpl: str = "./dataset/processed/NUMOSIM_Kanto/test/traj_test",
    time_tolerance_sec: int = 60,
    allow_shorter: bool = False,
    allow_size_override: bool = False,
    include_error_range: bool = False,
    precomputed_metadata: Optional[dict] = None,
    precomputed_column_map: Optional[Dict[str, str]] = None,
    ordered_agents: Optional[List[Any]] = None,
    agent_entry_counts: Optional[Dict[Any, int]] = None,
    native_sample_sec_hint: Optional[float] = None,
) -> dict:
    del time_tolerance_sec  # compatibility only; interval matching is handled by searchsorted semantics
    _log_thread_budget("extract_10min_traj")
    target_m, target_n = _resolve_target_sizes(
        m,
        n_points,
        allow_size_override=allow_size_override,
    )

    if intervals_sec is not None:
        target_intervals_sec = [int(x) for x in intervals_sec]
    else:
        target_intervals_min = intervals_min if intervals_min is not None else [2, 14, 20]
        if not target_intervals_min:
            raise ValueError("intervals_min must not be empty")
        target_intervals_sec = [int(x) * 60 for x in target_intervals_min]
    if not target_intervals_sec:
        raise ValueError("Target interval list must not be empty")

    column_map = _detect_column_map(parquet_dir, precomputed_column_map)
    metadata = precomputed_metadata if precomputed_metadata is not None else scan_parquet_metadata(
        parquet_dir,
        column_map=column_map,
    )
    all_parquet_paths = sorted(Path(parquet_dir).glob("*.parquet"))
    expected_test_files = {str(p) for p in all_parquet_paths[-int(TEST_SPLIT_PARQUET_COUNT):]}
    actual_files = set(metadata.keys())
    if not actual_files:
        raise ValueError("No metadata scanned from parquet files")
    if not actual_files.issubset(expected_test_files):
        raise RuntimeError(
            "Detected non-test parquet files in metadata scan. "
            f"Expected only last {int(TEST_SPLIT_PARQUET_COUNT)} parquet file(s) (test split). "
            f"Expected subset={sorted(expected_test_files)}, actual={sorted(actual_files)}"
        )

    interval_results = {}
    for interval_sec in target_intervals_sec:
        label = _format_interval_label(interval_sec)
        msg = (
            "[traj] Running fixed-gap extraction for "
            f"interval={label} ({int(interval_sec)}s) "
            f"target_m={int(target_m)} target_n={int(target_n)}"
        )
        logger.info(msg)
        _progress_print(msg)

        def _constant_gap_fn(size: int, _v=float(interval_sec)) -> np.ndarray:
            return np.full(int(size), _v, dtype=np.float64)

        one = _extract_trajectories_with_gap_sampler(
            parquet_dir=parquet_dir,
            output_dir=output_dir_tmpl.format(interval=label),
            target_m=target_m,
            target_n=target_n,
            prefix=f"fulltraj{label}",
            sample_gap_fn=_constant_gap_fn,
            sampler_name=f"constant_{label}",
            sampler_metadata={"interval_sec": int(interval_sec), "interval_label": label},
            allow_shorter=allow_shorter,
            metadata=metadata,
            column_map=column_map,
            ordered_agents=ordered_agents,
            agent_entry_counts=agent_entry_counts,
            include_error_range=include_error_range,
            is_native_sampler=False,
            sample_time_label=None,
            native_sample_sec_hint=native_sample_sec_hint,
        )
        interval_results[label] = {
            "interval_sec": int(interval_sec),
            "interval_label": label,
            "regular_path": one["output_path"],
            "n_trajectories": int(one["n_trajectories"]),
            "total_points": int(one["total_points"]),
            "avg_length": int(one.get("avg_length", 0)),
            "median_length": int(one["median_length"]),
            "min_length": int(one["min_length"]),
            "max_length": int(one["max_length"]),
            "extraction_failures": int(one["extraction_failures"]),
            "interval_stats_sec": one["interval_stats_sec"],
            "sample_time_label": one.get("sample_time_label"),
            "quality_stats": one.get("quality_stats", {}),
        }
        msg = (
            f"[traj] Completed interval={label}: "
            f"n_trajectories={int(one['n_trajectories'])} "
            f"total_points={int(one['total_points'])} "
            f"failures={int(one['extraction_failures'])} "
            f"output={one['output_path']}"
        )
        logger.info(msg)
        _progress_print(msg)

    return {
        "interval_results": interval_results,
        "intervals_sec": [int(x) for x in target_intervals_sec],
        "m": int(target_m),
        "n_points": int(target_n),
    }


def extract_sampletime_traj(
    parquet_dir: str = "./dataset/raw/NUMOSIM_Kanto",
    output_dir: str = "./dataset/processed/NUMOSIM_Kanto/test/traj_test",
    m: int | None = None,
    n_points: int | None = None,
    target_mean_sec: float = BLOGWATCH_TARGET_MEAN_SEC,
    target_median_sec: float = BLOGWATCH_TARGET_MEDIAN_SEC,
    target_std_sec: float = BLOGWATCH_TARGET_STD_SEC,
    fit_seed: int = 42,
    sample_seed: int = 7,
    shuffle_seed: int = 101,
    fit_iter: int = 20000,
    max_gap_seconds: float | None = None,
    allow_size_override: bool = False,
    allow_shorter: bool = False,
    include_error_range: bool = False,
    precomputed_metadata: Optional[dict] = None,
    precomputed_column_map: Optional[Dict[str, str]] = None,
    ordered_agents: Optional[List[Any]] = None,
    agent_entry_counts: Optional[Dict[Any, int]] = None,
    native_sample_sec_hint: Optional[float] = None,
) -> dict:
    try:
        from .blog_watcher_distribution import build_blogwatch_sample_time_generator
    except ImportError:
        from blog_watcher_distribution import build_blogwatch_sample_time_generator

    _log_thread_budget("extract_sampletime_traj")
    target_m, target_n = _resolve_target_sizes(
        m,
        n_points,
        allow_size_override=allow_size_override,
    )
    msg = (
        "[traj] Fitting sample-time generator: "
        f"target_mean={float(target_mean_sec):.3f} "
        f"target_median={float(target_median_sec):.3f} "
        f"target_std={float(target_std_sec):.3f} "
        f"iter={int(fit_iter)}"
    )
    logger.info(msg)
    _progress_print(msg)

    fit, generator = build_blogwatch_sample_time_generator(
        target_mean=float(target_mean_sec),
        target_median=float(target_median_sec),
        target_std=float(target_std_sec),
        fit_seed=int(fit_seed),
        sample_seed=int(sample_seed),
        n_iter=int(fit_iter),
    )

    def _gap_fn(size: int) -> np.ndarray:
        return generator.sample(int(size), min_seconds=1.0, round_to_int=False)

    result = _extract_trajectories_with_gap_sampler(
        parquet_dir=parquet_dir,
        output_dir=output_dir,
        target_m=target_m,
        target_n=target_n,
        prefix="fulltraj_sampletime",
        sample_gap_fn=_gap_fn,
        sampler_name="blogwatch_mimic",
        sampler_metadata={
            "fit": fit.as_dict(),
            "target_mean_sec": float(target_mean_sec),
            "target_median_sec": float(target_median_sec),
            "target_std_sec": float(target_std_sec),
        },
        allow_shorter=allow_shorter,
        max_gap_seconds=max_gap_seconds,
        shuffle_seed=int(shuffle_seed),
        metadata=precomputed_metadata,
        column_map=precomputed_column_map,
        ordered_agents=ordered_agents,
        agent_entry_counts=agent_entry_counts,
        include_error_range=include_error_range,
        is_native_sampler=False,
        sample_time_label=None,
        native_sample_sec_hint=native_sample_sec_hint,
    )
    result["fit"] = fit.as_dict()
    msg = (
        "[traj] Completed sample-time extraction: "
        f"n_trajectories={int(result['n_trajectories'])} "
        f"total_points={int(result['total_points'])} "
        f"failures={int(result['extraction_failures'])} "
        f"output={result['output_path']}"
    )
    logger.info(msg)
    _progress_print(msg)
    return result


def extract_native_traj(
    parquet_dir: str = "./dataset/raw/NUMOSIM_Kanto",
    output_dir: str = "./dataset/processed/NUMOSIM_Kanto/test/traj_test",
    m: int | None = None,
    n_points: int | None = None,
    allow_size_override: bool = False,
    allow_shorter: bool = False,
    include_error_range: bool = False,
    precomputed_metadata: Optional[dict] = None,
    precomputed_column_map: Optional[Dict[str, str]] = None,
    ordered_agents: Optional[List[Any]] = None,
    agent_entry_counts: Optional[Dict[Any, int]] = None,
) -> dict:
    _log_thread_budget("extract_native_traj")
    target_m, target_n = _resolve_target_sizes(
        m,
        n_points,
        allow_size_override=allow_size_override,
    )
    msg = (
        "[traj] Running native extraction: "
        f"target_m={int(target_m)} target_n={int(target_n)}"
    )
    logger.info(msg)
    _progress_print(msg)

    result = _extract_trajectories_with_gap_sampler(
        parquet_dir=parquet_dir,
        output_dir=output_dir,
        target_m=target_m,
        target_n=target_n,
        prefix="traj_native",
        sample_gap_fn=lambda size: np.ones(int(size), dtype=np.float64),
        sampler_name="native",
        sampler_metadata={"interval_label": "native"},
        allow_shorter=allow_shorter,
        metadata=precomputed_metadata,
        column_map=precomputed_column_map,
        ordered_agents=ordered_agents,
        agent_entry_counts=agent_entry_counts,
        include_error_range=include_error_range,
        is_native_sampler=True,
        sample_time_label="native",
        native_sample_sec_hint=None,
    )
    msg = (
        "[traj] Completed native extraction: "
        f"n_trajectories={int(result['n_trajectories'])} "
        f"total_points={int(result['total_points'])} "
        f"failures={int(result['extraction_failures'])} "
        f"output={result['output_path']}"
    )
    logger.info(msg)
    _progress_print(msg)
    return result


def load_test_trajectory(
    agent_id: Any,
    parquet_paths: List[str],
    n_points: int = 60480,
    include_error_range: bool = False,
    column_map: Optional[Dict[str, str]] = None,
) -> Optional[dict]:
    """
    Load N consecutive data points from a single agent across multiple parquet files.
    
    Purpose:
        Extract trajectory data for testing BF/DF denoising algorithms.
        Returns sorted, NaN-free GPS data for one agent.
    
    Parameters:
        agent_id (int): Target agent ID to extract
        parquet_paths (list[str]): List of parquet file paths to search
        n_points (int): Target number of points (default 60480 = 7 days). 
                       Returns shorter if agent has insufficient data.
    
    Return:
        out_load_test (dict | None): {
            "agent_id": int,
            "n_points": int,
            "longitude_n": np.ndarray,  # shape (N,)
            "latitude_n": np.ndarray,   # shape (N,)
            "longitude": np.ndarray,    # shape (N,)
            "latitude": np.ndarray,     # shape (N,)
            "timestamp": np.ndarray     # shape (N,), Unix timestamp (float)
        }
        Returns None if agent not found or has no valid data.
    
    Notes:
        - Filters NaN/null values in all 4 coordinate columns
        - Sorts by timestamp (ascending)
        - Searches files in order until n_points reached or files exhausted
        - Returns actual count (may be < n_points if agent data insufficient)
    """
    
    if not parquet_paths:
        return None

    column_map = _detect_column_map(str(Path(parquet_paths[0]).parent), column_map)
    agent_col = column_map["agent"]
    ts_col = column_map["timestamp"]
    lon_n_col = column_map["longitude_n"]
    lat_n_col = column_map["latitude_n"]
    lon_col = column_map["longitude"]
    lat_col = column_map["latitude"]
    err_col = column_map.get("error_range")

    collected_data = []
    total_collected = 0
    
    for pq_path in parquet_paths:
        if total_collected >= n_points:
            break
        
        path = Path(pq_path)
        if not path.exists():
            continue
        
        # Lazy scan with Polars
        df = (
            pl.scan_parquet(pq_path)
            .with_columns(_timestamp_expr(ts_col))
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
                & pl.col("_timestamp").is_not_null()
            )
        )

        if include_error_range and err_col is not None:
            df = df.filter(
                pl.col(err_col).is_not_null()
                & pl.col(err_col).is_finite()
            )

        select_cols = [lon_n_col, lat_n_col, lon_col, lat_col, "_timestamp"]
        if include_error_range and err_col is not None:
            select_cols.append(err_col)

        # Important fast-path: limit at lazy stage so we do not materialize
        # an entire per-agent file when only a prefix is needed.
        remaining = n_points - total_collected
        df = (
            df.select(select_cols)
            .sort("_timestamp")
            .limit(remaining)
            .collect()
        )
        
        if len(df) == 0:
            continue
        
        # Take what we need
        chunk = df.head(remaining)
        collected_data.append(chunk)
        total_collected += len(chunk)
    
    if total_collected == 0:
        return None
    
    # Concatenate all chunks and sort globally
    full_df = pl.concat(collected_data).sort("_timestamp")
    
    # Convert timestamp to Unix timestamp (float)
    timestamp_dt = full_df["_timestamp"].to_numpy()
    timestamp_unix = timestamp_dt.astype('datetime64[s]').astype(float)
    
    out = {
        "agent_id": agent_id,
        "n_points": len(full_df),
        "longitude_n": full_df[lon_n_col].to_numpy(),
        "latitude_n": full_df[lat_n_col].to_numpy(),
        "longitude": full_df[lon_col].to_numpy(),
        "latitude": full_df[lat_col].to_numpy(),
        "timestamp": timestamp_unix,
    }

    if include_error_range and err_col is not None:
        out["error_range"] = full_df[err_col].to_numpy()

    return out


def extract_single_trajectory(
    agent_id: Any,
    metadata: dict,
    target_length: int,
    max_gap_seconds: float = 1800.0,  # 30 minutes
    include_error_range: bool = False,
    column_map: Optional[Dict[str, str]] = None,
    agent_file_index: Optional[Dict[Any, List[Tuple[str, float, float]]]] = None,
) -> Optional[dict]:
    """
    Purpose:
        Extract one trajectory for given agent, stitching across files if needed.
    
    Parameters:
        agent_id (int): Target agent ID
        metadata (dict): Parquet metadata from scan_parquet_metadata
        target_length (int): Target number of points
        max_gap_seconds (float): Maximum time gap between files (default 1800s = 30min)
    
    Return:
        extracted_traj (dict | None): {
            "agent_id": int,
            "n_points": int,
            "longitude_n": np.ndarray,
            "latitude_n": np.ndarray,
            "longitude": np.ndarray,
            "latitude": np.ndarray,
            "timestamp": np.ndarray
        }
        Returns None if agent not found or no valid data.
    
    Notes:
        - Tries to stitch across multiple files if one file insufficient
        - Checks time gap between file boundaries (< 30min required)
        - Returns shorter trajectory if target_length not reachable
    """
    
    # Find all files containing this agent
    if agent_file_index is not None:
        agent_files = agent_file_index.get(agent_id, [])
    else:
        agent_files = find_agent_files(agent_id, metadata)
    
    if not agent_files:
        logger.debug(f"Agent {agent_id} not found in any file")
        return None
    
    # Start with earliest file
    collected_data = []
    total_points = 0
    last_end_time = None
    
    for pq_path, start_time, end_time in agent_files:
        if total_points >= target_length:
            break
        
        # Check time gap if not first file
        if last_end_time is not None:
            gap = start_time - last_end_time
            if gap > max_gap_seconds:
                logger.debug(f"Agent {agent_id}: gap {gap:.0f}s > {max_gap_seconds}s, stopping stitch")
                break
        
        # Load data from this file
        remaining = target_length - total_points
        traj = load_test_trajectory(
            agent_id,
            [pq_path],
            remaining,
            include_error_range=include_error_range,
            column_map=column_map,
        )
        
        if traj is None or traj['n_points'] == 0:
            continue
        
        collected_data.append(traj)
        total_points += traj['n_points']
        last_end_time = end_time
    
    if total_points == 0:
        return None
    
    # Merge all collected data
    if len(collected_data) == 1:
        return collected_data[0]
    
    # Concatenate multiple chunks
    merged = {
        "agent_id": agent_id,
        "n_points": total_points,
        "longitude_n": np.concatenate([t['longitude_n'] for t in collected_data]),
        "latitude_n": np.concatenate([t['latitude_n'] for t in collected_data]),
        "longitude": np.concatenate([t['longitude'] for t in collected_data]),
        "latitude": np.concatenate([t['latitude'] for t in collected_data]),
        "timestamp": np.concatenate([t['timestamp'] for t in collected_data])
    }

    if include_error_range:
        merged["error_range"] = np.concatenate([t['error_range'] for t in collected_data])
    
    return merged


def _coerce_fixed_targets(requested_m: int, requested_n: int, *, context: str) -> tuple[int, int]:
    if int(requested_m) != int(FIXED_TRAJ_COUNT) or int(requested_n) != int(FIXED_TRAJ_POINTS):
        logger.warning(
            "%s now uses fixed settings only: M=%d, N=%d. Ignoring requested M=%s, N=%s.",
            context,
            FIXED_TRAJ_COUNT,
            FIXED_TRAJ_POINTS,
            requested_m,
            requested_n,
        )
    return int(FIXED_TRAJ_COUNT), int(FIXED_TRAJ_POINTS)


def _insert_into_pool(
    trajectory_pool: List[Tuple[int, dict]],
    total_points: int,
    traj_dict: dict,
    target_total_points: int,
    target_m: int,
) -> tuple[bool, int]:
    del target_total_points
    n_points = int(traj_dict["n_points"])

    if len(trajectory_pool) < int(target_m):
        trajectory_pool.append((n_points, traj_dict))
        trajectory_pool.sort(key=lambda x: x[0], reverse=True)
        total_points = int(sum(t[0] for t in trajectory_pool))
        return True, total_points

    shortest_len = int(trajectory_pool[-1][0])
    # Pool full: strict replacement only if incoming trajectory is longer.
    if n_points <= shortest_len:
        return False, total_points

    trajectory_pool[-1] = (n_points, traj_dict)
    trajectory_pool.sort(key=lambda x: x[0], reverse=True)
    total_points = int(sum(t[0] for t in trajectory_pool))
    return True, total_points


def _pool_reached_target(trajectory_pool: List[Tuple[int, dict]], target_m: int, target_n: int) -> bool:
    if len(trajectory_pool) < int(target_m):
        return False
    return all(int(t[0]) >= int(target_n) for t in trajectory_pool)


def _extract_ranked_trajectory_pool(
    parquet_dir: str,
    M: int,
    N: int,
    *,
    include_error_range: bool = False,
    column_map: Optional[Dict[str, str]] = None,
) -> dict:
    _assert_not_calibration_or_processed_source(parquet_dir)
    column_map = _detect_column_map(parquet_dir, column_map)
    metadata = scan_parquet_metadata(parquet_dir, column_map=column_map)
    agent_file_index = build_agent_file_index(metadata)
    all_agents = list(agent_file_index.keys())
    if not all_agents:
        raise ValueError("No agents found in parquet directory")
    np.random.shuffle(all_agents)

    processor = data_processor_with_error_range if include_error_range else data_processor
    trajectory_pool: List[Tuple[int, dict]] = []
    total_points = 0
    target_total_points = int(M) * int(N)
    extraction_failures = 0
    agent_idx = 0

    while agent_idx < len(all_agents):
        agent_id = all_agents[agent_idx]
        agent_idx += 1

        if _pool_reached_target(trajectory_pool, M, N):
            logger.info(f"Early stop: all {M} trajectories reached length {N}")
            break

        extracted_traj = extract_single_trajectory(
            agent_id,
            metadata,
            int(N) * int(MAX_TRAJ_PER_AGENT),
            include_error_range=include_error_range,
            column_map=column_map,
            agent_file_index=agent_file_index,
        )
        if extracted_traj is None:
            extraction_failures += 1
            continue

        n_points = int(extracted_traj["n_points"])
        n_segments = min(int(MAX_TRAJ_PER_AGENT), int(n_points // int(N)))
        if n_segments == 0:
            extraction_failures += 1
            continue

        for seg_idx in range(n_segments):
            if _pool_reached_target(trajectory_pool, M, N):
                break
            start = seg_idx * int(N)
            end = start + int(N)
            seg_raw = _slice_raw_trajectory(
                extracted_traj,
                start,
                end,
                include_error_range=include_error_range,
            )
            processed = processor(seg_raw)
            _, total_points = _insert_into_pool(
                trajectory_pool,
                total_points,
                processed,
                target_total_points,
                M,
            )

    if not trajectory_pool:
        raise ValueError("No valid trajectories extracted")

    processed_trajectories = [t[1] for t in trajectory_pool]
    lengths = [int(t["n_points"]) for t in processed_trajectories]
    return {
        "processed_trajectories": processed_trajectories,
        "n_trajectories": int(len(processed_trajectories)),
        "total_points": int(sum(lengths)),
        "median_length": int(np.median(lengths)) if lengths else 0,
        "min_length": int(min(lengths)) if lengths else 0,
        "max_length": int(max(lengths)) if lengths else 0,
        "target_total_points": int(target_total_points),
        "extraction_failures": int(extraction_failures),
        "agents_sampled": int(agent_idx),
    }


def traj_extractor(
    parquet_dir: str,
    M: int,
    N: int = 60480,
    output_dir: str = "./dataset/processed/NUMOSIM_Kanto/test/traj_test/full_traj_10sec",
    column_map: Optional[Dict[str, str]] = None,
) -> dict:
    """
    Purpose:
        Extract M full trajectories from parquet directory.
        Each trajectory targets N points, stitches across files if needed.
        Keeps longest trajectories to ensure total points >= M*N.
        Processes and saves as PyTorch tensor file.
    
    Parameters:
        parquet_dir (str): Directory containing parquet files
        M (int): Number of trajectories to extract
        N (int): Target length per trajectory (default 60480 ≈ 7 days)
        output_dir (str): Output directory for processed trajectories
    
    Return:
        out_traj_extractor (dict): {
            "status": "completed",
            "n_trajectories": int,
            "total_points": int,
            "median_length": int,
            "output_file": str
        }
    
    Usage:
        Called to generate test set for BF/DF denoising comparison.
    
    TODO:
        1. Scan parquet directory and build agent metadata index
        2. Randomly sample agents WITHOUT REPLACEMENT (shuffled order)
        3. For each agent, extract trajectory with stitching logic
        4. Keep longest trajectories (priority queue/sorted list)
        5. Stop when M trajectories all have length N
        6. Process and save trajectories to PyTorch file
        7. Return statistics
    
    Strategy:
        - Maintain sorted list of trajectories by length (longest first)
        - If total_points >= M*N, replace shortest trajectory if new one is longer
        - Early stop: all M trajectories reach length N
        - Otherwise: continue until all agents exhausted
    
    Notes:
        - Guarantees total points >= M*N (best effort for M×N points)
        - Filename format: fulltraj_<n_trajectories>_<median_length>.pt
        - Each trajectory stored as dict with keys: agent_id, data, label, n_points
    """
    
    _log_thread_budget("traj_extractor")
    M, N = _coerce_fixed_targets(M, N, context="Trajectory extractor")
    logger.info(
        "Starting trajectory extraction: M=%d, N=%d, max_traj_per_agent=%d",
        M,
        N,
        MAX_TRAJ_PER_AGENT,
    )
    stats = _extract_ranked_trajectory_pool(
        parquet_dir=parquet_dir,
        M=M,
        N=N,
        include_error_range=False,
        column_map=column_map,
    )
    processed_trajectories = stats["processed_trajectories"]
    n_trajectories = int(stats["n_trajectories"])
    total_points = int(stats["total_points"])
    median_length = int(stats["median_length"])

    logger.info(f"Extraction complete: {n_trajectories} trajectories, {total_points} total points")
    logger.info(
        "Length stats - median: %d, min: %d, max: %d",
        int(stats["median_length"]),
        int(stats["min_length"]),
        int(stats["max_length"]),
    )
    logger.info(f"Extraction failures: {int(stats['extraction_failures'])}")
    logger.info(f"Agents sampled: {int(stats['agents_sampled'])}")
    if total_points >= int(stats["target_total_points"]):
        logger.info(f"✓ Target met: {total_points} >= {int(stats['target_total_points'])}")
    else:
        logger.warning(f"⚠ Target not met: {total_points} < {int(stats['target_total_points'])}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"fulltraj_{n_trajectories}_{median_length}.pt"
    output_file = output_path / filename
    
    # Convert to torch tensors for saving
    save_data = {
        "trajectories": [
            {
                "agent_id": t['agent_id'],
                "n_points": t['n_points'],
                "data": torch.tensor(t['data'], dtype=torch.float32),
                "label": torch.tensor(t['label'], dtype=torch.float32)
            }
            for t in processed_trajectories
        ],
        "metadata": {
            "n_trajectories": n_trajectories,
            "total_points": total_points,
            "median_length": median_length,
            "target_M": M,
            "target_N": N,
            "extraction_failures": int(stats["extraction_failures"]),
            "agents_sampled": int(stats["agents_sampled"]),
        }
    }
    
    torch.save(save_data, output_file)
    logger.debug(f"Saved trajectories to {output_file}")
    
    # ================================================================
    # 7. Return statistics
    # ================================================================
    return {
        "status": "completed",
        "n_trajectories": n_trajectories,
        "total_points": total_points,
        "median_length": median_length,
        "output_file": str(output_file)
    }


def traj_extractor_with_error_range(
    parquet_dir: str,
    M: int,
    N: int = 10000,
    output_dir: str = "./dataset/processed/NUMOSIM_Kanto/test/traj_test/full_traj_range",
    column_map: Optional[Dict[str, str]] = None,
) -> dict:
    """
    Purpose:
        Extract M full trajectories with error_range from parquet directory.
        This mirrors traj_extractor but keeps per-point error_range.
    """
    _log_thread_budget("traj_extractor_with_error_range")
    M, N = _coerce_fixed_targets(M, N, context="Trajectory extractor (error_range)")
    logger.info(
        "Starting trajectory extraction (error_range): M=%d, N=%d, max_traj_per_agent=%d",
        M,
        N,
        MAX_TRAJ_PER_AGENT,
    )
    stats = _extract_ranked_trajectory_pool(
        parquet_dir=parquet_dir,
        M=M,
        N=N,
        include_error_range=True,
        column_map=column_map,
    )
    processed_trajectories = stats["processed_trajectories"]
    n_trajectories = int(stats["n_trajectories"])
    total_points = int(stats["total_points"])
    median_length = int(stats["median_length"])

    logger.info(f"Extraction complete: {n_trajectories} trajectories, {total_points} total points")
    logger.info(
        "Length stats - median: %d, min: %d, max: %d",
        int(stats["median_length"]),
        int(stats["min_length"]),
        int(stats["max_length"]),
    )
    logger.info(f"Extraction failures: {int(stats['extraction_failures'])}")
    logger.info(f"Agents sampled: {int(stats['agents_sampled'])}")
    if total_points >= int(stats["target_total_points"]):
        logger.info(f"✓ Target met: {total_points} >= {int(stats['target_total_points'])}")
    else:
        logger.warning(f"⚠ Target not met: {total_points} < {int(stats['target_total_points'])}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = f"fulltraj_range_{n_trajectories}_{median_length}.pt"
    output_file = output_path / filename

    save_data = {
        "trajectories": [
            {
                "agent_id": t['agent_id'],
                "n_points": t['n_points'],
                "data": torch.tensor(t['data'], dtype=torch.float32),
                "label": torch.tensor(t['label'], dtype=torch.float32),
                "error_range": torch.tensor(t['error_range'], dtype=torch.float32),
                "timestamp": torch.tensor(t['timestamp'], dtype=torch.float32),
            }
            for t in processed_trajectories
        ],
        "metadata": {
            "n_trajectories": n_trajectories,
            "total_points": total_points,
            "median_length": median_length,
            "target_M": M,
            "target_N": N,
            "extraction_failures": int(stats["extraction_failures"]),
            "agents_sampled": int(stats["agents_sampled"]),
        }
    }

    torch.save(save_data, output_file)
    logger.info(f"Saved trajectories to {output_file}")

    return {
        "status": "completed",
        "n_trajectories": n_trajectories,
        "total_points": total_points,
        "median_length": median_length,
        "output_file": str(output_file)
    }


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Example usage: TEST SPLIT ONLY
    result = traj_extractor(
        parquet_dir="./dataset/raw/NUMOSIM_Kanto",
        M=200, # fixed internally
        N=5000, # fixed internally
        output_dir="./dataset/processed/NUMOSIM_Kanto/test/traj_test/full_traj_10sec"
    )
    
    print("\n" + "="*60)
    print("Trajectory Extraction Complete")
    print("="*60)
    print(f"Status: {result['status']}")
    print(f"Trajectories: {result['n_trajectories']}")
    print(f"Total points: {result['total_points']}")
    print(f"Median length: {result['median_length']}")
    print(f"Output file: {result['output_file']}")
    print("="*60)
