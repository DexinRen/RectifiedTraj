#!/usr/bin/env python3
"""
Extract M full trajectories of target length N from parquet directory.

Handles multi-file trajectory stitching with time continuity checks.
Processes and saves trajectories for BF/DF testing.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import polars as pl
import numpy as np
import torch
import logging
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)

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
        - Only uses TEST SPLIT (last 3 files when sorted), matching parquet_processor_test_only
        - Fast scanning: only queries unique agent IDs + min/max timestamp per agent
        - Skips corrupted files
    """
    
    logger.debug(f"Scanning parquet directory: {parquet_dir}")
    
    column_map = _detect_column_map(parquet_dir, column_map)
    agent_col = column_map["agent"]
    ts_col = column_map["timestamp"]
    lon_n_col = column_map["longitude_n"]
    lat_n_col = column_map["latitude_n"]
    lon_col = column_map["longitude"]
    lat_col = column_map["latitude"]

    # Get sorted parquet files and select TEST SPLIT ONLY (last 3 files)
    all_parquet_paths = sorted(Path(parquet_dir).glob("*.parquet"))
    logger.debug(f"Found {len(all_parquet_paths)} total parquet files")

    test_files = all_parquet_paths[-3:]
    logger.debug(f"Using TEST SPLIT: last {len(test_files)} file(s)")
    
    metadata = {}
    
    for pq_path in test_files:
        logger.debug(f"Scanning {pq_path.name}...")
        
        try:
            # FAST SCAN: Query aggregated data directly (much faster than filtering all rows)
            df = (
                pl.scan_parquet(pq_path)
                .with_columns(_timestamp_expr(ts_col))
                .filter(
                    pl.col(lon_n_col).is_not_null()
                    & pl.col(lat_n_col).is_not_null()
                    & pl.col(lon_col).is_not_null()
                    & pl.col(lat_col).is_not_null()
                    & pl.col(lon_n_col).is_finite()
                    & pl.col(lat_n_col).is_finite()
                    & pl.col(lon_col).is_finite()
                    & pl.col(lat_col).is_finite()
                    & pl.col("_timestamp").is_not_null()
                )
                .group_by(agent_col)
                .agg([
                    pl.col("_timestamp").min().alias("start_time"),
                    pl.col("_timestamp").max().alias("end_time"),
                ])
                .collect()
            )
            
            if len(df) == 0:
                logger.warning(f"No valid data in {pq_path.name}")
                continue
            
            # Build agent_ranges dict
            agent_ranges = {}
            for row in df.iter_rows(named=True):
                agent_id = row[agent_col]
                start_ts = row['start_time'].timestamp() if hasattr(row['start_time'], 'timestamp') else row['start_time'].astype('datetime64[s]').astype(float)
                end_ts = row['end_time'].timestamp() if hasattr(row['end_time'], 'timestamp') else row['end_time'].astype('datetime64[s]').astype(float)
                agent_ranges[agent_id] = (float(start_ts), float(end_ts))
            
            metadata[str(pq_path)] = agent_ranges
            logger.debug(f"Found {len(agent_ranges)} agents in {pq_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to scan {pq_path.name}: {e}")
            logger.warning(f"Skipping corrupted file: {pq_path.name}")
            continue
    
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

        df = (
            df.select([lon_n_col, lat_n_col, lon_col, lat_col, "_timestamp"] + ([err_col] if include_error_range and err_col is not None else []))
            .sort("_timestamp")
            .collect()
        )
        
        if len(df) == 0:
            continue
        
        # Take what we need
        remaining = n_points - total_collected
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


def traj_extractor(
    parquet_dir: str,
    M: int,
    N: int = 60480,
    output_dir: str = "./dataset/processed/full_traj",
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
    
    logger.info(f"Starting trajectory extraction: M={M}, N={N}")
    
    # ================================================================
    # 1. Scan parquet directory and build metadata index
    # ================================================================
    column_map = _detect_column_map(parquet_dir, column_map)
    metadata = scan_parquet_metadata(parquet_dir, column_map=column_map)
    
    # Collect all unique agent IDs across all files
    all_agents = set()
    for agent_ranges in metadata.values():
        all_agents.update(agent_ranges.keys())
    
    all_agents = list(all_agents)
    logger.info(f"Found {len(all_agents)} unique agents across all files")

    if len(all_agents) == 0:
        raise ValueError("No agents found in parquet directory")

    # RANDOM SAMPLING: Draw agents randomly without replacement
    # This ensures diverse coverage and avoids bias from sequential ordering
    np.random.shuffle(all_agents)
    logger.debug("Agents randomized for sampling (no repeats)")
    
    # ================================================================
    # 2. Extract and maintain longest trajectories
    # ================================================================
    # List of tuples: (n_points, processed_trajectory_dict)
    # Kept sorted by n_points descending (longest first)
    trajectory_pool = []
    total_points = 0
    target_total_points = M * N
    
    extraction_failures = 0
    agent_idx = 0
    
    def insert_trajectory(traj_dict):
        """Insert trajectory into sorted pool, maintaining M longest trajectories."""
        nonlocal trajectory_pool, total_points
        
        n_points = traj_dict['n_points']
        
        # If we have < M trajectories, always add
        if len(trajectory_pool) < M:
            trajectory_pool.append((n_points, traj_dict))
            trajectory_pool.sort(key=lambda x: x[0], reverse=True)  # Sort by length desc
            total_points = sum(t[0] for t in trajectory_pool)
            logger.debug(f"Added trajectory (pool size: {len(trajectory_pool)}/{M}, total: {total_points})")
            return True
        
        # If pool full and total_points < target, replace shortest if new is longer
        if total_points < target_total_points:
            shortest_len = trajectory_pool[-1][0]
            if n_points > shortest_len:
                # Replace shortest
                trajectory_pool[-1] = (n_points, traj_dict)
                trajectory_pool.sort(key=lambda x: x[0], reverse=True)
                total_points = sum(t[0] for t in trajectory_pool)
                logger.debug(f"Replaced shortest ({shortest_len}) with longer ({n_points}), total: {total_points}")
                return True
            else:
                logger.debug(f"Rejected trajectory ({n_points} <= {shortest_len})")
                return False
        
        # If total_points >= target, only replace if new trajectory is longer than shortest
        shortest_len = trajectory_pool[-1][0]
        if n_points > shortest_len:
            trajectory_pool[-1] = (n_points, traj_dict)
            trajectory_pool.sort(key=lambda x: x[0], reverse=True)
            total_points = sum(t[0] for t in trajectory_pool)
            logger.debug(f"Improved pool: replaced {shortest_len} with {n_points}, total: {total_points}")
            return True
        
        return False
    
    def check_early_stop():
        """Check if all M trajectories have reached length N."""
        if len(trajectory_pool) < M:
            return False
        return all(t[0] >= N for t in trajectory_pool)
    
    # ================================================================
    # 3. Sample agents and extract trajectories
    # ================================================================
    while agent_idx < len(all_agents):
        agent_id = all_agents[agent_idx]
        agent_idx += 1
        
        # Early stop: all M trajectories are full length N
        if check_early_stop():
            logger.info(f"Early stop: all {M} trajectories reached length {N}")
            break
        
        logger.debug(f"Extracting trajectory for agent {agent_id} ({agent_idx}/{len(all_agents)})...")
        
        # Extract raw trajectory
        extracted_traj = extract_single_trajectory(
            agent_id,
            metadata,
            N,
            include_error_range=False,
            column_map=column_map,
        )
        
        if extracted_traj is None:
            logger.debug(f"Agent {agent_id}: extraction failed")
            extraction_failures += 1
            continue
        
        n_points = extracted_traj['n_points']
        logger.debug(f"Agent {agent_id}: extracted {n_points} points (target: {N})")
        
        # Process trajectory to model format
        processed = data_processor(extracted_traj)
        
        # Try to insert into pool
        insert_trajectory(processed)
        
        # Log progress every 10 agents
        if agent_idx % 10 == 0:
            logger.info(f"Progress: {agent_idx}/{len(all_agents)} agents sampled, pool: {len(trajectory_pool)}/{M}, total: {total_points}/{target_total_points}")
    
    # ================================================================
    # 4. Final check
    # ================================================================
    if len(trajectory_pool) == 0:
        raise ValueError("No valid trajectories extracted")
    
    # Extract processed trajectories from pool
    processed_trajectories = [t[1] for t in trajectory_pool]
    
    # ================================================================
    # 5. Compute statistics
    # ================================================================
    n_trajectories = len(processed_trajectories)
    lengths = [t['n_points'] for t in processed_trajectories]
    median_length = int(np.median(lengths))
    total_points = sum(lengths)
    
    logger.info(f"Extraction complete: {n_trajectories} trajectories, {total_points} total points")
    logger.info(f"Length stats - median: {median_length}, min: {min(lengths)}, max: {max(lengths)}")
    logger.info(f"Extraction failures: {extraction_failures}")
    logger.info(f"Agents sampled: {agent_idx}/{len(all_agents)}")
    
    # Check if target met
    if total_points >= target_total_points:
        logger.info(f"✓ Target met: {total_points} >= {target_total_points}")
    else:
        logger.warning(f"⚠ Target not met: {total_points} < {target_total_points}")
    
    # ================================================================
    # 6. Save processed trajectories
    # ================================================================
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
            "extraction_failures": extraction_failures,
            "agents_sampled": agent_idx
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
    output_dir: str = "./dataset/processed/full_traj_range",
    column_map: Optional[Dict[str, str]] = None,
) -> dict:
    """
    Purpose:
        Extract M full trajectories with error_range from parquet directory.
        This mirrors traj_extractor but keeps per-point error_range.
    """
    logger.info(f"Starting trajectory extraction (error_range): M={M}, N={N}")

    column_map = _detect_column_map(parquet_dir, column_map)
    metadata = scan_parquet_metadata(parquet_dir, column_map=column_map)

    all_agents = set()
    for agent_ranges in metadata.values():
        all_agents.update(agent_ranges.keys())

    all_agents = list(all_agents)
    logger.debug(f"Found {len(all_agents)} unique agents across all files")

    if len(all_agents) == 0:
        raise ValueError("No agents found in parquet directory")

    np.random.shuffle(all_agents)
    logger.info(f"Agents randomized for sampling (no repeats)")

    trajectory_pool = []
    total_points = 0
    target_total_points = M * N

    extraction_failures = 0
    agent_idx = 0

    def insert_trajectory(traj_dict):
        nonlocal trajectory_pool, total_points

        n_points = traj_dict['n_points']

        if len(trajectory_pool) < M:
            trajectory_pool.append((n_points, traj_dict))
            trajectory_pool.sort(key=lambda x: x[0], reverse=True)
            total_points = sum(t[0] for t in trajectory_pool)
            return True

        if total_points < target_total_points:
            shortest_len = trajectory_pool[-1][0]
            if n_points > shortest_len:
                trajectory_pool[-1] = (n_points, traj_dict)
                trajectory_pool.sort(key=lambda x: x[0], reverse=True)
                total_points = sum(t[0] for t in trajectory_pool)
                return True
            return False

        shortest_len = trajectory_pool[-1][0]
        if n_points > shortest_len:
            trajectory_pool[-1] = (n_points, traj_dict)
            trajectory_pool.sort(key=lambda x: x[0], reverse=True)
            total_points = sum(t[0] for t in trajectory_pool)
            return True

        return False

    def check_early_stop():
        if len(trajectory_pool) < M:
            return False
        return all(t[0] >= N for t in trajectory_pool)

    while agent_idx < len(all_agents):
        agent_id = all_agents[agent_idx]
        agent_idx += 1

        if check_early_stop():
            logger.info(f"Early stop: all {M} trajectories reached length {N}")
            break

        logger.info(f"Extracting trajectory for agent {agent_id} ({agent_idx}/{len(all_agents)})...")

        extracted_traj = extract_single_trajectory(
            agent_id,
            metadata,
            N,
            include_error_range=True,
            column_map=column_map,
        )

        if extracted_traj is None:
            extraction_failures += 1
            continue

        n_points = extracted_traj['n_points']
        logger.info(f"Agent {agent_id}: extracted {n_points} points (target: {N})")

        processed = data_processor_with_error_range(extracted_traj)
        insert_trajectory(processed)

    if len(trajectory_pool) == 0:
        raise ValueError("No valid trajectories extracted")

    processed_trajectories = [t[1] for t in trajectory_pool]

    n_trajectories = len(processed_trajectories)
    lengths = [t['n_points'] for t in processed_trajectories]
    median_length = int(np.median(lengths))
    total_points = sum(lengths)

    logger.info(f"Extraction complete: {n_trajectories} trajectories, {total_points} total points")
    logger.debug(f"Length stats - median: {median_length}, min: {min(lengths)}, max: {max(lengths)}")
    logger.debug(f"Extraction failures: {extraction_failures}")
    logger.debug(f"Agents sampled: {agent_idx}/{len(all_agents)}")

    if total_points >= target_total_points:
        logger.info(f"✓ Target met: {total_points} >= {target_total_points}")
    else:
        logger.warning(f"⚠ Target not met: {total_points} < {target_total_points}")

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
            "extraction_failures": extraction_failures,
            "agents_sampled": agent_idx
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
        parquet_dir="./dataset/raw",
        M=20, # number of trajectories
        N=60480, # length of trajectories, 60480 = 7 days at minimum
        output_dir="./dataset/processed/full_traj"
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
