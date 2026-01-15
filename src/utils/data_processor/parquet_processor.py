"""
Parquet Processor for Trajectory Denoising Pipeline

This module processes large parquet datasets containing GPS trajectory data,
dicing them into overlapping chunks for rectified flow model training.

Key features:
- File-by-file processing with garbage collection
- Lazy loading with Polars
- Incremental Huber delta estimation
- ENU coordinate transformation
- Chunk generation with overlap (buckle points)

TODO list:
    - Implement each stage as standalone function.
    - Maintain logger injection.
    - Follow return-dict pattern with explicit keys per code.style.txt.
"""

import os
import gc
import json
import math
import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np
import polars as pl
import torch
import glob
from pymap3d import geodetic2enu


# Configure logger
logger = logging.getLogger(__name__)


def ds_dicer(ds_entry: dict, K: int = 256, Q: int = 1) -> Tuple[dict, dict]:
    """
    Purpose:
        Randomly sample ~50 users in one dataset and dice their full
        trajectories into K-point chunks with overlap Q. Stop when a
        total of 5000 chunks are generated.

    Parameters:
        ds_entry (dict): {"name": str, "ds": polars.LazyFrame}
        K (int): chunk size (default 256)
        Q (int): overlap size (default 1)

    Return:
        out_ds_dicer (dict): {usr_id: [(chunk_id, row_start, row_end), ...]}
        ds_record (dict): {"ds_name": str, "users": {usr_id: end_row, ...}}

    Notes:
        - First Q points of each chunk = last Q points of previous chunk
        - For first chunk of a user: duplicate first point Q times
        - Updates ./dataset/state/ds_records.jsonl
    """
    logger.info(f"Dicing dataset: {ds_entry['name']}")
    
    usr_num = 200
    chunk_num = 10000
    ds_name = ds_entry['name']
    ds = ds_entry['ds']
    
    # Load state directory
    state_dir = Path("./dataset/state")
    state_dir.mkdir(parents=True, exist_ok=True)
    records_path = state_dir / "ds_records.jsonl"
    
    # Load existing records to check starting points
    existing_records = {}
    with open(records_path, 'r') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if record['ds_name'] == ds_name:
                    existing_records = record.get('users', {})
                    break
    
    # Collect the data (lazy -> eager) with error handling
    try:
        df = ds.collect()
    except Exception as e:
        logger.error(f"Failed to collect data from {ds_name}: {e}")
        logger.warning(f"Skipping corrupted file: {ds_name}")
        # Return empty results to skip this file
        return {}, {"ds_name": ds_name, "users": {}}
    
    # Get unique user IDs
    all_users = df['agent'].unique().to_list()
    logger.info(f"Found {len(all_users)} unique users in dataset")
    
    # Randomly sample ~50 users
    n_users = min(usr_num, len(all_users))
    sampled_users = np.random.choice(all_users, size=n_users, replace=False)
    logger.info(f"Sampled {n_users} users for processing")
    
    out_ds_dicer = {}
    total_chunks = 0
    chunk_counter = 0
    ds_record_users = {}
    
    # Process each sampled user
    for usr_id in sampled_users:
        if total_chunks >= chunk_num:
            break
        
        # Filter data for this user, remove NaN/null, and sort by timestamp
        # Note: Polars distinguishes between null and NaN - we need to filter both
        n_points_before = len(df.filter(pl.col('agent') == usr_id))
        
        user_df = df.filter(
            (pl.col('agent') == usr_id) &
            pl.col('longitude_n').is_not_null() &
            pl.col('latitude_n').is_not_null() &
            pl.col('longitude').is_not_null() &
            pl.col('latitude').is_not_null() &
            pl.col('longitude_n').is_finite() &
            pl.col('latitude_n').is_finite() &
            pl.col('longitude').is_finite() &
            pl.col('latitude').is_finite()
        ).sort('timestamp')
        
        n_points = len(user_df)
        
        if n_points < n_points_before:
            logger.info(f"User {usr_id}: Filtered out {n_points_before - n_points} NaN/null points, {n_points} valid points remaining")
        
        # CRITICAL: Verify no NaN made it through
        if n_points > 0:
            lon_n_array = user_df['longitude_n'].to_numpy()
            lat_n_array = user_df['latitude_n'].to_numpy()
            lon_array = user_df['longitude'].to_numpy()
            lat_array = user_df['latitude'].to_numpy()
            
            has_nan = (np.any(~np.isfinite(lon_n_array)) or np.any(~np.isfinite(lat_n_array)) or
                      np.any(~np.isfinite(lon_array)) or np.any(~np.isfinite(lat_array)))
            
            if has_nan:
                logger.error(f"❌ BUG: User {usr_id} still has NaN after Polars filter!")
                logger.error(f"   lon_n NaN count: {np.sum(~np.isfinite(lon_n_array))}")
                logger.error(f"   lat_n NaN count: {np.sum(~np.isfinite(lat_n_array))}")
                logger.error(f"   lon NaN count: {np.sum(~np.isfinite(lon_array))}")
                logger.error(f"   lat NaN count: {np.sum(~np.isfinite(lat_array))}")
                logger.error(f"   Polars filter is NOT working correctly!")
                continue
        
        # Check if user has enough valid points
        if n_points < K:
            logger.debug(f"User {usr_id}: Only {n_points} valid points (need {K}), skipping user")
            continue
        
        # Determine starting row
        start_row = existing_records.get(str(usr_id), 0)
        
        if start_row >= n_points - K:
            logger.debug(f"User {usr_id} already fully processed, skipping")
            continue
        
        chunks_for_user = []
        current_row = start_row
        
        # Generate chunks with overlap
        # First chunk starts at start_row
        # Subsequent chunks overlap by Q points with previous chunk
        while current_row + K <= n_points and total_chunks < chunk_num:
            row_start = current_row
            row_end = current_row + K - 1  # inclusive
            
            chunks_for_user.append((chunk_counter, row_start, row_end))
            chunk_counter += 1
            total_chunks += 1
            
            # Move to next chunk: overlap by Q points
            # Next chunk starts at (current_row + K - Q)
            current_row += (K - Q)
        
        if chunks_for_user:
            out_ds_dicer[usr_id] = chunks_for_user
            # Record the last row index that was included in a chunk
            ds_record_users[str(usr_id)] = chunks_for_user[-1][2]  # last row_end
            logger.debug(f"User {usr_id}: generated {len(chunks_for_user)} chunks")
    
    logger.info(f"Total chunks generated: {total_chunks}")
    
    # Prepare ds_record
    ds_record = {
        "ds_name": ds_name,
        "users": ds_record_users
    }
    
    # Update ds_records.jsonl
    # Read all existing records
    all_records = []
    existing_ds_record = None
    if records_path.exists():
        with open(records_path, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    # Keep old record for this dataset for merging
                    if record['ds_name'] == ds_name:
                        existing_ds_record = record
                    else:
                        all_records.append(record)
    
    # Merge with existing record: preserve largest end_row per user
    if existing_ds_record:
        for usr_id, end_row in existing_ds_record.get('users', {}).items():
            # Only update if existing end_row is larger
            if usr_id not in ds_record_users or end_row > ds_record_users.get(usr_id, -1):
                ds_record_users[usr_id] = end_row
        
        # Update ds_record with merged users
        ds_record = {
            "ds_name": ds_name,
            "users": ds_record_users
        }
    
    # Add the new/updated record
    all_records.append(ds_record)
    
    # Write back
    with open(records_path, 'w') as f:
        for record in all_records:
            f.write(json.dumps(record) + '\n')
    
    logger.info(f"Updated ds_records.jsonl for {ds_name}")
    
    return out_ds_dicer, ds_record


def ds_assemble(ds_entry: dict, 
                usr_chunks: dict, 
                K: int = 256, 
                Q: int = 1) -> List[dict]:
    """
    Purpose:
        Build chunk pairs from diced trajectory segments.
        Extract [longitude_n, latitude_n, longitude, latitude, timestamp] and mark buckle points.

    Parameters:
        ds_entry (dict): {"name": str, "ds": polars.LazyFrame}
        usr_chunks (dict): {usr_id: [(chunk_id, row_start, row_end), ...]}
        K (int): chunk size (default 256)
        Q (int): overlap size (default 1)

    Return:
        out_ds_assemble (list[dict]): [
            {
                "usr_id": int,
                "chunk_id": int,
                "X1": [[longitude_n, latitude_n, timestamp, is_start], ...],
                "X0": [[longitude, latitude, timestamp, is_start], ...]
            },
            ...
        ]

    Notes:
        - First Q points have is_start = True
        - For first chunk only: duplicate first point Q times
        - Other chunks: use natural overlap from ds_dicer
    """
    logger.info(f"Assembling chunks from {ds_entry['name']}")
    
    ds_name = ds_entry['name']
    ds = ds_entry['ds']
    
    # Collect the full dataframe with error handling
    try:
        df = ds.collect()
    except Exception as e:
        logger.error(f"Failed to collect data from {ds_name}: {e}")
        logger.warning(f"Skipping corrupted file in assembly: {ds_name}")
        return []  # Return empty list to skip this file
    
    out_ds_assemble = []
    
    for usr_id, chunks in usr_chunks.items():
        # Get all data for this user
        user_df = df.filter(
            (pl.col('agent') == usr_id) &
            pl.col('longitude_n').is_not_null() &
            pl.col('latitude_n').is_not_null() &
            pl.col('longitude').is_not_null() &
            pl.col('latitude').is_not_null() &
            pl.col('longitude_n').is_finite() &
            pl.col('latitude_n').is_finite() &
            pl.col('longitude').is_finite() &
            pl.col('latitude').is_finite()
        ).sort('timestamp')
        
        for idx, (chunk_id, row_start, row_end) in enumerate(chunks):
            is_first_chunk = (idx == 0)
            
            # Extract the chunk slice (inclusive of row_end)
            chunk_df = user_df[row_start:row_end + 1]
            
            # Verify we have K points
            if len(chunk_df) != K:
                logger.warning(f"Chunk {chunk_id} has {len(chunk_df)} points, expected {K}. Skipping.")
                continue
            
            # Extract columns and make copies (Polars arrays are read-only)
            longitude_n = chunk_df['longitude_n'].to_numpy().copy()
            latitude_n = chunk_df['latitude_n'].to_numpy().copy()
            longitude = chunk_df['longitude'].to_numpy().copy()
            latitude = chunk_df['latitude'].to_numpy().copy()
            
            # Convert timestamp to Unix timestamp (seconds since epoch as float)
            timestamp_dt = chunk_df['timestamp'].to_numpy()
            timestamp = timestamp_dt.astype('datetime64[s]').astype(float)
            
            # Handle first chunk: duplicate the first point Q times
            if is_first_chunk:
                # Replace first Q points with duplicates of the very first point
                longitude_n[:Q] = longitude_n[0]
                latitude_n[:Q] = latitude_n[0]
                longitude[:Q] = longitude[0]
                latitude[:Q] = latitude[0]
                timestamp[:Q] = timestamp[0]
            
            # Mark buckle points (first Q points have is_start = True)
            is_start = np.zeros(K, dtype=bool)
            is_start[:Q] = True
            
            # Build X1 and X0 arrays
            X1 = np.stack([longitude_n, latitude_n, timestamp, is_start.astype(float)], axis=1)
            X0 = np.stack([longitude, latitude, timestamp, is_start.astype(float)], axis=1)
            
            one_raw_chunk = {
                "usr_id": usr_id,
                "chunk_id": chunk_id,
                "X1": X1.tolist(),
                "X0": X0.tolist()
            }
            
            out_ds_assemble.append(one_raw_chunk)
    
    logger.info(f"Assembled {len(out_ds_assemble)} chunks")
    return out_ds_assemble


def enu_transform(one_raw_chunk: dict) -> dict:
    """
    Purpose:
        Transform GPS coordinates (longitude, latitude) to local ENU coordinates.
        Use the first point as the origin (0, 0) for both X0 and X1.

    Parameters:
        one_raw_chunk (dict): {
            "usr_id": int,
            "chunk_id": int,
            "X1": [[longitude_n, latitude_n, timestamp, is_start], ...],
            "X0": [[longitude, latitude, timestamp, is_start], ...]
        }

    Return:
        out_enu_transform (dict): {
            "usr_id": int,
            "chunk_id": int,
            "z": {"lon0": float, "lat0": float, "lon1": float, "lat1": float},
            "chunk_enu": {
                "X1": [[e, n, timestamp, is_start], ...],
                "X0": [[e, n, timestamp, is_start], ...]
            }
        }

    Notes:
        - z stores GPS reference points for recovery
        - X0 and X1 are transformed independently
    """
    X1 = np.array(one_raw_chunk['X1'])
    X0 = np.array(one_raw_chunk['X0'])
    
    # Extract coordinates
    lon1, lat1 = X1[:, 0], X1[:, 1]
    lon0, lat0 = X0[:, 0], X0[:, 1]
    
    # Reference points (first point of each)
    lon1_ref, lat1_ref = lon1[0], lat1[0]
    lon0_ref, lat0_ref = lon0[0], lat0[0]
    
    # Transform to ENU (assuming h=0 for all points)
    # 46 is the altitude above ellipsoid in downtown LA
    # this is the height-projection adjustment on the 2D distance 
    e1, n1, _ = geodetic2enu(lat1, lon1, 46, lat1_ref, lon1_ref, 0)
    e0, n0, _ = geodetic2enu(lat0, lon0, 46, lat1_ref, lon1_ref, 0)
    # we use X1's first point as both X1 and X0 (Y)'s enu reference point
    # so they are in the same map, without adding extra noise caused by 
    # two different coordination system centered by two different point
    # and also avoid expose label info (since we use X1 as reference)
    
    # CRITICAL: Check if ENU transformation produced NaN/Inf
    
    # Build transformed arrays (keep timestamp and is_start)
    X1_enu = np.stack([e1, n1, X1[:, 2], X1[:, 3]], axis=1)
    X0_enu = np.stack([e0, n0, X0[:, 2], X0[:, 3]], axis=1)
    
    out_enu_transform = {
        "usr_id": one_raw_chunk['usr_id'],
        "chunk_id": one_raw_chunk['chunk_id'],
        "z": {
            "lon0": float(lon0_ref),
            "lat0": float(lat0_ref),
            "lon1": float(lon1_ref),
            "lat1": float(lat1_ref)
        },
        "chunk_enu": {
            "X1": X1_enu.tolist(),
            "X0": X0_enu.tolist()
        }
    }
    
    return out_enu_transform


def v_labelizer(enu_chunk: dict) -> dict:
    """
    Purpose:
        Compute velocity field label V = X1 - X0 (coordinates only).

    Parameters:
        enu_chunk (dict): {
            "usr_id": int,
            "chunk_id": int,
            "z": {...},
            "chunk_enu": {
                "X1": [[e, n, timestamp, is_start], ...],
                "X0": [[e, n, timestamp, is_start], ...]
            }
        }

    Return:
        out_v_labelizer (dict): {
            "usr_id": int,
            "chunk_id": int,
            "z": {...},
            "chunk_enu": {
                "X0": [[e, n, timestamp, is_start], ...],
                "V": [[ve, vn], ...]  # coordinates only
            }
        }
    """
    X1 = np.array(enu_chunk['chunk_enu']['X1'])
    X0 = np.array(enu_chunk['chunk_enu']['X0'])
    
    # Compute velocity (coordinates only: e and n)
    V = X1[:, :2] - X0[:, :2]
    
    out_v_labelizer = {
        "usr_id": enu_chunk['usr_id'],
        "chunk_id": enu_chunk['chunk_id'],
        "z": enu_chunk['z'],
        "chunk_enu": {
            "X0": X0.tolist(),
            "V": V.tolist()
        }
    }
    
    return out_v_labelizer

def huber_delta_estimator(enu_transform_list: List[dict]) -> dict:
    """
    Purpose:
        Incrementally estimate Huber loss threshold δ using the chunks
        produced by enu_transform. Computes displacement magnitudes in ENU
        coordinates: epsilon = sqrt((E1 - E0)^2 + (N1 - N0)^2).

    Parameters:
        enu_transform_list (list[dict]): List of out_enu_transform chunks with
            structure: {
                "usr_id": int,
                "chunk_id": int,
                "z": {...},
                "chunk_enu": {
                    "X1": [[e, n, timestamp, is_start], ...],
                    "X0": [[e, n, timestamp, is_start], ...]
                }
            }

    Return:
        out_huber_delta_estimator (dict): {
            "status": "appended",
            "n_samples": int
        }

    Notes:
        - Uses chunk (X0, X1) data directly from enu_transform
        - Computes epsilon for ALL POINTS in all chunks (point-wise)
        - Delta is calculated per-point, not per-chunk
        - Writes to ./dataset/state/huber_samples.tmp
        - Call finalize_huber_delta() after all training files processed
    """
    state_dir = Path("./dataset/state")
    state_dir.mkdir(parents=True, exist_ok=True)
    samples_path = state_dir / "huber_samples.tmp"
    
    epsilon_values = []
    
    # Process all chunks and all points
    for chunk in enu_transform_list:
        X0 = np.array(chunk['chunk_enu']['X0'])
        X1 = np.array(chunk['chunk_enu']['X1'])
        
        # Extract ENU coordinates (first 2 columns: e, n)
        E0 = X0[:, 0]
        N0 = X0[:, 1]
        E1 = X1[:, 0]
        N1 = X1[:, 1]
        
        # Compute epsilon = sqrt((E1 - E0)^2 + (N1 - N0)^2) for each point
        eps = np.sqrt((E1 - E0)**2 + (N1 - N0)**2)
        
        # Filter out NaN and Inf values
        eps_valid = eps[np.isfinite(eps)]
        if len(eps_valid) < len(eps):
            logger.warning(f"Filtered out {len(eps) - len(eps_valid)} non-finite epsilon values in chunk {chunk['chunk_id']}")
        
        epsilon_values.extend(eps_valid.tolist())
    
    # Append to temporary file
    with open(samples_path, 'a') as f:
        for eps in epsilon_values:
            f.write(f"{eps}\n")
    
    logger.info(f"Appended {len(epsilon_values)} epsilon samples to huber_samples.tmp")
    
    return {
        "status": "appended",
        "n_samples": len(epsilon_values)
    }

def finalize_huber_delta() -> dict:
    """
    Purpose:
        Compute global Huber delta from accumulated samples.
        δ = Q3 + 1.5 × IQR

    Return:
        out_finalize (dict): {
            "delta": float,
            "q1": float,
            "q3": float,
            "iqr": float
        }

    Notes:
        - Reads ./dataset/state/huber_samples.tmp
        - Writes result to ./dataset/state/huber_delta.json
        - Deletes temporary file after processing
    """
    state_dir = Path("./dataset/state")
    samples_path = state_dir / "huber_samples.tmp"
    delta_path = state_dir / "huber_delta.json"
    
    if not samples_path.exists():
        logger.warning("No huber_samples.tmp found. Returning default delta=1.0")
        return {"delta": 1.0, "q1": 0.0, "q3": 1.0, "iqr": 1.0}
    
    # Load all epsilon values
    epsilon_values = []
    with open(samples_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    val = float(line.strip())
                    # Filter out NaN and Inf values
                    if np.isfinite(val):
                        epsilon_values.append(val)
                except ValueError:
                    logger.warning(f"Skipping invalid epsilon value: {line.strip()}")
    
    logger.info(f"Loaded {len(epsilon_values)} epsilon samples from file")
    
    # Check if we have any valid values BEFORE converting to numpy
    if len(epsilon_values) == 0:
        logger.error("No valid epsilon values found! Returning default delta=1.0")
        return {"delta": 1.0, "q1": 0.0, "q3": 1.0, "iqr": 1.0}
    
    # Convert to numpy array
    epsilon_values = np.array(epsilon_values)
    logger.info(f"Epsilon stats - min: {np.min(epsilon_values):.6f}, max: {np.max(epsilon_values):.6f}, mean: {np.mean(epsilon_values):.6f}")
    
    # Compute quartiles
    q1 = np.percentile(epsilon_values, 25)
    q3 = np.percentile(epsilon_values, 75)
    iqr = q3 - q1
    delta = q3 + 1.5 * iqr
    
    # Validate results
    if not np.isfinite(delta):
        logger.error(f"Computed delta is not finite! Q1={q1}, Q3={q3}, IQR={iqr}")
        logger.error("Returning default delta=1.0")
        return {"delta": 1.0, "q1": 0.0, "q3": 1.0, "iqr": 1.0}
    
    result = {
        "delta": float(delta),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr)
    }
    
    # Save to JSON
    with open(delta_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Huber delta finalized: δ={delta:.6f} (Q1={q1:.6f}, Q3={q3:.6f}, IQR={iqr:.6f})")
    
    # Keep temporary file for debugging (don't delete)
    # samples_path.unlink()
    logger.info(f"Kept huber_samples.tmp for debugging at {samples_path}")
    logger.info(f"File contains {len(epsilon_values)} epsilon values (one per line)")
    
    return resul

def t_sampler(v_labelizer_chunk: dict, r: int = 5) -> List[dict]:
    """
    Purpose:
        Sample r independent time values t_i ∈ [0, 1) and create
        training records {X_t, t, V}.

    Parameters:
        v_labelizer_chunk (dict): {
            "usr_id": int,
            "chunk_id": int,
            "z": {...},
            "chunk_enu": {
                "X0": [[e, n, timestamp, is_start], ...],
                "V": [[ve, vn], ...]
            }
        }
        r (int): number of time samples per chunk (default 5)

    Return:
        training_samples (list[dict]): [
            {
                "usr_id": int,
                "chunk_id": int,
                "z": {...},
                "X_t": [[e, n, timestamp, is_start], ...],
                "t": float,
                "V": [[ve, vn], ...]
            },
            ...
        ]
    """
    X0 = np.array(v_labelizer_chunk['chunk_enu']['X0'])
    V = np.array(v_labelizer_chunk['chunk_enu']['V'])
    
    training_samples = []
    
    # Sample r time values
    t_values = np.random.uniform(0, 1, size=r)
    
    for t in t_values:
        # Compute X_t = X0 + t * V (only for coordinates)
        X_t_coords = X0[:, :2] + t * V
        
        # Reconstruct X_t with timestamp and is_start
        X_t = np.concatenate([X_t_coords, X0[:, 2:]], axis=1)
        
        sample = {
            "usr_id": v_labelizer_chunk['usr_id'],
            "chunk_id": v_labelizer_chunk['chunk_id'],
            "z": v_labelizer_chunk['z'],
            "X_t": X_t.tolist(),
            "t": float(t),
            "V": V.tolist()
        }
        
        training_samples.append(sample)
    
    return training_samples

def parquet_processor(K: int = 256, 
                     Q: int = 1, 
                     r: int = 5,
                     raw_ds_path: str = "./dataset/raw") -> dict:
   
    raw_path = Path(raw_ds_path)
    processed_path = Path("./dataset/processed")
    
    for split in ['train', 'val', 'test']:
        (processed_path / split).mkdir(parents=True, exist_ok=True)
    
    # Get all parquet files and sort by name
    parquet_files = sorted(raw_path.glob("*.parquet"))
    logger.info(f"Found {len(parquet_files)} parquet files")
    
    # Define splits
    train_files = parquet_files[:26]
    val_files = parquet_files[26:29]
    test_files = parquet_files[29:32]
    
    logger.info(f"Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    # Statistics
    chunk_counts = {
        "train": 0,
        "val": 0,
        "test": 0
    }
    
    corrupted_files = []  # Track corrupted files
    
    # Process each split
    for split_name, file_list in [("train", train_files), 
                                    ("val", val_files), 
                                    ("test", test_files)]:
        logger.info(f"Processing {split_name} split ({len(file_list)} files)")
        
        # Process each file individually
        for file_idx, file_path in enumerate(file_list):
            logger.info(f"Processing file {file_idx + 1}/{len(file_list)}: {file_path.name}")
            
            # Step 1: Read single file
            ds_result = ds_reader([str(file_path)])
            ds_entry = ds_result['datasets'][0]
            
            # Step 2: Dice trajectories
            out_ds_dicer, ds_record = ds_dicer(ds_entry, K=K, Q=Q)
            
            if not out_ds_dicer:
                logger.warning(f"No chunks generated from {file_path.name}")
                # Check if this was due to corruption by seeing if ds_record is empty
                if not ds_record.get('users'):
                    corrupted_files.append(file_path.name)
                gc.collect()
                continue
            
            # Step 3: Assemble chunks
            out_ds_assemble = ds_assemble(ds_entry, out_ds_dicer, K=K, Q=Q)
            logger.info(f"Assembled {len(out_ds_assemble)} chunks from {file_path.name}")
            
            if len(out_ds_assemble) == 0:
                logger.warning(f"No valid chunks assembled from {file_path.name} (all skipped due to NaN GPS data)")
                gc.collect()
                continue
            
            # Step 4: ENU transform
            out_enu_transform = [enu_transform(chunk) for chunk in out_ds_assemble]
            logger.info(f"ENU transformed {len(out_enu_transform)} chunks")
            
            # Step 5: Huber delta estimation (training split only)
            if split_name == "train":
                huber_result = huber_delta_estimator(out_enu_transform)
                logger.info(f"Huber samples: {huber_result['n_samples']}")
            
            # Step 6: V labelizer
            out_v_labelizer = [v_labelizer(chunk) for chunk in out_enu_transform]
            
            # Step 7: Time sampling
            all_training_samples = []
            for chunk in out_v_labelizer:
                samples = t_sampler(chunk, r=r)
                all_training_samples.extend(samples)
            
            # === Step 8: Save tensorized dataset ===
            N = len(all_training_samples)
            if N == 0:
                logger.warning(f"No training samples to save for {file_path.name}")
                continue

            K = len(all_training_samples[0]["X_t"])  # usually 256

            # allocate tensors
            X_t = torch.empty((N, K, 4), dtype=torch.float32)
            V   = torch.empty((N, K, 2), dtype=torch.float32)
            t   = torch.empty((N, 1), dtype=torch.float32)

            # fill tensors
            for i, s in enumerate(all_training_samples):
                X_t[i] = torch.tensor(s["X_t"], dtype=torch.float32)
                V[i]   = torch.tensor(s["V"], dtype=torch.float32)
                t[i, 0] = s["t"]

            tensor_pack = {"X_t": X_t, "V": V, "t": t}

            output_file = processed_path / split_name / f"chunks_{ds_entry['name'].replace('.parquet', '')}.pt"
            torch.save(tensor_pack, output_file)
            with open(f"{output_file}.len", "w") as f:
                f.write(str(N))

            logger.info(f"Saved tensor dataset {tensor_pack['X_t'].shape} -> {output_file}")
        
        # Finalize Huber delta after training split
        if split_name == "train":
            logger.info("Finalizing Huber delta computation")
            delta_result = finalize_huber_delta()
            logger.info(f"Final Huber delta: {delta_result['delta']:.6f}")
    
    # Final summary
    out_parquet_processor = {
        "status": "completed",
        "train_files": len(train_files),
        "val_files": len(val_files),
        "test_files": len(test_files),
        "total_chunks": chunk_counts,
        "corrupted_files": corrupted_files
    }
    
    logger.info("Parquet processing pipeline completed successfully")
    logger.info(f"Total chunks - Train: {chunk_counts['train']}, "
                f"Val: {chunk_counts['val']}, Test: {chunk_counts['test']}")
    
    if corrupted_files:
        logger.warning(f"Skipped {len(corrupted_files)} corrupted files: {corrupted_files}")
        
        # Save corrupted files list to ./dataset/state/corrupted_files.json
        state_dir = Path("./dataset/state")
        state_dir.mkdir(parents=True, exist_ok=True)
        corrupted_path = state_dir / "corrupted_files.json"
        
        with open(corrupted_path, 'w') as f:
            json.dump({
                "total_corrupted": len(corrupted_files),
                "files": corrupted_files,
                "timestamp": str(Path(__file__).stat().st_mtime)
            }, f, indent=2)
        
        logger.info(f"Saved corrupted files list to {corrupted_path}")
    
    return out_parquet_processor

def sample_quick_val(val_dir="./dataset/processed/val", sample_size=2000, output_path="./dataset/quick_val.pt"):
    val_files = [f for f in glob.glob(os.path.join(val_dir, "*.pt")) if not f.endswith(".len")]
    if not val_files:
        raise FileNotFoundError(f"[sample_quick_val] No .pt files found in {val_dir}")

    X_t_all, V_all, t_all = [], [], []
    for f in val_files:
        data = torch.load(f)
        X_t_all.append(data["X_t"])
        V_all.append(data["V"])
        t_all.append(data["t"])
    X_t = torch.cat(X_t_all, dim=0)
    V = torch.cat(V_all, dim=0)
    t = torch.cat(t_all, dim=0)

    n = len(X_t)
    idx = torch.randperm(n)[:sample_size]
    sample = {"X_t": X_t[idx], "V": V[idx], "t": t[idx]}
    torch.save(sample, output_path)
    print(f"[sample_quick_val] Saved {sample_size} samples to {output_path}")
    return sample



if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the pipeline
    result = parquet_processor(
        K=256,
        Q=1,
        r=5,
        raw_ds_path="./dataset/raw"
    )
    sample_quick_val()
    print("\n" + "="*50)
    print("Pipeline completed!")
    print(f"Status: {result['status']}")
    print(f"Files processed: {result['train_files']} train, "
          f"{result['val_files']} val, {result['test_files']} test")
    print(f"Total chunks: {result['total_chunks']}")
    if result.get('corrupted_files'):
        print(f"Corrupted files skipped: {len(result['corrupted_files'])}")
        print(f"  {result['corrupted_files']}")
    print("="*50)

def ds_reader(ds_path) -> dict:
    """
    Purpose:
        Lazily read multiple Parquet datasets using the Polars package
        and return their pointers as a list for later chunk processing.

    Parameters:
        ds_path_list (list[str]):
            List of dataset file paths under ./dataset/raw.

    Return Dict:
        "dataset": {"name": str (filename), "ds": polars.LazyFrame pointer}
    """
    logger.info(f"Loading {len(ds_path)} datasets with lazy evaluation")
    # Lazy load the parquet file
    ds = pl.scan_parquet(ds_path)
    return ds

def parquet_spliter(runtime):
    # Setup directories
    chunk_demand = 100000 # 99 % train, 1 % quick val = 
    chunk_count  = 0
    raw_path = Path(runtime["raw_ds_dir"])

    processed_path = Path("./dataset/processed")
    
    for split in ['train', 'val', 'test']:
        (processed_path / split).mkdir(parents=True, exist_ok=True)
    
    # Get all parquet files and sort by name
    parquet_files = sorted(raw_path.glob("*.parquet"))

    # train/val/test split interval
    unit_fold = (parquet_files.len() / 100) 
    fold_train_bound = int(math.ceil(unit_fold * 70))
    fold_val_bound   = fold_train_bound + int(math.ceil(unit_fold * 15))
    
    fold_train = parquet_files[ : fold_train_bound]
    fold_val   = parquet_files[fold_train_bound + 1 : fold_val_bound] 
    fold_test  = parquet_files[fold_val_bound   + 1 : ]

    logger.info(f"Found {len(parquet_files)} parquet files")

    for ds_path in parquet_files:
        # Step 1: Read single file
        ds_entry = pl.scan_parquet(ds_path)
        
        # Step 2: Dice trajectories
        out_ds_assemble = ds_dicer(ds_entry, K=K, Q=Q)
        
        
        # Step 4: ENU transform
        out_enu_transform = [enu_transform(chunk) for chunk in out_ds_assemble]
        
        # Step 5: Huber delta estimation (training split only)
        if ds_path in fold_train:
            huber_result = huber_delta_estimator(out_enu_transform)
        
        # Step 6: V labelizer
        out_v_labelizer = [v_labelizer(chunk) for chunk in out_enu_transform]
        
        # Step 7: Time sampling
        all_training_samples = []
        for chunk in out_v_labelizer:
            samples = t_sampler(chunk, r=r)
            all_training_samples.extend(samples)
        
        # === Step 8: Save tensorized dataset ===
        N = len(all_training_samples)
        if N == 0:
            logger.warning(f"No training samples to save for {file_path.name}")
            continue

        K = len(all_training_samples[0]["X_t"])  # usually 256

        # allocate tensors
        X_t = torch.empty((N, K, 4), dtype=torch.float32)
        V   = torch.empty((N, K, 2), dtype=torch.float32)
        t   = torch.empty((N, 1), dtype=torch.float32)

        # fill tensors
        for i, s in enumerate(all_training_samples):
            X_t[i] = torch.tensor(s["X_t"], dtype=torch.float32)
            V[i]   = torch.tensor(s["V"], dtype=torch.float32)
            t[i, 0] = s["t"]

        tensor_pack = {"X_t": X_t, "V": V, "t": t}

        output_file = processed_path / split_name / f"chunks_{ds_entry['name'].replace('.parquet', '')}.pt"
        torch.save(tensor_pack, output_file)
        with open(f"{output_file}.len", "w") as f:
            f.write(str(N))

        logger.info(f"Saved tensor dataset {tensor_pack['X_t'].shape} -> {output_file}")
        
        # Finalize Huber delta after training split
        if split_name == "train":
            logger.info("Finalizing Huber delta computation")
            delta_result = finalize_huber_delta()
            logger.info(f"Final Huber delta: {delta_result['delta']:.6f}")
    
    # Final summary
    out_parquet_processor = {
        "status": "completed",
        "train_files": len(train_files),
        "val_files": len(val_files),
        "test_files": len(test_files),
        "total_chunks": chunk_counts,
        "corrupted_files": corrupted_files
    }
    
    logger.info("Parquet processing pipeline completed successfully")
    logger.info(f"Total chunks - Train: {chunk_counts['train']}, "
                f"Val: {chunk_counts['val']}, Test: {chunk_counts['test']}")
    
    if corrupted_files:
        logger.warning(f"Skipped {len(corrupted_files)} corrupted files: {corrupted_files}")
        
        # Save corrupted files list to ./dataset/state/corrupted_files.json
        state_dir = Path("./dataset/state")
        state_dir.mkdir(parents=True, exist_ok=True)
        corrupted_path = state_dir / "corrupted_files.json"
        
        with open(corrupted_path, 'w') as f:
            json.dump({
                "total_corrupted": len(corrupted_files),
                "files": corrupted_files,
                "timestamp": str(Path(__file__).stat().st_mtime)
            }, f, indent=2)
        
        logger.info(f"Saved corrupted files list to {corrupted_path}")
    
    return out_parquet_processor

def parquet_spliter(runtime):
    for pq in runtime[""]
    all_users = df['agent'].unique().to_list()



def make(runtime = None):
    if runtime == None: 
        runtime = {
            "K":256, 
            "Q":1,
            "usr_num": 100,
            "size_per_usr": 100,
            "raw_ds_dir":"./ds/raw"
            }
    parquet_loader(runtime)  # runtime["raw_ds_pt"] = {"train", "val", "test"}
    parquet_spliter(runtime) # split & save on the same time






    

