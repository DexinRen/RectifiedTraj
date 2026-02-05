"""
Parquet Processor for Trajectory Denoising Pipeline

This module processes large parquet datasets containing GPS trajectory data,
dicing them into overlapping chunks for rectified flow model training.

Key features:
- File-by-file processing with garbage collection
- Lazy loading with Polars
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
import hashlib
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


def ds_reader(ds_path_list) -> dict:
    """
    Purpose:
        Lazily read multiple Parquet datasets using the Polars package
        and return their pointers as a list for later chunk processing.

    Parameters:
        ds_path_list (list[str]):
            List of dataset file paths under ./dataset/raw.

    Return Dict:
        "datasets": [{"name": str (filename), "ds": polars.LazyFrame pointer}, ...]
    """
    logger.info(f"Loading {len(ds_path_list)} datasets with lazy evaluation")
    datasets = []
    for ds_path in ds_path_list:
        ds = pl.scan_parquet(ds_path)
        datasets.append({"name": Path(ds_path).name, "ds": ds})
    return {"datasets": datasets}


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
            
            # Step 5: V labelizer
            out_v_labelizer = [v_labelizer(chunk) for chunk in out_enu_transform]
            
            # Step 6: Time sampling
            all_training_samples = []
            for chunk in out_v_labelizer:
                samples = t_sampler(chunk, r=r)
                all_training_samples.extend(samples)
            
            # === Step 7: Save tensorized dataset ===
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


def parquet_processor_test_only(
    K: int = 256,
    Q: int = 1,
    r: int = 5,
    raw_ds_path: str = "./dataset/raw",
    test_files: list[str] | None = None,
    output_dir: str = "./dataset/processed/test",
) -> dict:
    """
    Generate test-only chunk datasets (preserves timestamps in X_t).

    Args:
        test_files: list of parquet filenames (or full paths). If None, use all *.parquet in raw_ds_path.
        output_dir: output directory for test chunks.
    """
    raw_path = Path(raw_ds_path)
    processed_path = Path(output_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    if test_files:
        file_list = [Path(p) for p in test_files]
        file_list = [p if p.is_absolute() else raw_path / p for p in file_list]
    else:
        parquet_files = sorted(raw_path.glob("*.parquet"))
        file_list = parquet_files[-3:]

    logger.info(f"Processing test-only split ({len(file_list)} files)")

    chunk_count = 0
    corrupted_files = []

    for file_idx, file_path in enumerate(file_list):
        logger.info(f"Processing file {file_idx + 1}/{len(file_list)}: {file_path.name}")

        ds_result = ds_reader([str(file_path)])
        ds_entry = ds_result['datasets'][0]

        out_ds_dicer, ds_record = ds_dicer(ds_entry, K=K, Q=Q)
        if not out_ds_dicer:
            logger.warning(f"No chunks generated from {file_path.name}")
            if not ds_record.get('users'):
                corrupted_files.append(file_path.name)
            gc.collect()
            continue

        out_ds_assemble = ds_assemble(ds_entry, out_ds_dicer, K=K, Q=Q)
        logger.info(f"Assembled {len(out_ds_assemble)} chunks from {file_path.name}")

        if len(out_ds_assemble) == 0:
            logger.warning(f"No valid chunks assembled from {file_path.name} (all skipped due to NaN GPS data)")
            gc.collect()
            continue

        out_enu_transform = [enu_transform(chunk) for chunk in out_ds_assemble]
        logger.info(f"ENU transformed {len(out_enu_transform)} chunks")

        out_v_labelizer = [v_labelizer(chunk) for chunk in out_enu_transform]

        all_training_samples = []
        for chunk in out_v_labelizer:
            samples = t_sampler(chunk, r=r)
            all_training_samples.extend(samples)

        N = len(all_training_samples)
        if N == 0:
            logger.warning(f"No training samples to save for {file_path.name}")
            continue

        K = len(all_training_samples[0]["X_t"])
        # Keep X_t in float64 to preserve timestamp resolution at epoch scale.
        X_t = torch.empty((N, K, 4), dtype=torch.float64)
        V = torch.empty((N, K, 2), dtype=torch.float32)
        t = torch.empty((N, 1), dtype=torch.float32)

        for i, s in enumerate(all_training_samples):
            X_t[i] = torch.tensor(s["X_t"], dtype=torch.float64)
            V[i] = torch.tensor(s["V"], dtype=torch.float32)
            t[i, 0] = s["t"]

        tensor_pack = {"X_t": X_t, "V": V, "t": t}
        output_file = processed_path / f"chunks_{ds_entry['name'].replace('.parquet', '')}.pt"
        torch.save(tensor_pack, output_file)
        with open(f"{output_file}.len", "w") as f:
            f.write(str(N))

        chunk_count += int(N)
        logger.info(f"Saved tensor dataset {tensor_pack['X_t'].shape} -> {output_file}")

    out_parquet_processor = {
        "status": "completed",
        "test_files": len(file_list),
        "total_chunks": {"test": chunk_count},
        "corrupted_files": corrupted_files,
    }

    if corrupted_files:
        logger.warning(f"Skipped {len(corrupted_files)} corrupted files: {corrupted_files}")

    return out_parquet_processor

def _format_size_suffix(n: int) -> str:
    if n % 1_000_000 == 0:
        return f"{n // 1_000_000}m"
    if n % 1_000 == 0:
        return f"{n // 1_000}k"
    return str(n)


def _hash_tensor_en(x: torch.Tensor) -> bytes:
    x = x.contiguous()
    return hashlib.sha1(x.numpy().tobytes()).digest()


def _sample_is_valid(X_t: torch.Tensor, V: torch.Tensor, t: torch.Tensor) -> bool:
    if torch.isnan(X_t).any() or torch.isinf(X_t).any():
        return False
    if torch.isnan(V).any() or torch.isinf(V).any():
        return False
    if torch.isnan(t).any() or torch.isinf(t).any():
        return False
    is_start_vals = torch.unique(X_t[:, 3])
    for v in is_start_vals:
        if v.item() not in (0.0, 1.0):
            return False
    return True


def build_quick_val_sets(
    val_dir="./dataset/processed/val",
    output_dir="./dataset",
    small_size=10_000,
    mid_size=50_000,
    big_size=100_000,
    seed=42,
):
    """
    Build three non-overlapping quick-val sets from val shards.
    Files are named: quick_val_small.pt, quick_val.pt (mid), quick_val_big.pt
    """
    val_files = [f for f in glob.glob(os.path.join(val_dir, "*.pt")) if not f.endswith(".len")]
    if not val_files:
        raise FileNotFoundError(f"[build_quick_val_sets] No .pt files found in {val_dir}")

    X_t_all, V_all, t_all = [], [], []
    for f in sorted(val_files):
        data = torch.load(f, map_location="cpu")
        X_t_all.append(data["X_t"])
        V_all.append(data["V"])
        t_all.append(data["t"])

    X_t = torch.cat(X_t_all, dim=0)
    V = torch.cat(V_all, dim=0)
    t = torch.cat(t_all, dim=0)

    n = len(X_t)
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()

    targets = [
        ("small", small_size),
        ("mid", mid_size),
        ("big", big_size),
    ]
    selected = {name: [] for name, _ in targets}
    hash_set = set()

    for idx in perm:
        X_s = X_t[idx]
        V_s = V[idx]
        t_s = t[idx]

        if not _sample_is_valid(X_s, V_s, t_s):
            continue

        h = _hash_tensor_en(X_s[:, :2])
        if h in hash_set:
            continue

        for name, size in targets:
            if len(selected[name]) < size:
                selected[name].append(idx)
                hash_set.add(h)
                break

        if all(len(selected[name]) >= size for name, size in targets):
            break

    for name, size in targets:
        if len(selected[name]) < size:
            logger.warning(
                f"[build_quick_val_sets] Insufficient samples for {name}: {len(selected[name])} < {size}. "
                f"Using available samples."
            )

    def _save(name: str, size: int):
        idx_tensor = torch.tensor(selected[name], dtype=torch.long)
        pack = {
            "X_t": X_t.index_select(0, idx_tensor),
            "V": V.index_select(0, idx_tensor),
            "t": t.index_select(0, idx_tensor),
        }
        if name == "mid":
            out_name = "quick_val.pt"
        else:
            out_name = f"quick_val_{name}.pt"
        out_path = os.path.join(output_dir, out_name)
        torch.save(pack, out_path)
        print(f"[build_quick_val_sets] Saved {name} set: {out_path} (N={len(idx_tensor)})")

    _save("small", small_size)
    _save("mid", mid_size)
    _save("big", big_size)


def shuffle_train_pt_pairwise(train_dir="./dataset/processed/train", seed=42):
    """
    RAM-safe shuffle: load two train files at a time, shuffle, split back.
    Pairing uses a "least-mated" policy and runs for 2 * ceil(log2(N)) rounds.
    NOTE: Does NOT touch val/test directories.
    """
    train_files = [f for f in glob.glob(os.path.join(train_dir, "*.pt")) if not f.endswith(".len")]
    if not train_files:
        raise FileNotFoundError(f"[shuffle_train_pt_pairwise] No .pt files found in {train_dir}")

    files = sorted(train_files)
    n_files = len(files)
    rounds = int(2 * math.ceil(math.log2(n_files))) if n_files > 1 else 0

    # Track how often each pair has been mixed.
    mate_counts = {f: {} for f in files}

    def _mate_count(a, b):
        return mate_counts[a].get(b, 0)

    def _record_mate(a, b):
        mate_counts[a][b] = _mate_count(a, b) + 1
        mate_counts[b][a] = _mate_count(b, a) + 1

    # Greedy "least-mated" pairing with deterministic tie-break.
    for r in range(rounds):
        remaining = set(files)
        if len(remaining) % 2 == 1:
            # Rotate the "bye" file to avoid starving one file of mixing.
            total_mates = {f: sum(mate_counts[f].values()) for f in remaining}
            bye = max(sorted(remaining), key=lambda x: (total_mates[x], x))
            remaining.remove(bye)

        while len(remaining) > 1:
            a = sorted(remaining)[0]
            remaining.remove(a)

            # Choose b with the smallest mate count with a (tie-break by name).
            b = min(remaining, key=lambda x: (_mate_count(a, x), x))
            remaining.remove(b)

            d1 = torch.load(a, map_location="cpu")
            d2 = torch.load(b, map_location="cpu")

            n1 = d1["X_t"].shape[0]
            n2 = d2["X_t"].shape[0]

            X_t = torch.cat([d1["X_t"], d2["X_t"]], dim=0)
            V = torch.cat([d1["V"], d2["V"]], dim=0)
            t = torch.cat([d1["t"], d2["t"]], dim=0)

            g_pair = torch.Generator()
            g_pair.manual_seed(seed + r * 1000 + n1 + n2)
            perm = torch.randperm(n1 + n2, generator=g_pair)

            X_t = X_t[perm]
            V = V[perm]
            t = t[perm]

            pack1 = {"X_t": X_t[:n1], "V": V[:n1], "t": t[:n1]}
            pack2 = {"X_t": X_t[n1:], "V": V[n1:], "t": t[n1:]}

            torch.save(pack1, a)
            torch.save(pack2, b)
            with open(f"{a}.len", "w") as lf1:
                lf1.write(str(n1))
            with open(f"{b}.len", "w") as lf2:
                lf2.write(str(n2))

            _record_mate(a, b)

        # If odd file count, last file is left as-is for this round.

    print(f"[shuffle_train_pt_pairwise] Shuffled train files with {rounds} rounds (seed={seed})")



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
    shuffle_train_pt_pairwise(train_dir="./dataset/processed/train", seed=42)
    build_quick_val_sets()
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

    
