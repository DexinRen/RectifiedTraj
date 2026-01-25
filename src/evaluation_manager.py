#!/usr/bin/env python3
"""
evaluation_manager.py

PURPOSE:
    Orchestrate model evaluation with multiple evaluator types.
    - TrajectoryEvaluator: final trajectory denoising quality + timing
    - InTrainingEvaluator: quick validation during training (placeholder)
    - RegionalEvaluator: Q1/Q2 buckle accuracy analysis (placeholder)

STRUCTURE:
    - EvaluationManager: orchestration class
    - TrajectoryEvaluator: full trajectory evaluation
    - Placeholder evaluators for future implementation

USAGE:
    manager = EvaluationManager(output_dir="test_results")
    manager.run_trajectory_evaluation()  # all models
    manager.run_trajectory_evaluation(model_names=["model_A"])  # specific models
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch


# ================================================================
# TRAJECTORY EVALUATOR
# ================================================================
class TrajectoryEvaluator:
    """
    Purpose:
        Evaluate trajectory denoising quality at multiple granularities.
        
    Outputs:
        - Parquet: detailed results with byte/chunk-wise lists
        - CSV: point-wise summary for quick comparison
        
    Granularities:
        - Point-wise (pw): individual point L2 errors
        - Byte-wise (bw): 8-point groups
        - Chunk-wise (cw): per-chunk errors
    """
    
    def __init__(self, output_dir: str = "test_results"):
        """
        Purpose:
            Initialize evaluator with output directories.
            
        Parameters:
            output_dir (str): root directory for test results
            
        Return Dict:
            N/A (constructor)
            
        TODO:
            1. Create output directories
            2. Initialize logger
            3. Set up CSV file with headers
        """
        # 1. Create directories
        self.output_dir = Path(output_dir)
        self.parquet_dir = self.output_dir / "trajectory_evaluation_results"
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_path = self.output_dir / "trajectory_evaluation_summary.csv"
        
        # 2. Initialize logger
        self.logger = logging.getLogger("TrajectoryEvaluator")
        
        # 3. Create CSV with headers if doesn't exist
        if not self.csv_path.exists():
            header = (
                "model_name,denoise_method,avg_l2_err_pw,med_l2_err_pw,std_l2_err_pw,"
                "avg_denoise_time_sec,num_tested_trajectories,num_tested_points,test_timestamp\n"
            )
            self.csv_path.write_text(header)
    
    def evaluate_model(
        self,
        model_name: str,
        model_dir: str,
        checkpoint_name: str,
        denoise_method: str,
        test_trajectories: List,
        K: int = 256,
        Q1: int = 2,
        Q2: int = 2,
    ) -> Dict:
        """
        Purpose:
            Run full trajectory evaluation for one model + method combination.
            
        Parameters:
            model_name (str): model identifier
            model_dir (str): absolute path to model directory
            checkpoint_name (str): checkpoint filename (e.g., "best_ckpt_step_50000.pt")
            denoise_method (str): "BF" or "DF"
            test_trajectories (List): list of trajectory objects with noisy/clean data
            K (int): chunk size, default 256
            Q1 (int): head buckle size in bytes, default 2
            Q2 (int): tail buckle size in bytes, default 2
            
        Return Dict:
            "error_code": 0 (success) | -1 (error)
            "results": dict with all computed metrics | None
            
        TODO:
            1. Load model checkpoint
            2. Run denoising on all test trajectories
            3. Compute point-wise metrics
            4. Compute byte-wise metrics
            5. Compute chunk-wise metrics
            6. Measure timing on longest trajectory
            7. Assemble results dictionary
            8. Save to parquet and CSV
            9. Return results
        """
        self.logger.info(f"Evaluating {model_name} with {denoise_method}")
        
        # 1. Load model (placeholder - needs actual model loading logic)
        try:
            model = self._load_model(model_dir, checkpoint_name)
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return {"error_code": -1, "results": None}
        
        # 2. Run denoising (placeholder - needs actual denoising logic)
        try:
            denoised_trajectories, errors = self._denoise_trajectories(
                model, test_trajectories, denoise_method
            )
        except Exception as e:
            self.logger.error(f"Failed to denoise trajectories: {e}")
            return {"error_code": -1, "results": None}
        
        # 3. Compute point-wise metrics
        pw_metrics = self._compute_pointwise_metrics(errors)
        
        # 4. Compute byte-wise metrics
        bw_metrics = self._compute_bytewise_metrics(test_trajectories, errors)
        
        # 5. Compute chunk-wise metrics
        cw_metrics = self._compute_chunkwise_metrics(test_trajectories, errors, K, Q1, Q2)
        
        # 6. Measure timing
        longest_traj = max(test_trajectories, key=lambda t: len(t))
        avg_time = self._measure_timing(model, longest_traj, denoise_method)
        
        # 7. Assemble results
        results = {
            # Metadata
            "model_name": model_name,
            "model_dir": model_dir,
            "checkpoint_name": checkpoint_name,
            "K": K,
            "Q1": Q1,
            "Q2": Q2,
            "denoise_method": denoise_method,
            "test_timestamp": datetime.now().isoformat(),
            "num_tested_trajectories": len(test_trajectories),
            "num_tested_points": sum(len(t) for t in test_trajectories),
            "longest_trajectory_length": len(longest_traj),
            
            # Point-wise metrics
            "avg_l2_err_pw": pw_metrics["avg"],
            "med_l2_err_pw": pw_metrics["med"],
            "std_l2_err_pw": pw_metrics["std"],
            
            # Byte-wise data
            "avg_l2_err_bw": bw_metrics["avg_list"],
            "avg_l2_err_bw_norm": bw_metrics["avg_list_norm"],
            
            # Chunk-wise data
            "avg_l2_err_cw": cw_metrics["avg_list"],
            "avg_l2_err_cw_norm": cw_metrics["avg_list_norm"],
            
            # Timing
            "avg_denoise_time_sec": avg_time,
        }
        
        # 8. Save results
        save_result = self._save_results(results)
        if save_result["error_code"] != 0:
            return {"error_code": -1, "results": None}
        
        # 9. Return
        self.logger.info(f"Evaluation complete: {model_name} {denoise_method}")
        return {"error_code": 0, "results": results}
    
    def _load_model(self, model_dir: str, checkpoint_name: str):
        """
        Purpose:
            Load model checkpoint from disk.
            
        Parameters:
            model_dir (str): path to model directory
            checkpoint_name (str): checkpoint filename
            
        Return Dict:
            "error_code": 0 | -1
            "model": loaded model | None
            
        TODO:
            1. Read config.json from model_dir/log/
            2. Build model architecture
            3. Load checkpoint state_dict
            4. Return model in eval mode
        """
        # 1. Read config from model directory
        config_path = Path(model_dir) / "log" / "config.json"
        if not config_path.exists():
            self.logger.error(f"Config not found: {config_path}")
            return {"error_code": -1, "model": None}
        
        with open(config_path, "r") as f:
            cfg = json.load(f)
        
        # 2. Build model architecture (needs theta_model import)
        try:
            # This import assumes theta_model.py is in the same directory or PYTHONPATH
            # TODO: Adjust import path based on your project structure
            from theta_model import build_theta
            
            model_result = build_theta(
                model_type=cfg.get("model_type", "nn"),
                hidden=cfg.get("hidden", 512),
                layers=cfg.get("layers", 6),
                K=cfg.get("K", 256),
                dropout=cfg.get("dropout", 0.1)
            )
            model = model_result["model"]
            
        except ImportError as e:
            self.logger.error(f"Failed to import theta_model: {e}")
            return {"error_code": -1, "model": None}
        except Exception as e:
            self.logger.error(f"Failed to build model: {e}")
            return {"error_code": -1, "model": None}
        
        # 3. Load checkpoint state_dict
        # Try best_ckpt directory first, then ckpts
        ckpt_path = None
        for ckpt_dir_name in ["best_ckpt", "ckpts"]:
            ckpt_dir = Path(model_dir) / ckpt_dir_name
            if ckpt_dir.exists():
                test_path = ckpt_dir / checkpoint_name
                if test_path.exists():
                    ckpt_path = test_path
                    break
        
        if ckpt_path is None:
            self.logger.error(f"Checkpoint not found: {checkpoint_name}")
            return {"error_code": -1, "model": None}
        
        try:
            blob = torch.load(ckpt_path, map_location="cpu")
            
            # Load weights
            if isinstance(blob, dict) and "model_state_dict" in blob:
                model.load_state_dict(blob["model_state_dict"], strict=True)
            else:
                self.logger.error(f"Checkpoint format not recognized: missing model_state_dict")
                return {"error_code": -1, "model": None}
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return {"error_code": -1, "model": None}
        
        # 4. Return model in eval mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        
        self.logger.info(f"Loaded model from {checkpoint_name}")
        return {"error_code": 0, "model": model}
    
    def _denoise_trajectories(self, model, test_trajectories: List, method: str):
        """
        Purpose:
            Run denoising on all trajectories and compute errors.
            
        Parameters:
            model: loaded model in eval mode
            test_trajectories (List): trajectories with noisy/clean data
            method (str): "BF" or "DF"
            
        Return Dict:
            "error_code": 0 | -1
            "denoised": list of denoised trajectories | None
            "errors": numpy array of point-wise L2 errors | None
            
        TODO:
            1. For each trajectory, run denoising
            2. Compute L2 error per point
            3. Aggregate all errors into single array
            4. Return denoised trajectories and errors
        """
        # Validate method
        if method not in ["BF", "DF"]:
            self.logger.error(f"Invalid denoise method: {method}")
            return {"error_code": -1, "denoised": None, "errors": None}
        
        try:
            # This import assumes encoder_decoder.py is available
            # TODO: Adjust import path based on your project structure
            from encoder_decoder import EncoderDecoder
            
            # Initialize encoder-decoder with model
            # TODO: Get K, Q1, Q2 from config or pass as parameters
            decoder = EncoderDecoder(model=model, K=256, Q1=2, Q2=2)
            
        except ImportError as e:
            self.logger.error(f"Failed to import EncoderDecoder: {e}")
            return {"error_code": -1, "denoised": None, "errors": None}
        
        denoised_trajectories = []
        all_errors = []
        
        for idx, traj_obj in enumerate(test_trajectories):
            # TODO: Adjust based on your trajectory object structure
            # Assuming traj_obj has:
            #   - traj_obj.noisy_gps: (T, 2) noisy GPS trajectory
            #   - traj_obj.clean_gps: (T, 2) ground truth GPS trajectory
            
            try:
                noisy_gps = traj_obj.noisy_gps  # (T, 2) [lon, lat]
                clean_gps = traj_obj.clean_gps  # (T, 2) [lon, lat]
                
                # Run denoising based on method
                if method == "BF":
                    result = decoder.denoise_traj_BF(noisy_gps)
                else:  # DF
                    result = decoder.denoise_traj_DF(noisy_gps)
                
                if result["error_code"] != 0:
                    self.logger.warning(f"Denoising failed for trajectory {idx}")
                    continue
                
                denoised_gps = result["traj_clean"]  # (T', 2)
                
                # Align lengths (denoising may change length due to buckle stripping)
                T_denoised = len(denoised_gps)
                clean_gps_aligned = clean_gps[-T_denoised:]
                
                # Convert to ENU for error calculation (using first clean point as origin)
                ref_lat, ref_lon = float(clean_gps_aligned[0, 1]), float(clean_gps_aligned[0, 0])
                enu_denoised = self._gps_to_enu_batch(denoised_gps, ref_lat, ref_lon)
                enu_clean = self._gps_to_enu_batch(clean_gps_aligned, ref_lat, ref_lon)
                
                # Compute L2 errors per point (in meters)
                errors = np.linalg.norm(enu_denoised - enu_clean, axis=1)
                
                denoised_trajectories.append(denoised_gps)
                all_errors.append(errors)
                
            except Exception as e:
                self.logger.error(f"Error processing trajectory {idx}: {e}")
                continue
        
        if len(all_errors) == 0:
            self.logger.error("No trajectories successfully denoised")
            return {"error_code": -1, "denoised": None, "errors": None}
        
        # Concatenate all errors into single array
        all_errors_array = np.concatenate(all_errors, axis=0)
        
        self.logger.info(f"Denoised {len(denoised_trajectories)} trajectories with method {method}")
        return {
            "error_code": 0,
            "denoised": denoised_trajectories,
            "errors": all_errors_array
        }
    
    def _gps_to_enu_batch(self, gps_coords: np.ndarray, ref_lat: float, ref_lon: float) -> np.ndarray:
        """
        Purpose:
            Convert GPS coordinates to ENU (East-North-Up) in meters.
            
        Parameters:
            gps_coords (np.ndarray): (N, 2) array of [lon, lat]
            ref_lat (float): reference latitude
            ref_lon (float): reference longitude
            
        Return Dict:
            (N, 2) array of [east, north] in meters
            
        TODO:
            1. Use pymap3d for conversion
            2. Return ENU coordinates
        """
        try:
            import pymap3d as pm
        except ImportError:
            self.logger.error("pymap3d not installed, cannot convert GPS to ENU")
            raise
        
        lons = gps_coords[:, 0]
        lats = gps_coords[:, 1]
        
        # Convert to ENU (assume altitude = 0)
        e, n, u = pm.geodetic2enu(lats, lons, 0, ref_lat, ref_lon, 0)
        
        return np.stack([e, n], axis=1)  # (N, 2)
    
    def _compute_pointwise_metrics(self, errors: np.ndarray) -> Dict:
        """
        Purpose:
            Compute mean, median, std of point-wise L2 errors.
            
        Parameters:
            errors (np.ndarray): 1D array of L2 errors for all points
            
        Return Dict:
            "avg": float
            "med": float
            "std": float
            
        TODO:
            1. Compute mean
            2. Compute median
            3. Compute std
            4. Return dict
        """
        return {
            "avg": float(np.mean(errors)),
            "med": float(np.median(errors)),
            "std": float(np.std(errors)),
        }
    
    def _compute_bytewise_metrics(self, trajectories: List, errors: np.ndarray) -> Dict:
        """
        Purpose:
            Aggregate errors into bytes (8-point groups).
            
        Parameters:
            trajectories (List): original trajectories
            errors (np.ndarray): point-wise errors (same order as concatenated trajectories)
            
        Return Dict:
            "avg_list": list[float] of per-byte average errors
            "avg_list_norm": list[float] normalized version
            
        TODO:
            1. Build pw_list (per-point average across all trajectories)
            2. Group pw_list into bytes (8 points each)
            3. Compute average error per byte
            4. Normalize the list (divide by mean)
            5. Return both lists
        """
        # 1. Build pw_list - average error at each point position across all trajectories
        # Find longest trajectory length
        max_length = max(len(t.noisy_gps) for t in trajectories)
        
        # Initialize arrays to accumulate errors and counts per position
        pw_sum = np.zeros(max_length, dtype=float)
        pw_count = np.zeros(max_length, dtype=int)
        
        # Iterate through errors aligned with trajectory structure
        error_idx = 0
        for traj_obj in trajectories:
            traj_length = len(traj_obj.noisy_gps)
            
            # Get errors for this trajectory
            traj_errors = errors[error_idx : error_idx + traj_length]
            
            # Add to positional sum and count
            pw_sum[:traj_length] += traj_errors
            pw_count[:traj_length] += 1
            
            error_idx += traj_length
        
        # Compute average error per position (avoid division by zero)
        pw_list = np.divide(pw_sum, pw_count, where=pw_count>0, out=np.zeros_like(pw_sum))
        
        # 2. Group into bytes (8 points each)
        num_bytes = int(np.ceil(max_length / 8))
        avg_l2_err_bw = []
        
        for byte_idx in range(num_bytes):
            start = byte_idx * 8
            end = min(start + 8, max_length)
            
            # Average error for this byte (only use positions that have data)
            byte_errors = pw_list[start:end]
            byte_avg = float(np.mean(byte_errors[byte_errors > 0]))  # exclude zeros from empty positions
            avg_l2_err_bw.append(byte_avg)
        
        # 3. Normalize (divide by mean of non-zero elements)
        bw_mean = np.mean([x for x in avg_l2_err_bw if x > 0])
        avg_l2_err_bw_norm = [x / bw_mean if bw_mean > 0 else 0.0 for x in avg_l2_err_bw]
        
        return {
            "avg_list": avg_l2_err_bw,
            "avg_list_norm": avg_l2_err_bw_norm
        }
    
    def _compute_chunkwise_metrics(
        self, trajectories: List, errors: np.ndarray, K: int, Q1: int, Q2: int
    ) -> Dict:
        """
        Purpose:
            Aggregate errors by chunk boundaries.
            
        Parameters:
            trajectories (List): original trajectories
            errors (np.ndarray): point-wise errors
            K (int): chunk size (256)
            Q1 (int): head buckle bytes (each byte = 8 points)
            Q2 (int): tail buckle bytes
            
        Return Dict:
            "avg_list": list[float] of per-chunk average errors
            "avg_list_norm": list[float] normalized version
            
        TODO:
            1. Build pw_list (per-point average across all trajectories)
            2. Determine chunk boundaries based on K, Q1, Q2
            3. For each chunk position, compute average error
            4. Normalize the list
            5. Return both lists
        """
        # 1. Build pw_list - same as bytewise
        max_length = max(len(t.noisy_gps) for t in trajectories)
        
        pw_sum = np.zeros(max_length, dtype=float)
        pw_count = np.zeros(max_length, dtype=int)
        
        error_idx = 0
        for traj_obj in trajectories:
            traj_length = len(traj_obj.noisy_gps)
            traj_errors = errors[error_idx : error_idx + traj_length]
            
            pw_sum[:traj_length] += traj_errors
            pw_count[:traj_length] += 1
            
            error_idx += traj_length
        
        pw_list = np.divide(pw_sum, pw_count, where=pw_count>0, out=np.zeros_like(pw_sum))
        
        # 2. Calculate chunk boundaries
        # Chunk structure: [HEAD_BUCKLE (Q1*8 points) | PAYLOAD | TAIL_BUCKLE (Q2*8 points)]
        # Stride = K - Q1*8 - Q2*8 (payload size)
        Q1_points = Q1 * 8
        Q2_points = Q2 * 8
        stride = K - Q1_points - Q2_points
        
        # Calculate number of chunks for longest trajectory
        # First chunk: 0 to K
        # Subsequent chunks: overlap by (Q1+Q2)*8 points
        num_chunks = 1  # First chunk
        remaining = max_length - K
        if remaining > 0:
            num_chunks += int(np.ceil(remaining / stride))
        
        # 3. Compute average error per chunk position
        avg_l2_err_cw = []
        
        for chunk_idx in range(num_chunks):
            if chunk_idx == 0:
                # First chunk: 0 to K
                start = 0
                end = min(K, max_length)
            else:
                # Subsequent chunks: previous_start + stride
                start = (chunk_idx - 1) * stride + K - (Q1_points + Q2_points)
                end = min(start + K, max_length)
            
            # Average error for this chunk
            chunk_errors = pw_list[start:end]
            # Only average over positions that have data
            valid_errors = chunk_errors[chunk_errors > 0]
            if len(valid_errors) > 0:
                chunk_avg = float(np.mean(valid_errors))
            else:
                chunk_avg = 0.0
            
            avg_l2_err_cw.append(chunk_avg)
        
        # 4. Normalize
        cw_mean = np.mean([x for x in avg_l2_err_cw if x > 0])
        avg_l2_err_cw_norm = [x / cw_mean if cw_mean > 0 else 0.0 for x in avg_l2_err_cw]
        
        return {
            "avg_list": avg_l2_err_cw,
            "avg_list_norm": avg_l2_err_cw_norm
        }
    
    def _measure_timing(self, model, longest_trajectory, method: str) -> float:
        """
        Purpose:
            Measure average denoising time on longest trajectory (5 runs).
            
        Parameters:
            model: loaded model
            longest_trajectory: trajectory object with maximum length
            method (str): "BF" or "DF"
            
        Return Dict:
            "error_code": 0 | -1
            "avg_time_sec": float | None
            
        TODO:
            1. Run denoising 5 times
            2. Record time for each run
            3. Compute average
            4. Return average time
        """
        times = []
        for _ in range(5):
            start = time.time()
            # Placeholder - run actual denoising
            # self._denoise_single_trajectory(model, longest_trajectory, method)
            end = time.time()
            times.append(end - start)
        
        return float(np.mean(times))
    
    def _save_results(self, results: Dict) -> Dict:
        """
        Purpose:
            Save results to both parquet and CSV.
            
        Parameters:
            results (Dict): complete results dictionary
            
        Return Dict:
            "error_code": 0 (success) | -1 (error)
            
        TODO:
            1. Create DataFrame from results
            2. Save to parquet (overwrite if exists)
            3. Append point-wise summary to CSV
            4. Return success/error code
        """
        try:
            # 1. Create DataFrame
            df = pd.DataFrame([results])
            
            # 2. Save to parquet
            parquet_filename = f"{results['model_name']}_{results['denoise_method']}.parquet"
            parquet_path = self.parquet_dir / parquet_filename
            df.to_parquet(parquet_path, index=False)
            self.logger.info(f"Saved parquet: {parquet_path}")
            
            # 3. Append to CSV (point-wise summary only)
            csv_row = (
                f"{results['model_name']},{results['denoise_method']},"
                f"{results['avg_l2_err_pw']:.6f},{results['med_l2_err_pw']:.6f},"
                f"{results['std_l2_err_pw']:.6f},{results['avg_denoise_time_sec']:.6f},"
                f"{results['num_tested_trajectories']},{results['num_tested_points']},"
                f"{results['test_timestamp']}\n"
            )
            with open(self.csv_path, "a") as f:
                f.write(csv_row)
            self.logger.info(f"Appended to CSV: {self.csv_path}")
            
            return {"error_code": 0}
        
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            return {"error_code": -1}


# ================================================================
# PLACEHOLDER EVALUATORS
# ================================================================
class InTrainingEvaluator:
    """
    Purpose:
        Quick validation during training (not implemented yet).
        
    TODO:
        Implement quick validation metrics for training monitoring.
    """
    pass


class RegionalEvaluator:
    """
    Purpose:
        Q1/Q2 buckle region accuracy analysis (not implemented yet).
        
    TODO:
        Implement regional accuracy testing for buckle sections.
    """
    pass


# ================================================================
# EVALUATION MANAGER
# ================================================================
class EvaluationManager:
    """
    Purpose:
        Orchestrate all evaluation types across multiple models.
        
    Responsibilities:
        - Model discovery
        - Test type selection
        - Parallel execution coordination (future)
        - Result aggregation
    """
    
    def __init__(self, output_dir: str = "test_results"):
        """
        Purpose:
            Initialize manager with all evaluators.
            
        Parameters:
            output_dir (str): root directory for all test results
            
        Return Dict:
            N/A (constructor)
            
        TODO:
            1. Initialize all evaluators
            2. Set up logging
        """
        # 1. Initialize evaluators
        self.trajectory_evaluator = TrajectoryEvaluator(output_dir)
        self.intraining_evaluator = InTrainingEvaluator()  # placeholder
        self.regional_evaluator = RegionalEvaluator()  # placeholder
        
        # 2. Set up logging
        self.logger = logging.getLogger("EvaluationManager")
        self.output_dir = Path(output_dir)
    
    def run_trajectory_evaluation(
        self,
        model_names: Optional[List[str]] = None,
        denoise_methods: List[str] = None,
        model_root: str = "./bin/model",
        test_data_path: str = "./dataset/processed/full_traj",
    ) -> Dict:
        """
        Purpose:
            Run trajectory evaluation on specified models.
            
        Parameters:
            model_names (List[str] | None): specific models to test, or None for all
            denoise_methods (List[str] | None): methods to test, default ["BF", "DF"]
            model_root (str): root directory containing model subdirectories
            test_data_path (str): path to test trajectory data
            
        Return Dict:
            "error_code": 0 (success) | -1 (error)
            "results": list of result dicts | None
            
        TODO:
            1. Discover models if model_names is None
            2. Load test trajectories
            3. For each model and method, run evaluation
            4. Aggregate results
            5. Return summary
        """
        if denoise_methods is None:
            denoise_methods = ["BF", "DF"]
        
        # 1. Discover models
        if model_names is None:
            discover_result = self._discover_models(model_root)
            if discover_result["error_code"] != 0:
                return {"error_code": -1, "results": None}
            model_names = discover_result["model_names"]
        
        self.logger.info(f"Found {len(model_names)} models to evaluate")
        
        # 2. Load test trajectories
        load_result = self._load_test_trajectories(test_data_path)
        if load_result["error_code"] != 0:
            return {"error_code": -1, "results": None}
        
        test_trajectories = load_result["trajectories"]
        
        # 3. Run evaluations
        all_results = []
        for model_name in model_names:
            model_dir = Path(model_root) / model_name
            
            # Find best checkpoint
            ckpt_result = self._find_best_checkpoint(model_dir)
            if ckpt_result["error_code"] != 0:
                self.logger.warning(f"No checkpoint found for {model_name}, skipping")
                continue
            
            checkpoint_name = ckpt_result["checkpoint_name"]
            
            # Get configuration
            config = self._load_model_config(model_dir)
            K = config.get("K", 256)
            Q1 = config.get("Q1", 2)
            Q2 = config.get("Q2", 2)
            
            # Test each method
            for method in denoise_methods:
                self.logger.info(f"Testing {model_name} with {method}")
                
                result = self.trajectory_evaluator.evaluate_model(
                    model_name=model_name,
                    model_dir=str(model_dir.absolute()),
                    checkpoint_name=checkpoint_name,
                    denoise_method=method,
                    test_trajectories=test_trajectories,
                    K=K,
                    Q1=Q1,
                    Q2=Q2,
                )
                
                if result["error_code"] == 0:
                    all_results.append(result["results"])
                else:
                    self.logger.error(f"Failed to evaluate {model_name} {method}")
        
        # 4. Return summary
        self.logger.info(f"Completed {len(all_results)} evaluations")
        return {"error_code": 0, "results": all_results}
    
    def _discover_models(self, model_root: str) -> Dict:
        """
        Purpose:
            Find all models with best checkpoints.
            
        Parameters:
            model_root (str): root directory containing models
            
        Return Dict:
            "error_code": 0 | -1
            "model_names": list[str] | None
            
        TODO:
            1. Scan model_root for subdirectories
            2. Check each for checkpoints directory
            3. Return list of valid model names
        """
        model_root = Path(model_root)
        if not model_root.exists():
            self.logger.error(f"Model root not found: {model_root}")
            return {"error_code": -1, "model_names": None}
        
        model_names = []
        for model_dir in model_root.iterdir():
            if not model_dir.is_dir():
                continue
            
            # Check for either best_ckpt or ckpts directory with .pt files
            has_checkpoints = False
            for ckpt_dir_name in ["best_ckpt", "ckpts"]:
                ckpt_dir = model_dir / ckpt_dir_name
                if ckpt_dir.exists() and any(ckpt_dir.glob("*_full.pt")):
                    has_checkpoints = True
                    break
            
            if has_checkpoints:
                model_names.append(model_dir.name)
        
        return {"error_code": 0, "model_names": sorted(model_names)}
    
    def _find_best_checkpoint(self, model_dir: Path) -> Dict:
        """
        Purpose:
            Find best checkpoint in model directory.
            
        Parameters:
            model_dir (Path): path to model directory
            
        Return Dict:
            "error_code": 0 | -1
            "checkpoint_name": str | None
            
        TODO:
            1. Look for best_ckpt directory first
            2. If found, take the .pt file
            3. If not found, look in ckpts directory for most recent
            4. Return checkpoint filename
        """
        # Try best_ckpt directory first
        best_ckpt_dir = model_dir / "best_ckpt"
        if best_ckpt_dir.exists():
            best_ckpts = list(best_ckpt_dir.glob("*_full.pt"))
            if best_ckpts:
                return {"error_code": 0, "checkpoint_name": best_ckpts[0].name}
        
        # Fallback to ckpts directory
        ckpts_dir = model_dir / "ckpts"
        if not ckpts_dir.exists():
            return {"error_code": -1, "checkpoint_name": None}
        
        # Look for *_full.pt files (full checkpoints with optimizer state)
        all_ckpts = sorted(ckpts_dir.glob("*_full.pt"), key=lambda p: p.stat().st_mtime)
        if all_ckpts:
            return {"error_code": 0, "checkpoint_name": all_ckpts[-1].name}
        
        return {"error_code": -1, "checkpoint_name": None}
    
    def _load_model_config(self, model_dir: Path) -> Dict:
        """
        Purpose:
            Load model configuration from log/config.json.
            
        Parameters:
            model_dir (Path): path to model directory
            
        Return Dict:
            Configuration dictionary
            
        TODO:
            1. Read config.json from log directory
            2. Return parsed JSON
        """
        config_path = model_dir / "log" / "config.json"
        if not config_path.exists():
            self.logger.warning(f"Config not found: {config_path}")
            return {}
        
        with open(config_path, "r") as f:
            return json.load(f)
    
    def _load_test_trajectories(self, test_data_path: str = "./dataset/processed/full_traj") -> List:
        """
        Purpose:
            Load test dataset trajectories from .pt file.
            
        Parameters:
            test_data_path (str): path to directory containing trajectory .pt files
            
        Return Dict:
            "error_code": 0 | -1
            "trajectories": List of trajectory objects | None
            
        TODO:
            1. Find .pt file in test data directory
            2. Load torch file
            3. Extract trajectory list
            4. Convert to expected format
        """
        test_dir = Path(test_data_path)
        
        if not test_dir.exists():
            self.logger.error(f"Test data directory not found: {test_dir}")
            return {"error_code": -1, "trajectories": None}
        
        # Find .pt file (should be named like fulltraj_M_N.pt)
        pt_files = list(test_dir.glob("*.pt"))
        
        if len(pt_files) == 0:
            self.logger.error(f"No .pt files found in {test_dir}")
            return {"error_code": -1, "trajectories": None}
        
        # Use first .pt file found
        pt_file = pt_files[0]
        self.logger.info(f"Loading test trajectories from {pt_file.name}")
        
        try:
            data = torch.load(pt_file, map_location="cpu")
            
            # Extract trajectories from saved format
            # Format: {"trajectories": [...], "metadata": {...}}
            raw_trajectories = data["trajectories"]
            
            # Convert to simple object structure for evaluation
            # Create simple namespace objects for easy attribute access
            from types import SimpleNamespace
            
            trajectories = []
            for traj_dict in raw_trajectories:
                traj_obj = SimpleNamespace(
                    agent_id=traj_dict["agent_id"],
                    n_points=traj_dict["n_points"],
                    noisy_gps=traj_dict["data"].numpy(),   # (N, 2) [lon, lat] - noisy
                    clean_gps=traj_dict["label"].numpy(),  # (N, 2) [lon, lat] - clean
                )
                trajectories.append(traj_obj)
            
            self.logger.info(f"Loaded {len(trajectories)} trajectories")
            return {"error_code": 0, "trajectories": trajectories}
            
        except Exception as e:
            self.logger.error(f"Failed to load trajectories: {e}")
            return {"error_code": -1, "trajectories": None}


# ================================================================
# MAIN
# ================================================================
def main():
    """
    Purpose:
        Entry point for running evaluations.
        
    TODO:
        1. Set up logging
        2. Create manager
        3. Run trajectory evaluation
    """
    # 1. Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # 2. Create manager
    manager = EvaluationManager(output_dir="test_results")
    
    # 3. Run evaluation
    result = manager.run_trajectory_evaluation()
    
    if result["error_code"] == 0:
        print(f"\n✓ Evaluation complete: {len(result['results'])} tests run")
    else:
        print("\n✗ Evaluation failed")


if __name__ == "__main__":
    main()