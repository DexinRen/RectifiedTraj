#!/usr/bin/env python3
"""
evaluation_manager.py

PURPOSE:
    Orchestrate trajectory-wise denoising evaluation across multiple models.
    Supports BF (Breadth-First) and DF (Depth-First) denoising methods.
    Outputs multi-granularity error metrics (point/byte/chunk-wise).

USAGE:
    manager = EvaluationManager(output_dir="./bin/test_results")
    manager.run_trajectory_evaluation()
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

# Project imports
import encoder_decoder
from encoder_decoder import EncoderDecoder
from theta_model import build_theta_model


logger = logging.getLogger(__name__)


# ================================================================
# TRAJECTORY EVALUATOR
# ================================================================
class TrajectoryEvaluator:
    """
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
        """Initialize evaluator with output directories."""
        self.output_dir = Path(output_dir)
        self.parquet_dir = self.output_dir / "trajectory_evaluation_results"
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_path = self.output_dir / "trajectory_evaluation_summary.csv"
        self.logger = logging.getLogger("TrajectoryEvaluator")
        
        # Create CSV with headers if doesn't exist
        if not self.csv_path.exists():
            header = (
                "model_name,denoise_method,K,Q1,Q2,t_delta,N_steps,"
                "avg_l2_err_pw,med_l2_err_pw,std_l2_err_pw,"
                "avg_denoise_time_sec,num_tested_trajectories,num_tested_points,test_timestamp\n"
            )
            self.csv_path.write_text(header)
    
    
    # ============================================================
    # PUBLIC: Main evaluation entry point
    # ============================================================
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
        Run full trajectory evaluation for one model + method combination.
        
        Parameters:
            model_name: model identifier
            model_dir: absolute path to model directory
            checkpoint_name: checkpoint filename (e.g., "ckpt_e24_s860000_full.pt")
            denoise_method: "BF" or "DF"
            test_trajectories: list of trajectory objects with noisy/clean data
            K: chunk size
            Q1: head buckle size in bytes
            Q2: tail buckle size in bytes
            
        Returns:
            Dictionary with all computed metrics
        """
        self.logger.info(f"Evaluating {model_name} with {denoise_method}")
        
        # 1. Get checkpoint path
        checkpoint_path = self._get_checkpoint_path(model_dir, checkpoint_name)
        if checkpoint_path is None:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_name}")
        
        # 2. Run denoising and compute errors
        denoised_trajectories, errors, decoder = self._denoise_trajectories(
            checkpoint_path, test_trajectories, denoise_method
        )
        
        # 2.5. Extract denoising parameters from decoder instance
        # decoder.t_delta is the ACTUAL step size being used for denoising
        t_delta = decoder.t_delta
        N_steps = int(1.0 / t_delta) if t_delta > 0 else 10
        
        self.logger.info(f"Denoising parameters: t_delta={t_delta}, N_steps={N_steps}")
        
        # 3. Compute metrics at different granularities
        self.logger.info("Computing point-wise metrics...")
        pw_metrics = self._compute_pointwise_metrics(errors)
        
        self.logger.info("Computing byte-wise metrics...")
        bw_metrics = self._compute_bytewise_metrics(test_trajectories, errors)
        
        self.logger.info("Computing chunk-wise metrics...")
        cw_metrics = self._compute_chunkwise_metrics(test_trajectories, errors, K, Q1, Q2)
        
        # 4. Measure execution time
        longest_traj = max(test_trajectories, key=lambda t: len(t.noisy_gps))
        self.logger.info(f"Measuring timing (5 runs on longest trajectory: {len(longest_traj.noisy_gps)} points)...")
        avg_time = self._measure_timing(checkpoint_path, longest_traj, denoise_method)
        
        # 5. Assemble results
        results = {
            # Metadata
            "model_name": model_name,
            "model_dir": model_dir,
            "checkpoint_name": checkpoint_name,
            "K": K,
            "Q1": Q1,
            "Q2": Q2,
            "t_delta": t_delta,
            "N_steps": N_steps,
            "denoise_method": denoise_method,
            "test_timestamp": datetime.now().isoformat(),
            "num_tested_trajectories": len(test_trajectories),
            "num_tested_points": sum(len(t.noisy_gps) for t in test_trajectories),
            "longest_trajectory_length": len(longest_traj.noisy_gps),
            
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
        
        # 6. Save results
        self._save_results(results)
        
        self.logger.info(f"Evaluation complete: {model_name} {denoise_method}")
        return results
    
    
    # ============================================================
    # DENOISING: Run EncoderDecoder on trajectories
    # ============================================================
    def _denoise_trajectories(
        self, 
        checkpoint_path: str, 
        test_trajectories: List, 
        method: str
    ) -> tuple:
        """
        Run denoising on all trajectories and compute errors.
        
        Parameters:
            checkpoint_path: full path to checkpoint file
            test_trajectories: list of trajectory objects
            method: "BF" or "DF"
            
        Returns:
            (denoised_trajectories, all_errors_array, decoder)
        """
        assert method in ["BF", "DF"], f"Invalid method: {method}"
        
        # CRITICAL: Patch encoder_decoder checkpoint loading before use
        self._patch_encoder_decoder_checkpoint_loading()
        
        # Initialize EncoderDecoder
        decoder = EncoderDecoder(checkpoint_path)
        self.logger.info(f"Initialized EncoderDecoder with {checkpoint_path}")
        
        denoised_trajectories = []
        all_errors = []
        
        self.logger.info(f"Starting denoising of {len(test_trajectories)} trajectories...")
        
        for idx, traj_obj in enumerate(test_trajectories):
            self.logger.info(f"  Denoising trajectory {idx + 1}/{len(test_trajectories)} ({len(traj_obj.noisy_gps)} points)...")
            
            noisy_gps = traj_obj.noisy_gps  # (T, 2) [lon, lat]
            clean_gps = traj_obj.clean_gps  # (T, 2) [lon, lat]
            
            # Run denoising
            if method == "BF":
                denoised_gps = decoder.denoise_traj_BF(noisy_gps)  # Returns ndarray directly
            else:  # DF
                denoised_gps = decoder.denoise_traj_DF(noisy_gps)  # Returns ndarray directly
            
            # Align lengths (denoising strips buckles)
            T_denoised = len(denoised_gps)
            clean_gps_aligned = clean_gps[-T_denoised:]
            
            # Convert to ENU for error calculation (meters)
            ref_lat = float(clean_gps_aligned[0, 1])
            ref_lon = float(clean_gps_aligned[0, 0])
            enu_denoised = self._gps_to_enu_batch(denoised_gps, ref_lat, ref_lon)
            enu_clean = self._gps_to_enu_batch(clean_gps_aligned, ref_lat, ref_lon)
            
            # Compute L2 errors per point
            errors = np.linalg.norm(enu_denoised - enu_clean, axis=1)
            
            denoised_trajectories.append(denoised_gps)
            all_errors.append(errors)
        
        if len(all_errors) == 0:
            raise RuntimeError("No trajectories successfully denoised")
        
        all_errors_array = np.concatenate(all_errors, axis=0)
        
        self.logger.info(f"Denoised {len(denoised_trajectories)} trajectories")
        
        # Return decoder instance so we can access t_delta and other attributes
        return denoised_trajectories, all_errors_array, decoder
    
    
    def _gps_to_enu_batch(
        self, 
        gps_coords: np.ndarray, 
        ref_lat: float, 
        ref_lon: float
    ) -> np.ndarray:
        """
        Convert GPS coordinates to ENU (East-North-Up) in meters.
        
        Parameters:
            gps_coords: (N, 2) array of [lon, lat]
            ref_lat: reference latitude
            ref_lon: reference longitude
            
        Returns:
            (N, 2) array of [east, north] in meters
        """
        import pymap3d as pm
        
        lons = gps_coords[:, 0]
        lats = gps_coords[:, 1]
        
        e, n, u = pm.geodetic2enu(lats, lons, 0, ref_lat, ref_lon, 0)
        
        return np.stack([e, n], axis=1)
    
    
    @staticmethod
    def _patch_encoder_decoder_checkpoint_loading():
        """
        CRITICAL FIX: Patch load_model_from_config in encoder_decoder.py
        to properly handle *_full.pt checkpoints that contain nested dicts.
        
        The *_full.pt files have structure:
        {
            "model_state_dict": {...},
            "optimizer_state_dict": {...},
            "scheduler_state_dict": {...},
            ...
        }
        
        But load_model_from_config tries to load the whole dict as state_dict.
        This monkey-patches the function to extract model_state_dict properly.
        """
        original_load = encoder_decoder.load_model_from_config

        def patched_load(config_json_path: Path, ckpt_path: Path):
            cfg = json.loads(Path(config_json_path).read_text())
            runtime = {"config": cfg}
            
            model = build_theta_model(runtime).to(encoder_decoder.DEVICE)
            
            ckpt_path = Path(ckpt_path)
            
            # Load checkpoint
            if ckpt_path.suffix == ".safetensors":
                sd = encoder_decoder.load_safetensors(str(ckpt_path))
            else:
                # FIX: Handle *_full.pt files that contain nested dict
                blob = torch.load(str(ckpt_path), map_location=encoder_decoder.DEVICE)
                
                if isinstance(blob, dict) and "model_state_dict" in blob:
                    # Training checkpoint with optimizer/scheduler state
                    sd = blob["model_state_dict"]
                else:
                    # Plain state dict
                    sd = blob
            
            model.load_state_dict(sd)
            model.eval()
            return model, cfg
        
        # Monkey patch
        encoder_decoder.load_model_from_config = patched_load
    
    
    # ============================================================
    # METRICS: Point-wise
    # ============================================================
    def _compute_pointwise_metrics(self, errors: np.ndarray) -> Dict:
        """
        Compute mean, median, std of point-wise L2 errors.
        
        Parameters:
            errors: 1D array of L2 errors for all points
            
        Returns:
            {"avg": float, "med": float, "std": float}
        """
        return {
            "avg": float(np.mean(errors)),
            "med": float(np.median(errors)),
            "std": float(np.std(errors)),
        }
    
    
    # ============================================================
    # METRICS: Byte-wise (8-point groups)
    # ============================================================
    def _compute_bytewise_metrics(
        self, 
        trajectories: List, 
        errors: np.ndarray
    ) -> Dict:
        """
        Aggregate errors into bytes (8-point groups).
        
        Strategy:
            1. Build pw_list: per-point average across all trajectories
            2. Group pw_list into bytes (8 points each)
            3. Normalize by dividing by mean
            
        Parameters:
            trajectories: original trajectory objects
            errors: point-wise errors (concatenated across all trajectories)
            
        Returns:
            {"avg_list": list[float], "avg_list_norm": list[float]}
        """
        # Build per-point average (pw_list)
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
        
        # Group into bytes (8 points each)
        num_bytes = int(np.ceil(max_length / 8))
        avg_l2_err_bw = []
        
        for byte_idx in range(num_bytes):
            start = byte_idx * 8
            end = min(start + 8, max_length)
            
            byte_errors = pw_list[start:end]
            byte_avg = float(np.mean(byte_errors[byte_errors > 0]))
            avg_l2_err_bw.append(byte_avg)
        
        # Normalize
        bw_mean = np.mean([x for x in avg_l2_err_bw if x > 0])
        avg_l2_err_bw_norm = [
            x / bw_mean if bw_mean > 0 else 0.0 
            for x in avg_l2_err_bw
        ]
        
        return {
            "avg_list": avg_l2_err_bw,
            "avg_list_norm": avg_l2_err_bw_norm
        }
    
    
    # ============================================================
    # METRICS: Chunk-wise (by K, Q1, Q2 boundaries)
    # ============================================================
    def _compute_chunkwise_metrics(
        self, 
        trajectories: List, 
        errors: np.ndarray, 
        K: int, 
        Q1: int, 
        Q2: int
    ) -> Dict:
        """
        Aggregate errors by chunk boundaries.
        
        Strategy:
            1. Build per-point average (pw_list)
            2. Calculate chunk boundaries based on K, Q1, Q2
            3. Average errors within each chunk
            4. Normalize by dividing by mean
            
        Parameters:
            trajectories: original trajectory objects
            errors: point-wise errors
            K: chunk size (256)
            Q1: head buckle bytes
            Q2: tail buckle bytes
            
        Returns:
            {"avg_list": list[float], "avg_list_norm": list[float]}
        """
        # Build per-point average
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
        
        # Calculate chunk boundaries
        Q1_points = Q1 * 8
        Q2_points = Q2 * 8
        stride = K - Q1_points - Q2_points
        
        num_chunks = 1
        remaining = max_length - K
        if remaining > 0:
            num_chunks += int(np.ceil(remaining / stride))
        
        # Compute average error per chunk
        avg_l2_err_cw = []
        
        for chunk_idx in range(num_chunks):
            if chunk_idx == 0:
                start = 0
                end = min(K, max_length)
            else:
                start = (chunk_idx - 1) * stride + K - (Q1_points + Q2_points)
                end = min(start + K, max_length)
            
            chunk_errors = pw_list[start:end]
            valid_errors = chunk_errors[chunk_errors > 0]
            
            if len(valid_errors) > 0:
                chunk_avg = float(np.mean(valid_errors))
            else:
                chunk_avg = 0.0
            
            avg_l2_err_cw.append(chunk_avg)
        
        # Normalize
        cw_mean = np.mean([x for x in avg_l2_err_cw if x > 0])
        avg_l2_err_cw_norm = [
            x / cw_mean if cw_mean > 0 else 0.0 
            for x in avg_l2_err_cw
        ]
        
        return {
            "avg_list": avg_l2_err_cw,
            "avg_list_norm": avg_l2_err_cw_norm
        }
    
    
    # ============================================================
    # TIMING: Measure execution time
    # ============================================================
    def _measure_timing(
        self, 
        checkpoint_path: str, 
        longest_trajectory, 
        method: str
    ) -> float:
        """
        Measure average denoising time on longest trajectory (5 runs).
        
        Parameters:
            checkpoint_path: full path to checkpoint
            longest_trajectory: trajectory object with maximum length
            method: "BF" or "DF"
            
        Returns:
            Average time in seconds
        """
        from encoder_decoder import EncoderDecoder
        decoder = EncoderDecoder(checkpoint_path)
        
        times = []
        for run_idx in range(5):
            self.logger.info(f"  Timing run {run_idx + 1}/5...")
            start = time.time()
            
            if method == "BF":
                result = decoder.denoise_traj_BF(longest_trajectory.noisy_gps)
            else:  # DF
                _ = decoder.denoise_traj_DF(longest_trajectory.noisy_gps)
            
            end = time.time()
            elapsed = end - start
            times.append(elapsed)
            self.logger.info(f"  Run {run_idx + 1} completed in {elapsed:.2f}s")
        
        avg = float(np.mean(times))
        self.logger.info(f"Average timing: {avg:.2f}s over 5 runs")
        return avg
    
    
    # ============================================================
    # STORAGE: Save results to parquet and CSV
    # ============================================================
    def _save_results(self, results: Dict):
        """
        Save results to both parquet and CSV.
        
        Parameters:
            results: complete results dictionary
        """
        # Save to parquet (detailed results with lists)
        df = pd.DataFrame([results])
        parquet_filename = f"{results['model_name']}_{results['denoise_method']}.parquet"
        parquet_path = self.parquet_dir / parquet_filename
        df.to_parquet(parquet_path, index=False)
        self.logger.info(f"Saved parquet: {parquet_path}")
        
        # Append to CSV (point-wise summary only)
        csv_row = (
            f"{results['model_name']},{results['denoise_method']},"
            f"{results['K']},{results['Q1']},{results['Q2']},"
            f"{results['t_delta']:.4f},{results['N_steps']},"
            f"{results['avg_l2_err_pw']:.6f},{results['med_l2_err_pw']:.6f},"
            f"{results['std_l2_err_pw']:.6f},{results['avg_denoise_time_sec']:.6f},"
            f"{results['num_tested_trajectories']},{results['num_tested_points']},"
            f"{results['test_timestamp']}\n"
        )
        with open(self.csv_path, "a") as f:
            f.write(csv_row)
        self.logger.info(f"Appended to CSV: {self.csv_path}")
    
    
    # ============================================================
    # HELPER: Get checkpoint path
    # ============================================================
    def _get_checkpoint_path(self, model_dir: str, checkpoint_name: str) -> Optional[str]:
        """
        Find full path to checkpoint file.

        Parameters:
            model_dir: path to model directory
            checkpoint_name: checkpoint filename

        Returns:
            Full path to checkpoint, or None if not found
        """
        for ckpt_dir_name in ["best_ckpt", "ckpts"]:
            ckpt_dir = Path(model_dir) / ckpt_dir_name
            if ckpt_dir.exists():
                ckpt_path = ckpt_dir / checkpoint_name
                if ckpt_path.exists():
                    return str(ckpt_path.absolute())

        return None


    # ============================================================
    # GRID SEARCH: Evaluate with manual hyperparameter config
    # ============================================================
    def evaluate_model_with_config(
        self,
        model_name: str,
        model_dir: str,
        checkpoint_name: str,
        denoise_method: str,
        test_trajectories: List,
        manual_config: Dict,
    ) -> Dict:
        """
        Run evaluation with manual hyperparameter configuration.

        Parameters:
            model_name: model identifier
            model_dir: absolute path to model directory
            checkpoint_name: checkpoint filename
            denoise_method: "BF" or "DF"
            test_trajectories: list of trajectory objects
            manual_config: {"Q1": int, "Q2": int, "t_delta": float}

        Returns:
            Dictionary with all computed metrics
        """
        Q1 = manual_config["Q1"]
        Q2 = manual_config["Q2"]
        t_delta = manual_config["t_delta"]

        self.logger.info(
            f"Evaluating {model_name} with {denoise_method} "
            f"(Q1={Q1}, Q2={Q2}, t_delta={t_delta})"
        )

        # Get checkpoint path
        checkpoint_path = self._get_checkpoint_path(model_dir, checkpoint_name)
        if checkpoint_path is None:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_name}")

        # Run denoising with manual_config
        denoised_trajectories, errors, decoder = self._denoise_trajectories_with_config(
            checkpoint_path, test_trajectories, denoise_method, manual_config
        )

        # Extract actual parameters from decoder
        actual_t_delta = decoder.t_delta
        actual_Q1 = decoder.Q1_bytes
        actual_Q2 = decoder.Q2_bytes
        N_steps = int(1.0 / actual_t_delta) if actual_t_delta > 0 else 10

        self.logger.info(f"Actual denoising parameters: t_delta={actual_t_delta}, N_steps={N_steps}")

        # Compute metrics
        self.logger.info("Computing point-wise metrics...")
        pw_metrics = self._compute_pointwise_metrics(errors)

        self.logger.info("Computing byte-wise metrics...")
        bw_metrics = self._compute_bytewise_metrics(test_trajectories, errors)

        self.logger.info("Computing chunk-wise metrics...")
        cw_metrics = self._compute_chunkwise_metrics(
            test_trajectories, errors, decoder.K, decoder.Q1, decoder.Q2
        )

        # Measure timing
        longest_traj = max(test_trajectories, key=lambda t: len(t.noisy_gps))
        self.logger.info(f"Measuring timing (5 runs on longest trajectory: {len(longest_traj.noisy_gps)} points)...")
        avg_time = self._measure_timing_with_config(
            checkpoint_path, longest_traj, denoise_method, manual_config
        )

        # Assemble results
        results = {
            # Metadata
            "model_name": model_name,
            "model_dir": model_dir,
            "checkpoint_name": checkpoint_name,
            "K": decoder.K,
            "Q1": actual_Q1,
            "Q2": actual_Q2,
            "t_delta": actual_t_delta,
            "N_steps": N_steps,
            "denoise_method": denoise_method,
            "test_timestamp": datetime.now().isoformat(),
            "num_tested_trajectories": len(test_trajectories),
            "num_tested_points": sum(len(t.noisy_gps) for t in test_trajectories),
            "longest_trajectory_length": len(longest_traj.noisy_gps),

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

        # Save results
        self._save_results(results)

        self.logger.info(f"Evaluation complete: {model_name} {denoise_method}")
        return results


    def _denoise_trajectories_with_config(
        self,
        checkpoint_path: str,
        test_trajectories: List,
        method: str,
        manual_config: Dict,
    ) -> tuple:
        """
        Run denoising with manual configuration override.

        Parameters:
            checkpoint_path: full path to checkpoint file
            test_trajectories: list of trajectory objects
            method: "BF" or "DF"
            manual_config: {"Q1": int, "Q2": int, "t_delta": float}

        Returns:
            (denoised_trajectories, all_errors_array, decoder)
        """
        assert method in ["BF", "DF"], f"Invalid method: {method}"

        # CRITICAL: Patch encoder_decoder checkpoint loading before use
        self._patch_encoder_decoder_checkpoint_loading()

        # Initialize EncoderDecoder WITH manual_config
        decoder = EncoderDecoder(checkpoint_path, manual_config=manual_config)
        self.logger.info(
            f"Initialized EncoderDecoder with manual config: "
            f"Q1={manual_config['Q1']}, Q2={manual_config['Q2']}, t_delta={manual_config['t_delta']}"
        )

        # Rest is same as _denoise_trajectories...
        denoised_trajectories = []
        all_errors = []

        self.logger.info(f"Starting denoising of {len(test_trajectories)} trajectories...")

        for idx, traj_obj in enumerate(test_trajectories):
            self.logger.info(f"  Denoising trajectory {idx + 1}/{len(test_trajectories)} ({len(traj_obj.noisy_gps)} points)...")

            noisy_gps = traj_obj.noisy_gps
            clean_gps = traj_obj.clean_gps

            # Run denoising
            if method == "BF":
                denoised_gps = decoder.denoise_traj_BF(noisy_gps)
            else:  # DF
                denoised_gps = decoder.denoise_traj_DF(noisy_gps)

            # Align lengths
            T_denoised = len(denoised_gps)
            clean_gps_aligned = clean_gps[-T_denoised:]

            # Convert to ENU
            ref_lat = float(clean_gps_aligned[0, 1])
            ref_lon = float(clean_gps_aligned[0, 0])
            enu_denoised = self._gps_to_enu_batch(denoised_gps, ref_lat, ref_lon)
            enu_clean = self._gps_to_enu_batch(clean_gps_aligned, ref_lat, ref_lon)

            # Compute errors
            errors = np.linalg.norm(enu_denoised - enu_clean, axis=1)

            denoised_trajectories.append(denoised_gps)
            all_errors.append(errors)

        if len(all_errors) == 0:
            raise RuntimeError("No trajectories successfully denoised")

        all_errors_array = np.concatenate(all_errors, axis=0)

        self.logger.info(f"Denoised {len(denoised_trajectories)} trajectories")
        return denoised_trajectories, all_errors_array, decoder


    def _measure_timing_with_config(
        self,
        checkpoint_path: str,
        longest_trajectory,
        method: str,
        manual_config: Dict,
    ) -> float:
        """Measure timing with manual config."""
        decoder = EncoderDecoder(checkpoint_path, manual_config=manual_config)

        times = []
        for run_idx in range(5):
            self.logger.info(f"  Timing run {run_idx + 1}/5...")
            start = time.time()

            if method == "BF":
                _ = decoder.denoise_traj_BF(longest_trajectory.noisy_gps)
            else:  # DF
                _ = decoder.denoise_traj_DF(longest_trajectory.noisy_gps)

            end = time.time()
            elapsed = end - start
            times.append(elapsed)
            self.logger.info(f"  Run {run_idx + 1} completed in {elapsed:.2f}s")

        avg = float(np.mean(times))
        self.logger.info(f"Average timing: {avg:.2f}s over 5 runs")
        return avg


# ================================================================
# PLACEHOLDER EVALUATORS (Future Implementation)
# ================================================================
class InTrainingEvaluator:
    """Quick validation during training (not implemented yet)."""
    pass


class RegionalEvaluator:
    """Q1/Q2 buckle region accuracy analysis (not implemented yet)."""
    pass


# ================================================================
# EVALUATION MANAGER
# ================================================================
class EvaluationManager:
    """
    Orchestrate all evaluation types across multiple models.
    
    Responsibilities:
        - Model discovery
        - Test type selection
        - Parallel execution coordination (future)
        - Result aggregation
    """
    
    def __init__(self, output_dir: str = "test_results"):
        """Initialize manager with all evaluators."""
        self.trajectory_evaluator = TrajectoryEvaluator(output_dir)
        self.intraining_evaluator = InTrainingEvaluator()
        self.regional_evaluator = RegionalEvaluator()
        
        self.logger = logging.getLogger("EvaluationManager")
        self.output_dir = Path(output_dir)
    
    
    # ============================================================
    # PUBLIC: Run trajectory evaluation
    # ============================================================
    def run_trajectory_evaluation(
        self,
        model_names: Optional[List[str]] = None,
        denoise_methods: List[str] = None,
        model_root: str = "./bin/model",
        test_data_path: str = "./dataset/processed/full_traj",
        M: int = 20,
        D: Optional[float] = None,
        N: Optional[int] = None,
    ) -> List[Dict]:
        """
        Run trajectory evaluation on specified models.

        Parameters:
            model_names: specific models to test, or None for all
            denoise_methods: methods to test, default ["BF", "DF"]
            model_root: root directory containing model subdirectories
            test_data_path: path to test trajectory data
            M: number of trajectories to test (default 20)
            D: number of days per trajectory (default 7). Assumes 10-sec sampling (8640 points/day)
            N: number of points per trajectory (optional, overrides D if provided)

        Returns:
            List of result dictionaries
        """
        # Convert D (days) to N (points) if N not explicitly provided
        if N is None:
            if D is None:
                D = 7.0  # Default: 7 days
            N = int(D * 8640)  # 8640 points per day @ 10-sec sampling

        self.logger.info(f"Dataset configuration: M={M}, D={D if D else N/8640:.2f} days, N={N} points")
        if denoise_methods is None:
            denoise_methods = ["BF", "DF"]

        # Discover models
        if model_names is None:
            model_names = self._discover_models(model_root)

        self.logger.info(f"Found {len(model_names)} models to evaluate")

        # Load or generate test trajectories
        test_trajectories = self._load_or_generate_test_data(test_data_path, M, N)
        
        # Run evaluations
        all_results = []
        for model_name in model_names:
            model_dir = Path(model_root) / model_name
            
            # Find best checkpoint
            checkpoint_name = self._find_best_checkpoint(model_dir)
            if checkpoint_name is None:
                self.logger.warning(f"No checkpoint found for {model_name}, skipping")
                continue
            
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
                
                all_results.append(result)
        
        self.logger.info(f"Completed {len(all_results)} evaluations")
        return all_results
    
    
    # ================================================================
    # PART 2: EvaluationManager grid search
    # ================================================================
    def run_grid_search_evaluation(
        self,
        model_names: Optional[List[str]] = None,
        denoise_methods: List[str] = None,
        job_list: Optional[Dict] = None,
        model_root: str = "./bin/model",
        test_data_path: str = "./dataset/processed/full_traj",
        M: int = 20,
        D: Optional[float] = None,
        N: Optional[int] = None,
    ) -> List[Dict]:
        """
        Run hyperparameter grid search evaluation.

        Parameters:
            model_names: specific models to test, or None for all
            denoise_methods: methods to test, default ["BF", "DF"]
            job_list: hyperparameter grid configuration
                Example:
                {
                    "Q1": [1, 2, 3, 4],
                    "Q2": [0, 1, 2],
                    "t_delta": [0.1, 0.05, 0.01]
                }
            model_root: root directory containing model subdirectories
            test_data_path: path to test trajectory data
            M: number of trajectories to test (default 20)
            D: number of days per trajectory (default 7). Assumes 10-sec sampling (8640 points/day)
            N: number of points per trajectory (optional, overrides D if provided)

        Returns:
            List of result dictionaries for all combinations
        """
        # Convert D (days) to N (points) if N not explicitly provided
        if N is None:
            if D is None:
                D = 7.0  # Default: 7 days
            N = int(D * 8640)  # 8640 points per day @ 10-sec sampling

        self.logger.info(f"Dataset configuration: M={M}, D={D if D else N/8640:.2f} days, N={N} points")
        if denoise_methods is None:
            denoise_methods = ["BF", "DF"]
        
        # Default job_list if not provided
        if job_list is None:
            job_list = {
                "Q1": [1, 2],
                "Q2": [0, 1],
                "t_delta": [0.1, 0.05]
            }
        
        # Discover models
        if model_names is None:
            model_names = self._discover_models(model_root)
        
        self.logger.info(f"Found {len(model_names)} models to evaluate")
        self.logger.info(f"Grid search space:")
        self.logger.info(f"  Q1: {job_list['Q1']}")
        self.logger.info(f"  Q2: {job_list['Q2']}")
        self.logger.info(f"  t_delta: {job_list['t_delta']}")
        self.logger.info(f"  Methods: {denoise_methods}")
        
        total_combinations = (
            len(model_names) * 
            len(job_list['Q1']) * 
            len(job_list['Q2']) * 
            len(job_list['t_delta']) * 
            len(denoise_methods)
        )
        self.logger.info(f"Total combinations: {total_combinations}")

        # Load or generate test trajectories once
        test_trajectories = self._load_or_generate_test_data(test_data_path, M, N)

        # Run grid search
        all_results = []
        combination_idx = 0
        skipped_invalid = 0  # Invalid hyperparameter combinations
        skipped_errors = 0   # Other errors
        
        for model_name in model_names:
            model_dir = Path(model_root) / model_name
            
            # Find best checkpoint
            checkpoint_name = self._find_best_checkpoint(model_dir)
            if checkpoint_name is None:
                self.logger.warning(f"No checkpoint found for {model_name}, skipping")
                continue
            
            for Q1 in job_list['Q1']:
                for Q2 in job_list['Q2']:
                    for t_delta in job_list['t_delta']:
                        for method in denoise_methods:
                            combination_idx += 1
                            self.logger.info(
                                f"\n[{combination_idx}/{total_combinations}] "
                                f"Testing: {model_name} | "
                                f"Q1={Q1}, Q2={Q2}, t_delta={t_delta}, method={method}"
                            )
                            
                            # Create manual config for this combination
                            manual_config = {
                                "Q1": Q1,
                                "Q2": Q2,
                                "t_delta": t_delta
                            }
                            
                            # Run evaluation with manual config
                            try:
                                result = self.trajectory_evaluator.evaluate_model_with_config(
                                    model_name=model_name,
                                    model_dir=str(model_dir.absolute()),
                                    checkpoint_name=checkpoint_name,
                                    denoise_method=method,
                                    test_trajectories=test_trajectories,
                                    manual_config=manual_config,
                                )

                                all_results.append(result)

                            except AssertionError as e:
                                # Invalid hyperparameter combination (Q1/Q2 constraints violated)
                                skipped_invalid += 1
                                self.logger.warning(
                                    f"Skipping invalid hyperparameters: {model_name}, "
                                    f"Q1={Q1}, Q2={Q2}, t_delta={t_delta}, {method}"
                                )
                                self.logger.warning(f"Reason: {e}")
                                continue

                            except Exception as e:
                                # Other errors (file not found, computation errors, etc.)
                                skipped_errors += 1
                                self.logger.error(
                                    f"Failed combination: {model_name}, "
                                    f"Q1={Q1}, Q2={Q2}, t_delta={t_delta}, {method}"
                                )
                                self.logger.error(f"Error: {e}")
                                continue
        
        # Summary
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Grid Search Summary")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total combinations tested: {combination_idx}")
        self.logger.info(f"Successful evaluations: {len(all_results)}")
        self.logger.info(f"Skipped (invalid hyperparameters): {skipped_invalid}")
        self.logger.info(f"Skipped (errors): {skipped_errors}")
        self.logger.info(f"{'='*60}")

        return all_results

    # ============================================================
    # HELPER: Discover models
    # ============================================================
    def _discover_models(self, model_root: str) -> List[str]:
        """
        Find all models with best checkpoints.
        
        Parameters:
            model_root: root directory containing models
            
        Returns:
            List of model names
        """
        model_root = Path(model_root)
        
        if not model_root.exists():
            raise FileNotFoundError(f"Model root not found: {model_root}")
        
        model_names = []
        for model_dir in model_root.iterdir():
            if not model_dir.is_dir():
                continue
            
            # Check for checkpoints
            has_checkpoints = False
            for ckpt_dir_name in ["best_ckpt", "ckpts"]:
                ckpt_dir = model_dir / ckpt_dir_name
                if ckpt_dir.exists() and any(ckpt_dir.glob("*_full.pt")):
                    has_checkpoints = True
                    break
            
            if has_checkpoints:
                model_names.append(model_dir.name)
        
        return sorted(model_names)
    
    
    def _find_best_checkpoint(self, model_dir: Path) -> Optional[str]:
        """
        Find best checkpoint in model directory.
        
        Parameters:
            model_dir: path to model directory
            
        Returns:
            Checkpoint filename, or None if not found
        """
        # Try best_ckpt directory first
        best_ckpt_dir = model_dir / "best_ckpt"
        if best_ckpt_dir.exists():
            best_ckpts = list(best_ckpt_dir.glob("*_full.pt"))
            if best_ckpts:
                return best_ckpts[0].name
        
        # Fallback to ckpts directory
        ckpts_dir = model_dir / "ckpts"
        if not ckpts_dir.exists():
            return None
        
        all_ckpts = sorted(ckpts_dir.glob("*_full.pt"), key=lambda p: p.stat().st_mtime)
        if all_ckpts:
            return all_ckpts[-1].name
        
        return None
    
    
    def _load_model_config(self, model_dir: Path) -> Dict:
        """
        Load model configuration from log/config.json.
        
        Parameters:
            model_dir: path to model directory
            
        Returns:
            Configuration dictionary
        """
        config_path = model_dir / "log" / "config.json"
        if not config_path.exists():
            self.logger.warning(f"Config not found: {config_path}")
            return {}
        
        with open(config_path, "r") as f:
            return json.load(f)
    
    
    def _load_or_generate_test_data(self, test_data_path: str, M: int, N: int) -> List:
        """
        Load test trajectories from file or generate new dataset if not found.

        Parameters:
            test_data_path: path to directory containing trajectory .pt files
            M: number of trajectories
            N: number of points per trajectory

        Returns:
            List of trajectory objects
        """
        test_dir = Path(test_data_path)
        test_dir.mkdir(parents=True, exist_ok=True)

        # Calculate median length target (approximate)
        # The actual median might vary slightly, so we search for files close to target
        target_median_min = int(N * 0.8)  # Allow 80% of target
        target_median_max = int(N * 1.2)  # Allow 120% of target

        # Search for existing dataset matching M and N
        matching_file = None
        for pt_file in test_dir.glob("fulltraj_*.pt"):
            # Parse filename: fulltraj_{n_trajectories}_{median_length}.pt
            parts = pt_file.stem.split("_")
            if len(parts) != 3:
                continue

            try:
                file_M = int(parts[1])
                file_median = int(parts[2])

                # Check if M matches and median is in acceptable range
                if file_M == M and target_median_min <= file_median <= target_median_max:
                    matching_file = pt_file
                    self.logger.info(
                        f"Found matching dataset: {pt_file.name} "
                        f"(M={file_M}, median={file_median}, target N={N})"
                    )
                    break
            except (ValueError, IndexError):
                continue

        # If no matching file found, generate new dataset
        if matching_file is None:
            self.logger.info(
                f"No matching dataset found for M={M}, N={N}. "
                f"Generating new dataset..."
            )

            # Import traj_extractor dynamically
            extractor_path = Path("./src/utils/data_processor")
            if str(extractor_path) not in sys.path:
                sys.path.insert(0, str(extractor_path))

            from traj_extractor import traj_extractor  # noqa: E402 (import not at top)

            # Generate dataset
            result = traj_extractor(
                parquet_dir="./dataset/raw",
                M=M,
                N=N,
                output_dir=str(test_dir)
            )

            matching_file = Path(result["output_file"])
            self.logger.info(
                f"Generated new dataset: {matching_file.name} "
                f"({result['n_trajectories']} trajectories, "
                f"{result['total_points']} total points)"
            )

        # Load the dataset
        self.logger.info(f"Loading test trajectories from {matching_file.name}")
        data = torch.load(matching_file, map_location="cpu")
        raw_trajectories = data["trajectories"]

        # Convert to simple namespace objects
        trajectories = []
        for traj_dict in raw_trajectories:
            traj_obj = SimpleNamespace(
                agent_id=traj_dict["agent_id"],
                n_points=traj_dict["n_points"],
                noisy_gps=traj_dict["data"].numpy(),   # (N, 2) [lon, lat]
                clean_gps=traj_dict["label"].numpy(),  # (N, 2) [lon, lat]
            )
            trajectories.append(traj_obj)

        self.logger.info(f"Loaded {len(trajectories)} trajectories")
        return trajectories


# ================================================================
# MAIN
# ================================================================
def main():
    """Entry point for running evaluations."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    manager = EvaluationManager(output_dir="./bin/test_results")

    # ================================================================
    # Dataset configuration
    # ================================================================
    M = 200  # Number of trajectories
    D = 1  # Days per trajectory (1 day = 8640 points @ 10-sec sampling)

    # ================================================================
    # OPTION 1: Simple evaluation - Test models with default hyperparameters
    # ================================================================
    # Uses each model's trained Q1, Q2, t_delta from config.json
    # Tests BF vs DF methods only (fast)

    # results = manager.run_trajectory_evaluation(
    #     model_names=None,
    #     denoise_methods=["BF", "DF"],
    #     M=M,
    #     D=D
    # )

    # ================================================================
    # OPTION 2: Grid search - Find optimal hyperparameters (SLOW!)
    # ================================================================
    # Uncomment to test multiple Q1, Q2, t_delta combinations
    # Example: 4×4×2 params × 2 methods × 10 models = 640 runs!

    job_list = {
        "Q1": [1, 4, 8],
        "Q2": [4, 8, 16],
        "t_delta": [1, 0.5, 0.2, 0.1, 0.05]
    }
    
    results = manager.run_grid_search_evaluation(
        model_names=None,
        denoise_methods=["BF"],
        job_list=job_list,
        M=M,
        D=D
    )

    print(f"\n✓ Evaluation complete")
    

if __name__ == "__main__":
    main()