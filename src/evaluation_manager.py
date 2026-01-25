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
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch


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
                "model_name,denoise_method,avg_l2_err_pw,med_l2_err_pw,std_l2_err_pw,"
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
        denoised_trajectories, errors = self._denoise_trajectories(
            checkpoint_path, test_trajectories, denoise_method
        )
        
        # 3. Compute metrics at different granularities
        pw_metrics = self._compute_pointwise_metrics(errors)
        bw_metrics = self._compute_bytewise_metrics(test_trajectories, errors)
        cw_metrics = self._compute_chunkwise_metrics(test_trajectories, errors, K, Q1, Q2)
        
        # 4. Measure execution time
        longest_traj = max(test_trajectories, key=lambda t: len(t.noisy_gps))
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
            (denoised_trajectories, all_errors_array)
        """
        assert method in ["BF", "DF"], f"Invalid method: {method}"
        
        # Initialize EncoderDecoder
        from encoder_decoder import EncoderDecoder
        decoder = EncoderDecoder(checkpoint_path)
        self.logger.info(f"Initialized EncoderDecoder with {checkpoint_path}")
        
        denoised_trajectories = []
        all_errors = []
        
        for idx, traj_obj in enumerate(test_trajectories):
            noisy_gps = traj_obj.noisy_gps  # (T, 2) [lon, lat]
            clean_gps = traj_obj.clean_gps  # (T, 2) [lon, lat]
            
            # Run denoising
            if method == "BF":
                result = decoder.denoise_traj_BF(noisy_gps)
                if result["error_code"] != 0:
                    self.logger.warning(f"BF denoising failed for trajectory {idx}")
                    continue
                denoised_gps = result["traj_clean"]
            else:  # DF
                denoised_gps = decoder.denoise_traj_DF(noisy_gps)
            
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
        return denoised_trajectories, all_errors_array
    
    
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
        for _ in range(5):
            start = time.time()
            
            if method == "BF":
                result = decoder.denoise_traj_BF(longest_trajectory.noisy_gps)
            else:  # DF
                _ = decoder.denoise_traj_DF(longest_trajectory.noisy_gps)
            
            end = time.time()
            times.append(end - start)
        
        return float(np.mean(times))
    
    
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
    ) -> List[Dict]:
        """
        Run trajectory evaluation on specified models.
        
        Parameters:
            model_names: specific models to test, or None for all
            denoise_methods: methods to test, default ["BF", "DF"]
            model_root: root directory containing model subdirectories
            test_data_path: path to test trajectory data
            
        Returns:
            List of result dictionaries
        """
        if denoise_methods is None:
            denoise_methods = ["BF", "DF"]
        
        # Discover models
        if model_names is None:
            model_names = self._discover_models(model_root)
        
        self.logger.info(f"Found {len(model_names)} models to evaluate")
        
        # Load test trajectories
        test_trajectories = self._load_test_trajectories(test_data_path)
        
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
    
    
    def _load_test_trajectories(self, test_data_path: str) -> List:
        """
        Load test dataset trajectories from .pt file.
        
        Parameters:
            test_data_path: path to directory containing trajectory .pt files
            
        Returns:
            List of trajectory objects
        """
        test_dir = Path(test_data_path)
        
        if not test_dir.exists():
            raise FileNotFoundError(f"Test data directory not found: {test_dir}")
        
        # Find .pt file
        pt_files = list(test_dir.glob("*.pt"))
        if len(pt_files) == 0:
            raise FileNotFoundError(f"No .pt files found in {test_dir}")
        
        pt_file = pt_files[0]
        self.logger.info(f"Loading test trajectories from {pt_file.name}")
        
        data = torch.load(pt_file, map_location="cpu")
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
    results = manager.run_trajectory_evaluation()
    
    print(f"\n✓ Evaluation complete: {len(results)} tests run")


if __name__ == "__main__":
    main()