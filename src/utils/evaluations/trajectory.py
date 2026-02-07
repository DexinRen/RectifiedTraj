import csv
import json
import logging
import sys
import re
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

import encoder_decoder
from encoder_decoder import EncoderDecoder
from theta_model import build_theta_model

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
np.seterr(all="ignore")


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
        self.output_dir = Path(output_dir)
        self.parquet_dir = self.output_dir / "trajectory_evaluation_results"
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir: Optional[Path] = None

        self.csv_path = self.output_dir / "trajectory_evaluation_summary.csv"
        self.logger = logging.getLogger("TrajectoryEvaluator")

        header_cols = [
            "model_name",
            "model_tag",
            "dataset_name",
            "denoise_method",
            "K",
            "Q1",
            "Q2",
            "t_delta",
            "N_steps",
            "avg_l2_err_pw",
            "med_l2_err_pw",
            "std_l2_err_pw",
            "avg_denoise_time_sec",
            "avg_denoise_time_sec_per_point",
            "num_tested_trajectories",
            "num_tested_points",
            "test_timestamp",
        ]

        if not self.csv_path.exists():
            self.csv_path.write_text(",".join(header_cols) + "\n")
        else:
            with open(self.csv_path, "r", newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)

            if not rows:
                self.csv_path.write_text(",".join(header_cols) + "\n")
            else:
                header = rows[0]
                if header != header_cols:
                    fixed_rows = [header_cols]
                    for row in rows[1:]:
                        row_map = {k: v for k, v in zip(header, row)}
                        fixed_rows.append([row_map.get(col, "") for col in header_cols])
                    with open(self.csv_path, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(fixed_rows)

    def set_run_context(self, dataset_name: str) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"{dataset_name}_{timestamp}"
        self.parquet_dir = self.run_dir / "raw"
        self.parquet_dir.mkdir(parents=True, exist_ok=True)

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
        dataset_name: Optional[str] = None,
    ) -> Dict:
        self.logger.info(f"Evaluating {model_name} with {denoise_method}")

        checkpoint_path = self._get_checkpoint_path(model_dir, checkpoint_name)
        if checkpoint_path is None:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_name}")

        denoised_trajectories, errors, decoder = self._denoise_trajectories(
            checkpoint_path, test_trajectories, denoise_method
        )

        t_delta = decoder.t_delta
        N_steps = int(1.0 / t_delta) if t_delta > 0 else 10

        self.logger.info(f"Denoising parameters: t_delta={t_delta}, N_steps={N_steps}")

        self.logger.info("Computing point-wise metrics...")
        pw_metrics = self._compute_pointwise_metrics(errors)

        self.logger.info("Computing byte-wise metrics...")
        bw_metrics = self._compute_bytewise_metrics(test_trajectories, errors)

        self.logger.info("Computing chunk-wise metrics...")
        cw_metrics = self._compute_chunkwise_metrics(test_trajectories, errors, K, Q1, Q2)

        longest_traj = max(test_trajectories, key=lambda t: len(t.noisy_gps))
        self.logger.info(
            f"Measuring timing (5 runs on longest trajectory: {len(longest_traj.noisy_gps)} points)..."
        )
        avg_time = self._measure_timing(checkpoint_path, longest_traj, denoise_method)
        avg_time_per_point = avg_time / len(longest_traj.noisy_gps) if len(longest_traj.noisy_gps) else None

        results = {
            "model_name": model_name,
            "model_tag": "RectifiedTraj",
            "dataset_name": dataset_name,
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
            "avg_l2_err_pw": pw_metrics["avg"],
            "med_l2_err_pw": pw_metrics["med"],
            "std_l2_err_pw": pw_metrics["std"],
            "avg_l2_err_bw": bw_metrics["avg_list"],
            "avg_l2_err_bw_norm": bw_metrics["avg_list_norm"],
            "avg_l2_err_cw": cw_metrics["avg_list"],
            "avg_l2_err_cw_norm": cw_metrics["avg_list_norm"],
            "avg_denoise_time_sec": avg_time,
            "avg_denoise_time_sec_per_point": avg_time_per_point,
        }

        self._save_results(results)

        self.logger.info(f"Evaluation complete: {model_name} {denoise_method}")
        return results

    def evaluate_model_with_config(
        self,
        model_name: str,
        model_dir: str,
        checkpoint_name: str,
        denoise_method: str,
        test_trajectories: List,
        manual_config: Dict,
        dataset_name: Optional[str] = None,
    ) -> Dict:
        req_Q1 = manual_config["Q1"]
        req_Q2 = manual_config["Q2"]
        req_t_delta = manual_config["t_delta"]

        checkpoint_path = self._get_checkpoint_path(model_dir, checkpoint_name)
        if checkpoint_path is None:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_name}")

        denoised_trajectories, errors, decoder = self._denoise_trajectories_with_config(
            checkpoint_path, test_trajectories, denoise_method, manual_config
        )

        actual_t_delta = decoder.t_delta
        actual_Q1 = decoder.Q1_bytes
        actual_Q2 = decoder.Q2_bytes
        N_steps = int(1.0 / actual_t_delta) if actual_t_delta > 0 else 10

        pw_metrics = self._compute_pointwise_metrics(errors)
        bw_metrics = self._compute_bytewise_metrics(test_trajectories, errors)
        cw_metrics = self._compute_chunkwise_metrics(
            test_trajectories, errors, decoder.K, actual_Q1, actual_Q2
        )

        longest_traj = max(test_trajectories, key=lambda t: len(t.noisy_gps))
        avg_time = self._measure_timing_with_config(
            checkpoint_path, longest_traj, denoise_method, manual_config
        )
        avg_time_per_point = avg_time / len(longest_traj.noisy_gps) if len(longest_traj.noisy_gps) else None

        results = {
            "model_name": model_name,
            "model_tag": "RectifiedTraj",
            "dataset_name": dataset_name,
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
            "avg_l2_err_pw": pw_metrics["avg"],
            "med_l2_err_pw": pw_metrics["med"],
            "std_l2_err_pw": pw_metrics["std"],
            "avg_l2_err_bw": bw_metrics["avg_list"],
            "avg_l2_err_bw_norm": bw_metrics["avg_list_norm"],
            "avg_l2_err_cw": cw_metrics["avg_list"],
            "avg_l2_err_cw_norm": cw_metrics["avg_list_norm"],
            "avg_denoise_time_sec": avg_time,
            "avg_denoise_time_sec_per_point": avg_time_per_point,
        }

        self._save_results(results)
        return results

    def _denoise_trajectories_with_config(
        self,
        checkpoint_path: str,
        test_trajectories: List,
        method: str,
        manual_config: Dict,
    ) -> tuple:
        assert method in ["BF", "DF"], f"Invalid method: {method}"

        self._patch_encoder_decoder_checkpoint_loading()

        decoder = EncoderDecoder(checkpoint_path, manual_config=manual_config)

        denoised_trajectories = []
        all_errors = []

        for idx, traj_obj in enumerate(test_trajectories):
            if hasattr(self, "progress_tracker") and self.progress_tracker is not None:
                self.progress_tracker.update(traj=idx + 1, total_traj=len(test_trajectories))

            noisy_gps = traj_obj.noisy_gps
            clean_gps = traj_obj.clean_gps

            if method == "BF":
                denoised_gps = decoder.denoise_traj_BF(noisy_gps)
            else:
                denoised_gps = decoder.denoise_traj_DF(noisy_gps)

            T_denoised = len(denoised_gps)
            clean_gps_aligned = clean_gps[-T_denoised:]

            ref_lat = float(clean_gps_aligned[0, 1])
            ref_lon = float(clean_gps_aligned[0, 0])
            enu_denoised = self._gps_to_enu_batch(denoised_gps, ref_lat, ref_lon)
            enu_clean = self._gps_to_enu_batch(clean_gps_aligned, ref_lat, ref_lon)

            with np.errstate(all="ignore"):
                errors = np.linalg.norm(enu_denoised - enu_clean, axis=1)

            denoised_trajectories.append(denoised_gps)
            all_errors.append(errors)

        if len(all_errors) == 0:
            raise RuntimeError("No trajectories successfully denoised")

        all_errors_array = np.concatenate(all_errors, axis=0)
        return denoised_trajectories, all_errors_array, decoder

    def _measure_timing_with_config(
        self,
        checkpoint_path: str,
        longest_trajectory,
        method: str,
        manual_config: Dict,
    ) -> float:
        decoder = EncoderDecoder(checkpoint_path, manual_config=manual_config)

        times = []
        for _ in range(5):
            start = time.time()

            if method == "BF":
                _ = decoder.denoise_traj_BF(longest_trajectory.noisy_gps)
            else:
                _ = decoder.denoise_traj_DF(longest_trajectory.noisy_gps)

            end = time.time()
            times.append(end - start)

        avg = float(np.mean(times))
        return avg

    def _denoise_trajectories(
        self,
        checkpoint_path: str,
        test_trajectories: List,
        method: str
    ) -> tuple:
        assert method in ["BF", "DF"], f"Invalid method: {method}"

        self._patch_encoder_decoder_checkpoint_loading()

        decoder = EncoderDecoder(checkpoint_path)
        self.logger.info(f"Initialized EncoderDecoder with {checkpoint_path}")

        denoised_trajectories = []
        all_errors = []

        self.logger.info(f"Starting denoising of {len(test_trajectories)} trajectories...")

        for idx, traj_obj in enumerate(test_trajectories):
            if hasattr(self, "progress_tracker") and self.progress_tracker is not None:
                self.progress_tracker.update(traj=idx + 1, total_traj=len(test_trajectories))

            noisy_gps = traj_obj.noisy_gps
            clean_gps = traj_obj.clean_gps

            if method == "BF":
                denoised_gps = decoder.denoise_traj_BF(noisy_gps)
            else:
                denoised_gps = decoder.denoise_traj_DF(noisy_gps)

            T_denoised = len(denoised_gps)
            clean_gps_aligned = clean_gps[-T_denoised:]

            ref_lat = float(clean_gps_aligned[0, 1])
            ref_lon = float(clean_gps_aligned[0, 0])
            enu_denoised = self._gps_to_enu_batch(denoised_gps, ref_lat, ref_lon)
            enu_clean = self._gps_to_enu_batch(clean_gps_aligned, ref_lat, ref_lon)

            errors = np.linalg.norm(enu_denoised - enu_clean, axis=1)

            denoised_trajectories.append(denoised_gps)
            all_errors.append(errors)

        if len(all_errors) == 0:
            raise RuntimeError("No trajectories successfully denoised")

        all_errors_array = np.concatenate(all_errors, axis=0)

        self.logger.info(f"Denoised {len(denoised_trajectories)} trajectories")

        return denoised_trajectories, all_errors_array, decoder

    def _gps_to_enu_batch(
        self,
        gps_coords: np.ndarray,
        ref_lat: float,
        ref_lon: float
    ) -> np.ndarray:
        import pymap3d as pm

        lons = gps_coords[:, 0]
        lats = gps_coords[:, 1]

        e, n, u = pm.geodetic2enu(lats, lons, 0, ref_lat, ref_lon, 0)

        return np.stack([e, n], axis=1)

    @staticmethod
    def _patch_encoder_decoder_checkpoint_loading():
        original_load = encoder_decoder.load_model_from_config

        def patched_load(config_json_path: Path, ckpt_path: Path):
            cfg = json.loads(Path(config_json_path).read_text())
            runtime = {"config": cfg}

            model = build_theta_model(runtime).to(encoder_decoder.DEVICE)

            ckpt_path = Path(ckpt_path)

            if ckpt_path.suffix == ".safetensors":
                sd = encoder_decoder.load_safetensors(str(ckpt_path))
            else:
                blob = torch.load(str(ckpt_path), map_location=encoder_decoder.DEVICE)

                if isinstance(blob, dict) and "model_state_dict" in blob:
                    sd = blob["model_state_dict"]
                else:
                    sd = blob

            model.load_state_dict(sd)
            model.eval()
            return model, cfg

        encoder_decoder.load_model_from_config = patched_load

    def _compute_pointwise_metrics(self, errors: np.ndarray) -> Dict:
        return {
            "avg": float(np.mean(errors)),
            "med": float(np.median(errors)),
            "std": float(np.std(errors)),
        }

    def _compute_bytewise_metrics(
        self,
        trajectories: List,
        errors: np.ndarray
    ) -> Dict:
        max_length = max(len(t.noisy_gps) for t in trajectories)

        pw_sum = np.zeros(max_length, dtype=float)
        pw_count = np.zeros(max_length, dtype=int)

        error_idx = 0
        for traj_obj in trajectories:
            traj_length = len(traj_obj.noisy_gps)
            traj_errors = errors[error_idx: error_idx + traj_length]

            pw_sum[:traj_length] += traj_errors
            pw_count[:traj_length] += 1

            error_idx += traj_length

        pw_list = np.divide(pw_sum, pw_count, where=pw_count > 0, out=np.zeros_like(pw_sum))

        num_bytes = int(np.ceil(max_length / 8))
        avg_l2_err_bw = []

        for byte_idx in range(num_bytes):
            start = byte_idx * 8
            end = min(start + 8, max_length)

            byte_errors = pw_list[start:end]
            byte_avg = float(np.mean(byte_errors[byte_errors > 0]))
            avg_l2_err_bw.append(byte_avg)

        bw_mean = np.mean([x for x in avg_l2_err_bw if x > 0])
        avg_l2_err_bw_norm = [
            x / bw_mean if bw_mean > 0 else 0.0
            for x in avg_l2_err_bw
        ]

        return {
            "avg_list": avg_l2_err_bw,
            "avg_list_norm": avg_l2_err_bw_norm,
        }

    def _compute_chunkwise_metrics(
        self,
        trajectories: List,
        errors: np.ndarray,
        K: int,
        Q1: int,
        Q2: int
    ) -> Dict:
        max_length = max(len(t.noisy_gps) for t in trajectories)

        pw_sum = np.zeros(max_length, dtype=float)
        pw_count = np.zeros(max_length, dtype=int)

        error_idx = 0
        for traj_obj in trajectories:
            traj_length = len(traj_obj.noisy_gps)
            traj_errors = errors[error_idx: error_idx + traj_length]

            pw_sum[:traj_length] += traj_errors
            pw_count[:traj_length] += 1

            error_idx += traj_length

        pw_list = np.divide(pw_sum, pw_count, where=pw_count > 0, out=np.zeros_like(pw_sum))

        Q1_points = Q1 * 8
        Q2_points = Q2 * 8
        stride = K - Q1_points - Q2_points

        num_chunks = 1
        remaining = max_length - K
        if remaining > 0:
            num_chunks += int(np.ceil(remaining / stride))

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

        cw_mean = np.mean([x for x in avg_l2_err_cw if x > 0])
        avg_l2_err_cw_norm = [
            x / cw_mean if cw_mean > 0 else 0.0
            for x in avg_l2_err_cw
        ]

        return {
            "avg_list": avg_l2_err_cw,
            "avg_list_norm": avg_l2_err_cw_norm,
        }

    def evaluate_baseline(self, test_trajectories: List, dataset_name: Optional[str] = None) -> Dict:
        baseline_k = 256
        baseline_q1 = 1
        baseline_q2 = 12
        all_errors = []
        for traj_obj in test_trajectories:
            noisy_gps = traj_obj.noisy_gps
            clean_gps = traj_obj.clean_gps

            ref_lat = float(clean_gps[0, 1])
            ref_lon = float(clean_gps[0, 0])
            enu_noisy = self._gps_to_enu_batch(noisy_gps, ref_lat, ref_lon)
            enu_clean = self._gps_to_enu_batch(clean_gps, ref_lat, ref_lon)
            errors = np.linalg.norm(enu_noisy - enu_clean, axis=1)
            all_errors.append(errors)

        if not all_errors:
            raise RuntimeError("No trajectories for baseline evaluation")

        errors = np.concatenate(all_errors, axis=0)
        pw_metrics = self._compute_pointwise_metrics(errors)
        bw_metrics = self._compute_bytewise_metrics(test_trajectories, errors)
        cw_metrics = self._compute_chunkwise_metrics(
            test_trajectories, errors, baseline_k, baseline_q1, baseline_q2
        )

        results = {
            "model_name": "test data",
            "model_tag": "Baseline",
            "dataset_name": dataset_name,
            "model_dir": None,
            "checkpoint_name": None,
            "K": baseline_k,
            "Q1": baseline_q1,
            "Q2": baseline_q2,
            "t_delta": 1.0,
            "N_steps": 1,
            "denoise_method": "N/A",
            "test_timestamp": datetime.now().isoformat(),
            "num_tested_trajectories": len(test_trajectories),
            "num_tested_points": int(sum(len(t.noisy_gps) for t in test_trajectories)),
            "longest_trajectory_length": int(max(len(t.noisy_gps) for t in test_trajectories)),
            "avg_l2_err_pw": pw_metrics["avg"],
            "med_l2_err_pw": pw_metrics["med"],
            "std_l2_err_pw": pw_metrics["std"],
            "avg_l2_err_bw": bw_metrics["avg_list"],
            "avg_l2_err_bw_norm": bw_metrics["avg_list_norm"],
            "avg_l2_err_cw": cw_metrics["avg_list"],
            "avg_l2_err_cw_norm": cw_metrics["avg_list_norm"],
            "avg_denoise_time_sec": None,
            "avg_denoise_time_sec_per_point": None,
        }

        self._save_results(results)
        return results

    def _measure_timing(
        self,
        checkpoint_path: str,
        longest_trajectory,
        method: str
    ) -> float:
        decoder = EncoderDecoder(checkpoint_path)

        times = []
        for run_idx in range(5):
            self.logger.debug(f"  Timing run {run_idx + 1}/5...")
            start = time.time()

            if method == "BF":
                _ = decoder.denoise_traj_BF(longest_trajectory.noisy_gps)
            else:
                _ = decoder.denoise_traj_DF(longest_trajectory.noisy_gps)

            end = time.time()
            elapsed = end - start
            times.append(elapsed)
            self.logger.debug(f"  Run {run_idx + 1} completed in {elapsed:.2f}s")

        avg = float(np.mean(times))
        self.logger.info(f"Average timing: {avg:.2f}s over 5 runs")
        return avg

    def _save_results(self, results: Dict):
        def _fmt(value, fmt: str):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return "NA"
            return format(value, fmt)

        df = pd.DataFrame([results])

        def _tag(value, fmt: str, prefix: str):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return f"{prefix}NA"
            return f"{prefix}{format(value, fmt)}"

        def _safe_name(value: str) -> str:
            return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))

        safe_model = _safe_name(results.get("model_name", "unknown"))
        safe_method = _safe_name(results.get("denoise_method", "NA"))
        safe_dataset = _safe_name(results.get("dataset_name", "unknown"))
        parquet_filename = (
            f"{safe_model}_{safe_method}_{safe_dataset}"
            f"_{_tag(results.get('K'), 'd', 'K')}"
            f"_{_tag(results.get('Q1'), 'd', 'Q1')}"
            f"_{_tag(results.get('Q2'), 'd', 'Q2')}"
            f"_{_tag(results.get('t_delta'), '.4f', 'td')}"
            f"_{_tag(results.get('N_steps'), 'd', 'N')}"
            f"_{results.get('test_timestamp', '').replace(':', '').replace('-', '').replace('.', '')}.parquet"
        )
        parquet_path = self.parquet_dir / parquet_filename
        df.to_parquet(parquet_path, index=False)

        dataset_name = results.get("dataset_name") or "NA"
        csv_row = (
            f"{results['model_name']},{results.get('model_tag', 'NA')},{dataset_name},"
            f"{results['denoise_method']},"
            f"{_fmt(results.get('K'), 'd')},{_fmt(results.get('Q1'), 'd')},{_fmt(results.get('Q2'), 'd')},"
            f"{_fmt(results.get('t_delta'), '.4f')},{_fmt(results.get('N_steps'), 'd')},"
            f"{_fmt(results.get('avg_l2_err_pw'), '.6f')},{_fmt(results.get('med_l2_err_pw'), '.6f')},"
            f"{_fmt(results.get('std_l2_err_pw'), '.6f')},"
            f"{_fmt(results.get('avg_denoise_time_sec'), '.6f')},"
            f"{_fmt(results.get('avg_denoise_time_sec_per_point'), '.8f')},"
            f"{_fmt(results.get('num_tested_trajectories'), 'd')},{_fmt(results.get('num_tested_points'), 'd')},"
            f"{results.get('test_timestamp')}\n"
        )
        with open(self.csv_path, "a") as f:
            f.write(csv_row)

    def _get_checkpoint_path(self, model_dir: str, checkpoint_name: str) -> Optional[str]:
        for ckpt_dir_name in ["best_ckpt", "ckpts"]:
            ckpt_dir = Path(model_dir) / ckpt_dir_name
            if ckpt_dir.exists():
                ckpt_path = ckpt_dir / checkpoint_name
                if ckpt_path.exists():
                    return str(ckpt_path.absolute())

        return None


class ClassicBaselineEvaluator:
    """
    Evaluate classic baselines on GPS trajectories.

    Notes:
        - Uses GPS order as given (time order preserved).
        - No map data required.
        - Operates in ENU space for metric errors.
    """

    def __init__(self, trajectory_evaluator: TrajectoryEvaluator):
        self.trajectory_evaluator = trajectory_evaluator
        self.logger = logging.getLogger("ClassicBaselineEvaluator")
        self.progress_bar = True

    def _progress(self, message: str) -> None:
        if not self.progress_bar:
            return
        sys.stdout.write("\r\033[K" + message)
        sys.stdout.flush()

    def evaluate_classic_baselines(
        self,
        test_trajectories: List,
        dataset_name: Optional[str] = None,
        methods: Optional[List[str]] = None,
    ) -> List[Dict]:
        baseline_k = 256
        baseline_q1 = 1
        baseline_q2 = 12
        from baseline import classic as classic_baseline

        available_methods = [
            ("kalman_rts_ts", classic_baseline.kalman_rts_smoother),
            ("kalman_rts_notime", classic_baseline.kalman_rts_smoother),
            ("hampel", classic_baseline.hampel_filter),
            ("savgol", classic_baseline.savitzky_golay_filter),
            ("spline", classic_baseline.smoothing_spline),
            ("raw", classic_baseline.raw_baseline),
        ]
        if methods is None:
            selected = available_methods
        else:
            allowed = set(methods)
            selected = [(name, fn) for name, fn in available_methods if name in allowed]
            missing = [name for name in methods if name not in {n for n, _ in available_methods}]
            for name in missing:
                self.logger.warning("Unknown classic baseline ignored: %s", name)
        if not selected:
            self.logger.warning("No classic baselines selected; skipping classic baseline evaluation.")
            return []

        results = []
        total_methods = len(selected)
        longest_traj = max(test_trajectories, key=lambda t: len(t.noisy_gps))
        ref_lat = float(longest_traj.clean_gps[0, 1])
        ref_lon = float(longest_traj.clean_gps[0, 0])
        enu_longest = self.trajectory_evaluator._gps_to_enu_batch(
            longest_traj.noisy_gps, ref_lat, ref_lon
        )
        longest_points = len(longest_traj.noisy_gps)
        for idx, (method_name, method_fn) in enumerate(selected, start=1):
            label = f"Baseline [{idx}/{total_methods}]"
            if dataset_name:
                label = f"{label} {dataset_name}"
            self._progress(f"{label} {method_name}")
            self.logger.info("Running classic baseline: %s", method_name)
            all_errors = []

            for traj_obj in test_trajectories:
                noisy_gps = traj_obj.noisy_gps
                clean_gps = traj_obj.clean_gps

                ref_lat = float(clean_gps[0, 1])
                ref_lon = float(clean_gps[0, 0])
                enu_noisy = self.trajectory_evaluator._gps_to_enu_batch(
                    noisy_gps, ref_lat, ref_lon
                )
                enu_clean = self.trajectory_evaluator._gps_to_enu_batch(
                    clean_gps, ref_lat, ref_lon
                )

                if method_name == "kalman_rts_ts":
                    ts = getattr(traj_obj, "timestamps", None)
                    if ts is not None:
                        ts = np.asarray(ts, dtype=np.float64)
                        if ts.size and np.isfinite(ts[0]):
                            ts = ts - float(ts[0])
                    denoised_enu = method_fn(enu_noisy, timestamps=ts)
                elif method_name == "kalman_rts_notime":
                    denoised_enu = method_fn(enu_noisy, timestamps=None)
                else:
                    denoised_enu = method_fn(enu_noisy)

                errors = np.linalg.norm(denoised_enu - enu_clean, axis=1)
                all_errors.append(errors)

            if not all_errors:
                raise RuntimeError(f"No trajectories for classic baseline: {method_name}")

            errors = np.concatenate(all_errors, axis=0)
            pw_metrics = self.trajectory_evaluator._compute_pointwise_metrics(errors)
            bw_metrics = self.trajectory_evaluator._compute_bytewise_metrics(
                test_trajectories, errors
            )
            cw_metrics = self.trajectory_evaluator._compute_chunkwise_metrics(
                test_trajectories, errors, baseline_k, baseline_q1, baseline_q2
            )

            times = []
            for run_idx in range(5):
                start = time.time()
                if method_name == "kalman_rts_ts":
                    ts = getattr(longest_traj, "timestamps", None)
                    if ts is not None:
                        ts = np.asarray(ts, dtype=np.float64)
                        if ts.size and np.isfinite(ts[0]):
                            ts = ts - float(ts[0])
                    _ = method_fn(enu_longest, timestamps=ts)
                elif method_name == "kalman_rts_notime":
                    _ = method_fn(enu_longest, timestamps=None)
                else:
                    _ = method_fn(enu_longest)
                end = time.time()
                times.append(end - start)
            avg_time = float(np.mean(times)) if times else None
            avg_time_per_point = avg_time / longest_points if avg_time is not None and longest_points else None

            result = {
                "model_name": method_name,
                "model_tag": "Baseline",
                "dataset_name": dataset_name,
                "model_dir": None,
                "checkpoint_name": None,
                "K": baseline_k,
                "Q1": baseline_q1,
                "Q2": baseline_q2,
                "t_delta": 1.0,
                "N_steps": 1,
                "denoise_method": "N/A",
                "test_timestamp": datetime.now().isoformat(),
                "num_tested_trajectories": len(test_trajectories),
                "num_tested_points": int(sum(len(t.noisy_gps) for t in test_trajectories)),
                "longest_trajectory_length": int(max(len(t.noisy_gps) for t in test_trajectories)),
                "avg_l2_err_pw": pw_metrics["avg"],
                "med_l2_err_pw": pw_metrics["med"],
                "std_l2_err_pw": pw_metrics["std"],
                "avg_l2_err_bw": bw_metrics["avg_list"],
                "avg_l2_err_bw_norm": bw_metrics["avg_list_norm"],
                "avg_l2_err_cw": cw_metrics["avg_list"],
                "avg_l2_err_cw_norm": cw_metrics["avg_list_norm"],
                "avg_denoise_time_sec": avg_time,
                "avg_denoise_time_sec_per_point": avg_time_per_point,
            }

            self.trajectory_evaluator._save_results(result)
            results.append(result)

        if self.progress_bar:
            done_msg = "Baseline complete"
            if dataset_name:
                done_msg = f"{done_msg} ({dataset_name})"
            sys.stdout.write("\r\033[K" + done_msg + "\n")
            sys.stdout.flush()
        return results
