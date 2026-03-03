import csv
import json
import logging
import os
import psutil
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

import encoder_decoder
from encoder_decoder import EncoderDecoder
from theta_model import build_theta_model

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
np.seterr(all="ignore")


def _runtime_device_label() -> str:
    raw = str(
        os.getenv(
            "RECTIFIEDTRAJ_RUNTIME_DEVICE_EFFECTIVE",
            os.getenv("RECTIFIEDTRAJ_DEVICE", "unknown"),
        )
    ).strip().lower()
    if raw.startswith("cuda"):
        return "cuda"
    if raw == "cpu":
        return "cpu"
    return raw or "unknown"


def _normalize_device_label(raw_device) -> str:
    raw = str(raw_device or "").strip().lower()
    if raw.startswith("cuda"):
        return "cuda"
    if raw == "cpu":
        return "cpu"
    return raw or "unknown"


def _latency_summary(times_sec: list[float]) -> Dict[str, float | None]:
    if not times_sec:
        return {
            "avg_time_sec": None,
            "latency_p50_ms": None,
            "latency_p95_ms": None,
            "latency_max_ms": None,
        }
    arr = np.asarray(times_sec, dtype=float)
    return {
        "avg_time_sec": float(np.mean(arr)),
        "latency_p50_ms": float(np.percentile(arr, 50) * 1000.0),
        "latency_p95_ms": float(np.percentile(arr, 95) * 1000.0),
        "latency_max_ms": float(np.max(arr) * 1000.0),
    }


def _throughput_points_per_sec(avg_time_sec: float | None, n_points: int) -> float | None:
    if avg_time_sec is None:
        return None
    if avg_time_sec <= 0 or int(n_points) <= 0:
        return None
    return float(int(n_points) / float(avg_time_sec))


class TrajectoryEvaluator:
    """
    Evaluate trajectory denoising quality at multiple granularities.

    Outputs:
        - CSV: point-wise summary for quick comparison
        - CSV: trajectory_bytewise_summary.csv (one row per model+config with avg_l2_err_bw)

    Granularities:
        - Point-wise (pw): individual point L2 errors
        - Byte-wise (bw): 8-point groups
        - Chunk-wise (cw): per-chunk errors
    """

    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_dir = self.output_dir / "raw"
        self.run_dir: Optional[Path] = None
        self._active_dataset_name: Optional[str] = None
        self._run_dir_by_dataset: dict[str, Path] = {}

        self.csv_path = self.output_dir / "trajectory_evaluation_summary.csv"
        self.logger = logging.getLogger("TrajectoryEvaluator")

        header_cols = [
            "model_name",
            "model_tag",
            "device",
            "dataset_name",
            "denoise_method",
            "K",
            "Q1",
            "Q2",
            "t_delta",
            "N_steps",
            "avg_l2_err_pw",
            "med_l2_err_pw",
            "p95_l2_err_pw",
            "std_l2_err_pw",
            "avg_denoise_time_sec",
            "avg_denoise_time_sec_per_point",
            "latency_p50_ms",
            "latency_p95_ms",
            "latency_max_ms",
            "throughput_points_per_sec",
            "peak_rss_mb",
            "peak_vram_mb",
            "calibration_time_sec",
            "calibration_peak_rss_mb",
            "calibration_peak_vram_mb",
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
        dataset_key = str(dataset_name)
        cached = self._run_dir_by_dataset.get(dataset_key)
        if cached is not None:
            cached.mkdir(parents=True, exist_ok=True)
            self.run_dir = cached
            self.parquet_dir = self.run_dir / "raw"
            self._active_dataset_name = dataset_key
            return
        if (
            self.run_dir is not None
            and self._active_dataset_name == dataset_key
            and self.run_dir.exists()
        ):
            # Reuse the same dataset-specific run dir within one benchmark run.
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"{dataset_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_dir = self.run_dir / "raw"
        self._active_dataset_name = dataset_key
        self._run_dir_by_dataset[dataset_key] = self.run_dir

    @staticmethod
    def _profile_predict_runs(
        predict_once,
        *,
        repeats: int,
        n_points: int,
    ) -> Dict[str, float | None]:
        device = _runtime_device_label()
        use_cuda = (device == "cuda") and torch.cuda.is_available()
        proc = psutil.Process(os.getpid())
        peak_rss_mb: float | None = None
        peak_vram_mb: float | None = None
        times_sec: list[float] = []

        for _ in range(max(1, int(repeats))):
            rss_before = float(proc.memory_info().rss) / (1024.0 * 1024.0)
            if use_cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            _ = predict_once()
            if use_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            rss_after = float(proc.memory_info().rss) / (1024.0 * 1024.0)
            run_peak_rss = max(rss_before, rss_after)
            peak_rss_mb = run_peak_rss if peak_rss_mb is None else max(peak_rss_mb, run_peak_rss)

            if use_cuda:
                run_peak_vram = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
                peak_vram_mb = run_peak_vram if peak_vram_mb is None else max(peak_vram_mb, run_peak_vram)

            times_sec.append(float(t1 - t0))

        lat = _latency_summary(times_sec)
        avg_time_sec = lat["avg_time_sec"]
        return {
            "avg_time_sec": avg_time_sec,
            "avg_time_sec_per_point": (
                (float(avg_time_sec) / float(n_points))
                if avg_time_sec is not None and int(n_points) > 0
                else None
            ),
            "latency_p50_ms": lat["latency_p50_ms"],
            "latency_p95_ms": lat["latency_p95_ms"],
            "latency_max_ms": lat["latency_max_ms"],
            "throughput_points_per_sec": _throughput_points_per_sec(avg_time_sec, n_points),
            "peak_rss_mb": peak_rss_mb,
            "peak_vram_mb": peak_vram_mb,
        }

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
        model_tag: str = "RectifiedTraj",
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
        timing = self._measure_timing(checkpoint_path, longest_traj, denoise_method)

        results = {
            "model_name": model_name,
            "model_tag": model_tag,
            "device": _normalize_device_label(getattr(encoder_decoder, "DEVICE", None)),
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
            "p95_l2_err_pw": pw_metrics["p95"],
            "std_l2_err_pw": pw_metrics["std"],
            "avg_l2_err_bw": bw_metrics["avg_list"],
            "avg_l2_err_bw_norm": bw_metrics["avg_list_norm"],
            "avg_l2_err_cw": cw_metrics["avg_list"],
            "avg_l2_err_cw_norm": cw_metrics["avg_list_norm"],
            "avg_denoise_time_sec": timing["avg_time_sec"],
            "avg_denoise_time_sec_per_point": timing["avg_time_sec_per_point"],
            "latency_p50_ms": timing["latency_p50_ms"],
            "latency_p95_ms": timing["latency_p95_ms"],
            "latency_max_ms": timing["latency_max_ms"],
            "throughput_points_per_sec": timing["throughput_points_per_sec"],
            "peak_rss_mb": timing["peak_rss_mb"],
            "peak_vram_mb": timing["peak_vram_mb"],
            "calibration_time_sec": 0.0,
            "calibration_peak_rss_mb": None,
            "calibration_peak_vram_mb": None,
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
        model_tag: str = "RectifiedTraj",
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
        timing = self._measure_timing_with_config(
            checkpoint_path, longest_traj, denoise_method, manual_config
        )

        results = {
            "model_name": model_name,
            "model_tag": model_tag,
            "device": _normalize_device_label(getattr(encoder_decoder, "DEVICE", None)),
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
            "p95_l2_err_pw": pw_metrics["p95"],
            "std_l2_err_pw": pw_metrics["std"],
            "avg_l2_err_bw": bw_metrics["avg_list"],
            "avg_l2_err_bw_norm": bw_metrics["avg_list_norm"],
            "avg_l2_err_cw": cw_metrics["avg_list"],
            "avg_l2_err_cw_norm": cw_metrics["avg_list_norm"],
            "avg_denoise_time_sec": timing["avg_time_sec"],
            "avg_denoise_time_sec_per_point": timing["avg_time_sec_per_point"],
            "latency_p50_ms": timing["latency_p50_ms"],
            "latency_p95_ms": timing["latency_p95_ms"],
            "latency_max_ms": timing["latency_max_ms"],
            "throughput_points_per_sec": timing["throughput_points_per_sec"],
            "peak_rss_mb": timing["peak_rss_mb"],
            "peak_vram_mb": timing["peak_vram_mb"],
            "calibration_time_sec": 0.0,
            "calibration_peak_rss_mb": None,
            "calibration_peak_vram_mb": None,
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
    ) -> Dict[str, float | None]:
        decoder = EncoderDecoder(checkpoint_path, manual_config=manual_config)
        if method == "BF":
            predict_once = lambda: decoder.denoise_traj_BF(longest_trajectory.noisy_gps)
        else:
            predict_once = lambda: decoder.denoise_traj_DF(longest_trajectory.noisy_gps)
        return self._profile_predict_runs(
            predict_once,
            repeats=5,
            n_points=len(longest_trajectory.noisy_gps),
        )

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
            "p95": float(np.percentile(errors, 95)),
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
            "device": "cpu",
            "dataset_name": dataset_name,
            "model_dir": None,
            "checkpoint_name": None,
            "K": None,
            "Q1": None,
            "Q2": None,
            "t_delta": None,
            "N_steps": None,
            "denoise_method": "N/A",
            "test_timestamp": datetime.now().isoformat(),
            "num_tested_trajectories": len(test_trajectories),
            "num_tested_points": int(sum(len(t.noisy_gps) for t in test_trajectories)),
            "longest_trajectory_length": int(max(len(t.noisy_gps) for t in test_trajectories)),
            "avg_l2_err_pw": pw_metrics["avg"],
            "med_l2_err_pw": pw_metrics["med"],
            "p95_l2_err_pw": pw_metrics["p95"],
            "std_l2_err_pw": pw_metrics["std"],
            "avg_l2_err_bw": bw_metrics["avg_list"],
            "avg_l2_err_bw_norm": bw_metrics["avg_list_norm"],
            "avg_l2_err_cw": cw_metrics["avg_list"],
            "avg_l2_err_cw_norm": cw_metrics["avg_list_norm"],
            "avg_denoise_time_sec": None,
            "avg_denoise_time_sec_per_point": None,
            "latency_p50_ms": None,
            "latency_p95_ms": None,
            "latency_max_ms": None,
            "throughput_points_per_sec": None,
            "peak_rss_mb": None,
            "peak_vram_mb": None,
            "calibration_time_sec": 0.0,
            "calibration_peak_rss_mb": None,
            "calibration_peak_vram_mb": None,
        }

        self._save_results(results)
        return results

    def _measure_timing(
        self,
        checkpoint_path: str,
        longest_trajectory,
        method: str
    ) -> Dict[str, float | None]:
        decoder = EncoderDecoder(checkpoint_path)
        if method == "BF":
            predict_once = lambda: decoder.denoise_traj_BF(longest_trajectory.noisy_gps)
        else:
            predict_once = lambda: decoder.denoise_traj_DF(longest_trajectory.noisy_gps)
        out = self._profile_predict_runs(
            predict_once,
            repeats=5,
            n_points=len(longest_trajectory.noisy_gps),
        )
        if out["avg_time_sec"] is not None:
            self.logger.info(f"Average timing: {float(out['avg_time_sec']):.2f}s over 5 runs")
        return out

    def _save_results(self, results: Dict):
        results = dict(results)
        results.setdefault("device", _runtime_device_label())
        results.setdefault("calibration_time_sec", None)
        results.setdefault("calibration_peak_rss_mb", None)
        results.setdefault("calibration_peak_vram_mb", None)

        def _fmt(value, fmt: str):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return "NA"
            return format(value, fmt)

        dataset_name = results.get("dataset_name") or "NA"
        csv_row = (
            f"{results['model_name']},{results.get('model_tag', 'NA')},{results.get('device', 'unknown')},{dataset_name},"
            f"{results['denoise_method']},"
            f"{_fmt(results.get('K'), 'd')},{_fmt(results.get('Q1'), 'd')},{_fmt(results.get('Q2'), 'd')},"
            f"{_fmt(results.get('t_delta'), '.4f')},{_fmt(results.get('N_steps'), 'd')},"
            f"{_fmt(results.get('avg_l2_err_pw'), '.6f')},{_fmt(results.get('med_l2_err_pw'), '.6f')},"
            f"{_fmt(results.get('p95_l2_err_pw'), '.6f')},"
            f"{_fmt(results.get('std_l2_err_pw'), '.6f')},"
            f"{_fmt(results.get('avg_denoise_time_sec'), '.6f')},"
            f"{_fmt(results.get('avg_denoise_time_sec_per_point'), '.8f')},"
            f"{_fmt(results.get('latency_p50_ms'), '.4f')},"
            f"{_fmt(results.get('latency_p95_ms'), '.4f')},"
            f"{_fmt(results.get('latency_max_ms'), '.4f')},"
            f"{_fmt(results.get('throughput_points_per_sec'), '.4f')},"
            f"{_fmt(results.get('peak_rss_mb'), '.4f')},"
            f"{_fmt(results.get('peak_vram_mb'), '.4f')},"
            f"{_fmt(results.get('calibration_time_sec'), '.6f')},"
            f"{_fmt(results.get('calibration_peak_rss_mb'), '.4f')},"
            f"{_fmt(results.get('calibration_peak_vram_mb'), '.4f')},"
            f"{_fmt(results.get('num_tested_trajectories'), 'd')},{_fmt(results.get('num_tested_points'), 'd')},"
            f"{results.get('test_timestamp')}\n"
        )
        with open(self.csv_path, "a") as f:
            f.write(csv_row)

        self._append_trajectory_bytewise_row(results)
        self._append_valhalla_info_row(results)

    def _resolve_bytewise_model_label(self, results: Dict) -> str:
        model_tag = str(results.get("model_tag", "")).strip().lower()
        model_name = str(results.get("model_name", "")).strip() or "NA"
        model_dir = str(results.get("model_dir", "") or "").strip()

        if model_tag == "baseline":
            if model_name.startswith("kalman_rts@"):
                return model_name
            if model_name == "kalman_rts":
                mode = str(
                    results.get("calibration_mode")
                    or os.getenv("KALMAN_RTS_CALIBRATION_MODE", "")
                ).strip()
                if mode:
                    return f"kalman_rts@{mode}"
            return model_name

        model_tag_raw = str(results.get("model_tag", "")).strip()
        if model_dir:
            base = Path(model_dir).name.strip()
            if base:
                return f"{model_tag_raw}/{base}" if model_tag_raw else base
            return f"{model_tag_raw}/{model_dir}" if model_tag_raw else model_dir
        return f"{model_tag_raw}/{model_name}" if model_tag_raw else model_name

    def _append_trajectory_bytewise_row(self, results: Dict) -> None:
        bw = results.get("avg_l2_err_bw")
        if not isinstance(bw, (list, tuple, np.ndarray)) or len(bw) == 0:
            return

        out_dir = self.run_dir if self.run_dir is not None else self.output_dir
        out_csv = out_dir / "trajectory_bytewise_summary.csv"

        model_label = self._resolve_bytewise_model_label(results)
        dataset_name = str(results.get("dataset_name", "") or "NA").strip()
        method_name = str(results.get("denoise_method", "") or "NA").strip()

        q1_raw = results.get("Q1")
        q2_raw = results.get("Q2")
        t_delta_raw = results.get("t_delta")

        q1_str = "NA" if q1_raw is None else str(int(round(float(q1_raw))))
        q2_str = "NA" if q2_raw is None else str(int(round(float(q2_raw))))
        if t_delta_raw is None:
            t_delta_str = "NA"
            step_str = "NA"
        else:
            t_delta_value = float(t_delta_raw)
            if t_delta_value <= 0.0:
                t_delta_str = "NA"
                step_str = "NA"
            else:
                t_delta_str = f"{t_delta_value:.4f}"
                step_value = 1.0 / t_delta_value
                step_rounded = int(round(step_value))
                if abs(step_value - step_rounded) <= 1e-9:
                    step_str = str(step_rounded)
                else:
                    step_str = f"{step_value:.6f}".rstrip("0").rstrip(".")

        key = (
            model_label,
            dataset_name,
            method_name,
            q1_str,
            q2_str,
            t_delta_str,
            step_str,
        )

        merged: dict[tuple[str, str, str, str, str, str, str], list[float]] = {}
        if out_csv.exists():
            with out_csv.open("r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    label = str(row.get("model_dir", "")).strip()
                    ds = str(row.get("dataset_name", "")).strip()
                    md = str(row.get("denoise_method", "") or "NA").strip()
                    q1 = str(row.get("Q1", "") or "NA").strip()
                    q2 = str(row.get("Q2", "") or "NA").strip()
                    td = str(row.get("t_delta", "") or "NA").strip()
                    st = str(row.get("step", "") or "NA").strip()
                    if not label:
                        continue
                    vals: list[float] = []
                    i = 0
                    while True:
                        col = f"byte_{i}"
                        if col not in row:
                            break
                        raw = str(row.get(col, "")).strip()
                        if raw in {"", "NA"}:
                            vals.append(float("nan"))
                        else:
                            vals.append(float(raw))
                        i += 1
                    merged[(label, ds, md, q1, q2, td, st)] = vals

        merged[key] = [float(v) for v in np.asarray(bw, dtype=float)]

        max_len = max((len(v) for v in merged.values()), default=0)
        header = [
            "model_dir",
            "dataset_name",
            "denoise_method",
            "Q1",
            "Q2",
            "t_delta",
            "step",
        ] + [f"byte_{i}" for i in range(max_len)]

        with out_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for (label, ds, md, q1, q2, td, st), vals in sorted(merged.items(), key=lambda x: x[0]):
                padded = list(vals) + [float("nan")] * (max_len - len(vals))
                row = [label, ds, md, q1, q2, td, st] + [
                    ("NA" if (isinstance(v, float) and np.isnan(v)) else f"{float(v):.10f}")
                    for v in padded
                ]
                writer.writerow(row)

    def _append_valhalla_info_row(self, results: Dict) -> None:
        valhalla_cols = sorted(
            key for key in results.keys() if str(key).startswith("valhalla_")
        )
        if not valhalla_cols:
            return

        out_dir = self.run_dir if self.run_dir is not None else self.output_dir
        out_csv = out_dir / "baseline_info" / "valhalla.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        base_cols = [
            "test_timestamp",
            "dataset_name",
            "model_name",
            "model_tag",
            "device",
            "num_tested_trajectories",
            "num_tested_points",
        ]

        def _to_cell(value) -> str:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return "NA"
            return str(value)

        row = {
            "test_timestamp": results.get("test_timestamp"),
            "dataset_name": results.get("dataset_name"),
            "model_name": results.get("model_name"),
            "model_tag": results.get("model_tag"),
            "device": results.get("device", _runtime_device_label()),
            "num_tested_trajectories": results.get("num_tested_trajectories"),
            "num_tested_points": results.get("num_tested_points"),
        }
        for key in valhalla_cols:
            row[key] = results.get(key)

        existing_rows: list[dict[str, str]] = []
        merged_header = list(base_cols)
        for col in valhalla_cols:
            if col not in merged_header:
                merged_header.append(col)

        if out_csv.exists():
            with out_csv.open("r", newline="") as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)
                if reader.fieldnames:
                    merged_header = list(reader.fieldnames)
                for col in base_cols + valhalla_cols:
                    if col not in merged_header:
                        merged_header.append(col)

        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=merged_header)
            writer.writeheader()
            for old in existing_rows:
                writer.writerow({col: old.get(col, "NA") for col in merged_header})
            writer.writerow({col: _to_cell(row.get(col)) for col in merged_header})

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
        from baseline import (
            build_lat_lon_timestamp_sequence_from_lonlat,
            create_baseline_model,
            latlon_to_lonlat,
        )

        available_methods = [
            ("kalman_rts", classic_baseline.kalman_rts_smoother),
            ("hampel", classic_baseline.hampel_filter),
            ("savgol", classic_baseline.savitzky_golay_filter),
            ("spline", classic_baseline.smoothing_spline),
            ("raw", classic_baseline.raw_baseline),
            # Placeholder function is unused in model-based execution path.
            ("valhalla_meili", classic_baseline.raw_baseline),
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
        longest_points = len(longest_traj.noisy_gps)
        runtime_device = _runtime_device_label()
        use_cuda_timing = (runtime_device == "cuda") and torch.cuda.is_available()
        proc = psutil.Process(os.getpid())
        for idx, (method_name, _method_fn) in enumerate(selected, start=1):
            label = f"Baseline [{idx}/{total_methods}]"
            if dataset_name:
                label = f"{label} {dataset_name}"
            self._progress(f"{label} {method_name}")
            self.logger.info("Running classic baseline: %s", method_name)
            model = None
            calibration_time_sec = None
            calibration_peak_rss_mb = None
            calibration_peak_vram_mb = None
            try:
                cal_rss_before = float(proc.memory_info().rss) / (1024.0 * 1024.0)
                if use_cuda_timing:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                cal_t0 = time.perf_counter()
                model = create_baseline_model(
                    method_name=method_name,
                    dataset_name=dataset_name,
                )
                if use_cuda_timing:
                    torch.cuda.synchronize()
                cal_t1 = time.perf_counter()
                cal_rss_after = float(proc.memory_info().rss) / (1024.0 * 1024.0)
                calibration_time_sec = float(cal_t1 - cal_t0)
                calibration_peak_rss_mb = max(cal_rss_before, cal_rss_after)
                if use_cuda_timing:
                    calibration_peak_vram_mb = (
                        float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
                    )
            except Exception as exc:
                self.logger.warning("Classic baseline %s initialization failed: %s", method_name, exc)
                continue

            try:
                all_errors = []
                method_failed = False
                method_fail_reason = ""

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

                    try:
                        ts = getattr(traj_obj, "timestamps", None)
                        seq = build_lat_lon_timestamp_sequence_from_lonlat(noisy_gps, timestamps=ts)
                        denoised_latlon = model.predict(seq)
                        denoised_gps = latlon_to_lonlat(denoised_latlon)
                        denoised_enu = self.trajectory_evaluator._gps_to_enu_batch(
                            denoised_gps,
                            ref_lat,
                            ref_lon,
                        )
                    except Exception as exc:
                        method_failed = True
                        method_fail_reason = f"{type(exc).__name__}: {exc}"
                        break

                    errors = np.linalg.norm(denoised_enu - enu_clean, axis=1)
                    all_errors.append(errors)

                if method_failed:
                    self.logger.warning("Skipping classic baseline %s: %s", method_name, method_fail_reason)
                    continue

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

                try:
                    ts = getattr(longest_traj, "timestamps", None)
                    seq = build_lat_lon_timestamp_sequence_from_lonlat(
                        longest_traj.noisy_gps,
                        timestamps=ts,
                    )
                    timing = self.trajectory_evaluator._profile_predict_runs(
                        lambda: model.predict(seq),
                        repeats=5,
                        n_points=longest_points,
                    )
                except Exception:
                    timing = {
                        "avg_time_sec": None,
                        "avg_time_sec_per_point": None,
                        "latency_p50_ms": None,
                        "latency_p95_ms": None,
                        "latency_max_ms": None,
                        "throughput_points_per_sec": None,
                        "peak_rss_mb": None,
                        "peak_vram_mb": None,
                    }

                result = {
                    "model_name": method_name,
                    "model_tag": "Baseline",
                    "device": "cpu",
                    "dataset_name": dataset_name,
                    "model_dir": None,
                    "checkpoint_name": None,
                    "K": None,
                    "Q1": None,
                    "Q2": None,
                    "t_delta": None,
                    "N_steps": None,
                    "denoise_method": "N/A",
                    "test_timestamp": datetime.now().isoformat(),
                    "num_tested_trajectories": len(test_trajectories),
                    "num_tested_points": int(sum(len(t.noisy_gps) for t in test_trajectories)),
                    "longest_trajectory_length": int(max(len(t.noisy_gps) for t in test_trajectories)),
                    "avg_l2_err_pw": pw_metrics["avg"],
                    "med_l2_err_pw": pw_metrics["med"],
                    "p95_l2_err_pw": pw_metrics["p95"],
                    "std_l2_err_pw": pw_metrics["std"],
                    "avg_l2_err_bw": bw_metrics["avg_list"],
                    "avg_l2_err_bw_norm": bw_metrics["avg_list_norm"],
                    "avg_l2_err_cw": cw_metrics["avg_list"],
                    "avg_l2_err_cw_norm": cw_metrics["avg_list_norm"],
                    "avg_denoise_time_sec": timing["avg_time_sec"],
                    "avg_denoise_time_sec_per_point": timing["avg_time_sec_per_point"],
                    "latency_p50_ms": timing["latency_p50_ms"],
                    "latency_p95_ms": timing["latency_p95_ms"],
                    "latency_max_ms": timing["latency_max_ms"],
                    "throughput_points_per_sec": timing["throughput_points_per_sec"],
                    "peak_rss_mb": timing["peak_rss_mb"],
                    "peak_vram_mb": timing["peak_vram_mb"],
                    "calibration_time_sec": calibration_time_sec,
                    "calibration_peak_rss_mb": calibration_peak_rss_mb,
                    "calibration_peak_vram_mb": calibration_peak_vram_mb,
                }
                if method_name == "valhalla_meili":
                    diagnostics_fn = getattr(model, "diagnostics_snapshot", None)
                    if callable(diagnostics_fn):
                        try:
                            diagnostics = diagnostics_fn()
                            if isinstance(diagnostics, dict):
                                result.update(diagnostics)
                        except Exception as exc:
                            self.logger.warning(
                                "Failed to collect Valhalla diagnostics for result output: %s",
                                exc,
                            )

                self.trajectory_evaluator._save_results(result)
                results.append(result)
            finally:
                if model is not None:
                    try:
                        model.deconst()
                    except Exception:
                        pass

        if self.progress_bar:
            done_msg = "Baseline complete"
            if dataset_name:
                done_msg = f"{done_msg} ({dataset_name})"
            sys.stdout.write("\r\033[K" + done_msg + "\n")
            sys.stdout.flush()
        return results
