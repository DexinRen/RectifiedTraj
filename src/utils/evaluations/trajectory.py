import csv
import json
import logging
import os
import sys
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import psutil
import torch

import encoder_decoder
from learned_decoder import build_learned_decoder
from theta_model import build_theta_model
from utils.evaluations.result_io import write_rows_to_csv

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


def _clean_manual_config(manual_config: Optional[Dict]) -> Optional[Dict]:
    cfg = dict(manual_config or {})
    cfg.pop("denoise_method", None)
    return cfg or None


_PROCESS = psutil.Process(os.getpid())


def _current_rss_mb() -> float:
    return float(_PROCESS.memory_info().rss) / (1024.0 * 1024.0)


class _RssMonitor:
    """Sample process RSS during one evaluation task.

    `resource.ru_maxrss` is process-lifetime state, so it cannot distinguish
    sequential model/baseline rows. This monitor records the current RSS peak
    while a single task is active.
    """

    def __init__(self, interval_sec: float = 0.02):
        self.interval_sec = float(interval_sec)
        self.start_mb = 0.0
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "_RssMonitor":
        self.start_mb = _current_rss_mb()
        self.peak_mb = self.start_mb
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.sample()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval_sec * 2.0, 0.05))
        self.sample()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_sec):
            self.sample()

    def sample(self) -> None:
        rss_mb = _current_rss_mb()
        if rss_mb > self.peak_mb:
            self.peak_mb = rss_mb

    def telemetry(self) -> dict[str, float]:
        self.sample()
        return {
            "peak_rss_mb": float(self.peak_mb),
            "rss_delta_mb": max(float(self.peak_mb - self.start_mb), 0.0),
        }


def _sync_runtime_device() -> None:
    if str(getattr(encoder_decoder, "DEVICE", "")).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


class TrajectoryEvaluator:
    """
    Evaluate trajectory denoising quality at multiple granularities.

    Outputs:
        - CSV: point-wise summary for quick comparison
        - CSV: trajectory_pointwise_summary.csv (one row per model+config with per-position avg_l2_err_pw_profile)
        Heatmap PNGs are generated later by utils/data_visualizer/make_heatmaps.py
        from the aggregated summary CSVs.

    Granularities:
        - Point-wise (pw): individual point L2 errors
        - Byte-wise (bw): 8-point groups
        - Chunk-wise (cw): per-chunk errors
    """

    def __init__(self, output_dir: str = "./bin/test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_dir = self.output_dir / "raw"
        self.run_dir: Optional[Path] = None
        self._active_dataset_name: Optional[str] = None
        self._run_dir_by_dataset: dict[str, Path] = {}
        self._last_prediction_telemetry: dict[str, float] = {}

        self.csv_path = self.output_dir / "trajectory_evaluation_summary.csv"
        self.logger = logging.getLogger("TrajectoryEvaluator")

        header_cols = [
            "model_name",
            "model_tag",
            "device",
            "dataset_name",
            "K",
            "Q1",
            "Q2",
            "denoise_steps",
            "sample_steps",
            "t_delta",
            "avg_l1_err_pw",
            "med_l1_err_pw",
            "p95_l1_err_pw",
            "std_l1_err_pw",
            "avg_l2_err_pw",
            "med_l2_err_pw",
            "p95_l2_err_pw",
            "std_l2_err_pw",
            "avg_l1_err_tail",
            "avg_l2_err_tail",
            "num_tested_trajectories",
            "num_tested_points",
            "prediction_time_sec",
            "points_per_sec",
            "peak_rss_mb",
            "rss_delta_mb",
            "test_timestamp",
            "model_full_name",
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
    def _compute_error_norms(
        enu_pred: np.ndarray,
        enu_clean: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        delta = np.asarray(enu_pred, dtype=float) - np.asarray(enu_clean, dtype=float)
        with np.errstate(all="ignore"):
            l2 = np.linalg.norm(delta, axis=1)
            l1 = np.abs(delta).sum(axis=1)
        return l2, l1

    @staticmethod
    def _effective_traj_length(traj_obj) -> int:
        noisy = np.asarray(getattr(traj_obj, "noisy_gps", []), dtype=float)
        return int(len(encoder_decoder.remove_nan_rows(noisy)))

    def evaluate_model(
        self,
        model_name: str,
        model_dir: str,
        checkpoint_name: str,
        test_trajectories: List,
        K: int = 256,
        Q1: int = 2,
        Q2: int = 2,
        model_tag: str = "RectifiedTraj",
        dataset_name: Optional[str] = None,
    ) -> Dict:
        self.logger.info(f"Evaluating {model_name} with fixed chunk_stitch denoising")

        checkpoint_path = self._get_checkpoint_path(model_dir, checkpoint_name)
        if checkpoint_path is None:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_name}")

        denoised_trajectories, l2_errors, last_point_l2_errors, l1_errors, last_point_l1_errors, decoder = self._denoise_trajectories(
            checkpoint_path, test_trajectories
        )

        actual_Q1 = int(getattr(decoder, "Q1_bytes", Q1))
        actual_Q2 = int(getattr(decoder, "Q2_bytes", Q2))

        self.logger.info("Computing point-wise metrics...")
        pw_l2_metrics = self._compute_pointwise_metrics(l2_errors)
        pw_l1_metrics = self._compute_pointwise_metrics(l1_errors)
        pw_profile = self._compute_trajectory_pointwise_profile(test_trajectories, l2_errors)
        per_traj_l2_errors = self._split_error_array_by_trajectory(test_trajectories, l2_errors)
        per_traj_l1_errors = self._split_error_array_by_trajectory(test_trajectories, l1_errors)
        tail_l2_by_traj, tail_l1_by_traj = self._compute_chunk_tail_error_lists_from_errors(
            per_traj_l2_errors,
            per_traj_l1_errors,
            decoder,
        )
        chunk_tail_l2_errors = np.concatenate([x for x in tail_l2_by_traj if x.size > 0], axis=0) if any(x.size > 0 for x in tail_l2_by_traj) else np.asarray([], dtype=float)
        chunk_tail_l1_errors = np.concatenate([x for x in tail_l1_by_traj if x.size > 0], axis=0) if any(x.size > 0 for x in tail_l1_by_traj) else np.asarray([], dtype=float)
        tail_l2_metrics = self._compute_pointwise_metrics(chunk_tail_l2_errors)
        tail_l1_metrics = self._compute_pointwise_metrics(chunk_tail_l1_errors)

        self.logger.info("Computing byte-wise metrics...")
        bw_metrics = self._compute_bytewise_metrics(test_trajectories, l2_errors)

        self.logger.info("Computing chunk-wise metrics...")
        cw_metrics = self._compute_chunkwise_metrics(
            test_trajectories,
            l2_errors,
            decoder.K,
            actual_Q1,
            actual_Q2,
        )

        results = {
            "model_name": model_name,
            "model_full_name": Path(model_dir).name,
            "model_tag": model_tag,
            "device": _normalize_device_label(getattr(encoder_decoder, "DEVICE", None)),
            "dataset_name": dataset_name,
            "model_dir": model_dir,
            "checkpoint_name": checkpoint_name,
            "K": decoder.K,
            "Q1": actual_Q1,
            "Q2": actual_Q2,
            "denoise_steps": None,
            "sample_steps": getattr(decoder, "sample_steps", None),
            "t_delta": decoder.t_delta,
            "test_timestamp": datetime.now().isoformat(),
            "num_tested_trajectories": len(test_trajectories),
            "num_tested_points": sum(self._effective_traj_length(t) for t in test_trajectories),
            **self._last_prediction_telemetry,
            "longest_trajectory_length": max(self._effective_traj_length(t) for t in test_trajectories),
            "avg_l1_err_pw": pw_l1_metrics["avg"],
            "med_l1_err_pw": pw_l1_metrics["med"],
            "p95_l1_err_pw": pw_l1_metrics["p95"],
            "std_l1_err_pw": pw_l1_metrics["std"],
            "avg_l2_err_pw": pw_l2_metrics["avg"],
            "med_l2_err_pw": pw_l2_metrics["med"],
            "p95_l2_err_pw": pw_l2_metrics["p95"],
            "std_l2_err_pw": pw_l2_metrics["std"],
            "avg_l1_err_tail": tail_l1_metrics["avg"],
            "avg_l2_err_tail": tail_l2_metrics["avg"],
            "avg_l2_err_pw_profile": pw_profile["avg_list"],
            "avg_l2_err_pw_profile_norm": pw_profile["avg_list_norm"],
            "avg_l2_err_bw": bw_metrics["avg_list"],
            "avg_l2_err_bw_norm": bw_metrics["avg_list_norm"],
            "avg_l2_err_cw": cw_metrics["avg_list"],
            "avg_l2_err_cw_norm": cw_metrics["avg_list_norm"],
        }
        results["traj_p_val_rows"] = self._build_traj_p_val_rows_from_lists(
            test_trajectories,
            per_traj_l2_errors,
            per_traj_l1_errors,
            results,
            tail_l2_by_traj=tail_l2_by_traj,
            tail_l1_by_traj=tail_l1_by_traj,
        )

        self._save_results(results)

        self.logger.info("Evaluation complete: %s", model_name)
        return results

    def evaluate_model_with_config(
        self,
        model_name: str,
        model_dir: str,
        checkpoint_name: str,
        test_trajectories: List,
        manual_config: Dict,
        model_tag: str = "RectifiedTraj",
        dataset_name: Optional[str] = None,
    ) -> Dict:
        manual_config = _clean_manual_config(manual_config)

        checkpoint_path = self._get_checkpoint_path(model_dir, checkpoint_name)
        if checkpoint_path is None:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_name}")

        denoised_trajectories, l2_errors, last_point_l2_errors, l1_errors, last_point_l1_errors, decoder = self._denoise_trajectories_with_config(
            checkpoint_path, test_trajectories, manual_config
        )

        actual_Q1 = decoder.Q1_bytes
        actual_Q2 = decoder.Q2_bytes

        pw_l2_metrics = self._compute_pointwise_metrics(l2_errors)
        pw_l1_metrics = self._compute_pointwise_metrics(l1_errors)
        pw_profile = self._compute_trajectory_pointwise_profile(test_trajectories, l2_errors)
        per_traj_l2_errors = self._split_error_array_by_trajectory(test_trajectories, l2_errors)
        per_traj_l1_errors = self._split_error_array_by_trajectory(test_trajectories, l1_errors)
        tail_l2_by_traj, tail_l1_by_traj = self._compute_chunk_tail_error_lists_from_errors(
            per_traj_l2_errors,
            per_traj_l1_errors,
            decoder,
        )
        chunk_tail_l2_errors = np.concatenate([x for x in tail_l2_by_traj if x.size > 0], axis=0) if any(x.size > 0 for x in tail_l2_by_traj) else np.asarray([], dtype=float)
        chunk_tail_l1_errors = np.concatenate([x for x in tail_l1_by_traj if x.size > 0], axis=0) if any(x.size > 0 for x in tail_l1_by_traj) else np.asarray([], dtype=float)
        tail_l2_metrics = self._compute_pointwise_metrics(chunk_tail_l2_errors)
        tail_l1_metrics = self._compute_pointwise_metrics(chunk_tail_l1_errors)
        bw_metrics = self._compute_bytewise_metrics(test_trajectories, l2_errors)
        cw_metrics = self._compute_chunkwise_metrics(
            test_trajectories,
            l2_errors,
            decoder.K,
            actual_Q1,
            actual_Q2,
        )

        results = {
            "model_name": model_name,
            "model_full_name": Path(model_dir).name,
            "model_tag": model_tag,
            "device": _normalize_device_label(getattr(encoder_decoder, "DEVICE", None)),
            "dataset_name": dataset_name,
            "model_dir": model_dir,
            "checkpoint_name": checkpoint_name,
            "K": decoder.K,
            "Q1": actual_Q1,
            "Q2": actual_Q2,
            "denoise_steps": (manual_config or {}).get("denoise_steps"),
            "sample_steps": getattr(decoder, "sample_steps", (manual_config or {}).get("sample_steps")),
            "t_delta": decoder.t_delta,
            "test_timestamp": datetime.now().isoformat(),
            "num_tested_trajectories": len(test_trajectories),
            "num_tested_points": sum(self._effective_traj_length(t) for t in test_trajectories),
            **self._last_prediction_telemetry,
            "longest_trajectory_length": max(self._effective_traj_length(t) for t in test_trajectories),
            "avg_l1_err_pw": pw_l1_metrics["avg"],
            "med_l1_err_pw": pw_l1_metrics["med"],
            "p95_l1_err_pw": pw_l1_metrics["p95"],
            "std_l1_err_pw": pw_l1_metrics["std"],
            "avg_l2_err_pw": pw_l2_metrics["avg"],
            "med_l2_err_pw": pw_l2_metrics["med"],
            "p95_l2_err_pw": pw_l2_metrics["p95"],
            "std_l2_err_pw": pw_l2_metrics["std"],
            "avg_l1_err_tail": tail_l1_metrics["avg"],
            "avg_l2_err_tail": tail_l2_metrics["avg"],
            "avg_l2_err_pw_profile": pw_profile["avg_list"],
            "avg_l2_err_pw_profile_norm": pw_profile["avg_list_norm"],
            "avg_l2_err_bw": bw_metrics["avg_list"],
            "avg_l2_err_bw_norm": bw_metrics["avg_list_norm"],
            "avg_l2_err_cw": cw_metrics["avg_list"],
            "avg_l2_err_cw_norm": cw_metrics["avg_list_norm"],
        }
        results["traj_p_val_rows"] = self._build_traj_p_val_rows_from_lists(
            test_trajectories,
            per_traj_l2_errors,
            per_traj_l1_errors,
            results,
            tail_l2_by_traj=tail_l2_by_traj,
            tail_l1_by_traj=tail_l1_by_traj,
        )

        self._save_results(results)
        return results

    def _denoise_trajectories_with_config(
        self,
        checkpoint_path: str,
        test_trajectories: List,
        manual_config: Dict,
    ) -> tuple:
        return self._denoise_trajectories_core(
            checkpoint_path=checkpoint_path,
            test_trajectories=test_trajectories,
            manual_config=manual_config,
        )

    def _denoise_trajectories(
        self,
        checkpoint_path: str,
        test_trajectories: List,
    ) -> tuple:
        return self._denoise_trajectories_core(
            checkpoint_path=checkpoint_path,
            test_trajectories=test_trajectories,
            manual_config=None,
        )

    def _denoise_trajectories_core(
        self,
        checkpoint_path: str,
        test_trajectories: List,
        manual_config: Optional[Dict],
    ) -> tuple:
        manual_config = _clean_manual_config(manual_config)
        self._patch_encoder_decoder_checkpoint_loading()

        decoder = build_learned_decoder(checkpoint_path, manual_config=manual_config)
        self.logger.info(
            "Initialized %s with %s",
            decoder.__class__.__name__,
            checkpoint_path,
        )

        denoised_trajectories = []
        all_l2_errors = []
        all_l1_errors = []
        last_point_l2_errors = []
        last_point_l1_errors = []
        predicted_points = 0
        prediction_inputs = [
            (idx, traj_obj.noisy_gps, traj_obj.clean_gps)
            for idx, traj_obj in enumerate(test_trajectories)
        ]
        prediction_pairs = []
        self.logger.info(f"Starting denoising of {len(test_trajectories)} trajectories...")
        _sync_runtime_device()
        with _RssMonitor() as rss_monitor:
            predict_start = time.perf_counter()

            for idx, noisy_gps, clean_gps in prediction_inputs:
                if hasattr(self, "progress_tracker") and self.progress_tracker is not None:
                    self.progress_tracker.update(traj=idx + 1, total_traj=len(test_trajectories))

                denoised_gps = decoder.denoise_traj_DF(noisy_gps)
                rss_monitor.sample()
                predicted_points += int(len(denoised_gps))
                prediction_pairs.append((denoised_gps, clean_gps))

            _sync_runtime_device()
            prediction_time_sec = max(time.perf_counter() - predict_start, 0.0)
            rss_telemetry = rss_monitor.telemetry()

        for denoised_gps, clean_gps in prediction_pairs:
            T_denoised = len(denoised_gps)
            clean_gps_aligned = clean_gps[-T_denoised:]

            ref_lat = float(clean_gps_aligned[0, 1])
            ref_lon = float(clean_gps_aligned[0, 0])
            enu_denoised = self._gps_to_enu_batch(denoised_gps, ref_lat, ref_lon)
            enu_clean = self._gps_to_enu_batch(clean_gps_aligned, ref_lat, ref_lon)

            l2_errors, l1_errors = self._compute_error_norms(enu_denoised, enu_clean)

            denoised_trajectories.append(denoised_gps)
            all_l2_errors.append(l2_errors)
            all_l1_errors.append(l1_errors)
            if l2_errors.size > 0:
                last_point_l2_errors.append(float(l2_errors[-1]))
                last_point_l1_errors.append(float(l1_errors[-1]))

        if len(all_l2_errors) == 0:
            raise RuntimeError("No trajectories successfully denoised")

        self._last_prediction_telemetry = {
            "prediction_time_sec": prediction_time_sec,
            "points_per_sec": (
                float(predicted_points) / prediction_time_sec
                if prediction_time_sec > 0.0
                else 0.0
            ),
            **rss_telemetry,
        }

        all_l2_errors_array = np.concatenate(all_l2_errors, axis=0)
        all_l1_errors_array = np.concatenate(all_l1_errors, axis=0)
        last_point_l2_errors_array = np.asarray(last_point_l2_errors, dtype=float)
        last_point_l1_errors_array = np.asarray(last_point_l1_errors, dtype=float)

        self.logger.info(f"Denoised {len(denoised_trajectories)} trajectories")

        return (
            denoised_trajectories,
            all_l2_errors_array,
            last_point_l2_errors_array,
            all_l1_errors_array,
            last_point_l1_errors_array,
            decoder,
        )

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

    def _compute_chunk_tail_error_lists(
        self,
        trajectories: List,
        decoder,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Tail error = literal chunk-end prediction error.

        If the chunk end falls in right padding, use the last non-padded point of the
        chunk instead. This measures the real tail of each predicted chunk, not the
        stitched payload boundary.
        """
        tails_l2_by_traj: list[np.ndarray] = []
        tails_l1_by_traj: list[np.ndarray] = []

        def _chunk_errors(pred_gps: np.ndarray, clean_gps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            ref_lat = float(clean_gps[0, 1])
            ref_lon = float(clean_gps[0, 0])
            enu_pred = self._gps_to_enu_batch(pred_gps, ref_lat, ref_lon)
            enu_clean = self._gps_to_enu_batch(clean_gps, ref_lat, ref_lon)
            return self._compute_error_norms(enu_pred, enu_clean)

        for traj_obj in trajectories:
            noisy_gps = encoder_decoder.remove_nan_rows(np.asarray(traj_obj.noisy_gps, dtype=float))
            n_points = int(len(noisy_gps))
            if n_points <= 0:
                continue
            clean_gps = np.asarray(traj_obj.clean_gps, dtype=float)[-n_points:]

            stride = int(decoder.stride)
            q1 = int(decoder.Q1)
            noisy_padded, noisy_pad_mask, n_chunks, _ = decoder.build_padded_trajectory(noisy_gps)
            clean_padded, _, _, _ = decoder.build_padded_trajectory(clean_gps)

            traj_tail_l2: list[float] = []
            traj_tail_l1: list[float] = []
            for j in range(n_chunks):
                start = j * stride
                end = start + decoder.K
                noisy_chunk = noisy_padded[start:end]
                clean_chunk = clean_padded[start:end]
                chunk_pad_mask = noisy_pad_mask[start:end]
                pred_chunk = decoder.denoise_chunk(noisy_chunk, pad_mask=chunk_pad_mask)
                tail_local_idx = min(decoder.K - 1, q1 + n_points - 1 - start)
                chunk_l2_err, chunk_l1_err = _chunk_errors(pred_chunk, clean_chunk)
                traj_tail_l2.append(float(chunk_l2_err[tail_local_idx]))
                traj_tail_l1.append(float(chunk_l1_err[tail_local_idx]))
            tails_l2_by_traj.append(np.asarray(traj_tail_l2, dtype=float))
            tails_l1_by_traj.append(np.asarray(traj_tail_l1, dtype=float))

        return tails_l2_by_traj, tails_l1_by_traj

    def _compute_chunk_tail_errors(
        self,
        trajectories: List,
        decoder,
    ) -> tuple[np.ndarray, np.ndarray]:
        tails_l2_by_traj, tails_l1_by_traj = self._compute_chunk_tail_error_lists(
            trajectories,
            decoder,
        )
        tails_l2 = [arr for arr in tails_l2_by_traj if arr.size > 0]
        tails_l1 = [arr for arr in tails_l1_by_traj if arr.size > 0]
        out_l2 = np.concatenate(tails_l2, axis=0) if tails_l2 else np.asarray([], dtype=float)
        out_l1 = np.concatenate(tails_l1, axis=0) if tails_l1 else np.asarray([], dtype=float)
        return out_l2, out_l1

    def _compute_trajectory_pointwise_profile(
        self,
        trajectories: List,
        errors: np.ndarray,
    ) -> Dict:
        effective_lengths = [self._effective_traj_length(t) for t in trajectories]
        max_length = max(effective_lengths, default=0)

        pw_sum = np.zeros(max_length, dtype=float)
        pw_count = np.zeros(max_length, dtype=int)

        error_idx = 0
        for traj_obj, traj_length in zip(trajectories, effective_lengths):
            traj_errors = errors[error_idx: error_idx + traj_length]
            if traj_errors.shape[0] != traj_length:
                raise ValueError(
                    "Pointwise error array length does not match the effective trajectory lengths: "
                    f"needed {traj_length} values, got {traj_errors.shape[0]} at offset {error_idx}"
                )

            pw_sum[:traj_length] += traj_errors
            pw_count[:traj_length] += 1

            error_idx += traj_length

        if error_idx != errors.shape[0]:
            raise ValueError(
                "Unused pointwise errors remain after profile aggregation: "
                f"consumed {error_idx}, total {errors.shape[0]}"
            )

        pw_list = np.divide(
            pw_sum,
            pw_count,
            where=pw_count > 0,
            out=np.full_like(pw_sum, np.nan),
        )

        valid = pw_list[np.isfinite(pw_list)]
        mean_val = float(np.nanmean(valid)) if valid.size > 0 else 0.0
        if not np.isfinite(mean_val) or mean_val <= 0.0:
            avg_l2_err_pw_norm = np.zeros_like(pw_list)
        else:
            avg_l2_err_pw_norm = pw_list / mean_val

        return {
            "avg_list": [float(v) for v in pw_list],
            "avg_list_norm": [float(v) for v in avg_l2_err_pw_norm],
        }

    def _compute_bytewise_metrics(
        self,
        trajectories: List,
        errors: np.ndarray
    ) -> Dict:
        pw_list = np.asarray(
            self._compute_trajectory_pointwise_profile(trajectories, errors)["avg_list"],
            dtype=float,
        )
        max_length = int(pw_list.shape[0])

        num_bytes = int(np.ceil(max_length / 8))
        avg_l2_err_bw = []

        for byte_idx in range(num_bytes):
            start = byte_idx * 8
            end = min(start + 8, max_length)

            byte_errors = pw_list[start:end]
            valid = byte_errors[np.isfinite(byte_errors)]
            byte_avg = float(np.mean(valid)) if valid.size > 0 else 0.0
            avg_l2_err_bw.append(byte_avg)

        bw_valid = [x for x in avg_l2_err_bw if np.isfinite(x) and x > 0]
        bw_mean = float(np.mean(bw_valid)) if bw_valid else 0.0
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
        pw_list = np.asarray(
            self._compute_trajectory_pointwise_profile(trajectories, errors)["avg_list"],
            dtype=float,
        )
        max_length = int(pw_list.shape[0])

        Q1_points = encoder_decoder.q_config_to_points(Q1)
        Q2_points = encoder_decoder.q_config_to_points(Q2)
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
            valid_errors = chunk_errors[np.isfinite(chunk_errors)]

            if len(valid_errors) > 0:
                chunk_avg = float(np.mean(valid_errors))
            else:
                chunk_avg = 0.0

            avg_l2_err_cw.append(chunk_avg)

        cw_valid = [x for x in avg_l2_err_cw if np.isfinite(x) and x > 0]
        cw_mean = float(np.mean(cw_valid)) if cw_valid else 0.0
        avg_l2_err_cw_norm = [
            x / cw_mean if cw_mean > 0 else 0.0
            for x in avg_l2_err_cw
        ]

        return {
            "avg_list": avg_l2_err_cw,
            "avg_list_norm": avg_l2_err_cw_norm,
        }

    def _save_results(self, results: Dict):
        results = dict(results)
        results.setdefault("device", _runtime_device_label())
        results.setdefault("model_full_name", self._resolve_model_full_name(results))

        def _fmt(value, fmt: str):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return "NA"
            return format(value, fmt)

        dataset_name = results.get("dataset_name") or "NA"
        csv_row = (
            f"{results['model_name']},{results.get('model_tag', 'NA')},{results.get('device', 'unknown')},{dataset_name},"
            f"{_fmt(results.get('K'), 'd')},{_fmt(results.get('Q1'), 'd')},{_fmt(results.get('Q2'), 'd')},"
            f"{_fmt(results.get('denoise_steps'), 'd')},{_fmt(results.get('sample_steps'), 'd')},"
            f"{_fmt(results.get('t_delta'), '.8f')},"
            f"{_fmt(results.get('avg_l1_err_pw'), '.6f')},{_fmt(results.get('med_l1_err_pw'), '.6f')},"
            f"{_fmt(results.get('p95_l1_err_pw'), '.6f')},"
            f"{_fmt(results.get('std_l1_err_pw'), '.6f')},"
            f"{_fmt(results.get('avg_l2_err_pw'), '.6f')},{_fmt(results.get('med_l2_err_pw'), '.6f')},"
            f"{_fmt(results.get('p95_l2_err_pw'), '.6f')},"
            f"{_fmt(results.get('std_l2_err_pw'), '.6f')},"
            f"{_fmt(results.get('avg_l1_err_tail'), '.6f')},"
            f"{_fmt(results.get('avg_l2_err_tail'), '.6f')},"
            f"{_fmt(results.get('num_tested_trajectories'), 'd')},{_fmt(results.get('num_tested_points'), 'd')},"
            f"{_fmt(results.get('prediction_time_sec'), '.6f')},{_fmt(results.get('points_per_sec'), '.6f')},"
            f"{_fmt(results.get('peak_rss_mb'), '.3f')},{_fmt(results.get('rss_delta_mb'), '.3f')},"
            f"{results.get('test_timestamp')},{results.get('model_full_name')}\n"
        )
        with open(self.csv_path, "a") as f:
            f.write(csv_row)

        self._append_trajectory_pointwise_row(results)
        self._append_traj_p_val_rows(results)

    def _append_traj_p_val_rows(self, results: Dict) -> None:
        rows = results.get("traj_p_val_rows")
        if not isinstance(rows, list) or not rows:
            return

        out_dir = self.run_dir if self.run_dir is not None else self.output_dir
        out_csv = out_dir / "traj_p_val.csv"
        existing_rows: list[dict[str, str]] = []
        if out_csv.exists():
            with out_csv.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)

        merged_rows = existing_rows + [{str(k): v for k, v in row.items()} for row in rows]
        field_order = [
            "sample_index",
            "dataset_name",
            "model_name",
            "model_tag",
            "device",
            "K",
            "Q1",
            "Q2",
            "denoise_steps",
            "sample_steps",
            "t_delta",
            "n_points",
            "first_timestamp",
            "last_timestamp",
            "mean_l2_err",
            "median_l2_err",
            "p95_l2_err",
            "std_l2_err",
            "mean_l1_err",
            "median_l1_err",
            "p95_l1_err",
            "std_l1_err",
            "last_point_l2_err",
            "last_point_l1_err",
            "tail_mean_l2_err",
            "tail_mean_l1_err",
            "num_tail_chunks",
            "test_timestamp",
            "model_full_name",
        ]
        write_rows_to_csv(merged_rows, out_csv, field_order=field_order)

    def _split_error_array_by_trajectory(
        self,
        trajectories: List,
        errors: np.ndarray,
    ) -> list[np.ndarray]:
        per_traj: list[np.ndarray] = []
        error_idx = 0
        for traj_obj in trajectories:
            traj_length = self._effective_traj_length(traj_obj)
            traj_errors = np.asarray(errors[error_idx:error_idx + traj_length], dtype=float)
            if traj_errors.shape[0] != traj_length:
                raise ValueError(
                    "Per-trajectory split failed: "
                    f"needed {traj_length} values, got {traj_errors.shape[0]} at offset {error_idx}"
                )
            per_traj.append(traj_errors)
            error_idx += traj_length
        if error_idx != int(errors.shape[0]):
            raise ValueError(
                "Unused errors remain after per-trajectory split: "
                f"consumed {error_idx}, total {errors.shape[0]}"
            )
        return per_traj

    def _compute_chunk_tail_error_lists_from_errors(
        self,
        per_traj_l2_errors: list[np.ndarray],
        per_traj_l1_errors: list[np.ndarray],
        decoder,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        stride = max(1, int(getattr(decoder, "stride", 1)))
        tails_l2_by_traj: list[np.ndarray] = []
        tails_l1_by_traj: list[np.ndarray] = []
        for l2_arr, l1_arr in zip(per_traj_l2_errors, per_traj_l1_errors):
            l2_vals = np.asarray(l2_arr, dtype=float).reshape(-1)
            l1_vals = np.asarray(l1_arr, dtype=float).reshape(-1)
            n_points = int(min(l2_vals.size, l1_vals.size))
            if n_points <= 0:
                tails_l2_by_traj.append(np.asarray([], dtype=float))
                tails_l1_by_traj.append(np.asarray([], dtype=float))
                continue
            tail_indices = list(range(stride - 1, n_points, stride))
            if not tail_indices or tail_indices[-1] != n_points - 1:
                tail_indices.append(n_points - 1)
            idx = np.asarray(tail_indices, dtype=int)
            tails_l2_by_traj.append(l2_vals[idx])
            tails_l1_by_traj.append(l1_vals[idx])
        return tails_l2_by_traj, tails_l1_by_traj

    def _build_traj_p_val_rows_from_lists(
        self,
        trajectories: List,
        per_traj_l2_errors: list[np.ndarray],
        per_traj_l1_errors: list[np.ndarray],
        results: Dict,
        *,
        tail_l2_by_traj: list[np.ndarray] | None = None,
        tail_l1_by_traj: list[np.ndarray] | None = None,
    ) -> list[dict]:
        rows: list[dict] = []
        device = results.get("device", _runtime_device_label())
        dataset_name = results.get("dataset_name", "NA")
        model_name = results.get("model_name", "NA")
        model_tag = results.get("model_tag", "NA")
        model_full_name = results.get("model_full_name") or self._resolve_model_full_name(results)
        q1 = results.get("Q1")
        q2 = results.get("Q2")
        denoise_steps = results.get("denoise_steps")
        sample_steps = results.get("sample_steps")
        t_delta = results.get("t_delta")
        k_value = results.get("K")
        test_timestamp = results.get("test_timestamp")

        tail_l2_seq = tail_l2_by_traj or [np.asarray([], dtype=float) for _ in per_traj_l2_errors]
        tail_l1_seq = tail_l1_by_traj or [np.asarray([], dtype=float) for _ in per_traj_l1_errors]

        for idx, (traj_obj, l2_arr, l1_arr, tail_l2_arr, tail_l1_arr) in enumerate(
            zip(trajectories, per_traj_l2_errors, per_traj_l1_errors, tail_l2_seq, tail_l1_seq)
        ):
            l2_vals = np.asarray(l2_arr, dtype=float)
            l1_vals = np.asarray(l1_arr, dtype=float)
            if l2_vals.size == 0 or l1_vals.size == 0:
                continue

            ts = getattr(traj_obj, "timestamps", None)
            first_ts = ""
            last_ts = ""
            if ts is not None:
                ts_arr = np.asarray(ts, dtype=float).reshape(-1)
                n_points = int(l2_vals.size)
                if ts_arr.size >= n_points:
                    ts_arr = ts_arr[-n_points:]
                if ts_arr.size > 0 and np.isfinite(ts_arr[0]):
                    first_ts = float(ts_arr[0])
                if ts_arr.size > 0 and np.isfinite(ts_arr[-1]):
                    last_ts = float(ts_arr[-1])

            tail_l2_vals = np.asarray(tail_l2_arr, dtype=float).reshape(-1)
            tail_l1_vals = np.asarray(tail_l1_arr, dtype=float).reshape(-1)
            tail_l2_valid = tail_l2_vals[np.isfinite(tail_l2_vals)]
            tail_l1_valid = tail_l1_vals[np.isfinite(tail_l1_vals)]

            rows.append(
                {
                    "sample_index": idx,
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "model_tag": model_tag,
                    "model_full_name": model_full_name,
                    "device": device,
                    "K": k_value,
                    "Q1": q1,
                    "Q2": q2,
                    "denoise_steps": denoise_steps,
                    "sample_steps": sample_steps,
                    "t_delta": t_delta,
                    "n_points": int(l2_vals.size),
                    "first_timestamp": first_ts,
                    "last_timestamp": last_ts,
                    "mean_l2_err": float(np.mean(l2_vals)),
                    "median_l2_err": float(np.median(l2_vals)),
                    "p95_l2_err": float(np.percentile(l2_vals, 95)),
                    "std_l2_err": float(np.std(l2_vals)),
                    "mean_l1_err": float(np.mean(l1_vals)),
                    "median_l1_err": float(np.median(l1_vals)),
                    "p95_l1_err": float(np.percentile(l1_vals, 95)),
                    "std_l1_err": float(np.std(l1_vals)),
                    "last_point_l2_err": float(l2_vals[-1]),
                    "last_point_l1_err": float(l1_vals[-1]),
                    "tail_mean_l2_err": float(np.mean(tail_l2_valid)) if tail_l2_valid.size > 0 else "",
                    "tail_mean_l1_err": float(np.mean(tail_l1_valid)) if tail_l1_valid.size > 0 else "",
                    "num_tail_chunks": int(tail_l2_valid.size),
                    "test_timestamp": test_timestamp,
                }
            )
        return rows

    def _resolve_trajectory_model_label(self, results: Dict) -> str:
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

    def _resolve_model_full_name(self, results: Dict) -> str:
        raw = str(results.get("model_full_name", "") or "").strip()
        if raw:
            return raw
        model_dir = str(results.get("model_dir", "") or "").strip()
        if model_dir:
            base = Path(model_dir).name.strip()
            if base:
                return base
        return str(results.get("model_name", "NA") or "NA")

    def _append_trajectory_pointwise_row(self, results: Dict) -> None:
        values = results.get("avg_l2_err_pw_profile")
        if not isinstance(values, (list, tuple, np.ndarray)) or len(values) == 0:
            return
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size == 0:
            return

        out_dir = self.run_dir if self.run_dir is not None else self.output_dir
        out_csv = out_dir / "trajectory_pointwise_summary.csv"

        model_label = self._resolve_trajectory_model_label(results)
        model_full_name = self._resolve_model_full_name(results)
        dataset_name = str(results.get("dataset_name", "") or "NA").strip()

        q1_raw = results.get("Q1")
        q2_raw = results.get("Q2")
        denoise_steps_raw = results.get("denoise_steps")
        sample_steps_raw = results.get("sample_steps")

        q1_str = "NA" if q1_raw is None else str(int(round(float(q1_raw))))
        q2_str = "NA" if q2_raw is None else str(int(round(float(q2_raw))))
        denoise_steps_str = (
            "NA"
            if denoise_steps_raw is None
            else str(int(round(float(denoise_steps_raw))))
        )
        sample_steps_str = (
            "NA"
            if sample_steps_raw is None
            else str(int(round(float(sample_steps_raw))))
        )

        key = (
            model_label,
            dataset_name,
            q1_str,
            q2_str,
            denoise_steps_str,
            sample_steps_str,
        )

        merged: dict[tuple[str, str, str, str, str, str], list[float]] = {}
        full_names: dict[tuple[str, str, str, str, str, str], str] = {}
        if out_csv.exists():
            with out_csv.open("r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    label = str(row.get("model_dir", "")).strip()
                    ds = str(row.get("dataset_name", "")).strip()
                    q1 = str(row.get("Q1", "") or "NA").strip()
                    q2 = str(row.get("Q2", "") or "NA").strip()
                    denoise_steps = str(row.get("denoise_steps", "") or "NA").strip()
                    sample_steps = str(row.get("sample_steps", "") or "NA").strip()
                    model_full = str(row.get("model_full_name", "")).strip()
                    if not label:
                        continue
                    vals: list[float] = []
                    i = 0
                    while True:
                        col = f"point_{i}"
                        if col not in row:
                            break
                        raw = str(row.get(col, "")).strip()
                        if raw in {"", "NA"}:
                            vals.append(float("nan"))
                        else:
                            vals.append(float(raw))
                        i += 1
                    old_key = (label, ds, q1, q2, denoise_steps, sample_steps)
                    merged[old_key] = vals
                    if model_full:
                        full_names[old_key] = model_full

        merged[key] = [float(v) for v in values]
        full_names[key] = model_full_name

        max_len = max((len(v) for v in merged.values()), default=0)
        header = [
            "model_dir",
            "dataset_name",
            "Q1",
            "Q2",
            "denoise_steps",
            "sample_steps",
        ] + [f"point_{i}" for i in range(max_len)] + ["model_full_name"]

        with out_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for (label, ds, q1, q2, denoise_steps, sample_steps), vals in sorted(merged.items(), key=lambda x: x[0]):
                padded = list(vals) + [float("nan")] * (max_len - len(vals))
                row = [label, ds, q1, q2, denoise_steps, sample_steps] + [
                    ("NA" if (isinstance(v, float) and np.isnan(v)) else f"{float(v):.10f}")
                    for v in padded
                ] + [full_names.get((label, ds, q1, q2, denoise_steps, sample_steps), "")]
                writer.writerow(row)

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
        from utils.evaluations.classic_baseline_runner import run_classic_baselines_filtered

        class _ManagerAdapter:
            def __init__(self, trajectory_evaluator: TrajectoryEvaluator):
                self.trajectory_evaluator = trajectory_evaluator

        results = run_classic_baselines_filtered(
            manager=_ManagerAdapter(self.trajectory_evaluator),
            test_trajectories=test_trajectories,
            dataset_name=str(dataset_name or "NA"),
            methods=list(methods or [
                "alpha_beta",
                "causal_hampel",
                "kalman_filter",
                "kalman_rts",
                "hampel",
                "savgol",
                "raw",
            ]),
            dataset_name_hint=dataset_name,
        )

        if self.progress_bar:
            done_msg = "Baseline complete"
            if dataset_name:
                done_msg = f"{done_msg} ({dataset_name})"
            sys.stdout.write("\r\033[K" + done_msg + "\n")
            sys.stdout.flush()
        return results
