#!/usr/bin/env python3
"""
evaluation_manager.py

PURPOSE:
    Orchestrate trajectory-wise denoising evaluation across multiple models.
    Supports BF (Breadth-First) and DF (Depth-First) denoising methods.
    Outputs multi-granularity error metrics (point/byte/chunk-wise).

USAGE:
    manager = TestManager(output_dir="./bin/test_results")
    manager.run_trajectory_evaluation()
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import psutil
import torch

from encoder_decoder import EncoderDecoder

from utils.evaluations.base import EvaluationManager, InTrainingEvaluator, RegionalEvaluator
from utils.evaluations.chunk import ChunkEvaluator
from utils.evaluations.progress import ProgressTracker
from utils.evaluations.trajectory import ClassicBaselineEvaluator, TrajectoryEvaluator
from utils.evaluations.uncertainty import UncertaintyBandTrajectoryTest
from utils.evaluations.validation import quick_acc_test, time_test


# ================================================================
# TRAJECTORY TEST MANAGER
# ================================================================
class TestManager(EvaluationManager):
    def __init__(self, output_dir: str = "test_results"):
        super().__init__(output_dir)
        self.trajectory_evaluator = TrajectoryEvaluator(output_dir)
        self.chunk_evaluator = ChunkEvaluator(output_dir)
        self.intraining_evaluator = InTrainingEvaluator()
        self.regional_evaluator = RegionalEvaluator()
        self.uncertainty_band_tester = UncertaintyBandTrajectoryTest(output_dir)
        self.classic_baseline_evaluator = ClassicBaselineEvaluator(self.trajectory_evaluator)

    def _new_uncertainty_tester(self) -> UncertaintyBandTrajectoryTest:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(self.output_dir) / f"test_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return UncertaintyBandTrajectoryTest(str(run_dir))

    # ============================================================
    # PUBLIC: Run trajectory evaluation
    # ============================================================
    def run_trajectory_evaluation(
        self,
        model_names: Optional[List[str]] = None,
        denoise_methods: List[str] = None,
        model_root: str = "./bin/model/RectifiedTraj",
        model_tag: str = "RectifiedTraj",
        test_data_path: str = "./dataset/processed/NUMOSIM_Kanto/test/traj_test",
        M: int = 20,
        D: Optional[float] = None,
        N: Optional[int] = None,
        manual_config: Optional[Dict] = None,
        run_baselines: bool = True,
    ) -> List[Dict]:
        if N is None:
            if D is None:
                D = 7.0
            N = int(D * 8640)

        self.logger.info(
            f"Dataset configuration: M={M}, D={D if D else N/8640:.2f} days, N={N} points"
        )
        if denoise_methods is None:
            denoise_methods = ["BF", "DF"]

        if model_names is None:
            model_names = self._discover_models(model_root)

        self.logger.info(f"Found {len(model_names)} models to evaluate")

        test_trajectories, dataset_name = self._load_or_generate_test_data(test_data_path, M, N)
        self.trajectory_evaluator.set_run_context(dataset_name)
        if run_baselines:
            self.trajectory_evaluator.evaluate_baseline(test_trajectories, dataset_name=dataset_name)
            self.classic_baseline_evaluator.evaluate_classic_baselines(
                test_trajectories, dataset_name=dataset_name
            )

        progress_tracker = ProgressTracker(
            total_models=len(model_names),
            total_q1=1,
            total_q2=1,
            total_step=1,
            total_method=len(denoise_methods),
        )
        self.trajectory_evaluator.progress_tracker = progress_tracker
        progress_tracker.update(phase="trajectory", dataset=dataset_name)

        all_results = []
        for model_idx, model_name in enumerate(model_names):
            model_dir = Path(model_root) / model_name
            display_name = self._normalize_model_name(model_name)

            checkpoint_name = self._find_best_checkpoint(model_dir)
            if checkpoint_name is None:
                self.logger.warning(f"No checkpoint found for {model_name}, skipping")
                continue

            config = self._load_model_config(model_dir)
            K = config.get("K", 256)
            Q1 = config.get("Q1", 1)
            Q2 = config.get("Q2", 12)
            if manual_config is not None:
                Q1 = manual_config.get("Q1", Q1)
                Q2 = manual_config.get("Q2", Q2)

            for method_idx, method in enumerate(denoise_methods):
                progress_tracker.update(
                    model=display_name,
                    model_idx=model_idx,
                    q1=Q1,
                    q2=Q2,
                    q1_idx=0,
                    q2_idx=0,
                    step_idx=0,
                    method_idx=method_idx,
                    method=method,
                    t_delta=(manual_config or {}).get("t_delta", config.get("t_delta", 1.0)),
                )
                self.logger.info(f"Testing {model_name} with {method}")

                if manual_config is None:
                    result = self.trajectory_evaluator.evaluate_model(
                        model_name=display_name,
                        model_dir=str(model_dir.absolute()),
                        checkpoint_name=checkpoint_name,
                        denoise_method=method,
                        test_trajectories=test_trajectories,
                        K=K,
                        Q1=Q1,
                        Q2=Q2,
                        model_tag=model_tag,
                        dataset_name=dataset_name,
                    )
                else:
                    result = self.trajectory_evaluator.evaluate_model_with_config(
                        model_name=display_name,
                        model_dir=str(model_dir.absolute()),
                        checkpoint_name=checkpoint_name,
                        denoise_method=method,
                        test_trajectories=test_trajectories,
                        manual_config=manual_config,
                        model_tag=model_tag,
                        dataset_name=dataset_name,
                    )

                all_results.append(result)
                progress_tracker.update(job_finished=True)

        self.logger.info(f"Completed {len(all_results)} evaluations")
        return all_results

    # ============================================================
    # PART 2: EvaluationManager grid search
    # ============================================================
    def run_grid_search_evaluation(
        self,
        job_list: Dict,
        model_names: Optional[List[str]] = None,
        model_root: str = "./bin/model/RectifiedTraj",
        model_tag: str = "RectifiedTraj",
        test_data_path: str = "./dataset/processed/NUMOSIM_Kanto/test/traj_test",
        M: int = 20,
        D: Optional[float] = None,
        N: Optional[int] = None,
        run_baselines: bool = True,
    ) -> List[Dict]:
        if N is None:
            if D is None:
                D = 7.0
            N = int(D * 8640)

        self.logger.info(
            f"Dataset configuration: M={M}, D={D if D else N/8640:.2f} days, N={N} points"
        )
        if job_list is None:
            raise ValueError("job_list is required for grid search (use eval_joblist.json).")
        denoise_methods = job_list.get("methods") or ["BF", "DF"]

        if model_names is None:
            model_names = self._discover_models(model_root)

        self.logger.info(f"Found {len(model_names)} models to evaluate")
        self.logger.info("Grid search space:")
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

        test_trajectories, dataset_name = self._load_or_generate_test_data(test_data_path, M, N)
        self.trajectory_evaluator.set_run_context(dataset_name)
        if run_baselines:
            self.trajectory_evaluator.evaluate_baseline(test_trajectories, dataset_name=dataset_name)
            self.classic_baseline_evaluator.evaluate_classic_baselines(
                test_trajectories, dataset_name=dataset_name
            )

        progress_tracker = ProgressTracker(
            total_models=len(model_names),
            total_q1=len(job_list['Q1']),
            total_q2=len(job_list['Q2']),
            total_step=len(job_list['t_delta']),
            total_method=len(denoise_methods)
        )

        self.trajectory_evaluator.progress_tracker = progress_tracker
        progress_tracker.update(phase="trajectory", dataset=dataset_name)

        all_results = []
        combination_idx = 0
        skipped_invalid = 0
        skipped_errors = 0

        for model_idx, model_name in enumerate(model_names):
            model_dir = Path(model_root) / model_name
            display_name = self._normalize_model_name(model_name)

            checkpoint_name = self._find_best_checkpoint(model_dir)
            if checkpoint_name is None:
                self.logger.warning(f"No checkpoint found for {model_name}, skipping")
                continue

            for q1_idx, Q1 in enumerate(job_list['Q1']):
                for q2_idx, Q2 in enumerate(job_list['Q2']):
                    for step_idx, t_delta in enumerate(job_list['t_delta']):
                        for method_idx, method in enumerate(denoise_methods):
                            combination_idx += 1

                            progress_tracker.update(
                                model=display_name,
                                model_idx=model_idx,
                                q1=Q1,
                                q2=Q2,
                                q1_idx=q1_idx,
                                q2_idx=q2_idx,
                                step_idx=step_idx,
                                method_idx=method_idx,
                                method=method,
                                t_delta=t_delta,
                            )

                            manual_config = {
                                "Q1": Q1,
                                "Q2": Q2,
                                "t_delta": t_delta,
                            }

                            try:
                                result = self.trajectory_evaluator.evaluate_model_with_config(
                                    model_name=display_name,
                                    model_dir=str(model_dir.absolute()),
                                    checkpoint_name=checkpoint_name,
                                denoise_method=method,
                                test_trajectories=test_trajectories,
                                manual_config=manual_config,
                                model_tag=model_tag,
                                dataset_name=dataset_name,
                            )

                                all_results.append(result)
                                progress_tracker.update(job_finished=True)

                            except AssertionError as e:
                                self.logger.warning(
                                    f"SKIPPED (Invalid): {model_name} Q1={Q1} Q2={Q2} t_delta={t_delta} | {str(e)}"
                                )
                                skipped_invalid += 1
                                progress_tracker.update(job_finished=True)
                                continue

                            except Exception as e:
                                self.logger.warning(
                                    f"SKIPPED (Error): {model_name} Q1={Q1} Q2={Q2} t_delta={t_delta} | {type(e).__name__}: {str(e)}"
                                )
                                skipped_errors += 1
                                progress_tracker.update(job_finished=True)
                                continue

        if getattr(self, "brief_summary", False):
            self.logger.info(
                "Grid search summary: total=%d success=%d skipped_invalid=%d skipped_errors=%d",
                combination_idx,
                len(all_results),
                skipped_invalid,
                skipped_errors,
            )
        else:
            print("\n" * 3)
            print(f"{'='*60}")
            print("Grid Search Summary")
            print(f"{'='*60}")
            print(f"Total combinations tested: {combination_idx}")
            print(f"Successful evaluations: {len(all_results)}")
            print(f"Skipped (invalid hyperparameters): {skipped_invalid}")
            print(f"Skipped (errors): {skipped_errors}")
            print(f"{'='*60}")

        return all_results

    # ============================================================
    # PUBLIC: Run chunk-wise evaluation
    # ============================================================
    def run_chunk_evaluation(
        self,
        model_names: Optional[List[str]] = None,
        model_root: str = "./bin/model/RectifiedTraj",
        model_tag: str = "RectifiedTraj",
        test_dir: str = "./dataset/processed/NUMOSIM_Kanto/test/chunk_test",
        max_chunks: int = 5000,
        manual_config: Optional[Dict] = None,
        run_baselines: bool = True,
        baseline_methods: Optional[List[str]] = None,
    ) -> List[Dict]:
        from baseline import (
            build_lat_lon_timestamp_sequence_from_lonlat,
            create_baseline_model,
            latlon_to_lonlat,
        )
        from datetime import datetime
        import time
        from pymap3d import geodetic2enu

        self.logger.info("Starting chunk-wise evaluation...")
        self.logger.info(f"  test_dir: {test_dir}")
        self.logger.info(f"  max_chunks: {max_chunks}")

        if model_names is None:
            model_names = self._discover_models(model_root)
        self.logger.info(f"  models: {len(model_names)}")

        limit = int(max_chunks) if (max_chunks is not None and int(max_chunks) > 0) else None
        X0, X1, timestamps, file_count, chunk_coord_space = self._load_chunk_pairs_via_dataloader(
            test_dir,
            max_chunks=limit,
        )
        num_chunks = int(X0.shape[0])
        self.logger.info(
            "Loaded %d chunks via StandaloneDataLoader (files=%d, coord_space=%s)",
            num_chunks,
            file_count,
            chunk_coord_space,
        )

        test_path = Path(test_dir)
        dataset_root = self._infer_dataset_name_from_test_path(test_path)
        if test_path.is_file() and test_path.suffix == ".pt":
            if dataset_root:
                dataset_name = f"{dataset_root}_{test_path.stem}"
            else:
                dataset_name = test_path.stem
        else:
            dataset_name = dataset_root or "chunk_test"
        results = []
        bytewise_rows = []

        baseline_method_table = [
            "kalman_rts",
            "hampel",
            "savgol",
            "spline",
            "raw",
            "valhalla_meili",
        ]
        if baseline_methods is None:
            selected_baselines = baseline_method_table
        else:
            allowed = set(baseline_methods)
            selected_baselines = [name for name in baseline_method_table if name in allowed]
            unknown = [name for name in baseline_methods if name not in set(baseline_method_table)]
            for name in unknown:
                self.logger.warning("Unknown chunk baseline ignored: %s", name)

        Q1_bytes = (manual_config or {}).get("Q1", 1)
        Q2_bytes = (manual_config or {}).get("Q2", 12)
        Q1p = Q1_bytes * 8
        Q2p = Q2_bytes * 8
        bar_width = 30

        runtime_device = str(
            os.getenv(
                "RECTIFIEDTRAJ_RUNTIME_DEVICE_EFFECTIVE",
                os.getenv("RECTIFIEDTRAJ_DEVICE", "unknown"),
            )
        ).strip().lower()
        if runtime_device.startswith("cuda"):
            runtime_device = "cuda"
        elif runtime_device == "cpu":
            runtime_device = "cpu"
        use_cuda_timing = (runtime_device == "cuda") and torch.cuda.is_available()
        proc = psutil.Process(os.getpid())
        chunk_coord_space_norm = str(chunk_coord_space or "UNKNOWN").strip().upper()

        def _latency_stats(times_sec: list[float]) -> dict:
            if not times_sec:
                return {
                    "avg": None,
                    "p50_ms": None,
                    "p95_ms": None,
                    "max_ms": None,
                }
            arr = np.asarray(times_sec, dtype=float)
            return {
                "avg": float(np.mean(arr)),
                "p50_ms": float(np.percentile(arr, 50) * 1000.0),
                "p95_ms": float(np.percentile(arr, 95) * 1000.0),
                "max_ms": float(np.max(arr) * 1000.0),
            }

        def _chunk_bar(i: int, total: int, name: str, q1: int, q2: int, method: str, t_delta: float | None):
            progress = i / total if total > 0 else 0.0
            filled = int(bar_width * progress)
            bar = "#" * filled + "-" * (bar_width - filled)
            t_str = datetime.now().strftime("%H:%M:%S")
            t_disp = f"{t_delta:.2f}" if isinstance(t_delta, (int, float)) else "NA"
            sys.stdout.write(
                f"\r[{bar}] {i}/{total} | {name} | Q1={q1} Q2={q2} {method} tΔ={t_disp} | {t_str}"
            )
            sys.stdout.flush()

        def _lonlat_to_enu(lonlat: np.ndarray, ref_lon: float, ref_lat: float) -> np.ndarray:
            arr = np.asarray(lonlat, dtype=np.float64)
            lat = arr[:, 1]
            lon = arr[:, 0]
            e, n, _ = geodetic2enu(
                lat,
                lon,
                0.0,
                float(ref_lat),
                float(ref_lon),
                0.0,
            )
            return np.stack([e, n], axis=1).astype(np.float32, copy=False)

        def _prepare_chunk_eval_views(
            noisy_xy: np.ndarray,
            clean_xy: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float | None, float | None]:
            noisy_arr = np.asarray(noisy_xy, dtype=np.float32)
            clean_arr = np.asarray(clean_xy, dtype=np.float32)
            if chunk_coord_space_norm == "GPS":
                ref_lon = float(noisy_arr[0, 0])
                ref_lat = float(noisy_arr[0, 1])
                noisy_enu = _lonlat_to_enu(noisy_arr, ref_lon=ref_lon, ref_lat=ref_lat)
                clean_enu = _lonlat_to_enu(clean_arr, ref_lon=ref_lon, ref_lat=ref_lat)
                return noisy_enu, clean_enu, noisy_arr, ref_lon, ref_lat
            return noisy_arr, clean_arr, None, None, None

        if run_baselines:
            if not selected_baselines:
                self.logger.warning("No chunk classic baselines selected; skipping classic chunk baselines.")
            for method_name in selected_baselines:
                if method_name == "valhalla_meili" and chunk_coord_space_norm != "GPS":
                    self.logger.warning(
                        "Skipping valhalla_meili chunk baseline: coord_space=%s (GPS required).",
                        chunk_coord_space_norm,
                    )
                    continue
                model = None
                report_method_name = method_name
                if method_name == "kalman_rts":
                    kalman_mode = str(os.getenv("KALMAN_RTS_CALIBRATION_MODE", "dataset")).strip() or "dataset"
                    report_method_name = f"kalman_rts@{kalman_mode}"
                calibration_time_sec = None
                calibration_peak_rss_mb = None
                calibration_peak_vram_mb = None
                self.logger.info(f"[Baseline] {report_method_name}")
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
                    if method_name == "kalman_rts":
                        # Keep reporting keyed by requested fairness mode, not backend artifact source.
                        mode_label = str(kalman_mode).strip() or "dataset"
                        report_method_name = f"kalman_rts@{mode_label}"

                    errs_full = []
                    errs_mid = []
                    byte_sum = np.zeros(32, dtype=np.float64)
                    byte_cnt = np.zeros(32, dtype=np.int64)
                    times = []
                    peak_rss_mb = None
                    peak_vram_mb = None

                    for i in range(num_chunks):
                        _chunk_bar(i + 1, num_chunks, report_method_name, Q1_bytes, Q2_bytes, "N/A", None)
                        inp_raw = X1[i].numpy()
                        gt_raw = X0[i].numpy()
                        inp_metric, gt_metric, noisy_lonlat, ref_lon, ref_lat = _prepare_chunk_eval_views(
                            inp_raw,
                            gt_raw,
                        )
                        ts_abs = timestamps[i].numpy() if timestamps is not None else None
                        ts_rel = None
                        if ts_abs is not None:
                            ts_abs = ts_abs.astype(np.float64, copy=False)
                            ts_rel = ts_abs.copy()
                            if ts_rel.size and np.isfinite(ts_rel[0]):
                                ts_rel = ts_rel - float(ts_rel[0])

                        # Timing scope is prediction only; calibration already happened in create_baseline_model().
                        rss_before = float(proc.memory_info().rss) / (1024.0 * 1024.0)
                        if use_cuda_timing:
                            torch.cuda.synchronize()
                            torch.cuda.reset_peak_memory_stats()
                        t0 = time.perf_counter()
                        if method_name == "valhalla_meili":
                            if noisy_lonlat is None or ref_lon is None or ref_lat is None:
                                raise RuntimeError(
                                    "valhalla_meili chunk baseline requires GPS chunk inputs."
                                )
                            seq = build_lat_lon_timestamp_sequence_from_lonlat(
                                noisy_lonlat,
                                timestamps=ts_abs,
                            )
                            denoised_latlon = model.predict(seq)
                            denoised_lonlat = latlon_to_lonlat(denoised_latlon)
                            pred = _lonlat_to_enu(
                                denoised_lonlat,
                                ref_lon=ref_lon,
                                ref_lat=ref_lat,
                            )
                        else:
                            pred = model.predict_enu(inp_metric, timestamps=ts_rel)
                        if use_cuda_timing:
                            torch.cuda.synchronize()
                        t1 = time.perf_counter()
                        times.append(t1 - t0)
                        rss_after = float(proc.memory_info().rss) / (1024.0 * 1024.0)
                        run_peak_rss = max(rss_before, rss_after)
                        peak_rss_mb = run_peak_rss if peak_rss_mb is None else max(peak_rss_mb, run_peak_rss)
                        if use_cuda_timing:
                            run_peak_vram = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
                            peak_vram_mb = run_peak_vram if peak_vram_mb is None else max(peak_vram_mb, run_peak_vram)

                        diff_full = pred - gt_metric
                        l2_full = np.sqrt((diff_full * diff_full).sum(axis=-1))
                        errs_full.append(l2_full)

                        for b in range(32):
                            s = b * 8
                            e = s + 8
                            seg = l2_full[s:e]
                            if seg.size > 0:
                                byte_sum[b] += float(seg.sum())
                                byte_cnt[b] += int(seg.size)

                        if Q2p > 0:
                            pred_mid = pred[Q1p:-Q2p]
                            gt_mid = gt_metric[Q1p:-Q2p]
                        else:
                            pred_mid = pred[Q1p:]
                            gt_mid = gt_metric[Q1p:]
                        diff_mid = pred_mid - gt_mid
                        l2_mid = np.sqrt((diff_mid * diff_mid).sum(axis=-1))
                        errs_mid.append(l2_mid)

                    errs_full = np.stack(errs_full, axis=0)
                    errs_mid = np.stack(errs_mid, axis=0)
                    timing = _latency_stats(times)
                    avg_time = timing["avg"]
                    avg_time_per_point = (avg_time / X0.shape[1]) if avg_time is not None and X0.shape[1] else None
                    throughput = (
                        (float(X0.shape[1]) / float(avg_time))
                        if avg_time is not None and float(avg_time) > 0 and int(X0.shape[1]) > 0
                        else None
                    )
                    row = {
                        "model_name": report_method_name,
                        "model_tag": "Baseline",
                        "device": "cpu",
                        "dataset_name": dataset_name,
                        "denoise_method": "N/A",
                        "K": None,
                        "Q1": None,
                        "Q2": None,
                        "t_delta": None,
                        "N_steps": None,
                        "avg_time_s": float(avg_time),
                        "avg_time_per_point_s": float(avg_time_per_point) if avg_time_per_point is not None else None,
                        "err_mean_full": float(errs_full.mean()),
                        "err_median_full": float(np.median(errs_full)),
                        "err_p95_full": float(np.percentile(errs_full, 95)),
                        "err_std_full": float(errs_full.std()),
                        "err_mean_mid": float(errs_mid.mean()),
                        "err_median_mid": float(np.median(errs_mid)),
                        "err_p95_mid": float(np.percentile(errs_mid, 95)),
                        "err_std_mid": float(errs_mid.std()),
                        "latency_p50_ms": timing["p50_ms"],
                        "latency_p95_ms": timing["p95_ms"],
                        "latency_max_ms": timing["max_ms"],
                        "throughput_points_per_sec": throughput,
                        "peak_rss_mb": peak_rss_mb,
                        "peak_vram_mb": peak_vram_mb,
                        "calibration_time_sec": calibration_time_sec,
                        "calibration_peak_rss_mb": calibration_peak_rss_mb,
                        "calibration_peak_vram_mb": calibration_peak_vram_mb,
                        "num_tested_chunks": num_chunks,
                        "test_timestamp": datetime.now().isoformat(),
                    }
                    self.chunk_evaluator._append_row(row)
                    results.append(row)
                    byte_mean = np.divide(byte_sum, np.maximum(byte_cnt, 1))
                    bytewise_rows.append({
                        "model_name": report_method_name,
                        "model_tag": "Baseline",
                        "dataset_name": dataset_name,
                        "byte_mean": byte_mean,
                    })
                except Exception as exc:
                    self.logger.warning("Chunk baseline failed for %s: %s", report_method_name, exc)
                finally:
                    if model is not None:
                        try:
                            model.deconst()
                        except Exception:
                            pass
                    sys.stdout.write("\r\033[K")
                    sys.stdout.flush()

        for model_name in model_names:
            model_dir = Path(model_root) / model_name
            display_name = self._normalize_model_name(model_name)
            checkpoint_name = self._find_best_checkpoint(model_dir)
            if checkpoint_name is None:
                self.logger.warning(f"No checkpoint found for {model_name}, skipping")
                continue

            ckpt_path = self.trajectory_evaluator._get_checkpoint_path(str(model_dir), checkpoint_name)
            if ckpt_path is None:
                self.logger.warning(f"Checkpoint not found for {model_name}, skipping")
                continue

            self.logger.info(f"[Model] {display_name} | ckpt={checkpoint_name}")
            decoder = EncoderDecoder(ckpt_path, manual_config=manual_config)
            Q1p = decoder.Q1
            Q2p = decoder.Q2
            errs_full = []
            errs_mid = []
            byte_sum = np.zeros(32, dtype=np.float64)
            byte_cnt = np.zeros(32, dtype=np.int64)
            times = []
            peak_rss_mb = None
            peak_vram_mb = None

            for i in range(num_chunks):
                _chunk_bar(i + 1, num_chunks, display_name, decoder.Q1_bytes, decoder.Q2_bytes, "N/A", decoder.t_delta)
                inp_raw = X1[i].numpy()
                gt_raw = X0[i].numpy()
                inp_metric, gt_metric, noisy_lonlat, ref_lon, ref_lat = _prepare_chunk_eval_views(
                    inp_raw,
                    gt_raw,
                )
                rss_before = float(proc.memory_info().rss) / (1024.0 * 1024.0)
                if use_cuda_timing:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                if chunk_coord_space_norm == "GPS":
                    if noisy_lonlat is None or ref_lon is None or ref_lat is None:
                        raise RuntimeError("GPS chunk path requires noisy_lonlat/ref coords.")
                    pred_gps = decoder.denoise_chunk(np.asarray(noisy_lonlat, dtype=np.float64))
                    pred = _lonlat_to_enu(
                        pred_gps,
                        ref_lon=ref_lon,
                        ref_lat=ref_lat,
                    )
                else:
                    pred = decoder.denoise_chunk_enu(inp_metric)
                if use_cuda_timing:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append(t1 - t0)
                rss_after = float(proc.memory_info().rss) / (1024.0 * 1024.0)
                run_peak_rss = max(rss_before, rss_after)
                peak_rss_mb = run_peak_rss if peak_rss_mb is None else max(peak_rss_mb, run_peak_rss)
                if use_cuda_timing:
                    run_peak_vram = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
                    peak_vram_mb = run_peak_vram if peak_vram_mb is None else max(peak_vram_mb, run_peak_vram)

                diff_full = pred - gt_metric
                l2_full = np.sqrt((diff_full * diff_full).sum(axis=-1))
                errs_full.append(l2_full)

                for b in range(32):
                    s = b * 8
                    e = s + 8
                    seg = l2_full[s:e]
                    if seg.size > 0:
                        byte_sum[b] += float(seg.sum())
                        byte_cnt[b] += int(seg.size)

                if Q2p > 0:
                    pred_mid = pred[Q1p:-Q2p]
                    gt_mid = gt_metric[Q1p:-Q2p]
                else:
                    pred_mid = pred[Q1p:]
                    gt_mid = gt_metric[Q1p:]
                diff_mid = pred_mid - gt_mid
                l2_mid = np.sqrt((diff_mid * diff_mid).sum(axis=-1))
                errs_mid.append(l2_mid)

            errs_full = np.stack(errs_full, axis=0)
            errs_mid = np.stack(errs_mid, axis=0)
            timing = _latency_stats(times)
            avg_time = timing["avg"]
            avg_time_per_point = (avg_time / X0.shape[1]) if avg_time is not None and X0.shape[1] else None
            throughput = (
                (float(X0.shape[1]) / float(avg_time))
                if avg_time is not None and float(avg_time) > 0 and int(X0.shape[1]) > 0
                else None
            )
            row = {
                "model_name": display_name,
                "model_tag": model_tag,
                "device": runtime_device or "unknown",
                "dataset_name": dataset_name,
                "denoise_method": "N/A",
                "K": decoder.K,
                "Q1": decoder.Q1_bytes,
                "Q2": decoder.Q2_bytes,
                "t_delta": decoder.t_delta,
                "N_steps": int(1.0 / decoder.t_delta) if decoder.t_delta > 0 else None,
                "avg_time_s": avg_time,
                "avg_time_per_point_s": avg_time_per_point,
                "err_mean_full": float(errs_full.mean()),
                "err_median_full": float(np.median(errs_full)),
                "err_p95_full": float(np.percentile(errs_full, 95)),
                "err_std_full": float(errs_full.std()),
                "err_mean_mid": float(errs_mid.mean()),
                "err_median_mid": float(np.median(errs_mid)),
                "err_p95_mid": float(np.percentile(errs_mid, 95)),
                "err_std_mid": float(errs_mid.std()),
                "latency_p50_ms": timing["p50_ms"],
                "latency_p95_ms": timing["p95_ms"],
                "latency_max_ms": timing["max_ms"],
                "throughput_points_per_sec": throughput,
                "peak_rss_mb": peak_rss_mb,
                "peak_vram_mb": peak_vram_mb,
                "calibration_time_sec": 0.0,
                "calibration_peak_rss_mb": None,
                "calibration_peak_vram_mb": None,
                "num_tested_chunks": num_chunks,
                "test_timestamp": datetime.now().isoformat(),
            }
            self.chunk_evaluator._append_row(row)
            results.append(row)
            byte_mean = np.divide(byte_sum, np.maximum(byte_cnt, 1))
            bytewise_rows.append({
                "model_name": display_name,
                "model_tag": model_tag,
                "dataset_name": dataset_name,
                "byte_mean": byte_mean,
            })
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()

        self.chunk_evaluator.save_bytewise_heatmap(bytewise_rows, dataset_name=dataset_name)
        return results

    # ============================================================
    # UNCERTAINTY BAND TEST
    # ============================================================
    def run_uncertainty_band_test(
        self,
        model_names: Optional[List[str]] = None,
        denoise_methods: List[str] = None,
        model_root: str = "./bin/model/RectifiedTraj",
        model_tag: str = "RectifiedTraj",
        test_data_path: str = "./dataset/processed/NUMOSIM_Kanto/test/traj_test/full_traj_range",
        M: int = 200,
        N: int = 10000,
        run_baselines: bool = True,
        baseline_methods: Optional[List[str]] = None,
    ) -> List[Dict]:
        if denoise_methods is None:
            denoise_methods = ["BF", "DF"]

        if model_names is None:
            model_names = self._discover_models(model_root)

        self.logger.info(f"Found {len(model_names)} models to evaluate (uncertainty band)")

        test_trajectories = self._load_or_generate_uncertainty_test_data(
            test_data_path, M, N
        )

        tester = self._new_uncertainty_tester()
        try:
            tester.log_uncertainty_dataset_info(
                test_trajectories=test_trajectories,
                dataset_name=Path(test_data_path).name,
            )
        except Exception as exc:
            self.logger.warning("Failed to log uncertainty dataset info: %s", exc)
        all_results = []
        if run_baselines:
            self.logger.info("Running classic baselines (uncertainty band)")
            baseline_results = tester.evaluate_classic_baselines(
                test_trajectories=test_trajectories,
                methods=baseline_methods,
            )
            all_results.extend(baseline_results)

        for model_name in model_names:
            model_dir = Path(model_root) / model_name

            checkpoint_name = self._find_best_checkpoint(model_dir)
            if checkpoint_name is None:
                self.logger.warning(f"No checkpoint found for {model_name}, skipping")
                continue

            config = self._load_model_config(model_dir)
            K = config.get("K", 256)
            Q1 = config.get("Q1", 1)
            Q2 = config.get("Q2", 12)

            for method in denoise_methods:
                self.logger.info(f"Testing {model_name} with {method} (uncertainty band)")

                result = tester.evaluate_model(
                    model_name=model_name,
                    model_dir=str(model_dir.absolute()),
                    checkpoint_name=checkpoint_name,
                    denoise_method=method,
                    test_trajectories=test_trajectories,
                    K=K,
                    Q1=Q1,
                    Q2=Q2,
                    model_tag=model_tag,
                )

                all_results.append(result)

        self.logger.info(f"Completed {len(all_results)} uncertainty-band evaluations")
        return all_results

    def bounded_trajectory_test(
        self,
        model_name: str,
        denoise_methods: List[str] = None,
        model_root: str = "./bin/model/RectifiedTraj",
        model_tag: str = "RectifiedTraj",
        test_data_path: str = "./dataset/processed/NUMOSIM_Kanto/test/traj_test/full_traj_range",
        M: int = 200,
        N: int = 10000,
        checkpoint_name: Optional[str] = None,
        run_baselines: bool = False,
    ) -> List[Dict]:
        if denoise_methods is None:
            denoise_methods = ["BF", "DF"]

        model_dir = Path(model_root) / model_name
        if checkpoint_name is None:
            checkpoint_name = self._find_best_checkpoint(model_dir)
        if checkpoint_name is None:
            raise FileNotFoundError(f"No checkpoint found for {model_name}")

        config = self._load_model_config(model_dir)
        K = config.get("K", 256)
        Q1 = config.get("Q1", 1)
        Q2 = config.get("Q2", 12)

        test_trajectories = self._load_or_generate_uncertainty_test_data(
            test_data_path, M, N
        )

        tester = self._new_uncertainty_tester()
        try:
            tester.log_uncertainty_dataset_info(
                test_trajectories=test_trajectories,
                dataset_name=Path(test_data_path).name,
            )
        except Exception as exc:
            self.logger.warning("Failed to log uncertainty dataset info: %s", exc)
        results = []
        if run_baselines:
            baseline_results = tester.evaluate_classic_baselines(
                test_trajectories=test_trajectories
            )
            results.extend(baseline_results)

        for method in denoise_methods:
            result = tester.evaluate_model(
                model_name=model_name,
                model_dir=str(model_dir.absolute()),
                checkpoint_name=checkpoint_name,
                denoise_method=method,
                test_trajectories=test_trajectories,
                K=K,
                Q1=Q1,
                Q2=Q2,
                model_tag=model_tag,
            )
            results.append(result)

        return results

    def bounded_trajectory_test_all_models(
        self,
        model_names: Optional[List[str]] = None,
        denoise_methods: List[str] = None,
        model_root: str = "./bin/model/RectifiedTraj",
        model_tag: str = "RectifiedTraj",
        test_data_path: str = "./dataset/processed/NUMOSIM_Kanto/test/traj_test/full_traj_range",
        M: int = 200,
        N: int = 10000,
    ) -> List[Dict]:
        return self.run_uncertainty_band_test(
            model_names=model_names,
            denoise_methods=denoise_methods,
            model_root=model_root,
            model_tag=model_tag,
            test_data_path=test_data_path,
            M=M,
            N=N,
        )
