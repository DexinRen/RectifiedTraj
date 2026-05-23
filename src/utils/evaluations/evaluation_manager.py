#!/usr/bin/env python3
"""
evaluation_manager.py

PURPOSE:
    Orchestrate trajectory-wise denoising evaluation across multiple models.
    Standard learned-model evaluation uses fixed chunk_stitch denoising.
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

from encoder_decoder import EncoderDecoder, q_config_to_points

from utils.evaluations.base import EvaluationManager, InTrainingEvaluator, RegionalEvaluator
from utils.evaluations.chunk_batch_runner import run_chunk_batch as _run_chunk_batch
from utils.evaluations.chunk import ChunkEvaluator
from utils.evaluations.progress import ProgressTracker
from utils.evaluations.p_value import generate_pairwise_p_value_report
from utils.evaluations.trajectory import ClassicBaselineEvaluator, TrajectoryEvaluator
from utils.evaluations.trajectory_batch_runner import run_trajectory_batch as _run_trajectory_batch
from utils.evaluations.uncertainty_batch_runner import run_uncertainty_batch as _run_uncertainty_batch
from utils.evaluations.uncertainty import UncertaintyBandTrajectoryTest


# ================================================================
# TRAJECTORY TEST MANAGER
# ================================================================
class TestManager(EvaluationManager):
    def __init__(self, output_dir: str = "./bin/test_results"):
        super().__init__(output_dir)
        self.trajectory_evaluator = TrajectoryEvaluator(output_dir)
        self.chunk_evaluator = ChunkEvaluator(output_dir)
        self.intraining_evaluator = InTrainingEvaluator()
        self.regional_evaluator = RegionalEvaluator()
        self.uncertainty_band_tester = UncertaintyBandTrajectoryTest(output_dir)
        self.classic_baseline_evaluator = ClassicBaselineEvaluator(self.trajectory_evaluator)

    def _new_uncertainty_tester(self) -> UncertaintyBandTrajectoryTest:
        return self.uncertainty_band_tester

    # ============================================================
    # PUBLIC: Run trajectory evaluation
    # ============================================================
    def run_trajectory_evaluation(
        self,
        model_names: Optional[List[str]] = None,
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
        if model_names is None:
            model_names = self._discover_models(model_root)

        self.logger.info(f"Found {len(model_names)} models to evaluate")

        test_trajectories, dataset_name = self._load_or_generate_test_data(test_data_path, M, N)
        self.trajectory_evaluator.set_run_context(dataset_name)
        if run_baselines:
            self.classic_baseline_evaluator.evaluate_classic_baselines(
                test_trajectories, dataset_name=dataset_name
            )

        progress_tracker = ProgressTracker(
            total_models=len(model_names),
            total_q1=1,
            total_q2=1,
            total_step=1,
            total_method=1,
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

            progress_tracker.update(
                model=display_name,
                model_idx=model_idx,
                q1=Q1,
                q2=Q2,
                q1_idx=0,
                q2_idx=0,
                step_idx=0,
                method_idx=0,
            )
            self.logger.info(f"Testing {model_name} with fixed chunk_stitch denoising")

            if manual_config is None:
                result = self.trajectory_evaluator.evaluate_model(
                    model_name=display_name,
                    model_dir=str(model_dir.absolute()),
                    checkpoint_name=checkpoint_name,
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
                    test_trajectories=test_trajectories,
                    manual_config=manual_config,
                    model_tag=model_tag,
                    dataset_name=dataset_name,
                )

            all_results.append(result)
            progress_tracker.update(job_finished=True)

        self.logger.info(f"Completed {len(all_results)} evaluations")
        return all_results

    def run_trajectory_batch(
        self,
        *,
        task_specs: List[Dict],
        max_workers: int,
    ) -> None:
        _run_trajectory_batch(
            manager=self,
            task_specs=task_specs,
            max_workers=max_workers,
        )

    def run_uncertainty_batch(
        self,
        *,
        task_specs: List[Dict],
        max_workers: int,
    ) -> None:
        _run_uncertainty_batch(
            manager=self,
            task_specs=task_specs,
            max_workers=max_workers,
        )

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
        from baseline import create_baseline_model
        from datetime import datetime
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
            "Loaded %d chunks via DataLoader (files=%d, coord_space=%s)",
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
        pointwise_rows = []
        chunk_p_val_rows = []
        baseline_method_table = [
            "alpha_beta",
            "causal_hampel",
            "kalman_filter",
            "kalman_rts",
            "hampel",
            "savgol",
            "raw",
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
        Q1p = q_config_to_points(Q1_bytes)
        Q2p = q_config_to_points(Q2_bytes)
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
        chunk_coord_space_norm = str(chunk_coord_space or "UNKNOWN").strip().upper()

        def _chunk_bar(i: int, total: int, name: str, q1: int, q2: int):
            progress = i / total if total > 0 else 0.0
            filled = int(bar_width * progress)
            bar = "#" * filled + "-" * (bar_width - filled)
            t_str = datetime.now().strftime("%H:%M:%S")
            sys.stdout.write(
                f"\r[{bar}] {i}/{total} | {name} | Q1={q1} Q2={q2} | {t_str}"
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

        def _build_chunk_p_val_rows(
            *,
            model_name: str,
            model_full_name: str,
            model_tag_value: str,
            device_value: str,
            k_value,
            q1_value,
            q2_value,
            errs_full_arr: np.ndarray,
            errs_l1_full_arr: np.ndarray,
            errs_mid_arr: np.ndarray,
            errs_l1_mid_arr: np.ndarray,
            timestamp_value: str,
        ) -> list[dict]:
            rows: list[dict] = []
            for idx in range(int(errs_full_arr.shape[0])):
                l2_full = np.asarray(errs_full_arr[idx], dtype=float)
                l1_full = np.asarray(errs_l1_full_arr[idx], dtype=float)
                l2_mid = np.asarray(errs_mid_arr[idx], dtype=float)
                l1_mid = np.asarray(errs_l1_mid_arr[idx], dtype=float)
                rows.append(
                    {
                        "sample_index": idx,
                        "dataset_name": dataset_name,
                        "model_name": model_name,
                        "model_tag": model_tag_value,
                        "model_full_name": model_full_name,
                        "device": device_value,
                        "K": k_value,
                        "Q1": q1_value,
                        "Q2": q2_value,
                        "n_points_full": int(l2_full.size),
                        "n_points_mid": int(l2_mid.size),
                        "mean_l2_err_full": float(np.mean(l2_full)),
                        "median_l2_err_full": float(np.median(l2_full)),
                        "p95_l2_err_full": float(np.percentile(l2_full, 95)),
                        "std_l2_err_full": float(np.std(l2_full)),
                        "mean_l1_err_full": float(np.mean(l1_full)),
                        "median_l1_err_full": float(np.median(l1_full)),
                        "p95_l1_err_full": float(np.percentile(l1_full, 95)),
                        "std_l1_err_full": float(np.std(l1_full)),
                        "mean_l2_err_mid": float(np.mean(l2_mid)),
                        "median_l2_err_mid": float(np.median(l2_mid)),
                        "p95_l2_err_mid": float(np.percentile(l2_mid, 95)),
                        "std_l2_err_mid": float(np.std(l2_mid)),
                        "mean_l1_err_mid": float(np.mean(l1_mid)),
                        "median_l1_err_mid": float(np.median(l1_mid)),
                        "p95_l1_err_mid": float(np.percentile(l1_mid, 95)),
                        "std_l1_err_mid": float(np.std(l1_mid)),
                        "test_timestamp": timestamp_value,
                    }
                )
            return rows

        if run_baselines:
            if not selected_baselines:
                self.logger.warning("No chunk classic baselines selected; skipping classic chunk baselines.")
            for method_name in selected_baselines:
                model = None
                report_method_name = method_name
                if method_name == "kalman_rts":
                    kalman_mode = str(os.getenv("KALMAN_RTS_CALIBRATION_MODE", "dataset")).strip() or "dataset"
                    report_method_name = f"kalman_rts@{kalman_mode}"
                self.logger.info(f"[Baseline] {report_method_name}")
                try:
                    model = create_baseline_model(
                        method_name=method_name,
                        dataset_name=dataset_name,
                    )
                    if method_name == "kalman_rts":
                        # Keep reporting keyed by requested fairness mode, not backend artifact source.
                        mode_label = str(kalman_mode).strip() or "dataset"
                        report_method_name = f"kalman_rts@{mode_label}"

                    errs_full = []
                    errs_l1_full = []
                    errs_mid = []
                    errs_l1_mid = []

                    for i in range(num_chunks):
                        _chunk_bar(i + 1, num_chunks, report_method_name, Q1_bytes, Q2_bytes)
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

                        pred = model.predict_enu(inp_metric, timestamps=ts_rel)

                        diff_full = pred - gt_metric
                        l2_full = np.sqrt((diff_full * diff_full).sum(axis=-1))
                        l1_full = np.abs(diff_full).sum(axis=-1)
                        errs_full.append(l2_full)
                        errs_l1_full.append(l1_full)

                        if Q2p > 0:
                            pred_mid = pred[Q1p:-Q2p]
                            gt_mid = gt_metric[Q1p:-Q2p]
                        else:
                            pred_mid = pred[Q1p:]
                            gt_mid = gt_metric[Q1p:]
                        diff_mid = pred_mid - gt_mid
                        l2_mid = np.sqrt((diff_mid * diff_mid).sum(axis=-1))
                        l1_mid = np.abs(diff_mid).sum(axis=-1)
                        errs_mid.append(l2_mid)
                        errs_l1_mid.append(l1_mid)

                    errs_full = np.stack(errs_full, axis=0)
                    errs_l1_full = np.stack(errs_l1_full, axis=0)
                    errs_mid = np.stack(errs_mid, axis=0)
                    errs_l1_mid = np.stack(errs_l1_mid, axis=0)
                    point_mean = np.mean(errs_full, axis=0, dtype=np.float64)
                    row = {
                        "model_name": report_method_name,
                        "model_tag": "Baseline",
                        "device": "cpu",
                        "dataset_name": dataset_name,
                        "K": None,
                        "Q1": None,
                        "Q2": None,
                        "err_l1_mean_full": float(errs_l1_full.mean()),
                        "err_l1_median_full": float(np.median(errs_l1_full)),
                        "err_l1_p95_full": float(np.percentile(errs_l1_full, 95)),
                        "err_l1_std_full": float(errs_l1_full.std()),
                        "err_mean_full": float(errs_full.mean()),
                        "err_median_full": float(np.median(errs_full)),
                        "err_p95_full": float(np.percentile(errs_full, 95)),
                        "err_std_full": float(errs_full.std()),
                        "err_l1_mean_mid": float(errs_l1_mid.mean()),
                        "err_l1_median_mid": float(np.median(errs_l1_mid)),
                        "err_l1_p95_mid": float(np.percentile(errs_l1_mid, 95)),
                        "err_l1_std_mid": float(errs_l1_mid.std()),
                        "err_mean_mid": float(errs_mid.mean()),
                        "err_median_mid": float(np.median(errs_mid)),
                        "err_p95_mid": float(np.percentile(errs_mid, 95)),
                        "err_std_mid": float(errs_mid.std()),
                        "num_tested_chunks": num_chunks,
                        "test_timestamp": datetime.now().isoformat(),
                        "model_full_name": report_method_name,
                    }
                    self.chunk_evaluator._append_row(row)
                    results.append(row)
                    pointwise_rows.append({
                        "model_name": report_method_name,
                        "model_full_name": report_method_name,
                        "model_tag": "Baseline",
                        "dataset_name": dataset_name,
                        "model_root": "",
                        "Q1": None,
                        "Q2": None,
                        "point_mean": point_mean,
                    })
                    chunk_p_val_rows.extend(
                        _build_chunk_p_val_rows(
                            model_name=report_method_name,
                            model_full_name=report_method_name,
                            model_tag_value="Baseline",
                            device_value="cpu",
                            k_value=None,
                            q1_value=None,
                            q2_value=None,
                            errs_full_arr=errs_full,
                            errs_l1_full_arr=errs_l1_full,
                            errs_mid_arr=errs_mid,
                            errs_l1_mid_arr=errs_l1_mid,
                            timestamp_value=row["test_timestamp"],
                        )
                    )
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
            errs_l1_full = []
            errs_mid = []
            errs_l1_mid = []

            for i in range(num_chunks):
                _chunk_bar(i + 1, num_chunks, display_name, decoder.Q1_bytes, decoder.Q2_bytes)
                inp_raw = X1[i].numpy()
                gt_raw = X0[i].numpy()
                inp_metric, gt_metric, noisy_lonlat, ref_lon, ref_lat = _prepare_chunk_eval_views(
                    inp_raw,
                    gt_raw,
                )
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

                diff_full = pred - gt_metric
                l2_full = np.sqrt((diff_full * diff_full).sum(axis=-1))
                l1_full = np.abs(diff_full).sum(axis=-1)
                errs_full.append(l2_full)
                errs_l1_full.append(l1_full)

                if Q2p > 0:
                    pred_mid = pred[Q1p:-Q2p]
                    gt_mid = gt_metric[Q1p:-Q2p]
                else:
                    pred_mid = pred[Q1p:]
                    gt_mid = gt_metric[Q1p:]
                diff_mid = pred_mid - gt_mid
                l2_mid = np.sqrt((diff_mid * diff_mid).sum(axis=-1))
                l1_mid = np.abs(diff_mid).sum(axis=-1)
                errs_mid.append(l2_mid)
                errs_l1_mid.append(l1_mid)

            errs_full = np.stack(errs_full, axis=0)
            errs_l1_full = np.stack(errs_l1_full, axis=0)
            errs_mid = np.stack(errs_mid, axis=0)
            errs_l1_mid = np.stack(errs_l1_mid, axis=0)
            point_mean = np.mean(errs_full, axis=0, dtype=np.float64)
            row = {
                "model_name": display_name,
                "model_full_name": model_name,
                "model_tag": model_tag,
                "device": runtime_device or "unknown",
                "dataset_name": dataset_name,
                "K": decoder.K,
                "Q1": decoder.Q1_bytes,
                "Q2": decoder.Q2_bytes,
                "err_l1_mean_full": float(errs_l1_full.mean()),
                "err_l1_median_full": float(np.median(errs_l1_full)),
                "err_l1_p95_full": float(np.percentile(errs_l1_full, 95)),
                "err_l1_std_full": float(errs_l1_full.std()),
                "err_mean_full": float(errs_full.mean()),
                "err_median_full": float(np.median(errs_full)),
                "err_p95_full": float(np.percentile(errs_full, 95)),
                "err_std_full": float(errs_full.std()),
                "err_l1_mean_mid": float(errs_l1_mid.mean()),
                "err_l1_median_mid": float(np.median(errs_l1_mid)),
                "err_l1_p95_mid": float(np.percentile(errs_l1_mid, 95)),
                "err_l1_std_mid": float(errs_l1_mid.std()),
                "err_mean_mid": float(errs_mid.mean()),
                "err_median_mid": float(np.median(errs_mid)),
                "err_p95_mid": float(np.percentile(errs_mid, 95)),
                "err_std_mid": float(errs_mid.std()),
                "num_tested_chunks": num_chunks,
                "test_timestamp": datetime.now().isoformat(),
            }
            self.chunk_evaluator._append_row(row)
            results.append(row)
            pointwise_rows.append({
                "model_name": display_name,
                "model_full_name": model_name,
                "model_tag": model_tag,
                "dataset_name": dataset_name,
                "model_root": str(Path(model_root)),
                "Q1": decoder.Q1_bytes,
                "Q2": decoder.Q2_bytes,
                "point_mean": point_mean,
            })
            chunk_p_val_rows.extend(
                _build_chunk_p_val_rows(
                    model_name=display_name,
                    model_full_name=model_name,
                    model_tag_value=model_tag,
                    device_value=runtime_device or "unknown",
                    k_value=decoder.K,
                    q1_value=decoder.Q1_bytes,
                    q2_value=decoder.Q2_bytes,
                    errs_full_arr=errs_full,
                    errs_l1_full_arr=errs_l1_full,
                    errs_mid_arr=errs_mid,
                    errs_l1_mid_arr=errs_l1_mid,
                    timestamp_value=row["test_timestamp"],
                )
            )
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()

        self.chunk_evaluator.save_pointwise_heatmap(pointwise_rows, dataset_name=dataset_name)
        self.chunk_evaluator.save_chunk_p_val_rows(chunk_p_val_rows)
        return results

    def run_chunk_batch(
        self,
        *,
        job: Dict,
        model_root: str,
        model_names: Optional[List],
        classic_baselines: List[str],
        model_tag: str,
        run_baselines: bool,
        max_workers: int = 4,
        log_level: str = "INFO",
    ) -> None:
        _run_chunk_batch(
            manager=self,
            job=job,
            model_root=model_root,
            model_names=model_names,
            classic_baselines=classic_baselines,
            model_tag=model_tag,
            run_baselines=run_baselines,
            max_workers=max_workers,
            log_level=log_level,
        )

    # ============================================================
    # UNCERTAINTY BAND TEST
    # ============================================================
    def run_uncertainty_band_test(
        self,
        model_names: Optional[List[str]] = None,
        model_root: str = "./bin/model/RectifiedTraj",
        model_tag: str = "RectifiedTraj",
        test_data_path: str = "./dataset/processed/NUMOSIM_Kanto/test/traj_test/full_traj_range",
        M: int = 200,
        N: int = 10000,
        run_baselines: bool = True,
        baseline_methods: Optional[List[str]] = None,
        manual_config: Optional[Dict] = None,
        progress_unit_offset: int = 0,
        progress_total_units: Optional[int] = None,
    ) -> List[Dict]:
        if model_names is None:
            model_names = self._discover_models(model_root)

        self.logger.debug("Found %d models to evaluate (uncertainty band)", len(model_names))

        test_trajectories = self._load_or_generate_uncertainty_test_data(
            test_data_path, M, N
        )
        dataset_name = self._resolve_dataset_display_name(test_data_path)
        baseline_dataset_name = self._infer_dataset_name_from_test_path(Path(test_data_path)) or dataset_name

        tester = self._new_uncertainty_tester()
        try:
            tester.log_uncertainty_dataset_info(
                test_trajectories=test_trajectories,
                dataset_name=dataset_name,
            )
        except Exception as exc:
            self.logger.warning("Failed to log uncertainty dataset info: %s", exc)
        all_results = []
        baseline_count = len(list(baseline_methods or [])) if run_baselines else 0
        traj_count = len(test_trajectories)
        baseline_units = baseline_count * traj_count
        if run_baselines:
            self.logger.debug("Running classic baselines (uncertainty band)")
            baseline_results = tester.evaluate_classic_baselines(
                test_trajectories=test_trajectories,
                dataset_name=dataset_name,
                baseline_dataset_name=baseline_dataset_name,
                methods=baseline_methods,
                progress_unit_offset=progress_unit_offset,
                progress_total_units=progress_total_units,
            )
            all_results.extend(baseline_results)

        progress_tracker = ProgressTracker(
            total_models=len(model_names),
            total_q1=1,
            total_q2=1,
            total_step=1,
            total_method=1,
            unit_offset=progress_unit_offset + baseline_units,
            global_total_units=progress_total_units,
        )
        progress_tracker.update(
            phase="uncertainty",
            dataset=dataset_name,
            total_traj=traj_count,
        )

        for model_idx, model_name in enumerate(model_names):
            model_dir = Path(model_root) / model_name

            checkpoint_name = self._find_best_checkpoint(model_dir)
            if checkpoint_name is None:
                self.logger.warning(f"No checkpoint found for {model_name}, skipping")
                continue

            config = self._load_model_config(model_dir)
            K = config.get("K", 256)
            Q1 = config.get("Q1", 1)
            Q2 = config.get("Q2", 12)
            if manual_config is not None:
                Q1 = int(manual_config.get("Q1", Q1))
                Q2 = int(manual_config.get("Q2", Q2))

            result = tester.evaluate_model(
                model_name=model_name,
                model_dir=str(model_dir.absolute()),
                checkpoint_name=checkpoint_name,
                test_trajectories=test_trajectories,
                K=K,
                Q1=Q1,
                Q2=Q2,
                model_tag=model_tag,
                manual_config=manual_config,
                dataset_name=dataset_name,
                progress_tracker=progress_tracker,
                model_idx=model_idx,
            )

            all_results.append(result)

        uncertainty_pval_csv = Path(self.output_dir) / "uncertainty_traj_p_val.csv"
        if uncertainty_pval_csv.exists():
            generate_pairwise_p_value_report(
                uncertainty_pval_csv,
                Path(self.output_dir) / "uncertainty_p_value_summary",
                sample_type="uncertainty_trajectory",
                metric_column="pass_rate_points",
            )

        self.logger.debug("Completed %d uncertainty-band evaluations", len(all_results))
        return all_results

    def bounded_trajectory_test(
        self,
        model_name: str,
        model_root: str = "./bin/model/RectifiedTraj",
        model_tag: str = "RectifiedTraj",
        test_data_path: str = "./dataset/processed/NUMOSIM_Kanto/test/traj_test/full_traj_range",
        M: int = 200,
        N: int = 10000,
        checkpoint_name: Optional[str] = None,
        run_baselines: bool = False,
        manual_config: Optional[Dict] = None,
    ) -> List[Dict]:
        model_dir = Path(model_root) / model_name
        if checkpoint_name is None:
            checkpoint_name = self._find_best_checkpoint(model_dir)
        if checkpoint_name is None:
            raise FileNotFoundError(f"No checkpoint found for {model_name}")

        config = self._load_model_config(model_dir)
        K = config.get("K", 256)
        Q1 = config.get("Q1", 1)
        Q2 = config.get("Q2", 12)
        if manual_config is not None:
            Q1 = int(manual_config.get("Q1", Q1))
            Q2 = int(manual_config.get("Q2", Q2))

        test_trajectories = self._load_or_generate_uncertainty_test_data(
            test_data_path, M, N
        )
        dataset_name = self._resolve_dataset_display_name(test_data_path)
        baseline_dataset_name = self._infer_dataset_name_from_test_path(Path(test_data_path)) or dataset_name

        tester = self._new_uncertainty_tester()
        try:
            tester.log_uncertainty_dataset_info(
                test_trajectories=test_trajectories,
                dataset_name=dataset_name,
            )
        except Exception as exc:
            self.logger.warning("Failed to log uncertainty dataset info: %s", exc)
        results = []
        if run_baselines:
            baseline_results = tester.evaluate_classic_baselines(
                test_trajectories=test_trajectories,
                dataset_name=dataset_name,
                baseline_dataset_name=baseline_dataset_name,
            )
            results.extend(baseline_results)

        result = tester.evaluate_model(
            model_name=model_name,
            model_dir=str(model_dir.absolute()),
            checkpoint_name=checkpoint_name,
            test_trajectories=test_trajectories,
            K=K,
            Q1=Q1,
            Q2=Q2,
            model_tag=model_tag,
            manual_config=manual_config,
            dataset_name=dataset_name,
        )
        results.append(result)

        return results

    def bounded_trajectory_test_all_models(
        self,
        model_names: Optional[List[str]] = None,
        model_root: str = "./bin/model/RectifiedTraj",
        model_tag: str = "RectifiedTraj",
        test_data_path: str = "./dataset/processed/NUMOSIM_Kanto/test/traj_test/full_traj_range",
        M: int = 200,
        N: int = 10000,
    ) -> List[Dict]:
        return self.run_uncertainty_band_test(
            model_names=model_names,
            model_root=model_root,
            model_tag=model_tag,
            test_data_path=test_data_path,
            M=M,
            N=N,
        )
