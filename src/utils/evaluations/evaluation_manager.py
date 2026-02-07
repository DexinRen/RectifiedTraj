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
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
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
        model_root: str = "./bin/model",
        test_data_path: str = "./dataset/processed/full_traj",
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
                        dataset_name=dataset_name,
                    )

                all_results.append(result)
                progress_tracker.update(job_finished=True)

        self.logger.info(f"Completed {len(all_results)} evaluations")
        try:
            from utils.data_visualizer.visualizer_traj_test import main as visualize_trajectory_results
            visualize_trajectory_results(
                csv_path=str(self.trajectory_evaluator.csv_path),
                parquet_dir=str(self.trajectory_evaluator.parquet_dir),
                output_dir=str(self.trajectory_evaluator.run_dir or self.output_dir),
            )
        except Exception as exc:
            self.logger.warning(f"Visualization failed: {exc}")
        return all_results

    # ============================================================
    # PART 2: EvaluationManager grid search
    # ============================================================
    def run_grid_search_evaluation(
        self,
        job_list: Dict,
        model_names: Optional[List[str]] = None,
        model_root: str = "./bin/model",
        test_data_path: str = "./dataset/processed/full_traj",
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

        if getattr(self, "visualize_each_run", False):
            try:
                from utils.data_visualizer.visualizer_traj_test import main as visualize_trajectory_results
                visualize_trajectory_results(brief=getattr(self, "brief_visualizer", False))
            except Exception as exc:
                self.logger.warning(f"Visualization failed: {exc}")

        return all_results

    # ============================================================
    # PUBLIC: Run chunk-wise evaluation
    # ============================================================
    def run_chunk_evaluation(
        self,
        model_names: Optional[List[str]] = None,
        model_root: str = "./bin/model",
        test_dir: str = "./dataset/processed/test",
        max_chunks: int = 5000,
        manual_config: Optional[Dict] = None,
        run_baselines: bool = True,
        baseline_methods: Optional[List[str]] = None,
    ) -> List[Dict]:
        from theta_train import DataLoader
        from baseline import classic as classic_baseline
        from datetime import datetime
        import time

        self.logger.info("Starting chunk-wise evaluation...")
        self.logger.info(f"  test_dir: {test_dir}")
        self.logger.info(f"  max_chunks: {max_chunks}")

        if model_names is None:
            model_names = self._discover_models(model_root)
        self.logger.info(f"  models: {len(model_names)}")

        runtime_template = {
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "config": {
                "batch_size": 64,
                "train_dir": test_dir,
                "K": 256,
            },
        }

        test_files = sorted(Path(test_dir).glob("*.pt"))
        if not test_files:
            raise FileNotFoundError(f"No .pt files found under {test_dir}")
        self.logger.info(f"Found {len(test_files)} test chunk files")

        X0_list = []
        X1_list = []
        timestamps_list = []
        for i, _ in enumerate(test_files):
            rt = dict(runtime_template)
            rt["config"] = dict(runtime_template["config"])
            rt["config"]["train_dir"] = str(Path(test_dir))
            dl = DataLoader(rt)
            dl.set(i)
            # Keep model inputs float32 while preserving timestamps precision.
            Xt = dl.X_t[:, :, :2].float()
            ts = dl.X_t[:, :, 2].cpu()
            V = dl.V[:, :, :2]
            t = dl.t
            t_b = t.view(-1, 1, 1)
            X0 = Xt - V * t_b
            X1 = Xt + V * (1.0 - t_b)
            X0_list.append(X0.cpu())
            X1_list.append(X1.cpu())
            timestamps_list.append(ts)

        X0 = torch.cat(X0_list, dim=0)
        X1 = torch.cat(X1_list, dim=0)
        timestamps = torch.cat(timestamps_list, dim=0) if timestamps_list else None
        total = X0.shape[0]
        if max_chunks is not None and max_chunks > 0 and total > max_chunks:
            X0 = X0[:max_chunks]
            X1 = X1[:max_chunks]
            if timestamps is not None:
                timestamps = timestamps[:max_chunks]
        num_chunks = X0.shape[0]
        self.logger.info(f"Loaded {num_chunks} chunks (of {total})")

        dataset_name = "chunk_test"
        results = []
        bytewise_rows = []

        baseline_method_table = [
            ("kalman_rts_ts", classic_baseline.kalman_rts_smoother),
            ("kalman_rts_notime", classic_baseline.kalman_rts_smoother),
            ("hampel", classic_baseline.hampel_filter),
            ("savgol", classic_baseline.savitzky_golay_filter),
            ("spline", classic_baseline.smoothing_spline),
            ("raw", classic_baseline.raw_baseline),
        ]
        if baseline_methods is None:
            selected_baselines = baseline_method_table
        else:
            allowed = set(baseline_methods)
            selected_baselines = [(name, fn) for name, fn in baseline_method_table if name in allowed]
            unknown = [name for name in baseline_methods if name not in {n for n, _ in baseline_method_table}]
            for name in unknown:
                self.logger.warning("Unknown chunk baseline ignored: %s", name)

        Q1_bytes = (manual_config or {}).get("Q1", 1)
        Q2_bytes = (manual_config or {}).get("Q2", 12)
        Q1p = Q1_bytes * 8
        Q2p = Q2_bytes * 8
        bar_width = 30

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

        if run_baselines:
            if not selected_baselines:
                self.logger.warning("No chunk classic baselines selected; skipping classic chunk baselines.")
            for method_name, method_fn in selected_baselines:
                self.logger.info(f"[Baseline] {method_name}")
                errs_full = []
                errs_mid = []
                byte_sum = np.zeros(32, dtype=np.float64)
                byte_cnt = np.zeros(32, dtype=np.int64)
                times = []

                for i in range(num_chunks):
                    _chunk_bar(i + 1, num_chunks, method_name, Q1_bytes, Q2_bytes, "N/A", None)
                    inp = X1[i].numpy()
                    gt = X0[i].numpy()
                    t0 = time.perf_counter()
                    if method_name == "kalman_rts_ts":
                        ts = timestamps[i].numpy() if timestamps is not None else None
                        if ts is not None:
                            ts = ts.astype(np.float64, copy=False)
                            if ts.size and np.isfinite(ts[0]):
                                ts = ts - float(ts[0])
                        pred = method_fn(inp, timestamps=ts)
                    elif method_name == "kalman_rts_notime":
                        pred = method_fn(inp, timestamps=None)
                    else:
                        pred = method_fn(inp)
                    t1 = time.perf_counter()
                    times.append(t1 - t0)

                    diff_full = pred - gt
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
                        gt_mid = gt[Q1p:-Q2p]
                    else:
                        pred_mid = pred[Q1p:]
                        gt_mid = gt[Q1p:]
                    diff_mid = pred_mid - gt_mid
                    l2_mid = np.sqrt((diff_mid * diff_mid).sum(axis=-1))
                    errs_mid.append(l2_mid)

                errs_full = np.stack(errs_full, axis=0)
                errs_mid = np.stack(errs_mid, axis=0)
                avg_time = float(np.mean(times)) if times else None
                avg_time_per_point = (avg_time / X0.shape[1]) if avg_time is not None and X0.shape[1] else None
                row = {
                    "model_name": method_name,
                    "model_tag": "Baseline",
                    "dataset_name": dataset_name,
                    "denoise_method": "N/A",
                    "K": 256,
                    "Q1": Q1_bytes,
                    "Q2": Q2_bytes,
                    "t_delta": None,
                    "N_steps": None,
                    "avg_time_s": float(avg_time),
                    "avg_time_per_point_s": float(avg_time_per_point) if avg_time_per_point is not None else None,
                    "err_mean_full": float(errs_full.mean()),
                    "err_median_full": float(np.median(errs_full)),
                    "err_std_full": float(errs_full.std()),
                    "err_mean_mid": float(errs_mid.mean()),
                    "err_median_mid": float(np.median(errs_mid)),
                    "err_std_mid": float(errs_mid.std()),
                    "num_tested_chunks": num_chunks,
                    "test_timestamp": datetime.now().isoformat(),
                }
                self.chunk_evaluator._append_row(row)
                results.append(row)
                byte_mean = np.divide(byte_sum, np.maximum(byte_cnt, 1))
                bytewise_rows.append({
                    "model_name": method_name,
                    "model_tag": "Baseline",
                    "dataset_name": dataset_name,
                    "byte_mean": byte_mean,
                })
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

            for i in range(num_chunks):
                _chunk_bar(i + 1, num_chunks, display_name, decoder.Q1_bytes, decoder.Q2_bytes, "N/A", decoder.t_delta)
                inp = X1[i].numpy()
                gt = X0[i].numpy()
                t0 = time.perf_counter()
                pred = decoder.denoise_chunk_enu(inp)
                t1 = time.perf_counter()
                times.append(t1 - t0)

                diff_full = pred - gt
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
                    gt_mid = gt[Q1p:-Q2p]
                else:
                    pred_mid = pred[Q1p:]
                    gt_mid = gt[Q1p:]
                diff_mid = pred_mid - gt_mid
                l2_mid = np.sqrt((diff_mid * diff_mid).sum(axis=-1))
                errs_mid.append(l2_mid)

            errs_full = np.stack(errs_full, axis=0)
            errs_mid = np.stack(errs_mid, axis=0)
            avg_time = float(np.mean(times)) if times else None
            avg_time_per_point = (avg_time / X0.shape[1]) if avg_time is not None and X0.shape[1] else None
            row = {
                "model_name": display_name,
                "model_tag": "RectifiedTraj",
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
                "err_std_full": float(errs_full.std()),
                "err_mean_mid": float(errs_mid.mean()),
                "err_median_mid": float(np.median(errs_mid)),
                "err_std_mid": float(errs_mid.std()),
                "num_tested_chunks": num_chunks,
                "test_timestamp": datetime.now().isoformat(),
            }
            self.chunk_evaluator._append_row(row)
            results.append(row)
            byte_mean = np.divide(byte_sum, np.maximum(byte_cnt, 1))
            bytewise_rows.append({
                "model_name": display_name,
                "model_tag": "RectifiedTraj",
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
        model_root: str = "./bin/model",
        test_data_path: str = "./dataset/processed/full_traj_range",
        M: int = 200,
        N: int = 10000,
        run_baselines: bool = True,
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
        all_results = []
        if run_baselines:
            self.logger.info("Running classic baselines (uncertainty band)")
            baseline_results = tester.evaluate_classic_baselines(
                test_trajectories=test_trajectories
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
                )

                all_results.append(result)

        self.logger.info(f"Completed {len(all_results)} uncertainty-band evaluations")
        return all_results

    def bounded_trajectory_test(
        self,
        model_name: str,
        denoise_methods: List[str] = None,
        model_root: str = "./bin/model",
        test_data_path: str = "./dataset/processed/full_traj_range",
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
            )
            results.append(result)

        return results

    def bounded_trajectory_test_all_models(
        self,
        model_names: Optional[List[str]] = None,
        denoise_methods: List[str] = None,
        model_root: str = "./bin/model",
        test_data_path: str = "./dataset/processed/full_traj_range",
        M: int = 200,
        N: int = 10000,
    ) -> List[Dict]:
        return self.run_uncertainty_band_test(
            model_names=model_names,
            denoise_methods=denoise_methods,
            model_root=model_root,
            test_data_path=test_data_path,
            M=M,
            N=N,
        )
