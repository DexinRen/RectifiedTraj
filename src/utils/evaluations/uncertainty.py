import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from encoder_decoder import EncoderDecoder
from utils.evaluations.trajectory import TrajectoryEvaluator


class UncertaintyBandTrajectoryTest:
    """
    Evaluate trajectory denoising against reference centers with per-point error ranges.

    A point is a PASS if distance(denoised, reference) <= error_range.
    """

    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = Path(output_dir)
        self.csv_path = self.output_dir / "uncertainty_band_summary.csv"
        self.detail_dir = self.output_dir / "uncertainty_band_traj_test_result"
        self.detail_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("UncertaintyBandTrajectoryTest")

        header = (
            "model_name,denoise_method,K,Q1,Q2,t_delta,N_steps,"
            "pass_rate_points,pass_rate_trajectories,avg_outside_error,"
            "data_avg_sample_time_sec,data_median_sample_time_sec,data_std_sample_time_sec,"
            "mean_distance_all,mean_signed_margin_all,"
            "tier4_points_all,tier4_pass_rate_points_all,tier4_pass_rate_trajectories_all,tier4_mean_distance_all,tier4_mean_signed_margin_all,"
            "tier3_points_acc_leq_30,tier3_pass_rate_points_acc_leq_30,tier3_pass_rate_trajectories_acc_leq_30,tier3_mean_distance_acc_leq_30,tier3_mean_signed_margin_acc_leq_30,"
            "tier2_points_acc_leq_15,tier2_pass_rate_points_acc_leq_15,tier2_pass_rate_trajectories_acc_leq_15,tier2_mean_distance_acc_leq_15,tier2_mean_signed_margin_acc_leq_15,"
            "tier1_points_acc_leq_10,tier1_pass_rate_points_acc_leq_10,tier1_pass_rate_trajectories_acc_leq_10,tier1_mean_distance_acc_leq_10,tier1_mean_signed_margin_acc_leq_10,"
            "tier0_points_acc_leq_5,tier0_pass_rate_points_acc_leq_5,tier0_pass_rate_trajectories_acc_leq_5,tier0_mean_distance_acc_leq_5,tier0_mean_signed_margin_acc_leq_5,"
            "num_tested_trajectories,num_tested_points,longest_trajectory_length,test_timestamp\n"
        )

        if self.csv_path.exists():
            existing_header = self.csv_path.read_text().splitlines()[:1]
            if existing_header and existing_header[0] != header.strip():
                self.logger.warning("Uncertainty summary header mismatch. Writing to a new file.")
                self.csv_path = self.output_dir / "uncertainty_band_summary_v5.csv"

        if not self.csv_path.exists():
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
        manual_config: Optional[Dict] = None,
    ) -> Dict:
        self.logger.info(f"Evaluating {model_name} with {denoise_method} (uncertainty band)")

        checkpoint_path = self._get_checkpoint_path(model_dir, checkpoint_name)
        if checkpoint_path is None:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_name}")

        denoised_trajectories, decoder = self._denoise_trajectories(
            checkpoint_path, test_trajectories, denoise_method, manual_config=manual_config
        )

        t_delta = float(getattr(decoder, "t_delta", 1.0))
        N_steps = int(1.0 / t_delta) if t_delta > 0 else 1

        metrics = self._compute_pass_metrics(
            denoised_trajectories, test_trajectories
        )

        sample_stats = self._compute_sample_time_stats(test_trajectories)

        results = {
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
            "longest_trajectory_length": int(max(len(t.noisy_gps) for t in test_trajectories)) if test_trajectories else 0,
            "pass_rate_points": metrics["pass_rate_points"],
            "pass_rate_trajectories": metrics["pass_rate_trajectories"],
            "avg_outside_error": metrics["avg_outside_error"],
            "data_avg_sample_time_sec": sample_stats[0],
            "data_median_sample_time_sec": sample_stats[1],
            "data_std_sample_time_sec": sample_stats[2],
            "mean_distance_all": metrics["mean_distance_all"],
            "mean_signed_margin_all": metrics["mean_signed_margin_all"],
            "tier4_points": metrics["tier4_points"],
            "tier4_pass_rate_points": metrics["tier4_pass_rate_points"],
            "tier4_pass_rate_trajectories": metrics["tier4_pass_rate_trajectories"],
            "tier4_mean_distance": metrics["tier4_mean_distance"],
            "tier4_mean_signed_margin": metrics["tier4_mean_signed_margin"],
            "tier3_points": metrics["tier3_points"],
            "tier3_pass_rate_points": metrics["tier3_pass_rate_points"],
            "tier3_pass_rate_trajectories": metrics["tier3_pass_rate_trajectories"],
            "tier3_mean_distance": metrics["tier3_mean_distance"],
            "tier3_mean_signed_margin": metrics["tier3_mean_signed_margin"],
            "tier2_points": metrics["tier2_points"],
            "tier2_pass_rate_points": metrics["tier2_pass_rate_points"],
            "tier2_pass_rate_trajectories": metrics["tier2_pass_rate_trajectories"],
            "tier2_mean_distance": metrics["tier2_mean_distance"],
            "tier2_mean_signed_margin": metrics["tier2_mean_signed_margin"],
            "tier1_points": metrics["tier1_points"],
            "tier1_pass_rate_points": metrics["tier1_pass_rate_points"],
            "tier1_pass_rate_trajectories": metrics["tier1_pass_rate_trajectories"],
            "tier1_mean_distance": metrics["tier1_mean_distance"],
            "tier1_mean_signed_margin": metrics["tier1_mean_signed_margin"],
            "tier0_points": metrics["tier0_points"],
            "tier0_pass_rate_points": metrics["tier0_pass_rate_points"],
            "tier0_pass_rate_trajectories": metrics["tier0_pass_rate_trajectories"],
            "tier0_mean_distance": metrics["tier0_mean_distance"],
            "tier0_mean_signed_margin": metrics["tier0_mean_signed_margin"],
        }

        self._save_results(results)
        distances_list, error_ranges_list = self._compute_distances_and_ranges(
            denoised_trajectories, test_trajectories
        )
        self._save_pointwise_aggregates(
            distances_list=distances_list,
            error_ranges_list=error_ranges_list,
            model_name=model_name,
            denoise_method=denoise_method,
            K=K,
            Q1=Q1,
            Q2=Q2,
            t_delta=t_delta,
            N_steps=N_steps,
            test_timestamp=results["test_timestamp"],
        )
        self.logger.info(f"Uncertainty band evaluation complete: {model_name} {denoise_method}")
        return results

    def evaluate_classic_baselines(
        self,
        test_trajectories: List,
        dataset_name: Optional[str] = None,
    ) -> List[Dict]:
        from baseline import classic as classic_baseline

        methods = [
            ("kalman_rts_ts", classic_baseline.kalman_rts_smoother),
            ("kalman_rts_notime", classic_baseline.kalman_rts_smoother),
            ("hampel", classic_baseline.hampel_filter),
            ("savgol", classic_baseline.savitzky_golay_filter),
            ("spline", classic_baseline.smoothing_spline),
        ]

        results = []
        sample_stats = self._compute_sample_time_stats(test_trajectories)
        for method_name, method_fn in methods:
            distances_list = []
            error_ranges_list = []

            for traj_obj in test_trajectories:
                noisy_gps = traj_obj.noisy_gps
                ref_gps = traj_obj.ref_gps
                error_range = traj_obj.error_range

                ref_lat = float(ref_gps[0, 1])
                ref_lon = float(ref_gps[0, 0])
                enu_noisy = self._gps_to_enu_batch(noisy_gps, ref_lat, ref_lon)
                enu_ref = self._gps_to_enu_batch(ref_gps, ref_lat, ref_lon)

                try:
                    if method_name == "kalman_rts_ts":
                        denoised_enu = method_fn(
                            enu_noisy, timestamps=getattr(traj_obj, "timestamps", None)
                        )
                    elif method_name == "kalman_rts_notime":
                        denoised_enu = method_fn(enu_noisy, timestamps=None)
                    else:
                        denoised_enu = method_fn(enu_noisy)
                except TypeError:
                    denoised_enu = method_fn(enu_noisy, timestamps=None)

                T = min(len(denoised_enu), len(enu_ref), len(error_range))
                if T <= 0:
                    continue

                distances = np.linalg.norm(denoised_enu[:T] - enu_ref[:T], axis=1)
                distances_list.append(distances)
                error_ranges_list.append(error_range[:T])

            if not distances_list:
                self.logger.warning("No trajectories for classic baseline: %s", method_name)
                continue

            metrics = self._compute_pass_metrics_from_distances(distances_list, error_ranges_list)

            results_row = {
                "model_name": method_name,
                "model_dir": None,
                "checkpoint_name": None,
                "K": None,
                "Q1": None,
                "Q2": None,
                "t_delta": 1.0,
                "N_steps": 1,
                "denoise_method": "Baseline",
                "test_timestamp": datetime.now().isoformat(),
                "num_tested_trajectories": len(distances_list),
                "num_tested_points": int(sum(len(d) for d in distances_list)),
                "longest_trajectory_length": int(max(len(d) for d in distances_list)) if distances_list else 0,
                "pass_rate_points": metrics["pass_rate_points"],
                "pass_rate_trajectories": metrics["pass_rate_trajectories"],
                "avg_outside_error": metrics["avg_outside_error"],
                "data_avg_sample_time_sec": sample_stats[0],
                "data_median_sample_time_sec": sample_stats[1],
                "data_std_sample_time_sec": sample_stats[2],
                "mean_distance_all": metrics["mean_distance_all"],
                "mean_signed_margin_all": metrics["mean_signed_margin_all"],
                "tier4_points": metrics["tier4_points"],
                "tier4_pass_rate_points": metrics["tier4_pass_rate_points"],
                "tier4_pass_rate_trajectories": metrics["tier4_pass_rate_trajectories"],
                "tier4_mean_distance": metrics["tier4_mean_distance"],
                "tier4_mean_signed_margin": metrics["tier4_mean_signed_margin"],
                "tier3_points": metrics["tier3_points"],
                "tier3_pass_rate_points": metrics["tier3_pass_rate_points"],
                "tier3_pass_rate_trajectories": metrics["tier3_pass_rate_trajectories"],
                "tier3_mean_distance": metrics["tier3_mean_distance"],
                "tier3_mean_signed_margin": metrics["tier3_mean_signed_margin"],
                "tier2_points": metrics["tier2_points"],
                "tier2_pass_rate_points": metrics["tier2_pass_rate_points"],
                "tier2_pass_rate_trajectories": metrics["tier2_pass_rate_trajectories"],
                "tier2_mean_distance": metrics["tier2_mean_distance"],
                "tier2_mean_signed_margin": metrics["tier2_mean_signed_margin"],
                "tier1_points": metrics["tier1_points"],
                "tier1_pass_rate_points": metrics["tier1_pass_rate_points"],
                "tier1_pass_rate_trajectories": metrics["tier1_pass_rate_trajectories"],
                "tier1_mean_distance": metrics["tier1_mean_distance"],
                "tier1_mean_signed_margin": metrics["tier1_mean_signed_margin"],
                "tier0_points": metrics["tier0_points"],
                "tier0_pass_rate_points": metrics["tier0_pass_rate_points"],
                "tier0_pass_rate_trajectories": metrics["tier0_pass_rate_trajectories"],
                "tier0_mean_distance": metrics["tier0_mean_distance"],
                "tier0_mean_signed_margin": metrics["tier0_mean_signed_margin"],
            }

            self._save_results(results_row)
            self._save_pointwise_aggregates(
                distances_list=distances_list,
                error_ranges_list=error_ranges_list,
                model_name=method_name,
                denoise_method="Baseline",
                K=None,
                Q1=None,
                Q2=None,
                t_delta=1.0,
                N_steps=1,
                test_timestamp=results_row["test_timestamp"],
            )
            results.append(results_row)

        return results

    def evaluate_difftraj_baseline(
        self,
        test_trajectories: List,
        *,
        repo_dir: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        timesteps: int = 100,
        final_steps: Optional[int] = None,
        eta: float = 0.0,
        dataset_name: Optional[str] = None,
    ) -> List[Dict]:
        from baseline.difftraj import (
            DiffTrajPaths,
            difftraj_denoise_with_model,
            prepare_difftraj,
        )

        paths = DiffTrajPaths(repo_dir=repo_dir, checkpoint_path=checkpoint_path)
        try:
            config, model, device = prepare_difftraj(paths, device=device)
        except FileNotFoundError as exc:
            self.logger.warning("DiffTraj baseline unavailable: %s", str(exc))
            return []
        except Exception as exc:
            self.logger.warning(
                "DiffTraj baseline failed to load (%s): %s", type(exc).__name__, str(exc)
            )
            return []

        def _pad_tail(values: np.ndarray, target_len: int) -> np.ndarray:
            values = np.asarray(values, dtype=float)
            if values.shape[0] == target_len:
                return values
            if values.shape[0] > target_len:
                return values[:target_len]
            pad_count = target_len - values.shape[0]
            tail = np.repeat(values[-1:], pad_count, axis=0)
            return np.concatenate([values, tail], axis=0)

        def _denoise_chunked(enu_noisy: np.ndarray, target_len: int) -> np.ndarray:
            total_len = enu_noisy.shape[0]
            if total_len <= target_len:
                chunk = _pad_tail(enu_noisy, target_len)
                den = difftraj_denoise_with_model(
                    chunk,
                    config=config,
                    model=model,
                    device=device,
                    timesteps=timesteps,
                    final_steps=final_steps,
                    eta=eta,
                )
                return den[:total_len]

            stride = target_len
            outputs = []
            idx = 0
            while idx < total_len:
                chunk = enu_noisy[idx : idx + target_len]
                chunk = _pad_tail(chunk, target_len)
                den = difftraj_denoise_with_model(
                    chunk,
                    config=config,
                    model=model,
                    device=device,
                    timesteps=timesteps,
                    final_steps=final_steps,
                    eta=eta,
                )
                outputs.append(den)
                idx += stride

            stitched = np.concatenate(outputs, axis=0)
            return stitched[:total_len]

        distances_list = []
        error_ranges_list = []
        target_len = int(getattr(getattr(config, "data", None), "traj_length", 0) or 0)

        for traj_obj in test_trajectories:
            noisy_gps = traj_obj.noisy_gps
            ref_gps = traj_obj.ref_gps
            error_range = traj_obj.error_range

            if len(noisy_gps) == 0:
                continue

            ref_lat = float(ref_gps[0, 1])
            ref_lon = float(ref_gps[0, 0])
            enu_noisy = self._gps_to_enu_batch(noisy_gps, ref_lat, ref_lon)
            enu_ref = self._gps_to_enu_batch(ref_gps, ref_lat, ref_lon)

            try:
                if target_len > 0 and enu_noisy.shape[0] != target_len:
                    denoised_enu = _denoise_chunked(enu_noisy, target_len)
                else:
                    denoised_enu = difftraj_denoise_with_model(
                        enu_noisy,
                        config=config,
                        model=model,
                        device=device,
                        timesteps=timesteps,
                        final_steps=final_steps,
                        eta=eta,
                    )
            except Exception as exc:
                self.logger.warning(
                    "DiffTraj failed on one trajectory (%s): %s",
                    type(exc).__name__,
                    str(exc),
                )
                continue

            T = min(len(denoised_enu), len(enu_ref), len(error_range))
            if T <= 0:
                continue

            distances = np.linalg.norm(denoised_enu[:T] - enu_ref[:T], axis=1)
            distances_list.append(distances)
            error_ranges_list.append(error_range[:T])

        if not distances_list:
            self.logger.warning("No trajectories for DiffTraj baseline.")
            return []

        metrics = self._compute_pass_metrics_from_distances(distances_list, error_ranges_list)
        sample_stats = self._compute_sample_time_stats(test_trajectories)

        results_row = {
            "model_name": "difftraj",
            "model_dir": None,
            "checkpoint_name": None,
            "K": None,
            "Q1": None,
            "Q2": None,
            "t_delta": 1.0,
            "N_steps": 1,
            "denoise_method": "Baseline",
            "test_timestamp": datetime.now().isoformat(),
            "num_tested_trajectories": len(distances_list),
            "num_tested_points": int(sum(len(d) for d in distances_list)),
            "longest_trajectory_length": int(max(len(d) for d in distances_list)) if distances_list else 0,
            "pass_rate_points": metrics["pass_rate_points"],
            "pass_rate_trajectories": metrics["pass_rate_trajectories"],
            "avg_outside_error": metrics["avg_outside_error"],
            "data_avg_sample_time_sec": sample_stats[0],
            "data_median_sample_time_sec": sample_stats[1],
            "data_std_sample_time_sec": sample_stats[2],
            "mean_distance_all": metrics["mean_distance_all"],
            "mean_signed_margin_all": metrics["mean_signed_margin_all"],
            "tier4_points": metrics["tier4_points"],
            "tier4_pass_rate_points": metrics["tier4_pass_rate_points"],
            "tier4_pass_rate_trajectories": metrics["tier4_pass_rate_trajectories"],
            "tier4_mean_distance": metrics["tier4_mean_distance"],
            "tier4_mean_signed_margin": metrics["tier4_mean_signed_margin"],
            "tier3_points": metrics["tier3_points"],
            "tier3_pass_rate_points": metrics["tier3_pass_rate_points"],
            "tier3_pass_rate_trajectories": metrics["tier3_pass_rate_trajectories"],
            "tier3_mean_distance": metrics["tier3_mean_distance"],
            "tier3_mean_signed_margin": metrics["tier3_mean_signed_margin"],
            "tier2_points": metrics["tier2_points"],
            "tier2_pass_rate_points": metrics["tier2_pass_rate_points"],
            "tier2_pass_rate_trajectories": metrics["tier2_pass_rate_trajectories"],
            "tier2_mean_distance": metrics["tier2_mean_distance"],
            "tier2_mean_signed_margin": metrics["tier2_mean_signed_margin"],
            "tier1_points": metrics["tier1_points"],
            "tier1_pass_rate_points": metrics["tier1_pass_rate_points"],
            "tier1_pass_rate_trajectories": metrics["tier1_pass_rate_trajectories"],
            "tier1_mean_distance": metrics["tier1_mean_distance"],
            "tier1_mean_signed_margin": metrics["tier1_mean_signed_margin"],
            "tier0_points": metrics["tier0_points"],
            "tier0_pass_rate_points": metrics["tier0_pass_rate_points"],
            "tier0_pass_rate_trajectories": metrics["tier0_pass_rate_trajectories"],
            "tier0_mean_distance": metrics["tier0_mean_distance"],
            "tier0_mean_signed_margin": metrics["tier0_mean_signed_margin"],
        }

        self._save_results(results_row)
        self._save_pointwise_aggregates(
            distances_list=distances_list,
            error_ranges_list=error_ranges_list,
            model_name="difftraj",
            denoise_method="Baseline",
            K=None,
            Q1=None,
            Q2=None,
            t_delta=1.0,
            N_steps=1,
            test_timestamp=results_row["test_timestamp"],
        )

        return [results_row]

    def _compute_distances_and_ranges(
        self,
        denoised_trajectories: List,
        test_trajectories: List,
    ) -> tuple[List[np.ndarray], List[np.ndarray]]:
        distances_list = []
        error_ranges_list = []

        for denoised_gps, traj_obj in zip(denoised_trajectories, test_trajectories):
            ref_gps = traj_obj.ref_gps
            error_range = traj_obj.error_range

            T_denoised = len(denoised_gps)
            ref_gps_aligned = ref_gps[-T_denoised:]
            error_range_aligned = error_range[-T_denoised:]

            ref_lat = float(ref_gps_aligned[0, 1])
            ref_lon = float(ref_gps_aligned[0, 0])
            enu_denoised = self._gps_to_enu_batch(denoised_gps, ref_lat, ref_lon)
            enu_ref = self._gps_to_enu_batch(ref_gps_aligned, ref_lat, ref_lon)

            distances = np.linalg.norm(enu_denoised - enu_ref, axis=1)
            distances_list.append(distances)
            error_ranges_list.append(error_range_aligned)

        return distances_list, error_ranges_list

    @staticmethod
    def _tier_from_accuracy(acc: np.ndarray) -> np.ndarray:
        tier = np.full_like(acc, fill_value=-1, dtype=int)
        valid = ~np.isnan(acc)
        acc_valid = acc[valid]
        tier_vals = np.full_like(acc_valid, fill_value=4, dtype=int)
        tier_vals = np.where(acc_valid <= 30.0, 3, tier_vals)
        tier_vals = np.where(acc_valid <= 15.0, 2, tier_vals)
        tier_vals = np.where(acc_valid <= 10.0, 1, tier_vals)
        tier_vals = np.where(acc_valid <= 5.0, 0, tier_vals)
        tier[valid] = tier_vals
        return tier

    def _save_pointwise_aggregates(
        self,
        distances_list: List[np.ndarray],
        error_ranges_list: List[np.ndarray],
        model_name: str,
        denoise_method: str,
        K: Optional[int],
        Q1: Optional[int],
        Q2: Optional[int],
        t_delta: float,
        N_steps: int,
        test_timestamp: str,
    ) -> None:
        import pandas as pd

        if not distances_list:
            return

        def _safe_name(value: str) -> str:
            import re
            return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))

        safe_model = _safe_name(model_name)
        safe_method = _safe_name(denoise_method)
        ts_tag = test_timestamp.replace(":", "").replace("-", "").replace(".", "")

        # Trajectory-level pointwise averages (aligned to longest trajectory)
        max_len = max(len(d) for d in distances_list)
        pass_sum = np.zeros(max_len, dtype=float)
        dist_sum = np.zeros(max_len, dtype=float)
        margin_sum = np.zeros(max_len, dtype=float)
        acc_sum = np.zeros(max_len, dtype=float)
        count = np.zeros(max_len, dtype=int)
        dist_lists = [[] for _ in range(max_len)]
        margin_lists = [[] for _ in range(max_len)]
        acc_lists = [[] for _ in range(max_len)]

        for distances, acc in zip(distances_list, error_ranges_list):
            L = len(distances)
            if L == 0:
                continue
            signed_margin = distances - acc
            pass_flags = (signed_margin <= 0).astype(float)
            pass_sum[:L] += pass_flags
            dist_sum[:L] += distances
            margin_sum[:L] += signed_margin
            acc_sum[:L] += acc
            count[:L] += 1
            for i in range(L):
                dist_lists[i].append(float(distances[i]))
                margin_lists[i].append(float(signed_margin[i]))
                acc_lists[i].append(float(acc[i]))

        with np.errstate(divide="ignore", invalid="ignore"):
            pass_rate = np.divide(pass_sum, count, where=count > 0)
            mean_distance = np.divide(dist_sum, count, where=count > 0)
            mean_signed_margin = np.divide(margin_sum, count, where=count > 0)
            mean_accuracy = np.divide(acc_sum, count, where=count > 0)

        mean_accuracy[count == 0] = np.nan
        median_distance = np.array(
            [float(np.median(vals)) if vals else np.nan for vals in dist_lists]
        )
        median_signed_margin = np.array(
            [float(np.median(vals)) if vals else np.nan for vals in margin_lists]
        )
        median_accuracy = np.array(
            [float(np.median(vals)) if vals else np.nan for vals in acc_lists]
        )
        tier = self._tier_from_accuracy(mean_accuracy)

        traj_df = pd.DataFrame(
            {
                "position_index": np.arange(max_len, dtype=int),
                "pass_rate": pass_rate,
                "mean_signed_margin": mean_signed_margin,
                "mean_distance": mean_distance,
                "mean_accuracy": mean_accuracy,
                "median_signed_margin": median_signed_margin,
                "median_distance": median_distance,
                "median_accuracy": median_accuracy,
                "tier": tier,
                "aggregate_type": "trajectory_point_avg",
                "model_name": model_name,
                "denoise_method": denoise_method,
                "K": K,
                "Q1": Q1,
                "Q2": Q2,
                "t_delta": t_delta,
                "N_steps": N_steps,
                "test_timestamp": test_timestamp,
            }
        )

        traj_path = self.detail_dir / (
            f"{safe_model}_{safe_method}_K{K}_Q1{Q1}_Q2{Q2}_td{t_delta:.4f}_N{N_steps}_{ts_tag}_traj_point_avg.parquet"
        )
        traj_df.to_parquet(traj_path, index=False)

        # Chunk-level pointwise averages (aligned to chunk positions)
        if K is None or Q1 is None or Q2 is None:
            return

        Q1_points = Q1 * 8
        Q2_points = Q2 * 8
        stride = K - Q1_points - Q2_points
        if stride <= 0:
            self.logger.warning("Invalid stride for chunk aggregate: K=%s Q1=%s Q2=%s", K, Q1, Q2)
            return

        chunk_pass_sum = np.zeros(K, dtype=float)
        chunk_dist_sum = np.zeros(K, dtype=float)
        chunk_margin_sum = np.zeros(K, dtype=float)
        chunk_acc_sum = np.zeros(K, dtype=float)
        chunk_count = np.zeros(K, dtype=int)
        chunk_dist_lists = [[] for _ in range(K)]
        chunk_margin_lists = [[] for _ in range(K)]
        chunk_acc_lists = [[] for _ in range(K)]

        for distances, acc in zip(distances_list, error_ranges_list):
            N = len(distances)
            if N == 0:
                continue
            S = stride
            M = int(np.ceil(N / S))

            head = np.repeat(distances[0:1], Q1_points, axis=0) if Q1_points > 0 else np.zeros((0,))
            head_acc = np.repeat(acc[0:1], Q1_points, axis=0) if Q1_points > 0 else np.zeros((0,))
            payload_pad_len = M * S - N
            tail = np.repeat(distances[-1:], Q2_points, axis=0) if Q2_points > 0 else np.zeros((0,))
            tail_acc = np.repeat(acc[-1:], Q2_points, axis=0) if Q2_points > 0 else np.zeros((0,))
            pad = np.repeat(distances[-1:], payload_pad_len, axis=0) if payload_pad_len > 0 else np.zeros((0,))
            pad_acc = np.repeat(acc[-1:], payload_pad_len, axis=0) if payload_pad_len > 0 else np.zeros((0,))

            dist_padded = np.concatenate([head, distances, pad, tail], axis=0)
            acc_padded = np.concatenate([head_acc, acc, pad_acc, tail_acc], axis=0)

            for j in range(M):
                start = j * S
                end = start + K
                dist_chunk = dist_padded[start:end]
                acc_chunk = acc_padded[start:end]
                if len(dist_chunk) < K:
                    break
                signed_margin = dist_chunk - acc_chunk
                pass_flags = (signed_margin <= 0).astype(float)
                chunk_pass_sum += pass_flags
                chunk_dist_sum += dist_chunk
                chunk_margin_sum += signed_margin
                chunk_acc_sum += acc_chunk
                chunk_count += 1
                for i in range(K):
                    chunk_dist_lists[i].append(float(dist_chunk[i]))
                    chunk_margin_lists[i].append(float(signed_margin[i]))
                    chunk_acc_lists[i].append(float(acc_chunk[i]))

        with np.errstate(divide="ignore", invalid="ignore"):
            pass_rate = np.divide(chunk_pass_sum, chunk_count, where=chunk_count > 0)
            mean_distance = np.divide(chunk_dist_sum, chunk_count, where=chunk_count > 0)
            mean_signed_margin = np.divide(chunk_margin_sum, chunk_count, where=chunk_count > 0)
            mean_accuracy = np.divide(chunk_acc_sum, chunk_count, where=chunk_count > 0)

        mean_accuracy[chunk_count == 0] = np.nan
        median_distance = np.array(
            [float(np.median(vals)) if vals else np.nan for vals in chunk_dist_lists]
        )
        median_signed_margin = np.array(
            [float(np.median(vals)) if vals else np.nan for vals in chunk_margin_lists]
        )
        median_accuracy = np.array(
            [float(np.median(vals)) if vals else np.nan for vals in chunk_acc_lists]
        )
        tier = self._tier_from_accuracy(mean_accuracy)

        chunk_df = pd.DataFrame(
            {
                "position_index": np.arange(K, dtype=int),
                "pass_rate": pass_rate,
                "mean_signed_margin": mean_signed_margin,
                "mean_distance": mean_distance,
                "mean_accuracy": mean_accuracy,
                "median_signed_margin": median_signed_margin,
                "median_distance": median_distance,
                "median_accuracy": median_accuracy,
                "tier": tier,
                "aggregate_type": "chunk_point_avg",
                "model_name": model_name,
                "denoise_method": denoise_method,
                "K": K,
                "Q1": Q1,
                "Q2": Q2,
                "t_delta": t_delta,
                "N_steps": N_steps,
                "test_timestamp": test_timestamp,
            }
        )

        chunk_path = self.detail_dir / (
            f"{safe_model}_{safe_method}_K{K}_Q1{Q1}_Q2{Q2}_td{t_delta:.4f}_N{N_steps}_{ts_tag}_chunk_point_avg.parquet"
        )
        chunk_df.to_parquet(chunk_path, index=False)

    def _denoise_trajectories(
        self,
        checkpoint_path: str,
        test_trajectories: List,
        method: str,
        manual_config: Optional[Dict] = None,
    ) -> tuple:
        assert method in ["BF", "DF"], f"Invalid method: {method}"

        TrajectoryEvaluator._patch_encoder_decoder_checkpoint_loading()
        decoder = EncoderDecoder(checkpoint_path, manual_config=manual_config)

        denoised_trajectories = []
        for traj_obj in test_trajectories:
            noisy_gps = traj_obj.noisy_gps
            if method == "BF":
                denoised_gps = decoder.denoise_traj_BF(noisy_gps)
            else:
                denoised_gps = decoder.denoise_traj_DF(noisy_gps)
            denoised_trajectories.append(denoised_gps)

        return denoised_trajectories, decoder

    @staticmethod
    def _compute_sample_time_stats(test_trajectories: List) -> tuple[float, float, float]:
        deltas = []
        for traj_obj in test_trajectories:
            ts = getattr(traj_obj, "timestamps", None)
            if ts is None:
                continue
            if len(ts) < 2:
                continue
            diffs = np.diff(ts)
            diffs = diffs[diffs > 0]
            if diffs.size:
                deltas.append(diffs)

        if not deltas:
            return 0.0, 0.0, 0.0

        all_deltas = np.concatenate(deltas, axis=0)
        return float(all_deltas.mean()), float(np.median(all_deltas)), float(all_deltas.std())

    def _compute_pass_metrics(self, denoised_trajectories: List, test_trajectories: List) -> Dict[str, float]:
        distances_list = []
        error_ranges_list = []

        for denoised_gps, traj_obj in zip(denoised_trajectories, test_trajectories):
            ref_gps = traj_obj.ref_gps
            error_range = traj_obj.error_range

            T_denoised = len(denoised_gps)
            ref_gps_aligned = ref_gps[-T_denoised:]
            error_range_aligned = error_range[-T_denoised:]

            ref_lat = float(ref_gps_aligned[0, 1])
            ref_lon = float(ref_gps_aligned[0, 0])
            enu_denoised = self._gps_to_enu_batch(denoised_gps, ref_lat, ref_lon)
            enu_ref = self._gps_to_enu_batch(ref_gps_aligned, ref_lat, ref_lon)

            distances = np.linalg.norm(enu_denoised - enu_ref, axis=1)
            distances_list.append(distances)
            error_ranges_list.append(error_range_aligned)

        return self._compute_pass_metrics_from_distances(distances_list, error_ranges_list)

    def _compute_pass_metrics_from_distances(
        self,
        distances_list: List[np.ndarray],
        error_ranges_list: List[np.ndarray],
    ) -> Dict[str, float]:
        total_points = 0
        pass_points = 0
        outside_errors = []
        traj_pass_rates = []
        all_distances = []
        all_signed_margins = []

        tiers = [
            ("tier4", None),
            ("tier3", 30.0),
            ("tier2", 15.0),
            ("tier1", 10.0),
            ("tier0", 5.0),
        ]
        tier_point_totals = {name: 0 for name, _ in tiers}
        tier_point_pass = {name: 0 for name, _ in tiers}
        tier_traj_rates = {name: [] for name, _ in tiers}
        tier_distance_lists = {name: [] for name, _ in tiers}
        tier_signed_margin_lists = {name: [] for name, _ in tiers}

        for distances, error_range in zip(distances_list, error_ranges_list):
            if len(distances) == 0:
                continue

            signed_margin = distances - error_range
            pass_mask = signed_margin <= 0
            total_points += len(distances)
            pass_points += int(pass_mask.sum())
            traj_pass_rates.append(float(pass_mask.sum() / len(distances)))
            all_distances.extend(distances.tolist())
            all_signed_margins.extend(signed_margin.tolist())

            if (~pass_mask).any():
                outside_errors.extend((distances[~pass_mask] - error_range[~pass_mask]).tolist())

            for name, thr in tiers:
                if thr is None:
                    tier_mask = np.ones_like(error_range, dtype=bool)
                else:
                    tier_mask = error_range <= thr

                tier_points = int(tier_mask.sum())
                if tier_points == 0:
                    continue

                tier_pass = int((pass_mask & tier_mask).sum())
                tier_point_totals[name] += tier_points
                tier_point_pass[name] += tier_pass
                tier_traj_rates[name].append(float(tier_pass / tier_points))
                tier_distance_lists[name].extend(distances[tier_mask].tolist())
                tier_signed_margin_lists[name].extend(signed_margin[tier_mask].tolist())

        metrics = {
            "pass_rate_points": float(pass_points / total_points) if total_points > 0 else 0.0,
            "pass_rate_trajectories": float(np.mean(traj_pass_rates)) if traj_pass_rates else 0.0,
            "avg_outside_error": float(np.mean(outside_errors)) if outside_errors else 0.0,
            "mean_distance_all": float(np.mean(all_distances)) if all_distances else 0.0,
            "mean_signed_margin_all": float(np.mean(all_signed_margins)) if all_signed_margins else 0.0,
        }

        for name, _ in tiers:
            total = tier_point_totals[name]
            passed = tier_point_pass[name]
            traj_rates = tier_traj_rates[name]
            dist_list = tier_distance_lists[name]
            margin_list = tier_signed_margin_lists[name]
            metrics[f"{name}_points"] = int(total)
            metrics[f"{name}_pass_rate_points"] = float(passed / total) if total > 0 else 0.0
            metrics[f"{name}_pass_rate_trajectories"] = float(np.mean(traj_rates)) if traj_rates else 0.0
            metrics[f"{name}_mean_distance"] = float(np.mean(dist_list)) if dist_list else 0.0
            metrics[f"{name}_mean_signed_margin"] = float(np.mean(margin_list)) if margin_list else 0.0

        return metrics

    def _gps_to_enu_batch(self, gps_coords: np.ndarray, ref_lat: float, ref_lon: float) -> np.ndarray:
        import pymap3d as pm

        lons = gps_coords[:, 0]
        lats = gps_coords[:, 1]
        e, n, _ = pm.geodetic2enu(lats, lons, 0, ref_lat, ref_lon, 0)
        return np.stack([e, n], axis=1)

    def _save_results(self, results: Dict):
        csv_row = (
            f"{results['model_name']},{results['denoise_method']},"
            f"{results['K']},{results['Q1']},{results['Q2']},"
            f"{results['t_delta']:.4f},{results['N_steps']},"
            f"{results['pass_rate_points']:.6f},{results['pass_rate_trajectories']:.6f},"
            f"{results['avg_outside_error']:.6f},"
            f"{results['data_avg_sample_time_sec']:.6f},{results['data_median_sample_time_sec']:.6f},{results['data_std_sample_time_sec']:.6f},"
            f"{results['mean_distance_all']:.6f},{results['mean_signed_margin_all']:.6f},"
            f"{results['tier4_points']},{results['tier4_pass_rate_points']:.6f},{results['tier4_pass_rate_trajectories']:.6f},{results['tier4_mean_distance']:.6f},{results['tier4_mean_signed_margin']:.6f},"
            f"{results['tier3_points']},{results['tier3_pass_rate_points']:.6f},{results['tier3_pass_rate_trajectories']:.6f},{results['tier3_mean_distance']:.6f},{results['tier3_mean_signed_margin']:.6f},"
            f"{results['tier2_points']},{results['tier2_pass_rate_points']:.6f},{results['tier2_pass_rate_trajectories']:.6f},{results['tier2_mean_distance']:.6f},{results['tier2_mean_signed_margin']:.6f},"
            f"{results['tier1_points']},{results['tier1_pass_rate_points']:.6f},{results['tier1_pass_rate_trajectories']:.6f},{results['tier1_mean_distance']:.6f},{results['tier1_mean_signed_margin']:.6f},"
            f"{results['tier0_points']},{results['tier0_pass_rate_points']:.6f},{results['tier0_pass_rate_trajectories']:.6f},{results['tier0_mean_distance']:.6f},{results['tier0_mean_signed_margin']:.6f},"
            f"{results['num_tested_trajectories']},{results['num_tested_points']},{results['longest_trajectory_length']},"
            f"{results['test_timestamp']}\n"
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
