"""Classic baseline benchmark runner helpers.

Purpose:
    Keep classic baseline evaluation loops out of run_benchmarks so the entry
    point only dispatches phases.

Logic Chain:
    1. Build one initialized classic baseline model.
    2. Evaluate all test trajectories.
    3. Save one baseline result row through the trajectory evaluator.
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import encoder_decoder

from .benchmark_schema import split_baseline_spec
from .run_context import stage


def run_classic_baselines_filtered(
    manager,
    test_trajectories: list,
    dataset_name: str,
    methods: list[str],
    dataset_name_hint: str | None = None,
) -> None:
    """Run selected classic baselines and save trajectory-eval result rows."""
    baseline_k = 256
    baseline_q1 = 1
    baseline_q2 = 12

    from baseline import classic as classic_baseline
    from baseline import (
        build_lat_lon_timestamp_sequence_from_lonlat,
        create_baseline_model,
        latlon_to_lonlat,
    )

    method_table = {
        "alpha_beta": classic_baseline.alpha_beta_filter,
        "causal_hampel": classic_baseline.causal_hampel_filter,
        "kalman_filter": classic_baseline.kalman_filter,
        "kalman_rts": classic_baseline.kalman_rts_smoother,
        "hampel": classic_baseline.hampel_filter,
        "savgol": classic_baseline.savitzky_golay_filter,
        "raw": classic_baseline.raw_baseline,
    }

    selected: list[tuple[str, str, str | None]] = []
    for spec in methods:
        base_name, kalman_mode, display_name = split_baseline_spec(spec)
        if base_name not in method_table:
            logging.warning("Classic baseline %s ignored (unknown base=%s)", spec, base_name)
            continue
        selected.append((display_name, base_name, kalman_mode))
    if not selected:
        stage("Classic baseline list is empty; skipping classic baselines.")
        return

    for display_name, base_name, kalman_mode in selected:
        logging.info("Running classic baseline: %s", display_name)
        model = None
        baseline_dataset_name = str(dataset_name_hint or dataset_name or "").strip() or dataset_name

        try:
            model = create_baseline_model(
                method_name=base_name,
                dataset_name=baseline_dataset_name,
                kalman_calibration_mode=(
                    kalman_mode if base_name == "kalman_rts" else None
                ),
            )
        except Exception as exc:
            logging.warning("Classic baseline %s initialization failed: %s", display_name, exc)
            continue

        try:
            all_errors = []
            all_l1_errors = []
            last_point_l2_errors = []
            last_point_l1_errors = []

            for traj_obj in test_trajectories:
                noisy_gps = encoder_decoder.remove_nan_rows(np.asarray(traj_obj.noisy_gps, dtype=float))
                if noisy_gps.size == 0:
                    continue
                clean_gps = np.asarray(traj_obj.clean_gps, dtype=float)[-len(noisy_gps):]

                ref_lat = float(clean_gps[0, 1])
                ref_lon = float(clean_gps[0, 0])
                enu_clean = manager.trajectory_evaluator._gps_to_enu_batch(clean_gps, ref_lat, ref_lon)

                try:
                    timestamps = getattr(traj_obj, "timestamps", None)
                    seq = build_lat_lon_timestamp_sequence_from_lonlat(noisy_gps, timestamps=timestamps)
                    denoised_latlon = model.predict(seq)
                    denoised_gps = latlon_to_lonlat(denoised_latlon)
                    denoised_enu = manager.trajectory_evaluator._gps_to_enu_batch(
                        denoised_gps,
                        ref_lat,
                        ref_lon,
                    )
                except Exception as exc:
                    logging.warning("Classic baseline %s skipped: %s", display_name, exc)
                    all_errors = []
                    all_l1_errors = []
                    break

                diff = denoised_enu - enu_clean
                l2_errors = np.linalg.norm(diff, axis=1)
                l1_errors = np.abs(diff[:, 0]) + np.abs(diff[:, 1])
                all_errors.append(l2_errors)
                all_l1_errors.append(l1_errors)
                last_point_l2_errors.append(float(l2_errors[-1]))
                last_point_l1_errors.append(float(l1_errors[-1]))

            if not all_errors:
                continue

            errors = np.concatenate(all_errors, axis=0)
            l1_errors = np.concatenate(all_l1_errors, axis=0)
            pw_metrics = manager.trajectory_evaluator._compute_pointwise_metrics(errors)
            l1_metrics = manager.trajectory_evaluator._compute_pointwise_metrics(l1_errors)
            pw_profile = manager.trajectory_evaluator._compute_trajectory_pointwise_profile(
                test_trajectories,
                errors,
            )
            bw_metrics = manager.trajectory_evaluator._compute_bytewise_metrics(
                test_trajectories,
                errors,
            )
            cw_metrics = manager.trajectory_evaluator._compute_chunkwise_metrics(
                test_trajectories,
                errors,
                baseline_k,
                baseline_q1,
                baseline_q2,
            )

            result = {
                "model_name": display_name,
                "model_tag": "Baseline",
                "device": "cpu",
                "dataset_name": dataset_name,
                "model_dir": None,
                "checkpoint_name": None,
                "K": None,
                "Q1": None,
                "Q2": None,
                "test_timestamp": datetime.now().isoformat(),
                "num_tested_trajectories": len(test_trajectories),
                "num_tested_points": int(sum(manager.trajectory_evaluator._effective_traj_length(traj) for traj in test_trajectories)),
                "longest_trajectory_length": int(max(manager.trajectory_evaluator._effective_traj_length(traj) for traj in test_trajectories)),
                "avg_l2_err_pw": pw_metrics["avg"],
                "med_l2_err_pw": pw_metrics["med"],
                "p95_l2_err_pw": pw_metrics["p95"],
                "std_l2_err_pw": pw_metrics["std"],
                "avg_l1_err_pw": l1_metrics["avg"],
                "med_l1_err_pw": l1_metrics["med"],
                "p95_l1_err_pw": l1_metrics["p95"],
                "std_l1_err_pw": l1_metrics["std"],
                "avg_l1_err_tail": None,
                "avg_l2_err_tail": None,
                "avg_l2_err_pw_profile": pw_profile["avg_list"],
                "avg_l2_err_pw_profile_norm": pw_profile["avg_list_norm"],
                "avg_l2_err_bw": bw_metrics["avg_list"],
                "avg_l2_err_bw_norm": bw_metrics["avg_list_norm"],
                "avg_l2_err_cw": cw_metrics["avg_list"],
                "avg_l2_err_cw_norm": cw_metrics["avg_list_norm"],
            }
            result["traj_p_val_rows"] = manager.trajectory_evaluator._build_traj_p_val_rows_from_lists(
                test_trajectories,
                [np.asarray(x, dtype=float) for x in all_errors],
                [np.asarray(x, dtype=float) for x in all_l1_errors],
                result,
            )
            manager.trajectory_evaluator._save_results(result)
        finally:
            if model is not None:
                model.deconst()
