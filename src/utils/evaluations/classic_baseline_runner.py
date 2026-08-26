"""Classic and map-matching baseline trajectory evaluation helpers."""

from __future__ import annotations

import csv
import json
import logging
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import encoder_decoder
import numpy as np

from .benchmark_schema import split_baseline_spec
from .run_context import stage
from .trajectory import _RssMonitor


def _merge_diagnostic_counts(target: Counter[str], source: dict) -> dict:
    """
    Purpose:
        Merge one nonnegative diagnostic counter into an aggregate.
    Parameters:
        target (Counter[str]), mutable destination counter.
        source (dict), string-like keys and nonnegative integer counts.
    Return Dict:
        "error_code": int, 0 after a valid merge.
        "counter": Counter[str], updated destination.
    Usage:
        The trajectory runner aggregates Meili request diagnostics.
    TODO:
        1) Validate the source.
        2) Add every count.
        3) Return the destination.
    """

    # 1. Validate Source
    if not isinstance(source, dict):
        raise TypeError("Diagnostic counts must be a dict.")

    # 2. Add Every Count
    for key, value in source.items():
        count = int(value)
        if count < 0:
            raise ValueError("Diagnostic counts must be nonnegative.")
        target[str(key)] += count

    # 3. Return Destination
    return {"error_code": 0, "counter": target}


def _write_valhalla_diagnostics(output_dir: Path, summary: dict) -> dict:
    """
    Purpose:
        Persist machine-readable Valhalla rejection evidence for one evaluation.
    Parameters:
        output_dir (Path), child evaluation output directory.
        summary (dict), aggregate counts, rates, and sanitized request records.
    Return Dict:
        "error_code": int, 0 after all files are written.
        "summary_path": str, JSON summary path.
        "error_codes_path": str, error-code CSV path.
        "requests_path": str, sanitized request JSONL path.
    Usage:
        The trajectory runner writes proof even when all data is rejected.
    TODO:
        1) Create the diagnostics directory.
        2) Write the aggregate summary.
        3) Write normalized counter rows.
        4) Write sanitized request records.
    """

    # 1. Create Diagnostics Directory
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "valhalla_meili_summary.json"
    error_codes_path = output_dir / "valhalla_meili_error_codes.csv"
    requests_path = output_dir / "valhalla_meili_requests.jsonl"

    # 2. Write Aggregate Summary
    public_summary = dict(summary)
    request_records = list(public_summary.pop("request_records", []))
    summary_path.write_text(
        json.dumps(public_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # 3. Write Normalized Counter Rows
    counter_groups = {
        "http_status": summary["http_status_counts"],
        "valhalla_error_code": summary["valhalla_error_code_counts"],
        "adapter_error_code": summary["adapter_error_code_counts"],
        "transport_error": summary["transport_error_counts"],
        "matched_point_type": summary["point_type_counts"],
    }
    with error_codes_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["category", "code", "count"])
        writer.writeheader()
        for category, counts in counter_groups.items():
            for code, count in sorted(counts.items()):
                writer.writerow({"category": category, "code": code, "count": int(count)})

    # 4. Write Sanitized Request Records
    with requests_path.open("w", encoding="utf-8") as file_obj:
        for record in request_records:
            file_obj.write(json.dumps(record, sort_keys=True) + "\n")
    return {
        "error_code": 0,
        "summary_path": str(summary_path),
        "error_codes_path": str(error_codes_path),
        "requests_path": str(requests_path),
    }


def run_classic_baselines_filtered(
    manager,
    test_trajectories: list,
    dataset_name: str,
    methods: list[str],
    dataset_name_hint: str | None = None,
    baseline_config: dict | None = None,
    diagnostics_output_dir: str | Path | None = None,
) -> list[dict]:
    """
    Purpose:
        Evaluate selected baselines with raw fallback for rejected Meili points.
    Parameters:
        manager, evaluation manager exposing trajectory_evaluator.
        test_trajectories (list), ordered trajectory objects.
        dataset_name (str), result label.
        methods (list[str]), selected baseline specifications.
        dataset_name_hint (str | None), processor dataset identifier.
        baseline_config (dict | None), required for valhalla_meili.
        diagnostics_output_dir (str | Path | None), Meili evidence directory.
    Return Dict:
        Existing interface returns list[dict], one result row per baseline.
    Usage:
        Trajectory batch children and ClassicBaselineEvaluator call this helper.
    TODO:
        1) Validate methods and prepare inputs.
        2) Initialize each selected model.
        3) Predict every trajectory through acceptance packets.
        4) Score complete Meili-plus-raw-fallback trajectories.
        5) Persist strict Meili rejection evidence and result rows.
    """

    # 1. Validate Methods And Prepare Constants
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
        "valhalla_meili": None,
    }
    selected: list[tuple[str, str, str | None]] = []
    for spec in methods:
        base_name, kalman_mode, display_name = split_baseline_spec(spec)
        if base_name not in method_table:
            raise ValueError(f"Unknown baseline specification: {spec!r}")
        selected.append((display_name, base_name, kalman_mode))
    if not selected:
        stage("Classic baseline list is empty; skipping classic baselines.")
        return []

    prepared_inputs: list[tuple] = []
    attempted_points = 0
    for trajectory_index, traj_obj in enumerate(test_trajectories):
        noisy_gps = encoder_decoder.remove_nan_rows(np.asarray(traj_obj.noisy_gps, dtype=float))
        if noisy_gps.size == 0:
            continue
        clean_gps = np.asarray(traj_obj.clean_gps, dtype=float)[-len(noisy_gps):]
        ref_lat = float(clean_gps[0, 1])
        ref_lon = float(clean_gps[0, 0])
        enu_clean = manager.trajectory_evaluator._gps_to_enu_batch(
            clean_gps,
            ref_lat,
            ref_lon,
        )
        seq = build_lat_lon_timestamp_sequence_from_lonlat(
            noisy_gps,
            timestamps=getattr(traj_obj, "timestamps", None),
        )
        prepared_inputs.append((trajectory_index, traj_obj, seq, ref_lat, ref_lon, enu_clean))
        attempted_points += int(len(seq))

    # 2. Initialize Each Selected Model
    results: list[dict] = []
    for display_name, base_name, kalman_mode in selected:
        report_name = (
            "valhalla_meili_raw_fallback"
            if base_name == "valhalla_meili"
            else display_name
        )
        logging.info("Running classic baseline: %s", report_name)
        baseline_dataset_name = str(dataset_name_hint or dataset_name or "").strip()
        model = create_baseline_model(
            method_name=base_name,
            dataset_name=baseline_dataset_name,
            kalman_calibration_mode=(kalman_mode if base_name == "kalman_rts" else None),
            baseline_config=baseline_config if base_name == "valhalla_meili" else None,
        )
        try:
            all_errors: list[np.ndarray] = []
            all_l1_errors: list[np.ndarray] = []
            scored_trajectories: list = []
            accepted_points = 0
            accepted_trajectory_count = 0
            partial_trajectory_count = 0
            rejected_trajectory_count = 0
            attempted_requests = 0
            accepted_requests = 0
            rejected_requests = 0
            http_status_counts: Counter[str] = Counter()
            valhalla_error_counts: Counter[str] = Counter()
            adapter_error_counts: Counter[str] = Counter()
            transport_error_counts: Counter[str] = Counter()
            point_type_counts: Counter[str] = Counter()
            request_records: list[dict] = []

            # 3. Predict Every Trajectory Through Acceptance Packets
            resource_pids = (
                model.resource_usage_roots()["pids"]
                if base_name == "valhalla_meili"
                else []
            )
            with _RssMonitor(extra_root_pids=resource_pids) as rss_monitor:
                predict_start = time.perf_counter()
                for trajectory_index, traj_obj, seq, ref_lat, ref_lon, enu_clean in prepared_inputs:
                    packet = model.predict_packet(seq)
                    rss_monitor.sample()
                    mask = np.asarray(packet["accepted_mask"], dtype=bool)
                    if mask.shape != (len(seq),):
                        raise ValueError("Baseline acceptance mask is not point-aligned.")
                    accepted_points += int(np.count_nonzero(mask))

                    if base_name == "valhalla_meili":
                        diagnostics = packet["diagnostics"]
                        attempted_requests += int(diagnostics["attempted_requests"])
                        accepted_requests += int(diagnostics["accepted_requests"])
                        rejected_requests += int(diagnostics["rejected_requests"])
                        _merge_diagnostic_counts(http_status_counts, diagnostics["http_status_counts"])
                        _merge_diagnostic_counts(
                            valhalla_error_counts,
                            diagnostics["valhalla_error_code_counts"],
                        )
                        _merge_diagnostic_counts(
                            adapter_error_counts,
                            diagnostics["adapter_error_code_counts"],
                        )
                        _merge_diagnostic_counts(
                            transport_error_counts,
                            diagnostics["transport_error_counts"],
                        )
                        _merge_diagnostic_counts(point_type_counts, diagnostics["point_type_counts"])
                        for record in diagnostics["request_records"]:
                            request_records.append(
                                {"trajectory_index": int(trajectory_index), **dict(record)}
                            )

                    if bool(packet["complete"]):
                        accepted_trajectory_count += 1
                    elif bool(np.any(mask)):
                        partial_trajectory_count += 1
                        if base_name != "valhalla_meili":
                            continue
                    else:
                        rejected_trajectory_count += 1
                        if base_name != "valhalla_meili":
                            continue

                    # 4. Score Complete Meili-Plus-Raw-Fallback Trajectories
                    denoised_gps = latlon_to_lonlat(packet["positions_latlon"])
                    denoised_enu = manager.trajectory_evaluator._gps_to_enu_batch(
                        denoised_gps,
                        ref_lat,
                        ref_lon,
                    )
                    diff = denoised_enu - enu_clean
                    all_errors.append(np.linalg.norm(diff, axis=1))
                    all_l1_errors.append(np.abs(diff[:, 0]) + np.abs(diff[:, 1]))
                    scored_trajectories.append(traj_obj)

                prediction_time_sec = max(time.perf_counter() - predict_start, 0.0)
                rss_telemetry = rss_monitor.telemetry()

            errors = np.concatenate(all_errors, axis=0) if all_errors else np.asarray([])
            l1_errors = np.concatenate(all_l1_errors, axis=0) if all_l1_errors else np.asarray([])
            if errors.size:
                pw_metrics = manager.trajectory_evaluator._compute_pointwise_metrics(errors)
                l1_metrics = manager.trajectory_evaluator._compute_pointwise_metrics(l1_errors)
                pw_profile = manager.trajectory_evaluator._compute_trajectory_pointwise_profile(
                    scored_trajectories,
                    errors,
                )
                bw_metrics = manager.trajectory_evaluator._compute_bytewise_metrics(
                    scored_trajectories,
                    errors,
                )
                cw_metrics = manager.trajectory_evaluator._compute_chunkwise_metrics(
                    scored_trajectories,
                    errors,
                    baseline_k,
                    baseline_q1,
                    baseline_q2,
                )
            else:
                pw_metrics = {"avg": None, "med": None, "p95": None, "std": None}
                l1_metrics = {"avg": None, "med": None, "p95": None, "std": None}
                pw_profile = {"avg_list": [], "avg_list_norm": []}
                bw_metrics = {"avg_list": [], "avg_list_norm": []}
                cw_metrics = {"avg_list": [], "avg_list_norm": []}

            # 5. Persist Meili Evidence And Result Row
            attempted_trajectory_count = len(prepared_inputs)
            rejected_point_count = attempted_points - accepted_points
            rejected_trajectory_total = partial_trajectory_count + rejected_trajectory_count
            summary = {
                "dataset_name": dataset_name,
                "method": report_name,
                "attempted_trajectories": attempted_trajectory_count,
                "accepted_trajectories": accepted_trajectory_count,
                "partial_trajectories": partial_trajectory_count,
                "rejected_trajectories": rejected_trajectory_count,
                "trajectory_rejection_rate": (
                    float(rejected_trajectory_total) / float(attempted_trajectory_count)
                    if attempted_trajectory_count else 0.0
                ),
                "attempted_points": attempted_points,
                "accepted_points": accepted_points,
                "rejected_points": rejected_point_count,
                "point_rejection_rate": (
                    float(rejected_point_count) / float(attempted_points)
                    if attempted_points else 0.0
                ),
                "attempted_requests": attempted_requests,
                "accepted_requests": accepted_requests,
                "rejected_requests": rejected_requests,
                "request_rejection_rate": (
                    float(rejected_requests) / float(attempted_requests)
                    if attempted_requests else 0.0
                ),
                "http_status_counts": dict(sorted(http_status_counts.items())),
                "valhalla_error_code_counts": dict(sorted(valhalla_error_counts.items())),
                "adapter_error_code_counts": dict(sorted(adapter_error_counts.items())),
                "transport_error_counts": dict(sorted(transport_error_counts.items())),
                "point_type_counts": dict(sorted(point_type_counts.items())),
                "request_records": request_records,
            }
            if base_name == "valhalla_meili":
                summary["fallback_policy"] = "raw_input"
                summary["fallback_points"] = rejected_point_count
                summary["scored_trajectories"] = attempted_trajectory_count
                summary["scored_points"] = attempted_points
                if diagnostics_output_dir is None:
                    raise ValueError("valhalla_meili requires diagnostics_output_dir.")
                _write_valhalla_diagnostics(Path(diagnostics_output_dir), summary)

            result = {
                "model_name": report_name,
                "model_tag": "Baseline",
                "device": "cpu",
                "dataset_name": dataset_name,
                "model_dir": None,
                "checkpoint_name": None,
                "K": None,
                "Q1": None,
                "Q2": None,
                "test_timestamp": datetime.now().isoformat(),
                "num_tested_trajectories": attempted_trajectory_count,
                "num_tested_points": attempted_points,
                "prediction_time_sec": prediction_time_sec,
                "points_per_sec": (
                    float(attempted_points) / prediction_time_sec
                    if prediction_time_sec > 0.0 else 0.0
                ),
                **rss_telemetry,
                "longest_trajectory_length": max((len(item[2]) for item in prepared_inputs), default=0),
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
                **{key: value for key, value in summary.items() if key != "request_records"},
            }
            result["traj_p_val_rows"] = (
                manager.trajectory_evaluator._build_traj_p_val_rows_from_lists(
                    scored_trajectories,
                    [np.asarray(item, dtype=float) for item in all_errors],
                    [np.asarray(item, dtype=float) for item in all_l1_errors],
                    result,
                ) if all_errors else []
            )
            manager.trajectory_evaluator._save_results(result)
            results.append(result)
        finally:
            model.deconst()
    return results
