#!/usr/bin/env python3
"""Entry point for running evaluations using eval_joblist.json."""

import argparse
import csv
import time
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
import torch
import numpy as np

from utils.evaluations.evaluation_manager import TestManager
from utils.evaluations.wandb_logger import log_run_to_wandb
from utils.data_processor.traj_extractor import traj_extractor
from utils.data_processor.extract_10min_traj import extract_10min_traj


FULL_TRAJ_DIR = Path("./dataset/processed/full_traj")
DEBUG_FULL_TRAJ = FULL_TRAJ_DIR / "fulltraj_debug_mini.pt"

DEFAULT_TIME_NPY = Path("./dataset/time_test/source_list.npy")
DEFAULT_TIME_LOG = Path("./bin/log/time_test.csv")
DEFAULT_CLASSIC_BASELINES = [
    "kalman_rts_ts",
    "kalman_rts_notime",
    "hampel",
    "savgol",
    "raw",
]


def _stage(message: str) -> None:
    print(f"[run_benchmarks] {message}", flush=True)


def _latest_pt_file(directory: Path) -> Path:
    candidates = sorted(directory.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No .pt files found under {directory}")
    return candidates[0]


def _load_metadata(pt_path: Path) -> dict:
    data = torch.load(pt_path, map_location="cpu")
    meta = data.get("metadata", {})
    if not meta:
        raise ValueError(f"Missing metadata in {pt_path}")
    return meta


def _resolve_dataset(path_or_dir: Path, debug_path: Path | None, use_debug: bool) -> tuple[Path, int, int]:
    if use_debug:
        if debug_path is None:
            raise ValueError("Debug path not provided")
        pt_path = debug_path
    elif path_or_dir.is_file():
        pt_path = path_or_dir
    else:
        pt_path = _latest_pt_file(path_or_dir)

    meta = _load_metadata(pt_path)
    m = int(meta.get("n_trajectories", 0) or 0)
    n = int(meta.get("median_length", 0) or 0)
    if m <= 0 or n <= 0:
        raise ValueError(f"Invalid metadata in {pt_path}: n_trajectories={m}, median_length={n}")
    return pt_path, m, n


def _build_job_list(job: dict) -> dict:
    job_list = {
        "Q1": job.get("Q1"),
        "Q2": job.get("Q2"),
        "t_delta": job.get("t_delta"),
        "methods": list(job.get("methods", ["BF", "DF"])),
    }
    if not job_list["Q1"] or not job_list["Q2"] or not job_list["t_delta"]:
        raise ValueError("eval_joblist.json must include non-empty Q1, Q2, and t_delta lists.")
    return job_list


def _resolve_classic_baselines(job: dict) -> list[str]:
    raw = job.get("classic_baselines")
    if raw is None:
        return list(DEFAULT_CLASSIC_BASELINES)
    if isinstance(raw, str):
        requested = [x.strip() for x in raw.split(",") if x.strip()]
    elif isinstance(raw, list):
        requested = [str(x).strip() for x in raw if str(x).strip()]
    else:
        raise ValueError("classic_baselines must be a list of names or comma-separated string.")

    normalized = [("raw" if name == "spline" else name) for name in requested]
    allowed = set(DEFAULT_CLASSIC_BASELINES)
    selected = [name for name in normalized if name in allowed]
    unknown = [name for name in normalized if name not in allowed]
    for name in unknown:
        logging.warning("Unknown classic baseline ignored: %s", name)
    return selected


def _append_time_log_traj(
    logfile: Path,
    model_name: str,
    model_tag: str,
    dataset_name: str,
    denoise_method: str,
    avg_time_sec: float,
    avg_time_sec_per_point: float | None,
    num_points: int,
) -> None:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    file_exists = logfile.exists()
    with logfile.open("a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "model_name",
                "model_tag",
                "dataset_name",
                "denoise_method",
                "avg_time_sec",
                "avg_time_sec_per_point",
                "num_points",
                "test_timestamp",
            ])
        writer.writerow([
            model_name,
            model_tag,
            dataset_name,
            denoise_method,
            f"{avg_time_sec:.6f}",
            f"{avg_time_sec_per_point:.8f}" if avg_time_sec_per_point is not None else "NA",
            num_points,
            datetime.now().isoformat(),
        ])


def _parse_model_name(model_name: str) -> tuple[str, str]:
    parts = model_name.split("_")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return model_name, "NA"


def _generate_full_traj(output_dir: Path, use_new_traj: dict) -> Path:
    cfg = use_new_traj.get("full_traj", {}) if use_new_traj else {}
    if "M" not in cfg or "N" not in cfg:
        raise ValueError("use_new_traj.full_traj must include M and N when gen_new_test is true.")
    result = traj_extractor(
        parquet_dir="./dataset/raw",
        M=int(cfg["M"]),
        N=int(cfg["N"]),
        output_dir=str(output_dir),
    )
    return Path(result["output_file"])


def _resolve_existing_dataset(
    name: str,
    path_value: str | None,
    debug_path: Path | None,
    use_debug: bool,
) -> tuple[Path, int, int] | None:
    if use_debug:
        if debug_path is None:
            raise ValueError(f"Debug path not provided for {name}")
        pt_path = debug_path
        meta = _load_metadata(pt_path)
        return pt_path, int(meta["n_trajectories"]), int(meta["median_length"])

    if path_value is None or str(path_value).strip() == "":
        return None

    path = Path(path_value)
    if not path.exists():
        logging.error("Dataset path for %s not found: %s", name, path)
        return None

    pt_path, m, n = _resolve_dataset(path, None, use_debug=False)
    return pt_path, m, n


def _run_trajectory_eval(
    manager: TestManager,
    model_root: str,
    model_names: list | None,
    methods: list,
    dataset_entries: list[dict],
) -> None:
    for entry in dataset_entries:
        manager.run_trajectory_evaluation(
            model_names=model_names,
            denoise_methods=methods,
            model_root=model_root,
            test_data_path=str(entry["path"]),
            M=int(entry["M"]),
            D=None,
            N=int(entry["N"]),
            run_baselines=False,
        )


def _run_grid_eval(
    manager: TestManager,
    job_list: dict,
    model_root: str,
    model_names: list | None,
    dataset_entries: list[dict],
) -> None:
    for entry in dataset_entries:
        _stage(
            f"Grid eval start | dataset={entry['name']} M={int(entry['M'])} N={int(entry['N'])}"
        )
        manager.run_grid_search_evaluation(
            job_list=job_list,
            model_names=model_names,
            model_root=model_root,
            test_data_path=str(entry["path"]),
            M=int(entry["M"]),
            D=None,
            N=int(entry["N"]),
            run_baselines=False,
        )


def _run_bounded_eval(manager: TestManager, job: dict, model_root: str, model_names: list | None, methods: list) -> None:
    manager.run_uncertainty_band_test(
        model_names=model_names,
        denoise_methods=methods,
        model_root=model_root,
        test_data_path=job.get("test_data_path", "./dataset/processed/full_traj_range"),
        M=int(job.get("M", 200)),
        N=int(job.get("N", 10000)),
    )


def _run_difftraj_baseline_trajectory(
    manager: TestManager,
    test_trajectories: list,
    dataset_name: str,
    job: dict,
) -> None:
    try:
        from baseline.difftraj import DiffTrajPaths, difftraj_denoise_with_model, prepare_difftraj
    except Exception as exc:
        logging.warning("DiffTraj import failed; skipping trajectory baseline: %s", exc)
        return

    repo_dir = job.get("difftraj_repo", "./bin/baseline_model/difftraj")
    checkpoint_path = job.get("difftraj_ckpt", "")
    device = job.get("difftraj_device", "cuda")
    timesteps = int(job.get("difftraj_timesteps", 100))
    final_steps = job.get("difftraj_final_steps")
    final_steps = int(final_steps) if final_steps is not None else None
    eta = float(job.get("difftraj_eta", 0.0))

    paths = DiffTrajPaths(
        repo_dir=repo_dir or None,
        checkpoint_path=checkpoint_path or None,
    )

    try:
        config, model, device = prepare_difftraj(paths, device=device)
    except Exception as exc:
        logging.warning("DiffTraj unavailable; skipping trajectory baseline: %s", exc)
        return

    target_len = int(getattr(getattr(config, "data", None), "traj_length", 0) or 0)
    logging.warning(
        "DiffTraj trajectory baseline started (traj=%d, timesteps=%d, device=%s, target_len=%d).",
        len(test_trajectories),
        timesteps,
        device,
        target_len,
    )

    def _pad_tail(values: np.ndarray, tgt_len: int) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.shape[0] == tgt_len:
            return values
        if values.shape[0] > tgt_len:
            return values[:tgt_len]
        pad_count = tgt_len - values.shape[0]
        tail = np.repeat(values[-1:], pad_count, axis=0)
        return np.concatenate([values, tail], axis=0)

    def _denoise_chunked(enu_noisy: np.ndarray, tgt_len: int) -> np.ndarray:
        total_len = enu_noisy.shape[0]
        if total_len <= tgt_len:
            chunk = _pad_tail(enu_noisy, tgt_len)
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

        outputs = []
        idx = 0
        while idx < total_len:
            chunk = _pad_tail(enu_noisy[idx: idx + tgt_len], tgt_len)
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
            idx += tgt_len

        stitched = np.concatenate(outputs, axis=0)
        return stitched[:total_len]

    per_traj_errors = []
    total_traj = len(test_trajectories)
    for traj_idx, traj_obj in enumerate(test_trajectories, start=1):
        noisy_gps = traj_obj.noisy_gps
        clean_gps = traj_obj.clean_gps
        if len(noisy_gps) == 0:
            continue

        ref_lat = float(clean_gps[0, 1])
        ref_lon = float(clean_gps[0, 0])
        enu_noisy = manager.trajectory_evaluator._gps_to_enu_batch(noisy_gps, ref_lat, ref_lon)
        enu_clean = manager.trajectory_evaluator._gps_to_enu_batch(clean_gps, ref_lat, ref_lon)

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
            logging.warning("DiffTraj failed on one trajectory (%s): %s", type(exc).__name__, exc)
            continue

        T = min(len(denoised_enu), len(enu_clean))
        if T <= 0:
            continue
        errors = np.linalg.norm(denoised_enu[:T] - enu_clean[:T], axis=1)
        per_traj_errors.append(errors)
        if traj_idx == 1 or traj_idx % 10 == 0 or traj_idx == total_traj:
            logging.warning("DiffTraj trajectory baseline progress: %d/%d", traj_idx, total_traj)

    if not per_traj_errors:
        logging.warning("No valid DiffTraj outputs; skipping trajectory baseline logging.")
        return

    all_errors = np.concatenate(per_traj_errors, axis=0)
    pw_metrics = manager.trajectory_evaluator._compute_pointwise_metrics(all_errors)

    longest_traj = max(test_trajectories, key=lambda t: len(t.noisy_gps))
    ref_lat = float(longest_traj.clean_gps[0, 1])
    ref_lon = float(longest_traj.clean_gps[0, 0])
    enu_noisy = manager.trajectory_evaluator._gps_to_enu_batch(longest_traj.noisy_gps, ref_lat, ref_lon)

    times = []
    for _ in range(5):
        t0 = time.time()
        if target_len > 0 and enu_noisy.shape[0] != target_len:
            _ = _denoise_chunked(enu_noisy, target_len)
        else:
            _ = difftraj_denoise_with_model(
                enu_noisy,
                config=config,
                model=model,
                device=device,
                timesteps=timesteps,
                final_steps=final_steps,
                eta=eta,
            )
        t1 = time.time()
        times.append(t1 - t0)
    avg_time = float(np.mean(times)) if times else None
    avg_time_per_point = avg_time / len(longest_traj.noisy_gps) if avg_time and len(longest_traj.noisy_gps) else None

    result = {
        "model_name": "difftraj",
        "model_tag": "Baseline",
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
        "num_tested_trajectories": len(per_traj_errors),
        "num_tested_points": int(sum(len(e) for e in per_traj_errors)),
        "longest_trajectory_length": int(max(len(e) for e in per_traj_errors)),
        "avg_l2_err_pw": pw_metrics["avg"],
        "med_l2_err_pw": pw_metrics["med"],
        "std_l2_err_pw": pw_metrics["std"],
        "avg_l2_err_bw": [],
        "avg_l2_err_bw_norm": [],
        "avg_l2_err_cw": [],
        "avg_l2_err_cw_norm": [],
        "avg_denoise_time_sec": avg_time,
        "avg_denoise_time_sec_per_point": avg_time_per_point,
    }
    manager.trajectory_evaluator._save_results(result)
    logging.warning("DiffTraj trajectory baseline saved.")


def _run_classic_baselines_filtered_compat(
    manager: TestManager,
    test_trajectories: list,
    dataset_name: str,
    methods: list[str],
) -> None:
    from baseline import classic as classic_baseline

    method_table = [
        ("kalman_rts_ts", classic_baseline.kalman_rts_smoother),
        ("kalman_rts_notime", classic_baseline.kalman_rts_smoother),
        ("hampel", classic_baseline.hampel_filter),
        ("savgol", classic_baseline.savitzky_golay_filter),
        ("raw", classic_baseline.raw_baseline),
    ]
    selected = [(n, f) for n, f in method_table if n in set(methods)]
    if not selected:
        _stage("Classic baseline list is empty; skipping classic baselines.")
        return

    longest_traj = max(test_trajectories, key=lambda t: len(t.noisy_gps))
    ref_lat = float(longest_traj.clean_gps[0, 1])
    ref_lon = float(longest_traj.clean_gps[0, 0])
    enu_longest = manager.trajectory_evaluator._gps_to_enu_batch(longest_traj.noisy_gps, ref_lat, ref_lon)
    longest_points = len(longest_traj.noisy_gps)

    for method_name, method_fn in selected:
        logging.info("Running classic baseline (compat): %s", method_name)
        all_errors = []

        for traj_obj in test_trajectories:
            noisy_gps = traj_obj.noisy_gps
            clean_gps = traj_obj.clean_gps

            ref_lat = float(clean_gps[0, 1])
            ref_lon = float(clean_gps[0, 0])
            enu_noisy = manager.trajectory_evaluator._gps_to_enu_batch(noisy_gps, ref_lat, ref_lon)
            enu_clean = manager.trajectory_evaluator._gps_to_enu_batch(clean_gps, ref_lat, ref_lon)

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
            continue

        errors = np.concatenate(all_errors, axis=0)
        pw_metrics = manager.trajectory_evaluator._compute_pointwise_metrics(errors)

        times = []
        for _ in range(5):
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
            "std_l2_err_pw": pw_metrics["std"],
            "avg_l2_err_bw": [],
            "avg_l2_err_bw_norm": [],
            "avg_l2_err_cw": [],
            "avg_l2_err_cw_norm": [],
            "avg_denoise_time_sec": avg_time,
            "avg_denoise_time_sec_per_point": avg_time_per_point,
        }
        manager.trajectory_evaluator._save_results(result)


def _run_time_tests(
    manager: TestManager,
    job: dict,
    job_list: dict,
    model_root: str,
    model_names: list | None,
    dataset_entries: list[dict],
    methods: list,
    classic_baselines: list[str],
    include_difftraj_timing: bool = False,
) -> None:
    if not job.get("time_test", True):
        logging.info("Skipping time test (time_test=false).")
        return
    default_time_log = Path(manager.output_dir) / "time_test.csv"
    time_log = Path(job.get("time_log_path", str(default_time_log)))

    dataset_entry = next((d for d in dataset_entries if d.get("name") == "full_traj"), None)
    if dataset_entry is None:
        dataset_entry = dataset_entries[0] if dataset_entries else None
    if dataset_entry is None:
        logging.warning("No datasets available for time test. Skipping.")
        return

    test_trajectories, dataset_name = manager._load_or_generate_test_data(
        test_data_path=str(dataset_entry["path"]),
        M=int(dataset_entry["M"]),
        N=int(dataset_entry["N"]),
    )
    if not test_trajectories:
        logging.warning("No trajectories loaded for time test. Skipping.")
        return

    longest_traj = max(test_trajectories, key=lambda t: len(t.noisy_gps))
    num_points = len(longest_traj.noisy_gps)

    if model_names is None:
        model_names = manager._discover_models(model_root)

    time_config = job.get("time_config")
    if time_config is None:
        time_config = {
            "Q1": job_list["Q1"][0],
            "Q2": job_list["Q2"][0],
            "t_delta": job_list["t_delta"][0],
        }

    for model_name in model_names:
        model_dir = Path(model_root) / model_name
        ckpt_name = manager._find_best_checkpoint(model_dir)
        if ckpt_name is None:
            manager.logger.warning(f"No checkpoint found for {model_name}, skipping time test")
            continue
        ckpt_path = manager.trajectory_evaluator._get_checkpoint_path(str(model_dir), ckpt_name)
        if ckpt_path is None:
            manager.logger.warning(f"Checkpoint not found for {model_name}, skipping time test")
            continue

        for method in methods:
            avg_time = manager.trajectory_evaluator._measure_timing_with_config(
                ckpt_path, longest_traj, method, time_config
            )
            avg_time_per_point = avg_time / num_points if num_points else None
            _append_time_log_traj(
                logfile=time_log,
                model_name=model_name,
                model_tag="RectifiedTraj",
                dataset_name=dataset_name,
                denoise_method=method,
                avg_time_sec=avg_time,
                avg_time_sec_per_point=avg_time_per_point,
                num_points=num_points,
            )

    # Baseline timing (classic methods)
    from baseline import classic as classic_baseline
    baseline_method_table = [
        ("kalman_rts_ts", classic_baseline.kalman_rts_smoother),
        ("kalman_rts_notime", classic_baseline.kalman_rts_smoother),
        ("hampel", classic_baseline.hampel_filter),
        ("savgol", classic_baseline.savitzky_golay_filter),
        ("raw", classic_baseline.raw_baseline),
    ]
    baseline_methods = [(n, f) for n, f in baseline_method_table if n in set(classic_baselines)]
    ref_lat = float(longest_traj.clean_gps[0, 1])
    ref_lon = float(longest_traj.clean_gps[0, 0])
    enu_noisy = manager.trajectory_evaluator._gps_to_enu_batch(
        longest_traj.noisy_gps, ref_lat, ref_lon
    )

    for method_name, method_fn in baseline_methods:
        times = []
        for run_idx in range(5):
            start = time.time()
            try:
                if method_name == "kalman_rts_ts":
                    _ = method_fn(enu_noisy, timestamps=getattr(longest_traj, "timestamps", None))
                elif method_name == "kalman_rts_notime":
                    _ = method_fn(enu_noisy, timestamps=None)
                else:
                    _ = method_fn(enu_noisy)
            except TypeError:
                _ = method_fn(enu_noisy, timestamps=None)
            end = time.time()
            times.append(end - start)
        avg_time = float(sum(times) / len(times)) if times else 0.0
        avg_time_per_point = avg_time / num_points if num_points else None
        _append_time_log_traj(
            logfile=time_log,
            model_name=method_name,
            model_tag="Baseline",
            dataset_name=dataset_name,
            denoise_method="N/A",
            avg_time_sec=avg_time,
            avg_time_sec_per_point=avg_time_per_point,
            num_points=num_points,
        )

    if include_difftraj_timing:
        # Baseline timing (DiffTraj)
        try:
            from baseline.difftraj import DiffTrajPaths, difftraj_denoise_with_model, prepare_difftraj

            repo_dir = job.get("difftraj_repo", "./bin/baseline_model/difftraj")
            checkpoint_path = job.get("difftraj_ckpt", "")
            device = job.get("difftraj_device", "cuda")
            timesteps = int(job.get("difftraj_timesteps", 100))
            final_steps = job.get("difftraj_final_steps")
            final_steps = int(final_steps) if final_steps is not None else None
            eta = float(job.get("difftraj_eta", 0.0))

            paths = DiffTrajPaths(
                repo_dir=repo_dir or None,
                checkpoint_path=checkpoint_path or None,
            )
            config, model, device = prepare_difftraj(paths, device=device)
            target_len = int(getattr(getattr(config, "data", None), "traj_length", 0) or 0)
            logging.warning(
                "DiffTraj timing started (timesteps=%d, device=%s, target_len=%d).",
                timesteps,
                device,
                target_len,
            )

            def _pad_tail(values: np.ndarray, tgt_len: int) -> np.ndarray:
                values = np.asarray(values, dtype=np.float32)
                if values.shape[0] == tgt_len:
                    return values
                if values.shape[0] > tgt_len:
                    return values[:tgt_len]
                tail = np.repeat(values[-1:], tgt_len - values.shape[0], axis=0)
                return np.concatenate([values, tail], axis=0)

            def _denoise_once(enu_xy: np.ndarray) -> np.ndarray:
                if target_len <= 0 or enu_xy.shape[0] == target_len:
                    return difftraj_denoise_with_model(
                        enu_xy,
                        config=config,
                        model=model,
                        device=device,
                        timesteps=timesteps,
                        final_steps=final_steps,
                        eta=eta,
                    )
                if enu_xy.shape[0] < target_len:
                    return difftraj_denoise_with_model(
                        _pad_tail(enu_xy, target_len),
                        config=config,
                        model=model,
                        device=device,
                        timesteps=timesteps,
                        final_steps=final_steps,
                        eta=eta,
                    )[:enu_xy.shape[0]]
                out = []
                i = 0
                while i < enu_xy.shape[0]:
                    chunk = _pad_tail(enu_xy[i: i + target_len], target_len)
                    out.append(
                        difftraj_denoise_with_model(
                            chunk,
                            config=config,
                            model=model,
                            device=device,
                            timesteps=timesteps,
                            final_steps=final_steps,
                            eta=eta,
                        )
                    )
                    i += target_len
                return np.concatenate(out, axis=0)[:enu_xy.shape[0]]

            times = []
            for run_idx in range(5):
                start = time.time()
                _ = _denoise_once(enu_noisy)
                end = time.time()
                times.append(end - start)
                logging.warning("DiffTraj timing run %d/5: %.3fs", run_idx + 1, times[-1])

            avg_time = float(sum(times) / len(times)) if times else 0.0
            avg_time_per_point = avg_time / num_points if num_points else None
            _append_time_log_traj(
                logfile=time_log,
                model_name="difftraj",
                model_tag="Baseline",
                dataset_name=dataset_name,
                denoise_method="N/A",
                avg_time_sec=avg_time,
                avg_time_sec_per_point=avg_time_per_point,
                num_points=num_points,
            )
            logging.warning("DiffTraj timing saved.")
        except Exception as exc:
            logging.warning("DiffTraj timing baseline unavailable: %s", exc)


def _run_difftraj_chunk_baseline(
    manager: TestManager,
    job: dict,
    manual_config: dict | None,
    max_chunks: int | None,
) -> None:
    try:
        from theta_train import DataLoader
        from baseline.difftraj import DiffTrajPaths, difftraj_denoise_with_model, prepare_difftraj
    except Exception as exc:
        logging.warning("DiffTraj chunk baseline import failed: %s", exc)
        return

    test_dir = str(job.get("chunk_test_dir", "./dataset/processed/test"))
    test_files = sorted(Path(test_dir).glob("*.pt"))
    if not test_files:
        logging.warning("No chunk test files found for DiffTraj baseline: %s", test_dir)
        return

    runtime_template = {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "config": {
            "batch_size": 64,
            "train_dir": test_dir,
            "K": 256,
        },
    }

    x0_list = []
    x1_list = []
    for i, _ in enumerate(test_files):
        rt = dict(runtime_template)
        rt["config"] = dict(runtime_template["config"])
        dl = DataLoader(rt)
        dl.set(i)
        xt = dl.X_t[:, :, :2].float()
        v = dl.V[:, :, :2]
        t = dl.t
        t_b = t.view(-1, 1, 1)
        x0 = xt - v * t_b
        x1 = xt + v * (1.0 - t_b)
        x0_list.append(x0.cpu())
        x1_list.append(x1.cpu())

    x0 = torch.cat(x0_list, dim=0)
    x1 = torch.cat(x1_list, dim=0)
    total = x0.shape[0]
    if max_chunks is not None and max_chunks > 0 and total > max_chunks:
        x0 = x0[:max_chunks]
        x1 = x1[:max_chunks]
    num_chunks = x0.shape[0]
    if num_chunks <= 0:
        logging.warning("No chunks loaded for DiffTraj chunk baseline.")
        return

    repo_dir = job.get("difftraj_repo", "./bin/baseline_model/difftraj")
    checkpoint_path = job.get("difftraj_ckpt", "")
    device = job.get("difftraj_device", "cuda")
    timesteps = int(job.get("difftraj_timesteps", 100))
    final_steps = job.get("difftraj_final_steps")
    final_steps = int(final_steps) if final_steps is not None else None
    eta = float(job.get("difftraj_eta", 0.0))

    paths = DiffTrajPaths(
        repo_dir=repo_dir or None,
        checkpoint_path=checkpoint_path or None,
    )
    try:
        config, model, device = prepare_difftraj(paths, device=device)
    except Exception as exc:
        logging.warning("DiffTraj chunk baseline unavailable: %s", exc)
        return

    target_len = int(getattr(getattr(config, "data", None), "traj_length", 0) or 0)
    logging.warning(
        "DiffTraj chunk baseline started (chunks=%d, timesteps=%d, device=%s, target_len=%d).",
        num_chunks,
        timesteps,
        device,
        target_len,
    )

    def _pad_tail(values: np.ndarray, tgt_len: int) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.shape[0] == tgt_len:
            return values
        if values.shape[0] > tgt_len:
            return values[:tgt_len]
        tail = np.repeat(values[-1:], tgt_len - values.shape[0], axis=0)
        return np.concatenate([values, tail], axis=0)

    def _denoise_once(enu_xy: np.ndarray) -> np.ndarray:
        if target_len <= 0 or enu_xy.shape[0] == target_len:
            return difftraj_denoise_with_model(
                enu_xy,
                config=config,
                model=model,
                device=device,
                timesteps=timesteps,
                final_steps=final_steps,
                eta=eta,
            )
        if enu_xy.shape[0] < target_len:
            den = difftraj_denoise_with_model(
                _pad_tail(enu_xy, target_len),
                config=config,
                model=model,
                device=device,
                timesteps=timesteps,
                final_steps=final_steps,
                eta=eta,
            )
            return den[:enu_xy.shape[0]]

        out = []
        i = 0
        while i < enu_xy.shape[0]:
            chunk = _pad_tail(enu_xy[i: i + target_len], target_len)
            out.append(
                difftraj_denoise_with_model(
                    chunk,
                    config=config,
                    model=model,
                    device=device,
                    timesteps=timesteps,
                    final_steps=final_steps,
                    eta=eta,
                )
            )
            i += target_len
        return np.concatenate(out, axis=0)[:enu_xy.shape[0]]

    q1_bytes = (manual_config or {}).get("Q1", 1)
    q2_bytes = (manual_config or {}).get("Q2", 12)
    q1_points = int(q1_bytes) * 8
    q2_points = int(q2_bytes) * 8

    errs_full = []
    errs_mid = []
    failed = 0
    for i in range(num_chunks):
        inp = x1[i].numpy()
        gt = x0[i].numpy()
        try:
            pred = _denoise_once(inp)
        except Exception:
            failed += 1
            continue

        diff_full = pred - gt
        l2_full = np.sqrt((diff_full * diff_full).sum(axis=-1))
        errs_full.append(l2_full)

        if q2_points > 0:
            pred_mid = pred[q1_points:-q2_points]
            gt_mid = gt[q1_points:-q2_points]
        else:
            pred_mid = pred[q1_points:]
            gt_mid = gt[q1_points:]
        diff_mid = pred_mid - gt_mid
        l2_mid = np.sqrt((diff_mid * diff_mid).sum(axis=-1))
        errs_mid.append(l2_mid)
        if i == 0 or (i + 1) % 10 == 0 or (i + 1) == num_chunks:
            logging.warning("DiffTraj chunk baseline progress: %d/%d", i + 1, num_chunks)

    if not errs_full:
        logging.warning("DiffTraj chunk baseline produced no valid outputs (failed=%d).", failed)
        return

    errs_full = np.stack(errs_full, axis=0)
    errs_mid = np.stack(errs_mid, axis=0)
    row = {
        "model_name": "difftraj",
        "model_tag": "Baseline",
        "dataset_name": "chunk_test",
        "denoise_method": "N/A",
        "K": int(x0.shape[1]),
        "Q1": int(q1_bytes),
        "Q2": int(q2_bytes),
        "t_delta": None,
        "N_steps": None,
        "err_mean_full": float(errs_full.mean()),
        "err_median_full": float(np.median(errs_full)),
        "err_std_full": float(errs_full.std()),
        "err_mean_mid": float(errs_mid.mean()),
        "err_median_mid": float(np.median(errs_mid)),
        "err_std_mid": float(errs_mid.std()),
        "num_tested_chunks": int(errs_full.shape[0]),
        "test_timestamp": datetime.now().isoformat(),
    }
    manager.chunk_evaluator._append_row(row)
    logging.warning(
        "DiffTraj chunk baseline logged (chunks=%d, failed=%d).",
        int(errs_full.shape[0]),
        failed,
    )


def _run_chunk_eval(
    manager: TestManager,
    job: dict,
    model_root: str,
    model_names: list | None,
    classic_baselines: list[str],
    include_difftraj: bool = False,
) -> None:
    test_dir = job.get("chunk_test_dir", "./dataset/processed/test")
    _stage(f"Chunk eval start | test_dir={test_dir}")
    max_chunks = job.get("chunk_max_chunks", 5000)
    if max_chunks is not None:
        max_chunks = int(max_chunks)
    manual_config = job.get("chunk_config")
    if manual_config is None:
        q1 = job.get("Q1")
        q2 = job.get("Q2")
        t_delta = job.get("t_delta")
        if q1 is not None or q2 is not None or t_delta is not None:
            manual_config = {
                "Q1": q1[0] if isinstance(q1, list) and q1 else q1,
                "Q2": q2[0] if isinstance(q2, list) and q2 else q2,
                "t_delta": t_delta[0] if isinstance(t_delta, list) and t_delta else t_delta,
            }
            manual_config = {k: v for k, v in manual_config.items() if v is not None}
    run_baseline = job.get("run_baseline", job.get("baseline_once", True))
    try:
        manager.run_chunk_evaluation(
            model_names=model_names,
            model_root=model_root,
            test_dir=test_dir,
            max_chunks=max_chunks,
            manual_config=manual_config,
            run_baselines=bool(run_baseline),
            baseline_methods=classic_baselines,
        )
    except TypeError:
        logging.warning(
            "run_chunk_evaluation() does not support baseline filtering in this environment; "
            "running chunk eval without classic baselines to avoid unintended methods."
        )
        manager.run_chunk_evaluation(
            model_names=model_names,
            model_root=model_root,
            test_dir=test_dir,
            max_chunks=max_chunks,
            manual_config=manual_config,
            run_baselines=False,
        )
        if bool(run_baseline) and classic_baselines:
            logging.warning("Skipped classic chunk baselines due to API mismatch.")
    if include_difftraj and bool(run_baseline):
        _run_difftraj_chunk_baseline(
            manager=manager,
            job=job,
            manual_config=manual_config,
            max_chunks=max_chunks,
        )


def _run_difftraj_late_phase(
    manager: TestManager,
    job: dict,
    datasets: list[dict],
) -> None:
    _stage("Late DiffTraj phase start")
    for entry in datasets:
        test_trajectories, dataset_name = manager._load_or_generate_test_data(
            test_data_path=str(entry["path"]),
            M=int(entry["M"]),
            N=int(entry["N"]),
        )
        manager.trajectory_evaluator.set_run_context(dataset_name)
        _run_difftraj_baseline_trajectory(
            manager=manager,
            test_trajectories=test_trajectories,
            dataset_name=dataset_name,
            job=job,
        )

    if job.get("chunk_test", True):
        _stage("Late DiffTraj chunk phase start")
        test_dir = job.get("chunk_test_dir", "./dataset/processed/test")
        max_chunks = job.get("chunk_max_chunks", 5000)
        if max_chunks is not None:
            max_chunks = int(max_chunks)
        manual_config = job.get("chunk_config")
        if manual_config is None:
            q1 = job.get("Q1")
            q2 = job.get("Q2")
            t_delta = job.get("t_delta")
            if q1 is not None or q2 is not None or t_delta is not None:
                manual_config = {
                    "Q1": q1[0] if isinstance(q1, list) and q1 else q1,
                    "Q2": q2[0] if isinstance(q2, list) and q2 else q2,
                    "t_delta": t_delta[0] if isinstance(t_delta, list) and t_delta else t_delta,
                }
                manual_config = {k: v for k, v in manual_config.items() if v is not None}
        _run_difftraj_chunk_baseline(
            manager=manager,
            job=job,
            manual_config=manual_config,
            max_chunks=max_chunks,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trajectory benchmarks")
    parser.add_argument(
        "-test",
        action="store_true",
        help="Use debug_mini datasets automatically (benchmark mode)",
    )
    parser.add_argument("--wandb", action="store_true", help="Upload results to Weights & Biases")
    parser.add_argument("--wandb_project", default="", help="W&B project name")
    parser.add_argument("--wandb_entity", default="", help="W&B entity/team (optional)")
    parser.add_argument("--wandb_run_name", default="", help="W&B run name (optional)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    joblist_path = Path("./src/eval_joblist.json")
    if not joblist_path.exists():
        raise FileNotFoundError(f"Missing job list: {joblist_path}")

    with joblist_path.open("r") as f:
        job = json.load(f)

    model_root = job.get("model_root", "./bin/model")
    model_names = job.get("model_names")
    if model_names is not None and not model_names:
        model_names = None

    methods = job.get("methods", ["BF", "DF"])
    if isinstance(methods, str):
        methods = [m.strip() for m in methods.split(",") if m.strip()]
    if not methods:
        methods = ["BF", "DF"]

    progress_only = False
    log_level_name = str(job.get("log_level", "INFO")).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logging.getLogger().setLevel(log_level)
    _stage(f"Log level set to {logging.getLevelName(log_level)}")

    job_list = _build_job_list(job)
    classic_baselines = _resolve_classic_baselines(job)
    _stage(f"Classic baselines selected: {classic_baselines if classic_baselines else '[]'}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manager = TestManager(output_dir=str(Path("./bin/test_results") / f"test_{timestamp}"))
    manager.brief_summary = job.get("brief_summary", True)
    manager.brief_visualizer = job.get("brief_visualizer", True)
    manager.visualize_each_run = False
    wandb_enabled = bool(job.get("wandb", False)) or bool(args.wandb)
    wandb_project = args.wandb_project or job.get("wandb_project", "rectifiedtraj_benchmarks")
    wandb_entity = args.wandb_entity or job.get("wandb_entity") or None
    wandb_run_name = args.wandb_run_name or job.get("wandb_run_name") or Path(manager.output_dir).name

    gen_new_test = bool(job.get("gen_new_test", False))
    use_new_traj = job.get("use_new_traj", {}) or {}
    traj_paths = job.get("traj_paths", {}) or {}

    datasets: list[dict] = []

    full_entry = _resolve_existing_dataset(
        name="full_traj",
        path_value=traj_paths.get("full_traj"),
        debug_path=DEBUG_FULL_TRAJ,
        use_debug=args.test,
    )
    if full_entry is None and gen_new_test and not args.test:
        logging.info("Generating full_traj dataset...")
        output_dir = FULL_TRAJ_DIR
        if traj_paths.get("full_traj"):
            full_path = Path(traj_paths["full_traj"])
            output_dir = full_path if full_path.suffix == "" else full_path.parent
        pt_path = _generate_full_traj(output_dir, use_new_traj)
        meta = _load_metadata(pt_path)
        full_entry = (pt_path, int(meta["n_trajectories"]), int(meta["median_length"]))
    if full_entry is not None:
        datasets.append({"name": "full_traj", "path": full_entry[0], "M": full_entry[1], "N": full_entry[2]})

    if not datasets:
        if job.get("chunk_test", True):
            _run_chunk_eval(manager, job, model_root, model_names, classic_baselines)
            return
        raise ValueError("No valid trajectory datasets found. Provide traj_paths or enable gen_new_test.")

    run_baseline = job.get("run_baseline", job.get("baseline_once", True))
    if run_baseline:
        _stage("Baseline phase start")
        for entry in datasets:
            test_trajectories, dataset_name = manager._load_or_generate_test_data(
                test_data_path=str(entry["path"]),
                M=int(entry["M"]),
                N=int(entry["N"]),
            )
            _stage(
                f"Baseline dataset loaded | dataset={dataset_name} trajectories={len(test_trajectories)}"
            )
            manager.trajectory_evaluator.set_run_context(dataset_name)
            manager.classic_baseline_evaluator.progress_bar = bool(job.get("baseline_progress", True))
            manager.trajectory_evaluator.evaluate_baseline(test_trajectories, dataset_name=dataset_name)
            if classic_baselines:
                try:
                    manager.classic_baseline_evaluator.evaluate_classic_baselines(
                        test_trajectories, dataset_name=dataset_name, methods=classic_baselines
                    )
                except TypeError:
                    _run_classic_baselines_filtered_compat(
                        manager=manager,
                        test_trajectories=test_trajectories,
                        dataset_name=dataset_name,
                        methods=classic_baselines,
                    )
            else:
                _stage("Classic baseline list is empty; skipping classic baselines.")

    # Run trajectory tests over the provided hyperparameter grid only.
    _stage("Grid phase start")
    _run_grid_eval(manager, job_list, model_root, model_names, datasets)

    if job.get("range_test"):
        _stage("Range test phase start")
        _run_bounded_eval(manager, job, model_root, model_names, methods)

    _stage("Time test phase start")
    _run_time_tests(
        manager,
        job,
        job_list,
        model_root,
        model_names,
        datasets,
        methods,
        classic_baselines,
        include_difftraj_timing=False,
    )

    if job.get("chunk_test", True):
        _stage("Chunk test phase start")
        _run_chunk_eval(manager, job, model_root, model_names, classic_baselines)

    # Visualizer intentionally disabled for progress-only runs.

    if wandb_enabled:
        try:
            log_run_to_wandb(
                run_dir=manager.output_dir,
                project=wandb_project,
                entity=wandb_entity,
                run_name=wandb_run_name,
            )
        except Exception as exc:
            logging.warning("W&B upload failed: %s", exc)

    run_difftraj_late = bool(job.get("run_difftraj_baseline", True)) and bool(run_baseline)
    if run_difftraj_late:
        _run_difftraj_late_phase(
            manager=manager,
            job=job,
            datasets=datasets,
        )
        if wandb_enabled:
            try:
                log_run_to_wandb(
                    run_dir=manager.output_dir,
                    project=wandb_project,
                    entity=wandb_entity,
                    run_name=f"{wandb_run_name}_difftraj",
                )
            except Exception as exc:
                logging.warning("Second W&B upload (DiffTraj) failed: %s", exc)

    print("\n✓ Evaluation complete")


if __name__ == "__main__":
    main()
