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

from utils.evaluations.evaluation_manager import TestManager
from utils.data_processor.traj_extractor import traj_extractor
from utils.data_processor.extract_10min_traj import extract_10min_traj


FULL_TRAJ_DIR = Path("./dataset/processed/full_traj")
DEBUG_FULL_TRAJ = FULL_TRAJ_DIR / "fulltraj_debug_mini.pt"

DEFAULT_TIME_NPY = Path("./dataset/time_test/source_list.npy")
DEFAULT_TIME_LOG = Path("./bin/log/time_test.csv")


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


def _run_time_tests(
    manager: TestManager,
    job: dict,
    job_list: dict,
    model_root: str,
    model_names: list | None,
    dataset_entries: list[dict],
    methods: list,
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
    baseline_methods = [
        ("kalman_rts_ts", classic_baseline.kalman_rts_smoother),
        ("kalman_rts_notime", classic_baseline.kalman_rts_smoother),
        ("hampel", classic_baseline.hampel_filter),
        ("savgol", classic_baseline.savitzky_golay_filter),
        ("spline", classic_baseline.smoothing_spline),
    ]
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


def _run_chunk_eval(
    manager: TestManager,
    job: dict,
    model_root: str,
    model_names: list | None,
) -> None:
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
    run_baseline = job.get("run_baseline", job.get("baseline_once", True))
    manager.run_chunk_evaluation(
        model_names=model_names,
        model_root=model_root,
        test_dir=test_dir,
        max_chunks=max_chunks,
        manual_config=manual_config,
        run_baselines=bool(run_baseline),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trajectory benchmarks")
    parser.add_argument(
        "-test",
        action="store_true",
        help="Use debug_mini datasets automatically (benchmark mode)",
    )
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
    logging.getLogger().setLevel(logging.WARNING)

    job_list = _build_job_list(job)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manager = TestManager(output_dir=str(Path("./bin/test_results") / f"test_{timestamp}"))
    manager.brief_summary = job.get("brief_summary", True)
    manager.brief_visualizer = job.get("brief_visualizer", True)
    manager.visualize_each_run = False

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
            _run_chunk_eval(manager, job, model_root, model_names)
            return
        raise ValueError("No valid trajectory datasets found. Provide traj_paths or enable gen_new_test.")

    run_baseline = job.get("run_baseline", job.get("baseline_once", True))
    if run_baseline:
        for entry in datasets:
            test_trajectories, dataset_name = manager._load_or_generate_test_data(
                test_data_path=str(entry["path"]),
                M=int(entry["M"]),
                N=int(entry["N"]),
            )
            manager.trajectory_evaluator.set_run_context(dataset_name)
            manager.classic_baseline_evaluator.progress_bar = bool(job.get("baseline_progress", True))
            manager.trajectory_evaluator.evaluate_baseline(test_trajectories, dataset_name=dataset_name)
            manager.classic_baseline_evaluator.evaluate_classic_baselines(
                test_trajectories, dataset_name=dataset_name
            )

    # Run trajectory tests over the provided hyperparameter grid only.
    _run_grid_eval(manager, job_list, model_root, model_names, datasets)

    if job.get("range_test"):
        _run_bounded_eval(manager, job, model_root, model_names, methods)

    _run_time_tests(manager, job, job_list, model_root, model_names, datasets, methods)

    if job.get("chunk_test", True):
        _run_chunk_eval(manager, job, model_root, model_names)

    # Visualizer intentionally disabled for progress-only runs.

    print("\n✓ Evaluation complete")


if __name__ == "__main__":
    main()
