#!/usr/bin/env python3
"""
uncertainty_grid_runner.py

End-to-end grid evaluation for external uncertainty-band parquet data.
Creates (or reuses) a processed trajectory dataset, then runs grid search
over Q1/Q2/t_delta for BF/DF methods, including baselines.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.data_processor.traj_extractor import traj_extractor_with_error_range
from utils.evaluations.evaluation_manager import TestManager
from utils.evaluations.progress import ProgressTracker
from utils.evaluations.uncertainty import UncertaintyBandTrajectoryTest
from utils.evaluations.wandb_logger import log_run_to_wandb


JOBLIST_PATH = Path("./src/eval_joblist.json")
DEFAULT_PARQUET_DIR = "./dataset/external_uncertainty"
DEFAULT_METHODS = ["BF", "DF"]
DEFAULT_Q1_LIST = [1]
DEFAULT_Q2_LIST = [12]
DEFAULT_T_DELTA_LIST = [1.0]
DEFAULT_M = 1000
DEFAULT_N = 5000
TEST_M = 1
TEST_N = 3000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="External uncertainty-band evaluation runner")
    parser.add_argument("--parquet_dir", default="", help="Parquet directory (optional override)")
    parser.add_argument(
        "--processed_dir",
        default="./dataset/external_uncertainty/processed",
        help="Processed output directory",
    )
    parser.add_argument("--output_dir", default="./bin/test_result_uncertainty", help="Test results output directory")
    parser.add_argument("--model_root", default="./bin/model/RectifiedTraj", help="Model root directory")
    parser.add_argument("--model_names", default="", help="Comma-separated model names (empty = all)")
    parser.add_argument("--M", type=int, default=1000, help="Target number of trajectories")
    parser.add_argument("--N", type=int, default=5000, help="Target points per trajectory (threshold)")
    parser.add_argument("--reuse", action="store_true", help="Reuse latest processed dataset if available")
    parser.add_argument("--test", action="store_true", help="Quick test run (M=1, N=200)")
    parser.add_argument("-csv", "--csv", action="store_true", help="Save detailed results as CSV instead of parquet")
    parser.add_argument("--wandb", action="store_true", help="Upload results to Weights & Biases")
    parser.add_argument("--wandb_project", default="uncertainty_band", help="W&B project name")
    parser.add_argument("--wandb_entity", default="", help="W&B entity/team (optional)")
    parser.add_argument("--wandb_run_name", default="", help="W&B run name (optional)")
    parser.add_argument(
        "--difftraj_repo",
        default=str(Path("./bin/baseline_model/difftraj")),
        help="DiffTraj repo dir (or set DIFFTRAJ_REPO)",
    )
    parser.add_argument("--difftraj_ckpt", default="", help="DiffTraj checkpoint path override")
    parser.add_argument("--difftraj_device", default="cuda", help="DiffTraj device (cuda or cpu)")
    parser.add_argument("--difftraj_timesteps", type=int, default=100, help="DiffTraj sampling timesteps")
    parser.add_argument(
        "--difftraj_final_steps",
        type=int,
        default=None,
        help="Optional final number of steps for DiffTraj sampling",
    )
    parser.add_argument("--difftraj_eta", type=float, default=0.0, help="DiffTraj DDIM eta")
    return parser.parse_args()


def _load_joblist() -> dict:
    if not JOBLIST_PATH.exists():
        return {}
    with JOBLIST_PATH.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid joblist format: {JOBLIST_PATH}")
    return data


def _latest_pt_file(directory: Path) -> Path | None:
    candidates = sorted(directory.glob("fulltraj_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")

    joblist = _load_joblist()
    parquet_dir_value = args.parquet_dir or joblist.get("parquet_dir") or DEFAULT_PARQUET_DIR
    q1_list = joblist.get("Q1") or DEFAULT_Q1_LIST
    q2_list = joblist.get("Q2") or DEFAULT_Q2_LIST
    t_delta_list = joblist.get("t_delta") or DEFAULT_T_DELTA_LIST
    methods = joblist.get("methods") or DEFAULT_METHODS
    default_m = int(joblist.get("M", DEFAULT_M))
    default_n = int(joblist.get("N", DEFAULT_N))

    parquet_dir = Path(parquet_dir_value)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    args.M = default_m
    args.N = default_n

    test_data_path: Path | None = None
    if args.reuse:
        test_data_path = _latest_pt_file(processed_dir)

    if test_data_path is None:
        result = traj_extractor_with_error_range(
            parquet_dir=str(parquet_dir),
            M=args.M,
            N=args.N,
            output_dir=str(processed_dir),
        )
        test_data_path = Path(result["output_file"])

    model_names = [m.strip() for m in args.model_names.split(",") if m.strip()] or None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output = Path(args.output_dir) / f"test_{timestamp}"
    manager = TestManager(output_dir=str(run_output))

    dataset_name = Path(test_data_path).stem if test_data_path else "unknown"
    test_trajectories = manager._load_or_generate_uncertainty_test_data(
        str(test_data_path), args.M, args.N
    )

    detail_format = "csv" if args.csv else "parquet"
    tester = UncertaintyBandTrajectoryTest(str(run_output), detail_format=detail_format)
    tester.evaluate_classic_baselines(test_trajectories=test_trajectories, dataset_name=dataset_name)
    sys.stdout.write("\n")
    sys.stdout.flush()
    tester.evaluate_difftraj_baseline(
        test_trajectories=test_trajectories,
        dataset_name=dataset_name,
        repo_dir=args.difftraj_repo or None,
        checkpoint_path=args.difftraj_ckpt or None,
        device=args.difftraj_device,
        timesteps=args.difftraj_timesteps,
        final_steps=args.difftraj_final_steps,
        eta=args.difftraj_eta,
    )

    model_root = Path(args.model_root)
    if model_names is None:
        model_names = manager._discover_models(str(model_root))
    model_jobs = []
    for model_name in model_names:
        model_dir = model_root / model_name
        checkpoint_name = manager._find_best_checkpoint(model_dir)
        if checkpoint_name is None:
            logging.warning("No checkpoint found for %s, skipping", model_name)
            continue

        config = manager._load_model_config(model_dir)
        K = config.get("K", 256)
        model_jobs.append((model_name, model_dir, checkpoint_name, K))

    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.setLevel(logging.WARNING)

    progress_tracker = ProgressTracker(
        total_models=len(model_jobs),
        total_q1=len(q1_list),
        total_q2=len(q2_list),
        total_step=len(t_delta_list),
        total_method=len(methods),
    )
    progress_tracker.update(phase="uncertainty", dataset=dataset_name)

    for model_idx, (model_name, model_dir, checkpoint_name, K) in enumerate(model_jobs):
        progress_tracker.update(model=model_name, model_idx=model_idx)

        for q1_idx, Q1 in enumerate(q1_list):
            for q2_idx, Q2 in enumerate(q2_list):
                for step_idx, t_delta in enumerate(t_delta_list):
                    for method_idx, method in enumerate(methods):
                        progress_tracker.update(
                            q1=Q1,
                            q2=Q2,
                            q1_idx=q1_idx,
                            q2_idx=q2_idx,
                            step_idx=step_idx,
                            method_idx=method_idx,
                            method=method,
                            t_delta=t_delta,
                        )
                        logging.info(
                            "Testing %s with %s (uncertainty band grid) Q1=%s Q2=%s t_delta=%s",
                            model_name,
                            method,
                            Q1,
                            Q2,
                            t_delta,
                        )
                        manual_config = {
                            "Q1": Q1,
                            "Q2": Q2,
                            "t_delta": t_delta,
                        }
                        try:
                            tester.evaluate_model(
                                model_name=model_name,
                                model_dir=str(model_dir.absolute()),
                                checkpoint_name=checkpoint_name,
                                denoise_method=method,
                                test_trajectories=test_trajectories,
                                K=K,
                                Q1=Q1,
                                Q2=Q2,
                                manual_config=manual_config,
                            )
                        except AssertionError as exc:
                            logging.warning(
                                "SKIPPED (Invalid) %s Q1=%s Q2=%s t_delta=%s | %s",
                                model_name,
                                Q1,
                                Q2,
                                t_delta,
                                str(exc),
                            )
                        except Exception as exc:
                            logging.warning(
                                "SKIPPED (Error) %s Q1=%s Q2=%s t_delta=%s | %s: %s",
                                model_name,
                                Q1,
                                Q2,
                                t_delta,
                                type(exc).__name__,
                                str(exc),
                            )
                        finally:
                            progress_tracker.update(job_finished=True)

    sys.stdout.write("\n")
    sys.stdout.flush()
    root_logger.setLevel(previous_level)

    logging.info("Completed uncertainty-band evaluation. Results in %s", run_output)

    if args.wandb:
        entity = args.wandb_entity or None
        run_name = args.wandb_run_name or run_output.name
        log_run_to_wandb(
            run_dir=str(run_output),
            project=args.wandb_project,
            entity=entity,
            run_name=run_name,
        )


if __name__ == "__main__":
    main()
