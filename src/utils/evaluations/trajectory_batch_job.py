#!/usr/bin/env python3
import argparse
import json
import logging
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.evaluations.classic_baseline_runner import run_classic_baselines_filtered
from utils.evaluations.evaluation_manager import TestManager


def _copy_if_exists(source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _find_single_recursive(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.rglob(pattern))
    return matches[0] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one trajectory evaluation job from a JSON spec.")
    parser.add_argument("--spec-json", required=True, help="Path to the job spec JSON.")
    args = parser.parse_args()

    spec_path = Path(args.spec_json).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    output_dir = Path(spec["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "job.log"
    logging.basicConfig(
        level=getattr(logging, str(spec.get("log_level", "INFO")).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )

    manager = TestManager(output_dir=str(output_dir))
    task_type = str(spec.get("task_type", "learned_model") or "learned_model").strip().lower()
    results = None
    if task_type == "classic_baseline":
        test_trajectories, dataset_name = manager._load_or_generate_test_data(
            test_data_path=str(spec["test_data_path"]),
            M=int(spec["M"]),
            N=int(spec["N"]),
        )
        manager.trajectory_evaluator.set_run_context(dataset_name)
        results = run_classic_baselines_filtered(
            manager=manager,
            test_trajectories=test_trajectories,
            dataset_name=dataset_name,
            methods=[str(spec["baseline_method"])],
            dataset_name_hint=dataset_name,
            baseline_config=spec.get("baseline_config"),
            diagnostics_output_dir=output_dir,
        )
    elif task_type == "learned_model":
        results = manager.run_trajectory_evaluation(
            model_names=[str(spec["model_name"])],
            model_root=str(spec["model_root"]),
            model_tag=str(spec["model_tag"]),
            test_data_path=str(spec["test_data_path"]),
            M=int(spec["M"]),
            N=int(spec["N"]),
            manual_config=dict(spec["manual_config"]),
            run_baselines=False,
        )
        if not results:
            raise RuntimeError("No trajectory evaluation result was produced.")
    else:
        raise ValueError(f"Unsupported trajectory batch task type: {task_type!r}")

    traj_summary = output_dir / "trajectory_evaluation_summary.csv"
    pointwise_summary = _find_single_recursive(output_dir, "trajectory_pointwise_summary.csv")
    traj_p_val = _find_single_recursive(output_dir, "traj_p_val.csv")

    _copy_if_exists(traj_summary, output_dir / "traj_result.csv")
    if pointwise_summary is not None:
        _copy_if_exists(pointwise_summary, output_dir / "pw_result.csv")
    if traj_p_val is not None:
        _copy_if_exists(traj_p_val, output_dir / "traj_p_val.csv")

    manifest = {
        "spec": spec,
        "traj_result_csv": str((output_dir / "traj_result.csv").resolve()),
        "pw_result_csv": str((output_dir / "pw_result.csv").resolve()) if (output_dir / "pw_result.csv").exists() else None,
        "traj_p_val_csv": str((output_dir / "traj_p_val.csv").resolve()) if (output_dir / "traj_p_val.csv").exists() else None,
        "result_json": str((output_dir / "result.json").resolve()) if results else None,
        "status": "ok",
    }
    if results:
        (output_dir / "result.json").write_text(
            json.dumps(results[0], indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
