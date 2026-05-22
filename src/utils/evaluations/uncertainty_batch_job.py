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

from utils.evaluations.evaluation_manager import TestManager


def _copy_if_exists(source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one uncertainty evaluation job from a JSON spec.")
    parser.add_argument("--spec-json", required=True, help="Path to the job spec JSON.")
    args = parser.parse_args()

    spec_path = Path(args.spec_json).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    output_dir = Path(spec["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, str(spec.get("log_level", "INFO")).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(output_dir / "job.log", encoding="utf-8")],
    )

    manager = TestManager(output_dir=str(output_dir))
    task_type = str(spec.get("task_type", "learned_model") or "learned_model").strip().lower()
    if task_type == "classic_baseline":
        results = manager.run_uncertainty_band_test(
            model_names=[],
            model_root=str(spec.get("model_root", "./bin/model/RectifiedTraj")),
            model_tag="Baseline",
            test_data_path=str(spec["test_data_path"]),
            M=int(spec["M"]),
            N=int(spec["N"]),
            run_baselines=True,
            baseline_methods=[str(spec["baseline_method"])],
        )
    elif task_type == "learned_model":
        results = manager.run_uncertainty_band_test(
            model_names=[str(spec["model_name"])],
            model_root=str(spec["model_root"]),
            model_tag=str(spec["model_tag"]),
            test_data_path=str(spec["test_data_path"]),
            M=int(spec["M"]),
            N=int(spec["N"]),
            run_baselines=False,
            manual_config=dict(spec["manual_config"]),
        )
    else:
        raise ValueError(f"Unsupported uncertainty batch task type: {task_type!r}")

    if not results:
        raise RuntimeError("No uncertainty evaluation result was produced.")

    _copy_if_exists(output_dir / "uncertainty_band_summary.csv", output_dir / "uncertainty_result.csv")
    _copy_if_exists(output_dir / "uncertainty_traj_p_val.csv", output_dir / "uncertainty_p_val.csv")
    manifest = {
        "spec": spec,
        "uncertainty_result_csv": str((output_dir / "uncertainty_result.csv").resolve())
        if (output_dir / "uncertainty_result.csv").exists()
        else None,
        "uncertainty_p_val_csv": str((output_dir / "uncertainty_p_val.csv").resolve())
        if (output_dir / "uncertainty_p_val.csv").exists()
        else None,
        "status": "ok",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
