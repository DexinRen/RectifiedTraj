#!/usr/bin/env python3
import argparse
import json
import logging
import os
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
    parser = argparse.ArgumentParser(description="Run one chunk evaluation job from a JSON spec.")
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

    task_type = str(spec.get("task_type", "learned_model") or "learned_model").strip().lower()
    manager = TestManager(output_dir=str(output_dir))
    if task_type == "classic_baseline":
        previous_mode = os.getenv("KALMAN_RTS_CALIBRATION_MODE")
        try:
            if spec.get("kalman_mode"):
                os.environ["KALMAN_RTS_CALIBRATION_MODE"] = str(spec["kalman_mode"])
            manager.run_chunk_evaluation(
                model_names=[],
                model_root=str(spec.get("model_root", "./bin/model/RectifiedTraj")),
                model_tag="Baseline",
                test_dir=str(spec["test_dir"]),
                max_chunks=spec.get("max_chunks"),
                manual_config=spec.get("manual_config"),
                run_baselines=True,
                baseline_methods=[str(spec["baseline_method"])],
                baseline_config=spec.get("baseline_config"),
                diagnostics_output_dir=output_dir,
            )
        finally:
            if previous_mode is None:
                os.environ.pop("KALMAN_RTS_CALIBRATION_MODE", None)
            else:
                os.environ["KALMAN_RTS_CALIBRATION_MODE"] = previous_mode
    elif task_type == "learned_model":
        manager.run_chunk_evaluation(
            model_names=[str(spec["model_name"])],
            model_root=str(spec["model_root"]),
            model_tag=str(spec["model_tag"]),
            test_dir=str(spec["test_dir"]),
            max_chunks=spec.get("max_chunks"),
            manual_config=spec.get("manual_config"),
            run_baselines=False,
        )
    else:
        raise ValueError(f"Unsupported chunk batch task type: {task_type!r}")

    _copy_if_exists(output_dir / "chunk_evaluation_summary.csv", output_dir / "chunk_result.csv")
    _copy_if_exists(output_dir / "chunk_pointwise_summary.csv", output_dir / "chunk_pointwise_result.csv")
    _copy_if_exists(output_dir / "chunk_bytewise_summary.csv", output_dir / "chunk_bytewise_result.csv")
    _copy_if_exists(output_dir / "chunk_p_val.csv", output_dir / "chunk_p_val_result.csv")
    manifest = {
        "spec": spec,
        "chunk_result_csv": str((output_dir / "chunk_result.csv").resolve())
        if (output_dir / "chunk_result.csv").exists()
        else None,
        "status": "ok",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
