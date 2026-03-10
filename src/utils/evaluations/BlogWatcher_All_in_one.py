#!/usr/bin/env python3
"""
BlogWatcher_All_in_one.py

End-to-end BlogWatcher bounded evaluation runner.

Workflow:
1. Find the parquet dataset placed under ./dataset/raw/BlogWatcher.
2. Run parquet_processor in test-only mode on that file.
3. Run run_benchmarks.py using the current src/eval_joblist.json.
4. Optionally upload only the generated benchmark result directory to W&B.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHONPATH_ROOT = REPO_ROOT / "src"
DEFAULT_RAW_DS_PATH = REPO_ROOT / "dataset" / "raw" / "BlogWatcher"
DEFAULT_JOBLIST_PATH = REPO_ROOT / "src" / "eval_joblist.json"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "bin" / "test_results"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BlogWatcher all-in-one benchmark flow.")
    parser.add_argument(
        "--raw-ds-path",
        default=str(DEFAULT_RAW_DS_PATH),
        help="Directory containing the BlogWatcher parquet dataset.",
    )
    parser.add_argument(
        "--file",
        default="",
        help="Optional parquet filename/path override. If omitted, auto-detect inside raw-ds-path.",
    )
    parser.add_argument("--wandb", action="store_true", help="Upload the final benchmark result dir to W&B.")
    parser.add_argument("--wandb_project", default="rectifiedtraj_benchmarks", help="W&B project name.")
    parser.add_argument("--wandb_entity", default="", help="W&B entity/team (optional).")
    parser.add_argument("--wandb_run_name", default="", help="W&B run name (optional).")
    return parser.parse_args()


def _build_subprocess_env() -> dict:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "").strip()
    src_value = str(PYTHONPATH_ROOT)
    env["PYTHONPATH"] = src_value if not existing else f"{src_value}{os.pathsep}{existing}"
    return env


def _resolve_parquet_file(raw_ds_path: Path, file_hint: str) -> Path:
    if file_hint:
        hinted = Path(file_hint)
        if not hinted.is_absolute():
            hinted = (raw_ds_path / hinted.name) if len(hinted.parts) == 1 else (REPO_ROOT / hinted)
        hinted = hinted.resolve()
        if not hinted.exists():
            raise FileNotFoundError(f"Specified parquet file does not exist: {hinted}")
        if hinted.suffix.lower() != ".parquet":
            raise ValueError(f"Specified file is not a parquet file: {hinted}")
        return hinted

    candidates = sorted(
        raw_ds_path.glob("*.parquet"),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No parquet files found under {raw_ds_path}")
    if len(candidates) > 1:
        logging.info(
            "Multiple parquet files found under %s; using newest: %s",
            raw_ds_path,
            candidates[0].name,
        )
    return candidates[0].resolve()


def _existing_result_dirs(results_root: Path) -> set[Path]:
    if not results_root.exists():
        return set()
    return {one.resolve() for one in results_root.iterdir() if one.is_dir()}


def _detect_new_result_dir(results_root: Path, before: set[Path], started_at: float) -> Path:
    if not results_root.exists():
        raise FileNotFoundError(f"Benchmark results root does not exist: {results_root}")

    candidates = [one.resolve() for one in results_root.iterdir() if one.is_dir()]
    new_dirs = [one for one in candidates if one not in before]
    if new_dirs:
        return max(new_dirs, key=lambda path: path.stat().st_mtime)

    recent_dirs = [one for one in candidates if one.stat().st_mtime >= started_at - 1.0]
    if recent_dirs:
        return max(recent_dirs, key=lambda path: path.stat().st_mtime)

    raise RuntimeError("run_benchmarks.py finished, but no new result directory was detected.")


def _run_command(cmd: list[str], *, env: dict, stage: str) -> None:
    logging.info("[%s] %s", stage, shlex.join(cmd))
    subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
    )


@contextlib.contextmanager
def _temporary_disable_joblist_wandb(joblist_path: Path):
    original_text = joblist_path.read_text(encoding="utf-8")
    mutated = False
    try:
        payload = json.loads(original_text)
        if isinstance(payload, dict) and bool(payload.get("wandb", False)):
            payload["wandb"] = False
            joblist_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            mutated = True
        yield
    finally:
        if mutated:
            joblist_path.write_text(original_text, encoding="utf-8")


def _init_wandb(args: argparse.Namespace, raw_ds_path: Path, parquet_file: Path):
    try:
        import wandb
    except Exception as exc:
        raise RuntimeError(f"wandb not available: {exc}") from exc

    run_name = args.wandb_run_name or f"BlogWatcher_All_in_one_{datetime.now():%Y%m%d_%H%M%S}"
    run = wandb.init(
        project=args.wandb_project,
        entity=(args.wandb_entity or None),
        name=run_name,
    )
    run.config.update(
        {
            "runner": "BlogWatcher_All_in_one",
            "raw_dataset_dir": str(raw_ds_path.relative_to(REPO_ROOT)),
            "selected_parquet_file": parquet_file.name,
            "joblist": str(DEFAULT_JOBLIST_PATH.relative_to(REPO_ROOT)),
        },
        allow_val_change=True,
    )
    return run


def _upload_result_dir_to_wandb(run, result_dir: Path, parquet_file: Path) -> None:
    import wandb

    run.summary["selected_parquet_file"] = parquet_file.name
    run.summary["benchmark_result_dir"] = str(result_dir.relative_to(REPO_ROOT))
    artifact = wandb.Artifact(
        name=result_dir.name,
        type="benchmark_run",
    )
    artifact.add_dir(str(result_dir))
    run.log_artifact(artifact)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")

    raw_ds_path = Path(args.raw_ds_path).resolve()
    if not raw_ds_path.exists():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {raw_ds_path}")
    if not raw_ds_path.is_dir():
        raise NotADirectoryError(raw_ds_path)

    if not DEFAULT_JOBLIST_PATH.exists():
        raise FileNotFoundError(f"Missing eval joblist: {DEFAULT_JOBLIST_PATH}")

    parquet_file = _resolve_parquet_file(raw_ds_path, args.file)
    logging.info("Selected parquet file: %s", parquet_file.name)
    logging.info(
        "Dataset folder name drives processing identity; parquet basename can be arbitrary under %s.",
        raw_ds_path,
    )

    env = _build_subprocess_env()
    wandb_run = _init_wandb(args, raw_ds_path, parquet_file) if args.wandb else None
    results_before = _existing_result_dirs(DEFAULT_RESULTS_ROOT)
    benchmark_start = 0.0

    try:
        _run_command(
            [
                sys.executable,
                "-m",
                "utils.data_processor.parquet_processor",
                "--mode",
                "test-only",
                "--raw-ds-path",
                str(raw_ds_path),
                "--files",
                parquet_file.name,
            ],
            env=env,
            stage="parquet_processor",
        )

        benchmark_start = time.time()
        with _temporary_disable_joblist_wandb(DEFAULT_JOBLIST_PATH):
            _run_command(
                [
                    sys.executable,
                    str(REPO_ROOT / "src" / "run_benchmarks.py"),
                ],
                env=env,
                stage="run_benchmarks",
            )

        result_dir = _detect_new_result_dir(
            DEFAULT_RESULTS_ROOT,
            before=results_before,
            started_at=benchmark_start,
        )
        logging.info("Detected benchmark result dir: %s", result_dir)

        if wandb_run is not None:
            _upload_result_dir_to_wandb(wandb_run, result_dir, parquet_file)
    except subprocess.CalledProcessError as exc:
        if wandb_run is not None:
            wandb_run.summary["failed_stage"] = "subprocess"
            wandb_run.summary["return_code"] = int(exc.returncode)
            wandb_run.finish()
        raise SystemExit(exc.returncode) from exc
    except Exception:
        if wandb_run is not None:
            wandb_run.summary["failed_stage"] = "python"
            wandb_run.finish()
        raise
    else:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
