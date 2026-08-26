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
import ast
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
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Validate BlogWatcher runner inputs and joblist prerequisites, then exit before running jobs.",
    )
    parser.add_argument(
        "--check-git-tracked",
        action="store_true",
        help="With --smoke-test, fail if required repository source files are not tracked by git.",
    )
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


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _read_joblist(joblist_path: Path) -> dict:
    try:
        payload = json.loads(joblist_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {joblist_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Joblist must be a JSON object: {joblist_path}")
    return payload


def _local_module_file(module_name: str) -> Path | None:
    if not module_name:
        return None
    roots = {"utils", "baseline", "encoder_decoder", "theta_model", "theta_train", "run_benchmarks"}
    root = module_name.split(".", 1)[0]
    if root not in roots:
        return None

    parts = module_name.split(".")
    module_path = PYTHONPATH_ROOT.joinpath(*parts)
    file_path = module_path.with_suffix(".py")
    if file_path.exists():
        return file_path.resolve()

    package_init = module_path / "__init__.py"
    if package_init.exists():
        return package_init.resolve()
    return None


def _module_name_for_file(file_path: Path) -> str:
    rel = file_path.resolve().relative_to(PYTHONPATH_ROOT)
    if rel.name == "__init__.py":
        rel = rel.parent
    else:
        rel = rel.with_suffix("")
    return ".".join(rel.parts)


def _resolve_relative_import(current_module: str, module: str | None, level: int) -> str:
    parts = current_module.split(".")
    if Path(*parts).name != "__init__":
        parts = parts[:-1]
    if level > 0:
        parts = parts[: max(len(parts) - level + 1, 0)]
    if module:
        parts.extend(module.split("."))
    return ".".join(part for part in parts if part)


def _collect_local_source_dependencies(entrypoints: list[Path]) -> tuple[set[Path], list[str]]:
    discovered: set[Path] = set()
    pending = [path.resolve() for path in entrypoints if path.exists()]
    warnings: list[str] = []

    while pending:
        file_path = pending.pop()
        if file_path in discovered:
            continue
        discovered.add(file_path)

        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        except SyntaxError as exc:
            warnings.append(f"Could not parse {file_path}: {exc}")
            continue

        current_module = _module_name_for_file(file_path)
        for node in ast.walk(tree):
            module_names: list[str] = []
            if isinstance(node, ast.Import):
                module_names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base_module = _resolve_relative_import(current_module, node.module, node.level)
                    module_names.append(base_module)
                    for alias in node.names:
                        if alias.name != "*":
                            module_names.append(f"{base_module}.{alias.name}")
                elif node.module:
                    module_names.append(node.module)
                    for alias in node.names:
                        module_names.append(f"{node.module}.{alias.name}")

            for module_name in module_names:
                dep_file = _local_module_file(module_name)
                if dep_file is not None and dep_file not in discovered:
                    pending.append(dep_file)

    return discovered, warnings


def _job_model_groups(job: dict) -> list[dict]:
    groups = job.get("model_groups")
    if isinstance(groups, list) and groups:
        return [group for group in groups if isinstance(group, dict)]

    out: list[dict] = []
    for legacy_key in ("rectifiedtraj", "residualreg"):
        block = job.get(legacy_key)
        if isinstance(block, dict):
            out.append(block)
    return out


def _model_names_from_group(group: dict, model_root: Path) -> list[str]:
    configured = group.get("model_names", group.get("models"))
    names = [str(item).strip() for item in _as_list(configured) if str(item).strip()]
    if names:
        return names

    if not model_root.exists():
        return []
    out: list[str] = []
    for model_dir in sorted(model_root.iterdir()):
        if model_dir.is_dir() and (
            any((model_dir / "best_ckpt").glob("*.safetensors"))
            or any((model_dir / "best_ckpt").glob("*_full.pt"))
            or any((model_dir / "ckpts").glob("*.safetensors"))
            or any((model_dir / "ckpts").glob("*_full.pt"))
        ):
            out.append(model_dir.name)
    return out


def _representative_checkpoint(model_dir: Path) -> Path | None:
    best_ckpts = sorted((model_dir / "best_ckpt").glob("*.safetensors"))
    if best_ckpts:
        return best_ckpts[0].resolve()

    best_ckpts = sorted((model_dir / "best_ckpt").glob("*_full.pt"))
    if best_ckpts:
        return best_ckpts[0].resolve()

    ckpts = sorted((model_dir / "ckpts").glob("*.safetensors"), key=lambda path: path.stat().st_mtime)
    if ckpts:
        return ckpts[-1].resolve()

    ckpts = sorted((model_dir / "ckpts").glob("*_full.pt"), key=lambda path: path.stat().st_mtime)
    if ckpts:
        return ckpts[-1].resolve()
    return None


def _collect_job_prerequisites(job: dict) -> tuple[set[Path], set[Path], list[str]]:
    required_files: set[Path] = {
        DEFAULT_JOBLIST_PATH.resolve(),
        (REPO_ROOT / "src" / "run_benchmarks.py").resolve(),
        (REPO_ROOT / "src" / "utils" / "data_processor" / "parquet_processor.py").resolve(),
    }
    dataset_paths: set[Path] = set()
    errors: list[str] = []

    test_files = job.get("test_files", {})
    if isinstance(test_files, dict):
        for path_value in _as_list(test_files.get("traj_files")) + _as_list(test_files.get("chunk_files")):
            if str(path_value).strip():
                dataset_paths.add(_resolve_repo_path(path_value))

    data_source = job.get("data_source", {})
    if isinstance(data_source, dict):
        raw_dataset_dir = str(data_source.get("raw_dataset_dir", "") or "").strip()
        if raw_dataset_dir and raw_dataset_dir.lower() not in {"none", "null"}:
            dataset_paths.add(_resolve_repo_path(raw_dataset_dir))
        for raw_file in _as_list(data_source.get("raw_test_files")):
            if str(raw_file).strip():
                dataset_paths.add(_resolve_repo_path(raw_file))

    for group in _job_model_groups(job):
        model_root_raw = str(group.get("model_root", "") or "").strip()
        if not model_root_raw:
            errors.append("model group has empty model_root")
            continue
        model_root = _resolve_repo_path(model_root_raw)
        if not model_root.exists() or not model_root.is_dir():
            errors.append(f"model_root missing or not a directory: {model_root}")
            continue

        for model_name in _model_names_from_group(group, model_root):
            model_dir = model_root / model_name
            if not model_dir.exists():
                errors.append(f"model directory missing: {model_dir}")
                continue
            config_path = model_dir / "log" / "config.json"
            if config_path.exists():
                required_files.add(config_path.resolve())
            else:
                errors.append(f"model config missing: {config_path}")
            checkpoint_path = _representative_checkpoint(model_dir)
            if checkpoint_path is not None:
                required_files.add(checkpoint_path)
            else:
                errors.append(f"no .safetensors or *_full.pt checkpoint found under: {model_dir}")

    return required_files, dataset_paths, errors


def _git_untracked_required_files(required_files: set[Path]) -> list[Path]:
    rel_paths = sorted(
        _repo_relative(path)
        for path in required_files
        if not path.resolve().is_relative_to((REPO_ROOT / "bin" / "model").resolve())
    )
    if not rel_paths:
        return []

    result = subprocess.run(
        ["git", "ls-files", "--", *rel_paths],
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    tracked = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    return [REPO_ROOT / rel_path for rel_path in rel_paths if rel_path not in tracked]


def _run_smoke_test(args: argparse.Namespace, raw_ds_path: Path, parquet_file: Path) -> None:
    errors: list[str] = []
    required_files: set[Path] = {Path(__file__).resolve()}

    for path in [
        DEFAULT_JOBLIST_PATH,
        REPO_ROOT / "src" / "run_benchmarks.py",
        REPO_ROOT / "src" / "utils" / "data_processor" / "parquet_processor.py",
        REPO_ROOT / "src" / "utils" / "evaluations" / "trajectory_batch_job.py",
        REPO_ROOT / "src" / "utils" / "evaluations" / "uncertainty_batch_job.py",
        REPO_ROOT / "src" / "utils" / "evaluations" / "chunk_batch_job.py",
    ]:
        if path.exists():
            required_files.add(path.resolve())
        else:
            errors.append(f"required file missing: {path}")

    job: dict | None = None
    if DEFAULT_JOBLIST_PATH.exists():
        try:
            job = _read_joblist(DEFAULT_JOBLIST_PATH)
        except ValueError as exc:
            errors.append(str(exc))

    source_deps, source_warnings = _collect_local_source_dependencies(
        [
            Path(__file__).resolve(),
            REPO_ROOT / "src" / "run_benchmarks.py",
            REPO_ROOT / "src" / "utils" / "data_processor" / "parquet_processor.py",
            REPO_ROOT / "src" / "utils" / "evaluations" / "trajectory_batch_job.py",
            REPO_ROOT / "src" / "utils" / "evaluations" / "uncertainty_batch_job.py",
            REPO_ROOT / "src" / "utils" / "evaluations" / "chunk_batch_job.py",
        ]
    )
    required_files.update(source_deps)
    for warning in source_warnings:
        logging.warning("[smoke] %s", warning)

    dataset_paths = {parquet_file.resolve(), raw_ds_path.resolve()}
    if job is not None:
        job_required, job_dataset_paths, job_errors = _collect_job_prerequisites(job)
        required_files.update(job_required)
        dataset_paths.update(job_dataset_paths)
        errors.extend(job_errors)

    for data_path in sorted(dataset_paths):
        if not data_path.exists():
            errors.append(f"dataset path missing: {data_path}")

    if args.check_git_tracked:
        try:
            untracked = _git_untracked_required_files(required_files)
        except subprocess.CalledProcessError as exc:
            errors.append(f"git tracking check failed: {exc.stderr.strip() or exc}")
            untracked = []
        if untracked:
            errors.append(
                "required source files are not git-tracked:\n"
                + "\n".join(f"  - {_repo_relative(path)}" for path in untracked)
            )

    logging.info("[smoke] Selected parquet: %s", parquet_file)
    logging.info("[smoke] Required source/model files checked: %d", len(required_files))
    logging.info("[smoke] Dataset artifacts checked but excluded from git tracking: %d", len(dataset_paths))

    if errors:
        raise RuntimeError("BlogWatcher smoke test failed:\n" + "\n".join(f"- {error}" for error in errors))

    logging.info("[smoke] Static prerequisite check passed.")
    logging.info(
        "[smoke] No parquet processing, map cutting, Docker startup, or benchmark was executed."
    )


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

    if args.smoke_test:
        _run_smoke_test(args, raw_ds_path, parquet_file)
        return

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
