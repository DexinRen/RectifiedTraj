#!/usr/bin/env python3
"""Entry point for benchmark execution based on a JSON job list.

Purpose:
    Act as a benchmark lifter only.
    Parse eval_joblist, build phase packets, and dispatch evaluation phases.

Logic Chain:
    1. Load and normalize eval_joblist.
    2. Resolve explicit input datasets or generate them from parquet.
    3. Build the output run context.
    4. Dispatch baseline, trajectory, range, and chunk phases.
    5. Optionally upload the finished run to W&B.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from utils.evaluations.benchmark_inputs import (
    collect_missing_inputs_for_autogen,
    ensure_expected_test_paths_from_data_source,
    generate_full_traj,
    infer_dataset_name_from_path,
    load_metadata,
    preflight_validate_job,
    resolve_existing_dataset,
    run_data_generation_mode,
)
from utils.evaluations.benchmark_runtime import (
    apply_cpu_dataset_caps,
    apply_kalman_calibration_overrides,
    configure_encoder_decoder_device,
    resolve_traj_parallel,
)
from utils.evaluations.benchmark_schema import (
    as_list,
    build_job_list_from_group,
    dedupe_model_groups,
    normalize_job_schema,
    normalize_model_group_schema_entry,
    resolve_classic_baselines,
)
from utils.evaluations.bounded_runner import run_bounded_eval_batch
from utils.evaluations.evaluation_manager import TestManager
from utils.evaluations.progress import ProgressAwareStreamHandler
from utils.evaluations.run_context import (
    build_result_folder_name,
    stage,
    write_system_info,
)
from utils.evaluations.trajectory_batch_runner import (
    build_classic_baseline_task_specs,
    build_trajectory_task_specs,
)
from utils.evaluations.wandb_logger import log_run_to_wandb
from utils.data_visualizer.make_heatmaps import generate_run_heatmaps
from utils.data_processor.dataset_facts import build_dataset_facts, summarize_trajectory_file


FULL_TRAJ_DIR = Path("./dataset/processed/NUMOSIM_Kanto/test/traj_test")
DEBUG_FULL_TRAJ = FULL_TRAJ_DIR / "traj_debug_mini.pt"
JOBLIST_PATH = Path("./src/eval_joblist.json")


# ================================================================
# === Phase Packet Builders
# ================================================================
def build_group_runs(job: dict) -> list[dict]:
    """Build one compact run packet per learned-model group."""
    model_groups = list(job.get("model_groups") or [])
    if not model_groups:
        job["model_groups"] = []
        return []

    normalized_groups: list[dict] = []
    for idx, raw_group in enumerate(model_groups):
        group = normalize_model_group_schema_entry(
            raw_group,
            default_group=None,
            context=f"model_groups[{idx}]",
        )
        model_names = group.get("model_names")
        if model_names is not None and not model_names:
            group["model_names"] = None
        normalized_groups.append(group)

    model_groups = dedupe_model_groups(normalized_groups)
    if not model_groups:
        job["model_groups"] = []
        return []
    job["model_groups"] = model_groups

    primary_group = model_groups[0]
    job["data_hypothesis"] = primary_group["data_hypothesis"]
    job["model_root"] = primary_group["model_root"]
    job["model_names"] = primary_group["model_names"]
    job["Q1"] = list(primary_group["Q1"])
    job["Q2"] = list(primary_group["Q2"])
    job["denoise_steps"] = list(primary_group.get("denoise_steps") or [None])
    job["sample_steps"] = list(primary_group.get("sample_steps") or [None])

    out: list[dict] = []
    for group in model_groups:
        out.append(
            {
                "group": group,
                "job_list": build_job_list_from_group(group),
            }
        )
    return out


def resolve_explicit_test_dirs(job: dict) -> tuple[list[str], list[str]]:
    """Resolve the explicit trajectory/chunk test paths from the normalized job."""
    traj_paths = job.get("traj_paths", {}) or {}
    traj_dirs = [str(path).strip() for path in as_list(job.get("traj_dirs")) if str(path).strip()]
    if not traj_dirs and traj_paths.get("full_traj"):
        traj_dirs = [str(traj_paths.get("full_traj")).strip()]

    chunk_dirs = [str(path).strip() for path in as_list(job.get("chunk_dirs")) if str(path).strip()]
    if not chunk_dirs:
        chunk_fallback = str(job.get("chunk_test_dir", "") or "").strip()
        if chunk_fallback:
            chunk_dirs = [chunk_fallback]

    return ensure_expected_test_paths_from_data_source(job, traj_dirs, chunk_dirs)


def collect_trajectory_datasets(
    job: dict,
    traj_dirs: list[str],
    *,
    use_debug: bool,
) -> list[dict]:
    """Resolve trajectory dataset packets for baseline and learned trajectory phases."""
    datasets: list[dict] = []
    for idx, path_value in enumerate(traj_dirs):
        entries = resolve_existing_dataset(
            name=f"traj_{idx}",
            path_value=path_value,
            debug_path=DEBUG_FULL_TRAJ if (use_debug and idx == 0) else None,
            use_debug=bool(use_debug and idx == 0),
        )
        for file_idx, entry in enumerate(entries):
            dataset_family = infer_dataset_name_from_path(entry[0]) or infer_dataset_name_from_path(path_value)
            dataset_stem = Path(entry[0]).stem or Path(path_value).stem or f"traj_{idx}_{file_idx}"
            if dataset_family:
                dataset_name = f"{dataset_family}_{dataset_stem}"
            else:
                dataset_name = dataset_stem
            datasets.append(
                {
                    "name": dataset_name,
                    "path": entry[0],
                    "M": entry[1],
                    "N": entry[2],
                }
            )

    if datasets or not bool(job.get("gen_new_test", False)) or use_debug:
        return datasets

    stage("Generating full_traj dataset from raw parquet input")
    traj_paths = job.get("traj_paths", {}) or {}
    use_new_traj = job.get("use_new_traj", {}) or {}
    output_dir = FULL_TRAJ_DIR
    if traj_paths.get("full_traj"):
        full_path = Path(traj_paths["full_traj"])
        output_dir = full_path if full_path.suffix == "" else full_path.parent

    pt_path = generate_full_traj(output_dir, use_new_traj)
    meta = load_metadata(pt_path)
    datasets.append(
        {
            "name": "full_traj",
            "path": pt_path,
            "M": int(meta["n_trajectories"]),
            "N": int(meta["median_length"]),
        }
    )
    return datasets


def interleave_task_specs(*task_groups: list[dict]) -> list[dict]:
    """Round-robin multiple task groups into one unified waiting queue."""
    queues = [list(group) for group in task_groups if group]
    merged: list[dict] = []
    while queues:
        next_queues: list[list[dict]] = []
        for queue in queues:
            if not queue:
                continue
            merged.append(queue.pop(0))
            if queue:
                next_queues.append(queue)
        queues = next_queues
    return merged


def _append_dataset_facts_for_path(
    snapshot: dict,
    path_value: str,
    *,
    source: str,
) -> None:
    """Append dataset facts for one file or directory to a run-local snapshot."""
    path = Path(str(path_value))
    if not path.exists():
        snapshot.setdefault("missing", []).append({"path": str(path_value), "source": source})
        return

    try:
        if path.is_file():
            kind, summary = summarize_trajectory_file(path, kind="auto")
            snapshot.setdefault(kind, {})[summary["file"]] = summary
        else:
            facts = build_dataset_facts(path)
            for kind in ("exact", "uncertainty"):
                entries = facts.get(kind, {})
                if isinstance(entries, dict):
                    snapshot.setdefault(kind, {}).update(entries)
    except Exception as exc:
        snapshot.setdefault("errors", []).append(
            {
                "path": str(path_value),
                "source": source,
                "error": str(exc),
            }
        )


def write_used_dataset_facts(
    output_dir: str | Path,
    *,
    datasets: list[dict],
    job: dict,
) -> Path:
    """Write facts for trajectory/uncertainty datasets used by this run."""
    snapshot: dict = {
        "version": 1,
        "generated_at": datetime.now().isoformat(),
        "units": {
            "sample_time": "seconds",
            "point_per_traj": "points",
            "error_per_point_l1": "meters",
            "distance_to_ref_l1": "meters",
            "radius": "meters",
        },
        "exact": {},
        "uncertainty": {},
        "missing": [],
        "errors": [],
    }

    seen: set[str] = set()

    def _add(path_value: str | None, source: str) -> None:
        if not path_value:
            return
        token = str(path_value).strip()
        if not token or token in seen:
            return
        seen.add(token)
        _append_dataset_facts_for_path(snapshot, token, source=source)

    for dataset in datasets:
        _add(str(dataset.get("path", "") or ""), "trajectory_dataset")

    if bool(job.get("range_test", False)):
        for path_value in as_list(job.get("test_data_paths")):
            _add(str(path_value), "uncertainty_dataset")
        _add(str(job.get("test_data_path", "") or ""), "uncertainty_dataset")

    out_path = Path(output_dir) / "fact_used_dataset.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


# ================================================================
# === Phase Dispatchers
# ================================================================
def run_trajectory_phase(
    manager: TestManager,
    job: dict,
    group_runs: list[dict],
    datasets: list[dict],
    classic_baselines: list[str],
    *,
    log_level_name: str,
) -> None:
    """Run one unified trajectory evaluation queue for baselines and learned models."""
    if not bool(job.get("traj_test", True)):
        return
    if not datasets:
        return

    stage("Trajectory evaluation phase start")
    traj_parallel = resolve_traj_parallel(job)
    baseline_tasks: list[dict] = []
    if bool(job.get("run_baseline", job.get("baseline_once", True))) and classic_baselines:
        baseline_tasks = build_classic_baseline_task_specs(
            dataset_entries=datasets,
            classic_baselines=classic_baselines,
            baseline_options=dict(job.get("baseline_options") or {}),
            log_level=log_level_name,
        )
    manager.classic_baseline_evaluator.progress_bar = bool(job.get("baseline_progress", True))
    trajectory_tasks = build_trajectory_task_specs(
        manager=manager,
        group_runs=group_runs,
        dataset_entries=datasets,
        log_level=log_level_name,
    )
    all_tasks = interleave_task_specs(baseline_tasks, trajectory_tasks)
    if not all_tasks:
        return
    stage(
        "Unified trajectory task list resolved | total=%d learned=%d baselines=%d datasets=%d parallel=%d"
        % (len(all_tasks), len(trajectory_tasks), len(baseline_tasks), len(datasets), traj_parallel)
    )
    manager.run_trajectory_batch(
        task_specs=all_tasks,
        max_workers=traj_parallel,
    )


def run_range_phase(
    manager: TestManager,
    job: dict,
    group_runs: list[dict],
    classic_baselines: list[str],
    *,
    log_level_name: str,
) -> None:
    """Run bounded/range evaluation for all learned-model groups."""
    if not bool(job.get("range_test", False)):
        return

    stage("Range test phase start")
    run_bounded_eval_batch(
        manager=manager,
        job=job,
        group_runs=group_runs,
        classic_baselines=classic_baselines,
        log_level=log_level_name,
    )


def run_chunk_phase(
    manager: TestManager,
    job: dict,
    group_runs: list[dict],
    classic_baselines: list[str],
    *,
    log_level_name: str,
) -> None:
    """Run chunk evaluation for all learned-model groups."""
    if not bool(job.get("chunk_test", False)):
        return

    stage("Chunk test phase start")
    chunk_parallel = resolve_traj_parallel(job)
    for run_idx, run_item in enumerate(group_runs):
        group = run_item["group"]
        run_baselines_here = run_idx == 0
        stage(
            "Chunk learned group start | idx=%d hypothesis=%s model_root=%s run_baselines=%s"
            % (
                run_idx,
                group["data_hypothesis"],
                group["model_root"],
                run_baselines_here,
            )
        )
        manager.run_chunk_batch(
            job=job,
            model_root=group["model_root"],
            model_names=group.get("model_names"),
            classic_baselines=classic_baselines,
            model_tag=group["data_hypothesis"],
            run_baselines=run_baselines_here,
            max_workers=chunk_parallel,
            log_level=log_level_name,
        )


# ================================================================
# === Main
# ================================================================
def main() -> None:
    """Run the benchmark job described by the selected JSON job list."""
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
    parser.add_argument(
        "--job-list",
        type=Path,
        default=JOBLIST_PATH,
        help="Evaluation job-list JSON (default: src/eval_joblist.json)",
    )
    args = parser.parse_args()

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    handler = ProgressAwareStreamHandler(stream=sys.stdout)
    handler.setLevel(logging.NOTSET)
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger.addHandler(handler)

    joblist_path = args.job_list
    if not joblist_path.exists():
        raise FileNotFoundError(f"Missing job list: {joblist_path}")

    with joblist_path.open("r", encoding="utf-8") as file_obj:
        job_raw = json.load(file_obj)

    job = normalize_job_schema(job_raw)
    runtime_device = configure_encoder_decoder_device(job)
    stage(f"RectifiedTraj runtime.device: {runtime_device}")
    kalman_mode, kalman_dataset = apply_kalman_calibration_overrides(job)
    stage(f"Kalman calibration mode: {kalman_mode} (dataset source: {kalman_dataset})")
    stage(f"Test type: {job.get('test_type', 'exact')}")

    group_runs = build_group_runs(job)
    if not group_runs and (
        not bool(job.get("run_baseline", False))
        or bool(job.get("range_test", False))
        or bool(job.get("chunk_test", False))
    ):
        raise ValueError(
            "No learned model_groups are configured. "
            "Provide explicit model_groups (or rectifiedtraj/directreg blocks) for learned evaluation, "
            "or run trajectory baselines only with run_baseline=true and chunk_test/range_test disabled."
        )

    log_level_name = str(job.get("log_level", "INFO")).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logging.getLogger().setLevel(log_level)
    stage(f"Log level set to {logging.getLevelName(log_level)}")

    classic_baselines = resolve_classic_baselines(job)
    stage(f"Classic baselines selected: {classic_baselines if classic_baselines else '[]'}")
    for idx, run_item in enumerate(group_runs):
        group = run_item["group"]
        stage(
            "Learned model group[%d] | hypothesis=%s model_root=%s models=%s"
            % (
                idx,
                group["data_hypothesis"],
                group["model_root"],
                group.get("model_names"),
            )
        )

    traj_dirs, chunk_dirs = resolve_explicit_test_dirs(job)
    missing_inputs = collect_missing_inputs_for_autogen(
        job,
        traj_dirs=traj_dirs,
        chunk_dirs=chunk_dirs,
    )
    if missing_inputs:
        if not bool(job.get("gen_new_test", False)):
            raise FileNotFoundError(
                "Invalid eval_joblist input: %s. "
                "Provide existing paths or enable gen_new_test."
                % "; ".join(missing_inputs)
            )
        stage("Generating evaluation datasets from raw parquet input")
        traj_dirs, chunk_dirs = run_data_generation_mode(job)
        stage(f"Data generation finished | traj={traj_dirs} chunk={chunk_dirs}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_folder_name = build_result_folder_name(
        job,
        traj_dirs,
        chunk_dirs,
        runtime_device=runtime_device,
        timestamp=timestamp,
    )
    manager = TestManager(output_dir=str(Path("./bin/test_results") / result_folder_name))
    manager.brief_summary = job.get("brief_summary", True)
    manager.brief_visualizer = job.get("brief_visualizer", True)
    manager.visualize_each_run = False

    job_copy_path = Path(manager.output_dir) / "eval_joblist.json"
    job_copy_path.parent.mkdir(parents=True, exist_ok=True)
    job_copy_path.write_text(json.dumps(job_raw, indent=2), encoding="utf-8")
    stage(f"Saved eval joblist snapshot: {job_copy_path}")

    system_info_path = write_system_info(Path(manager.output_dir), runtime_device)
    stage(f"Saved system info snapshot: {system_info_path}")

    stage("Preflight validation start")
    preflight_validate_job(
        job,
        model_groups=job["model_groups"],
        traj_dirs=traj_dirs,
        chunk_dirs=chunk_dirs,
        classic_baselines=classic_baselines,
    )
    stage("Preflight validation passed")

    datasets = collect_trajectory_datasets(
        job,
        traj_dirs,
        use_debug=bool(args.test),
    )
    datasets = apply_cpu_dataset_caps(job, datasets)
    fact_used_path = write_used_dataset_facts(
        manager.output_dir,
        datasets=datasets,
        job=job,
    )
    stage(f"Saved used dataset facts snapshot: {fact_used_path}")
    if bool(job.get("traj_test", True)) and not datasets and not bool(job.get("range_test", False)):
        raise ValueError(
            "No valid trajectory datasets found. Provide test_files.traj_files or enable gen_new_test."
        )

    run_trajectory_phase(
        manager,
        job,
        group_runs,
        datasets,
        classic_baselines,
        log_level_name=log_level_name,
    )
    run_range_phase(manager, job, group_runs, classic_baselines, log_level_name=log_level_name)
    run_chunk_phase(manager, job, group_runs, classic_baselines, log_level_name=log_level_name)

    stage("Post-aggregation heatmap generation start")
    try:
        heatmap_counts = generate_run_heatmaps(Path(manager.output_dir))
        stage(
            "Post-aggregation heatmaps finished | trajectory=%d chunk_point=%d chunk_byte=%d"
            % (
                int(heatmap_counts.get("trajectory_pointwise", 0)),
                int(heatmap_counts.get("chunk_pointwise", 0)),
                int(heatmap_counts.get("chunk_bytewise", 0)),
            )
        )
    except Exception as exc:
        logging.warning("Post-aggregation heatmap generation failed: %s", exc)

    wandb_enabled = bool(job.get("wandb", False)) or bool(args.wandb)
    if wandb_enabled:
        wandb_project = args.wandb_project or job.get("wandb_project", "rectifiedtraj_benchmarks")
        wandb_entity = args.wandb_entity or job.get("wandb_entity") or None
        wandb_run_name = args.wandb_run_name or job.get("wandb_run_name") or Path(manager.output_dir).name
        try:
            log_run_to_wandb(
                run_dir=manager.output_dir,
                project=wandb_project,
                entity=wandb_entity,
                run_name=wandb_run_name,
            )
        except Exception as exc:
            logging.warning("W&B upload failed: %s", exc)

    print("\n✓ Evaluation complete")


if __name__ == "__main__":
    main()
