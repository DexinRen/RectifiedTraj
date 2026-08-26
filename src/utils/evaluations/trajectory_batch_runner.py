import json
import logging
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from utils.evaluations.progress import ProgressTracker
from utils.evaluations.result_io import aggregate_csv_folder
from utils.evaluations.p_value import generate_pairwise_p_value_report


REPO_ROOT = Path(__file__).resolve().parents[3]
JOB_SCRIPT = REPO_ROOT / "src" / "utils" / "evaluations" / "trajectory_batch_job.py"
VALHALLA_COMPOSE_FILE = (
    REPO_ROOT / "src" / "baseline" / "models" / "valhalla_meili" / "docker-compose.yml"
)
VALHALLA_MAP_TOOLS_DOCKERFILE = (
    REPO_ROOT / "src" / "baseline" / "models" / "valhalla_meili" / "Dockerfile.map_tools"
)


def resolve_valhalla_profile(dataset_name: str, baseline_options: dict) -> dict:
    """
    Purpose:
        Resolve one dataset-specific Valhalla profile into a full configuration.
    Parameters:
        dataset_name (str), concrete evaluation dataset label.
        baseline_options (dict), normalized baseline.options object.
    Return Dict:
        "error_code": int, 0 for one unambiguous profile.
        "config": dict, complete ValhallaMeiliBaselineModel configuration.
        "profile_name": str, matched dataset profile key.
    Usage:
        build_classic_baseline_task_specs prepares isolated child-job configs.
    TODO:
        1) Require a Valhalla profile table.
        2) Match exactly one dataset prefix.
        3) Require map, source, costing, and port selections.
        4) Expand the approved fixed benchmark settings.
        5) Prepare the tailored map before child jobs are launched.
    """

    # 1. Require Profile Table
    options = baseline_options.get("valhalla_meili")
    if not isinstance(options, dict):
        raise ValueError(
            "Selecting valhalla_meili requires baseline.options.valhalla_meili."
        )
    profiles = options.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("valhalla_meili.profiles must be a non-empty JSON object.")

    # 2. Match Exactly One Dataset Prefix
    dataset_token = str(dataset_name).strip()
    matches = [
        str(profile_name)
        for profile_name in profiles
        if dataset_token == str(profile_name)
        or dataset_token.startswith(str(profile_name) + "_")
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Dataset {dataset_token!r} must match exactly one Valhalla profile; "
            f"matched {matches!r}."
        )
    profile_name = matches[0]
    profile = profiles[profile_name]
    if not isinstance(profile, dict):
        raise ValueError(f"Valhalla profile {profile_name!r} must be a JSON object.")

    # 3. Require Dataset-Specific Choices
    required = {"map_id", "source", "costing", "port"}
    missing = sorted(required - set(profile.keys()))
    if missing:
        raise ValueError(
            f"Valhalla profile {profile_name!r} is missing: {', '.join(missing)}"
        )
    port = int(profile["port"])
    source = str(profile["source"]).strip().lower()
    if source not in {"japan", "georgia"}:
        raise ValueError(
            f"Valhalla profile {profile_name!r} source must be 'japan' or 'georgia'."
        )

    # 4. Expand Approved Fixed Settings
    config = {
        "base_url": f"http://127.0.0.1:{port}",
        "costing": str(profile["costing"]),
        "shape_match": "map_snap",
        "window_points": 500,
        "overlap_points": 50,
        "timeout_sec": 60.0,
        "auto_start": True,
        "map_id": str(profile["map_id"]),
        "compose_file": str(VALHALLA_COMPOSE_FILE),
        "map_tools_dockerfile": str(VALHALLA_MAP_TOOLS_DOCKERFILE),
        "raw_map_root": str(REPO_ROOT / "dataset" / "raw" / "map"),
        "processed_map_root": str(REPO_ROOT / "dataset" / "processed" / "map"),
        "state_file": str(
            REPO_ROOT / "dataset" / "state" / f"state_{str(profile['map_id']).strip()}.json"
        ),
        "source": source,
        "buffer_km": 1.0,
        "port": port,
        "startup_timeout_sec": 120.0,
        "build_timeout_sec": 7200.0,
    }
    from baseline.models.valhalla_meili.map_tools import ensure_dataset_map
    from baseline.models.valhalla_meili.model import validate_valhalla_config

    config = validate_valhalla_config(config)["config"]
    preparation = ensure_dataset_map(config)
    return {
        "error_code": 0,
        "config": config,
        "profile_name": profile_name,
        "map_preparation": preparation,
    }


def spec_task_type(spec: dict) -> str:
    token = str(spec.get("task_type", "learned_model") or "learned_model").strip().lower()
    if token not in {"learned_model", "classic_baseline"}:
        raise ValueError(f"Unsupported trajectory batch task type: {token!r}")
    return token


def safe_name(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._@-]+", "_", str(value))
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "item"


def task_display_label(spec: dict) -> str:
    if spec_task_type(spec) == "classic_baseline":
        return f"baseline:{str(spec.get('baseline_method', 'NA'))}"
    model_tag = str(spec.get("model_tag", "model") or "model")
    model_name = str(spec.get("model_name", "NA") or "NA")
    return f"{model_tag}/{model_name}"


def task_config_label(spec: dict) -> str:
    if spec_task_type(spec) == "classic_baseline":
        return "baseline"
    cfg = dict(spec.get("manual_config") or {})
    label = "Q1={q1} Q2={q2}".format(
        q1=cfg.get("Q1", "NA"),
        q2=cfg.get("Q2", "NA"),
    )
    if cfg.get("denoise_steps") is not None:
        label = f"{label} steps={cfg['denoise_steps']}"
    if cfg.get("sample_steps") is not None:
        label = f"{label} sample={cfg['sample_steps']}"
    return label


def progress_bar(finished: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "] 0/0 (0.0%)"
    ratio = max(0.0, min(1.0, float(finished) / float(total)))
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {finished}/{total} ({ratio * 100.0:5.1f}%)"


def render_active_jobs(
    active_jobs: list[dict],
    *,
    finished_jobs: int,
    total_jobs: int,
    max_workers: int,
) -> str:
    lines = [
        "Trajectory Evaluation Batch",
        f"{progress_bar(finished_jobs, total_jobs)} | active {len(active_jobs)}/{max_workers}",
        "+----------------------------------------------------------------------------------------------------+",
        "| Task               | Config                | Dataset                      | Start Time           |",
        "+----------------------------------------------------------------------------------------------------+",
    ]
    if not active_jobs:
        lines.append("| <idle>             | <none>                | <none>                       | <none>               |")
    else:
        for job in active_jobs:
            lines.append(
                "| {model:<18} | {cfg:<21} | {dataset:<28} | {start:<20} |".format(
                    model=task_display_label(job["spec"])[:18],
                    cfg=task_config_label(job["spec"])[:21],
                    dataset=str(job["spec"].get("dataset_name", "NA"))[:28],
                    start=job["start_time"].strftime("%Y-%m-%d %H:%M"),
                )
            )
    lines.append("+----------------------------------------------------------------------------------------------------+")
    return "\n".join(lines)


def emit_snapshot(active_jobs: list[dict], *, finished_jobs: int, total_jobs: int, max_workers: int) -> None:
    ProgressTracker._emit_log_message(
        sys.stdout,
        render_active_jobs(
            active_jobs,
            finished_jobs=finished_jobs,
            total_jobs=total_jobs,
            max_workers=max_workers,
        ),
    )


def job_dir_name(spec: dict) -> str:
    if spec_task_type(spec) == "classic_baseline":
        return "__".join(
            [
                safe_name(spec.get("dataset_name", "dataset")),
                "Baseline",
                safe_name(spec.get("baseline_method", "baseline")),
            ]
        )
    manual_config = dict(spec["manual_config"])
    q1 = int(manual_config["Q1"])
    q2 = int(manual_config["Q2"])
    denoise_steps = manual_config.get("denoise_steps")
    sample_steps = manual_config.get("sample_steps")
    dataset_label = str(spec.get("dataset_name", "")).strip()
    if not dataset_label:
        test_data_path = Path(str(spec.get("test_data_path", "")))
        dataset_label = test_data_path.stem or "dataset"
    parts = [
        safe_name(dataset_label),
        safe_name(spec.get("model_tag", "NA")),
        safe_name(Path(str(spec.get("model_root", ""))).name or "model_root"),
        safe_name(spec["model_name"]),
        f"Q1_{q1}",
        f"Q2_{q2}",
    ]
    if denoise_steps is not None:
        parts.append(f"steps_{int(denoise_steps)}")
    if sample_steps is not None:
        parts.append(f"sample_{int(sample_steps)}")
    return "__".join(parts)


def collect_job_outputs(
    *,
    job_dir: Path,
    job_key: str,
    traj_results_dir: Path,
    pw_result_dir: Path,
    traj_p_val_dir: Path,
    valhalla_diagnostics_dir: Path,
) -> None:
    traj_src = job_dir / "traj_result.csv"
    pw_src = job_dir / "pw_result.csv"
    pval_src = job_dir / "traj_p_val.csv"
    if traj_src.exists():
        shutil.copy2(traj_src, traj_results_dir / f"{job_key}.csv")
    if pw_src.exists():
        shutil.copy2(pw_src, pw_result_dir / f"{job_key}.csv")
    if pval_src.exists():
        shutil.copy2(pval_src, traj_p_val_dir / f"{job_key}.csv")
    for filename in (
        "valhalla_meili_summary.json",
        "valhalla_meili_error_codes.csv",
        "valhalla_meili_requests.jsonl",
    ):
        source = job_dir / filename
        if source.exists():
            suffix = "".join(Path(filename).suffixes)
            stem = filename[: -len(suffix)] if suffix else filename
            shutil.copy2(
                source,
                valhalla_diagnostics_dir / f"{job_key}__{stem}{suffix}",
            )


def build_trajectory_task_specs(
    *,
    manager,
    group_runs: list[dict],
    dataset_entries: list[dict],
    log_level: str = "INFO",
) -> list[dict]:
    """Build one flat learned-trajectory task list across all model groups.

    Purpose:
        Resolve every learned trajectory evaluation item into one flat task list.
    Parameters:
        manager (TestManager), used only for model discovery when a group omits model_names.
        group_runs (list[dict]), normalized learned-model groups with job_list packets.
        dataset_entries (list[dict]), resolved trajectory dataset packets.
        log_level (str), logging level copied into each child-job spec.
    Return Dict:
        Not used. Returns a flat list of task spec dicts.
    Usage:
        Called by run_benchmarks before launching the trajectory batch runner.
    TODO:
        1) Iterate normalized learned-model groups.
        2) Resolve model names for each group.
        3) Expand datasets x models x Q1 x Q2 x methods into flat specs.
        4) Return one combined task list for one batch run.
    """
    task_specs: list[dict] = []
    for run_item in group_runs:
        group = dict(run_item["group"])
        job_list = dict(run_item["job_list"])
        resolved_model_names = group.get("model_names") or manager._discover_models(group["model_root"])
        q1_vals = list(job_list.get("Q1") or [1])
        q2_vals = list(job_list.get("Q2") or [12])
        denoise_step_vals = list(job_list.get("denoise_steps") or [None])
        sample_step_vals = list(job_list.get("sample_steps") or [None])

        for entry in dataset_entries:
            for model_name in resolved_model_names:
                for q1 in q1_vals:
                    for q2 in q2_vals:
                        for denoise_steps in denoise_step_vals:
                            for sample_steps in sample_step_vals:
                                manual_config = {
                                    "Q1": int(q1),
                                    "Q2": int(q2),
                                }
                                if denoise_steps is not None:
                                    step_count = int(denoise_steps)
                                    manual_config["denoise_steps"] = step_count
                                    manual_config["t_delta"] = 1.0 / float(step_count)
                                if sample_steps is not None:
                                    manual_config["sample_steps"] = int(sample_steps)
                                task_specs.append(
                                    {
                                        "task_type": "learned_model",
                                        "dataset_name": str(entry["name"]),
                                        "test_data_path": str(entry["path"]),
                                        "M": int(entry["M"]),
                                        "N": int(entry["N"]),
                                        "model_name": str(model_name),
                                        "model_root": str(group["model_root"]),
                                        "model_tag": str(group["data_hypothesis"]),
                                        "manual_config": manual_config,
                                        "log_level": str(log_level).upper(),
                                    }
                                )
    return task_specs


def build_classic_baseline_task_specs(
    *,
    dataset_entries: list[dict],
    classic_baselines: list[str],
    baseline_options: dict,
    log_level: str = "INFO",
) -> list[dict]:
    task_specs: list[dict] = []
    for entry in dataset_entries:
        for method in classic_baselines:
            baseline_config = None
            if str(method).strip().lower() == "valhalla_meili":
                baseline_config = resolve_valhalla_profile(
                    str(entry["name"]),
                    baseline_options,
                )["config"]
            task_specs.append(
                {
                    "task_type": "classic_baseline",
                    "dataset_name": str(entry["name"]),
                    "test_data_path": str(entry["path"]),
                    "M": int(entry["M"]),
                    "N": int(entry["N"]),
                    "baseline_method": str(method),
                    "baseline_config": baseline_config,
                    "log_level": str(log_level).upper(),
                }
            )
    return task_specs


def launch_pending_jobs_until_full(
    *,
    pending_specs: list[dict],
    active_jobs: list[dict],
    specs_root: Path,
    jobs_root: Path,
    max_workers: int,
) -> None:
    while pending_specs and len(active_jobs) < int(max_workers):
        spec = pending_specs.pop(0)
        job_key = job_dir_name(spec)
        job_dir = jobs_root / job_key
        job_dir.mkdir(parents=True, exist_ok=True)
        spec["output_dir"] = str(job_dir)
        spec_path = specs_root / f"{job_key}.json"
        spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        stdout_path = job_dir / "stdout.log"
        stdout_stream = stdout_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, str(JOB_SCRIPT), "--spec-json", str(spec_path)],
            cwd=str(REPO_ROOT),
            stdout=stdout_stream,
            stderr=subprocess.STDOUT,
        )
        active_jobs.append(
            {
                "proc": proc,
                "spec": spec,
                "job_key": job_key,
                "job_dir": job_dir,
                "stdout_stream": stdout_stream,
                "start_time": datetime.now(),
            }
        )


def run_trajectory_batch(
    *,
    manager,
    task_specs: list[dict],
    max_workers: int,
) -> None:
    if not task_specs:
        return

    batch_root = Path(manager.output_dir)
    jobs_root = batch_root / "trajectory_jobs"
    specs_root = jobs_root / "specs"
    traj_results_dir = batch_root / "traj_results"
    pw_result_dir = batch_root / "pw_result"
    traj_p_val_dir = batch_root / "traj_p_val"
    valhalla_diagnostics_dir = batch_root / "valhalla_meili_diagnostics"
    for path in [
        jobs_root,
        specs_root,
        traj_results_dir,
        pw_result_dir,
        traj_p_val_dir,
        valhalla_diagnostics_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Trajectory evaluation batch start | jobs=%d parallel=%d output_root=%s",
        len(task_specs),
        int(max_workers),
        jobs_root,
    )

    pending_specs = list(task_specs)
    active_jobs: list[dict] = []
    finished_jobs = 0

    launch_pending_jobs_until_full(
        pending_specs=pending_specs,
        active_jobs=active_jobs,
        specs_root=specs_root,
        jobs_root=jobs_root,
        max_workers=int(max_workers),
    )

    emit_snapshot(
        active_jobs,
        finished_jobs=finished_jobs,
        total_jobs=len(task_specs),
        max_workers=int(max_workers),
    )

    while pending_specs or active_jobs:
        time.sleep(0.2)
        for job_info in list(active_jobs):
            ret = job_info["proc"].poll()
            if ret is None:
                continue

            job_info["stdout_stream"].close()
            active_jobs.remove(job_info)
            if int(ret) != 0:
                for other in active_jobs:
                    other["proc"].terminate()
                    other["stdout_stream"].close()
                log_hint = job_info["job_dir"] / "stdout.log"
                task_type = spec_task_type(job_info["spec"])
                if task_type == "classic_baseline":
                    task_label = (
                        f"baseline={job_info['spec']['baseline_method']} "
                        f"dataset={job_info['spec']['dataset_name']}"
                    )
                else:
                    task_label = (
                        f"model={job_info['spec']['model_name']} "
                        f"dataset={job_info['spec']['dataset_name']}"
                    )
                raise RuntimeError(
                    f"Trajectory evaluation batch job failed: {task_label} exit={ret} log={log_hint}"
                )

            collect_job_outputs(
                job_dir=job_info["job_dir"],
                job_key=job_info["job_key"],
                traj_results_dir=traj_results_dir,
                pw_result_dir=pw_result_dir,
                traj_p_val_dir=traj_p_val_dir,
                valhalla_diagnostics_dir=valhalla_diagnostics_dir,
            )
            finished_jobs += 1

            launch_pending_jobs_until_full(
                pending_specs=pending_specs,
                active_jobs=active_jobs,
                specs_root=specs_root,
                jobs_root=jobs_root,
                max_workers=int(max_workers),
            )

            emit_snapshot(
                active_jobs,
                finished_jobs=finished_jobs,
                total_jobs=len(task_specs),
                max_workers=int(max_workers),
            )

    logging.info("Trajectory evaluation batch workers finished | aggregating per-job CSVs")
    aggregate_csv_folder(
        traj_results_dir,
        traj_results_dir / "aggregated.csv",
        skip_names={"aggregated.csv"},
    )
    aggregate_csv_folder(
        pw_result_dir,
        pw_result_dir / "aggregated.csv",
        skip_names={"aggregated.csv"},
    )
    aggregate_csv_folder(
        traj_p_val_dir,
        traj_p_val_dir / "aggregated.csv",
        skip_names={"aggregated.csv"},
    )
    traj_agg = traj_results_dir / "aggregated.csv"
    pw_agg = pw_result_dir / "aggregated.csv"
    pval_agg = traj_p_val_dir / "aggregated.csv"
    if traj_agg.exists():
        shutil.copy2(traj_agg, batch_root / "trajectory_evaluation_summary.csv")
    if pw_agg.exists():
        shutil.copy2(pw_agg, batch_root / "trajectory_pointwise_summary.csv")
    if pval_agg.exists():
        shutil.copy2(pval_agg, batch_root / "traj_p_val.csv")
        generate_pairwise_p_value_report(
            batch_root / "traj_p_val.csv",
            batch_root / "traj_p_value_summary",
            sample_type="trajectory",
            metric_column="mean_l2_err",
            report_label="mean_error",
        )
        generate_pairwise_p_value_report(
            batch_root / "traj_p_val.csv",
            batch_root / "traj_p_value_summary",
            sample_type="trajectory",
            metric_column="tail_mean_l2_err",
            report_label="tail_error",
        )

    logging.info(
        "Trajectory evaluation batch complete | jobs=%d traj_csv=%s pw_csv=%s pval_csv=%s",
        len(task_specs),
        traj_agg,
        pw_agg,
        pval_agg,
    )
