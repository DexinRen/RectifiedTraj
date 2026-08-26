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
JOB_SCRIPT = REPO_ROOT / "src" / "utils" / "evaluations" / "uncertainty_batch_job.py"


def _safe_name(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._@-]+", "_", str(value))
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "item"


def _task_type(spec: dict) -> str:
    token = str(spec.get("task_type", "learned_model") or "learned_model").strip().lower()
    if token not in {"learned_model", "classic_baseline"}:
        raise ValueError(f"Unsupported uncertainty batch task type: {token!r}")
    return token


def _task_label(spec: dict) -> str:
    if _task_type(spec) == "classic_baseline":
        return f"baseline:{spec.get('baseline_method', 'NA')}"
    return f"{spec.get('model_tag', 'model')}/{spec.get('model_name', 'NA')}"


def _config_label(spec: dict) -> str:
    if _task_type(spec) == "classic_baseline":
        return "baseline"
    cfg = dict(spec.get("manual_config") or {})
    return f"Q1={cfg.get('Q1', 'NA')} Q2={cfg.get('Q2', 'NA')}"


def _job_dir_name(spec: dict) -> str:
    dataset = _safe_name(spec.get("dataset_name", "dataset"))
    if _task_type(spec) == "classic_baseline":
        return "__".join([dataset, "Baseline", _safe_name(spec.get("baseline_method", "baseline"))])
    cfg = dict(spec.get("manual_config") or {})
    return "__".join(
        [
            dataset,
            _safe_name(spec.get("model_tag", "NA")),
            _safe_name(Path(str(spec.get("model_root", ""))).name or "model_root"),
            _safe_name(spec.get("model_name", "model")),
            f"Q1_{int(cfg.get('Q1', 0))}",
            f"Q2_{int(cfg.get('Q2', 0))}",
        ]
    )


def _progress_bar(finished: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "] 0/0 (0.0%)"
    ratio = max(0.0, min(1.0, float(finished) / float(total)))
    filled = int(width * ratio)
    return f"[{'#' * filled}{'-' * (width - filled)}] {finished}/{total} ({ratio * 100.0:5.1f}%)"


def _render_active_jobs(active_jobs: list[dict], *, finished_jobs: int, total_jobs: int, max_workers: int) -> str:
    lines = [
        "Uncertainty Evaluation Batch",
        f"{_progress_bar(finished_jobs, total_jobs)} | active {len(active_jobs)}/{max_workers}",
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
                    model=_task_label(job["spec"])[:18],
                    cfg=_config_label(job["spec"])[:21],
                    dataset=str(job["spec"].get("dataset_name", "NA"))[:28],
                    start=job["start_time"].strftime("%Y-%m-%d %H:%M"),
                )
            )
    lines.append("+----------------------------------------------------------------------------------------------------+")
    return "\n".join(lines)


def _emit_snapshot(active_jobs: list[dict], *, finished_jobs: int, total_jobs: int, max_workers: int) -> None:
    ProgressTracker._emit_log_message(
        sys.stdout,
        _render_active_jobs(
            active_jobs,
            finished_jobs=finished_jobs,
            total_jobs=total_jobs,
            max_workers=max_workers,
        ),
    )


def _launch_pending_jobs_until_full(
    *,
    pending_specs: list[dict],
    active_jobs: list[dict],
    specs_root: Path,
    jobs_root: Path,
    max_workers: int,
) -> None:
    while pending_specs and len(active_jobs) < int(max_workers):
        spec = pending_specs.pop(0)
        job_key = _job_dir_name(spec)
        job_dir = jobs_root / job_key
        job_dir.mkdir(parents=True, exist_ok=True)
        spec["output_dir"] = str(job_dir)
        spec_path = specs_root / f"{job_key}.json"
        spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        stdout_stream = (job_dir / "stdout.log").open("w", encoding="utf-8")
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


def _collect_job_outputs(
    *,
    job_dir: Path,
    job_key: str,
    uncertainty_results_dir: Path,
    uncertainty_p_val_dir: Path,
    valhalla_diagnostics_dir: Path,
) -> None:
    summary = job_dir / "uncertainty_result.csv"
    pval = job_dir / "uncertainty_p_val.csv"
    if summary.exists():
        shutil.copy2(summary, uncertainty_results_dir / f"{job_key}.csv")
    if pval.exists():
        shutil.copy2(pval, uncertainty_p_val_dir / f"{job_key}.csv")
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


def run_uncertainty_batch(
    *,
    manager,
    task_specs: list[dict],
    max_workers: int,
) -> None:
    if not task_specs:
        return

    batch_root = Path(manager.output_dir)
    jobs_root = batch_root / "uncertainty_jobs"
    specs_root = jobs_root / "specs"
    uncertainty_results_dir = batch_root / "uncertainty_results"
    uncertainty_p_val_dir = batch_root / "uncertainty_p_val"
    valhalla_diagnostics_dir = batch_root / "valhalla_meili_diagnostics"
    for path in [
        jobs_root,
        specs_root,
        uncertainty_results_dir,
        uncertainty_p_val_dir,
        valhalla_diagnostics_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Uncertainty evaluation batch start | jobs=%d parallel=%d output_root=%s",
        len(task_specs),
        int(max_workers),
        jobs_root,
    )
    pending_specs = list(task_specs)
    active_jobs: list[dict] = []
    finished_jobs = 0

    _launch_pending_jobs_until_full(
        pending_specs=pending_specs,
        active_jobs=active_jobs,
        specs_root=specs_root,
        jobs_root=jobs_root,
        max_workers=int(max_workers),
    )
    _emit_snapshot(active_jobs, finished_jobs=finished_jobs, total_jobs=len(task_specs), max_workers=int(max_workers))

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
                raise RuntimeError(
                    "Uncertainty evaluation batch job failed: "
                    f"{_task_label(job_info['spec'])} dataset={job_info['spec'].get('dataset_name')} "
                    f"exit={ret} log={job_info['job_dir'] / 'stdout.log'}"
                )

            _collect_job_outputs(
                job_dir=job_info["job_dir"],
                job_key=job_info["job_key"],
                uncertainty_results_dir=uncertainty_results_dir,
                uncertainty_p_val_dir=uncertainty_p_val_dir,
                valhalla_diagnostics_dir=valhalla_diagnostics_dir,
            )
            finished_jobs += 1
            _launch_pending_jobs_until_full(
                pending_specs=pending_specs,
                active_jobs=active_jobs,
                specs_root=specs_root,
                jobs_root=jobs_root,
                max_workers=int(max_workers),
            )
            _emit_snapshot(active_jobs, finished_jobs=finished_jobs, total_jobs=len(task_specs), max_workers=int(max_workers))

    aggregate_csv_folder(
        uncertainty_results_dir,
        uncertainty_results_dir / "aggregated.csv",
        skip_names={"aggregated.csv"},
    )
    aggregate_csv_folder(
        uncertainty_p_val_dir,
        uncertainty_p_val_dir / "aggregated.csv",
        skip_names={"aggregated.csv"},
    )
    summary_agg = uncertainty_results_dir / "aggregated.csv"
    pval_agg = uncertainty_p_val_dir / "aggregated.csv"
    if summary_agg.exists():
        shutil.copy2(summary_agg, batch_root / "uncertainty_band_summary.csv")
    if pval_agg.exists():
        shutil.copy2(pval_agg, batch_root / "uncertainty_traj_p_val.csv")
        generate_pairwise_p_value_report(
            batch_root / "uncertainty_traj_p_val.csv",
            batch_root / "uncertainty_p_value_summary",
            sample_type="uncertainty_trajectory",
            metric_column="pass_rate_points",
        )

    logging.info(
        "Uncertainty evaluation batch complete | jobs=%d summary_csv=%s pval_csv=%s",
        len(task_specs),
        summary_agg,
        pval_agg,
    )
