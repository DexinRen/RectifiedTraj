import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from utils.evaluations.progress import ProgressTracker
from utils.evaluations.result_io import aggregate_csv_folder
from utils.evaluations.benchmark_inputs import infer_dataset_name_from_path
from utils.evaluations.p_value import generate_pairwise_p_value_report
from utils.evaluations.trajectory_batch_runner import resolve_valhalla_profile


REPO_ROOT = Path(__file__).resolve().parents[3]
JOB_SCRIPT = REPO_ROOT / "src" / "utils" / "evaluations" / "chunk_batch_job.py"


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return [value]


def _safe_name(value: object) -> str:
    import re

    text = re.sub(r"[^A-Za-z0-9._@-]+", "_", str(value))
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "item"


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        token = str(value)
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _split_baseline_spec(spec: str) -> tuple[str, str | None, str]:
    token = str(spec or "").strip()
    if not token:
        return "", None, ""
    if "@" not in token:
        return token, None, token
    base, mode = token.split("@", 1)
    base = base.strip()
    mode = mode.strip()
    if not base:
        return "", None, token
    display = f"{base}@{mode}" if mode else base
    return base, (mode or None), display


def _job_dir_name(spec: dict) -> str:
    test_dir = str(spec.get("test_dir", "chunk_test"))
    model_root = str(spec.get("model_root", "model_root"))
    model_tag = str(spec.get("model_tag", "NA"))
    manual_config = dict(spec.get("manual_config") or {})
    config = dict(manual_config or {})
    q1 = config.get("Q1", "NA")
    q2 = config.get("Q2", "NA")
    dataset_family = infer_dataset_name_from_path(test_dir)
    dataset_stem = Path(str(test_dir)).stem or "chunk_test"
    if dataset_family:
        dataset_label = f"{dataset_family}_{dataset_stem}"
    else:
        dataset_label = dataset_stem
    parts = [_safe_name(dataset_label)]
    task_type = str(spec.get("task_type", "learned_model") or "learned_model").strip().lower()
    if task_type == "classic_baseline":
        parts.extend(["Baseline", _safe_name(spec.get("baseline_method", "baseline"))])
        if spec.get("kalman_mode"):
            parts.append(_safe_name(spec["kalman_mode"]))
    else:
        parts.extend(
            [
                _safe_name(model_tag),
                _safe_name(Path(str(model_root)).name or "model_root"),
                _safe_name(spec.get("model_name", "model")),
            ]
        )
    parts.extend([f"Q1_{q1}", f"Q2_{q2}"])
    return "__".join(parts)


def _next_available_csv(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    idx = 2
    while True:
        candidate = path.with_name(f"{stem}__{idx}{suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def _task_label(spec: dict) -> str:
    task_type = str(spec.get("task_type", "learned_model") or "learned_model").strip().lower()
    if task_type == "classic_baseline":
        method = str(spec.get("baseline_method", "baseline"))
        mode = spec.get("kalman_mode")
        return f"baseline:{method}@{mode}" if mode else f"baseline:{method}"
    return f"{spec.get('model_tag', 'model')}/{spec.get('model_name', 'NA')}"


def _config_label(spec: dict) -> str:
    cfg = dict(spec.get("manual_config") or {})
    return f"Q1={cfg.get('Q1', 'NA')} Q2={cfg.get('Q2', 'NA')}"


def _progress_bar(finished: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "] 0/0 (0.0%)"
    ratio = max(0.0, min(1.0, float(finished) / float(total)))
    filled = int(width * ratio)
    return f"[{'#' * filled}{'-' * (width - filled)}] {finished}/{total} ({ratio * 100.0:5.1f}%)"


def _render_active_jobs(active_jobs: list[dict], *, finished_jobs: int, total_jobs: int, max_workers: int) -> str:
    lines = [
        "Chunk Evaluation Batch",
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
                    dataset=Path(str(job["spec"].get("test_dir", "NA"))).stem[:28],
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
    chunk_results_dir: Path,
    chunk_pointwise_results_dir: Path,
    chunk_bytewise_results_dir: Path,
    chunk_p_val_dir: Path,
    valhalla_diagnostics_dir: Path,
) -> None:
    for filename, target_dir in [
        ("chunk_result.csv", chunk_results_dir),
        ("chunk_pointwise_result.csv", chunk_pointwise_results_dir),
        ("chunk_bytewise_result.csv", chunk_bytewise_results_dir),
        ("chunk_p_val_result.csv", chunk_p_val_dir),
    ]:
        source = job_dir / filename
        if source.exists():
            shutil.copy2(source, _next_available_csv(target_dir / f"{job_key}.csv"))
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


def _augment_chunk_summary_with_tail_metrics(
    *,
    summary_csv: Path,
    pointwise_csv: Path,
) -> None:
    if not summary_csv.exists() or not pointwise_csv.exists():
        return

    def norm(value: object) -> str:
        token = str(value or "").strip()
        return "" if token.lower() in {"na", "none", "nan"} else token

    def key(row: dict[str, str]) -> tuple[str, ...]:
        return tuple(norm(row.get(c)) for c in ["dataset_name", "model_tag", "model_full_name", "Q1", "Q2"])

    tail_lookup: dict[tuple[str, ...], tuple[str, str]] = {}
    with pointwise_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            point_cols = sorted(
                ((int(c[6:]), c) for c in row if c.startswith("point_") and c[6:].isdigit()),
                reverse=True,
            )
            for idx, col in point_cols:
                try:
                    value = float(str(row.get(col, "")).strip())
                except ValueError:
                    continue
                if value == value:
                    tail_lookup[key(row)] = (f"{value:.6f}", str(idx))
                    break

    if not tail_lookup:
        return

    with summary_csv.open("r", newline="", encoding="utf-8") as f:
        summary_reader = csv.DictReader(f)
        fieldnames = list(summary_reader.fieldnames or [])
        summary_rows = list(summary_reader)
    if not fieldnames or not summary_rows:
        return

    for col in ["err_mean_tail", "tail_point_index"]:
        if col not in fieldnames:
            fieldnames.append(col)

    for row in summary_rows:
        match = tail_lookup.get(key(row))
        if match is None:
            continue
        tail_value, tail_index = match
        row["err_mean_tail"] = tail_value
        row["tail_point_index"] = tail_index

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def run_chunk_batch(
    *,
    manager,
    job: dict,
    model_root: str,
    model_names: Optional[list],
    classic_baselines: list[str],
    model_tag: str,
    run_baselines: bool,
    max_workers: int = 4,
    log_level: str = "INFO",
) -> None:
    chunk_paths = [str(p).strip() for p in _as_list(job.get("chunk_dirs")) if str(p).strip()]
    if not chunk_paths:
        fallback = str(job.get("chunk_test_dir", "") or "").strip()
        if fallback:
            chunk_paths = [fallback]
    if not chunk_paths:
        raise ValueError("No chunk evaluation paths configured.")

    runtime_cfg = job.get("runtime", {}) if isinstance(job.get("runtime"), dict) else {}
    if str(runtime_cfg.get("device_effective", runtime_cfg.get("device", "cuda"))).strip().lower() == "cpu":
        default_max_chunks = int(runtime_cfg.get("cpu_chunk_max_chunks", 2000))
    else:
        default_max_chunks = 5000
    raw_max_chunks = job.get("chunk_max_chunks", default_max_chunks)
    if raw_max_chunks is None:
        max_chunks = None
    else:
        try:
            max_chunks = int(raw_max_chunks)
        except Exception:
            max_chunks = int(default_max_chunks)
        if max_chunks <= 0:
            max_chunks = None

    manual_configs: list[dict | None] = []
    if bool(job.get("chunk_grid_search", False)):
        q1_vals = _as_list(job.get("Q1")) or [1]
        q2_vals = _as_list(job.get("Q2")) or [12]
        for q1 in q1_vals:
            for q2 in q2_vals:
                manual_configs.append(
                    {
                        "Q1": int(q1),
                        "Q2": int(q2),
                    }
                )
    else:
        manual_config = job.get("chunk_config")
        if manual_config is None:
            q1 = job.get("Q1")
            q2 = job.get("Q2")
            if q1 is not None or q2 is not None:
                manual_config = {
                    "Q1": q1[0] if isinstance(q1, list) and q1 else q1,
                    "Q2": q2[0] if isinstance(q2, list) and q2 else q2,
                }
                manual_config = {k: v for k, v in manual_config.items() if v is not None}
        manual_configs.append(manual_config)

    chunk_jobs_dir = Path(manager.output_dir) / "chunk_jobs"
    chunk_results_dir = Path(manager.output_dir) / "chunk_results"
    chunk_pointwise_results_dir = Path(manager.output_dir) / "chunk_pointwise_results"
    chunk_bytewise_results_dir = Path(manager.output_dir) / "chunk_bytewise_results"
    chunk_p_val_dir = Path(manager.output_dir) / "chunk_p_val"
    valhalla_diagnostics_dir = Path(manager.output_dir) / "valhalla_meili_diagnostics"
    chunk_jobs_dir.mkdir(parents=True, exist_ok=True)
    chunk_results_dir.mkdir(parents=True, exist_ok=True)
    chunk_pointwise_results_dir.mkdir(parents=True, exist_ok=True)
    chunk_bytewise_results_dir.mkdir(parents=True, exist_ok=True)
    chunk_p_val_dir.mkdir(parents=True, exist_ok=True)
    valhalla_diagnostics_dir.mkdir(parents=True, exist_ok=True)

    non_kalman_baselines: list[str] = []
    kalman_modes: list[str] = []
    for spec in classic_baselines:
        base_name, kalman_mode, _display_name = _split_baseline_spec(spec)
        if not base_name:
            continue
        if base_name != "kalman_rts":
            non_kalman_baselines.append(base_name)
            continue
        kalman_modes.append(kalman_mode or "dataset")
    non_kalman_baselines = _dedupe_keep_order(non_kalman_baselines)
    kalman_modes = _dedupe_keep_order(kalman_modes)

    task_specs: list[dict] = []
    resolved_model_names = list(model_names or manager._discover_models(model_root))
    for test_dir in chunk_paths:
        for config_index, manual_config in enumerate(manual_configs):
            run_baseline_here = bool(run_baselines and job.get("run_baseline", job.get("baseline_once", True)))
            if run_baseline_here:
                for baseline in non_kalman_baselines:
                    if baseline == "valhalla_meili" and config_index > 0:
                        continue
                    baseline_config = None
                    if baseline == "valhalla_meili":
                        dataset_profile_name = (
                            infer_dataset_name_from_path(test_dir)
                            or Path(str(test_dir)).stem
                        )
                        baseline_config = resolve_valhalla_profile(
                            dataset_profile_name,
                            dict(job.get("baseline_options") or {}),
                        )["config"]
                    task_specs.append(
                        {
                            "task_type": "classic_baseline",
                            "test_dir": test_dir,
                            "max_chunks": max_chunks,
                            "manual_config": manual_config,
                            "baseline_method": baseline,
                            "baseline_config": baseline_config,
                            "log_level": str(log_level).upper(),
                        }
                    )
                for kalman_mode in kalman_modes:
                    task_specs.append(
                        {
                            "task_type": "classic_baseline",
                            "test_dir": test_dir,
                            "max_chunks": max_chunks,
                            "manual_config": manual_config,
                            "baseline_method": "kalman_rts",
                            "kalman_mode": kalman_mode,
                            "log_level": str(log_level).upper(),
                        }
                    )
            for model_name in resolved_model_names:
                task_specs.append(
                    {
                        "task_type": "learned_model",
                        "test_dir": test_dir,
                        "max_chunks": max_chunks,
                        "manual_config": manual_config,
                        "model_name": str(model_name),
                        "model_root": str(model_root),
                        "model_tag": str(model_tag),
                        "log_level": str(log_level).upper(),
                    }
                )

    if not task_specs:
        return

    logging.info(
        "Chunk batch start | jobs=%d parallel=%d output_root=%s",
        len(task_specs),
        int(max_workers),
        chunk_jobs_dir,
    )

    specs_root = chunk_jobs_dir / "specs"
    specs_root.mkdir(parents=True, exist_ok=True)
    pending_specs = list(task_specs)
    active_jobs: list[dict] = []
    finished_jobs = 0
    _launch_pending_jobs_until_full(
        pending_specs=pending_specs,
        active_jobs=active_jobs,
        specs_root=specs_root,
        jobs_root=chunk_jobs_dir,
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
                    "Chunk evaluation batch job failed: "
                    f"{_task_label(job_info['spec'])} dir={job_info['spec'].get('test_dir')} "
                    f"exit={ret} log={job_info['job_dir'] / 'stdout.log'}"
                )

            _collect_job_outputs(
                job_dir=job_info["job_dir"],
                job_key=job_info["job_key"],
                chunk_results_dir=chunk_results_dir,
                chunk_pointwise_results_dir=chunk_pointwise_results_dir,
                chunk_bytewise_results_dir=chunk_bytewise_results_dir,
                chunk_p_val_dir=chunk_p_val_dir,
                valhalla_diagnostics_dir=valhalla_diagnostics_dir,
            )
            finished_jobs += 1
            _launch_pending_jobs_until_full(
                pending_specs=pending_specs,
                active_jobs=active_jobs,
                specs_root=specs_root,
                jobs_root=chunk_jobs_dir,
                max_workers=int(max_workers),
            )
            _emit_snapshot(active_jobs, finished_jobs=finished_jobs, total_jobs=len(task_specs), max_workers=int(max_workers))

    aggregate_csv_folder(
        chunk_results_dir,
        chunk_results_dir / "aggregated.csv",
        skip_names={"aggregated.csv"},
    )
    aggregate_csv_folder(
        chunk_pointwise_results_dir,
        chunk_pointwise_results_dir / "aggregated.csv",
        skip_names={"aggregated.csv"},
    )
    aggregate_csv_folder(
        chunk_bytewise_results_dir,
        chunk_bytewise_results_dir / "aggregated.csv",
        skip_names={"aggregated.csv"},
    )
    aggregate_csv_folder(
        chunk_p_val_dir,
        chunk_p_val_dir / "aggregated.csv",
        skip_names={"aggregated.csv"},
    )
    aggregated = chunk_results_dir / "aggregated.csv"
    point_aggregated = chunk_pointwise_results_dir / "aggregated.csv"
    _augment_chunk_summary_with_tail_metrics(
        summary_csv=aggregated,
        pointwise_csv=point_aggregated,
    )
    if aggregated.exists():
        shutil.copy2(aggregated, Path(manager.output_dir) / "chunk_evaluation_summary.csv")
    if point_aggregated.exists():
        shutil.copy2(point_aggregated, Path(manager.output_dir) / "chunk_pointwise_summary.csv")
    byte_aggregated = chunk_bytewise_results_dir / "aggregated.csv"
    if byte_aggregated.exists():
        shutil.copy2(byte_aggregated, Path(manager.output_dir) / "chunk_bytewise_summary.csv")
    pval_aggregated = chunk_p_val_dir / "aggregated.csv"
    if pval_aggregated.exists():
        shutil.copy2(pval_aggregated, Path(manager.output_dir) / "chunk_p_val.csv")
        generate_pairwise_p_value_report(
            Path(manager.output_dir) / "chunk_p_val.csv",
            Path(manager.output_dir) / "chunk_p_value_summary",
            sample_type="chunk",
            metric_column="mean_l2_err_full",
        )

    logging.info(
        "Chunk batch complete | output_csv=%s pointwise_csv=%s bytewise_csv=%s pval_csv=%s",
        aggregated,
        point_aggregated,
        byte_aggregated,
        pval_aggregated,
    )
