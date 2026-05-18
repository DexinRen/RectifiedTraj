import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from utils.evaluations.result_io import aggregate_csv_folder
from utils.evaluations.benchmark_inputs import infer_dataset_name_from_path
from utils.evaluations.p_value import generate_pairwise_p_value_report


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


def _job_dir_name(
    *,
    test_dir: str,
    model_root: str,
    model_tag: str,
    manual_config: dict | None,
) -> str:
    config = dict(manual_config or {})
    q1 = config.get("Q1", "NA")
    q2 = config.get("Q2", "NA")
    dataset_family = infer_dataset_name_from_path(test_dir)
    dataset_stem = Path(str(test_dir)).stem or "chunk_test"
    if dataset_family:
        dataset_label = f"{dataset_family}_{dataset_stem}"
    else:
        dataset_label = dataset_stem
    return "__".join(
        [
            _safe_name(dataset_label),
            _safe_name(model_tag),
            _safe_name(Path(str(model_root)).name or "model_root"),
            f"Q1_{q1}",
            f"Q2_{q2}",
        ]
    )


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


def run_chunk_batch(
    *,
    manager,
    job: dict,
    model_root: str,
    model_names: Optional[list],
    classic_baselines: list[str],
    model_tag: str,
    run_baselines: bool,
) -> None:
    from utils.evaluations.evaluation_manager import TestManager

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
    chunk_jobs_dir.mkdir(parents=True, exist_ok=True)
    chunk_results_dir.mkdir(parents=True, exist_ok=True)
    chunk_pointwise_results_dir.mkdir(parents=True, exist_ok=True)
    chunk_bytewise_results_dir.mkdir(parents=True, exist_ok=True)
    chunk_p_val_dir.mkdir(parents=True, exist_ok=True)

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

    logging.info(
        "Chunk batch start | dirs=%d output_root=%s",
        len(chunk_paths),
        chunk_jobs_dir,
    )

    for test_dir in chunk_paths:
        for manual_config in manual_configs:
            job_dir = chunk_jobs_dir / _job_dir_name(
                test_dir=test_dir,
                model_root=model_root,
                model_tag=model_tag,
                manual_config=manual_config,
            )
            job_dir.mkdir(parents=True, exist_ok=True)

            logging.info(
                "Chunk batch run | dir=%s model_root=%s Q1=%s Q2=%s",
                test_dir,
                model_root,
                (manual_config or {}).get("Q1"),
                (manual_config or {}).get("Q2"),
            )

            local_manager = TestManager(output_dir=str(job_dir))
            run_baseline_here = bool(run_baselines and job.get("run_baseline", job.get("baseline_once", True)))
            local_manager.run_chunk_evaluation(
                model_names=model_names,
                model_root=model_root,
                model_tag=model_tag,
                test_dir=test_dir,
                max_chunks=max_chunks,
                manual_config=manual_config,
                run_baselines=bool(run_baseline_here and non_kalman_baselines),
                baseline_methods=non_kalman_baselines,
            )

            if run_baseline_here and kalman_modes:
                prev_mode = os.getenv("KALMAN_RTS_CALIBRATION_MODE")
                try:
                    for kalman_mode in kalman_modes:
                        os.environ["KALMAN_RTS_CALIBRATION_MODE"] = str(kalman_mode)
                        logging.info(
                            "Chunk batch kalman run | dir=%s mode=%s Q1=%s Q2=%s",
                            test_dir,
                            kalman_mode,
                            (manual_config or {}).get("Q1"),
                            (manual_config or {}).get("Q2"),
                        )
                        local_manager.run_chunk_evaluation(
                            model_names=[],
                            model_root=model_root,
                            model_tag=model_tag,
                            test_dir=test_dir,
                            max_chunks=max_chunks,
                            manual_config=manual_config,
                            run_baselines=True,
                            baseline_methods=["kalman_rts"],
                        )
                finally:
                    if prev_mode is None:
                        os.environ.pop("KALMAN_RTS_CALIBRATION_MODE", None)
                    else:
                        os.environ["KALMAN_RTS_CALIBRATION_MODE"] = prev_mode

            chunk_csv = job_dir / "chunk_evaluation_summary.csv"
            if chunk_csv.exists():
                target = _next_available_csv(chunk_results_dir / f"{job_dir.name}.csv")
                shutil.copy2(chunk_csv, target)
            chunk_point_csv = job_dir / "chunk_pointwise_summary.csv"
            if chunk_point_csv.exists():
                target = _next_available_csv(chunk_pointwise_results_dir / f"{job_dir.name}.csv")
                shutil.copy2(chunk_point_csv, target)
            chunk_byte_csv = job_dir / "chunk_bytewise_summary.csv"
            if chunk_byte_csv.exists():
                target = _next_available_csv(chunk_bytewise_results_dir / f"{job_dir.name}.csv")
                shutil.copy2(chunk_byte_csv, target)
            chunk_pval_csv = job_dir / "chunk_p_val.csv"
            if chunk_pval_csv.exists():
                target = _next_available_csv(chunk_p_val_dir / f"{job_dir.name}.csv")
                shutil.copy2(chunk_pval_csv, target)

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
    if aggregated.exists():
        shutil.copy2(aggregated, Path(manager.output_dir) / "chunk_evaluation_summary.csv")
    point_aggregated = chunk_pointwise_results_dir / "aggregated.csv"
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
