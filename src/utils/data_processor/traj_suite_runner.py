#!/usr/bin/env python3
"""Trajectory extraction suite runner for parquet processor integration."""

from __future__ import annotations

import logging
import os
import json
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path


TRAJ_RESERVED_CPU_CORES = 2
TRAJ_MAX_CPU_CORES = 4
TRAJ_MAX_MEMORY_GB = 16.0
TRAJ_FULL_COUNT = 200
TRAJ_FULL_POINTS = 5000
TRAJ_DEBUG_COUNT = 2
TRAJ_DEBUG_POINTS = 20
TRAJ_ALLOW_SHORTER = True


def _env_int(name: str, default: int, *, min_value: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return max(int(min_value), int(raw))
    except Exception:
        return int(default)


def _env_float(
    name: str,
    default: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    raw = os.environ.get(name)
    if raw is None:
        value = float(default)
    else:
        try:
            value = float(raw)
        except Exception:
            value = float(default)
    if min_value is not None:
        value = max(float(min_value), float(value))
    if max_value is not None:
        value = min(float(max_value), float(value))
    return float(value)


TRAJ_ESTIMATE_MAX_AGENTS = _env_int("TRAJ_ESTIMATE_MAX_AGENTS", 200, min_value=1)
TRAJ_ESTIMATE_POINTS_PER_AGENT = _env_int("TRAJ_ESTIMATE_POINTS_PER_AGENT", 5000, min_value=2)
TRAJ_FULL_COUNT = _env_int("TRAJ_FULL_COUNT", TRAJ_FULL_COUNT, min_value=1)
TRAJ_FULL_POINTS = _env_int("TRAJ_FULL_POINTS", TRAJ_FULL_POINTS, min_value=2)
TRAJ_SAMPLETIME_TARGETS = [
    ("10s", 10.0),
    ("30s", 30.0),
    ("1min", 60.0),
    ("2min", 120.0),
]
TRAJ_SAMPLETIME_TARGETS_BY_DATASET = {
    "blogwatcher": [
        ("1min", 60.0),
        ("2min", 120.0),
    ],
    "pol_5s": [
        ("10s", 10.0),
        ("30s", 30.0),
        ("1min", 60.0),
        ("2min", 120.0),
    ],
}
TRAJ_NATIVE_EQUIV_INTERVALS_BY_DATASET_SEC = {
    "numosim_kanto": frozenset({10.0}),
    "pol_1min": frozenset({60.0}),
}
TRAJ_MIN_INTERVAL_RATIO_FROM_NATIVE_MEDIAN = _env_float(
    "TRAJ_MIN_INTERVAL_RATIO_FROM_NATIVE_MEDIAN",
    0.75,
    min_value=0.0,
    max_value=1.0,
)
TRAJ_BANNED_INTERVALS_SEC = frozenset({300.0})
TRAJ_NATIVE_LABEL = "native"
THREAD_LIMIT_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "POLARS_MAX_THREADS",
)
HEARTBEAT_SILENT_STAGES = frozenset(
    {
        "estimate_dataset_sample_time",
        "run_full_suite",
        "run_debug_suite",
    }
)


logger = logging.getLogger(__name__)

def _is_blogwatcher_dataset(raw_ds_path: str) -> bool:
    return Path(raw_ds_path).name.lower() == "blogwatcher"


def _target_traj_cpu_cores(total_cores: int, reserve_cores: int) -> int:
    base = max(1, int(total_cores) - int(reserve_cores))
    max_cap_raw = os.environ.get("TRAJ_MAX_CPU_CORES", str(int(TRAJ_MAX_CPU_CORES)))
    try:
        max_cap = max(1, int(max_cap_raw))
    except Exception:
        max_cap = int(TRAJ_MAX_CPU_CORES)
    return max(1, min(int(base), int(max_cap)))


def _suite_print(msg: str) -> None:
    print(msg, flush=True)


def _build_interval_targets(raw_ds_path: str) -> list[tuple[str, float]]:
    dataset_key = Path(raw_ds_path).name.lower()
    target_source = TRAJ_SAMPLETIME_TARGETS_BY_DATASET.get(dataset_key, TRAJ_SAMPLETIME_TARGETS)
    targets: list[tuple[str, float]] = []
    for label, sec in target_source:
        sec_f = float(sec)
        if any(abs(sec_f - float(b)) < 1e-9 for b in TRAJ_BANNED_INTERVALS_SEC):
            _suite_print(
                f"[traj_suite] banned interval skipped: label={str(label)} sec={float(sec_f):.3f}"
            )
            continue
        targets.append((str(label), float(sec_f)))
    return targets


@contextmanager
def _suite_stage_heartbeat(stage_ref: dict, interval_sec: int = 30):
    stop_event = threading.Event()
    started_at = time.monotonic()

    def _worker() -> None:
        while not stop_event.wait(float(interval_sec)):
            elapsed = int(time.monotonic() - started_at)
            stage = str(stage_ref.get("value", "unknown"))
            # Live trajectory progress uses cursor-up redraw; extra heartbeat lines
            # in these stages break the visual overwrite behavior.
            if stage in HEARTBEAT_SILENT_STAGES:
                continue
            _suite_print(f"[traj_suite] heartbeat: elapsed={elapsed}s stage={stage}")

    worker = threading.Thread(target=_worker, name="traj-suite-heartbeat", daemon=True)
    _suite_print(f"[traj_suite] heartbeat started (interval={int(interval_sec)}s)")
    worker.start()
    try:
        yield
    finally:
        stop_event.set()
        worker.join(timeout=1.0)
        elapsed = int(time.monotonic() - started_at)
        stage = str(stage_ref.get("value", "unknown"))
        _suite_print(f"[traj_suite] heartbeat stopped (elapsed={elapsed}s stage={stage})")


@contextmanager
def _traj_resource_guard(
    reserve_cores: int = TRAJ_RESERVED_CPU_CORES,
    max_memory_gb: float = TRAJ_MAX_MEMORY_GB,
):
    prev_affinity = None
    prev_env = {key: os.environ.get(key) for key in THREAD_LIMIT_ENV_VARS}
    target_cores = max(1, int(os.cpu_count() or 1))

    try:
        _suite_print("[traj_suite] resource_guard: entering")
        # Keep at least `reserve_cores` CPUs free for the desktop/system.
        if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
            prev_affinity = os.sched_getaffinity(0)
            allowed = sorted(prev_affinity)
            target_cores = _target_traj_cpu_cores(len(allowed), int(reserve_cores))
            new_affinity = set(allowed[:target_cores])
            if new_affinity and new_affinity != prev_affinity:
                os.sched_setaffinity(0, new_affinity)
        else:
            target_cores = _target_traj_cpu_cores(int(os.cpu_count() or 1), int(reserve_cores))
        _suite_print(
            f"[traj_suite] resource_guard: cpu affinity applied (target_cores={int(target_cores)})"
        )

        for key in THREAD_LIMIT_ENV_VARS:
            os.environ[key] = str(int(target_cores))
        _suite_print("[traj_suite] resource_guard: thread env vars applied")
        _suite_print(
            "[traj_suite] resource_guard: memory cap disabled "
            "(RLIMIT_AS removed to avoid allocator false failures)"
        )

        # Do not mutate torch threadpools here; this can hang in some runtime states.
        # Thread limits are enforced via env vars and traj_extractor module-level setup.
        _suite_print("[traj_suite] resource_guard: skipping runtime torch thread mutation")

        logger.info(
            "Trajectory extraction resource guard active: cpu_cores=%d reserve_cores=%d max_memory_gb=%.2f",
            int(target_cores),
            int(reserve_cores),
            float(max_memory_gb),
        )
        _suite_print(
            "[traj_suite] resource_guard active: "
            f"cpu_cores={int(target_cores)} reserve_cores={int(reserve_cores)} "
            f"max_memory_gb={float(max_memory_gb):.2f}"
        )
        yield {
            "cpu_cores": int(target_cores),
            "reserve_cores": int(reserve_cores),
            "max_memory_gb": float(max_memory_gb),
        }
    finally:
        if prev_affinity is not None:
            try:
                os.sched_setaffinity(0, prev_affinity)
            except Exception as exc:
                logger.warning("Failed to restore original CPU affinity: %s", exc)

        for key, value in prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_traj_extraction_functions():
    _suite_print("[traj_suite] import attempt: relative .traj_extractor")
    try:
        from .traj_extractor import (
            extract_10min_traj as _extract_10min,
            extract_native_traj as _extract_native,
            estimate_dataset_sample_time_seconds as _estimate_sampletime,
            build_traj_extraction_context as _build_traj_context,
        )
        _suite_print("[traj_suite] import success: relative .traj_extractor")
    except ImportError as exc:
        _suite_print(f"[traj_suite] relative import failed: {exc}; trying fallback traj_extractor")
        from traj_extractor import (
            extract_10min_traj as _extract_10min,
            extract_native_traj as _extract_native,
            estimate_dataset_sample_time_seconds as _estimate_sampletime,
            build_traj_extraction_context as _build_traj_context,
        )
        _suite_print("[traj_suite] import success: fallback traj_extractor")
    return _extract_10min, _extract_native, _estimate_sampletime, _build_traj_context


def _run_single_traj_suite(
    raw_ds_path: str,
    output_root: Path,
    *,
    extract_10min_fn,
    extract_native_fn,
    target_m: int,
    target_n: int,
    interval_targets: list[tuple[str, float]],
    dataset_sample_median_sec: float | None,
    run_resampled_intervals: bool = True,
    include_error_range: bool = False,
    extraction_context: Optional[dict] = None,
) -> dict:
    dataset_sample_hint = None
    try:
        if dataset_sample_median_sec is not None:
            candidate = float(dataset_sample_median_sec)
            if candidate > 0.0:
                dataset_sample_hint = float(candidate)
    except Exception:
        dataset_sample_hint = None
    suite = {
        "target_M": int(target_m),
        "target_N": int(target_n),
        "dataset_sample_median_sec": dataset_sample_hint,
        "runs": {},
        "skipped": {},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    ctx = extraction_context or {}
    native_interval_stats = {}
    is_blogwatcher_dataset = _is_blogwatcher_dataset(raw_ds_path)

    # Always run native first.
    try:
        _suite_print(
            f"[traj_suite:{output_root.name}] start interval=native "
            f"target_m={int(target_m)} target_n={int(target_n)}"
        )
        native_result = extract_native_fn(
            parquet_dir=raw_ds_path,
            output_dir=str(output_root),
            m=int(target_m),
            n_points=int(target_n),
            allow_size_override=True,
            allow_shorter=bool(TRAJ_ALLOW_SHORTER),
            include_error_range=include_error_range,
            precomputed_metadata=ctx.get("metadata"),
            precomputed_column_map=ctx.get("column_map"),
            ordered_agents=ctx.get("ordered_agents"),
            agent_entry_counts=ctx.get("agent_entry_counts"),
        )
        native_interval_stats = native_result.get("interval_stats_sec", {}) or {}
        requested_native_sample_sec = dataset_sample_hint
        if requested_native_sample_sec is None:
            try:
                requested_native_sample_sec = float(
                    native_interval_stats.get(
                        "median",
                        native_interval_stats.get("mean", 0.0),
                    )
                )
            except Exception:
                requested_native_sample_sec = None
            if requested_native_sample_sec is not None and not (requested_native_sample_sec > 0.0):
                requested_native_sample_sec = None
        suite["runs"][TRAJ_NATIVE_LABEL] = {
            "status": "completed",
            "requested_sample_sec": requested_native_sample_sec,
            "is_native_target": True,
            "output_file": native_result.get("output_path"),
            "n_trajectories": int(native_result.get("n_trajectories", 0)),
            "total_points": int(native_result.get("total_points", 0)),
            "avg_length": int(native_result.get("avg_length", 0)),
            "median_length": int(native_result.get("median_length", 0)),
            "min_length": int(native_result.get("min_length", 0)),
            "max_length": int(native_result.get("max_length", 0)),
            "extraction_failures": int(native_result.get("extraction_failures", 0)),
            "interval_stats_sec": native_interval_stats,
            "sample_time_label": native_result.get("sample_time_label"),
            "quality_stats": native_result.get("quality_stats", {}),
        }
        _suite_print(
            f"[traj_suite:{output_root.name}] done interval=native "
            f"n_trajectories={int(native_result.get('n_trajectories', 0))} "
            f"failures={int(native_result.get('extraction_failures', 0))}"
        )
    except Exception as e:
        suite["runs"][TRAJ_NATIVE_LABEL] = {
            "status": "failed",
            "requested_sample_sec": dataset_sample_hint,
            "is_native_target": True,
            "error": str(e),
        }
        _suite_print(f"[traj_suite:{output_root.name}] failed interval=native: {e}")

    native_sample_sec = float(dataset_sample_hint or 0.0)
    if native_interval_stats:
        try:
            native_sample_sec = float(native_interval_stats.get("mean", native_sample_sec))
        except Exception:
            native_sample_sec = float(dataset_sample_hint or 0.0)
    if not (native_sample_sec > 0.0):
        native_sample_sec = float(dataset_sample_hint or 0.0)

    native_median_sec = float(dataset_sample_hint or 0.0)
    if native_interval_stats:
        try:
            native_median_sec = float(native_interval_stats.get("median", native_median_sec))
        except Exception:
            native_median_sec = float(dataset_sample_hint or 0.0)
    if not (native_median_sec > 0.0):
        native_median_sec = float(dataset_sample_hint or 0.0)

    min_interval_sec = None
    min_interval_ratio = float(TRAJ_MIN_INTERVAL_RATIO_FROM_NATIVE_MEDIAN)
    if min_interval_ratio > 0.0 and native_median_sec > 0.0:
        min_interval_sec = float(min_interval_ratio) * float(native_median_sec)
        _suite_print(
            f"[traj_suite:{output_root.name}] interval floor active: "
            f"ratio={float(min_interval_ratio):.3f} "
            f"native_median_sec={float(native_median_sec):.3f} "
            f"min_interval_sec={float(min_interval_sec):.3f}"
        )
    suite["native_sample_median_sec"] = float(native_median_sec)
    suite["min_interval_ratio_from_native_median"] = float(min_interval_ratio)
    suite["min_interval_floor_sec"] = (
        float(min_interval_sec) if min_interval_sec is not None else None
    )
    dataset_key = Path(raw_ds_path).name.lower()
    dataset_native_equiv_intervals = TRAJ_NATIVE_EQUIV_INTERVALS_BY_DATASET_SEC.get(
        dataset_key,
        frozenset(),
    )
    suite["dataset_native_equivalent_intervals_sec"] = [
        float(x) for x in sorted(dataset_native_equiv_intervals)
    ]

    if not bool(run_resampled_intervals):
        skip_reason = "skipped_native_only_dataset_mode"
        for label, target_sec in interval_targets:
            suite["runs"][label] = {
                "status": "skipped",
                "requested_sample_sec": float(target_sec),
                "is_native_target": False,
                "reason": skip_reason,
            }
            suite["skipped"][label] = skip_reason
        _suite_print(
            f"[traj_suite:{output_root.name}] native-only mode active; "
            "skipping resampled interval extraction."
        )
        return suite

    auto_fail_threshold_sec = None

    for label, target_sec in interval_targets:
        if any(
            abs(float(target_sec) - float(native_eq_sec)) < 1e-9
            for native_eq_sec in dataset_native_equiv_intervals
        ):
            skip_reason = (
                "skipped_dataset_native_equivalent_interval: "
                f"dataset={dataset_key} "
                f"target_sec={float(target_sec):.3f}"
            )
            suite["runs"][label] = {
                "status": "skipped",
                "requested_sample_sec": float(target_sec),
                "is_native_target": False,
                "reason": skip_reason,
            }
            suite["skipped"][label] = skip_reason
            _suite_print(
                f"[traj_suite:{output_root.name}] skipped interval={label} "
                f"(dataset-native-equivalent target_sec={float(target_sec):.3f})"
            )
            continue
        if min_interval_sec is not None and float(target_sec) < float(min_interval_sec):
            skip_reason = (
                "skipped_interval_below_native_median_floor: "
                f"target_sec={float(target_sec):.3f} "
                f"min_interval_sec={float(min_interval_sec):.3f} "
                f"native_median_sec={float(native_median_sec):.3f} "
                f"ratio={float(min_interval_ratio):.3f}"
            )
            suite["runs"][label] = {
                "status": "skipped",
                "requested_sample_sec": float(target_sec),
                "is_native_target": False,
                "reason": skip_reason,
            }
            suite["skipped"][label] = skip_reason
            _suite_print(
                f"[traj_suite:{output_root.name}] skipped interval={label} "
                f"(target_sec={float(target_sec):.3f} < floor={float(min_interval_sec):.3f})"
            )
            continue
        if auto_fail_threshold_sec is not None and float(target_sec) > float(auto_fail_threshold_sec):
            suite["runs"][label] = {
                "status": "failed",
                "requested_sample_sec": float(target_sec),
                "is_native_target": False,
                "error": (
                    "auto_skipped_larger_interval_after_empty_failure: "
                    f"trigger_interval_sec={float(auto_fail_threshold_sec):.3f}"
                ),
            }
            _suite_print(
                f"[traj_suite:{output_root.name}] auto-skip interval={label} "
                f"(>{float(auto_fail_threshold_sec):.3f}s) after empty failure"
            )
            continue
        try:
            _suite_print(
                f"[traj_suite:{output_root.name}] start interval={label} "
                f"target_m={int(target_m)} target_n={int(target_n)}"
            )
            interval_result = extract_10min_fn(
                parquet_dir=raw_ds_path,
                m=int(target_m),
                n_points=int(target_n),
                intervals_sec=[int(target_sec)],
                output_dir_tmpl=str(output_root),
                allow_shorter=bool(TRAJ_ALLOW_SHORTER),
                allow_size_override=True,
                include_error_range=include_error_range,
                precomputed_metadata=ctx.get("metadata"),
                precomputed_column_map=ctx.get("column_map"),
                ordered_agents=ctx.get("ordered_agents"),
                agent_entry_counts=ctx.get("agent_entry_counts"),
                native_sample_sec_hint=float(native_sample_sec),
            )
            runs = interval_result.get("interval_results", {})
            one = next(iter(runs.values()), {})
            suite["runs"][label] = {
                "status": "completed",
                "requested_sample_sec": float(target_sec),
                "is_native_target": False,
                "output_file": one.get("regular_path"),
                "n_trajectories": int(one.get("n_trajectories", 0)),
                "total_points": int(one.get("total_points", 0)),
                "avg_length": int(one.get("avg_length", 0)),
                "median_length": int(one.get("median_length", 0)),
                "min_length": int(one.get("min_length", 0)),
                "max_length": int(one.get("max_length", 0)),
                "extraction_failures": int(one.get("extraction_failures", 0)),
                "interval_stats_sec": one.get("interval_stats_sec", {}),
                "sample_time_label": one.get("sample_time_label"),
                "quality_stats": one.get("quality_stats", {}),
            }
            if (
                int(one.get("n_trajectories", 0)) <= 0
                and float(target_sec) > float(native_sample_sec)
            ):
                auto_fail_threshold_sec = float(target_sec)
                _suite_print(
                    f"[traj_suite:{output_root.name}] auto-fail armed at interval={label} "
                    f"(target_sec={float(target_sec):.3f} > native_sample_sec={float(native_sample_sec):.3f})"
                )
            _suite_print(
                f"[traj_suite:{output_root.name}] done interval={label} "
                f"n_trajectories={int(one.get('n_trajectories', 0))} "
                f"failures={int(one.get('extraction_failures', 0))}"
            )
        except Exception as e:
            suite["runs"][label] = {
                "status": "failed",
                "requested_sample_sec": float(target_sec),
                "is_native_target": False,
                "error": str(e),
            }
            _suite_print(f"[traj_suite:{output_root.name}] failed interval={label}: {e}")
            err_text = str(e).lower()
            no_traj_failed = (
                "no valid trajectories extracted" in err_text
                or "no trajectory" in err_text
            )
            if no_traj_failed and float(target_sec) > float(native_sample_sec):
                auto_fail_threshold_sec = float(target_sec)
                _suite_print(
                    f"[traj_suite:{output_root.name}] auto-fail armed at interval={label} "
                    f"(target_sec={float(target_sec):.3f} > native_sample_sec={float(native_sample_sec):.3f})"
                )

    return suite


def run_traj_extraction_suites(raw_ds_path: str, output_base_dir: str = "./dataset/processed") -> dict:
    """
    Run full and debug trajectory extraction suites.

    Rules:
    - Uses testing split only through traj_extractor metadata scanning.
    - Skips target sample-times below configured native-median ratio floor.
    """
    _suite_print(f"[traj_suite] starting for raw_ds_path={raw_ds_path}")

    stage = {"value": "resource_guard_init"}
    with _suite_stage_heartbeat(stage_ref=stage, interval_sec=30):
        with _traj_resource_guard(
            reserve_cores=int(TRAJ_RESERVED_CPU_CORES),
            max_memory_gb=float(TRAJ_MAX_MEMORY_GB),
        ) as resource_guard:
            stage["value"] = "load_extraction_functions"
            _suite_print("[traj_suite] loading extraction functions...")
            (
                extract_10min_fn,
                extract_native_fn,
                estimate_sampletime_fn,
                build_traj_context_fn,
            ) = _load_traj_extraction_functions()
            _suite_print("[traj_suite] extraction functions loaded.")

            is_blogwatcher_dataset = _is_blogwatcher_dataset(raw_ds_path)

            stage["value"] = "build_traj_context"
            _suite_print("[traj_suite] building one-time trajectory extraction context...")
            try:
                extraction_context = build_traj_context_fn(
                    raw_ds_path,
                    shuffle_seed=101,
                    sort_users_by_entries=not bool(is_blogwatcher_dataset),
                )
                _suite_print(
                    "[traj_suite] extraction context ready: "
                    f"ordered_agents={int(len(extraction_context.get('ordered_agents', [])))} "
                    f"order={'entry_count_desc' if not bool(is_blogwatcher_dataset) else 'random_shuffle'}"
                )
            except Exception as e:
                stage["value"] = "failed_build_traj_context"
                _suite_print(f"[traj_suite] extraction context build failed: {e}")
                return {
                    "status": "failed",
                    "reason": "traj_context_build_failed",
                    "error": str(e),
                    "resource_guard": resource_guard,
                }
            ds_name = Path(raw_ds_path).name
            include_error_range = bool(is_blogwatcher_dataset)
            output_base = Path(output_base_dir) / ds_name
            output_test_base = output_base / "test"
            interval_targets = _build_interval_targets(raw_ds_path)
            if is_blogwatcher_dataset:
                sample_time_stats = {
                    "status": "skipped",
                    "reason": "blogwatcher_native_only",
                    "sampled_agents": 0,
                    "n_intervals": 0,
                    "mean_sec": 0.0,
                    "median_sec": 0.0,
                    "std_sec": 0.0,
                }
                dataset_sample_median_sec = None
                interval_target_policy = {
                    "native_mode": "native_only_skip_sampletime",
                    "native_mean_sec": 0.0,
                }
                _suite_print(
                    "[traj_suite] BlogWatcher detected; skipping sample-time estimation "
                    "and using native interval stats for configured resampled extraction."
                )
            else:
                stage["value"] = "estimate_dataset_sample_time"
                try:
                    _suite_print("[traj_suite] estimating dataset sample-time stats...")
                    _suite_print(
                        "[traj_suite] sample-time estimation config: "
                        f"max_agents={int(TRAJ_ESTIMATE_MAX_AGENTS)} "
                        f"points_per_agent={int(TRAJ_ESTIMATE_POINTS_PER_AGENT)}"
                    )
                    sample_time_stats = estimate_sampletime_fn(
                        raw_ds_path,
                        max_agents=int(TRAJ_ESTIMATE_MAX_AGENTS),
                        points_per_agent=int(TRAJ_ESTIMATE_POINTS_PER_AGENT),
                        precomputed_metadata=extraction_context.get("metadata"),
                        precomputed_agent_file_index=extraction_context.get("agent_file_index"),
                        ordered_agents=extraction_context.get("ordered_agents"),
                    )
                    _suite_print(
                        "[traj_suite] sample-time stats ready: "
                        f"median_sec={float(sample_time_stats.get('median_sec', 0.0)):.3f} "
                        f"sampled_agents={int(sample_time_stats.get('sampled_agents', 0))}"
                    )
                except Exception as e:
                    stage["value"] = "failed_sample_time_estimation"
                    _suite_print(f"[traj_suite] sample-time estimation failed: {e}")
                    return {
                        "status": "failed",
                        "reason": "sample_time_estimation_failed",
                        "error": str(e),
                        "resource_guard": resource_guard,
                    }
                dataset_sample_median_sec = float(sample_time_stats.get("median_sec", 0.0))
                interval_target_policy = {
                    "native_mode": "explicit_native_sampler",
                    "native_mean_sec": float(sample_time_stats.get("mean_sec", 0.0)),
                }
            _suite_print(
                "[traj_suite] interval target policy: "
                f"mode={interval_target_policy.get('native_mode')} "
                f"native_mean_sec={float(interval_target_policy.get('native_mean_sec', 0.0)):.3f}"
            )
            _suite_print(
                "[traj_suite] interval targets: "
                + ", ".join(
                    f"{label}={float(sec):.3f}s"
                    for label, sec in interval_targets
                )
            )

            stage["value"] = "run_full_suite"
            _suite_print("[traj_suite] running full suite...")
            full_suite = _run_single_traj_suite(
                raw_ds_path,
                output_test_base / "traj_test",
                extract_10min_fn=extract_10min_fn,
                extract_native_fn=extract_native_fn,
                target_m=int(TRAJ_FULL_COUNT),
                target_n=int(TRAJ_FULL_POINTS),
                interval_targets=interval_targets,
                dataset_sample_median_sec=dataset_sample_median_sec,
                run_resampled_intervals=True,
                include_error_range=include_error_range,
                extraction_context=extraction_context,
            )
            stage["value"] = "run_debug_suite"
            _suite_print("[traj_suite] running debug suite...")
            debug_suite = _run_single_traj_suite(
                raw_ds_path,
                output_test_base / "traj_test_debug",
                extract_10min_fn=extract_10min_fn,
                extract_native_fn=extract_native_fn,
                target_m=int(TRAJ_DEBUG_COUNT),
                target_n=int(TRAJ_DEBUG_POINTS),
                interval_targets=interval_targets,
                dataset_sample_median_sec=dataset_sample_median_sec,
                run_resampled_intervals=True,
                include_error_range=include_error_range,
                extraction_context=extraction_context,
            )
            stage["value"] = "completed"
            _suite_print("[traj_suite] completed.")
            return {
                "status": "completed",
                "only_test_split": True,
                "dataset_sample_time_sec": sample_time_stats,
                "interval_target_policy": interval_target_policy,
                "extraction_context": {
                    "sort_users_by_entries": bool(extraction_context.get("sort_users_by_entries", False)),
                    "ordered_agents": int(len(extraction_context.get("ordered_agents", []))),
                },
                "resource_guard": resource_guard,
                "full": full_suite,
                "debug": debug_suite,
            }


def run_traj_extraction_suites_isolated(
    raw_ds_path: str,
    output_base_dir: str = "./dataset/processed",
) -> dict:
    """
    Run trajectory suites in a fresh Python process.

    Why:
    - parquet_processor can hold large allocator state after chunk generation.
    - running suites in-process may hit allocation failures immediately when RLIMIT_AS is applied.
    """
    module_dir = Path(__file__).resolve().parent
    result_file = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="traj_suite_result_",
            suffix=".json",
            delete=False,
        ) as tmp:
            result_file = Path(tmp.name)

        child_code = (
            "import json, sys\n"
            "from pathlib import Path\n"
            "module_dir = Path(sys.argv[4])\n"
            "if str(module_dir) not in sys.path:\n"
            "    sys.path.insert(0, str(module_dir))\n"
            "import traj_suite_runner\n"
            "result = traj_suite_runner.run_traj_extraction_suites(sys.argv[1], sys.argv[2])\n"
            "Path(sys.argv[3]).write_text(json.dumps(result), encoding='utf-8')\n"
        )

        cmd = [
            sys.executable,
            "-c",
            child_code,
            str(raw_ds_path),
            str(output_base_dir),
            str(result_file),
            str(module_dir),
        ]
        _suite_print(
            "[traj_suite] launching isolated process for trajectory extraction suites"
        )
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            return {
                "status": "failed",
                "reason": "isolated_traj_process_failed",
                "returncode": int(completed.returncode),
            }

        if result_file is None or not result_file.exists():
            return {
                "status": "failed",
                "reason": "isolated_traj_result_missing",
            }

        try:
            return json.loads(result_file.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "status": "failed",
                "reason": "isolated_traj_result_parse_failed",
                "error": str(exc),
            }
    finally:
        if result_file is not None:
            try:
                result_file.unlink(missing_ok=True)
            except Exception:
                pass
