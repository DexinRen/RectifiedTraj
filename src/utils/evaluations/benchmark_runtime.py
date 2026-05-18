"""Benchmark runtime helpers for run_benchmarks.

Purpose:
    Hold device policy, batch parallelism, and global Kalman calibration
    environment setup outside the benchmark entry point.

Logic Chain:
    1. Resolve effective runtime device.
    2. Apply strict integer runtime limits.
    3. Publish the active Kalman calibration mode to child components.
"""

from __future__ import annotations

import logging
import os

import torch

from .benchmark_schema import (
    normalize_kalman_calibration_mode_token,
    runtime_cfg,
    runtime_device_for_defaults,
    runtime_device_requested,
)
from .run_context import stage


# ================================================================
# === Strict Value Parsing
# ================================================================
def parse_positive_int(value, *, context: str, default: int | None = None) -> int:
    """Parse one positive integer config value with strict validation."""
    if value is None:
        if default is None:
            raise ValueError(f"{context} is required and must be a positive integer.")
        value = default
    out = int(value)
    if out <= 0:
        raise ValueError(f"{context} must be a positive integer, got {value!r}.")
    return out


# ================================================================
# === Device Policy
# ================================================================
def runtime_device_effective(job: dict) -> str:
    """Resolve the effective benchmark runtime device."""
    cfg = runtime_cfg(job)
    requested = runtime_device_requested(job)
    strict_init = bool(cfg.get("strict_init", True))
    if requested == "cuda" and not torch.cuda.is_available():
        if strict_init:
            raise RuntimeError(
                "runtime.device is set to 'cuda' but CUDA is not available. "
                "Set runtime.device='cpu' for CPU-only runs."
            )
        logging.warning(
            "runtime.device=cuda requested but CUDA unavailable; falling back to cpu "
            "(strict_init=false)."
        )
        return "cpu"
    return requested


def configure_encoder_decoder_device(job: dict) -> str:
    """Push the resolved runtime device into encoder_decoder global state."""
    from encoder_decoder import get_runtime_device, set_runtime_device

    effective = runtime_device_effective(job)
    cfg = runtime_cfg(job)
    cfg["device_effective"] = effective
    job["runtime"] = cfg
    os.environ["RECTIFIEDTRAJ_RUNTIME_DEVICE_EFFECTIVE"] = effective
    os.environ["RECTIFIEDTRAJ_DEVICE"] = effective
    set_runtime_device(effective)
    return get_runtime_device()


def resolve_cpu_traj_caps(job: dict, default_m: int = 50, default_n: int = 1500) -> tuple[int, int]:
    """Read strict CPU trajectory-size caps from runtime config."""
    cfg = runtime_cfg(job)
    m_value = parse_positive_int(
        cfg.get("cpu_traj_M"),
        context="runtime.cpu_traj_M",
        default=default_m,
    )
    n_value = parse_positive_int(
        cfg.get("cpu_traj_N"),
        context="runtime.cpu_traj_N",
        default=default_n,
    )
    return m_value, n_value


def cpu_shrink_enabled(job: dict) -> bool:
    """Check whether CPU-only runs should cap dataset sizes."""
    if runtime_device_for_defaults(job) != "cpu":
        return False
    return bool(runtime_cfg(job).get("cpu_shrink", True))


def apply_cpu_dataset_caps(job: dict, datasets: list[dict]) -> list[dict]:
    """Apply CPU dataset caps to resolved dataset entries."""
    if not cpu_shrink_enabled(job):
        return datasets

    cap_m, cap_n = resolve_cpu_traj_caps(job)
    capped: list[dict] = []
    for entry in datasets:
        m_old = int(entry.get("M", cap_m))
        n_old = int(entry.get("N", cap_n))
        m_new = min(m_old, cap_m)
        n_new = min(n_old, cap_n)
        if m_new != m_old or n_new != n_old:
            stage(
                "CPU shrink applied | dataset=%s M:%d->%d N:%d->%d"
                % (entry.get("name", "unknown"), m_old, m_new, n_old, n_new)
            )
        capped.append(
            {
                "name": entry.get("name"),
                "path": entry.get("path"),
                "M": int(m_new),
                "N": int(n_new),
            }
        )
    return capped


def resolve_traj_parallel(job: dict, default_value: int = 4) -> int:
    """Resolve strict trajectory batch parallelism from runtime config."""
    cfg = runtime_cfg(job)
    raw = cfg.get("traj_parallel", job.get("traj_parallel", default_value))
    return parse_positive_int(
        raw,
        context="runtime.traj_parallel",
        default=default_value,
    )


# ================================================================
# === Global Evaluation Runtime
# ================================================================
def apply_kalman_calibration_overrides(job: dict) -> tuple[str, str]:
    """Publish one active Kalman calibration mode for the whole benchmark run."""
    raw_mode = str(job.get("kalman_calibration_mode", "") or "").strip()
    dataset = str(job.get("kalman_calibration_dataset", "") or "").strip()
    mode = normalize_kalman_calibration_mode_token(raw_mode) if raw_mode else ""

    if mode:
        os.environ["KALMAN_RTS_CALIBRATION_MODE"] = mode
    mode_effective = str(os.getenv("KALMAN_RTS_CALIBRATION_MODE", "dataset")).strip()

    if dataset:
        os.environ["KALMAN_RTS_CALIBRATION_DATASET"] = dataset
    dataset_effective = str(
        os.getenv("KALMAN_RTS_CALIBRATION_DATASET", "NUMOSIM_Kanto")
    ).strip()
    return mode_effective, dataset_effective

