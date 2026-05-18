"""Benchmark run-context helpers for run_benchmarks.

Purpose:
    Hold the top-level benchmark run metadata and console-stage helpers.

Logic Chain:
    1. Emit stage messages in the shared progress-aware stream.
    2. Build deterministic run-folder labels.
    3. Snapshot host/runtime metadata for later inspection.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import psutil
import torch

from .benchmark_schema import normalize_device_label
from .progress import ProgressTracker


# ================================================================
# === Progress Logging
# ================================================================
def stage(message: str) -> None:
    """Emit one benchmark stage message through the progress-aware logger."""
    ProgressTracker._emit_log_message(sys.stdout, f"[run_benchmarks] {message}")


# ================================================================
# === Output Folder Naming
# ================================================================
def safe_folder_token(value: str | None, fallback: str = "unknown") -> str:
    """Convert free text into a filesystem-safe folder token."""
    text = str(value or "").strip()
    if not text:
        text = fallback
    out: list[str] = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-"}:
            out.append(ch)
        else:
            out.append("_")
    token = "".join(out).strip("_")
    return token or fallback


def resolve_run_scope_label(job: dict) -> str:
    """Summarize which benchmark phases are enabled for the run."""
    traj_enabled = bool(job.get("traj_test", True))
    chunk_enabled = bool(job.get("chunk_test", False))
    if traj_enabled and chunk_enabled:
        return "full"
    if chunk_enabled:
        return "chunk"
    if traj_enabled:
        return "traj"
    return "full"


def build_result_folder_name(
    job: dict,
    traj_dirs: list[str],
    chunk_dirs: list[str],
    *,
    runtime_device: str | None,
    timestamp: str,
) -> str:
    """Build the top-level benchmark output folder name."""
    del traj_dirs
    del chunk_dirs
    scope = resolve_run_scope_label(job)
    device_norm = normalize_device_label(runtime_device)
    if device_norm == "cpu":
        device_label = "CPU"
    elif device_norm == "cuda":
        device_label = "CUDA"
    else:
        device_label = "UNKNOWN"
    return (
        f"{safe_folder_token(scope, 'full')}_"
        f"{safe_folder_token(device_label, 'unknown_device')}_"
        f"{safe_folder_token(timestamp, 'ts')}"
    )


# ================================================================
# === System Snapshot
# ================================================================
def read_cpu_model_name() -> str:
    """Read the CPU model name from /proc/cpuinfo when available."""
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        for line in cpuinfo_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    model = parts[1].strip()
                    if model:
                        return model
    return platform.processor() or "unknown"


def collect_system_info(runtime_device: str) -> dict:
    """Collect runtime host/device metadata for one benchmark run."""
    vm = psutil.virtual_memory()
    info = {
        "test_timestamp": datetime.now().isoformat(),
        "runtime_device_effective": normalize_device_label(runtime_device),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "os_system": platform.system(),
        "os_release": platform.release(),
        "python_version": sys.version.split()[0],
        "torch_version": getattr(torch, "__version__", "unknown"),
        "torch_cuda_version": str(getattr(torch.version, "cuda", None)),
        "cuda_available": bool(torch.cuda.is_available()),
        "cpu_model": read_cpu_model_name(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gb": round(float(vm.total) / (1024.0 ** 3), 2),
    }

    gpu_names: list[str] = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            gpu_names.append(str(torch.cuda.get_device_name(idx)))
    info["gpu_count"] = len(gpu_names)
    info["gpu_names"] = gpu_names
    return info


def write_system_info(output_dir: Path, runtime_device: str) -> Path:
    """Write one system-info JSON snapshot into the run output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "system_info.json"
    payload = collect_system_info(runtime_device)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
