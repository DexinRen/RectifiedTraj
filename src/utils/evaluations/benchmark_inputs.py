"""Benchmark dataset/input helpers for run_benchmarks.

Purpose:
    Keep dataset path resolution, optional parquet-to-test generation, and
    benchmark preflight validation out of the top-level benchmark entry point.

Logic Chain:
    1. Resolve dataset files/directories and load metadata.
    2. Validate explicit eval_joblist paths without guessy fallback logic.
    3. Generate processed benchmark inputs only when gen_new_test=true.
    4. Build dataset-entry packets reused by trajectory/range/chunk phases.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from .benchmark_runtime import parse_positive_int
from .benchmark_schema import (
    as_list,
    dedupe_keep_order,
    normalize_kalman_calibration_mode_token,
    split_baseline_spec,
)
from .run_context import stage


# ================================================================
# === Dataset Metadata
# ================================================================
def latest_pt_file(directory: Path) -> Path:
    """Return the newest .pt file under one dataset directory."""
    candidates = sorted(
        directory.glob("*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No .pt files found under {directory}")
    return candidates[0]


def load_metadata(pt_path: Path) -> dict:
    """Load the metadata packet from one processed .pt dataset file."""
    data = torch.load(pt_path, map_location="cpu")
    meta = data.get("metadata", {})
    if not meta:
        raise ValueError(f"Missing metadata in {pt_path}")
    return meta


def resolve_dataset(
    path_or_dir: Path,
    debug_path: Path | None,
    use_debug: bool,
) -> tuple[Path, int, int]:
    """Resolve one dataset path into concrete file path plus M/N metadata."""
    if use_debug:
        if debug_path is None:
            raise ValueError("Debug path not provided")
        pt_path = debug_path
    elif path_or_dir.is_file():
        pt_path = path_or_dir
    else:
        pt_path = latest_pt_file(path_or_dir)

    meta = load_metadata(pt_path)
    m_value = int(meta.get("n_trajectories", 0) or 0)
    n_value = int(meta.get("median_length", 0) or 0)
    if m_value <= 0 or n_value <= 0:
        raise ValueError(
            f"Invalid metadata in {pt_path}: "
            f"n_trajectories={m_value}, median_length={n_value}"
        )
    return pt_path, m_value, n_value


def infer_dataset_name_from_path(path_value: str | Path | None) -> str | None:
    """Infer processed dataset name from one processed dataset path."""
    if path_value is None:
        return None
    path = Path(path_value)
    parts = list(path.parts)
    for idx, part in enumerate(parts):
        if str(part).lower() == "processed" and idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def resolve_existing_dataset(
    name: str,
    path_value: str | None,
    debug_path: Path | None,
    use_debug: bool,
) -> list[tuple[Path, int, int]]:
    """Resolve one dataset input into concrete file entries."""
    if use_debug:
        if debug_path is None:
            raise ValueError(f"Debug path not provided for {name}")
        pt_path = debug_path
        meta = load_metadata(pt_path)
        return [(pt_path, int(meta["n_trajectories"]), int(meta["median_length"]))]

    if path_value is None or str(path_value).strip() == "":
        return []

    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path for {name} not found: {path}")

    if path.is_file():
        pt_path, m_value, n_value = resolve_dataset(path, None, use_debug=False)
        return [(pt_path, m_value, n_value)]

    pt_files = sorted(path.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found under dataset directory for {name}: {path}")

    out: list[tuple[Path, int, int]] = []
    for pt_file in pt_files:
        pt_path, m_value, n_value = resolve_dataset(pt_file, None, use_debug=False)
        out.append((pt_path, m_value, n_value))
    return out


# ================================================================
# === Explicit Path Policy
# ================================================================
def resolve_raw_dataset_dir(job: dict) -> str | None:
    """Read the explicit raw parquet dataset directory from eval_joblist."""
    data_source = job.get("data_source", {})
    if isinstance(data_source, dict):
        raw_dataset_dir = str(data_source.get("raw_dataset_dir", "") or "").strip()
        if raw_dataset_dir.lower() in {"none", "null"}:
            raw_dataset_dir = ""
        if raw_dataset_dir:
            return raw_dataset_dir

    raw_dataset_dir = str(job.get("raw_dataset_dir", "") or "").strip()
    if raw_dataset_dir.lower() in {"none", "null"}:
        raw_dataset_dir = ""
    return raw_dataset_dir or None


def default_processed_paths_for_dataset(dataset_name: str) -> tuple[str, str]:
    """Return canonical processed test directories for one dataset name."""
    root = Path("./dataset/processed") / str(dataset_name) / "test"
    return str(root / "traj_test"), str(root / "chunk_test")


def ensure_expected_test_paths_from_data_source(
    job: dict,
    traj_dirs: list[str],
    chunk_dirs: list[str],
) -> tuple[list[str], list[str]]:
    """Return only the explicit benchmark input paths provided by eval_joblist."""
    del job
    out_traj = [str(path).strip() for path in traj_dirs if str(path).strip()]
    out_chunk = [str(path).strip() for path in chunk_dirs if str(path).strip()]
    return out_traj, out_chunk


def collect_missing_inputs_for_autogen(
    job: dict,
    *,
    traj_dirs: list[str],
    chunk_dirs: list[str],
) -> list[str]:
    """Collect missing required inputs before the benchmark starts."""
    missing: list[str] = []
    traj_enabled = bool(job.get("traj_test", True))
    chunk_enabled = bool(job.get("chunk_test", False))
    range_enabled = bool(job.get("range_test", False))

    if traj_enabled:
        if not traj_dirs:
            missing.append("trajectory test path missing")
        else:
            for path in traj_dirs:
                if not Path(path).exists():
                    missing.append(f"trajectory test path does not exist: {path}")
                    break

    if chunk_enabled:
        if not chunk_dirs:
            missing.append("chunk test path missing")
        else:
            for path in chunk_dirs:
                if not Path(path).exists():
                    missing.append(f"chunk test path does not exist: {path}")
                    break

    if range_enabled:
        test_data_paths = [
            str(path).strip()
            for path in as_list(job.get("test_data_paths"))
            if str(path).strip()
        ]
        if not test_data_paths:
            fallback = str(job.get("test_data_path", "") or "").strip()
            if fallback:
                test_data_paths = [fallback]
        if not test_data_paths:
            missing.append("uncertainty test path missing")
        else:
            for path in test_data_paths:
                if not Path(path).exists():
                    missing.append(f"uncertainty test path does not exist: {path}")
                    break

    return dedupe_keep_order(missing)


def apply_generated_paths_to_job(job: dict, dataset_name: str) -> tuple[list[str], list[str]]:
    """Attach canonical generated processed paths back onto the normalized job."""
    traj_dir, chunk_dir = default_processed_paths_for_dataset(dataset_name)
    traj_dirs = [traj_dir]
    chunk_dirs = [chunk_dir]
    job["traj_dirs"] = traj_dirs
    job["traj_paths"] = {"full_traj": traj_dir}
    job["chunk_dirs"] = chunk_dirs
    job["chunk_test_dir"] = chunk_dir
    if bool(job.get("range_test", False)):
        job["test_data_path"] = str(job.get("test_data_path", "") or "").strip() or traj_dir
        job["test_data_paths"] = [job["test_data_path"]]
    return traj_dirs, chunk_dirs


# ================================================================
# === Optional Processed Data Generation
# ================================================================
def run_data_generation_mode(job: dict) -> tuple[list[str], list[str]]:
    """Generate processed eval inputs from explicit raw parquet inputs."""
    raw_dataset_path = resolve_raw_dataset_dir(job)
    if not raw_dataset_path:
        raise RuntimeError(
            "Data generation mode requires raw dataset location. "
            "Set data_source.raw_dataset_dir in eval_joblist.json."
        )

    raw_dataset_dir = Path(raw_dataset_path)
    if not raw_dataset_dir.exists() or not raw_dataset_dir.is_dir():
        raise FileNotFoundError(
            f"raw_dataset_dir does not exist or is not a directory: {raw_dataset_dir}"
        )
    if not any(raw_dataset_dir.glob("*.parquet")):
        raise RuntimeError(f"raw_dataset_dir contains no parquet files: {raw_dataset_dir}")

    data_source = job.get("data_source", {})
    if not isinstance(data_source, dict):
        data_source = {}

    raw_test_files = data_source.get("raw_test_files")
    if isinstance(raw_test_files, tuple):
        raw_test_files = list(raw_test_files)
    if isinstance(raw_test_files, str):
        raw_test_files = [raw_test_files]
    if raw_test_files is not None:
        raw_test_files = [str(value).strip() for value in raw_test_files if str(value).strip()]
        if not raw_test_files:
            raw_test_files = None

    k_value = parse_positive_int(
        job.get("K", job.get("chunk_K", 256)),
        context="K",
        default=256,
    )
    q_value = parse_positive_int(
        job.get("Q1", job.get("chunk_Q1", job.get("chunk_Q", 1))),
        context="Q1",
        default=1,
    )
    run_traj_extraction = bool(
        job.get("traj_test", True)
        or job.get("range_test", False)
        or job.get("run_baseline", True)
    )

    stage(
        "Data generation mode: running parquet_processor_test_only "
        f"(raw_dataset_dir={raw_dataset_dir}, K={k_value}, Q1={q_value}, run_traj_extraction={run_traj_extraction})"
    )

    from utils.data_processor.parquet_processor import parquet_processor_test_only

    parquet_processor_test_only(
        K=k_value,
        Q=q_value,
        raw_ds_path=str(raw_dataset_dir),
        test_files=raw_test_files,
        run_traj_extraction=run_traj_extraction,
    )
    return apply_generated_paths_to_job(job, dataset_name=raw_dataset_dir.name)


def generate_full_traj(output_dir: Path, use_new_traj: dict) -> Path:
    """Generate one full trajectory dataset from raw NUMOSIM parquet input."""
    from utils.data_processor.traj_extractor import traj_extractor

    cfg = use_new_traj.get("full_traj", {}) if use_new_traj else {}
    if "M" not in cfg or "N" not in cfg:
        raise ValueError("use_new_traj.full_traj must include M and N when gen_new_test is true.")
    result = traj_extractor(
        parquet_dir="./dataset/raw/NUMOSIM_Kanto",
        M=int(cfg["M"]),
        N=int(cfg["N"]),
        output_dir=str(output_dir),
    )
    return Path(result["output_file"])


# ================================================================
# === Benchmark Preflight
# ================================================================
def discover_models_for_preflight(model_root: Path, model_names: list | None) -> list[str]:
    """Resolve model names for preflight checkpoint validation."""
    if model_names is not None:
        return [str(name).strip() for name in model_names if str(name).strip()]

    discovered: list[str] = []
    for model_dir in sorted(model_root.iterdir()):
        if not model_dir.is_dir():
            continue
        has_ckpt = False
        for ckpt_dir_name in ("best_ckpt", "ckpts"):
            ckpt_dir = model_dir / ckpt_dir_name
            if ckpt_dir.exists() and any(ckpt_dir.glob("*_full.pt")):
                has_ckpt = True
                break
        if has_ckpt:
            discovered.append(model_dir.name)
    return discovered


def find_model_checkpoint_for_preflight(model_dir: Path) -> Path | None:
    """Return one representative checkpoint path for preflight validation."""
    best_ckpt_dir = model_dir / "best_ckpt"
    if best_ckpt_dir.exists():
        best = sorted(best_ckpt_dir.glob("*_full.pt"))
        if best:
            return best[0]

    ckpts_dir = model_dir / "ckpts"
    if ckpts_dir.exists():
        all_ckpts = sorted(ckpts_dir.glob("*_full.pt"), key=lambda path: path.stat().st_mtime)
        if all_ckpts:
            return all_ckpts[-1]
    return None


def load_model_k_for_preflight(model_dir: Path) -> int | None:
    """Read K from a model config when available."""
    config_path = model_dir / "log" / "config.json"
    if not config_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        k_value = int(config.get("K", 0) or 0)
    except Exception:
        return None
    return k_value if k_value > 0 else None


def validate_buckle_grid_for_preflight(
    *,
    errors: list[str],
    label: str,
    k_value: int | None,
    q1_values: list,
    q2_values: list,
) -> None:
    """Validate byte-level Q1/Q2 combinations before launching batch workers."""
    if k_value is None:
        return
    max_total_q = max((int(k_value) - 1) // 8, 0)
    for q1_raw in q1_values:
        for q2_raw in q2_values:
            try:
                q1 = int(q1_raw)
                q2 = int(q2_raw)
            except Exception:
                errors.append(f"{label}: Q values must be integers, got Q1={q1_raw!r} Q2={q2_raw!r}.")
                continue
            if q1 < 0 or q2 < 0:
                errors.append(f"{label}: Q values must be nonnegative, got Q1={q1} Q2={q2}.")
                continue
            if k_value <= (q1 + q2) * 8:
                errors.append(
                    f"{label}: invalid buckle Q1={q1} Q2={q2} for K={k_value}; "
                    f"requires Q1+Q2 <= {max_total_q}."
                )


def preflight_validate_job(
    job: dict,
    *,
    model_groups: list[dict],
    traj_dirs: list[str],
    chunk_dirs: list[str],
    classic_baselines: list[str],
) -> None:
    """Run strict benchmark preflight validation before long eval loops."""
    from baseline import classic as classic_baseline

    errors: list[str] = []
    for group in model_groups:
        data_hypothesis = str(group.get("data_hypothesis", "RectifiedTraj"))
        model_root = str(group.get("model_root", "")).strip()
        model_names = group.get("model_names")

        if not model_root:
            errors.append(f"{data_hypothesis}: model_root is empty.")
            continue

        root = Path(model_root)
        if not root.exists() or not root.is_dir():
            errors.append(f"{data_hypothesis}: model_root is missing or not a directory: {root}")
            continue

        resolved_models = discover_models_for_preflight(root, model_names)
        if not resolved_models and not bool(job.get("run_baseline", True)):
            errors.append(
                f"{data_hypothesis}: no model checkpoints found and run_baseline=false. "
                "Provide valid models or enable baselines."
            )
        for model_name in resolved_models:
            model_dir = root / model_name
            if not model_dir.exists():
                errors.append(f"{data_hypothesis}: model directory missing: {model_dir}")
                continue
            ckpt = find_model_checkpoint_for_preflight(model_dir)
            if ckpt is None:
                errors.append(f"{data_hypothesis}: no *_full.pt checkpoint found for model: {model_dir}")
            validate_buckle_grid_for_preflight(
                errors=errors,
                label=f"{data_hypothesis}/{model_name}",
                k_value=load_model_k_for_preflight(model_dir),
                q1_values=as_list(group.get("Q1")) or [1],
                q2_values=as_list(group.get("Q2")) or [12],
            )

    traj_enabled = bool(job.get("traj_test", True))
    chunk_enabled = bool(job.get("chunk_test", False))
    range_enabled = bool(job.get("range_test", False))

    if traj_enabled:
        if not traj_dirs and not bool(job.get("gen_new_test", False)):
            errors.append("traj_test=true but no trajectory test path is configured (test_files.traj_files).")
        for path in traj_dirs:
            if not Path(path).exists():
                errors.append(f"trajectory test path does not exist: {path}")

    if chunk_enabled:
        if not chunk_dirs:
            errors.append("chunk_test=true but no chunk test path is configured (test_files.chunk_files).")
        for path in chunk_dirs:
            if not Path(path).exists():
                errors.append(f"chunk test path does not exist: {path}")

    if range_enabled:
        test_data_paths = [
            str(path).strip()
            for path in as_list(job.get("test_data_paths"))
            if str(path).strip()
        ]
        if not test_data_paths:
            fallback = str(job.get("test_data_path", "") or "").strip()
            if fallback:
                test_data_paths = [fallback]
        if not test_data_paths:
            errors.append("range_test=true but test_data_path/uncertainty_path is missing.")
        else:
            for path in test_data_paths:
                if not Path(path).exists():
                    errors.append(f"uncertainty test path does not exist: {path}")
                    break

    if bool(job.get("run_baseline", True)):
        kalman_specs = [
            spec
            for spec in classic_baselines
            if split_baseline_spec(spec)[0] in {"kalman_filter", "kalman_rts"}
        ]

        dataset_hints: list[str] = []
        for path in traj_dirs + chunk_dirs:
            hint = infer_dataset_name_from_path(path)
            if hint:
                dataset_hints.append(hint)
        dataset_hints = dedupe_keep_order(dataset_hints)

        for spec in kalman_specs:
            base_name, mode_name, _display = split_baseline_spec(spec)
            effective_mode = (
                normalize_kalman_calibration_mode_token(mode_name)
                if base_name == "kalman_rts"
                else "dataset"
            )

            if effective_mode == "numosim_kanto":
                source_dataset = (
                    str(os.getenv("KALMAN_RTS_CALIBRATION_DATASET", "")).strip()
                    or "NUMOSIM_Kanto"
                )
                calibration_file = classic_baseline.resolve_kalman_calibration_file_from_state(
                    dataset_name_hint=source_dataset
                )
                if not calibration_file:
                    errors.append(
                        f"{base_name}@numosim_kanto requires calibration artifact for "
                        f"{source_dataset}, but none was found under dataset/state."
                    )
                continue

            if not dataset_hints:
                errors.append(
                    f"{base_name}@dataset selected but dataset name cannot be inferred from test paths."
                )
                continue

            for dataset_name in dataset_hints:
                calibration_file = classic_baseline.resolve_kalman_calibration_file_from_state(
                    dataset_name_hint=dataset_name
                )
                if not calibration_file:
                    errors.append(
                        f"{base_name}@dataset requires calibration artifact for "
                        f"{dataset_name}, but none was found under dataset/state."
                    )

    if errors:
        joined = "\n - " + "\n - ".join(errors)
        raise RuntimeError(f"Preflight validation failed:{joined}")


# ================================================================
# === Range-Test Dataset Packets
# ================================================================
def resolve_bounded_dataset_entries(job: dict) -> list[dict]:
    """Build range-test dataset packets from explicit dataset paths."""
    path_values = [
        str(path).strip()
        for path in as_list(job.get("test_data_paths"))
        if str(path).strip()
    ]
    if not path_values:
        path_values = [
            str(path).strip()
            for path in as_list(job.get("traj_dirs"))
            if str(path).strip()
        ]
    if not path_values:
        fallback = str(job.get("test_data_path", "") or "").strip()
        if fallback:
            path_values = [fallback]
    if not path_values:
        raise ValueError("No bounded test dataset path is configured.")

    entries: list[dict] = []
    for idx, path_value in enumerate(path_values):
        resolved = resolve_existing_dataset(
            name=f"range_{idx}",
            path_value=path_value,
            debug_path=None,
            use_debug=False,
        )
        for file_idx, entry in enumerate(resolved):
            dataset_family = infer_dataset_name_from_path(entry[0]) or infer_dataset_name_from_path(path_value)
            dataset_stem = Path(entry[0]).stem or Path(path_value).stem or f"range_{idx}_{file_idx}"
            if dataset_family:
                dataset_name = f"{dataset_family}_{dataset_stem}"
            else:
                dataset_name = dataset_stem
            entries.append(
                {
                    "name": dataset_name,
                    "path": entry[0],
                    "M": int(entry[1]),
                    "N": int(entry[2]),
                }
            )
    return entries


def build_bounded_manual_configs(job_list: dict) -> list[dict]:
    """Build manual Q1/Q2 configs for range-test evaluation."""
    q1_values = as_list(job_list.get("Q1")) or [1]
    q2_values = as_list(job_list.get("Q2")) or [12]
    manual_configs: list[dict] = []
    for q1_value in q1_values:
        for q2_value in q2_values:
            manual_configs.append(
                {
                    "Q1": int(q1_value),
                    "Q2": int(q2_value),
                }
            )
    return manual_configs
