#!/usr/bin/env python3
"""Entry point for running evaluations using eval_joblist.json."""

import argparse
import time
import json
import logging
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
import torch
import numpy as np
import psutil

from utils.evaluations.evaluation_manager import TestManager
from utils.evaluations.wandb_logger import log_run_to_wandb
from utils.data_loader_standalone import StandaloneDataLoader
from utils.data_processor.traj_extractor import traj_extractor


FULL_TRAJ_DIR = Path("./dataset/processed/NUMOSIM_Kanto/test/traj_test")
DEBUG_FULL_TRAJ = FULL_TRAJ_DIR / "traj_debug_mini.pt"

DEFAULT_TIME_NPY = Path("./dataset/time_test/source_list.npy")
DEFAULT_TIME_LOG = Path("./bin/log/time_test.csv")
DEFAULT_CLASSIC_BASELINES = [
    "kalman_rts",
    "hampel",
    "savgol",
    "spline",
    "raw",
]
ALLOWED_CLASSIC_BASELINES = list(DEFAULT_CLASSIC_BASELINES) + ["valhalla_meili"]
DIFFTRAJ_ENABLED = False


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


def _normalize_positive_float_list(values, *, context: str) -> list[float]:
    """Normalize a numeric list and enforce strictly-positive values."""
    out: list[float] = []
    for raw_value in _as_list(values):
        value = float(raw_value)
        if value <= 0.0:
            raise ValueError(f"{context} must be > 0, got {raw_value!r}.")
        out.append(value)
    return out


def _resolve_t_delta_values(
    raw_scope: dict | None,
    *,
    default_scope: dict | None = None,
    default_values: list[float] | None = None,
    context: str,
) -> list[float]:
    """
    Resolve denoising delta-t values from schema aliases.

    Priority:
      1) delta_t
      2) t_delta
      3) step (legacy: interpreted as number of denoise steps, mapped to 1/step)
      4) default scope
      5) fallback default_values
    """
    fallback = list(default_values) if default_values is not None else [1.0]

    def _pick(scope: dict | None):
        if not isinstance(scope, dict):
            return None, None
        if "delta_t" in scope:
            return scope.get("delta_t"), "delta_t"
        if "t_delta" in scope:
            return scope.get("t_delta"), "t_delta"
        if "step" in scope:
            return scope.get("step"), "step"
        return None, None

    selected_values, key_name = _pick(raw_scope)
    if key_name is None:
        selected_values, key_name = _pick(default_scope)
    if key_name is None:
        return fallback

    values = _normalize_positive_float_list(
        selected_values,
        context=f"{context}.{key_name}",
    )
    if not values:
        return fallback
    if key_name == "step":
        return [1.0 / step_count for step_count in values]
    return values


def _normalize_data_hypothesis(raw, default: str = "RectifiedTraj") -> str:
    token = str(raw if raw is not None else "").strip().lower().replace("-", "_")
    if token in {"", "rf", "rectified_flow", "rectified", "rectifiedtraj", "rectified_traj"}:
        return "RectifiedTraj"
    if token in {"rr", "residualreg", "residual_reg", "residual", "residual_regression"}:
        return "ResidualReg"
    text = str(raw).strip() if raw is not None else ""
    return text if text else str(default)


def _validate_supported_data_hypothesis(data_hypothesis: str, *, context: str) -> None:
    """Validate that the hypothesis token is one of the supported canonical names."""
    if data_hypothesis in {"RectifiedTraj", "ResidualReg"}:
        return
    raise ValueError(
        f"{context} has unsupported data_hypothesis={data_hypothesis!r}. "
        "Supported values: RectifiedTraj, ResidualReg."
    )


def _validate_model_root_matches_hypothesis(model_root: str, data_hypothesis: str, *, context: str) -> None:
    """Validate explicit hypothesis folder names in model_root."""
    root_name = Path(model_root).name.strip().lower()
    if root_name in {"rectifiedtraj", "residualreg"} and root_name != data_hypothesis.lower():
        raise ValueError(
            f"{context} has model_root={model_root!r} but data_hypothesis={data_hypothesis!r}. "
            "model_root hypothesis folder must match data_hypothesis."
        )


def _default_model_root_for_hypothesis(data_hypothesis: str) -> str:
    return str(Path("./bin/model") / _normalize_data_hypothesis(data_hypothesis))


def _normalize_model_root_with_hypothesis(model_root_value, data_hypothesis: str) -> str:
    model_root_text = str(model_root_value).strip()
    if not model_root_text:
        return _default_model_root_for_hypothesis(data_hypothesis)
    base = Path(model_root_text)
    norm = _normalize_data_hypothesis(data_hypothesis)
    if base.as_posix().rstrip("/") in {"./bin/model", "bin/model"}:
        return str(base / norm)
    return str(base)


def _normalize_methods_for_hypothesis(methods: list, data_hypothesis: str) -> list[str]:
    """Normalize rectified-traj methods according to hypothesis policy."""
    raw_methods = _as_list(methods)
    if data_hypothesis != "ResidualReg":
        return raw_methods or ["BF", "DF"]

    # ResidualReg is one-shot denoising. Treat BF/DF/RR/ONE_SHOT as one path.
    alias_set = {"BF", "DF", "RR", "ONE_SHOT"}
    normalized: list[str] = []
    for token in raw_methods:
        method = str(token).strip().upper()
        if not method:
            continue
        if method not in alias_set:
            raise ValueError(
                f"Unsupported ResidualReg method token: {token!r}. "
                "Use one of: BF, DF, RR, ONE_SHOT."
            )
        normalized.append("DF")

    out = _dedupe_keep_order(normalized)
    return out or ["DF"]


def _normalize_model_group_schema_entry(
    raw_group: dict,
    *,
    default_group: dict | None,
    context: str,
) -> dict:
    """Normalize one learned-model group entry from eval_joblist schema."""
    if not isinstance(raw_group, dict):
        raise ValueError(f"{context} must be a JSON object.")

    default_hypothesis = (
        str(default_group.get("data_hypothesis", "RectifiedTraj"))
        if isinstance(default_group, dict)
        else "RectifiedTraj"
    )
    data_hypothesis = _normalize_data_hypothesis(
        raw_group.get(
            "data_hypothesis",
            raw_group.get("data_hypothetis", default_hypothesis),
        )
    )
    _validate_supported_data_hypothesis(data_hypothesis, context=context)

    if "model_root" in raw_group:
        model_root = _normalize_model_root_with_hypothesis(
            raw_group.get("model_root"),
            data_hypothesis,
        )
    else:
        model_root = _default_model_root_for_hypothesis(data_hypothesis)
    _validate_model_root_matches_hypothesis(
        model_root,
        data_hypothesis,
        context=context,
    )

    if "models" in raw_group or "model_names" in raw_group:
        raw_models = raw_group.get("models", raw_group.get("model_names"))
    else:
        raw_models = default_group.get("model_names") if isinstance(default_group, dict) else None
    model_names = None if raw_models is None else _as_list(raw_models)

    if "method" in raw_group or "methods" in raw_group:
        raw_methods = raw_group.get("method", raw_group.get("methods"))
    else:
        raw_methods = (
            default_group.get("methods")
            if isinstance(default_group, dict)
            else ["BF", "DF"]
        )
    methods = _normalize_methods_for_hypothesis(
        raw_methods if raw_methods is not None else ["BF", "DF"],
        data_hypothesis,
    )

    raw_q1 = raw_group.get(
        "Q1",
        default_group.get("Q1", [1]) if isinstance(default_group, dict) else [1],
    )
    raw_q2 = raw_group.get(
        "Q2",
        default_group.get("Q2", [12]) if isinstance(default_group, dict) else [12],
    )
    group = {
        "data_hypothesis": data_hypothesis,
        "model_root": model_root,
        "model_names": model_names,
        "methods": methods,
        "Q1": _as_list(raw_q1) or [1],
        "Q2": _as_list(raw_q2) or [12],
        "t_delta": _resolve_t_delta_values(
            raw_group,
            default_scope=default_group if isinstance(default_group, dict) else None,
            default_values=[1.0],
            context=context,
        ),
    }
    return group


def _build_primary_model_group_from_job(job: dict) -> dict:
    """Build the primary learned-model group from normalized job fields."""
    return {
        "data_hypothesis": _normalize_data_hypothesis(job.get("data_hypothesis", "RectifiedTraj")),
        "model_root": str(job.get("model_root", _default_model_root_for_hypothesis("RectifiedTraj"))),
        "model_names": job.get("model_names"),
        "methods": list(job.get("methods", ["BF", "DF"])),
        "Q1": _as_list(job.get("Q1")) or [1],
        "Q2": _as_list(job.get("Q2")) or [12],
        "t_delta": _as_list(job.get("t_delta")) or [1.0],
    }


def _dedupe_model_groups(groups: list[dict]) -> list[dict]:
    """De-duplicate groups by hypothesis/root while preserving order."""
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for group in groups:
        key = (
            str(group.get("data_hypothesis", "")).strip(),
            str(group.get("model_root", "")).strip(),
        )
        if not key[0] or not key[1] or key in seen:
            continue
        seen.add(key)
        out.append(group)
    return out


def _dedupe_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        item = str(raw).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _normalize_kalman_calibration_mode_token(raw: str | None) -> str:
    token = str(raw or "").strip().lower().replace("-", "_")
    if token in {"", "default", "textbook", "textbook_default"}:
        return "textbook_default"
    if token in {"numosim", "numosim_kanto", "numosim_only", "kanto"}:
        return "numosim_kanto"
    if token in {"on_dataset", "dataset", "per_dataset"}:
        return "dataset"
    raise ValueError(
        f"Unsupported Kalman calibration mode token: {raw}. "
        "Use default/textbook_default, NUMOSIM/numosim_kanto, or on_dataset/dataset."
    )


def _split_baseline_spec(spec: str) -> tuple[str, str | None, str]:
    token = str(spec).strip()
    if not token:
        return "", None, ""
    if "@" in token:
        base, mode = token.split("@", 1)
    else:
        base, mode = token, None
    base_norm = str(base).strip().lower()
    mode_norm = None
    if base_norm == "kalman_rts":
        mode_norm = _normalize_kalman_calibration_mode_token(mode)
    display = f"{base_norm}@{mode_norm}" if (base_norm == "kalman_rts" and mode_norm) else base_norm
    return base_norm, mode_norm, display


def _expand_baseline_specs(
    baseline_models,
    calibration_cfg,
) -> list[str]:
    models = _as_list(baseline_models)
    if not models:
        models = list(DEFAULT_CLASSIC_BASELINES)
    if not isinstance(calibration_cfg, dict):
        calibration_cfg = {}

    out: list[str] = []
    for raw_model in models:
        model = str(raw_model).strip().lower()
        if not model:
            continue
        if model != "kalman_rts":
            out.append(model)
            continue
        raw_modes = calibration_cfg.get("kalman_rts")
        modes = _as_list(raw_modes) if raw_modes is not None else ["textbook_default"]
        if not modes:
            modes = ["textbook_default"]
        for mode in modes:
            mode_norm = _normalize_kalman_calibration_mode_token(str(mode))
            out.append(f"kalman_rts@{mode_norm}")
    return _dedupe_keep_order(out)


def _extract_test_dirs(test_files) -> tuple[list[str], list[str], str | None]:
    traj_dirs: list[str] = []
    chunk_dirs: list[str] = []
    uncertainty_path: str | None = None

    def _push_paths(target: list[str], values) -> None:
        for v in _as_list(values):
            p = str(v).strip()
            if p:
                target.append(p)

    traj_aliases = (
        "traj_files",
        "traj_file",
        "traj_dirs",
        "traj_dir",
        "traj_folder_dirs",
        "traj_folder_dir",
        "traj folder dirs",
        "traj folder dir",
    )
    chunk_aliases = (
        "chunk_files",
        "chunk_file",
        "chunk_dirs",
        "chunk_dir",
        "chunk_folder_dirs",
        "chunk_folder_dir",
        "chunk folder dirs",
        "chunk folder dir",
    )

    if isinstance(test_files, list):
        for row in test_files:
            if not isinstance(row, dict):
                continue
            for key in traj_aliases:
                _push_paths(traj_dirs, row.get(key))
            for key in chunk_aliases:
                _push_paths(chunk_dirs, row.get(key))
            u = row.get("uncertainty_path")
            if uncertainty_path is None and isinstance(u, str) and u.strip():
                uncertainty_path = str(u).strip()
    elif isinstance(test_files, dict):
        for key in traj_aliases:
            _push_paths(traj_dirs, test_files.get(key))
        for key in chunk_aliases:
            _push_paths(chunk_dirs, test_files.get(key))
        u = test_files.get("uncertainty_path")
        if isinstance(u, str) and u.strip():
            uncertainty_path = str(u).strip()
        datasets = test_files.get("datasets")
        if isinstance(datasets, list):
            sub_traj, sub_chunk, sub_u = _extract_test_dirs(datasets)
            traj_dirs.extend(sub_traj)
            chunk_dirs.extend(sub_chunk)
            if uncertainty_path is None:
                uncertainty_path = sub_u

    return _dedupe_keep_order(traj_dirs), _dedupe_keep_order(chunk_dirs), uncertainty_path


def _normalize_job_schema(raw_job: dict) -> dict:
    if not isinstance(raw_job, dict):
        raise ValueError("eval_joblist.json must be a JSON object.")

    has_new_schema = any(
        key in raw_job
        for key in (
            "test_type",
            "test_items",
            "test_files",
            "rectifiedtraj",
            "baseline",
            "baselines",
            "data_source",
        )
    )
    if not has_new_schema:
        job = dict(raw_job)
        job.setdefault("test_type", "exact")
        job.setdefault("traj_test", True)
        data_hypothesis = _normalize_data_hypothesis(
            job.get("data_hypothesis", job.get("data_hypothetis", "RectifiedTraj"))
        )
        _validate_supported_data_hypothesis(data_hypothesis, context="eval_joblist (legacy schema)")
        job["data_hypothesis"] = data_hypothesis
        if "model_root" in job:
            job["model_root"] = _normalize_model_root_with_hypothesis(job.get("model_root"), data_hypothesis)
        else:
            job["model_root"] = _default_model_root_for_hypothesis(data_hypothesis)
        _validate_model_root_matches_hypothesis(
            job["model_root"],
            data_hypothesis,
            context="eval_joblist (legacy schema)",
        )
        job["methods"] = _normalize_methods_for_hypothesis(job.get("methods", ["BF", "DF"]), data_hypothesis)
        job["t_delta"] = _resolve_t_delta_values(
            job,
            default_values=[1.0],
            context="eval_joblist (legacy schema)",
        )
        runtime_cfg = raw_job.get("runtime", {})
        if not isinstance(runtime_cfg, dict):
            runtime_cfg = {}
        job["runtime"] = dict(runtime_cfg)
        job["runtime"].setdefault("device", "cuda")
        job["runtime"].setdefault("strict_init", True)
        raw_dataset_dir = str(job.get("raw_dataset_dir", "") or "").strip()
        if raw_dataset_dir:
            job["raw_dataset_dir"] = raw_dataset_dir
            job["data_source"] = {
                "raw_dataset_dir": raw_dataset_dir,
                "raw_map_path": None,
                "raw_map_dir": "./dataset/raw_map",
                "raw_test_files": None,
                "map_padding_km": 5.0,
            }
        job["model_groups"] = [_build_primary_model_group_from_job(job)]
        return job

    job = dict(raw_job)
    test_type = str(raw_job.get("test_type", "exact")).strip().lower()
    if test_type not in {"exact", "uncertainty", "tuning"}:
        raise ValueError("test_type must be one of: exact, uncertainty, tuning")

    test_items = raw_job.get("test_items", {})
    if not isinstance(test_items, dict):
        test_items = {}
    rectified = raw_job.get("rectifiedtraj", {})
    if not isinstance(rectified, dict):
        rectified = {}
    baseline_cfg = raw_job.get("baseline", raw_job.get("baselines", {}))
    if not isinstance(baseline_cfg, dict):
        baseline_cfg = {}
    data_source = raw_job.get("data_source", {})
    if not isinstance(data_source, dict):
        data_source = {}

    traj_dirs, chunk_dirs, uncertainty_path = _extract_test_dirs(raw_job.get("test_files"))

    # RectifiedTraj settings
    data_hypothesis = _normalize_data_hypothesis(
        rectified.get(
            "data_hypothesis",
            rectified.get(
                "data_hypothetis",
                raw_job.get("data_hypothesis", raw_job.get("data_hypothetis", "RectifiedTraj")),
            ),
        )
    )
    _validate_supported_data_hypothesis(data_hypothesis, context="eval_joblist (new schema)")
    job["data_hypothesis"] = data_hypothesis
    model_root_raw = rectified.get("model_root", raw_job.get("model_root"))
    if model_root_raw is None:
        job["model_root"] = _default_model_root_for_hypothesis(data_hypothesis)
    else:
        job["model_root"] = _normalize_model_root_with_hypothesis(model_root_raw, data_hypothesis)
    _validate_model_root_matches_hypothesis(
        job["model_root"],
        data_hypothesis,
        context="eval_joblist (new schema)",
    )
    models = rectified.get("models", raw_job.get("model_names"))
    job["model_names"] = None if models is None else _as_list(models)
    methods = rectified.get("method", rectified.get("methods", raw_job.get("methods", ["BF", "DF"])))
    job["methods"] = _normalize_methods_for_hypothesis(methods, data_hypothesis)
    job["Q1"] = _as_list(rectified.get("Q1", raw_job.get("Q1", [1]))) or [1]
    job["Q2"] = _as_list(rectified.get("Q2", raw_job.get("Q2", [1, 12, 24]))) or [1, 12, 24]
    # Public schema uses `delta_t`; `t_delta` is accepted as alias.
    # Legacy `step` is interpreted as step count and converted via delta_t=1/step.
    job["t_delta"] = _resolve_t_delta_values(
        rectified,
        default_scope=raw_job,
        default_values=[1.0],
        context="eval_joblist.rectifiedtraj",
    )

    # Dataset/file selectors
    if traj_dirs:
        job["traj_dirs"] = traj_dirs
        job["traj_paths"] = {"full_traj": traj_dirs[0]}
    if chunk_dirs:
        job["chunk_dirs"] = chunk_dirs
        job["chunk_test_dir"] = chunk_dirs[0]

    # Baselines + calibration variants
    baseline_models = baseline_cfg.get("models", raw_job.get("classic_baselines"))
    baseline_calibration = baseline_cfg.get("calibration", {})
    expanded_baselines = _expand_baseline_specs(baseline_models, baseline_calibration)
    job["classic_baselines"] = expanded_baselines

    raw_dataset_dir = str(
        data_source.get("raw_dataset_dir", raw_job.get("raw_dataset_dir", ""))
    ).strip()
    if raw_dataset_dir:
        job["raw_dataset_dir"] = raw_dataset_dir
    raw_map_path = data_source.get("raw_map_path", raw_job.get("raw_map_path"))
    if isinstance(raw_map_path, str):
        raw_map_path = raw_map_path.strip() or None
    elif raw_map_path is not None:
        raw_map_path = str(raw_map_path).strip() or None
    raw_map_dir = str(data_source.get("raw_map_dir", raw_job.get("raw_map_dir", "./dataset/raw_map"))).strip()
    raw_test_files = data_source.get("raw_test_files", raw_job.get("raw_test_files"))
    if isinstance(raw_test_files, str):
        raw_test_files = [raw_test_files]
    elif isinstance(raw_test_files, tuple):
        raw_test_files = list(raw_test_files)
    if raw_test_files is not None:
        raw_test_files = [str(v).strip() for v in raw_test_files if str(v).strip()]
        if not raw_test_files:
            raw_test_files = None
    map_padding_km_raw = data_source.get("map_padding_km", raw_job.get("map_padding_km", 5.0))
    try:
        map_padding_km = float(map_padding_km_raw)
    except Exception:
        map_padding_km = 5.0
    job["data_source"] = {
        "raw_dataset_dir": raw_dataset_dir or None,
        "raw_map_path": raw_map_path,
        "raw_map_dir": raw_map_dir or "./dataset/raw_map",
        "raw_test_files": raw_test_files,
        "map_padding_km": max(0.0, map_padding_km),
    }

    # Test flags
    traj_default = test_type == "exact"
    chunk_default = False
    time_default = True
    uncertainty_default = test_type == "uncertainty"
    run_baseline_default = test_type != "tuning"
    if test_type == "tuning":
        traj_default = False
        chunk_default = True
        uncertainty_default = False

    job["test_type"] = test_type
    job["traj_test"] = bool(test_items.get("traj_test", raw_job.get("traj_test", traj_default)))
    job["chunk_test"] = bool(test_items.get("chunk_test", raw_job.get("chunk_test", chunk_default)))
    job["time_test"] = bool(test_items.get("time_test", raw_job.get("time_test", time_default)))
    job["range_test"] = bool(
        test_items.get(
            "uncertainty_test",
            raw_job.get("range_test", uncertainty_default),
        )
    )
    job["run_baseline"] = bool(raw_job.get("run_baseline", run_baseline_default))

    # Tuning mode: quick-val chunk-only RectifiedTraj run.
    if test_type == "tuning":
        quick_val_path = str(
            raw_job.get(
                "quick_val_path",
                "./dataset/processed/NUMOSIM_Kanto/val/quick_val_chunk_50k.pt",
            )
        ).strip()
        job["quick_val_path"] = quick_val_path
        if not job.get("chunk_test_dir"):
            job["chunk_test_dir"] = quick_val_path
        job["run_baseline"] = False
        job["classic_baselines"] = []
        job["chunk_grid_search"] = True

    # Uncertainty path
    if uncertainty_path:
        job["test_data_path"] = uncertainty_path
    elif job.get("range_test") and traj_dirs:
        # Fallback: use first trajectory path as uncertainty input when explicit path is absent.
        job["test_data_path"] = traj_dirs[0]

    runtime_cfg = raw_job.get("runtime", {})
    if not isinstance(runtime_cfg, dict):
        runtime_cfg = {}
    job["runtime"] = dict(runtime_cfg)
    job["runtime"].setdefault("device", "cuda")
    job["runtime"].setdefault("strict_init", True)

    # Learned-model groups:
    # - primary group from rectifiedtraj block (backward-compatible)
    # - optional additional groups from model_groups list and/or residualreg block
    primary_group = _build_primary_model_group_from_job(job)
    model_groups: list[dict] = [primary_group]
    raw_model_groups = raw_job.get("model_groups")
    if raw_model_groups is not None and not isinstance(raw_model_groups, list):
        raise ValueError("model_groups must be a list of objects.")
    if isinstance(raw_model_groups, list):
        for idx, raw_group in enumerate(raw_model_groups):
            group = _normalize_model_group_schema_entry(
                raw_group,
                default_group=primary_group,
                context=f"eval_joblist.model_groups[{idx}]",
            )
            model_groups.append(group)

    residualreg_block = raw_job.get("residualreg")
    if isinstance(residualreg_block, dict):
        rr_group = _normalize_model_group_schema_entry(
            residualreg_block,
            default_group=primary_group,
            context="eval_joblist.residualreg",
        )
        model_groups.append(rr_group)

    model_groups = _dedupe_model_groups(model_groups)
    if not model_groups:
        raise ValueError("No valid learned-model group is configured.")
    job["model_groups"] = model_groups

    return job


def _stage(message: str) -> None:
    print(f"[run_benchmarks] {message}", flush=True)


def _runtime_cfg(job: dict) -> dict:
    cfg = job.get("runtime", {})
    return cfg if isinstance(cfg, dict) else {}


def _normalize_runtime_device_token(raw) -> str:
    token = str(raw or "cuda").strip().lower()
    if token.startswith("cuda"):
        return "cuda"
    if token == "cpu":
        return "cpu"
    raise ValueError(
        f"Unsupported runtime.device: {raw}. Use 'cuda' or 'cpu'."
    )


def _normalize_device_label(raw) -> str:
    token = str(raw or "").strip().lower()
    if token.startswith("cuda"):
        return "cuda"
    if token == "cpu":
        return "cpu"
    return token or "unknown"


def _runtime_device_requested(job: dict) -> str:
    cfg = _runtime_cfg(job)
    return _normalize_runtime_device_token(cfg.get("device", "cuda"))


def _runtime_device_for_defaults(job: dict) -> str:
    cfg = _runtime_cfg(job)
    eff = cfg.get("device_effective")
    if eff is not None:
        return _normalize_runtime_device_token(eff)
    return _runtime_device_requested(job)


def _runtime_device_effective(job: dict) -> str:
    cfg = _runtime_cfg(job)
    requested = _runtime_device_requested(job)
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


def _configure_encoder_decoder_device(job: dict) -> str:
    from encoder_decoder import get_runtime_device, set_runtime_device

    effective = _runtime_device_effective(job)
    runtime_cfg = _runtime_cfg(job)
    runtime_cfg["device_effective"] = effective
    job["runtime"] = runtime_cfg
    os.environ["RECTIFIEDTRAJ_RUNTIME_DEVICE_EFFECTIVE"] = effective
    os.environ["RECTIFIEDTRAJ_DEVICE"] = effective
    set_runtime_device(effective)
    # Keep DiffTraj on the same runtime device unless explicitly overridden.
    if not str(job.get("difftraj_device", "") or "").strip():
        job["difftraj_device"] = effective
    return get_runtime_device()


def _resolve_cpu_traj_caps(job: dict, default_m: int = 50, default_n: int = 1500) -> tuple[int, int]:
    cfg = _runtime_cfg(job)

    def _as_pos_int(raw, fallback: int) -> int:
        try:
            out = int(raw)
        except Exception:
            out = int(fallback)
        return max(1, out)

    m = _as_pos_int(cfg.get("cpu_traj_M", default_m), default_m)
    n = _as_pos_int(cfg.get("cpu_traj_N", default_n), default_n)
    return m, n


def _cpu_shrink_enabled(job: dict) -> bool:
    if _runtime_device_for_defaults(job) != "cpu":
        return False
    cfg = _runtime_cfg(job)
    return bool(cfg.get("cpu_shrink", True))


def _apply_cpu_dataset_caps(job: dict, datasets: list[dict]) -> list[dict]:
    if not _cpu_shrink_enabled(job):
        return datasets
    cap_m, cap_n = _resolve_cpu_traj_caps(job)
    capped: list[dict] = []
    for entry in datasets:
        m_old = int(entry.get("M", cap_m))
        n_old = int(entry.get("N", cap_n))
        m_new = min(m_old, cap_m)
        n_new = min(n_old, cap_n)
        if m_new != m_old or n_new != n_old:
            _stage(
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


def _resolve_chunk_max_chunks(job: dict, default_gpu: int = 5000, default_cpu: int = 2000) -> int | None:
    cfg = _runtime_cfg(job)
    if _runtime_device_for_defaults(job) == "cpu":
        default = int(cfg.get("cpu_chunk_max_chunks", default_cpu))
    else:
        default = int(default_gpu)
    raw = job.get("chunk_max_chunks", default)
    if raw is None:
        return None
    try:
        n = int(raw)
    except Exception:
        n = int(default)
    if n <= 0:
        return None
    return int(n)


def _require_job_field(job: dict, key: str, context: str) -> object:
    value = job.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        logging.error("Missing required job field '%s' for %s. Please set it in eval_joblist.json.", key, context)
        raise SystemExit(2)
    return value


def _latest_pt_file(directory: Path) -> Path:
    candidates = sorted(directory.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No .pt files found under {directory}")
    return candidates[0]


def _load_metadata(pt_path: Path) -> dict:
    data = torch.load(pt_path, map_location="cpu")
    meta = data.get("metadata", {})
    if not meta:
        raise ValueError(f"Missing metadata in {pt_path}")
    return meta


def _resolve_dataset(path_or_dir: Path, debug_path: Path | None, use_debug: bool) -> tuple[Path, int, int]:
    if use_debug:
        if debug_path is None:
            raise ValueError("Debug path not provided")
        pt_path = debug_path
    elif path_or_dir.is_file():
        pt_path = path_or_dir
    else:
        pt_path = _latest_pt_file(path_or_dir)

    meta = _load_metadata(pt_path)
    m = int(meta.get("n_trajectories", 0) or 0)
    n = int(meta.get("median_length", 0) or 0)
    if m <= 0 or n <= 0:
        raise ValueError(f"Invalid metadata in {pt_path}: n_trajectories={m}, median_length={n}")
    return pt_path, m, n


def _build_job_list_from_group(group: dict) -> dict:
    """Build trajectory grid config from one learned-model group."""
    job_list = {
        "Q1": group.get("Q1"),
        "Q2": group.get("Q2"),
        "t_delta": group.get("t_delta"),
        "methods": list(group.get("methods", ["BF", "DF"])),
    }
    if not job_list["Q1"] or not job_list["Q2"] or not job_list["t_delta"]:
        raise ValueError("Each model group must include non-empty Q1, Q2, and t_delta lists.")
    return job_list


def _build_job_list(job: dict) -> dict:
    job_list = {
        "Q1": job.get("Q1"),
        "Q2": job.get("Q2"),
        "t_delta": job.get("t_delta"),
        "methods": list(job.get("methods", ["BF", "DF"])),
    }
    if not job_list["Q1"] or not job_list["Q2"] or not job_list["t_delta"]:
        raise ValueError("eval_joblist.json must include non-empty Q1, Q2, and t_delta lists.")
    return job_list


def _resolve_classic_baselines(job: dict) -> list[str]:
    raw = job.get("classic_baselines")
    if raw is None:
        return list(DEFAULT_CLASSIC_BASELINES)
    if not isinstance(raw, (str, list, tuple)):
        raise ValueError("classic_baselines must be a list of names or comma-separated string.")
    requested = _as_list(raw)

    allowed = set(ALLOWED_CLASSIC_BASELINES)
    selected: list[str] = []
    for item in requested:
        base, mode, display = _split_baseline_spec(str(item))
        if not base:
            continue
        if base not in allowed:
            logging.warning("Unknown classic baseline ignored: %s", item)
            continue
        if base == "kalman_rts":
            selected.append(f"kalman_rts@{mode or 'textbook_default'}")
        else:
            selected.append(display or base)
    return _dedupe_keep_order(selected)


def _apply_kalman_calibration_overrides(job: dict) -> tuple[str, str]:
    """
    Apply one active Kalman calibration mode for the whole benchmark process.

    Supported modes:
    - dataset
    - numosim_kanto
    - textbook_default
    """
    mode = str(job.get("kalman_calibration_mode", "") or "").strip()
    dataset = str(job.get("kalman_calibration_dataset", "") or "").strip()
    if mode:
        os.environ["KALMAN_RTS_CALIBRATION_MODE"] = mode
    mode_eff = str(os.getenv("KALMAN_RTS_CALIBRATION_MODE", "dataset")).strip()

    if dataset:
        os.environ["KALMAN_RTS_CALIBRATION_DATASET"] = dataset
    dataset_eff = str(
        os.getenv("KALMAN_RTS_CALIBRATION_DATASET", "NUMOSIM_Kanto")
    ).strip()
    return mode_eff, dataset_eff


def _append_trajectory_timing_summary_row(
    manager: TestManager,
    model_name: str,
    model_tag: str,
    dataset_name: str,
    denoise_method: str,
    avg_time_sec: float,
    avg_time_sec_per_point: float | None,
    latency_p50_ms: float | None,
    latency_p95_ms: float | None,
    latency_max_ms: float | None,
    throughput_points_per_sec: float | None,
    peak_rss_mb: float | None,
    peak_vram_mb: float | None,
    num_points: int,
    calibration_time_sec: float | None = None,
    calibration_peak_rss_mb: float | None = None,
    calibration_peak_vram_mb: float | None = None,
    *,
    k: int | None = None,
    q1: int | None = None,
    q2: int | None = None,
    t_delta: float | None = None,
    n_steps: int | None = None,
    device: str | None = None,
) -> None:
    def _fmt(value, fmt: str) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "NA"
        return format(value, fmt)

    device_label = _normalize_device_label(
        device
        if device is not None
        else os.getenv(
            "RECTIFIEDTRAJ_RUNTIME_DEVICE_EFFECTIVE",
            os.getenv("RECTIFIEDTRAJ_DEVICE", "unknown"),
        )
    )

    row = (
        f"{model_name},{model_tag},{device_label},{dataset_name},{denoise_method},"
        f"{_fmt(k, 'd')},{_fmt(q1, 'd')},{_fmt(q2, 'd')},"
        f"{_fmt(t_delta, '.4f')},{_fmt(n_steps, 'd')},"
        f"NA,NA,NA,NA,"
        f"{_fmt(float(avg_time_sec), '.6f')},{_fmt(avg_time_sec_per_point, '.8f')},"
        f"{_fmt(latency_p50_ms, '.4f')},{_fmt(latency_p95_ms, '.4f')},{_fmt(latency_max_ms, '.4f')},"
        f"{_fmt(throughput_points_per_sec, '.4f')},{_fmt(peak_rss_mb, '.4f')},{_fmt(peak_vram_mb, '.4f')},"
        f"{_fmt(calibration_time_sec, '.6f')},{_fmt(calibration_peak_rss_mb, '.4f')},{_fmt(calibration_peak_vram_mb, '.4f')},"
        f"{_fmt(1, 'd')},{_fmt(num_points, 'd')},"
        f"{datetime.now().isoformat()}\n"
    )
    with open(manager.trajectory_evaluator.csv_path, "a") as f:
        f.write(row)


def _append_chunk_timing_summary_row(
    manager: TestManager,
    model_name: str,
    model_tag: str,
    dataset_name: str,
    avg_time_sec: float,
    avg_time_sec_per_point: float | None,
    latency_p50_ms: float | None,
    latency_p95_ms: float | None,
    latency_max_ms: float | None,
    throughput_points_per_sec: float | None,
    peak_rss_mb: float | None,
    peak_vram_mb: float | None,
    calibration_time_sec: float | None = None,
    calibration_peak_rss_mb: float | None = None,
    calibration_peak_vram_mb: float | None = None,
    *,
    k: int | None = 256,
    q1: int | None = None,
    q2: int | None = None,
    t_delta: float | None = None,
    n_steps: int | None = None,
    device: str | None = None,
) -> None:
    timing_csv_path = Path(manager.output_dir) / "chunk_timing_summary.csv"
    header_cols = [
        "model_name",
        "model_tag",
        "device",
        "dataset_name",
        "denoise_method",
        "K",
        "Q1",
        "Q2",
        "t_delta",
        "N_steps",
        "avg_denoise_time_sec",
        "avg_denoise_time_sec_per_point",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_max_ms",
        "throughput_points_per_sec",
        "peak_rss_mb",
        "peak_vram_mb",
        "calibration_time_sec",
        "calibration_peak_rss_mb",
        "calibration_peak_vram_mb",
        "num_tested_chunks",
        "test_timestamp",
    ]
    if not timing_csv_path.exists():
        timing_csv_path.write_text(",".join(header_cols) + "\n", encoding="utf-8")

    def _fmt(value, fmt: str) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "NA"
        return format(value, fmt)

    device_label = _normalize_device_label(
        device
        if device is not None
        else os.getenv(
            "RECTIFIEDTRAJ_RUNTIME_DEVICE_EFFECTIVE",
            os.getenv("RECTIFIEDTRAJ_DEVICE", "unknown"),
        )
    )
    row = (
        f"{model_name},{model_tag},{device_label},{dataset_name},N/A,"
        f"{_fmt(k, 'd')},{_fmt(q1, 'd')},{_fmt(q2, 'd')},"
        f"{_fmt(t_delta, '.4f')},{_fmt(n_steps, 'd')},"
        f"{_fmt(float(avg_time_sec), '.6f')},{_fmt(avg_time_sec_per_point, '.8f')},"
        f"{_fmt(latency_p50_ms, '.4f')},{_fmt(latency_p95_ms, '.4f')},{_fmt(latency_max_ms, '.4f')},"
        f"{_fmt(throughput_points_per_sec, '.4f')},"
        f"{_fmt(peak_rss_mb, '.4f')},{_fmt(peak_vram_mb, '.4f')},"
        f"{_fmt(calibration_time_sec, '.6f')},"
        f"{_fmt(calibration_peak_rss_mb, '.4f')},{_fmt(calibration_peak_vram_mb, '.4f')},"
        f"{_fmt(1, 'd')},{datetime.now().isoformat()}\n"
    )
    with open(timing_csv_path, "a", encoding="utf-8") as f:
        f.write(row)


def _resolve_time_repeats(job: dict, default_gpu: int = 5, default_cpu: int = 3) -> int:
    runtime_cfg = _runtime_cfg(job)
    if _runtime_device_for_defaults(job) == "cpu":
        default = int(runtime_cfg.get("cpu_time_repeats", default_cpu))
    else:
        default = int(default_gpu)
    raw = None
    raw = runtime_cfg.get("time_repeats")
    if raw is None:
        raw = job.get("time_repeats", default)
    try:
        n = int(raw)
    except Exception:
        n = int(default)
    return max(1, n)


def _measure_predict_repeats(
    predict_once,
    repeats: int,
) -> dict:
    device = str(
        os.getenv(
            "RECTIFIEDTRAJ_RUNTIME_DEVICE_EFFECTIVE",
            os.getenv("RECTIFIEDTRAJ_DEVICE", "unknown"),
        )
    ).strip().lower()
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    proc = psutil.Process(os.getpid())
    times: list[float] = []
    peak_rss_mb: float | None = None
    peak_vram_mb: float | None = None
    for _ in range(int(repeats)):
        rss_before = float(proc.memory_info().rss) / (1024.0 * 1024.0)
        if use_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        _ = predict_once()
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(float(t1 - t0))
        rss_after = float(proc.memory_info().rss) / (1024.0 * 1024.0)
        run_peak_rss = max(rss_before, rss_after)
        peak_rss_mb = run_peak_rss if peak_rss_mb is None else max(peak_rss_mb, run_peak_rss)
        if use_cuda:
            run_peak_vram = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
            peak_vram_mb = run_peak_vram if peak_vram_mb is None else max(peak_vram_mb, run_peak_vram)

    if not times:
        return {
            "avg_time_sec": 0.0,
            "latency_p50_ms": None,
            "latency_p95_ms": None,
            "latency_max_ms": None,
            "peak_rss_mb": peak_rss_mb,
            "peak_vram_mb": peak_vram_mb,
        }
    arr = np.asarray(times, dtype=float)
    return {
        "avg_time_sec": float(np.mean(arr)),
        "latency_p50_ms": float(np.percentile(arr, 50) * 1000.0),
        "latency_p95_ms": float(np.percentile(arr, 95) * 1000.0),
        "latency_max_ms": float(np.max(arr) * 1000.0),
        "peak_rss_mb": peak_rss_mb,
        "peak_vram_mb": peak_vram_mb,
    }


def _load_chunk_time_sample(job: dict) -> tuple[np.ndarray, np.ndarray | None, str, str] | None:
    chunk_paths = [str(p).strip() for p in _as_list(job.get("chunk_dirs")) if str(p).strip()]
    if not chunk_paths:
        fallback = str(job.get("chunk_test_dir", "") or "").strip()
        if fallback:
            chunk_paths = [fallback]
    if not chunk_paths:
        return None

    best_xy: np.ndarray | None = None
    best_ts: np.ndarray | None = None
    best_name: str | None = None
    best_coord_space: str | None = None

    def _as_numpy(x) -> np.ndarray:
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _normalize_coord_space_token(token: object) -> str:
        text = str(token or "").strip().upper()
        return text if text else "UNKNOWN"

    def _infer_coord_space_from_xy(xy: np.ndarray) -> str:
        arr = np.asarray(xy, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return "UNKNOWN"
        lon = arr[:, 0].reshape(-1)
        lat = arr[:, 1].reshape(-1)
        if lon.size <= 0 or lat.size <= 0:
            return "UNKNOWN"
        mask = np.isfinite(lon) & np.isfinite(lat)
        if not np.any(mask):
            return "UNKNOWN"
        lon = lon[mask]
        lat = lat[mask]
        if np.all((lon >= -180.0) & (lon <= 180.0) & (lat >= -90.0) & (lat <= 90.0)):
            return "GPS"
        if np.any(np.abs(lon) > 180.0) or np.any(np.abs(lat) > 90.0):
            return "ENU"
        return "UNKNOWN"

    def _record_to_chunk_xy_ts(rec: dict) -> tuple[np.ndarray, np.ndarray | None, str] | None:
        payload = rec.get("payload", {})
        rtype = rec.get("record_type")
        if rtype == "chunk_pair":
            x1 = _as_numpy(payload.get("X1"))
            if x1.ndim != 2 or x1.shape[1] < 2:
                return None
            xy = np.asarray(x1[:, :2], dtype=np.float32)
            ts = np.asarray(x1[:, 2], dtype=np.float64) if int(x1.shape[1]) >= 3 else None
            coord_space = _normalize_coord_space_token(payload.get("coord_space"))
            if coord_space == "UNKNOWN":
                coord_space = _infer_coord_space_from_xy(xy)
            return xy, ts, coord_space
        if rtype == "train_triplet":
            xt = _as_numpy(payload.get("X_t"))
            v = _as_numpy(payload.get("V"))
            t_arr = _as_numpy(payload.get("t")).reshape(-1)
            if xt.ndim != 2 or v.ndim != 2 or xt.shape[1] < 2 or v.shape[1] < 2:
                return None
            if t_arr.size <= 0:
                return None
            t_scalar = float(t_arr[0])
            x1 = xt[:, :2] + v[:, :2] * (1.0 - t_scalar)
            return np.asarray(x1, dtype=np.float32), None, "ENU"
        return None

    for test_dir in chunk_paths:
        test_path = Path(test_dir)
        path_for_hint = test_path.parent if test_path.is_file() else test_path
        dataset_hint = _infer_dataset_name_from_path(path_for_hint) or path_for_hint.name
        if dataset_hint.lower() in {"chunk_test", "test", "validation"} and path_for_hint.parent is not None:
            parent_name = _infer_dataset_name_from_path(path_for_hint.parent) or path_for_hint.parent.name
            if parent_name:
                dataset_hint = parent_name
        if test_path.is_file():
            loader_data_dir = str(test_path.parent)
            loader_pattern = test_path.name
        else:
            loader_data_dir = str(test_path)
            loader_pattern = "*.pt"
        try:
            loader = StandaloneDataLoader(
                mode="test",
                data_dir=loader_data_dir,
                file_pattern=loader_pattern,
                shuffle=False,
            )
        except Exception:
            continue
        for rec in loader.iter_test_records():
            converted = _record_to_chunk_xy_ts(rec)
            if converted is None:
                continue
            cand_xy, cand_ts, cand_space = converted
            if int(cand_xy.shape[0]) <= 0:
                continue
            if best_xy is None or int(cand_xy.shape[0]) > int(best_xy.shape[0]):
                best_xy = cand_xy
                best_ts = cand_ts
                best_name = dataset_hint
                best_coord_space = cand_space

    if best_xy is None or best_name is None:
        return None
    return np.asarray(best_xy, dtype=np.float32), best_ts, best_name, str(best_coord_space or "UNKNOWN").strip().upper()


def _infer_dataset_name_from_path(path_value: str | Path | None) -> str | None:
    if path_value is None:
        return None
    path = Path(path_value)
    parts = list(path.parts)
    for idx, part in enumerate(parts):
        if str(part).lower() == "processed" and idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def _validate_pt_source(path_value: str, *, label: str) -> None:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"{label} path does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() != ".pt":
            raise RuntimeError(f"{label} file must be a .pt file: {path}")
        return
    if not path.is_dir():
        raise RuntimeError(f"{label} path is neither file nor directory: {path}")
    if not any(path.glob("*.pt")):
        raise RuntimeError(f"{label} directory contains no .pt files: {path}")


def _resolve_raw_dataset_dir(job: dict) -> str | None:
    data_source = job.get("data_source", {})
    if isinstance(data_source, dict):
        raw_ds = str(data_source.get("raw_dataset_dir", "") or "").strip()
        if raw_ds:
            return raw_ds
    raw_ds = str(job.get("raw_dataset_dir", "") or "").strip()
    return raw_ds or None


def _default_processed_paths_for_dataset(dataset_name: str) -> tuple[str, str]:
    root = Path("./dataset/processed") / str(dataset_name) / "test"
    return str(root / "traj_test"), str(root / "chunk_test")


def _infer_dataset_hints_for_inputs(
    job: dict,
    traj_dirs: list[str],
    chunk_dirs: list[str],
) -> list[str]:
    hints: list[str] = []
    for p in traj_dirs + chunk_dirs:
        ds = _infer_dataset_name_from_path(p)
        if ds:
            hints.append(ds)
    raw_ds = _resolve_raw_dataset_dir(job)
    if raw_ds:
        hints.append(Path(raw_ds).name)
    return _dedupe_keep_order(hints)


def _safe_folder_token(value: str | None, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-"}:
            out.append(ch)
        else:
            out.append("_")
    token = "".join(out).strip("_")
    return token or fallback


def _resolve_run_scope_label(job: dict) -> str:
    traj_enabled = bool(job.get("traj_test", True))
    chunk_enabled = bool(job.get("chunk_test", False))
    if traj_enabled and chunk_enabled:
        return "full"
    if chunk_enabled:
        return "chunk"
    if traj_enabled:
        return "traj"
    # Fallback for special runs where both are disabled.
    return "full"


def _load_dataset_avg_sample_time_sec(dataset_name: str | None) -> float | None:
    ds = str(dataset_name or "").strip()
    if not ds:
        return None
    state_path = Path("./dataset/state") / f"state_{ds}.json"
    if not state_path.exists():
        repo_root = Path(__file__).resolve().parent.parent
        alt = repo_root / "dataset" / "state" / f"state_{ds}.json"
        if alt.exists():
            state_path = alt
    if not state_path.exists():
        return None
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    candidate_paths = [
        ("parquet_processor", "trajectory_extraction", "dataset_sample_time_sec", "mean_sec"),
        ("trajectory_extraction", "dataset_sample_time_sec", "mean_sec"),
    ]
    for keys in candidate_paths:
        try:
            obj = payload
            for k in keys:
                if not isinstance(obj, dict):
                    obj = None
                    break
                obj = obj.get(k)
            if obj is None:
                continue
            sec = float(obj)
            if np.isfinite(sec) and sec > 0:
                return sec
        except Exception:
            continue
    return None


def _format_sample_time_label(avg_sec: float | None) -> str:
    if avg_sec is None:
        return "unknown"
    sec_rounded = int(round(float(avg_sec)))
    if sec_rounded <= 0:
        return "unknown"
    # Requirement: 60sec should be rendered as 1min.
    if sec_rounded >= 60:
        min_rounded = int(round(sec_rounded / 60.0))
        min_rounded = max(1, min_rounded)
        return f"{min_rounded}min"
    return f"{sec_rounded}sec"


def _build_result_folder_name(
    job: dict,
    traj_dirs: list[str],
    chunk_dirs: list[str],
    *,
    runtime_device: str | None,
    timestamp: str,
) -> str:
    del traj_dirs, chunk_dirs
    scope = _resolve_run_scope_label(job)
    device_norm = _normalize_device_label(runtime_device)
    if device_norm == "cpu":
        device_label = "CPU"
    elif device_norm == "cuda":
        device_label = "CUDA"
    else:
        device_label = "UNKNOWN"
    return (
        f"{_safe_folder_token(scope, 'full')}_"
        f"{_safe_folder_token(device_label, 'unknown_device')}_"
        f"{_safe_folder_token(timestamp, 'ts')}"
    )


def _read_cpu_model_name() -> str:
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        try:
            for line in cpuinfo_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.lower().startswith("model name"):
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        model = parts[1].strip()
                        if model:
                            return model
        except Exception:
            pass
    return platform.processor() or "unknown"


def _collect_system_info(runtime_device: str) -> dict:
    vm = psutil.virtual_memory()
    info = {
        "test_timestamp": datetime.now().isoformat(),
        "runtime_device_effective": _normalize_device_label(runtime_device),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "os_system": platform.system(),
        "os_release": platform.release(),
        "python_version": sys.version.split()[0],
        "torch_version": getattr(torch, "__version__", "unknown"),
        "torch_cuda_version": str(getattr(torch.version, "cuda", None)),
        "cuda_available": bool(torch.cuda.is_available()),
        "cpu_model": _read_cpu_model_name(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gb": round(float(vm.total) / (1024.0 ** 3), 2),
    }

    gpu_names: list[str] = []
    if torch.cuda.is_available():
        try:
            for idx in range(torch.cuda.device_count()):
                gpu_names.append(str(torch.cuda.get_device_name(idx)))
        except Exception:
            pass
    info["gpu_count"] = len(gpu_names)
    info["gpu_names"] = gpu_names
    return info


def _write_system_info(output_dir: Path, runtime_device: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "system_info.json"
    payload = _collect_system_info(runtime_device)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _ensure_expected_test_paths_from_data_source(
    job: dict,
    traj_dirs: list[str],
    chunk_dirs: list[str],
) -> tuple[list[str], list[str]]:
    out_traj = [str(p).strip() for p in traj_dirs if str(p).strip()]
    out_chunk = [str(p).strip() for p in chunk_dirs if str(p).strip()]
    raw_ds = _resolve_raw_dataset_dir(job)
    if not raw_ds:
        return out_traj, out_chunk

    dataset_name = Path(raw_ds).name
    default_traj, default_chunk = _default_processed_paths_for_dataset(dataset_name)
    if bool(job.get("traj_test", True)) and not out_traj:
        out_traj = [default_traj]
        job["traj_dirs"] = list(out_traj)
        job["traj_paths"] = {"full_traj": default_traj}
    if bool(job.get("chunk_test", False)) and not out_chunk:
        out_chunk = [default_chunk]
        job["chunk_dirs"] = list(out_chunk)
        job["chunk_test_dir"] = default_chunk
    if bool(job.get("range_test", False)) and not str(job.get("test_data_path", "") or "").strip():
        if out_traj:
            job["test_data_path"] = out_traj[0]
    return out_traj, out_chunk


def _needs_valhalla_map(classic_baselines: list[str]) -> bool:
    return any(_split_baseline_spec(s)[0] == "valhalla_meili" for s in classic_baselines)


def _resolve_valhalla_pbf_for_dataset(dataset_name: str) -> Path | None:
    ds = str(dataset_name or "").strip()
    if not ds:
        return None
    expected = (Path.cwd() / "dataset" / "map_processed" / f"map_{ds}.osm.pbf").resolve()
    if expected.exists():
        return expected
    try:
        from baseline.artifacts import resolve_baseline_artifacts_from_state

        artifacts = resolve_baseline_artifacts_from_state(
            dataset_name_hint=ds,
            strict_dataset_hint=True,
        )
    except Exception:
        return None
    raw = getattr(artifacts, "map_file", None)
    if not raw:
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        return None
    if not str(path.name).lower().endswith(".pbf"):
        return None
    return path


def _collect_missing_inputs_for_autogen(
    job: dict,
    *,
    traj_dirs: list[str],
    chunk_dirs: list[str],
    classic_baselines: list[str],
) -> list[str]:
    missing: list[str] = []
    traj_enabled = bool(job.get("traj_test", True))
    chunk_enabled = bool(job.get("chunk_test", False))
    range_enabled = bool(job.get("range_test", False))
    run_baseline = bool(job.get("run_baseline", True))

    if traj_enabled:
        if not traj_dirs:
            missing.append("trajectory test path missing")
        else:
            for p in traj_dirs:
                try:
                    _validate_pt_source(p, label="trajectory test")
                except Exception:
                    missing.append(f"trajectory test data missing/invalid: {p}")
                    break

    if chunk_enabled:
        if not chunk_dirs:
            missing.append("chunk test path missing")
        else:
            for p in chunk_dirs:
                try:
                    _validate_pt_source(p, label="chunk test")
                except Exception:
                    missing.append(f"chunk test data missing/invalid: {p}")
                    break

    if range_enabled:
        test_data_path = str(job.get("test_data_path", "") or "").strip()
        if not test_data_path:
            missing.append("uncertainty test path missing")
        else:
            try:
                _validate_pt_source(test_data_path, label="uncertainty test")
            except Exception:
                missing.append(f"uncertainty test data missing/invalid: {test_data_path}")

    if run_baseline and _needs_valhalla_map(classic_baselines):
        dataset_hints = _infer_dataset_hints_for_inputs(job, traj_dirs, chunk_dirs)
        if not dataset_hints:
            missing.append("valhalla map check failed: dataset name unavailable")
        for ds in dataset_hints:
            resolved = _resolve_valhalla_pbf_for_dataset(ds)
            if resolved is None:
                expected = (Path.cwd() / "dataset" / "map_processed" / f"map_{ds}.osm.pbf").resolve()
                missing.append(
                    "valhalla_meili requires a .pbf map, but none resolved for "
                    f"dataset={ds}. expected={expected} (or state-resolved .pbf)."
                )

    return _dedupe_keep_order(missing)


def _apply_generated_paths_to_job(job: dict, dataset_name: str) -> tuple[list[str], list[str]]:
    traj_dir, chunk_dir = _default_processed_paths_for_dataset(dataset_name)
    traj_dirs = [traj_dir]
    chunk_dirs = [chunk_dir]
    job["traj_dirs"] = traj_dirs
    job["traj_paths"] = {"full_traj": traj_dir}
    job["chunk_dirs"] = chunk_dirs
    job["chunk_test_dir"] = chunk_dir
    if bool(job.get("range_test", False)):
        # Use traj_test root as uncertainty input; evaluator can read .pt files from this directory.
        job["test_data_path"] = str(job.get("test_data_path", "") or "").strip() or traj_dir
    return traj_dirs, chunk_dirs


def _run_data_generation_mode(job: dict) -> tuple[list[str], list[str]]:
    raw_ds_path = _resolve_raw_dataset_dir(job)
    if not raw_ds_path:
        raise RuntimeError(
            "Data generation mode requires raw dataset location. "
            "Set data_source.raw_dataset_dir in eval_joblist.json."
        )

    raw_ds_dir = Path(raw_ds_path)
    if not raw_ds_dir.exists() or not raw_ds_dir.is_dir():
        raise FileNotFoundError(f"raw_dataset_dir does not exist or is not a directory: {raw_ds_dir}")
    if not any(raw_ds_dir.glob("*.parquet")):
        raise RuntimeError(f"raw_dataset_dir contains no parquet files: {raw_ds_dir}")

    data_source = job.get("data_source", {})
    if not isinstance(data_source, dict):
        data_source = {}
    raw_test_files = data_source.get("raw_test_files")
    if isinstance(raw_test_files, tuple):
        raw_test_files = list(raw_test_files)
    if isinstance(raw_test_files, str):
        raw_test_files = [raw_test_files]
    if raw_test_files is not None:
        raw_test_files = [str(v).strip() for v in raw_test_files if str(v).strip()]
        if not raw_test_files:
            raw_test_files = None

    try:
        map_padding_km = float(data_source.get("map_padding_km", 5.0))
    except Exception:
        map_padding_km = 5.0
    map_padding_km = max(0.0, map_padding_km)
    raw_map_path = data_source.get("raw_map_path")
    if isinstance(raw_map_path, str):
        raw_map_path = raw_map_path.strip() or None
    elif raw_map_path is not None:
        raw_map_path = str(raw_map_path).strip() or None

    def _as_pos_int(value, default: int) -> int:
        try:
            out = int(value)
        except Exception:
            out = int(default)
        return max(1, out)

    k_value = _as_pos_int(job.get("K", job.get("chunk_K", 256)), 256)
    q_value = _as_pos_int(job.get("Q1", job.get("chunk_Q1", job.get("chunk_Q", 1))), 1)
    run_traj_extraction = bool(
        job.get("traj_test", True)
        or job.get("range_test", False)
        or job.get("run_baseline", True)
    )

    _stage(
        "Data generation mode: running parquet_processor_test_only "
        f"(raw_dataset_dir={raw_ds_dir}, K={k_value}, Q1={q_value}, run_traj_extraction={run_traj_extraction})"
    )
    from utils.data_processor.parquet_processor import parquet_processor_test_only

    parquet_processor_test_only(
        K=k_value,
        Q=q_value,
        raw_ds_path=str(raw_ds_dir),
        test_files=raw_test_files,
        run_traj_extraction=run_traj_extraction,
        map_padding_km=map_padding_km,
        raw_map_path=raw_map_path,
        run_map_slice=True,
    )
    return _apply_generated_paths_to_job(job, dataset_name=raw_ds_dir.name)


def _discover_models_for_preflight(model_root: Path, model_names: list | None) -> list[str]:
    if model_names is not None:
        out = [str(m).strip() for m in model_names if str(m).strip()]
        return out
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


def _find_model_checkpoint_for_preflight(model_dir: Path) -> Path | None:
    best_ckpt_dir = model_dir / "best_ckpt"
    if best_ckpt_dir.exists():
        best = sorted(best_ckpt_dir.glob("*_full.pt"))
        if best:
            return best[0]
    ckpts_dir = model_dir / "ckpts"
    if ckpts_dir.exists():
        all_ckpts = sorted(ckpts_dir.glob("*_full.pt"), key=lambda p: p.stat().st_mtime)
        if all_ckpts:
            return all_ckpts[-1]
    return None


def _preflight_validate_job(
    job: dict,
    *,
    model_groups: list[dict],
    traj_dirs: list[str],
    chunk_dirs: list[str],
    classic_baselines: list[str],
) -> None:
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

        resolved_models = _discover_models_for_preflight(root, model_names)
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
            ckpt = _find_model_checkpoint_for_preflight(model_dir)
            if ckpt is None:
                errors.append(f"{data_hypothesis}: no *_full.pt checkpoint found for model: {model_dir}")

    traj_enabled = bool(job.get("traj_test", True))
    chunk_enabled = bool(job.get("chunk_test", False))
    range_enabled = bool(job.get("range_test", False))

    if traj_enabled:
        if not traj_dirs and not bool(job.get("gen_new_test", False)):
            errors.append("traj_test=true but no trajectory test path is configured (test_files.traj_files).")
        for p in traj_dirs:
            try:
                _validate_pt_source(p, label="trajectory test")
            except Exception as exc:
                errors.append(str(exc))

    if chunk_enabled:
        if not chunk_dirs:
            errors.append("chunk_test=true but no chunk test path is configured (test_files.chunk_files).")
        for p in chunk_dirs:
            try:
                _validate_pt_source(p, label="chunk test")
            except Exception as exc:
                errors.append(str(exc))

    if range_enabled:
        test_data_path = str(job.get("test_data_path", "") or "").strip()
        if not test_data_path:
            errors.append("range_test=true but test_data_path/uncertainty_path is missing.")
        else:
            try:
                _validate_pt_source(test_data_path, label="uncertainty test")
            except Exception as exc:
                errors.append(str(exc))

    # Baseline artifact checks (fail-fast before long benchmark loops).
    if bool(job.get("run_baseline", True)):
        needs_valhalla = any(_split_baseline_spec(s)[0] == "valhalla_meili" for s in classic_baselines)
        kalman_specs = [s for s in classic_baselines if _split_baseline_spec(s)[0] == "kalman_rts"]

        dataset_hints: list[str] = []
        for p in traj_dirs + chunk_dirs:
            hint = _infer_dataset_name_from_path(p)
            if hint:
                dataset_hints.append(hint)
        dataset_hints = _dedupe_keep_order(dataset_hints)

        if needs_valhalla and not dataset_hints:
            errors.append(
                "valhalla_meili baseline selected but dataset name cannot be inferred from test paths "
                "(expected path like ./dataset/processed/<dataset>/test/...)."
            )

        if needs_valhalla:
            for ds in dataset_hints:
                resolved = _resolve_valhalla_pbf_for_dataset(ds)
                if resolved is None:
                    expected = (Path.cwd() / "dataset" / "map_processed" / f"map_{ds}.osm.pbf").resolve()
                    errors.append(
                        "valhalla_meili requires a .pbf map, but none resolved for "
                        f"dataset={ds}. expected={expected} (or state-resolved .pbf)."
                    )

        for spec in kalman_specs:
            _base, mode, _display = _split_baseline_spec(spec)
            mode_eff = _normalize_kalman_calibration_mode_token(mode)
            if mode_eff == "textbook_default":
                continue
            if mode_eff == "numosim_kanto":
                source_ds = (
                    str(os.getenv("KALMAN_RTS_CALIBRATION_DATASET", "")).strip()
                    or "NUMOSIM_Kanto"
                )
                cal = classic_baseline.resolve_kalman_calibration_file_from_state(
                    dataset_name_hint=source_ds
                )
                if not cal:
                    errors.append(
                        "kalman_rts@numosim_kanto requires calibration artifact for "
                        f"{source_ds}, but none was found under dataset/state."
                    )
                continue
            # dataset mode
            if not dataset_hints:
                errors.append(
                    "kalman_rts@dataset selected but dataset name cannot be inferred from test paths."
                )
                continue
            for ds in dataset_hints:
                cal = classic_baseline.resolve_kalman_calibration_file_from_state(
                    dataset_name_hint=ds
                )
                if not cal:
                    errors.append(
                        "kalman_rts@dataset requires calibration artifact for "
                        f"{ds}, but none was found under dataset/state."
                    )

    if errors:
        joined = "\n - " + "\n - ".join(errors)
        raise RuntimeError(f"Preflight validation failed:{joined}")


def _parse_model_name(model_name: str) -> tuple[str, str]:
    parts = model_name.split("_")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return model_name, "NA"


def _generate_full_traj(output_dir: Path, use_new_traj: dict) -> Path:
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


def _resolve_existing_dataset(
    name: str,
    path_value: str | None,
    debug_path: Path | None,
    use_debug: bool,
) -> tuple[Path, int, int] | None:
    if use_debug:
        if debug_path is None:
            raise ValueError(f"Debug path not provided for {name}")
        pt_path = debug_path
        meta = _load_metadata(pt_path)
        return pt_path, int(meta["n_trajectories"]), int(meta["median_length"])

    if path_value is None or str(path_value).strip() == "":
        return None

    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path for {name} not found: {path}")

    pt_path, m, n = _resolve_dataset(path, None, use_debug=False)
    return pt_path, m, n


def _run_trajectory_eval(
    manager: TestManager,
    model_root: str,
    model_names: list | None,
    methods: list,
    model_tag: str,
    dataset_entries: list[dict],
) -> None:
    for entry in dataset_entries:
        manager.run_trajectory_evaluation(
            model_names=model_names,
            denoise_methods=methods,
            model_root=model_root,
            model_tag=model_tag,
            test_data_path=str(entry["path"]),
            M=int(entry["M"]),
            D=None,
            N=int(entry["N"]),
            run_baselines=False,
        )


def _run_grid_eval(
    manager: TestManager,
    job_list: dict,
    model_root: str,
    model_names: list | None,
    model_tag: str,
    dataset_entries: list[dict],
) -> None:
    for entry in dataset_entries:
        _stage(
            f"Grid eval start | dataset={entry['name']} M={int(entry['M'])} N={int(entry['N'])}"
        )
        manager.run_grid_search_evaluation(
            job_list=job_list,
            model_names=model_names,
            model_root=model_root,
            model_tag=model_tag,
            test_data_path=str(entry["path"]),
            M=int(entry["M"]),
            D=None,
            N=int(entry["N"]),
            run_baselines=False,
        )


def _run_bounded_eval(
    manager: TestManager,
    job: dict,
    model_root: str,
    model_names: list | None,
    methods: list,
    classic_baselines: list[str],
    model_tag: str,
    run_baselines: bool,
) -> None:
    test_data_path = str(_require_job_field(job, "test_data_path", "uncertainty band evaluation"))
    if _cpu_shrink_enabled(job):
        cpu_m, cpu_n = _resolve_cpu_traj_caps(job)
        default_m = cpu_m
        default_n = cpu_n
    else:
        default_m = 200
        default_n = 10000
    manager.run_uncertainty_band_test(
        model_names=model_names,
        denoise_methods=methods,
        model_root=model_root,
        test_data_path=test_data_path,
        M=int(job.get("M", default_m)),
        N=int(job.get("N", default_n)),
        model_tag=model_tag,
        run_baselines=run_baselines,
        baseline_methods=classic_baselines,
    )


def _run_difftraj_baseline_trajectory(
    manager: TestManager,
    test_trajectories: list,
    dataset_name: str,
    job: dict,
) -> None:
    baseline_k = 256
    baseline_q1 = 1
    baseline_q2 = 12
    try:
        from baseline.difftraj import DiffTrajPaths, difftraj_denoise_with_model, prepare_difftraj
    except Exception as exc:
        logging.warning("DiffTraj import failed; skipping trajectory baseline: %s", exc)
        return

    repo_dir = job.get("difftraj_repo", "./bin/baseline_model/difftraj")
    checkpoint_path = job.get("difftraj_ckpt", "")
    device = job.get("difftraj_device", "cuda")
    timesteps = int(job.get("difftraj_timesteps", 100))
    final_steps = job.get("difftraj_final_steps")
    final_steps = int(final_steps) if final_steps is not None else None
    eta = float(job.get("difftraj_eta", 0.0))

    paths = DiffTrajPaths(
        repo_dir=repo_dir or None,
        checkpoint_path=checkpoint_path or None,
    )

    try:
        config, model, device = prepare_difftraj(paths, device=device)
    except Exception as exc:
        logging.warning("DiffTraj unavailable; skipping trajectory baseline: %s", exc)
        return

    target_len = int(getattr(getattr(config, "data", None), "traj_length", 0) or 0)
    logging.warning(
        "DiffTraj trajectory baseline started (traj=%d, timesteps=%d, device=%s, target_len=%d).",
        len(test_trajectories),
        timesteps,
        device,
        target_len,
    )

    def _pad_tail(values: np.ndarray, tgt_len: int) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.shape[0] == tgt_len:
            return values
        if values.shape[0] > tgt_len:
            return values[:tgt_len]
        pad_count = tgt_len - values.shape[0]
        tail = np.repeat(values[-1:], pad_count, axis=0)
        return np.concatenate([values, tail], axis=0)

    def _denoise_chunked(enu_noisy: np.ndarray, tgt_len: int) -> np.ndarray:
        total_len = enu_noisy.shape[0]
        if total_len <= tgt_len:
            chunk = _pad_tail(enu_noisy, tgt_len)
            den = difftraj_denoise_with_model(
                chunk,
                config=config,
                model=model,
                device=device,
                timesteps=timesteps,
                final_steps=final_steps,
                eta=eta,
            )
            return den[:total_len]

        outputs = []
        idx = 0
        while idx < total_len:
            chunk = _pad_tail(enu_noisy[idx: idx + tgt_len], tgt_len)
            den = difftraj_denoise_with_model(
                chunk,
                config=config,
                model=model,
                device=device,
                timesteps=timesteps,
                final_steps=final_steps,
                eta=eta,
            )
            outputs.append(den)
            idx += tgt_len

        stitched = np.concatenate(outputs, axis=0)
        return stitched[:total_len]

    per_traj_errors = []
    total_traj = len(test_trajectories)
    for traj_idx, traj_obj in enumerate(test_trajectories, start=1):
        noisy_gps = traj_obj.noisy_gps
        clean_gps = traj_obj.clean_gps
        if len(noisy_gps) == 0:
            continue

        ref_lat = float(clean_gps[0, 1])
        ref_lon = float(clean_gps[0, 0])
        enu_noisy = manager.trajectory_evaluator._gps_to_enu_batch(noisy_gps, ref_lat, ref_lon)
        enu_clean = manager.trajectory_evaluator._gps_to_enu_batch(clean_gps, ref_lat, ref_lon)

        try:
            if target_len > 0 and enu_noisy.shape[0] != target_len:
                denoised_enu = _denoise_chunked(enu_noisy, target_len)
            else:
                denoised_enu = difftraj_denoise_with_model(
                    enu_noisy,
                    config=config,
                    model=model,
                    device=device,
                    timesteps=timesteps,
                    final_steps=final_steps,
                    eta=eta,
                )
        except Exception as exc:
            logging.warning("DiffTraj failed on one trajectory (%s): %s", type(exc).__name__, exc)
            continue

        T = min(len(denoised_enu), len(enu_clean))
        if T <= 0:
            continue
        errors = np.linalg.norm(denoised_enu[:T] - enu_clean[:T], axis=1)
        per_traj_errors.append(errors)
        if traj_idx == 1 or traj_idx % 10 == 0 or traj_idx == total_traj:
            logging.warning("DiffTraj trajectory baseline progress: %d/%d", traj_idx, total_traj)

    if not per_traj_errors:
        logging.warning("No valid DiffTraj outputs; skipping trajectory baseline logging.")
        return

    all_errors = np.concatenate(per_traj_errors, axis=0)
    pw_metrics = manager.trajectory_evaluator._compute_pointwise_metrics(all_errors)
    bw_metrics = manager.trajectory_evaluator._compute_bytewise_metrics(
        test_trajectories, all_errors
    )
    cw_metrics = manager.trajectory_evaluator._compute_chunkwise_metrics(
        test_trajectories, all_errors, baseline_k, baseline_q1, baseline_q2
    )

    longest_traj = max(test_trajectories, key=lambda t: len(t.noisy_gps))
    ref_lat = float(longest_traj.clean_gps[0, 1])
    ref_lon = float(longest_traj.clean_gps[0, 0])
    enu_noisy = manager.trajectory_evaluator._gps_to_enu_batch(longest_traj.noisy_gps, ref_lat, ref_lon)

    timing = _measure_predict_repeats(
        lambda: (
            _denoise_chunked(enu_noisy, target_len)
            if (target_len > 0 and enu_noisy.shape[0] != target_len)
            else difftraj_denoise_with_model(
                enu_noisy,
                config=config,
                model=model,
                device=device,
                timesteps=timesteps,
                final_steps=final_steps,
                eta=eta,
            )
        ),
        repeats=5,
    )
    avg_time = float(timing["avg_time_sec"])
    avg_time_per_point = avg_time / len(longest_traj.noisy_gps) if avg_time and len(longest_traj.noisy_gps) else None
    throughput = (
        (float(len(longest_traj.noisy_gps)) / avg_time)
        if avg_time > 0 and len(longest_traj.noisy_gps) > 0
        else None
    )

    result = {
        "model_name": "difftraj",
        "model_tag": "Baseline",
        "device": _normalize_device_label(device),
        "dataset_name": dataset_name,
        "model_dir": None,
        "checkpoint_name": None,
        "K": None,
        "Q1": None,
        "Q2": None,
        "t_delta": None,
        "N_steps": None,
        "denoise_method": "N/A",
        "test_timestamp": datetime.now().isoformat(),
        "num_tested_trajectories": len(per_traj_errors),
        "num_tested_points": int(sum(len(e) for e in per_traj_errors)),
        "longest_trajectory_length": int(max(len(e) for e in per_traj_errors)),
        "avg_l2_err_pw": pw_metrics["avg"],
        "med_l2_err_pw": pw_metrics["med"],
        "p95_l2_err_pw": pw_metrics["p95"],
        "std_l2_err_pw": pw_metrics["std"],
        "avg_l2_err_bw": bw_metrics["avg_list"],
        "avg_l2_err_bw_norm": bw_metrics["avg_list_norm"],
        "avg_l2_err_cw": cw_metrics["avg_list"],
        "avg_l2_err_cw_norm": cw_metrics["avg_list_norm"],
        "avg_denoise_time_sec": avg_time,
        "avg_denoise_time_sec_per_point": avg_time_per_point,
        "latency_p50_ms": timing["latency_p50_ms"],
        "latency_p95_ms": timing["latency_p95_ms"],
        "latency_max_ms": timing["latency_max_ms"],
        "throughput_points_per_sec": throughput,
        "peak_rss_mb": timing["peak_rss_mb"],
        "peak_vram_mb": timing["peak_vram_mb"],
        "calibration_time_sec": 0.0,
        "calibration_peak_rss_mb": None,
        "calibration_peak_vram_mb": None,
    }
    manager.trajectory_evaluator._save_results(result)
    logging.warning("DiffTraj trajectory baseline saved.")


def _run_classic_baselines_filtered_compat(
    manager: TestManager,
    test_trajectories: list,
    dataset_name: str,
    methods: list[str],
) -> None:
    baseline_k = 256
    baseline_q1 = 1
    baseline_q2 = 12
    from baseline import classic as classic_baseline
    from baseline import (
        build_lat_lon_timestamp_sequence_from_lonlat,
        create_baseline_model,
        latlon_to_lonlat,
    )

    method_table = {
        "kalman_rts": classic_baseline.kalman_rts_smoother,
        "hampel": classic_baseline.hampel_filter,
        "savgol": classic_baseline.savitzky_golay_filter,
        "spline": classic_baseline.smoothing_spline,
        "raw": classic_baseline.raw_baseline,
        # Placeholder function is unused in model-based execution path.
        "valhalla_meili": classic_baseline.raw_baseline,
    }
    selected: list[tuple[str, str, str | None]] = []
    for spec in methods:
        base_name, kalman_mode, display_name = _split_baseline_spec(spec)
        if base_name not in method_table:
            logging.warning("Classic baseline %s ignored (unknown base=%s)", spec, base_name)
            continue
        selected.append((display_name, base_name, kalman_mode))
    if not selected:
        _stage("Classic baseline list is empty; skipping classic baselines.")
        return

    longest_traj = max(test_trajectories, key=lambda t: len(t.noisy_gps))
    longest_points = len(longest_traj.noisy_gps)
    runtime_device = str(
        os.getenv(
            "RECTIFIEDTRAJ_RUNTIME_DEVICE_EFFECTIVE",
            os.getenv("RECTIFIEDTRAJ_DEVICE", "unknown"),
        )
    ).strip().lower()
    use_cuda_timing = runtime_device.startswith("cuda") and torch.cuda.is_available()
    proc = psutil.Process(os.getpid())

    for display_name, base_name, kalman_mode in selected:
        logging.info("Running classic baseline (compat): %s", display_name)
        model = None
        calibration_time_sec = None
        calibration_peak_rss_mb = None
        calibration_peak_vram_mb = None
        try:
            cal_rss_before = float(proc.memory_info().rss) / (1024.0 * 1024.0)
            if use_cuda_timing:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            cal_t0 = time.perf_counter()
            model = create_baseline_model(
                method_name=base_name,
                dataset_name=dataset_name,
                kalman_calibration_mode=(
                    kalman_mode if base_name == "kalman_rts" else None
                ),
            )
            if use_cuda_timing:
                torch.cuda.synchronize()
            cal_t1 = time.perf_counter()
            cal_rss_after = float(proc.memory_info().rss) / (1024.0 * 1024.0)
            calibration_time_sec = float(cal_t1 - cal_t0)
            calibration_peak_rss_mb = max(cal_rss_before, cal_rss_after)
            if use_cuda_timing:
                calibration_peak_vram_mb = (
                    float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
                )
        except Exception as exc:
            logging.warning("Classic baseline %s initialization failed: %s", display_name, exc)
            continue

        try:
            all_errors = []

            for traj_obj in test_trajectories:
                noisy_gps = traj_obj.noisy_gps
                clean_gps = traj_obj.clean_gps

                ref_lat = float(clean_gps[0, 1])
                ref_lon = float(clean_gps[0, 0])
                enu_noisy = manager.trajectory_evaluator._gps_to_enu_batch(noisy_gps, ref_lat, ref_lon)
                enu_clean = manager.trajectory_evaluator._gps_to_enu_batch(clean_gps, ref_lat, ref_lon)

                try:
                    ts = getattr(traj_obj, "timestamps", None)
                    seq = build_lat_lon_timestamp_sequence_from_lonlat(noisy_gps, timestamps=ts)
                    denoised_latlon = model.predict(seq)
                    denoised_gps = latlon_to_lonlat(denoised_latlon)
                    denoised_enu = manager.trajectory_evaluator._gps_to_enu_batch(
                        denoised_gps, ref_lat, ref_lon
                    )
                except Exception as exc:
                    logging.warning("Classic baseline %s skipped: %s", display_name, exc)
                    all_errors = []
                    break

                errors = np.linalg.norm(denoised_enu - enu_clean, axis=1)
                all_errors.append(errors)

            if not all_errors:
                continue

            errors = np.concatenate(all_errors, axis=0)
            pw_metrics = manager.trajectory_evaluator._compute_pointwise_metrics(errors)
            bw_metrics = manager.trajectory_evaluator._compute_bytewise_metrics(
                test_trajectories, errors
            )
            cw_metrics = manager.trajectory_evaluator._compute_chunkwise_metrics(
                test_trajectories, errors, baseline_k, baseline_q1, baseline_q2
            )

            try:
                ts = getattr(longest_traj, "timestamps", None)
                seq = build_lat_lon_timestamp_sequence_from_lonlat(
                    longest_traj.noisy_gps,
                    timestamps=ts,
                )
                timing = _measure_predict_repeats(
                    lambda: model.predict(seq),
                    repeats=5,
                )
                avg_time = float(timing["avg_time_sec"])
            except Exception:
                timing = {
                    "latency_p50_ms": None,
                    "latency_p95_ms": None,
                    "latency_max_ms": None,
                    "peak_rss_mb": None,
                    "peak_vram_mb": None,
                }
                avg_time = None
            avg_time_per_point = avg_time / longest_points if avg_time is not None and longest_points else None
            throughput = (
                (float(longest_points) / float(avg_time))
                if avg_time is not None and float(avg_time) > 0 and int(longest_points) > 0
                else None
            )

            result = {
                "model_name": display_name,
                "model_tag": "Baseline",
                "device": "cpu",
                "dataset_name": dataset_name,
                "model_dir": None,
                "checkpoint_name": None,
                "K": None,
                "Q1": None,
                "Q2": None,
                "t_delta": None,
                "N_steps": None,
                "denoise_method": "N/A",
                "test_timestamp": datetime.now().isoformat(),
                "num_tested_trajectories": len(test_trajectories),
                "num_tested_points": int(sum(len(t.noisy_gps) for t in test_trajectories)),
                "longest_trajectory_length": int(max(len(t.noisy_gps) for t in test_trajectories)),
                "avg_l2_err_pw": pw_metrics["avg"],
                "med_l2_err_pw": pw_metrics["med"],
                "p95_l2_err_pw": pw_metrics["p95"],
                "std_l2_err_pw": pw_metrics["std"],
                "avg_l2_err_bw": bw_metrics["avg_list"],
                "avg_l2_err_bw_norm": bw_metrics["avg_list_norm"],
                "avg_l2_err_cw": cw_metrics["avg_list"],
                "avg_l2_err_cw_norm": cw_metrics["avg_list_norm"],
                "avg_denoise_time_sec": avg_time,
                "avg_denoise_time_sec_per_point": avg_time_per_point,
                "latency_p50_ms": timing["latency_p50_ms"],
                "latency_p95_ms": timing["latency_p95_ms"],
                "latency_max_ms": timing["latency_max_ms"],
                "throughput_points_per_sec": throughput,
                "peak_rss_mb": timing["peak_rss_mb"],
                "peak_vram_mb": timing["peak_vram_mb"],
                "calibration_time_sec": calibration_time_sec,
                "calibration_peak_rss_mb": calibration_peak_rss_mb,
                "calibration_peak_vram_mb": calibration_peak_vram_mb,
            }
            if base_name == "valhalla_meili":
                diagnostics_fn = getattr(model, "diagnostics_snapshot", None)
                if callable(diagnostics_fn):
                    try:
                        diagnostics = diagnostics_fn()
                        if isinstance(diagnostics, dict):
                            result.update(diagnostics)
                    except Exception as exc:
                        logging.warning(
                            "Failed to collect Valhalla diagnostics for result output: %s",
                            exc,
                        )
            manager.trajectory_evaluator._save_results(result)
        finally:
            if model is not None:
                try:
                    model.deconst()
                except Exception:
                    pass


def _run_time_tests(
    manager: TestManager,
    job: dict,
    job_list: dict,
    model_root: str,
    model_names: list | None,
    dataset_entries: list[dict],
    methods: list,
    classic_baselines: list[str],
    model_tag: str = "RectifiedTraj",
    run_classic_baselines: bool = True,
    include_difftraj_timing: bool = False,
) -> None:
    from encoder_decoder import EncoderDecoder
    from baseline import (
        build_lat_lon_timestamp_sequence_from_lonlat,
        create_baseline_model,
    )
    from pymap3d import geodetic2enu

    repeats = _resolve_time_repeats(job)
    traj_timing_enabled = bool(job.get("traj_test", True))
    chunk_timing_enabled = bool(job.get("chunk_test", False))
    chunk_timing_csv = Path(manager.output_dir) / "chunk_timing_summary.csv"
    _stage(
        "Time test config | repeats=%d traj_csv=%s chunk_timing_csv=%s"
        % (
            repeats,
            manager.trajectory_evaluator.csv_path,
            chunk_timing_csv,
        )
    )

    resolved_model_names = (
        manager._discover_models(model_root) if model_names is None else list(model_names)
    )
    time_config = job.get("time_config")
    if time_config is None:
        time_config = {
            "Q1": job_list["Q1"][0],
            "Q2": job_list["Q2"][0],
            "t_delta": job_list["t_delta"][0],
        }

    q1_cfg = time_config.get("Q1")
    q2_cfg = time_config.get("Q2")
    td_cfg = time_config.get("t_delta")
    n_steps_cfg = int(round(1.0 / float(td_cfg))) if td_cfg not in (None, 0) else None

    # ------------------------------------------------------------
    # Trajectory timing
    # ------------------------------------------------------------
    if traj_timing_enabled:
        dataset_entry = next((d for d in dataset_entries if d.get("name") == "full_traj"), None)
        if dataset_entry is None:
            dataset_entry = dataset_entries[0] if dataset_entries else None
        if dataset_entry is None:
            raise RuntimeError(
                "Trajectory timing is enabled but no trajectory dataset is available."
            )

        test_trajectories, dataset_name = manager._load_or_generate_test_data(
            test_data_path=str(dataset_entry["path"]),
            M=int(dataset_entry["M"]),
            N=int(dataset_entry["N"]),
        )
        if not test_trajectories:
            raise RuntimeError("Trajectory timing is enabled but no trajectories were loaded.")

        longest_traj = max(test_trajectories, key=lambda t: len(t.noisy_gps))
        traj_num_points = int(len(longest_traj.noisy_gps))

        for model_name in resolved_model_names:
            model_dir = Path(model_root) / model_name
            ckpt_name = manager._find_best_checkpoint(model_dir)
            if ckpt_name is None:
                manager.logger.warning("No checkpoint found for %s, skipping trajectory time test", model_name)
                continue
            ckpt_path = manager.trajectory_evaluator._get_checkpoint_path(str(model_dir), ckpt_name)
            if ckpt_path is None:
                manager.logger.warning("Checkpoint not found for %s, skipping trajectory time test", model_name)
                continue

            for method in methods:
                method_norm = str(method).strip().upper()
                if method_norm not in {"BF", "DF"}:
                    logging.warning("Unknown trajectory denoise method ignored in timing: %s", method)
                    continue
                try:
                    decoder = EncoderDecoder(ckpt_path, manual_config=time_config)
                    timing = _measure_predict_repeats(
                        (lambda: decoder.denoise_traj_BF(longest_traj.noisy_gps))
                        if method_norm == "BF"
                        else (lambda: decoder.denoise_traj_DF(longest_traj.noisy_gps)),
                        repeats,
                    )
                except Exception as exc:
                    logging.warning(
                        "RectifiedTraj trajectory timing failed for %s/%s: %s",
                        model_name,
                        method_norm,
                        exc,
                    )
                    continue
                avg_time = float(timing["avg_time_sec"])
                avg_time_per_point = avg_time / traj_num_points if traj_num_points else None
                throughput = (
                    (float(traj_num_points) / avg_time)
                    if avg_time > 0 and traj_num_points > 0
                    else None
                )
                _append_trajectory_timing_summary_row(
                    manager=manager,
                    model_name=model_name,
                    model_tag=model_tag,
                    dataset_name=dataset_name,
                    denoise_method=method_norm,
                    avg_time_sec=float(avg_time),
                    avg_time_sec_per_point=avg_time_per_point,
                    latency_p50_ms=timing["latency_p50_ms"],
                    latency_p95_ms=timing["latency_p95_ms"],
                    latency_max_ms=timing["latency_max_ms"],
                    throughput_points_per_sec=throughput,
                    peak_rss_mb=timing["peak_rss_mb"],
                    peak_vram_mb=timing["peak_vram_mb"],
                    calibration_time_sec=0.0,
                    calibration_peak_rss_mb=None,
                    calibration_peak_vram_mb=None,
                    num_points=traj_num_points,
                    k=int(getattr(decoder, "K", 0)) if getattr(decoder, "K", None) is not None else None,
                    q1=int(getattr(decoder, "Q1_bytes", 0)) if getattr(decoder, "Q1_bytes", None) is not None else None,
                    q2=int(getattr(decoder, "Q2_bytes", 0)) if getattr(decoder, "Q2_bytes", None) is not None else None,
                    t_delta=float(getattr(decoder, "t_delta", 0.0)) if getattr(decoder, "t_delta", None) is not None else None,
                    n_steps=(
                        int(round(1.0 / float(getattr(decoder, "t_delta", 0.0))))
                        if getattr(decoder, "t_delta", None) not in (None, 0)
                        else None
                    ),
                )

        if run_classic_baselines:
            traj_seq = build_lat_lon_timestamp_sequence_from_lonlat(
                longest_traj.noisy_gps,
                timestamps=getattr(longest_traj, "timestamps", None),
            )
            baseline_method_allow = {
                "kalman_rts",
                "hampel",
                "savgol",
                "spline",
                "raw",
                "valhalla_meili",
            }
            baseline_specs: list[tuple[str, str, str | None]] = []
            for spec in classic_baselines:
                base_name, kalman_mode, display_name = _split_baseline_spec(spec)
                if base_name not in baseline_method_allow:
                    logging.warning("Classic timing baseline %s ignored (unknown base=%s)", spec, base_name)
                    continue
                baseline_specs.append((display_name, base_name, kalman_mode))

            for display_name, base_name, kalman_mode in baseline_specs:
                model = None
                calibration_time_sec = None
                calibration_peak_rss_mb = None
                calibration_peak_vram_mb = None
                try:
                    proc = psutil.Process(os.getpid())
                    use_cuda = str(
                        os.getenv(
                            "RECTIFIEDTRAJ_RUNTIME_DEVICE_EFFECTIVE",
                            os.getenv("RECTIFIEDTRAJ_DEVICE", "unknown"),
                        )
                    ).strip().lower().startswith("cuda") and torch.cuda.is_available()
                    cal_rss_before = float(proc.memory_info().rss) / (1024.0 * 1024.0)
                    if use_cuda:
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    cal_t0 = time.perf_counter()
                    model = create_baseline_model(
                        method_name=base_name,
                        dataset_name=dataset_name,
                        kalman_calibration_mode=(
                            kalman_mode if base_name == "kalman_rts" else None
                        ),
                    )
                    if use_cuda:
                        torch.cuda.synchronize()
                    cal_t1 = time.perf_counter()
                    cal_rss_after = float(proc.memory_info().rss) / (1024.0 * 1024.0)
                    calibration_time_sec = float(cal_t1 - cal_t0)
                    calibration_peak_rss_mb = max(cal_rss_before, cal_rss_after)
                    if use_cuda:
                        calibration_peak_vram_mb = (
                            float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
                        )
                    timing = _measure_predict_repeats(
                        lambda: model.predict(traj_seq),
                        repeats,
                    )
                except Exception as exc:
                    logging.warning("Classic trajectory timing baseline %s failed: %s", display_name, exc)
                    continue
                finally:
                    if model is not None:
                        try:
                            model.deconst()
                        except Exception:
                            pass

                avg_time = float(timing["avg_time_sec"])
                avg_time_per_point = avg_time / traj_num_points if traj_num_points else None
                throughput = (
                    (float(traj_num_points) / avg_time)
                    if avg_time > 0 and traj_num_points > 0
                    else None
                )
                _append_trajectory_timing_summary_row(
                    manager=manager,
                    model_name=display_name,
                    model_tag="Baseline",
                    dataset_name=dataset_name,
                    denoise_method="N/A",
                    avg_time_sec=float(avg_time),
                    avg_time_sec_per_point=avg_time_per_point,
                    latency_p50_ms=timing["latency_p50_ms"],
                    latency_p95_ms=timing["latency_p95_ms"],
                    latency_max_ms=timing["latency_max_ms"],
                    throughput_points_per_sec=throughput,
                    peak_rss_mb=timing["peak_rss_mb"],
                    peak_vram_mb=timing["peak_vram_mb"],
                    calibration_time_sec=calibration_time_sec,
                    calibration_peak_rss_mb=calibration_peak_rss_mb,
                    calibration_peak_vram_mb=calibration_peak_vram_mb,
                    num_points=traj_num_points,
                    k=None,
                    q1=None,
                    q2=None,
                    t_delta=None,
                    n_steps=None,
                    device="cpu",
                )

        if include_difftraj_timing and DIFFTRAJ_ENABLED:
            try:
                from baseline.difftraj import DiffTrajPaths, difftraj_denoise_with_model, prepare_difftraj

                repo_dir = job.get("difftraj_repo", "./bin/baseline_model/difftraj")
                checkpoint_path = job.get("difftraj_ckpt", "")
                device = job.get("difftraj_device", "cuda")
                timesteps = int(job.get("difftraj_timesteps", 100))
                final_steps = job.get("difftraj_final_steps")
                final_steps = int(final_steps) if final_steps is not None else None
                eta = float(job.get("difftraj_eta", 0.0))

                paths = DiffTrajPaths(
                    repo_dir=repo_dir or None,
                    checkpoint_path=checkpoint_path or None,
                )
                config, model, device = prepare_difftraj(paths, device=device)
                target_len = int(getattr(getattr(config, "data", None), "traj_length", 0) or 0)

                ref_lat = float(longest_traj.noisy_gps[0, 1])
                ref_lon = float(longest_traj.noisy_gps[0, 0])
                enu_noisy = manager.trajectory_evaluator._gps_to_enu_batch(
                    longest_traj.noisy_gps,
                    ref_lat,
                    ref_lon,
                ).astype(np.float32, copy=False)

                def _pad_tail(values: np.ndarray, tgt_len: int) -> np.ndarray:
                    values = np.asarray(values, dtype=np.float32)
                    if values.shape[0] == tgt_len:
                        return values
                    if values.shape[0] > tgt_len:
                        return values[:tgt_len]
                    tail = np.repeat(values[-1:], tgt_len - values.shape[0], axis=0)
                    return np.concatenate([values, tail], axis=0)

                def _denoise_once(enu_xy: np.ndarray) -> np.ndarray:
                    if target_len <= 0 or enu_xy.shape[0] == target_len:
                        return difftraj_denoise_with_model(
                            enu_xy,
                            config=config,
                            model=model,
                            device=device,
                            timesteps=timesteps,
                            final_steps=final_steps,
                            eta=eta,
                        )
                    if enu_xy.shape[0] < target_len:
                        return difftraj_denoise_with_model(
                            _pad_tail(enu_xy, target_len),
                            config=config,
                            model=model,
                            device=device,
                            timesteps=timesteps,
                            final_steps=final_steps,
                            eta=eta,
                        )[:enu_xy.shape[0]]
                    out = []
                    i = 0
                    while i < enu_xy.shape[0]:
                        chunk = _pad_tail(enu_xy[i: i + target_len], target_len)
                        out.append(
                            difftraj_denoise_with_model(
                                chunk,
                                config=config,
                                model=model,
                                device=device,
                                timesteps=timesteps,
                                final_steps=final_steps,
                                eta=eta,
                            )
                        )
                        i += target_len
                    return np.concatenate(out, axis=0)[:enu_xy.shape[0]]

                timing = _measure_predict_repeats(lambda: _denoise_once(enu_noisy), repeats)
                avg_time = float(timing["avg_time_sec"])
                avg_time_per_point = avg_time / traj_num_points if traj_num_points else None
                throughput = (
                    (float(traj_num_points) / avg_time)
                    if avg_time > 0 and traj_num_points > 0
                    else None
                )
                _append_trajectory_timing_summary_row(
                    manager=manager,
                    model_name="difftraj",
                    model_tag="Baseline",
                    dataset_name=dataset_name,
                    denoise_method="N/A",
                    avg_time_sec=float(avg_time),
                    avg_time_sec_per_point=avg_time_per_point,
                    latency_p50_ms=timing["latency_p50_ms"],
                    latency_p95_ms=timing["latency_p95_ms"],
                    latency_max_ms=timing["latency_max_ms"],
                    throughput_points_per_sec=throughput,
                    peak_rss_mb=timing["peak_rss_mb"],
                    peak_vram_mb=timing["peak_vram_mb"],
                    calibration_time_sec=0.0,
                    calibration_peak_rss_mb=None,
                    calibration_peak_vram_mb=None,
                    num_points=traj_num_points,
                    k=None,
                    q1=None,
                    q2=None,
                    t_delta=None,
                    n_steps=None,
                    device=_normalize_device_label(device),
                )
            except Exception as exc:
                logging.warning("DiffTraj timing baseline unavailable: %s", exc)

    # ------------------------------------------------------------
    # Chunk timing
    # ------------------------------------------------------------
    if chunk_timing_enabled:
        chunk_sample = _load_chunk_time_sample(job)
        if chunk_sample is None:
            raise RuntimeError(
                "Chunk timing is enabled but no usable chunk test data is available."
            )

        chunk_xy_raw, chunk_ts, chunk_dataset_name, chunk_coord_space = chunk_sample
        chunk_coord_space_norm = str(chunk_coord_space or "UNKNOWN").strip().upper()
        chunk_num_points = int(chunk_xy_raw.shape[0])
        chunk_ts_abs = None
        if chunk_ts is not None:
            chunk_ts_abs = np.asarray(chunk_ts, dtype=np.float64).reshape(-1)
        chunk_ts_rel = None
        if chunk_ts_abs is not None:
            chunk_ts_rel = chunk_ts_abs.copy()
            if chunk_ts_rel.size > 0 and np.isfinite(chunk_ts_rel[0]):
                chunk_ts_rel = chunk_ts_rel - float(chunk_ts_rel[0])

        chunk_xy_enu = np.asarray(chunk_xy_raw, dtype=np.float32)
        chunk_lonlat = None
        chunk_seq_latlon_t = None
        if chunk_coord_space_norm == "GPS":
            chunk_lonlat = np.asarray(chunk_xy_raw, dtype=np.float64)
            ref_lon = float(chunk_lonlat[0, 0])
            ref_lat = float(chunk_lonlat[0, 1])
            e, n, _ = geodetic2enu(
                chunk_lonlat[:, 1],
                chunk_lonlat[:, 0],
                0.0,
                ref_lat,
                ref_lon,
                0.0,
            )
            chunk_xy_enu = np.stack([e, n], axis=1).astype(np.float32, copy=False)
            chunk_seq_latlon_t = build_lat_lon_timestamp_sequence_from_lonlat(
                chunk_lonlat,
                timestamps=chunk_ts_abs,
            )
        elif chunk_coord_space_norm not in {"ENU", "UNKNOWN"}:
            logging.warning(
                "Unsupported chunk coord_space '%s' in chunk timing; defaulting to ENU path.",
                chunk_coord_space_norm,
            )

        for model_name in resolved_model_names:
            model_dir = Path(model_root) / model_name
            ckpt_name = manager._find_best_checkpoint(model_dir)
            if ckpt_name is None:
                manager.logger.warning("No checkpoint found for %s, skipping chunk time test", model_name)
                continue
            ckpt_path = manager.trajectory_evaluator._get_checkpoint_path(str(model_dir), ckpt_name)
            if ckpt_path is None:
                manager.logger.warning("Checkpoint not found for %s, skipping chunk time test", model_name)
                continue
            try:
                decoder = EncoderDecoder(ckpt_path, manual_config=time_config)
                if chunk_coord_space_norm == "GPS":
                    if chunk_lonlat is None:
                        raise RuntimeError("GPS chunk timing path expected chunk_lonlat.")
                    timing = _measure_predict_repeats(
                        lambda: decoder.denoise_chunk(chunk_lonlat),
                        repeats,
                    )
                else:
                    timing = _measure_predict_repeats(
                        lambda: decoder.denoise_chunk_enu(chunk_xy_enu),
                        repeats,
                    )
            except Exception as exc:
                logging.warning("RectifiedTraj chunk timing failed for %s: %s", model_name, exc)
                continue
            avg_time = float(timing["avg_time_sec"])
            avg_time_per_point = avg_time / chunk_num_points if chunk_num_points else None
            throughput = (
                (float(chunk_num_points) / avg_time)
                if avg_time > 0 and chunk_num_points > 0
                else None
            )
            _append_chunk_timing_summary_row(
                manager=manager,
                model_name=model_name,
                model_tag=model_tag,
                dataset_name=chunk_dataset_name,
                avg_time_sec=float(avg_time),
                avg_time_sec_per_point=avg_time_per_point,
                latency_p50_ms=timing["latency_p50_ms"],
                latency_p95_ms=timing["latency_p95_ms"],
                latency_max_ms=timing["latency_max_ms"],
                throughput_points_per_sec=throughput,
                peak_rss_mb=timing["peak_rss_mb"],
                peak_vram_mb=timing["peak_vram_mb"],
                calibration_time_sec=0.0,
                calibration_peak_rss_mb=None,
                calibration_peak_vram_mb=None,
                k=int(getattr(decoder, "K", 0)) if getattr(decoder, "K", None) is not None else None,
                q1=int(getattr(decoder, "Q1_bytes", 0)) if getattr(decoder, "Q1_bytes", None) is not None else None,
                q2=int(getattr(decoder, "Q2_bytes", 0)) if getattr(decoder, "Q2_bytes", None) is not None else None,
                t_delta=float(getattr(decoder, "t_delta", 0.0)) if getattr(decoder, "t_delta", None) is not None else None,
                n_steps=(
                    int(round(1.0 / float(getattr(decoder, "t_delta", 0.0))))
                    if getattr(decoder, "t_delta", None) not in (None, 0)
                    else None
                ),
            )

        if run_classic_baselines:
            chunk_method_allow = {
                "kalman_rts",
                "hampel",
                "savgol",
                "spline",
                "raw",
                "valhalla_meili",
            }
            for spec in classic_baselines:
                base_name, kalman_mode, display_name = _split_baseline_spec(spec)
                if base_name not in chunk_method_allow:
                    logging.warning("Chunk timing baseline %s ignored (unknown base=%s)", spec, base_name)
                    continue
                if base_name == "valhalla_meili" and chunk_seq_latlon_t is None:
                    logging.warning(
                        "Chunk timing baseline %s skipped: requires GPS chunk data.",
                        display_name or base_name,
                    )
                    continue

                model = None
                calibration_time_sec = None
                calibration_peak_rss_mb = None
                calibration_peak_vram_mb = None
                try:
                    mode = (
                        _normalize_kalman_calibration_mode_token(kalman_mode)
                        if base_name == "kalman_rts"
                        else None
                    )
                    proc = psutil.Process(os.getpid())
                    use_cuda = str(
                        os.getenv(
                            "RECTIFIEDTRAJ_RUNTIME_DEVICE_EFFECTIVE",
                            os.getenv("RECTIFIEDTRAJ_DEVICE", "unknown"),
                        )
                    ).strip().lower().startswith("cuda") and torch.cuda.is_available()
                    cal_rss_before = float(proc.memory_info().rss) / (1024.0 * 1024.0)
                    if use_cuda:
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    cal_t0 = time.perf_counter()
                    model = create_baseline_model(
                        method_name=base_name,
                        dataset_name=chunk_dataset_name,
                        kalman_calibration_mode=(
                            mode if base_name == "kalman_rts" else None
                        ),
                    )
                    if use_cuda:
                        torch.cuda.synchronize()
                    cal_t1 = time.perf_counter()
                    cal_rss_after = float(proc.memory_info().rss) / (1024.0 * 1024.0)
                    calibration_time_sec = float(cal_t1 - cal_t0)
                    calibration_peak_rss_mb = max(cal_rss_before, cal_rss_after)
                    if use_cuda:
                        calibration_peak_vram_mb = (
                            float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
                        )
                    if base_name == "valhalla_meili":
                        timing = _measure_predict_repeats(
                            lambda: model.predict(chunk_seq_latlon_t),
                            repeats,
                        )
                    else:
                        timing = _measure_predict_repeats(
                            lambda: model.predict_enu(chunk_xy_enu, timestamps=chunk_ts_rel),
                            repeats,
                        )
                    avg_time = float(timing["avg_time_sec"])
                    avg_time_per_point = avg_time / chunk_num_points if chunk_num_points else None
                    throughput = (
                        (float(chunk_num_points) / avg_time)
                        if avg_time > 0 and chunk_num_points > 0
                        else None
                    )
                    reported_name = display_name or base_name
                    if base_name == "kalman_rts":
                        # Keep timing rows keyed by requested calibration mode for fairness reporting.
                        mode_label = str(mode or "dataset").strip() or "dataset"
                        reported_name = f"kalman_rts@{mode_label}"
                    _append_chunk_timing_summary_row(
                        manager=manager,
                        model_name=reported_name,
                        model_tag="Baseline",
                        dataset_name=chunk_dataset_name,
                        avg_time_sec=float(avg_time),
                        avg_time_sec_per_point=avg_time_per_point,
                        latency_p50_ms=timing["latency_p50_ms"],
                        latency_p95_ms=timing["latency_p95_ms"],
                        latency_max_ms=timing["latency_max_ms"],
                        throughput_points_per_sec=throughput,
                        peak_rss_mb=timing["peak_rss_mb"],
                        peak_vram_mb=timing["peak_vram_mb"],
                        calibration_time_sec=calibration_time_sec,
                        calibration_peak_rss_mb=calibration_peak_rss_mb,
                        calibration_peak_vram_mb=calibration_peak_vram_mb,
                        k=None,
                        q1=None,
                        q2=None,
                        t_delta=None,
                        n_steps=None,
                        device="cpu",
                    )
                except Exception as exc:
                    logging.warning("Chunk timing baseline %s failed: %s", display_name or base_name, exc)
                finally:
                    if model is not None:
                        try:
                            model.deconst()
                        except Exception:
                            pass


def _run_difftraj_chunk_baseline(
    manager: TestManager,
    job: dict,
    manual_config: dict | None,
    max_chunks: int | None,
) -> None:
    try:
        from baseline.difftraj import DiffTrajPaths, difftraj_denoise_with_model, prepare_difftraj
    except Exception as exc:
        logging.warning("DiffTraj chunk baseline import failed: %s", exc)
        return

    chunk_paths = [str(p).strip() for p in _as_list(job.get("chunk_dirs")) if str(p).strip()]
    if chunk_paths:
        test_dir = chunk_paths[0]
    else:
        test_dir = str(_require_job_field(job, "chunk_test_dir", "DiffTraj chunk baseline"))

    test_path = Path(test_dir)
    if test_path.is_file():
        loader_data_dir = str(test_path.parent)
        loader_pattern = test_path.name
    else:
        loader_data_dir = str(test_path)
        loader_pattern = "*.pt"
    try:
        loader = StandaloneDataLoader(
            mode="test",
            data_dir=loader_data_dir,
            file_pattern=loader_pattern,
            shuffle=False,
        )
    except Exception as exc:
        logging.warning("No chunk test files found for DiffTraj baseline (%s): %s", test_dir, exc)
        return

    def _as_tensor(x) -> torch.Tensor:
        if torch.is_tensor(x):
            return x.detach().cpu()
        return torch.as_tensor(x)

    x0_list: list[torch.Tensor] = []
    x1_list: list[torch.Tensor] = []
    limit = int(max_chunks) if (max_chunks is not None and int(max_chunks) > 0) else None
    for rec in loader.iter_test_records():
        payload = rec.get("payload", {})
        rtype = rec.get("record_type")
        x0_one = None
        x1_one = None
        if rtype == "chunk_pair":
            x0 = _as_tensor(payload.get("X0")).float()
            x1 = _as_tensor(payload.get("X1")).float()
            if x0.ndim == 2 and x1.ndim == 2 and x0.shape[1] >= 2 and x1.shape[1] >= 2:
                x0_one = x0[:, :2]
                x1_one = x1[:, :2]
        elif rtype == "train_triplet":
            xt = _as_tensor(payload.get("X_t")).float()
            v = _as_tensor(payload.get("V")).float()
            t_arr = _as_tensor(payload.get("t")).float().reshape(-1)
            if xt.ndim == 2 and v.ndim == 2 and xt.shape[1] >= 2 and v.shape[1] >= 2 and int(t_arr.numel()) > 0:
                t_scalar = float(t_arr[0].item())
                x0_one = xt[:, :2] - v[:, :2] * t_scalar
                x1_one = xt[:, :2] + v[:, :2] * (1.0 - t_scalar)
        if x0_one is None or x1_one is None:
            continue
        if x0_one.shape != x1_one.shape:
            continue
        if int(x0_one.shape[0]) <= 0:
            continue
        x0_list.append(x0_one)
        x1_list.append(x1_one)
        if limit is not None and limit > 0 and len(x0_list) >= limit:
            break

    if not x0_list:
        logging.warning("No chunks loaded for DiffTraj chunk baseline.")
        return

    x0 = torch.stack(x0_list, dim=0)
    x1 = torch.stack(x1_list, dim=0)
    num_chunks = int(x0.shape[0])
    if num_chunks <= 0:
        logging.warning("No chunks loaded for DiffTraj chunk baseline.")
        return

    repo_dir = job.get("difftraj_repo", "./bin/baseline_model/difftraj")
    checkpoint_path = job.get("difftraj_ckpt", "")
    device = job.get("difftraj_device", "cuda")
    timesteps = int(job.get("difftraj_timesteps", 100))
    final_steps = job.get("difftraj_final_steps")
    final_steps = int(final_steps) if final_steps is not None else None
    eta = float(job.get("difftraj_eta", 0.0))

    paths = DiffTrajPaths(
        repo_dir=repo_dir or None,
        checkpoint_path=checkpoint_path or None,
    )
    try:
        config, model, device = prepare_difftraj(paths, device=device)
    except Exception as exc:
        logging.warning("DiffTraj chunk baseline unavailable: %s", exc)
        return

    target_len = int(getattr(getattr(config, "data", None), "traj_length", 0) or 0)
    logging.warning(
        "DiffTraj chunk baseline started (chunks=%d, timesteps=%d, device=%s, target_len=%d).",
        num_chunks,
        timesteps,
        device,
        target_len,
    )

    def _pad_tail(values: np.ndarray, tgt_len: int) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.shape[0] == tgt_len:
            return values
        if values.shape[0] > tgt_len:
            return values[:tgt_len]
        tail = np.repeat(values[-1:], tgt_len - values.shape[0], axis=0)
        return np.concatenate([values, tail], axis=0)

    def _denoise_once(enu_xy: np.ndarray) -> np.ndarray:
        if target_len <= 0 or enu_xy.shape[0] == target_len:
            return difftraj_denoise_with_model(
                enu_xy,
                config=config,
                model=model,
                device=device,
                timesteps=timesteps,
                final_steps=final_steps,
                eta=eta,
            )
        if enu_xy.shape[0] < target_len:
            den = difftraj_denoise_with_model(
                _pad_tail(enu_xy, target_len),
                config=config,
                model=model,
                device=device,
                timesteps=timesteps,
                final_steps=final_steps,
                eta=eta,
            )
            return den[:enu_xy.shape[0]]

        out = []
        i = 0
        while i < enu_xy.shape[0]:
            chunk = _pad_tail(enu_xy[i: i + target_len], target_len)
            out.append(
                difftraj_denoise_with_model(
                    chunk,
                    config=config,
                    model=model,
                    device=device,
                    timesteps=timesteps,
                    final_steps=final_steps,
                    eta=eta,
                )
            )
            i += target_len
        return np.concatenate(out, axis=0)[:enu_xy.shape[0]]

    q1_bytes = (manual_config or {}).get("Q1", 1)
    q2_bytes = (manual_config or {}).get("Q2", 12)
    q1_points = int(q1_bytes) * 8
    q2_points = int(q2_bytes) * 8

    errs_full = []
    errs_mid = []
    failed = 0
    for i in range(num_chunks):
        inp = x1[i].numpy()
        gt = x0[i].numpy()
        try:
            pred = _denoise_once(inp)
        except Exception:
            failed += 1
            continue

        diff_full = pred - gt
        l2_full = np.sqrt((diff_full * diff_full).sum(axis=-1))
        errs_full.append(l2_full)

        if q2_points > 0:
            pred_mid = pred[q1_points:-q2_points]
            gt_mid = gt[q1_points:-q2_points]
        else:
            pred_mid = pred[q1_points:]
            gt_mid = gt[q1_points:]
        diff_mid = pred_mid - gt_mid
        l2_mid = np.sqrt((diff_mid * diff_mid).sum(axis=-1))
        errs_mid.append(l2_mid)
        if i == 0 or (i + 1) % 10 == 0 or (i + 1) == num_chunks:
            logging.warning("DiffTraj chunk baseline progress: %d/%d", i + 1, num_chunks)

    if not errs_full:
        logging.warning("DiffTraj chunk baseline produced no valid outputs (failed=%d).", failed)
        return

    errs_full = np.stack(errs_full, axis=0)
    errs_mid = np.stack(errs_mid, axis=0)
    row = {
        "model_name": "difftraj",
        "model_tag": "Baseline",
        "device": _normalize_device_label(device),
        "dataset_name": "chunk_test",
        "denoise_method": "N/A",
        "K": None,
        "Q1": None,
        "Q2": None,
        "t_delta": None,
        "N_steps": None,
        "err_mean_full": float(errs_full.mean()),
        "err_median_full": float(np.median(errs_full)),
        "err_p95_full": float(np.percentile(errs_full, 95)),
        "err_std_full": float(errs_full.std()),
        "err_mean_mid": float(errs_mid.mean()),
        "err_median_mid": float(np.median(errs_mid)),
        "err_p95_mid": float(np.percentile(errs_mid, 95)),
        "err_std_mid": float(errs_mid.std()),
        "calibration_time_sec": 0.0,
        "calibration_peak_rss_mb": None,
        "calibration_peak_vram_mb": None,
        "num_tested_chunks": int(errs_full.shape[0]),
        "test_timestamp": datetime.now().isoformat(),
    }
    manager.chunk_evaluator._append_row(row)
    logging.warning(
        "DiffTraj chunk baseline logged (chunks=%d, failed=%d).",
        int(errs_full.shape[0]),
        failed,
    )


def _run_chunk_eval(
    manager: TestManager,
    job: dict,
    model_root: str,
    model_names: list | None,
    classic_baselines: list[str],
    model_tag: str = "RectifiedTraj",
    run_baselines: bool = True,
    include_difftraj: bool = False,
) -> None:
    chunk_paths = [str(p).strip() for p in _as_list(job.get("chunk_dirs")) if str(p).strip()]
    if not chunk_paths:
        chunk_paths = [str(_require_job_field(job, "chunk_test_dir", "chunk evaluation"))]
    _stage(f"Chunk eval start | test_dirs={chunk_paths}")

    max_chunks = _resolve_chunk_max_chunks(job)
    manual_configs: list[dict | None] = []
    chunk_grid = bool(job.get("chunk_grid_search", False))
    if chunk_grid:
        q1_vals = _as_list(job.get("Q1")) or [1]
        q2_vals = _as_list(job.get("Q2")) or [12]
        t_vals = _as_list(job.get("t_delta")) or [1.0]
        for q1 in q1_vals:
            for q2 in q2_vals:
                for t_delta in t_vals:
                    manual_configs.append(
                        {
                            "Q1": int(q1),
                            "Q2": int(q2),
                            "t_delta": float(t_delta),
                        }
                    )
    else:
        manual_config = job.get("chunk_config")
        if manual_config is None:
            q1 = job.get("Q1")
            q2 = job.get("Q2")
            t_delta = job.get("t_delta")
            if q1 is not None or q2 is not None or t_delta is not None:
                manual_config = {
                    "Q1": q1[0] if isinstance(q1, list) and q1 else q1,
                    "Q2": q2[0] if isinstance(q2, list) and q2 else q2,
                    "t_delta": t_delta[0] if isinstance(t_delta, list) and t_delta else t_delta,
                }
                manual_config = {k: v for k, v in manual_config.items() if v is not None}
        manual_configs.append(manual_config)

    run_baseline = bool(run_baselines and job.get("run_baseline", job.get("baseline_once", True)))
    non_kalman_baselines: list[str] = []
    kalman_modes: list[str] = []
    for spec in classic_baselines:
        base_name, kalman_mode, _display = _split_baseline_spec(spec)
        if not base_name:
            continue
        if base_name != "kalman_rts":
            non_kalman_baselines.append(base_name)
            continue
        kalman_modes.append(kalman_mode or "textbook_default")
    non_kalman_baselines = _dedupe_keep_order(non_kalman_baselines)
    kalman_modes = _dedupe_keep_order(kalman_modes)

    try:
        for test_dir in chunk_paths:
            for manual_config in manual_configs:
                _stage(
                    "Chunk config run | dir=%s Q1=%s Q2=%s step=%s"
                    % (
                        test_dir,
                        (manual_config or {}).get("Q1"),
                        (manual_config or {}).get("Q2"),
                        (manual_config or {}).get("t_delta"),
                    )
                )

                # 1) Run RectifiedTraj + non-Kalman chunk baselines once.
                manager.run_chunk_evaluation(
                    model_names=model_names,
                    model_root=model_root,
                    model_tag=model_tag,
                    test_dir=test_dir,
                    max_chunks=max_chunks,
                    manual_config=manual_config,
                    run_baselines=bool(run_baseline and non_kalman_baselines),
                    baseline_methods=non_kalman_baselines,
                )

                # 2) Run Kalman chunk baseline once per requested calibration mode.
                if bool(run_baseline):
                    prev_mode = os.getenv("KALMAN_RTS_CALIBRATION_MODE")
                    try:
                        for kalman_mode in kalman_modes:
                            os.environ["KALMAN_RTS_CALIBRATION_MODE"] = str(kalman_mode)
                            _stage(
                                "Chunk kalman run | dir=%s mode=%s Q1=%s Q2=%s step=%s"
                                % (
                                    test_dir,
                                    kalman_mode,
                                    (manual_config or {}).get("Q1"),
                                    (manual_config or {}).get("Q2"),
                                    (manual_config or {}).get("t_delta"),
                                )
                            )
                            manager.run_chunk_evaluation(
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
    except TypeError:
        logging.warning(
            "run_chunk_evaluation() does not support baseline filtering in this environment; "
            "running chunk eval without classic baselines to avoid unintended methods."
        )
        for test_dir in chunk_paths:
            for manual_config in manual_configs:
                manager.run_chunk_evaluation(
                    model_names=model_names,
                    model_root=model_root,
                    model_tag=model_tag,
                    test_dir=test_dir,
                    max_chunks=max_chunks,
                    manual_config=manual_config,
                    run_baselines=False,
                )
        if bool(run_baseline) and classic_baselines:
            logging.warning("Skipped classic chunk baselines due to API mismatch.")
    if include_difftraj and bool(run_baseline) and DIFFTRAJ_ENABLED:
        _run_difftraj_chunk_baseline(
            manager=manager,
            job=job,
            manual_config=manual_configs[0] if manual_configs else None,
            max_chunks=max_chunks,
        )


def _run_difftraj_late_phase(
    manager: TestManager,
    job: dict,
    datasets: list[dict],
) -> None:
    _stage("Late DiffTraj phase start")
    for entry in datasets:
        test_trajectories, dataset_name = manager._load_or_generate_test_data(
            test_data_path=str(entry["path"]),
            M=int(entry["M"]),
            N=int(entry["N"]),
        )
        manager.trajectory_evaluator.set_run_context(dataset_name)
        _run_difftraj_baseline_trajectory(
            manager=manager,
            test_trajectories=test_trajectories,
            dataset_name=dataset_name,
            job=job,
        )

    if job.get("chunk_test", True):
        chunk_paths = [str(p).strip() for p in _as_list(job.get("chunk_dirs")) if str(p).strip()]
        if not chunk_paths:
            chunk_paths = [str(_require_job_field(job, "chunk_test_dir", "late DiffTraj chunk phase"))]
        _stage(f"Late DiffTraj chunk phase start | test_dirs={chunk_paths}")
        max_chunks = _resolve_chunk_max_chunks(job)
        manual_config = job.get("chunk_config")
        if manual_config is None:
            q1 = job.get("Q1")
            q2 = job.get("Q2")
            t_delta = job.get("t_delta")
            if q1 is not None or q2 is not None or t_delta is not None:
                manual_config = {
                    "Q1": q1[0] if isinstance(q1, list) and q1 else q1,
                    "Q2": q2[0] if isinstance(q2, list) and q2 else q2,
                    "t_delta": t_delta[0] if isinstance(t_delta, list) and t_delta else t_delta,
                }
                manual_config = {k: v for k, v in manual_config.items() if v is not None}
        for chunk_path in chunk_paths:
            job_local = dict(job)
            job_local["chunk_dirs"] = [chunk_path]
            job_local["chunk_test_dir"] = chunk_path
            _run_difftraj_chunk_baseline(
                manager=manager,
                job=job_local,
                manual_config=manual_config,
                max_chunks=max_chunks,
            )


def main() -> None:
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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    joblist_path = Path("./src/eval_joblist.json")
    if not joblist_path.exists():
        raise FileNotFoundError(f"Missing job list: {joblist_path}")

    with joblist_path.open("r") as f:
        job_raw = json.load(f)
    job = _normalize_job_schema(job_raw)
    runtime_device = _configure_encoder_decoder_device(job)
    _stage(f"RectifiedTraj runtime.device: {runtime_device}")
    kalman_mode, kalman_dataset = _apply_kalman_calibration_overrides(job)
    _stage(f"Kalman calibration mode: {kalman_mode} (dataset source: {kalman_dataset})")
    _stage(f"Test type: {job.get('test_type', 'exact')}")

    model_groups = list(job.get("model_groups") or [])
    if not model_groups:
        model_groups = [_build_primary_model_group_from_job(job)]
    normalized_groups: list[dict] = []
    for idx, raw_group in enumerate(model_groups):
        group = _normalize_model_group_schema_entry(
            raw_group,
            default_group=None,
            context=f"model_groups[{idx}]",
        )
        model_names = group.get("model_names")
        if model_names is not None and not model_names:
            group["model_names"] = None
        normalized_groups.append(group)
    model_groups = _dedupe_model_groups(normalized_groups)
    if not model_groups:
        raise ValueError("No valid model_groups after normalization.")
    job["model_groups"] = model_groups

    # Keep backward-compatible top-level learned-model fields aligned to primary group.
    primary_group = model_groups[0]
    job["data_hypothesis"] = primary_group["data_hypothesis"]
    job["model_root"] = primary_group["model_root"]
    job["model_names"] = primary_group["model_names"]
    job["methods"] = list(primary_group["methods"])
    job["Q1"] = list(primary_group["Q1"])
    job["Q2"] = list(primary_group["Q2"])
    job["t_delta"] = list(primary_group["t_delta"])

    progress_only = False
    log_level_name = str(job.get("log_level", "INFO")).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logging.getLogger().setLevel(log_level)
    _stage(f"Log level set to {logging.getLevelName(log_level)}")

    group_runs: list[dict] = []
    for group in model_groups:
        group_runs.append(
            {
                "group": group,
                "job_list": _build_job_list_from_group(group),
            }
        )
    classic_baselines = _resolve_classic_baselines(job)
    _stage(f"Classic baselines selected: {classic_baselines if classic_baselines else '[]'}")
    for idx, gr in enumerate(group_runs):
        group = gr["group"]
        _stage(
            "Learned model group[%d] | hypothesis=%s model_root=%s models=%s methods=%s"
            % (
                idx,
                group["data_hypothesis"],
                group["model_root"],
                group.get("model_names"),
                group["methods"],
            )
        )

    gen_new_test = bool(job.get("gen_new_test", False))
    use_new_traj = job.get("use_new_traj", {}) or {}
    traj_paths = job.get("traj_paths", {}) or {}
    traj_dirs = [str(p).strip() for p in _as_list(job.get("traj_dirs")) if str(p).strip()]
    if not traj_dirs and traj_paths.get("full_traj"):
        traj_dirs = [str(traj_paths.get("full_traj")).strip()]
    chunk_dirs = [str(p).strip() for p in _as_list(job.get("chunk_dirs")) if str(p).strip()]
    if not chunk_dirs:
        chunk_fallback = str(job.get("chunk_test_dir", "") or "").strip()
        if chunk_fallback:
            chunk_dirs = [chunk_fallback]
    traj_dirs, chunk_dirs = _ensure_expected_test_paths_from_data_source(job, traj_dirs, chunk_dirs)

    missing_autogen_inputs = _collect_missing_inputs_for_autogen(
        job,
        traj_dirs=traj_dirs,
        chunk_dirs=chunk_dirs,
        classic_baselines=classic_baselines,
    )
    if missing_autogen_inputs:
        _stage("Missing eval artifacts detected; entering data generation mode")
        for msg in missing_autogen_inputs:
            _stage(f" - {msg}")
        traj_dirs, chunk_dirs = _run_data_generation_mode(job)
        _stage(
            "Data generation finished; using generated paths "
            f"(traj={traj_dirs}, chunk={chunk_dirs})"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_folder_name = _build_result_folder_name(
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
    _stage(f"Saved eval joblist snapshot: {job_copy_path}")
    system_info_path = _write_system_info(Path(manager.output_dir), runtime_device)
    _stage(f"Saved system info snapshot: {system_info_path}")
    wandb_enabled = bool(job.get("wandb", False)) or bool(args.wandb)
    wandb_project = args.wandb_project or job.get("wandb_project", "rectifiedtraj_benchmarks")
    wandb_entity = args.wandb_entity or job.get("wandb_entity") or None
    wandb_run_name = args.wandb_run_name or job.get("wandb_run_name") or Path(manager.output_dir).name

    _stage("Preflight validation start")
    _preflight_validate_job(
        job,
        model_groups=model_groups,
        traj_dirs=traj_dirs,
        chunk_dirs=chunk_dirs,
        classic_baselines=classic_baselines,
    )
    _stage("Preflight validation passed")

    datasets: list[dict] = []
    for idx, path_value in enumerate(traj_dirs):
        entry = _resolve_existing_dataset(
            name=f"traj_{idx}",
            path_value=path_value,
            debug_path=DEBUG_FULL_TRAJ if (args.test and idx == 0) else None,
            use_debug=bool(args.test and idx == 0),
        )
        if entry is None:
            continue
        ds_name = Path(path_value).stem or f"traj_{idx}"
        datasets.append({"name": ds_name, "path": entry[0], "M": entry[1], "N": entry[2]})

    if not datasets and gen_new_test and not args.test:
        logging.info("Generating full_traj dataset...")
        output_dir = FULL_TRAJ_DIR
        if traj_paths.get("full_traj"):
            full_path = Path(traj_paths["full_traj"])
            output_dir = full_path if full_path.suffix == "" else full_path.parent
        pt_path = _generate_full_traj(output_dir, use_new_traj)
        meta = _load_metadata(pt_path)
        datasets.append(
            {
                "name": "full_traj",
                "path": pt_path,
                "M": int(meta["n_trajectories"]),
                "N": int(meta["median_length"]),
            }
        )

    datasets = _apply_cpu_dataset_caps(job, datasets)

    traj_test_enabled = bool(job.get("traj_test", True))
    if traj_test_enabled and not datasets and not job.get("range_test", False):
        raise ValueError("No valid trajectory datasets found. Provide test_files.traj_files or enable gen_new_test.")

    run_baseline = job.get("run_baseline", job.get("baseline_once", True))
    if traj_test_enabled and run_baseline and datasets:
        _stage("Baseline phase start")
        for entry in datasets:
            test_trajectories, dataset_name = manager._load_or_generate_test_data(
                test_data_path=str(entry["path"]),
                M=int(entry["M"]),
                N=int(entry["N"]),
            )
            _stage(
                f"Baseline dataset loaded | dataset={dataset_name} trajectories={len(test_trajectories)}"
            )
            manager.trajectory_evaluator.set_run_context(dataset_name)
            manager.classic_baseline_evaluator.progress_bar = bool(job.get("baseline_progress", True))
            manager.trajectory_evaluator.evaluate_baseline(test_trajectories, dataset_name=dataset_name)
            if classic_baselines:
                _run_classic_baselines_filtered_compat(
                    manager=manager,
                    test_trajectories=test_trajectories,
                    dataset_name=dataset_name,
                    methods=classic_baselines,
                )
            else:
                _stage("Classic baseline list is empty; skipping classic baselines.")

    # Run trajectory tests over the provided hyperparameter grid only.
    if traj_test_enabled and datasets:
        _stage("Grid phase start")
        for run_idx, run_item in enumerate(group_runs):
            group = run_item["group"]
            group_job_list = run_item["job_list"]
            _stage(
                "Grid learned group start | idx=%d hypothesis=%s model_root=%s"
                % (run_idx, group["data_hypothesis"], group["model_root"])
            )
            _run_grid_eval(
                manager,
                group_job_list,
                group["model_root"],
                group.get("model_names"),
                group["data_hypothesis"],
                datasets,
            )

    if job.get("range_test"):
        _stage("Range test phase start")
        for run_idx, run_item in enumerate(group_runs):
            group = run_item["group"]
            run_baselines_here = run_idx == 0
            _stage(
                "Range learned group start | idx=%d hypothesis=%s model_root=%s run_baselines=%s"
                % (
                    run_idx,
                    group["data_hypothesis"],
                    group["model_root"],
                    run_baselines_here,
                )
            )
            _run_bounded_eval(
                manager,
                job,
                group["model_root"],
                group.get("model_names"),
                group["methods"],
                classic_baselines,
                model_tag=group["data_hypothesis"],
                run_baselines=run_baselines_here,
            )

    if bool(job.get("time_test", True)):
        _stage("Time test phase start")
        for run_idx, run_item in enumerate(group_runs):
            group = run_item["group"]
            group_job_list = run_item["job_list"]
            run_baselines_here = run_idx == 0
            _stage(
                "Time learned group start | idx=%d hypothesis=%s model_root=%s run_baselines=%s"
                % (
                    run_idx,
                    group["data_hypothesis"],
                    group["model_root"],
                    run_baselines_here,
                )
            )
            _run_time_tests(
                manager,
                job,
                group_job_list,
                group["model_root"],
                group.get("model_names"),
                datasets,
                group["methods"],
                classic_baselines,
                model_tag=group["data_hypothesis"],
                run_classic_baselines=run_baselines_here,
                include_difftraj_timing=False,
            )

    if job.get("chunk_test", False):
        _stage("Chunk test phase start")
        for run_idx, run_item in enumerate(group_runs):
            group = run_item["group"]
            run_baselines_here = run_idx == 0
            _stage(
                "Chunk learned group start | idx=%d hypothesis=%s model_root=%s run_baselines=%s"
                % (
                    run_idx,
                    group["data_hypothesis"],
                    group["model_root"],
                    run_baselines_here,
                )
            )
            _run_chunk_eval(
                manager,
                job,
                group["model_root"],
                group.get("model_names"),
                classic_baselines,
                model_tag=group["data_hypothesis"],
                run_baselines=run_baselines_here,
            )

    # Visualizer intentionally disabled for progress-only runs.

    if wandb_enabled:
        try:
            log_run_to_wandb(
                run_dir=manager.output_dir,
                project=wandb_project,
                entity=wandb_entity,
                run_name=wandb_run_name,
            )
        except Exception as exc:
            logging.warning("W&B upload failed: %s", exc)

    run_difftraj_late = (
        bool(job.get("run_difftraj_baseline", True))
        and bool(run_baseline)
        and DIFFTRAJ_ENABLED
    )
    if bool(job.get("run_difftraj_baseline", True)) and not DIFFTRAJ_ENABLED:
        logging.info("DiffTraj is hard-disabled in this build; skipping DiffTraj baseline phase.")
    if run_difftraj_late:
        _run_difftraj_late_phase(
            manager=manager,
            job=job,
            datasets=datasets,
        )
        if wandb_enabled:
            try:
                log_run_to_wandb(
                    run_dir=manager.output_dir,
                    project=wandb_project,
                    entity=wandb_entity,
                    run_name=f"{wandb_run_name}_difftraj",
                )
            except Exception as exc:
                logging.warning("Second W&B upload (DiffTraj) failed: %s", exc)

    print("\n✓ Evaluation complete")


if __name__ == "__main__":
    main()
