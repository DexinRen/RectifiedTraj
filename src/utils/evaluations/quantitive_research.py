#!/usr/bin/env python3
"""Run hardcoded quantitative trajectory research exports and plots."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-rectifiedtraj")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from pymap3d import geodetic2enu


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


DATASET_PATH = (
    REPO_ROOT
    / "dataset/processed/mini_map/"
    "traj_map_capture_native_from_first_in_region_stop_when_full_200_5000_"
    "lon_139p65300_139p85300_lat_35p58500_35p78500.pt"
)
OUTPUT_ROOT = REPO_ROOT / "bin/test_results"
OUTPUT_PREFIX = "quantitive_research"
CALIBRATION_DATASET = "NUMOSIM_Kanto"
LEARNED_MANUAL_CONFIG = {"Q1": 0, "Q2": 0}
MEILI_SAMPLE_INTERVAL_SEC = 10.0
TRAJECTORY_BLUE = np.asarray([31.0 / 255.0, 119.0 / 255.0, 180.0 / 255.0], dtype=np.float64)
MINI_MAP_CALIBRATION_KEY = "mini_map"
MINI_MAP_BASELINE_CALIBRATION_ENTRY = {
    "dataset_name": MINI_MAP_CALIBRATION_KEY,
    "calibration_source": "hardcoded_quantitive_research_mini_map_direct",
    "timestamp_policy": "No dataset timestamp/dt fields are used; calibration and prediction use unit sample index only.",
    "kalman_rts_params": {
        "process_var": 17.921876744826385,
        "meas_var": 175.02876054572263,
        "init_pos_var": 169.00558168178534,
        "init_vel_var": 326.5402844234792,
    },
    "alpha_beta_params": {
        "alpha": 0.6375,
        "beta": 0.04,
    },
    "causal_hampel_params": {
        "window_size": 5,
        "n_sigma": 2.0,
    },
}

LEARNED_MODELS = [
    {
        "key": "rt_cnn",
        "method": "RectifiedTraj_cnn_online_1M_20260518_181708",
        "model_root": str(REPO_ROOT / "bin/model/RectifiedTraj_online"),
        "model_name": "cnn_online_1M_20260518_181708",
        "model_tag": "RectifiedTraj",
    },
    {
        "key": "rt_hybrid",
        "method": "RectifiedTraj_hybrid_online_1M_20260518_181215",
        "model_root": str(REPO_ROOT / "bin/model/RectifiedTraj_online"),
        "model_name": "hybrid_online_1M_20260518_181215",
        "model_tag": "RectifiedTraj",
    },
    {
        "key": "rt_transformer",
        "method": "RectifiedTraj_transformer_online_1M_20260518_181440",
        "model_root": str(REPO_ROOT / "bin/model/RectifiedTraj_online"),
        "model_name": "transformer_online_1M_20260518_181440",
        "model_tag": "RectifiedTraj",
    },
    {
        "key": "causal_mlp",
        "method": "CausalMLP_causal_mlp_1M_20260825_134854",
        "model_root": str(REPO_ROOT / "bin/model/DirectReg_online"),
        "model_name": "causal_mlp_1M_20260825_134854",
        "model_tag": "DirectReg",
    },
    {
        "key": "directreg_hybrid",
        "method": "DirectReg_hybrid_online_1M_20260523_180637",
        "model_root": str(REPO_ROOT / "bin/model/DirectReg_online"),
        "model_name": "hybrid_online_1M_20260523_180637",
        "model_tag": "DirectReg",
    },
    {
        "key": "ddim_hybrid",
        "method": "DDIM_hybrid_online_1M_20260809_161937_sample_500",
        "model_root": str(REPO_ROOT / "bin/model/Diffusion_online"),
        "model_name": "diffusion_hybrid_online_1M_20260809_161937",
        "model_tag": "Diffusion",
        "manual_config": {"sample_steps": 500},
    },
]

BASELINE_METHODS = [
    "alpha_beta",
    "causal_hampel",
    "kalman_filter",
    "kalman_rts",
    "hampel",
    "savgol",
    "raw",
    "valhalla_meili",
]
MINI_MAP_CALIBRATED_BASELINES = {
    "alpha_beta",
    "causal_hampel",
    "kalman_filter",
    "kalman_rts",
}
PAPER_PANEL_FILENAMES = {
    "ground_truth": "qual_ground_truth.png",
    "baseline_raw": "qual_raw.png",
    "RectifiedTraj_hybrid_online_1M_20260518_181215": "qual_rt_hybrid.png",
    "RectifiedTraj_transformer_online_1M_20260518_181440": "qual_rt_trans.png",
    "RectifiedTraj_cnn_online_1M_20260518_181708": "qual_rt_cnn.png",
    "CausalMLP_causal_mlp_1M_20260825_134854": "qual_causal_mlp.png",
    "DirectReg_hybrid_online_1M_20260523_180637": "qual_directreg_hybrid.png",
    "DDIM_hybrid_online_1M_20260809_161937_sample_500": "qual_ddim_hybrid.png",
    "baseline_valhalla_meili": "qual_valhalla_meili.png",
    "baseline_hampel": "qual_hampel.png",
    "baseline_kalman_rts": "qual_kalman_rts.png",
    "baseline_savgol": "qual_savgol.png",
}
VALHALLA_BASELINE_OPTIONS = {
    "valhalla_meili": {
        "profiles": {
            "NUMOSIM_Kanto": {
                "map_id": "NUMOSIM_Kanto",
                "source": "japan",
                "costing": "auto",
                "port": 8003,
            }
        }
    }
}


@dataclass(frozen=True)
class MethodSpec:
    key: str
    method: str
    method_type: str
    model_root: str | None = None
    model_name: str | None = None
    model_tag: str | None = None
    baseline_name: str | None = None
    baseline_calibration_file: str | None = None
    baseline_calibration_entry: dict[str, Any] | None = None
    manual_config: dict[str, Any] | None = None
    baseline_config: dict[str, Any] | None = None
    dataset_name_override: str | None = None
    sample_interval_sec: float | None = None


def _safe_name(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._@-]+", "_", str(value or ""))
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "item"


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_dataset(
    dataset_path: Path,
    *,
    max_trajectories: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    blob = torch.load(dataset_path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        raise ValueError(f"Unsupported dataset payload type in {dataset_path}: {type(blob)}")
    rows = blob.get("trajectories")
    if not isinstance(rows, list):
        raise ValueError(f"Dataset has no trajectories list: {dataset_path}")
    metadata = blob.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = dict(metadata)
    if max_trajectories is not None:
        limit = int(max_trajectories)
        if limit <= 0:
            raise ValueError("max_trajectories must be positive when provided.")
        rows = rows[:limit]
        metadata["qualitative_subset_trajectories"] = int(len(rows))
        metadata["qualitative_subset_policy"] = "first_n_in_saved_dataset_order"
    return rows, metadata


def _infer_dataset_name_from_path(path_value: str | Path | None) -> str | None:
    if path_value is None:
        return None
    parts = list(Path(path_value).parts)
    for idx, part in enumerate(parts):
        if str(part).lower() == "processed" and idx + 1 < len(parts):
            return str(parts[idx + 1])
    return None


def _dataset_display_name(dataset_path: Path, metadata: dict[str, Any]) -> str:
    root_name = _infer_dataset_name_from_path(dataset_path)
    source_dataset = str(metadata.get("source_dataset", "") or "").strip()
    if root_name:
        return root_name
    if source_dataset:
        return source_dataset
    return dataset_path.stem


def _clean_lonlat_pair(noisy: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    noisy = np.asarray(noisy, dtype=np.float64)
    clean = np.asarray(clean, dtype=np.float64)
    n = min(int(noisy.shape[0]), int(clean.shape[0]))
    noisy = noisy[:n]
    clean = clean[:n]
    mask = (
        np.isfinite(noisy[:, 0])
        & np.isfinite(noisy[:, 1])
        & np.isfinite(clean[:, 0])
        & np.isfinite(clean[:, 1])
    )
    return noisy[mask], clean[mask]


def _trajectory_l2_error_m(pred_lonlat: np.ndarray, gt_lonlat: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred_lonlat, dtype=np.float64)
    gt = np.asarray(gt_lonlat, dtype=np.float64)
    n = min(int(pred.shape[0]), int(gt.shape[0]))
    if n <= 0:
        return np.asarray([], dtype=np.float64)
    pred = pred[:n]
    gt = gt[:n]
    ref_lat = float(gt[0, 1])
    ref_lon = float(gt[0, 0])
    pred_e, pred_n, _ = geodetic2enu(pred[:, 1], pred[:, 0], 0.0, ref_lat, ref_lon, 0.0)
    gt_e, gt_n, _ = geodetic2enu(gt[:, 1], gt[:, 0], 0.0, ref_lat, ref_lon, 0.0)
    delta_e = np.asarray(pred_e, dtype=np.float64) - np.asarray(gt_e, dtype=np.float64)
    delta_n = np.asarray(pred_n, dtype=np.float64) - np.asarray(gt_n, dtype=np.float64)
    return np.sqrt(delta_e * delta_e + delta_n * delta_n)


def _build_result_payload(
    *,
    method: str,
    method_type: str,
    dataset_path: Path,
    metadata: dict[str, Any],
    predictions: list[tuple[np.ndarray, dict[str, Any]]],
    started_at: str,
    elapsed_sec: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trajectories: list[dict[str, Any]] = []
    all_point_errors: list[np.ndarray] = []
    for pred, row in predictions:
        noisy, gt = _clean_lonlat_pair(_to_numpy(row["data"]), _to_numpy(row["label"]))
        pred_arr = np.asarray(pred, dtype=np.float64)
        n = min(int(noisy.shape[0]), int(gt.shape[0]), int(pred_arr.shape[0]))
        if n <= 0:
            continue
        noisy = noisy[:n]
        gt = gt[:n]
        pred_arr = pred_arr[:n]
        point_l2 = _trajectory_l2_error_m(pred_arr, gt)
        if point_l2.size <= 0:
            continue
        avg_l2 = float(np.mean(point_l2, dtype=np.float64))
        all_point_errors.append(point_l2)
        trajectories.append(
            {
                "agent_id": row.get("agent_id"),
                "record_index": len(trajectories),
                "n_points": int(n),
                "noisy": noisy.astype(np.float32, copy=False),
                "ground_truth": gt.astype(np.float32, copy=False),
                "prediction": pred_arr.astype(np.float32, copy=False),
                "point_l2_error_m": point_l2.astype(np.float32, copy=False),
                "avg_l2_error_m": avg_l2,
                "log_norm_avg_error": None,
                "line_width": None,
            }
        )

    if not trajectories:
        raise RuntimeError(f"No valid trajectories produced for {method}")

    if all_point_errors:
        flat_errors = np.concatenate(all_point_errors, axis=0)
        avg_point_error = float(np.mean(flat_errors, dtype=np.float64))
    else:
        avg_point_error = float("nan")

    return {
        "schema_version": 1,
        "method": method,
        "method_type": method_type,
        "dataset_path": str(dataset_path),
        "metadata": _jsonable(metadata),
        "started_at": started_at,
        "elapsed_sec": float(elapsed_sec),
        "avg_point_l2_error_m": avg_point_error,
        "num_trajectories": int(len(trajectories)),
        "num_points": int(sum(int(t["n_points"]) for t in trajectories)),
        "extra": _jsonable(extra or {}),
        "trajectories": trajectories,
    }


def _find_checkpoint(model_dir: Path) -> Path:
    for rel in ["best_ckpt", "ckpts"]:
        ckpt_dir = model_dir / rel
        if not ckpt_dir.exists():
            continue
        safetensors = sorted(ckpt_dir.glob("*.safetensors"))
        if safetensors:
            return safetensors[0]
        full_pts = sorted(ckpt_dir.glob("*_full.pt"))
        if full_pts:
            return full_pts[0]
    raise FileNotFoundError(f"No checkpoint found under {model_dir}")


def _evaluate_learned(spec: MethodSpec, rows: list[dict[str, Any]]) -> tuple[list[tuple[np.ndarray, dict[str, Any]]], dict[str, Any]]:
    from learned_decoder import build_learned_decoder

    if not spec.model_root or not spec.model_name:
        raise ValueError(f"Invalid learned spec: {spec}")
    model_dir = Path(spec.model_root) / spec.model_name
    ckpt_path = _find_checkpoint(model_dir)
    manual_config = dict(LEARNED_MANUAL_CONFIG)
    manual_config.update(dict(spec.manual_config or {}))
    decoder = build_learned_decoder(str(ckpt_path), manual_config=manual_config)
    predictions: list[tuple[np.ndarray, dict[str, Any]]] = []
    for row in rows:
        noisy, _gt = _clean_lonlat_pair(_to_numpy(row["data"]), _to_numpy(row["label"]))
        pred = decoder.denoise_traj_DF(noisy)
        predictions.append((np.asarray(pred, dtype=np.float64), row))
    t_delta = getattr(decoder, "t_delta", None)
    data_hypothesis = getattr(decoder, "data_hypothesis", None)
    if data_hypothesis is None:
        data_hypothesis = dict(getattr(decoder, "cfg", {}) or {}).get(
            "data_hypothesis",
            spec.model_tag,
        )
    return predictions, {
        "key": spec.key,
        "model_root": spec.model_root,
        "model_name": spec.model_name,
        "model_tag": spec.model_tag,
        "checkpoint_path": str(ckpt_path),
        "manual_config": manual_config,
        "eval_q": 0,
        "K": int(decoder.K),
        "Q1": int(decoder.Q1_bytes),
        "Q2": int(decoder.Q2_bytes),
        "t_delta": None if t_delta is None else float(t_delta),
        "sample_steps": getattr(decoder, "sample_steps", None),
        "data_hypothesis": str(data_hypothesis),
    }


def _apply_hardcoded_baseline_calibration(
    model: Any,
    *,
    baseline_name: str,
    calibration_entry: dict[str, Any],
) -> dict[str, Any]:
    from baseline import classic as classic_baseline

    summary = {
        "status": "ok",
        "mode": "hardcoded_quantitive_research",
        "dataset_name": str(calibration_entry.get("dataset_name", MINI_MAP_CALIBRATION_KEY)),
        "calibration_source": str(calibration_entry.get("calibration_source", "")),
        "timestamp_policy": str(calibration_entry.get("timestamp_policy", "")),
    }
    name = str(baseline_name).strip().lower()
    if name in {"kalman_filter", "kalman_rts"}:
        raw = calibration_entry["kalman_rts_params"]
        params = classic_baseline.KalmanParams(
            process_var=float(raw["process_var"]),
            meas_var=float(raw["meas_var"]),
            init_pos_var=float(raw["init_pos_var"]),
            init_vel_var=float(raw["init_vel_var"]),
        )
        impl = getattr(model, "_impl", None)
        if impl is None:
            raise RuntimeError(f"{baseline_name} model has no initialized Kalman implementation.")
        impl.params = params
        summary["params"] = _jsonable(raw)
        impl.calibration_summary = dict(summary)
    elif name == "alpha_beta":
        raw = calibration_entry["alpha_beta_params"]
        model.params = classic_baseline.AlphaBetaParams(
            alpha=float(raw["alpha"]),
            beta=float(raw["beta"]),
        )
        summary["params"] = _jsonable(raw)
    elif name == "causal_hampel":
        raw = calibration_entry["causal_hampel_params"]
        model.params = classic_baseline.CausalHampelParams(
            window_size=max(1, int(round(float(raw["window_size"])))),
            n_sigma=float(raw["n_sigma"]),
        )
        summary["params"] = _jsonable(raw)
    else:
        summary = {
            "status": "not_applicable",
            "mode": "hardcoded_quantitive_research",
            "reason": f"{baseline_name} has no mini-map calibration override.",
        }

    model.calibration_summary = dict(summary)
    return summary


def _evaluate_baseline(
    spec: MethodSpec,
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
) -> tuple[list[tuple[np.ndarray, dict[str, Any]]], dict[str, Any]]:
    from baseline import build_lat_lon_timestamp_sequence_from_lonlat, create_baseline_model, latlon_to_lonlat

    if not spec.baseline_name:
        raise ValueError(f"Invalid baseline spec: {spec}")
    effective_dataset_name = str(spec.dataset_name_override or dataset_name)
    model_calibration_file = spec.baseline_calibration_file
    calibration_entry = spec.baseline_calibration_entry
    calibration_override_summary = None
    if calibration_entry is not None:
        model_calibration_file = None
    elif spec.baseline_name == "alpha_beta" and spec.baseline_calibration_file:
        from baseline import classic as classic_baseline

        alpha_beta_params, calibration_summary = (
            classic_baseline.estimate_alpha_beta_params_from_calibration_file(
                spec.baseline_calibration_file,
                default_params=classic_baseline.AlphaBetaParams(),
                seed_kalman_params=None,
            )
        )
        calibration_override_summary = {
            "status": "ok",
            "mode": "artifact_default_alpha_beta_seed",
            **calibration_summary,
        }
        # Avoid AlphaBetaBaselineModel's default Kalman-seeded calibration path,
        # which is too narrow for the mini-map diagnostic dataset.
        model_calibration_file = None

    model = create_baseline_model(
        method_name=spec.baseline_name,
        dataset_name=effective_dataset_name,
        calibration_file=model_calibration_file,
        fallback_dataset=CALIBRATION_DATASET,
        baseline_config=spec.baseline_config,
    )
    if calibration_entry is not None:
        calibration_override_summary = _apply_hardcoded_baseline_calibration(
            model,
            baseline_name=spec.baseline_name,
            calibration_entry=calibration_entry,
        )
    elif spec.baseline_name == "alpha_beta" and calibration_override_summary is not None:
        model.params = alpha_beta_params
        model.calibration_summary = calibration_override_summary
    predictions: list[tuple[np.ndarray, dict[str, Any]]] = []
    meili_attempted_points = 0
    meili_accepted_points = 0
    meili_fallback_points = 0
    meili_complete_trajectories = 0
    meili_partial_trajectories = 0
    meili_rejected_trajectories = 0
    try:
        for row in rows:
            noisy, _gt = _clean_lonlat_pair(_to_numpy(row["data"]), _to_numpy(row["label"]))
            timestamps = None
            if spec.sample_interval_sec is not None:
                timestamps = np.arange(len(noisy), dtype=np.float64) * float(
                    spec.sample_interval_sec
                )
            seq = build_lat_lon_timestamp_sequence_from_lonlat(
                noisy,
                timestamps=timestamps,
            )
            if spec.baseline_name == "valhalla_meili":
                packet = model.predict_packet(seq)
                denoised_latlon = np.asarray(packet["positions_latlon"], dtype=np.float64)
                diagnostics = dict(packet.get("diagnostics") or {})
                attempted = int(diagnostics.get("attempted_points", len(noisy)))
                accepted = int(diagnostics.get("accepted_points", 0))
                fallback = int(diagnostics.get("fallback_points", attempted - accepted))
                meili_attempted_points += attempted
                meili_accepted_points += accepted
                meili_fallback_points += fallback
                if bool(packet.get("complete")):
                    meili_complete_trajectories += 1
                elif accepted > 0:
                    meili_partial_trajectories += 1
                else:
                    meili_rejected_trajectories += 1
            else:
                denoised_latlon = model.predict(seq)
            pred = latlon_to_lonlat(denoised_latlon)
            predictions.append((np.asarray(pred, dtype=np.float64), row))
    finally:
        model.deconst()
    extra = {
        "key": spec.key,
        "baseline_name": spec.baseline_name,
        "dataset_name": effective_dataset_name,
        "calibration_dataset": CALIBRATION_DATASET,
        "baseline_calibration_file": spec.baseline_calibration_file,
        "calibration_mode": (
            "mini_map_hardcoded"
            if calibration_entry is not None
            else "mini_map_file"
            if spec.baseline_calibration_file
            else str(os.getenv("KALMAN_RTS_CALIBRATION_MODE", "dataset")).strip()
        ),
        "calibration_summary": getattr(model, "calibration_summary", {}),
    }
    if spec.baseline_name == "valhalla_meili":
        extra["timestamp_policy"] = (
            f"synthetic regular timestamps at {float(spec.sample_interval_sec):g}s intervals"
        )
        extra["meili_coverage"] = {
            "attempted_points": meili_attempted_points,
            "accepted_points": meili_accepted_points,
            "fallback_points": meili_fallback_points,
            "point_fallback_rate": (
                float(meili_fallback_points) / float(meili_attempted_points)
                if meili_attempted_points
                else 0.0
            ),
            "complete_trajectories": meili_complete_trajectories,
            "partial_trajectories": meili_partial_trajectories,
            "rejected_trajectories": meili_rejected_trajectories,
            "fallback_policy": "raw_input",
        }
    return predictions, extra


def _evaluate_method_worker(
    spec_data: dict[str, Any],
    dataset_path: str,
    output_path: str,
    max_trajectories: int | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    started_at = datetime.now().isoformat()
    spec = MethodSpec(**spec_data)
    output = Path(output_path)
    try:
        rows, metadata = _load_dataset(
            Path(dataset_path),
            max_trajectories=max_trajectories,
        )
        if spec.method_type == "learned":
            predictions, extra = _evaluate_learned(spec, rows)
        elif spec.method_type == "baseline":
            predictions, extra = _evaluate_baseline(
                spec,
                rows,
                dataset_name=_dataset_display_name(Path(dataset_path), metadata),
            )
        else:
            raise ValueError(f"Unsupported method_type={spec.method_type!r}")
        payload = _build_result_payload(
            method=spec.method,
            method_type=spec.method_type,
            dataset_path=Path(dataset_path),
            metadata=metadata,
            predictions=predictions,
            started_at=started_at,
            elapsed_sec=time.perf_counter() - start,
            extra=extra,
        )
        torch.save(payload, output)
        return {
            "method": spec.method,
            "method_type": spec.method_type,
            "status": "ok",
            "output_path": str(output),
            "elapsed_sec": payload["elapsed_sec"],
            "avg_point_l2_error_m": payload["avg_point_l2_error_m"],
            "num_trajectories": payload["num_trajectories"],
            "num_points": payload["num_points"],
        }
    except Exception as exc:
        return {
            "method": spec.method,
            "method_type": spec.method_type,
            "status": "failed",
            "output_path": str(output),
            "error": str(exc),
            "traceback": traceback.format_exc(),
    }


def _configure_runtime_device(*, use_cpu: bool) -> str:
    device = "cpu" if bool(use_cpu) else "cuda"
    os.environ["RECTIFIEDTRAJ_DEVICE"] = device
    os.environ["RECTIFIEDTRAJ_RUNTIME_DEVICE_EFFECTIVE"] = device
    os.environ["KALMAN_RTS_CALIBRATION_MODE"] = "numosim_kanto"
    os.environ["KALMAN_RTS_CALIBRATION_DATASET"] = CALIBRATION_DATASET
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for the normal quantitative run, but PyTorch cannot see a CUDA device. "
                "Use --cpu only for debug smoke checks."
            )
    return device


def _write_ground_truth_payload(
    dataset_path: Path,
    output_path: Path,
    *,
    max_trajectories: int | None = None,
) -> dict[str, Any]:
    rows, metadata = _load_dataset(
        dataset_path,
        max_trajectories=max_trajectories,
    )
    predictions = []
    for row in rows:
        _noisy, gt = _clean_lonlat_pair(_to_numpy(row["data"]), _to_numpy(row["label"]))
        predictions.append((gt, row))
    payload = _build_result_payload(
        method="ground_truth",
        method_type="ground_truth",
        dataset_path=dataset_path,
        metadata=metadata,
        predictions=predictions,
        started_at=datetime.now().isoformat(),
        elapsed_sec=0.0,
        extra={"description": "Ground-truth trajectories with zero prediction error."},
    )
    torch.save(payload, output_path)
    return {
        "method": "ground_truth",
        "method_type": "ground_truth",
        "status": "ok",
        "output_path": str(output_path),
        "elapsed_sec": 0.0,
        "avg_point_l2_error_m": 0.0,
        "num_trajectories": payload["num_trajectories"],
        "num_points": payload["num_points"],
    }


def _write_mini_map_hardcoded_calibration(dataset_path: Path, output_path: Path) -> dict[str, Any]:
    _rows, metadata = _load_dataset(dataset_path)
    entry = {
        **MINI_MAP_BASELINE_CALIBRATION_ENTRY,
        "dataset_path": str(dataset_path),
        "metadata": _jsonable(metadata),
    }
    payload = {
        "calib_json_equivalent": {
            MINI_MAP_CALIBRATION_KEY: entry,
        },
        "note": (
            "This run uses hard-coded mini-map baseline params inside quantitive_research.py. "
            "The shared dataset/state/calib.json is intentionally not modified."
        ),
    }
    _write_json(output_path, payload)
    return {
        "path": str(output_path),
        "entry": entry,
    }


def _iter_payload_paths(prediction_dir: Path) -> list[Path]:
    return sorted(prediction_dir.glob("*.pt"), key=lambda p: p.name)


def _normalize_average_errors(avg_errors: np.ndarray, *, width_scale: str) -> tuple[np.ndarray, dict[str, Any]]:
    scale = str(width_scale).strip().lower()
    if scale == "log":
        scaled = np.log1p(avg_errors)
        min_value = float(np.min(scaled))
        max_value = float(np.max(scaled))
        denom = max_value - min_value
        if denom <= 0.0:
            norm = np.zeros_like(scaled, dtype=np.float64)
        else:
            norm = (scaled - min_value) / denom
        return norm, {
            "width_scale": "log",
            "log_min": min_value,
            "log_max": max_value,
            "formula": "line_width = 1 + normalized(log1p(avg_error_m)) * (max_linewidth - 1)",
        }

    max_error = float(np.max(avg_errors))
    if max_error <= 0.0:
        norm = np.zeros_like(avg_errors, dtype=np.float64)
    elif scale == "sqrt":
        norm = np.sqrt(avg_errors / max_error)
    elif scale == "linear":
        norm = avg_errors / max_error
    else:
        raise ValueError(f"Unsupported width scale: {width_scale!r}. Use sqrt, log, or linear.")
    return norm, {
        "width_scale": scale,
        "raw_error_max_m": max_error,
        "formula": f"line_width = 1 + {scale}(avg_error_m / global_max_error_m) * (max_linewidth - 1)",
    }


def _apply_global_error_width_normalization(
    prediction_dir: Path,
    *,
    max_linewidth: float,
    width_scale: str,
) -> dict[str, Any]:
    paths = _iter_payload_paths(prediction_dir)
    if not paths:
        raise FileNotFoundError(f"No prediction payloads found in {prediction_dir}")

    avg_errors: list[float] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for traj in payload.get("trajectories", []):
            value = float(traj.get("avg_l2_error_m", 0.0) or 0.0)
            if np.isfinite(value):
                avg_errors.append(max(value, 0.0))
    if not avg_errors:
        raise RuntimeError("No finite trajectory average errors found for normalization.")

    avg_error_arr = np.asarray(avg_errors, dtype=np.float64)
    normalized_errors, norm_summary = _normalize_average_errors(
        avg_error_arr,
        width_scale=width_scale,
    )
    norm_by_error = {
        float(err): float(norm)
        for err, norm in zip(avg_error_arr.tolist(), normalized_errors.tolist(), strict=True)
    }
    max_width = max(float(max_linewidth), 1.0)

    method_summaries: list[dict[str, Any]] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        norm_values: list[float] = []
        width_values: list[float] = []
        for traj in payload.get("trajectories", []):
            err = max(float(traj.get("avg_l2_error_m", 0.0) or 0.0), 0.0)
            norm = norm_by_error.get(float(err))
            if norm is None:
                norm, _unused = _normalize_average_errors(
                    np.asarray([err], dtype=np.float64),
                    width_scale=width_scale,
                )
                norm = float(norm[0])
            norm = float(np.clip(norm, 0.0, 1.0))
            width = 1.0 + norm * (max_width - 1.0)
            traj["norm_avg_error"] = norm
            traj["width_scale"] = norm_summary["width_scale"]
            traj["log_norm_avg_error"] = norm
            traj["line_width"] = float(width)
            norm_values.append(norm)
            width_values.append(float(width))
        payload["global_error_width_normalization"] = {
            "included_ground_truth": True,
            "max_linewidth": max_width,
            **norm_summary,
        }
        payload["global_log_normalization"] = payload["global_error_width_normalization"]
        torch.save(payload, path)
        method_summaries.append(
            {
                "method": payload.get("method"),
                "path": str(path),
                "mean_norm_avg_error": float(np.mean(norm_values)) if norm_values else None,
                "min_line_width": float(np.min(width_values)) if width_values else None,
                "max_line_width": float(np.max(width_values)) if width_values else None,
            }
        )

    return {
        "included_ground_truth": True,
        "num_trajectory_errors": int(len(avg_errors)),
        "raw_error_min_m": float(np.min(avg_error_arr)),
        "raw_error_max_m": float(np.max(avg_error_arr)),
        "max_linewidth": max_width,
        **norm_summary,
        "method_summaries": method_summaries,
    }


def _resolve_bbox(payload: dict[str, Any]) -> tuple[float, float, float, float]:
    metadata = payload.get("metadata", {})
    bbox = metadata.get("region_bbox") if isinstance(metadata, dict) else None
    if isinstance(bbox, dict):
        return (
            float(bbox["min_lon"]),
            float(bbox["max_lon"]),
            float(bbox["min_lat"]),
            float(bbox["max_lat"]),
        )
    coords: list[np.ndarray] = []
    for traj in payload.get("trajectories", []):
        pred = np.asarray(traj.get("prediction"), dtype=np.float64)
        if pred.ndim == 2 and pred.shape[1] >= 2:
            coords.append(pred[:, :2])
    if not coords:
        raise ValueError(f"Cannot resolve plotting bbox for {payload.get('method')}")
    xy = np.concatenate(coords, axis=0)
    margin_lon = max((float(np.nanmax(xy[:, 0])) - float(np.nanmin(xy[:, 0]))) * 0.02, 1e-6)
    margin_lat = max((float(np.nanmax(xy[:, 1])) - float(np.nanmin(xy[:, 1]))) * 0.02, 1e-6)
    return (
        float(np.nanmin(xy[:, 0]) - margin_lon),
        float(np.nanmax(xy[:, 0]) + margin_lon),
        float(np.nanmin(xy[:, 1]) - margin_lat),
        float(np.nanmax(xy[:, 1]) + margin_lat),
    )


def _inside_bbox(xy: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    min_lon, max_lon, min_lat, max_lat = bbox
    arr = np.asarray(xy, dtype=np.float64)
    return (
        np.isfinite(arr[:, 0])
        & np.isfinite(arr[:, 1])
        & (arr[:, 0] >= min_lon)
        & (arr[:, 0] <= max_lon)
        & (arr[:, 1] >= min_lat)
        & (arr[:, 1] <= max_lat)
    )


def _plot_payload(payload_path: Path, plot_dir: Path, *, alpha: float = 0.72, dpi: int = 180) -> dict[str, Any]:
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    method = str(payload.get("method", payload_path.stem))
    bbox = _resolve_bbox(payload)

    segments: list[np.ndarray] = []
    widths: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    visible_trajectories = 0
    visible_segments = 0
    for traj in payload.get("trajectories", []):
        pred = np.asarray(traj.get("prediction"), dtype=np.float64)
        if pred.ndim != 2 or pred.shape[1] < 2 or pred.shape[0] < 2:
            continue
        gt = np.asarray(traj.get("ground_truth"), dtype=np.float64)
        n = min(int(pred.shape[0]), int(gt.shape[0]))
        if n < 2:
            continue
        pred = pred[:n]
        gt = gt[:n]
        pred_point_mask = _inside_bbox(pred[:, :2], bbox)
        gt_point_mask = _inside_bbox(gt[:, :2], bbox)
        seg_mask = (
            pred_point_mask[:-1]
            & pred_point_mask[1:]
            & gt_point_mask[:-1]
            & gt_point_mask[1:]
        )
        if not np.any(seg_mask):
            continue
        width = traj.get("line_width")
        if width is None:
            width = 1.0
        width = float(width)
        seg = np.stack([pred[:-1, :2][seg_mask], pred[1:, :2][seg_mask]], axis=1).astype(
            np.float32,
            copy=False,
        )
        segments.append(seg)
        widths.append(np.full((seg.shape[0],), width, dtype=np.float32))
        colors.append(np.repeat(TRAJECTORY_BLUE[None, :], seg.shape[0], axis=0).astype(np.float32, copy=False))
        visible_trajectories += 1
        visible_segments += int(seg.shape[0])

    fig, ax = plt.subplots(figsize=(8.4, 8.4), dpi=int(dpi))
    if segments:
        collection = LineCollection(
            np.concatenate(segments, axis=0),
            colors=np.concatenate(colors, axis=0),
            linewidths=np.concatenate(widths, axis=0),
            alpha=float(alpha),
            capstyle="round",
            joinstyle="round",
        )
        ax.add_collection(collection)

    min_lon, max_lon, min_lat, max_lat = bbox
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)

    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / f"{_safe_name(method)}.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    return {
        "method": method,
        "plot_path": str(out_path),
        "visible_trajectories": int(visible_trajectories),
        "visible_segments": int(visible_segments),
        "bbox_segment_filter": "prediction_and_ground_truth_both_endpoints_inside",
    }


def _plot_all(prediction_dir: Path, plot_dir: Path) -> list[dict[str, Any]]:
    paths = _iter_payload_paths(prediction_dir)
    active_plot_names = {_safe_name(path.stem) + ".png" for path in paths}
    if plot_dir.exists():
        for old_plot in plot_dir.glob("*.png"):
            if old_plot.name not in active_plot_names:
                old_plot.unlink()
    return [_plot_payload(path, plot_dir) for path in paths]


def _export_paper_panels(
    plot_results: list[dict[str, Any]],
    export_dir: Path,
) -> dict[str, Any]:
    export_dir.mkdir(parents=True, exist_ok=True)
    plot_by_method = {
        str(item["method"]): Path(str(item["plot_path"]))
        for item in plot_results
    }
    exported: list[dict[str, str]] = []
    missing: list[str] = []
    for method, filename in PAPER_PANEL_FILENAMES.items():
        source = plot_by_method.get(method)
        if source is None or not source.is_file():
            missing.append(method)
            continue
        destination = export_dir / filename
        shutil.copy2(source, destination)
        exported.append(
            {
                "method": method,
                "source": str(source),
                "destination": str(destination),
            }
        )
    summary = {
        "export_dir": str(export_dir),
        "exported": exported,
        "missing": missing,
    }
    _write_json(export_dir / "paper_panel_manifest.json", summary)
    return summary


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8")


def _build_specs(
    *,
    baseline_calibration_file: str | None = None,
    baseline_calibration_entry: dict[str, Any] | None = None,
    baselines_only: bool = False,
    selected_keys: set[str] | None = None,
    valhalla_config: dict[str, Any] | None = None,
) -> list[MethodSpec]:
    specs: list[MethodSpec] = []
    if not baselines_only:
        specs.extend(
            MethodSpec(
                key=str(item["key"]),
                method=str(item["method"]),
                method_type="learned",
                model_root=str(item["model_root"]),
                model_name=str(item["model_name"]),
                model_tag=str(item["model_tag"]),
                manual_config=dict(item.get("manual_config") or {}),
            )
            for item in LEARNED_MODELS
            if selected_keys is None or str(item["key"]) in selected_keys
        )
    for name in BASELINE_METHODS:
        if selected_keys is not None and name not in selected_keys:
            continue
        if name == "valhalla_meili" and valhalla_config is None:
            raise ValueError("Valhalla Meili was selected without a resolved map profile.")
        specs.append(
            MethodSpec(
                key=name,
                method=f"baseline_{name}",
                method_type="baseline",
                baseline_name=name,
                baseline_calibration_file=baseline_calibration_file,
                baseline_calibration_entry=(
                    baseline_calibration_entry
                    if baseline_calibration_entry is not None
                    and name in MINI_MAP_CALIBRATED_BASELINES
                    else None
                ),
                baseline_config=valhalla_config if name == "valhalla_meili" else None,
                dataset_name_override=(
                    CALIBRATION_DATASET if name == "valhalla_meili" else None
                ),
                sample_interval_sec=(
                    MEILI_SAMPLE_INTERVAL_SEC if name == "valhalla_meili" else None
                ),
            ),
        )
    return specs


def _parse_selected_method_keys(raw_values: list[str] | None) -> set[str] | None:
    if not raw_values:
        return None
    selected: set[str] = set()
    for raw in raw_values:
        selected.update(token.strip() for token in str(raw).split(",") if token.strip())
    available = {str(item["key"]) for item in LEARNED_MODELS} | set(BASELINE_METHODS)
    unknown = sorted(selected - available)
    if unknown:
        raise ValueError(
            f"Unknown qualitative method key(s): {', '.join(unknown)}. "
            f"Available keys: {', '.join(sorted(available))}."
        )
    return selected


def _resolve_qualitative_valhalla_profile() -> dict[str, Any]:
    from utils.evaluations.trajectory_batch_runner import resolve_valhalla_profile

    return resolve_valhalla_profile(
        CALIBRATION_DATASET,
        VALHALLA_BASELINE_OPTIONS,
    )


def _make_output_dir(output_root: Path) -> Path:
    date_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_root / f"{OUTPUT_PREFIX}_{date_tag}"
    if not base.exists():
        return base
    suffix = 1
    while True:
        candidate = output_root / f"{OUTPUT_PREFIX}_{date_tag}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def _payload_result_summary(
    path: Path,
    *,
    reused: bool = False,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported qualitative payload type in {path}: {type(payload)}")
    return {
        "method": str(payload.get("method", path.stem)),
        "method_type": str(payload.get("method_type", "unknown")),
        "status": "ok",
        "output_path": str(path),
        "elapsed_sec": float(payload.get("elapsed_sec", 0.0) or 0.0),
        "avg_point_l2_error_m": float(payload.get("avg_point_l2_error_m", float("nan"))),
        "num_trajectories": int(payload.get("num_trajectories", 0) or 0),
        "num_points": int(payload.get("num_points", 0) or 0),
        "reused": bool(reused),
    }


def run_full(args: argparse.Namespace) -> Path:
    runtime_device = _configure_runtime_device(use_cpu=bool(args.cpu))
    dataset_path = DATASET_PATH
    if not dataset_path.exists():
        raise FileNotFoundError(f"Hardcoded dataset not found: {dataset_path}")

    selected_keys = _parse_selected_method_keys(args.methods)
    max_trajectories = args.max_trajectories
    if max_trajectories is not None and int(max_trajectories) <= 0:
        raise ValueError("--max-trajectories must be positive.")
    if args.resume_run and max_trajectories is not None:
        raise ValueError(
            "--resume-run cannot be combined with --max-trajectories because it could mix "
            "full-dataset and subset payloads in one result directory."
        )

    if args.resume_run:
        run_dir = Path(args.resume_run).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Qualitative run directory not found: {run_dir}")
    else:
        run_dir = _make_output_dir(Path(args.output_root).resolve())
    prediction_dir = run_dir / "trajectory_data"
    plot_dir = run_dir / "plots"
    calibration_dir = run_dir / "calibration"
    paper_panel_dir = run_dir / "paper_panels"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    calibration_dir.mkdir(parents=True, exist_ok=True)
    paper_panel_dir.mkdir(parents=True, exist_ok=True)

    baseline_calibration = None
    baseline_calibration_file = None
    baseline_calibration_entry = None
    if not bool(args.numosim_baseline_calibration):
        baseline_calibration_path = calibration_dir / "mini_map_hardcoded_calibration.json"
        baseline_calibration = _write_mini_map_hardcoded_calibration(
            dataset_path,
            baseline_calibration_path,
        )
        baseline_calibration_entry = baseline_calibration["entry"]

    needs_valhalla = selected_keys is None or "valhalla_meili" in selected_keys
    valhalla_profile = (
        _resolve_qualitative_valhalla_profile() if needs_valhalla else None
    )
    specs = _build_specs(
        baseline_calibration_file=baseline_calibration_file,
        baseline_calibration_entry=baseline_calibration_entry,
        baselines_only=bool(args.baselines_only),
        selected_keys=selected_keys,
        valhalla_config=(
            dict(valhalla_profile["config"])
            if valhalla_profile is not None
            else None
        ),
    )
    if not specs:
        raise ValueError("No qualitative methods remain after applying the CLI selection.")
    manifest: dict[str, Any] = {
        "status": "running",
        "created_at": datetime.now().isoformat(),
        "dataset_path": str(dataset_path),
        "output_dir": str(run_dir),
        "trajectory_data_dir": str(prediction_dir),
        "plot_dir": str(plot_dir),
        "paper_panel_dir": str(paper_panel_dir),
        "calibration_dir": str(calibration_dir),
        "calibration_dataset": CALIBRATION_DATASET,
        "baseline_calibration_source": (
            "numosim_kanto"
            if bool(args.numosim_baseline_calibration)
            else "mini_map_hardcoded"
        ),
        "baseline_calibration": baseline_calibration,
        "runtime_device": runtime_device,
        "parallel_workers": int(args.max_workers),
        "max_linewidth": float(args.max_linewidth),
        "width_scale": str(args.width_scale),
        "baselines_only": bool(args.baselines_only),
        "selected_method_keys": sorted(selected_keys) if selected_keys is not None else None,
        "max_trajectories": max_trajectories,
        "resume_run": str(run_dir) if args.resume_run else None,
        "force": bool(args.force),
        "valhalla_profile": valhalla_profile,
        "learned_models": LEARNED_MODELS,
        "baselines": BASELINE_METHODS,
        "results": [],
    }
    _write_json(run_dir / "manifest.json", manifest)

    ground_truth_path = prediction_dir / "ground_truth.pt"
    if ground_truth_path.is_file() and not bool(args.force):
        results = [_payload_result_summary(ground_truth_path, reused=True)]
        print("[quantitive] reused ground_truth.pt")
    else:
        results = [
            _write_ground_truth_payload(
                dataset_path,
                ground_truth_path,
                max_trajectories=max_trajectories,
            )
        ]
        print("[quantitive] saved ground_truth.pt")

    worker_count = max(1, int(args.max_workers))
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as pool:
        futures = {}
        for spec in specs:
            out_path = prediction_dir / f"{_safe_name(spec.method)}.pt"
            if out_path.is_file() and not bool(args.force):
                results.append(_payload_result_summary(out_path, reused=True))
                print(f"[quantitive] reused: {spec.method}")
                continue
            futures[
                pool.submit(
                    _evaluate_method_worker,
                    spec.__dict__,
                    str(dataset_path),
                    str(out_path),
                    max_trajectories,
                )
            ] = spec
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = result.get("status")
            print(f"[quantitive] {status}: {result.get('method')}")

    failures = [item for item in results if item.get("status") != "ok"]
    if failures:
        manifest.update(
            {
                "status": "failed",
                "finished_at": datetime.now().isoformat(),
                "results": results,
                "failures": failures,
            }
        )
        _write_json(run_dir / "manifest.json", manifest)
        raise RuntimeError(f"{len(failures)} quantitative task(s) failed. See {run_dir / 'manifest.json'}")

    normalization = _apply_global_error_width_normalization(
        prediction_dir,
        max_linewidth=float(args.max_linewidth),
        width_scale=str(args.width_scale),
    )
    plot_results = _plot_all(prediction_dir, plot_dir)
    paper_panel_export = _export_paper_panels(plot_results, paper_panel_dir)

    manifest.update(
        {
            "status": "ok",
            "finished_at": datetime.now().isoformat(),
            "results": results,
            "global_error_width_normalization": normalization,
            "global_log_normalization": normalization,
            "plots": plot_results,
            "paper_panel_export": paper_panel_export,
        }
    )
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(
        run_dir / "plot_manifest.json",
        {
            "plots": plot_results,
            "paper_panel_export": paper_panel_export,
        },
    )
    return run_dir


def run_plot_only(args: argparse.Namespace) -> Path:
    _configure_runtime_device(use_cpu=True)
    run_dir = Path(args.plot_only).resolve()
    prediction_dir = run_dir / "trajectory_data"
    plot_dir = run_dir / "plots"
    normalization = _apply_global_error_width_normalization(
        prediction_dir,
        max_linewidth=float(args.max_linewidth),
        width_scale=str(args.width_scale),
    )
    plot_results = _plot_all(prediction_dir, plot_dir)
    paper_panel_export = _export_paper_panels(
        plot_results,
        run_dir / "paper_panels",
    )
    _write_json(
        run_dir / "plot_manifest.json",
        {
            "updated_at": datetime.now().isoformat(),
            "global_error_width_normalization": normalization,
            "global_log_normalization": normalization,
            "plots": plot_results,
            "paper_panel_export": paper_panel_export,
        },
    )
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run hardcoded quantitative research predictions and plots."
    )
    parser.add_argument(
        "--output-root",
        default=str(OUTPUT_ROOT),
        help="Root directory for quantitive_research_<date> outputs.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=min(4, max(1, (os.cpu_count() or 1))),
        help="Parallel evaluation worker count.",
    )
    parser.add_argument(
        "--max-linewidth",
        type=float,
        default=10.0,
        help="Maximum plotted trajectory linewidth after global error-width normalization.",
    )
    parser.add_argument(
        "--width-scale",
        choices=("sqrt", "log", "linear"),
        default="log",
        help="Scale used to map trajectory average error to plotted linewidth.",
    )
    parser.add_argument(
        "--plot-only",
        default=None,
        help="Existing quantitive_research run directory to re-normalize and re-plot.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Debug only: run learned-model inference on CPU instead of requiring CUDA.",
    )
    parser.add_argument(
        "--numosim-baseline-calibration",
        action="store_true",
        help="Use the old NUMOSIM_Kanto baseline calibration instead of hard-coded mini_map baseline params.",
    )
    parser.add_argument(
        "--baselines-only",
        action="store_true",
        help="Run only ground truth and classic baselines; useful for calibration diagnosis.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help=(
            "Optional method keys, separated by spaces or commas. Available learned keys: "
            + ", ".join(str(item["key"]) for item in LEARNED_MODELS)
            + "; baseline keys: "
            + ", ".join(BASELINE_METHODS)
            + "."
        ),
    )
    parser.add_argument(
        "--resume-run",
        default=None,
        help="Existing qualitative run directory in which missing selected payloads are added.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute selected payloads even when their .pt files already exist.",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Deterministic first-N trajectory subset for smoke testing a new run.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.plot_only:
        run_dir = run_plot_only(args)
    else:
        run_dir = run_full(args)
    print(f"[quantitive] output: {run_dir}")


if __name__ == "__main__":
    main()
