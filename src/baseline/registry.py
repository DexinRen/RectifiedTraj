from __future__ import annotations

import logging
import os

from .artifacts import resolve_baseline_artifacts_from_state
from .base import BaselineModel
from .models import (
    EuclideanFilterBaselineModel,
    KalmanRTSBaselineModel,
    ValhallaMeiliBaselineModel,
)

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _normalize_kalman_calibration_mode(raw: str | None) -> str:
    if raw is None:
        raw = os.getenv("KALMAN_RTS_CALIBRATION_MODE", "dataset")
    token = str(raw).strip().lower().replace("-", "_")
    if token in {"dataset", "on_dataset", "ondataset", "per_dataset"}:
        return "dataset"
    if token in {"numosim_kanto", "numosim", "kanto", "numosim_only"}:
        return "numosim_kanto"
    if token in {"textbook_default", "textbook", "default", "defaults"}:
        return "textbook_default"
    raise ValueError(
        "Unsupported kalman calibration mode. "
        "Use one of: dataset, numosim_kanto, textbook_default."
    )


# ================================================================
# === Baseline Factory (Single Entry for Evaluators/Benchmarks)
# ================================================================
def create_baseline_model(
    method_name: str,
    *,
    dataset_name: str | None = None,
    calibration_file: str | None = None,
    map_file: str | None = None,
    state_dir: str = "./dataset/state",
    fallback_dataset: str = "NUMOSIM_Kanto",
    kalman_calibration_mode: str | None = None,
    kalman_calibration_dataset: str | None = None,
) -> BaselineModel:
    """
    Resolve artifacts + instantiate + initialize the requested baseline model.
    """
    name = str(method_name).strip().lower()
    strict_init = _env_bool("BASELINE_STRICT_INIT", True)

    # 1) Resolve optional dataset artifacts from state files.
    artifacts = resolve_baseline_artifacts_from_state(
        dataset_name_hint=dataset_name,
        state_dir=state_dir,
        fallback_dataset=fallback_dataset,
        strict_dataset_hint=bool(strict_init and dataset_name),
    )
    resolved_cal = calibration_file or artifacts.calibration_file
    resolved_map = map_file or artifacts.map_file
    kalman_mode = "dataset"
    allow_textbook_default = False
    kalman_source_dataset = None

    # 2) Route by method name.
    if name == "kalman_rts":
        kalman_mode = _normalize_kalman_calibration_mode(kalman_calibration_mode)
        if kalman_mode == "textbook_default":
            allow_textbook_default = True
            # Explicit textbook mode must not use any resolved artifact calibration.
            resolved_cal = None
        elif kalman_mode == "numosim_kanto" and calibration_file is None:
            kalman_source_dataset = (
                str(kalman_calibration_dataset).strip()
                if kalman_calibration_dataset
                else str(os.getenv("KALMAN_RTS_CALIBRATION_DATASET", "")).strip()
            ) or "NUMOSIM_Kanto"
            source_artifacts = resolve_baseline_artifacts_from_state(
                dataset_name_hint=kalman_source_dataset,
                state_dir=state_dir,
                fallback_dataset=fallback_dataset,
                strict_dataset_hint=True,
            )
            resolved_cal = source_artifacts.calibration_file

        # Kalman-RTS: timestamp-aware numeric calibration baseline.
        model = KalmanRTSBaselineModel(
            dataset_name=dataset_name,
            use_timestamps=True,
            allow_textbook_default=allow_textbook_default,
            fallback_dataset=fallback_dataset,
        )
    elif name == "valhalla_meili":
        # Map-matching baseline backed by Valhalla Meili HTTP service.
        model = ValhallaMeiliBaselineModel(dataset_name=dataset_name, use_timestamps=True)
    elif name in {"hampel", "savgol", "spline", "raw"}:
        # Euclidean filters: no calibration; deterministic smoothing.
        model = EuclideanFilterBaselineModel(
            method_name=name,
            dataset_name=dataset_name,
            use_timestamps=True,
        )
    else:
        raise ValueError(f"Unsupported baseline method: {method_name}")

    if strict_init and name == "kalman_rts" and kalman_mode != "textbook_default" and not resolved_cal:
        source_msg = (
            f"source_dataset={kalman_source_dataset} "
            if kalman_mode == "numosim_kanto"
            else ""
        )
        raise RuntimeError(
            "Baseline initialization rejected by strict fairness policy: "
            f"method={method_name} dataset={dataset_name} mode={kalman_mode} "
            f"{source_msg}requires calibration artifact but none was resolved. "
            "Set BASELINE_STRICT_INIT=0 to allow fallback behavior."
        )

    # 3) Initialize before timing; this may include calibration/server setup.
    summary = model.initialize(calibration_file=resolved_cal, map_file=resolved_map)
    status = str(summary.get("status", "unknown")).strip().lower()
    mode = str(summary.get("mode", "unknown")).strip().lower()

    # Fairness guard: required calibration baselines must be truly calibrated.
    if strict_init and bool(getattr(model, "requires_calibration", False)) and status != "ok":
        raise RuntimeError(
            "Baseline initialization rejected by strict fairness policy: "
            f"method={method_name} dataset={dataset_name} requires calibration, "
            f"got status={status} mode={mode}. "
            "Set BASELINE_STRICT_INIT=0 to allow fallback behavior."
        )

    logger.info(
        "Baseline initialized | method=%s dataset=%s calibration_mode=%s calibration_status=%s calibration_file=%s map_file=%s",
        method_name,
        dataset_name,
        mode,
        status,
        resolved_cal,
        resolved_map,
    )
    return model


__all__ = ["create_baseline_model"]
