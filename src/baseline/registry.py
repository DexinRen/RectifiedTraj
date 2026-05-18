from __future__ import annotations

import logging
import os

from . import classic as classic_baseline
from .artifacts import resolve_baseline_artifacts_from_state
from .base import BaselineModel
from .models import (
    AlphaBetaBaselineModel,
    CausalHampelBaselineModel,
    EuclideanFilterBaselineModel,
    KalmanFilterBaselineModel,
    KalmanRTSBaselineModel,
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
    token = str(raw).strip()
    if token == "dataset":
        return "dataset"
    if token == "numosim_kanto":
        return "numosim_kanto"
    raise ValueError(
        f"Unsupported kalman calibration mode={raw!r}. "
        "Recognized values: dataset, numosim_kanto."
    )


def create_baseline_model(
    method_name: str,
    *,
    dataset_name: str | None = None,
    calibration_file: str | None = None,
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

    artifacts = resolve_baseline_artifacts_from_state(
        dataset_name_hint=dataset_name,
        state_dir=state_dir,
        fallback_dataset=fallback_dataset,
        strict_dataset_hint=bool(strict_init and dataset_name),
    )
    resolved_cal = calibration_file or artifacts.calibration_file
    kalman_mode = "dataset"
    kalman_source_dataset = None

    if name in {"kalman_rts", "kalman_filter"}:
        kalman_mode = _normalize_kalman_calibration_mode(kalman_calibration_mode)
        params_entry = classic_baseline.resolve_kalman_params_entry_from_state(
            dataset_name_hint=dataset_name,
            state_dir=state_dir,
            fallback_dataset=fallback_dataset,
        )
        if calibration_file is None and params_entry is not None:
            resolved_cal = None
        if kalman_mode == "numosim_kanto" and calibration_file is None:
            kalman_source_dataset = (
                str(kalman_calibration_dataset).strip()
                if kalman_calibration_dataset
                else str(os.getenv("KALMAN_RTS_CALIBRATION_DATASET", "")).strip()
            ) or "NUMOSIM_Kanto"
            params_entry = classic_baseline.resolve_kalman_params_entry_from_state(
                dataset_name_hint=kalman_source_dataset,
                state_dir=state_dir,
                fallback_dataset=fallback_dataset,
            )
            source_artifacts = resolve_baseline_artifacts_from_state(
                dataset_name_hint=kalman_source_dataset,
                state_dir=state_dir,
                fallback_dataset=fallback_dataset,
                strict_dataset_hint=True,
            )
            resolved_cal = None if params_entry is not None else source_artifacts.calibration_file

        if name == "kalman_rts":
            model = KalmanRTSBaselineModel(
                dataset_name=dataset_name,
                use_timestamps=True,
                fallback_dataset=fallback_dataset,
            )
        else:
            model = KalmanFilterBaselineModel(
                dataset_name=dataset_name,
                use_timestamps=True,
                fallback_dataset=fallback_dataset,
            )
    elif name == "alpha_beta":
        model = AlphaBetaBaselineModel(
            dataset_name=dataset_name,
            use_timestamps=True,
            fallback_dataset=fallback_dataset,
        )
    elif name == "causal_hampel":
        model = CausalHampelBaselineModel(
            dataset_name=dataset_name,
            use_timestamps=True,
            fallback_dataset=fallback_dataset,
        )
    elif name in {"hampel", "savgol", "raw"}:
        model = EuclideanFilterBaselineModel(
            method_name=name,
            dataset_name=dataset_name,
            use_timestamps=True,
        )
    else:
        raise ValueError(f"Unsupported baseline method: {method_name}")

    if strict_init and name in {"kalman_rts", "kalman_filter"} and not resolved_cal:
        params_entry = classic_baseline.resolve_kalman_params_entry_from_state(
            dataset_name_hint=kalman_source_dataset if kalman_mode == "numosim_kanto" else dataset_name,
            state_dir=state_dir,
            fallback_dataset=fallback_dataset,
        )
        if params_entry is not None:
            resolved_cal = None
        else:
            source_msg = (
                f"source_dataset={kalman_source_dataset} "
                if kalman_mode == "numosim_kanto"
                else ""
            )
            raise RuntimeError(
                "Baseline initialization rejected by strict fairness policy: "
                f"method={method_name} dataset={dataset_name} mode={kalman_mode} "
                f"{source_msg}requires calibration artifact or state_json params but none was resolved. "
                "Set BASELINE_STRICT_INIT=0 to allow fallback behavior."
            )

    summary = model.initialize(calibration_file=resolved_cal)
    status = str(summary.get("status", "unknown")).strip().lower()
    mode = str(summary.get("mode", "unknown")).strip().lower()

    if strict_init and bool(getattr(model, "requires_calibration", False)) and status != "ok":
        raise RuntimeError(
            "Baseline initialization rejected by strict fairness policy: "
            f"method={method_name} dataset={dataset_name} requires calibration, "
            f"got status={status} mode={mode}. "
            "Set BASELINE_STRICT_INIT=0 to allow fallback behavior."
        )

    logger.debug(
        "Baseline initialized | method=%s dataset=%s calibration_mode=%s calibration_status=%s calibration_file=%s",
        method_name,
        dataset_name,
        mode,
        status,
        resolved_cal,
    )
    return model


__all__ = ["create_baseline_model"]
