#!/usr/bin/env python3
"""
Baseline denoising methods for 2D trajectories.

Implements:
  1) Kalman filter + RTS smoother
  2) Hampel-like spike removal
  3) Savitzky-Golay filter
  4) Smoothing spline on x(t), y(t)
  5) difftraj (stub; to be implemented later)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
from pymap3d import enu2geodetic, geodetic2enu
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)


@dataclass
class KalmanParams:
    process_var: float = 1.0
    meas_var: float = 5.0
    init_pos_var: float = 10.0
    init_vel_var: float = 10.0


def _safe_dataset_token(name: str | None) -> str | None:
    if not name:
        return None
    token = str(name).strip()
    if not token:
        return None
    for sep in ["/", "\\", ":", "."]:
        token = token.replace(sep, "_")
    token = "_".join(part for part in token.split("_") if part)
    return token or None


def _state_candidate_files(
    dataset_name_hint: str | None,
    state_dir: str,
    fallback_dataset: str,
) -> list[Path]:
    state_root = Path(state_dir)
    all_states = sorted(state_root.glob("state_*.json"))
    out: list[Path] = []

    def _push(path: Path) -> None:
        if path not in out:
            out.append(path)

    hint = _safe_dataset_token(dataset_name_hint)
    if hint:
        _push(state_root / f"state_{hint}.json")
        hint_lower = hint.lower()
        for path in all_states:
            if hint_lower in path.stem.lower():
                _push(path)

    fallback = _safe_dataset_token(fallback_dataset)
    if fallback:
        _push(state_root / f"state_{fallback}.json")

    for path in all_states:
        _push(path)
    return out


def _extract_kalman_params_from_payload(payload: dict) -> KalmanParams | None:
    if not isinstance(payload, dict):
        return None
    raw = payload.get("kalman_rts_params")
    if not isinstance(raw, dict):
        return None
    keys = ("process_var", "meas_var", "init_pos_var", "init_vel_var")
    vals: dict[str, float] = {}
    for key in keys:
        if key not in raw:
            return None
        vals[key] = float(raw[key])
    return KalmanParams(**vals)


def resolve_kalman_calibration_file_from_state(
    dataset_name_hint: str | None = None,
    state_dir: str = "./dataset/state",
    fallback_dataset: str = "NUMOSIM_Kanto",
) -> str | None:
    """
    Resolve canonical calibration artifact path from dataset state payload.
    """
    for state_path in _state_candidate_files(dataset_name_hint, state_dir, fallback_dataset):
        if not state_path.exists():
            continue
        try:
            with open(state_path, "r") as f:
                payload = json.load(f)
            parquet = payload.get("parquet_processor", {}) if isinstance(payload, dict) else {}
            calibration = parquet.get("calibration_native", {}) if isinstance(parquet, dict) else {}
            raw_path = (
                calibration.get("path")
                if isinstance(calibration, dict)
                else None
            ) or (
                calibration.get("native_source_output")
                if isinstance(calibration, dict)
                else None
            )
            if not raw_path:
                continue
            path = Path(str(raw_path))
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            if path.exists():
                return str(path)
        except Exception:
            continue
    return None


def load_kalman_params_from_state(
    dataset_name_hint: str | None = None,
    state_dir: str = "./dataset/state",
    fallback_dataset: str = "NUMOSIM_Kanto",
) -> KalmanParams:
    """
    Load Kalman params from dataset state file with fallback to NUMOSIM defaults.
    """
    default = KalmanParams()
    for path in _state_candidate_files(dataset_name_hint, state_dir, fallback_dataset):
        if not path.exists():
            continue
        try:
            with open(path, "r") as f:
                payload = json.load(f)
            params = _extract_kalman_params_from_payload(payload)
            if params is not None:
                return params
        except Exception:
            continue

    return default


def _as_float_array(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _prepare_timestamps(timestamps: Optional[np.ndarray], n: int) -> np.ndarray:
    if timestamps is None:
        return np.arange(n, dtype=float)
    t = _as_float_array(timestamps).reshape(-1)
    if t.size != n:
        raise ValueError(f"timestamps length {t.size} != n {n}")
    # Fix non-increasing steps to avoid degenerate dynamics
    dt = np.diff(t)
    if dt.size == 0:
        return t
    positive_dt = dt[dt > 0]
    fallback = float(np.median(positive_dt)) if positive_dt.size else 1.0
    dt = np.where(dt <= 0, fallback, dt)
    t_fixed = np.concatenate([[t[0]], t[0] + np.cumsum(dt)])
    return t_fixed


def _collapse_duplicate_t(t: np.ndarray, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    t = _as_float_array(t).reshape(-1)
    pos = _as_float_array(positions)
    if t.size != pos.shape[0]:
        raise ValueError("timestamps and positions length mismatch")
    uniq, inv = np.unique(t, return_inverse=True)
    if uniq.size == t.size:
        return t, pos
    sums = np.zeros((uniq.size, 2), dtype=float)
    counts = np.zeros(uniq.size, dtype=float)
    for i, idx in enumerate(inv):
        sums[idx] += pos[i]
        counts[idx] += 1.0
    return uniq, sums / counts[:, None]


def kalman_rts_smoother(
    positions: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    params: Optional[KalmanParams] = None,
) -> np.ndarray:
    """Kalman filter with RTS smoothing for 2D positions.

    positions: (N, 2) array of [x, y]
    timestamps: optional (N,) array, used for variable dt
    """
    if params is None:
        params = KalmanParams()
    pos = _as_float_array(positions)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("positions must be shape (N, 2)")
    n = pos.shape[0]
    if n == 0:
        return pos.copy()

    t = _prepare_timestamps(timestamps, n)
    dt = np.diff(t, prepend=t[0])
    dt[0] = dt[1] if n > 1 else 1.0

    # State: [x, y, vx, vy]
    x = np.zeros((n, 4), dtype=float)
    P = np.zeros((n, 4, 4), dtype=float)
    x_pred = np.zeros((n, 4), dtype=float)
    P_pred = np.zeros((n, 4, 4), dtype=float)

    # Init
    x[0, 0:2] = pos[0]
    x[0, 2:4] = 0.0
    P[0] = np.diag(
        [params.init_pos_var, params.init_pos_var, params.init_vel_var, params.init_vel_var]
    )

    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    R = np.eye(2, dtype=float) * params.meas_var

    for k in range(1, n):
        dtk = float(dt[k])
        F = np.array(
            [[1, 0, dtk, 0], [0, 1, 0, dtk], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=float,
        )
        q = params.process_var
        q11 = (dtk**4) / 4.0 * q
        q12 = (dtk**3) / 2.0 * q
        q22 = (dtk**2) * q
        Q = np.array(
            [[q11, 0, q12, 0], [0, q11, 0, q12], [q12, 0, q22, 0], [0, q12, 0, q22]],
            dtype=float,
        )

        # Predict
        x_pred[k] = F @ x[k - 1]
        P_pred[k] = F @ P[k - 1] @ F.T + Q

        # Update (skip if NaN measurement)
        z = pos[k]
        if np.any(~np.isfinite(z)):
            x[k] = x_pred[k]
            P[k] = P_pred[k]
            continue

        y = z - (H @ x_pred[k])
        S = H @ P_pred[k] @ H.T + R
        K = P_pred[k] @ H.T @ np.linalg.inv(S)
        x[k] = x_pred[k] + K @ y
        P[k] = (np.eye(4) - K @ H) @ P_pred[k]

    # RTS smoother
    x_s = x.copy()
    P_s = P.copy()
    for k in range(n - 2, -1, -1):
        dtk = float(dt[k + 1])
        F = np.array(
            [[1, 0, dtk, 0], [0, 1, 0, dtk], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=float,
        )
        if np.linalg.cond(P_pred[k + 1]) > 1e12:
            continue
        Ck = P[k] @ F.T @ np.linalg.inv(P_pred[k + 1])
        x_s[k] = x[k] + Ck @ (x_s[k + 1] - x_pred[k + 1])
        P_s[k] = P[k] + Ck @ (P_s[k + 1] - P_pred[k + 1]) @ Ck.T

    return x_s[:, 0:2]


def _to_numpy_array(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return np.asarray(x.detach().cpu().numpy())
    return np.asarray(x)


def estimate_kalman_params_from_calibration_file(
    calibration_file: str | Path,
    *,
    default_params: KalmanParams | None = None,
    max_trajectories: int | None = None,
) -> tuple[KalmanParams, dict]:
    """
    Estimate Kalman-RTS hyperparameters from calibration trajectory file.

    Expected file format:
      {
        "trajectories": [
          {"data": Tensor[N,2], "label": Tensor[N,2], "timestamp": Tensor[N]?},
          ...
        ]
      }
    where GPS order is [lon, lat].
    """
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("Torch is required for Kalman calibration file loading.") from exc

    path = Path(str(calibration_file))
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    blob = torch.load(path, map_location="cpu")
    trajectories = blob.get("trajectories", []) if isinstance(blob, dict) else []
    if not isinstance(trajectories, list) or not trajectories:
        raise ValueError(f"Calibration file has no trajectories: {path}")

    if max_trajectories is not None and max_trajectories > 0:
        trajectories = trajectories[: int(max_trajectories)]

    meas_sq_all: list[np.ndarray] = []
    init_pos_sq_all: list[float] = []
    init_vel_sq_all: list[float] = []
    accel_sq_all: list[np.ndarray] = []
    points_used = 0
    traj_used = 0

    for row in trajectories:
        if not isinstance(row, dict):
            continue

        data = row.get("data")
        label = row.get("label")
        if data is None or label is None:
            continue
        noisy_gps = _to_numpy_array(data)
        clean_gps = _to_numpy_array(label)
        if noisy_gps.ndim != 2 or clean_gps.ndim != 2:
            continue
        if noisy_gps.shape[1] < 2 or clean_gps.shape[1] < 2:
            continue

        t_len = min(noisy_gps.shape[0], clean_gps.shape[0])
        if t_len < 2:
            continue
        noisy_gps = noisy_gps[:t_len, :2].astype(np.float64, copy=False)
        clean_gps = clean_gps[:t_len, :2].astype(np.float64, copy=False)

        valid = np.isfinite(noisy_gps).all(axis=1) & np.isfinite(clean_gps).all(axis=1)
        if int(np.sum(valid)) < 2:
            continue
        noisy_gps = noisy_gps[valid]
        clean_gps = clean_gps[valid]
        n = noisy_gps.shape[0]

        timestamps = None
        if "timestamp" in row:
            ts = _to_numpy_array(row.get("timestamp")).reshape(-1)
            if ts.size >= t_len:
                ts = ts[:t_len][valid]
                if ts.size == n:
                    timestamps = ts.astype(np.float64, copy=False)

        ref_lon = float(clean_gps[0, 0])
        ref_lat = float(clean_gps[0, 1])
        enu_noisy = np.stack(
            geodetic2enu(
                noisy_gps[:, 1],
                noisy_gps[:, 0],
                0.0,
                ref_lat,
                ref_lon,
                0.0,
            )[:2],
            axis=1,
        )
        enu_clean = np.stack(
            geodetic2enu(
                clean_gps[:, 1],
                clean_gps[:, 0],
                0.0,
                ref_lat,
                ref_lon,
                0.0,
            )[:2],
            axis=1,
        )
        residual = enu_noisy - enu_clean
        meas_sq = np.sum(residual * residual, axis=1)
        meas_sq = meas_sq[np.isfinite(meas_sq)]
        if meas_sq.size == 0:
            continue
        meas_sq_all.append(meas_sq)
        init_pos_sq_all.append(float(np.dot(residual[0], residual[0])))

        t = _prepare_timestamps(timestamps, n)
        dt = np.diff(t)
        positive_dt = dt[dt > 0]
        fallback_dt = float(np.median(positive_dt)) if positive_dt.size else 1.0
        dt = np.where(dt <= 0, fallback_dt, dt)

        if dt.size > 0:
            v_ref = (enu_clean[1:] - enu_clean[:-1]) / dt[:, None]
            v_noisy = (enu_noisy[1:] - enu_noisy[:-1]) / dt[:, None]
            init_vel_err = v_noisy[0] - v_ref[0]
            init_vel_sq_all.append(float(np.dot(init_vel_err, init_vel_err)))
            if v_ref.shape[0] >= 2:
                dv = v_ref[1:] - v_ref[:-1]
                dt_v = dt[1:]
                acc = dv / dt_v[:, None]
                acc_sq = np.sum(acc * acc, axis=1)
                acc_sq = acc_sq[np.isfinite(acc_sq)]
                if acc_sq.size:
                    accel_sq_all.append(acc_sq)

        points_used += int(n)
        traj_used += 1

    params = default_params if default_params is not None else KalmanParams()
    if meas_sq_all:
        meas_sq = np.concatenate(meas_sq_all)
        meas_var = float(max(np.mean(meas_sq) / 2.0, 1e-9))
        init_pos_var = float(max(np.mean(np.asarray(init_pos_sq_all)) / 2.0, 1e-9))
        init_vel_var = (
            float(max(np.mean(np.asarray(init_vel_sq_all)) / 2.0, 1e-9))
            if init_vel_sq_all
            else meas_var
        )
        process_var = (
            float(max(np.mean(np.concatenate(accel_sq_all)) / 2.0, 1e-9))
            if accel_sq_all
            else meas_var
        )
        params = KalmanParams(
            process_var=process_var,
            meas_var=meas_var,
            init_pos_var=init_pos_var,
            init_vel_var=init_vel_var,
        )

    summary = {
        "calibration_file": str(path),
        "n_trajectories_total": int(len(trajectories)),
        "n_trajectories_used": int(traj_used),
        "n_points_used": int(points_used),
        "params": asdict(params),
    }
    return params, summary


class KalmanRTS:
    """
    Instance-based Kalman-RTS baseline.

    Calibration (if available) runs once at initialization and is excluded from denoise timing.
    """

    def __init__(
        self,
        *,
        calibration_file: str | Path | None = None,
        dataset_name_hint: str | None = None,
        state_dir: str = "./dataset/state",
        fallback_dataset: str = "NUMOSIM_Kanto",
        params: KalmanParams | None = None,
        calibrate: bool = True,
        max_calibration_trajectories: int | None = None,
        allow_textbook_default: bool = False,
    ) -> None:
        self.dataset_name_hint = dataset_name_hint
        self.state_dir = str(state_dir)
        self.fallback_dataset = str(fallback_dataset)
        self.allow_textbook_default = bool(allow_textbook_default)
        textbook_default_mode = self.allow_textbook_default and calibration_file is None
        self.params = (
            params
            if params is not None
            else (KalmanParams() if textbook_default_mode else load_kalman_params_from_state(
                dataset_name_hint=dataset_name_hint,
                state_dir=state_dir,
                fallback_dataset=fallback_dataset,
            ))
        )
        self.calibration_file: str | None = None
        self.calibration_summary: dict = {
            "status": "not_requested",
            "mode": "artifact",
            "params": asdict(self.params),
        }

        if textbook_default_mode:
            self.calibration_summary = {
                "status": "ok",
                "mode": "textbook_default",
                "calibration_file": None,
                "params": asdict(self.params),
            }
            return

        resolved = (
            str(calibration_file)
            if calibration_file is not None
            else resolve_kalman_calibration_file_from_state(
                dataset_name_hint=dataset_name_hint,
                state_dir=state_dir,
                fallback_dataset=fallback_dataset,
            )
        )
        if not resolved:
            self.calibration_summary = {
                "status": "not_found",
                "mode": "artifact",
                "params": asdict(self.params),
            }
            return

        self.calibration_file = str(resolved)
        if not calibrate:
            self.calibration_summary = {
                "status": "skipped",
                "mode": "artifact",
                "calibration_file": self.calibration_file,
                "params": asdict(self.params),
            }
            return

        try:
            tuned, summary = estimate_kalman_params_from_calibration_file(
                self.calibration_file,
                default_params=self.params,
                max_trajectories=max_calibration_trajectories,
            )
            self.params = tuned
            self.calibration_summary = {"status": "ok", "mode": "artifact", **summary}
        except Exception as exc:
            logger.warning(
                "KalmanRTS calibration failed for %s: %s. Falling back to state/default params.",
                self.calibration_file,
                exc,
            )
            self.calibration_summary = {
                "status": "failed",
                "mode": "artifact",
                "calibration_file": self.calibration_file,
                "reason": f"{type(exc).__name__}: {exc}",
                "params": asdict(self.params),
            }

    def denoise_enu(
        self,
        positions_enu: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        *,
        use_timestamps: bool = True,
    ) -> np.ndarray:
        ts = timestamps if use_timestamps else None
        return kalman_rts_smoother(positions_enu, timestamps=ts, params=self.params)

    def denoise_gps(
        self,
        gps_positions: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        *,
        use_timestamps: bool = True,
        ref_lat: float | None = None,
        ref_lon: float | None = None,
    ) -> np.ndarray:
        gps = _as_float_array(gps_positions)
        if gps.ndim != 2 or gps.shape[1] != 2:
            raise ValueError("gps_positions must be shape (N, 2) in [lon, lat] order")
        if gps.shape[0] == 0:
            return gps.copy()
        if ref_lat is None:
            ref_lat = float(gps[0, 1])
        if ref_lon is None:
            ref_lon = float(gps[0, 0])

        e, n, _ = geodetic2enu(gps[:, 1], gps[:, 0], 0.0, ref_lat, ref_lon, 0.0)
        enu = np.stack([e, n], axis=1)
        denoised_enu = self.denoise_enu(enu, timestamps=timestamps, use_timestamps=use_timestamps)

        lat, lon, _ = enu2geodetic(
            denoised_enu[:, 0],
            denoised_enu[:, 1],
            0.0,
            ref_lat,
            ref_lon,
            0.0,
        )
        return np.stack([lon, lat], axis=1)

    def denoise(
        self,
        positions: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        *,
        coord_space: str = "GPS",
        use_timestamps: bool = True,
        ref_lat: float | None = None,
        ref_lon: float | None = None,
    ) -> np.ndarray:
        space = str(coord_space).strip().upper()
        if space == "ENU":
            return self.denoise_enu(positions, timestamps=timestamps, use_timestamps=use_timestamps)
        if space == "GPS":
            return self.denoise_gps(
                positions,
                timestamps=timestamps,
                use_timestamps=use_timestamps,
                ref_lat=ref_lat,
                ref_lon=ref_lon,
            )
        raise ValueError("coord_space must be 'GPS' or 'ENU'")


def hampel_filter(
    positions: np.ndarray,
    window_size: int = 7,
    n_sigma: float = 3.0,
) -> np.ndarray:
    """Hampel-like spike removal per dimension.

    Replaces outliers with local median within the window.
    """
    pos = _as_float_array(positions)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("positions must be shape (N, 2)")
    n = pos.shape[0]
    if n == 0:
        return pos.copy()

    k = max(1, int(window_size) // 2)
    out = pos.copy()
    for dim in range(2):
        series = pos[:, dim]
        for i in range(n):
            start = max(0, i - k)
            end = min(n, i + k + 1)
            window = series[start:end]
            window = window[np.isfinite(window)]
            if window.size == 0:
                continue
            med = np.median(window)
            mad = np.median(np.abs(window - med))
            if mad == 0:
                continue
            thresh = n_sigma * 1.4826 * mad
            if np.isfinite(series[i]) and abs(series[i] - med) > thresh:
                out[i, dim] = med
    return out


def savitzky_golay_filter(
    positions: np.ndarray,
    window_length: int = 11,
    polyorder: int = 2,
) -> np.ndarray:
    """Savitzky-Golay filter for 2D positions."""
    pos = _as_float_array(positions)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("positions must be shape (N, 2)")
    n = pos.shape[0]
    if n < 3:
        return pos.copy()

    w = int(window_length)
    if w % 2 == 0:
        w += 1
    w = min(w, n if n % 2 == 1 else n - 1)
    min_w = polyorder + 2
    if min_w % 2 == 0:
        min_w += 1
    w = max(w, min_w)
    if w > n:
        return pos.copy()

    out = np.zeros_like(pos)
    for dim in range(2):
        out[:, dim] = savgol_filter(pos[:, dim], window_length=w, polyorder=polyorder)
    return out


def smoothing_spline(
    positions: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    smoothing: float = 1.0,
    spline_order: int = 3,
) -> np.ndarray:
    """Smoothing spline on x(t), y(t)."""
    pos = _as_float_array(positions)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("positions must be shape (N, 2)")
    n = pos.shape[0]
    if n <= 2:
        return pos.copy()

    t_raw = _prepare_timestamps(timestamps, n)
    t_fit, pos_fit = _collapse_duplicate_t(t_raw, pos)
    if t_fit.size <= 2:
        return pos.copy()

    k = int(spline_order)
    k = min(max(k, 1), min(3, t_fit.size - 1))
    s = float(smoothing) * t_fit.size

    out = np.zeros_like(pos)
    for dim in range(2):
        spline = UnivariateSpline(t_fit, pos_fit[:, dim], s=s, k=k)
        out[:, dim] = spline(t_raw)
    return out


def raw_baseline(
    positions: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    **_kwargs,
) -> np.ndarray:
    """Disabled: return input trajectory unchanged."""
    pos = _as_float_array(positions)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("positions must be shape (N, 2)")
    return pos.copy()


def difftraj(*_args, **_kwargs):
    """Placeholder for difftraj baseline (to be implemented later)."""
    raise NotImplementedError("difftraj baseline will be implemented later")


def google_maps_baseline(
    positions,
    timestamps,
    api_key,
):
    """Snap-to-roads baseline using the Google Maps Roads API.

    positions: (N, 2) array of [longitude, latitude] (project convention)
    timestamps: ignored (kept for interface consistency)
    api_key: Google Maps API key with Roads API enabled
    """
    import requests

    pos = _as_float_array(positions)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("positions must be shape (N, 2)")
    n = pos.shape[0]
    if n == 0:
        return pos.copy()

    out = pos.copy()
    BATCH_SIZE = 100
    URL = "https://roads.googleapis.com/v1/snapToRoads"

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch = pos[start:end]

        # API expects lat,lng — project stores [lon, lat]
        path_str = "|".join(f"{lat},{lon}" for lon, lat in batch)
        params = {
            "path": path_str,
            "interpolate": "false",
            "key": api_key,
        }

        resp = requests.get(URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for pt in data.get("snappedPoints", []):
            orig_idx = pt.get("originalIndex")
            if orig_idx is not None:
                loc = pt["location"]
                # Convert back to [lon, lat] project convention
                out[start + orig_idx] = [loc["longitude"], loc["latitude"]]

    return out


def _generate_synthetic_traj(n: int = 400, seed: int = 7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 20.0, n)
    x = np.cos(t) * 100.0 + t * 2.0
    y = np.sin(t) * 100.0 + t * 1.5
    clean = np.stack([x, y], axis=1)
    noise = rng.normal(scale=5.0, size=clean.shape)
    noisy = clean + noise
    # Add spikes
    spike_idx = rng.choice(n, size=max(3, n // 40), replace=False)
    noisy[spike_idx] += rng.normal(scale=60.0, size=(spike_idx.size, 2))
    return t, clean, noisy


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run_smoke_tests(plot: bool = False) -> None:
    """Simple synthetic test harness for baselines."""
    t, clean, noisy = _generate_synthetic_traj()

    kf = kalman_rts_smoother(noisy, timestamps=t)
    hp = hampel_filter(noisy, window_size=9, n_sigma=3.0)
    sg = savitzky_golay_filter(noisy, window_length=11, polyorder=2)
    sp = smoothing_spline(noisy, timestamps=t, smoothing=1.0)
    rw = raw_baseline(noisy, timestamps=t)

    print("RMSE (noisy):", _rmse(noisy, clean))
    print("RMSE (kalman+rts):", _rmse(kf, clean))
    print("RMSE (hampel):", _rmse(hp, clean))
    print("RMSE (savgol):", _rmse(sg, clean))
    print("RMSE (spline):", _rmse(sp, clean))
    print("RMSE (raw):", _rmse(rw, clean))

    if plot:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover - optional
            print(f"Plotting unavailable: {exc}")
            return
        plt.figure(figsize=(8, 6))
        plt.plot(clean[:, 0], clean[:, 1], label="clean", linewidth=2)
        plt.scatter(noisy[:, 0], noisy[:, 1], s=8, label="noisy", alpha=0.5)
        plt.plot(kf[:, 0], kf[:, 1], label="kalman+rts")
        plt.plot(hp[:, 0], hp[:, 1], label="hampel")
        plt.plot(sg[:, 0], sg[:, 1], label="savgol")
        plt.plot(sp[:, 0], sp[:, 1], label="spline")
        plt.plot(rw[:, 0], rw[:, 1], label="raw")
        plt.legend()
        plt.title("Baseline Denoisers (Synthetic)")
        plt.tight_layout()
        plt.show()


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Baseline denoising methods")
    parser.add_argument("--demo", action="store_true", help="run synthetic smoke test")
    parser.add_argument("--plot", action="store_true", help="plot demo results")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if args.demo:
        run_smoke_tests(plot=args.plot)
    else:
        print("Nothing to do. Try: python -m src.baseline.classic --demo")
