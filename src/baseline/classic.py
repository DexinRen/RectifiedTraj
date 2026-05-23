#!/usr/bin/env python3
"""
Baseline denoising methods for 2D trajectories.

Implements:
  1) Kalman filter + RTS smoother
  2) Hampel-like spike removal
  3) Savitzky-Golay filter
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
from pymap3d import enu2geodetic, geodetic2enu
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)
CALIBRATION_INDEX_FILENAME = "calib.json"


@dataclass
class KalmanParams:
    process_var: float = 1.0
    meas_var: float = 5.0
    init_pos_var: float = 10.0
    init_vel_var: float = 10.0


@dataclass
class AlphaBetaParams:
    alpha: float = 0.85
    beta: float = 0.02


@dataclass
class CausalHampelParams:
    window_size: int = 9
    n_sigma: float = 3.0


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


def _extract_named_params_dict(
    entry: Any,
    *,
    field_name: str,
    required_keys: tuple[str, ...],
) -> dict[str, float] | None:
    if not isinstance(entry, dict):
        return None
    raw = entry.get(field_name)
    if not isinstance(raw, dict):
        return None
    vals: dict[str, float] = {}
    for key in required_keys:
        if key not in raw:
            return None
        vals[key] = float(raw[key])
    return vals


def _calibration_index_path(state_dir: str) -> Path:
    return Path(state_dir) / CALIBRATION_INDEX_FILENAME


def _load_calibration_index(state_dir: str) -> dict[str, Any]:
    path = _calibration_index_path(state_dir)
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _extract_kalman_params_from_calibration_index_entry(entry: Any) -> KalmanParams | None:
    if not isinstance(entry, dict):
        return None
    raw = entry.get("kalman_rts_params")
    if not isinstance(raw, dict):
        raw = entry.get("params")
    if not isinstance(raw, dict):
        return None
    keys = ("process_var", "meas_var", "init_pos_var", "init_vel_var")
    vals: dict[str, float] = {}
    for key in keys:
        if key not in raw:
            return None
        vals[key] = float(raw[key])
    return KalmanParams(**vals)


def _extract_alpha_beta_params_from_calibration_index_entry(entry: Any) -> AlphaBetaParams | None:
    vals = _extract_named_params_dict(
        entry,
        field_name="alpha_beta_params",
        required_keys=("alpha", "beta"),
    )
    if vals is None:
        return None
    return AlphaBetaParams(alpha=float(vals["alpha"]), beta=float(vals["beta"]))


def _extract_causal_hampel_params_from_calibration_index_entry(
    entry: Any,
) -> CausalHampelParams | None:
    vals = _extract_named_params_dict(
        entry,
        field_name="causal_hampel_params",
        required_keys=("window_size", "n_sigma"),
    )
    if vals is None:
        return None
    return CausalHampelParams(
        window_size=max(1, int(round(vals["window_size"]))),
        n_sigma=float(vals["n_sigma"]),
    )


def _calibration_key_candidates(
    dataset_name_hint: str | None,
    fallback_dataset: str | None,
) -> list[str]:
    out: list[str] = []

    def _push(raw: str | None) -> None:
        key = _safe_dataset_token(raw)
        if key and key not in out:
            out.append(key)

    _push(dataset_name_hint)
    hint = _safe_dataset_token(dataset_name_hint)
    if hint:
        # Benchmark trajectory datasets are named like:
        #   PoL_5s_traj_10s_200_5000
        # Prefer that exact key, then same-dataset native sample-time params,
        # then the base dataset key for legacy calibration indexes.
        marker = "_traj_"
        if marker in hint:
            base, suffix = hint.split(marker, 1)
            parts = suffix.split("_")
            if len(parts) > 1:
                _push(f"{base}{marker}native_{'_'.join(parts[1:])}")
            _push(base)
    _push(fallback_dataset)
    return out


def resolve_kalman_params_entry_from_state(
    dataset_name_hint: str | None = None,
    state_dir: str = "./dataset/state",
    fallback_dataset: str = "NUMOSIM_Kanto",
) -> dict[str, Any] | None:
    calib_index = _load_calibration_index(state_dir)
    for key in _calibration_key_candidates(dataset_name_hint, fallback_dataset):
        entry = calib_index.get(key)
        if isinstance(entry, dict) and _extract_kalman_params_from_calibration_index_entry(entry) is not None:
            return dict(entry)
    return None


def _resolve_named_calibration_entry_from_state(
    dataset_name_hint: str | None,
    *,
    state_dir: str,
    fallback_dataset: str,
    extractor,
) -> dict[str, Any] | None:
    calib_index = _load_calibration_index(state_dir)
    for key in _calibration_key_candidates(dataset_name_hint, fallback_dataset):
        entry = calib_index.get(key)
        if isinstance(entry, dict) and extractor(entry) is not None:
            return dict(entry)
    return None


def resolve_alpha_beta_params_entry_from_state(
    dataset_name_hint: str | None = None,
    state_dir: str = "./dataset/state",
    fallback_dataset: str = "NUMOSIM_Kanto",
) -> dict[str, Any] | None:
    return _resolve_named_calibration_entry_from_state(
        dataset_name_hint,
        state_dir=state_dir,
        fallback_dataset=fallback_dataset,
        extractor=_extract_alpha_beta_params_from_calibration_index_entry,
    )


def resolve_causal_hampel_params_entry_from_state(
    dataset_name_hint: str | None = None,
    state_dir: str = "./dataset/state",
    fallback_dataset: str = "NUMOSIM_Kanto",
) -> dict[str, Any] | None:
    return _resolve_named_calibration_entry_from_state(
        dataset_name_hint,
        state_dir=state_dir,
        fallback_dataset=fallback_dataset,
        extractor=_extract_causal_hampel_params_from_calibration_index_entry,
    )


def resolve_kalman_calibration_file_from_state(
    dataset_name_hint: str | None = None,
    state_dir: str = "./dataset/state",
    fallback_dataset: str = "NUMOSIM_Kanto",
) -> str | None:
    """
    Resolve a calibration artifact path from dataset/state/calib.json only.

    The function name is retained for compatibility with older call sites; it
    no longer scans state_<dataset>.json fallback metadata.
    """
    calib_index = _load_calibration_index(state_dir)
    for key in _calibration_key_candidates(dataset_name_hint, fallback_dataset):
        entry = calib_index.get(key)
        if not isinstance(entry, dict):
            continue
        raw_path = entry.get("calibration_file")
        if not raw_path:
            continue
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if path.exists():
            return str(path)
    return None


def load_kalman_params_from_state(
    dataset_name_hint: str | None = None,
    state_dir: str = "./dataset/state",
    fallback_dataset: str = "NUMOSIM_Kanto",
) -> KalmanParams:
    """
    Load Kalman params from shared calibration index, with fallback to defaults.
    """
    default = KalmanParams()
    calib_entry = resolve_kalman_params_entry_from_state(
        dataset_name_hint=dataset_name_hint,
        state_dir=state_dir,
        fallback_dataset=fallback_dataset,
    )
    params = _extract_kalman_params_from_calibration_index_entry(calib_entry)
    if params is not None:
        return params

    return default


def load_alpha_beta_params_from_state(
    dataset_name_hint: str | None = None,
    state_dir: str = "./dataset/state",
    fallback_dataset: str = "NUMOSIM_Kanto",
) -> AlphaBetaParams:
    default = AlphaBetaParams()
    entry = resolve_alpha_beta_params_entry_from_state(
        dataset_name_hint=dataset_name_hint,
        state_dir=state_dir,
        fallback_dataset=fallback_dataset,
    )
    params = _extract_alpha_beta_params_from_calibration_index_entry(entry)
    return params if params is not None else default


def load_causal_hampel_params_from_state(
    dataset_name_hint: str | None = None,
    state_dir: str = "./dataset/state",
    fallback_dataset: str = "NUMOSIM_Kanto",
) -> CausalHampelParams:
    default = CausalHampelParams()
    entry = resolve_causal_hampel_params_entry_from_state(
        dataset_name_hint=dataset_name_hint,
        state_dir=state_dir,
        fallback_dataset=fallback_dataset,
    )
    params = _extract_causal_hampel_params_from_calibration_index_entry(entry)
    return params if params is not None else default


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


def _timestamps_from_row_time_fields(row: dict, t_len: int, valid: np.ndarray, n: int) -> Optional[np.ndarray]:
    if "timestamp" in row:
        ts = _to_numpy_array(row.get("timestamp")).reshape(-1)
        if ts.size >= t_len:
            ts = ts[:t_len][valid]
            if ts.size == n:
                return ts.astype(np.float64, copy=False)
    if "dt_sec" in row:
        dt = _to_numpy_array(row.get("dt_sec")).reshape(-1)
        if dt.size >= t_len:
            dt = dt[:t_len][valid].astype(np.float64, copy=False)
            if dt.size == n:
                ts = np.cumsum(dt, dtype=np.float64)
                return ts - ts[0] if ts.size else ts
    return None


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


def kalman_filter(
    positions: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    params: Optional[KalmanParams] = None,
) -> np.ndarray:
    """Forward-only Kalman filter for 2D positions."""
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

        x_pred = F @ x[k - 1]
        P_pred = F @ P[k - 1] @ F.T + Q

        z = pos[k]
        if np.any(~np.isfinite(z)):
            x[k] = x_pred
            P[k] = P_pred
            continue

        y = z - (H @ x_pred)
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x[k] = x_pred + K @ y
        P[k] = (np.eye(4) - K @ H) @ P_pred

    return x[:, 0:2]


def _to_numpy_array(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return np.asarray(x.detach().cpu().numpy())
    return np.asarray(x)


def _trajectory_rows_to_enu_sequences(
    trajectories: list[dict],
    *,
    max_trajectories: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not isinstance(trajectories, list) or not trajectories:
        raise ValueError("Calibration source has no trajectories.")
    rows = (
        trajectories[: int(max_trajectories)]
        if max_trajectories is not None and max_trajectories > 0
        else trajectories
    )

    sequences: list[dict[str, Any]] = []
    points_used = 0
    traj_used = 0

    for row in rows:
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

        timestamps = _timestamps_from_row_time_fields(row, t_len, valid, n)

        ref_lon = float(clean_gps[0, 0])
        ref_lat = float(clean_gps[0, 1])
        noisy_e, noisy_n, _ = geodetic2enu(
            noisy_gps[:, 1],
            noisy_gps[:, 0],
            0.0,
            ref_lat,
            ref_lon,
            0.0,
        )
        clean_e, clean_n, _ = geodetic2enu(
            clean_gps[:, 1],
            clean_gps[:, 0],
            0.0,
            ref_lat,
            ref_lon,
            0.0,
        )
        sequences.append(
            {
                "noisy_enu": np.stack([noisy_e, noisy_n], axis=1),
                "clean_enu": np.stack([clean_e, clean_n], axis=1),
                "timestamps": timestamps,
            }
        )
        points_used += int(n)
        traj_used += 1

    summary = {
        "n_trajectories_total": int(len(rows)),
        "n_trajectories_used": int(traj_used),
        "n_points_used": int(points_used),
    }
    return sequences, summary


def _median_positive_dt_from_sequences(sequences: list[dict[str, Any]]) -> float:
    dt_values: list[np.ndarray] = []
    for seq in sequences:
        timestamps = seq.get("timestamps")
        noisy = seq["noisy_enu"]
        t = _prepare_timestamps(timestamps, int(noisy.shape[0]))
        dt = np.diff(t)
        positive = dt[dt > 0]
        if positive.size:
            dt_values.append(positive.astype(np.float64, copy=False))
    if not dt_values:
        return 1.0
    return float(np.median(np.concatenate(dt_values)))


def _steady_state_alpha_beta_from_kalman_params(
    params: KalmanParams,
    *,
    dt: float,
    max_iter: int = 256,
) -> AlphaBetaParams:
    dtk = max(float(dt), 1e-6)
    p = np.diag([float(params.init_pos_var), float(params.init_vel_var)])
    f = np.array([[1.0, dtk], [0.0, 1.0]], dtype=float)
    q = float(params.process_var)
    q_mat = np.array(
        [[(dtk**4) * q / 4.0, (dtk**3) * q / 2.0], [(dtk**3) * q / 2.0, (dtk**2) * q]],
        dtype=float,
    )
    h = np.array([[1.0, 0.0]], dtype=float)
    r = np.array([[max(float(params.meas_var), 1e-9)]], dtype=float)
    k = np.zeros((2, 1), dtype=float)
    eye = np.eye(2, dtype=float)

    for _ in range(max_iter):
        p_pred = f @ p @ f.T + q_mat
        s = h @ p_pred @ h.T + r
        k = p_pred @ h.T @ np.linalg.inv(s)
        p = (eye - k @ h) @ p_pred

    alpha = float(np.clip(k[0, 0], 1e-4, 0.9999))
    beta = float(np.clip(dtk * k[1, 0], 1e-5, 1.999))
    return AlphaBetaParams(alpha=alpha, beta=beta)


def alpha_beta_filter(
    positions: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    params: Optional[AlphaBetaParams] = None,
) -> np.ndarray:
    """Forward-only alpha-beta tracker for 2D positions."""
    cfg = params if params is not None else AlphaBetaParams()
    pos = _as_float_array(positions)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("positions must be shape (N, 2)")
    n = pos.shape[0]
    if n == 0:
        return pos.copy()

    t = _prepare_timestamps(timestamps, n)
    dt = np.diff(t, prepend=t[0])
    dt[0] = dt[1] if n > 1 else 1.0

    alpha = float(np.clip(cfg.alpha, 1e-6, 0.999999))
    beta = float(max(cfg.beta, 0.0))
    out = np.zeros_like(pos)
    out[0] = pos[0]
    vel = np.zeros(2, dtype=float)

    if n > 1 and np.isfinite(pos[0]).all() and np.isfinite(pos[1]).all():
        dtk = max(float(dt[1]), 1e-6)
        vel = (pos[1] - pos[0]) / dtk

    for k in range(1, n):
        dtk = max(float(dt[k]), 1e-6)
        pred = out[k - 1] + vel * dtk
        z = pos[k]
        if np.any(~np.isfinite(z)):
            out[k] = pred
            continue
        resid = z - pred
        out[k] = pred + alpha * resid
        vel = vel + (beta / dtk) * resid
    return out


def _evaluate_alpha_beta_candidates(
    sequences: list[dict[str, Any]],
    candidates: np.ndarray,
) -> np.ndarray:
    alphas = np.asarray(candidates[:, 0], dtype=np.float64)
    betas = np.asarray(candidates[:, 1], dtype=np.float64)
    total_sq = np.zeros(alphas.shape[0], dtype=np.float64)
    total_pts = 0

    for seq in sequences:
        noisy = np.asarray(seq["noisy_enu"], dtype=np.float64)
        clean = np.asarray(seq["clean_enu"], dtype=np.float64)
        n = int(min(noisy.shape[0], clean.shape[0]))
        if n < 2:
            continue
        noisy = noisy[:n]
        clean = clean[:n]
        t = _prepare_timestamps(seq.get("timestamps"), n)
        dt = np.diff(t, prepend=t[0])
        dt[0] = dt[1] if n > 1 else 1.0

        pos = np.repeat(noisy[0][None, :], alphas.shape[0], axis=0)
        vel = np.zeros((alphas.shape[0], 2), dtype=np.float64)
        if n > 1 and np.isfinite(noisy[0]).all() and np.isfinite(noisy[1]).all():
            dtk1 = max(float(dt[1]), 1e-6)
            vel[:] = (noisy[1] - noisy[0])[None, :] / dtk1

        for idx in range(1, n):
            dtk = max(float(dt[idx]), 1e-6)
            pred = pos + vel * dtk
            z = noisy[idx]
            if np.any(~np.isfinite(z)):
                pos = pred
            else:
                resid = z[None, :] - pred
                pos = pred + alphas[:, None] * resid
                vel = vel + (betas[:, None] / dtk) * resid
            bad = (
                ~np.isfinite(pos).all(axis=1)
                | ~np.isfinite(vel).all(axis=1)
                | (np.max(np.abs(pos), axis=1) > 1e8)
                | (np.max(np.abs(vel), axis=1) > 1e8)
            )
            if np.any(bad):
                total_sq[bad] = np.inf
                pos[bad] = 0.0
                vel[bad] = 0.0
            diff = pos - clean[idx][None, :]
            total_sq += np.sum(diff * diff, axis=1)
        total_pts += max(0, n - 1)

    if total_pts <= 0:
        raise ValueError("No usable points for alpha-beta calibration.")
    return total_sq / float(total_pts)


def _estimate_alpha_beta_params_from_trajectory_rows(
    trajectories: list[dict],
    *,
    default_params: AlphaBetaParams | None = None,
    seed_kalman_params: KalmanParams | None = None,
    max_trajectories: int | None = None,
    source_label: str | None = None,
) -> tuple[AlphaBetaParams, dict]:
    sequences, counts = _trajectory_rows_to_enu_sequences(
        trajectories,
        max_trajectories=max_trajectories,
    )
    if not sequences:
        raise ValueError("Calibration source has no usable trajectories.")

    dt_med = _median_positive_dt_from_sequences(sequences)
    if seed_kalman_params is not None:
        seed = _steady_state_alpha_beta_from_kalman_params(seed_kalman_params, dt=dt_med)
    else:
        seed = default_params if default_params is not None else AlphaBetaParams()

    alpha_grid = np.clip(
        np.linspace(max(0.05, seed.alpha * 0.75), min(0.99, seed.alpha * 1.25), 5),
        1e-4,
        0.9999,
    )
    beta_grid = np.clip(
        np.linspace(max(1e-4, seed.beta * 0.6), min(1.25, max(seed.beta * 1.4, 0.04)), 5),
        1e-5,
        1.999,
    )
    candidates = np.asarray(
        [[a, b] for a in alpha_grid for b in beta_grid],
        dtype=np.float64,
    )
    mse = _evaluate_alpha_beta_candidates(sequences, candidates)
    best_idx = int(np.argmin(mse))
    best = AlphaBetaParams(
        alpha=float(candidates[best_idx, 0]),
        beta=float(candidates[best_idx, 1]),
    )
    summary = {
        "calibration_source": str(source_label or "in_memory_rows"),
        **counts,
        "median_dt": float(dt_med),
        "params": asdict(best),
        "seed_params": asdict(seed),
        "rmse_enu": float(np.sqrt(mse[best_idx])),
    }
    return best, summary


def estimate_alpha_beta_params_from_calibration_file(
    calibration_file: str | Path,
    *,
    default_params: AlphaBetaParams | None = None,
    seed_kalman_params: KalmanParams | None = None,
    max_trajectories: int | None = None,
) -> tuple[AlphaBetaParams, dict]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("Torch is required for alpha-beta calibration file loading.") from exc

    path = Path(str(calibration_file))
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    blob = torch.load(path, map_location="cpu")
    trajectories = blob.get("trajectories", []) if isinstance(blob, dict) else []
    if not isinstance(trajectories, list) or not trajectories:
        raise ValueError(f"Calibration file has no trajectories: {path}")

    params, summary = _estimate_alpha_beta_params_from_trajectory_rows(
        trajectories,
        default_params=default_params,
        seed_kalman_params=seed_kalman_params,
        max_trajectories=max_trajectories,
        source_label=str(path),
    )
    summary["calibration_file"] = str(path)
    return params, summary


def _causal_hampel_stats(series: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    window = max(1, int(window_size))
    if window <= 1:
        median = np.asarray(series, dtype=np.float64).copy()
        mad = np.zeros_like(median)
        return median, mad
    padded = np.pad(np.asarray(series, dtype=np.float64), (window - 1, 0), mode="constant", constant_values=np.nan)
    views = np.lib.stride_tricks.sliding_window_view(padded, window_shape=window)
    median = np.nanmedian(views, axis=1)
    mad = np.nanmedian(np.abs(views - median[:, None]), axis=1)
    return median, mad


def causal_hampel_filter(
    positions: np.ndarray,
    window_size: int = 9,
    n_sigma: float = 3.0,
) -> np.ndarray:
    """Past-only Hampel filter using a trailing window."""
    pos = _as_float_array(positions)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("positions must be shape (N, 2)")
    n = pos.shape[0]
    if n == 0:
        return pos.copy()

    out = pos.copy()
    scale = float(max(n_sigma, 0.0)) * 1.4826
    window = max(1, int(window_size))
    for dim in range(2):
        series = pos[:, dim]
        median, mad = _causal_hampel_stats(series, window)
        thresh = scale * mad
        mask = np.isfinite(series) & (mad > 0.0) & (np.abs(series - median) > thresh)
        out[mask, dim] = median[mask]
    return out


def _evaluate_causal_hampel_candidates(
    sequences: list[dict[str, Any]],
    window_candidates: list[int],
    sigma_candidates: np.ndarray,
) -> tuple[CausalHampelParams, float]:
    best_params = CausalHampelParams()
    best_mse = float("inf")

    for window in window_candidates:
        total_sq = np.zeros(sigma_candidates.shape[0], dtype=np.float64)
        total_pts = 0
        for seq in sequences:
            noisy = np.asarray(seq["noisy_enu"], dtype=np.float64)
            clean = np.asarray(seq["clean_enu"], dtype=np.float64)
            n = int(min(noisy.shape[0], clean.shape[0]))
            if n == 0:
                continue
            dim_out = np.repeat(noisy[:n][None, :, :], sigma_candidates.shape[0], axis=0)
            for dim in range(2):
                series = noisy[:n, dim]
                median, mad = _causal_hampel_stats(series, window)
                for idx_sigma, n_sigma in enumerate(sigma_candidates):
                    thresh = float(n_sigma) * 1.4826 * mad
                    filtered = np.where(
                        np.isfinite(series) & (mad > 0.0) & (np.abs(series - median) > thresh),
                        median,
                        series,
                    )
                    dim_out[idx_sigma, :, dim] = filtered
            diff = dim_out - clean[:n][None, :, :]
            total_sq += np.sum(diff * diff, axis=(1, 2))
            total_pts += n

        if total_pts <= 0:
            continue
        mse = total_sq / float(total_pts)
        idx = int(np.argmin(mse))
        if float(mse[idx]) < best_mse:
            best_mse = float(mse[idx])
            best_params = CausalHampelParams(
                window_size=int(window),
                n_sigma=float(sigma_candidates[idx]),
            )

    if not np.isfinite(best_mse):
        raise ValueError("No usable points for causal Hampel calibration.")
    return best_params, best_mse


def _estimate_causal_hampel_params_from_trajectory_rows(
    trajectories: list[dict],
    *,
    default_params: CausalHampelParams | None = None,
    max_trajectories: int | None = None,
    source_label: str | None = None,
) -> tuple[CausalHampelParams, dict]:
    sequences, counts = _trajectory_rows_to_enu_sequences(
        trajectories,
        max_trajectories=max_trajectories,
    )
    if not sequences:
        raise ValueError("Calibration source has no usable trajectories.")

    seed = default_params if default_params is not None else CausalHampelParams()
    window_candidates = sorted(
        {
            max(3, int(seed.window_size) - 4),
            max(3, int(seed.window_size) - 2),
            max(3, int(seed.window_size)),
            max(3, int(seed.window_size) + 2),
            max(3, int(seed.window_size) + 4),
            5,
            7,
            9,
            11,
            15,
        }
    )
    sigma_candidates = np.asarray([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0], dtype=np.float64)
    best, best_mse = _evaluate_causal_hampel_candidates(
        sequences,
        window_candidates=window_candidates,
        sigma_candidates=sigma_candidates,
    )
    summary = {
        "calibration_source": str(source_label or "in_memory_rows"),
        **counts,
        "params": asdict(best),
        "rmse_enu": float(np.sqrt(best_mse)),
    }
    return best, summary


def estimate_causal_hampel_params_from_calibration_file(
    calibration_file: str | Path,
    *,
    default_params: CausalHampelParams | None = None,
    max_trajectories: int | None = None,
) -> tuple[CausalHampelParams, dict]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("Torch is required for causal Hampel calibration file loading.") from exc

    path = Path(str(calibration_file))
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    blob = torch.load(path, map_location="cpu")
    trajectories = blob.get("trajectories", []) if isinstance(blob, dict) else []
    if not isinstance(trajectories, list) or not trajectories:
        raise ValueError(f"Calibration file has no trajectories: {path}")

    params, summary = _estimate_causal_hampel_params_from_trajectory_rows(
        trajectories,
        default_params=default_params,
        max_trajectories=max_trajectories,
        source_label=str(path),
    )
    summary["calibration_file"] = str(path)
    return params, summary


def _estimate_kalman_params_from_trajectory_rows(
    trajectories: list[dict],
    *,
    default_params: KalmanParams | None = None,
    max_trajectories: int | None = None,
    source_label: str | None = None,
) -> tuple[KalmanParams, dict]:
    if not isinstance(trajectories, list) or not trajectories:
        raise ValueError("Calibration source has no trajectories.")
    rows = trajectories[: int(max_trajectories)] if max_trajectories is not None and max_trajectories > 0 else trajectories

    meas_sq_all: list[np.ndarray] = []
    init_pos_sq_all: list[float] = []
    init_vel_sq_all: list[float] = []
    accel_sq_all: list[np.ndarray] = []
    points_used = 0
    traj_used = 0

    for row in rows:
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

        timestamps = _timestamps_from_row_time_fields(row, t_len, valid, n)

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
        "calibration_source": str(source_label or "in_memory_rows"),
        "n_trajectories_total": int(len(rows)),
        "n_trajectories_used": int(traj_used),
        "n_points_used": int(points_used),
        "params": asdict(params),
    }
    return params, summary


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

    params, summary = _estimate_kalman_params_from_trajectory_rows(
        trajectories,
        default_params=default_params,
        max_trajectories=max_trajectories,
        source_label=str(path),
    )
    summary["calibration_file"] = str(path)
    return params, summary


def estimate_kalman_params_from_rows(
    trajectories: list[dict],
    *,
    default_params: KalmanParams | None = None,
    max_trajectories: int | None = None,
    source_label: str | None = None,
) -> tuple[KalmanParams, dict]:
    return _estimate_kalman_params_from_trajectory_rows(
        trajectories,
        default_params=default_params,
        max_trajectories=max_trajectories,
        source_label=source_label,
    )


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
    ) -> None:
        self.dataset_name_hint = dataset_name_hint
        self.state_dir = str(state_dir)
        self.fallback_dataset = str(fallback_dataset)
        self.params = (
            params
            if params is not None
            else load_kalman_params_from_state(
                dataset_name_hint=dataset_name_hint,
                state_dir=state_dir,
                fallback_dataset=fallback_dataset,
            )
        )
        self.calibration_file: str | None = None
        self.calibration_summary: dict = {
            "status": "not_requested",
            "mode": "artifact",
            "params": asdict(self.params),
        }
        calib_entry = resolve_kalman_params_entry_from_state(
            dataset_name_hint=dataset_name_hint,
            state_dir=state_dir,
            fallback_dataset=fallback_dataset,
        )

        resolved = (
            str(calibration_file)
            if calibration_file is not None
            else (
                None
                if calib_entry is not None
                else resolve_kalman_calibration_file_from_state(
                    dataset_name_hint=dataset_name_hint,
                    state_dir=state_dir,
                    fallback_dataset=fallback_dataset,
                )
            )
        )
        if not resolved:
            if calib_entry is not None:
                self.calibration_summary = {
                    "status": "ok",
                    "mode": "calib_json",
                    "params": asdict(self.params),
                    "calibration_index_entry": calib_entry,
                }
                return
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


class KalmanFilter(KalmanRTS):
    """Instance-based forward-only Kalman baseline."""

    def denoise_enu(
        self,
        positions_enu: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        *,
        use_timestamps: bool = True,
    ) -> np.ndarray:
        ts = timestamps if use_timestamps else None
        return kalman_filter(positions_enu, timestamps=ts, params=self.params)


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
    rw = raw_baseline(noisy, timestamps=t)

    print("RMSE (noisy):", _rmse(noisy, clean))
    print("RMSE (kalman+rts):", _rmse(kf, clean))
    print("RMSE (hampel):", _rmse(hp, clean))
    print("RMSE (savgol):", _rmse(sg, clean))
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
