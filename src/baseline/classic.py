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

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter


@dataclass
class KalmanParams:
    process_var: float = 1.0
    meas_var: float = 5.0
    init_pos_var: float = 10.0
    init_vel_var: float = 10.0


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
    params: KalmanParams = KalmanParams(),
) -> np.ndarray:
    """Kalman filter with RTS smoothing for 2D positions.

    positions: (N, 2) array of [x, y]
    timestamps: optional (N,) array, used for variable dt
    """
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
