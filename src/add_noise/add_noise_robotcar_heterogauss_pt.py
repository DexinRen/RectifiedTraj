#!/usr/bin/env python3
"""
Add heteroscedastic Gaussian noise (in ENU meters) to RobotCar trajectory PT files.

Input schema:
- top-level dict with "trajectories" (list[dict])
- each trajectory contains:
  - "label" (preferred clean reference) in [lon, lat]
  - or fallback to "data" if "label" is absent

Output schema:
- preserves trajectory fields
- writes noisy coordinates to trajectory["data"] in [lon, lat]
- stores clean copy in trajectory["data_clean"]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pymap3d import enu2geodetic, geodetic2enu


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add heteroscedastic Gaussian noise (meters, ENU) to RobotCar PT trajectories."
    )
    parser.add_argument(
        "--input-pt",
        type=str,
        required=True,
        help="Path to extracted RobotCar .pt file (clean/reference labels expected).",
    )
    parser.add_argument(
        "--output-pt",
        type=str,
        default=None,
        help="Output .pt path. Default: <input_dir>/RobotCar_HeteroGauss.pt",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11,
        help="Random seed.",
    )
    parser.add_argument(
        "--target-mean-m",
        type=float,
        default=15.0,
        help="Target overall mean error (meters).",
    )
    parser.add_argument(
        "--target-median-m",
        type=float,
        default=15.0,
        help="Target overall median error (meters).",
    )
    parser.add_argument(
        "--target-std-m",
        type=float,
        default=9.0,
        help="Target overall std error (meters).",
    )
    parser.add_argument(
        "--scale-mode",
        type=str,
        default="median",
        choices=["mean", "median", "weighted"],
        help="How to compute global scale from base-noise stats.",
    )
    parser.add_argument(
        "--weight-mean",
        type=float,
        default=1.0,
        help="Weighted mode: mean weight.",
    )
    parser.add_argument(
        "--weight-median",
        type=float,
        default=1.0,
        help="Weighted mode: median weight.",
    )
    parser.add_argument(
        "--weight-std",
        type=float,
        default=1.0,
        help="Weighted mode: std weight.",
    )
    parser.add_argument(
        "--sigma-cv",
        type=float,
        default=0.15,
        help="Coefficient of variation for per-point sigma_t (heteroscedastic strength).",
    )
    parser.add_argument(
        "--ar-rho",
        type=float,
        default=0.96,
        help="AR(1) correlation for log-sigma series in each trajectory.",
    )
    parser.add_argument(
        "--sigma-factor-min",
        type=float,
        default=0.35,
        help="Lower clamp for sigma_t multiplicative factor.",
    )
    parser.add_argument(
        "--sigma-factor-max",
        type=float,
        default=3.00,
        help="Upper clamp for sigma_t multiplicative factor.",
    )
    return parser


def _to_np(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _make_sigma_factors(
    n: int,
    *,
    sigma_cv: float,
    ar_rho: float,
    clamp_min: float,
    clamp_max: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.float64)

    cv = max(0.0, float(sigma_cv))
    rho = float(np.clip(ar_rho, -0.999, 0.999))
    cmin = max(1e-6, float(clamp_min))
    cmax = max(cmin, float(clamp_max))

    if cv <= 0.0:
        return np.ones((n,), dtype=np.float64)

    sigma_ln = float(np.sqrt(np.log(1.0 + cv * cv)))
    # median(exp(sigma_ln * z)) = 1
    x = np.zeros((n,), dtype=np.float64)
    x[0] = rng.normal(0.0, 1.0)
    white_scale = np.sqrt(max(1e-12, 1.0 - rho * rho))
    for i in range(1, n):
        x[i] = rho * x[i - 1] + white_scale * rng.normal(0.0, 1.0)
    factors = np.exp(sigma_ln * x)
    factors = np.clip(factors, cmin, cmax)
    return factors


def _stats(x: np.ndarray) -> dict[str, float]:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "p95": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


def _resolve_scale(
    base_stats: dict[str, float],
    *,
    target_mean_m: float,
    target_median_m: float,
    target_std_m: float,
    mode: str,
    w_mean: float,
    w_median: float,
    w_std: float,
) -> float:
    bm = max(1e-12, float(base_stats["mean"]))
    bmed = max(1e-12, float(base_stats["median"]))
    bstd = max(1e-12, float(base_stats["std"]))
    tm = max(1e-12, float(target_mean_m))
    tmed = max(1e-12, float(target_median_m))
    ts = max(1e-12, float(target_std_m))

    if mode == "mean":
        return float(tm / bm)
    if mode == "median":
        return float(tmed / bmed)

    wm = max(0.0, float(w_mean))
    wmed = max(0.0, float(w_median))
    ws = max(0.0, float(w_std))
    denom = wm * bm * bm + wmed * bmed * bmed + ws * bstd * bstd
    if denom <= 0.0:
        return float(tm / bm)
    numer = wm * bm * tm + wmed * bmed * tmed + ws * bstd * ts
    return float(numer / denom)


def add_heteroscedastic_gaussian_noise(
    input_pt: str,
    *,
    output_pt: str | None = None,
    seed: int = 11,
    target_mean_m: float = 15.0,
    target_median_m: float = 15.0,
    target_std_m: float = 9.0,
    scale_mode: str = "median",
    weight_mean: float = 1.0,
    weight_median: float = 1.0,
    weight_std: float = 1.0,
    sigma_cv: float = 0.15,
    ar_rho: float = 0.96,
    sigma_factor_min: float = 0.35,
    sigma_factor_max: float = 3.0,
) -> dict[str, Any]:
    in_path = Path(input_pt).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input PT not found: {in_path}")

    if output_pt:
        out_path = Path(output_pt).resolve()
    else:
        out_path = in_path.with_name("RobotCar_HeteroGauss.pt")

    payload = torch.load(in_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Input PT must contain a dict payload.")
    trajectories = payload.get("trajectories")
    if not isinstance(trajectories, list):
        raise ValueError("Input PT payload missing list key: trajectories")

    rng = np.random.default_rng(int(seed))

    clean_lonlat_list: list[np.ndarray] = []
    ts_list: list[np.ndarray | None] = []
    factor_list: list[np.ndarray] = []
    base_noise_e_list: list[np.ndarray] = []
    base_noise_n_list: list[np.ndarray] = []
    base_dist_list: list[np.ndarray] = []

    for i, traj in enumerate(trajectories):
        if not isinstance(traj, dict):
            raise ValueError(f"Trajectory[{i}] must be dict.")
        clean = traj.get("label", traj.get("data"))
        if clean is None:
            raise ValueError(f"Trajectory[{i}] missing both label and data.")
        clean_arr = _to_np(clean).astype(np.float64, copy=False)
        if clean_arr.ndim != 2 or clean_arr.shape[1] != 2:
            raise ValueError(
                f"Trajectory[{i}] clean coordinates must be shape (N,2) in [lon,lat], got {clean_arr.shape}"
            )
        n = int(clean_arr.shape[0])
        if n <= 0:
            clean_lonlat_list.append(clean_arr.copy())
            ts_list.append(None)
            factor_list.append(np.zeros((0,), dtype=np.float64))
            base_noise_e_list.append(np.zeros((0,), dtype=np.float64))
            base_noise_n_list.append(np.zeros((0,), dtype=np.float64))
            base_dist_list.append(np.zeros((0,), dtype=np.float64))
            continue

        factors = _make_sigma_factors(
            n,
            sigma_cv=float(sigma_cv),
            ar_rho=float(ar_rho),
            clamp_min=float(sigma_factor_min),
            clamp_max=float(sigma_factor_max),
            rng=rng,
        )
        z_e = rng.normal(0.0, 1.0, n)
        z_n = rng.normal(0.0, 1.0, n)
        base_e = factors * z_e
        base_n = factors * z_n
        base_d = np.sqrt(base_e * base_e + base_n * base_n)

        ts_raw = traj.get("timestamp")
        ts_np = None if ts_raw is None else _to_np(ts_raw).reshape(-1).astype(np.float64, copy=False)

        clean_lonlat_list.append(clean_arr.copy())
        ts_list.append(ts_np)
        factor_list.append(factors)
        base_noise_e_list.append(base_e)
        base_noise_n_list.append(base_n)
        base_dist_list.append(base_d)

    flat_base = np.concatenate(base_dist_list, axis=0) if base_dist_list else np.zeros((0,), dtype=np.float64)
    base_stats = _stats(flat_base)
    scale_m = _resolve_scale(
        base_stats,
        target_mean_m=float(target_mean_m),
        target_median_m=float(target_median_m),
        target_std_m=float(target_std_m),
        mode=str(scale_mode),
        w_mean=float(weight_mean),
        w_median=float(weight_median),
        w_std=float(weight_std),
    )

    out_trajs: list[dict[str, Any]] = []
    realized_dist_list: list[np.ndarray] = []
    total_points = 0

    for i, traj in enumerate(trajectories):
        clean_arr = clean_lonlat_list[i]
        n = int(clean_arr.shape[0])
        one = dict(traj)
        if n <= 0:
            one["data_clean"] = torch.empty((0, 2), dtype=torch.float32)
            one["data"] = torch.empty((0, 2), dtype=torch.float32)
            out_trajs.append(one)
            continue

        lon = clean_arr[:, 0]
        lat = clean_arr[:, 1]
        ref_lon = float(lon[0])
        ref_lat = float(lat[0])

        e_clean, n_clean, _ = geodetic2enu(lat, lon, 0.0, ref_lat, ref_lon, 0.0)
        e_clean = np.asarray(e_clean, dtype=np.float64)
        n_clean = np.asarray(n_clean, dtype=np.float64)

        de = float(scale_m) * base_noise_e_list[i]
        dn = float(scale_m) * base_noise_n_list[i]
        e_noisy = e_clean + de
        n_noisy = n_clean + dn

        lat_noisy, lon_noisy, _ = enu2geodetic(e_noisy, n_noisy, 0.0, ref_lat, ref_lon, 0.0)
        noisy_lonlat = np.column_stack([lon_noisy, lat_noisy]).astype(np.float32, copy=False)

        one["data_clean"] = torch.from_numpy(clean_arr.astype(np.float32, copy=False))
        one["data"] = torch.from_numpy(noisy_lonlat)
        out_trajs.append(one)

        realized_dist = np.sqrt(de * de + dn * dn)
        realized_dist_list.append(realized_dist)
        total_points += n

    realized = np.concatenate(realized_dist_list, axis=0) if realized_dist_list else np.zeros((0,), dtype=np.float64)
    realized_stats = _stats(realized)

    out_payload = dict(payload)
    out_payload["trajectories"] = out_trajs
    out_payload["noise"] = {
        "type": "heteroscedastic_gaussian",
        "coord_space": "ENU_meters",
        "seed": int(seed),
        "sigma_cv": float(sigma_cv),
        "ar_rho": float(ar_rho),
        "sigma_factor_min": float(sigma_factor_min),
        "sigma_factor_max": float(sigma_factor_max),
        "scale_mode": str(scale_mode),
        "weights": {
            "mean": float(weight_mean),
            "median": float(weight_median),
            "std": float(weight_std),
        },
        "target_error_stats_m": {
            "mean": float(target_mean_m),
            "median": float(target_median_m),
            "std": float(target_std_m),
        },
        "base_error_stats_m": base_stats,
        "global_scale_m": float(scale_m),
        "realized_error_stats_m": realized_stats,
        "notes": "Applied to data only; label/timestamp/error_range are preserved.",
    }
    out_payload["source_input_pt"] = str(in_path)
    out_payload["output_pt"] = str(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_payload, out_path)

    return {
        "status": "completed",
        "input_pt": str(in_path),
        "output_pt": str(out_path),
        "n_trajectories": int(len(out_trajs)),
        "total_points": int(total_points),
        "scale_mode": str(scale_mode),
        "sigma_cv": float(sigma_cv),
        "ar_rho": float(ar_rho),
        "global_scale_m": float(scale_m),
        "target_error_stats_m": {
            "mean": float(target_mean_m),
            "median": float(target_median_m),
            "std": float(target_std_m),
        },
        "realized_error_stats_m": realized_stats,
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    result = add_heteroscedastic_gaussian_noise(
        input_pt=str(args.input_pt),
        output_pt=str(args.output_pt) if args.output_pt else None,
        seed=int(args.seed),
        target_mean_m=float(args.target_mean_m),
        target_median_m=float(args.target_median_m),
        target_std_m=float(args.target_std_m),
        scale_mode=str(args.scale_mode),
        weight_mean=float(args.weight_mean),
        weight_median=float(args.weight_median),
        weight_std=float(args.weight_std),
        sigma_cv=float(args.sigma_cv),
        ar_rho=float(args.ar_rho),
        sigma_factor_min=float(args.sigma_factor_min),
        sigma_factor_max=float(args.sigma_factor_max),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
