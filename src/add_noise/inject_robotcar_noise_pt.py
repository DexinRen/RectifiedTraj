#!/usr/bin/env python3
"""
Inject configurable noise models into RobotCar trajectory PT files.

Supported noise models:
- HeteroGaussian
- MixtureSwitching
- PiecewiseBiasJitter
- OU  (Ornstein-Uhlenbeck-like correlated process in ENU meters)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pymap3d import enu2geodetic, geodetic2enu


def _to_np(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _stats(arr: np.ndarray) -> dict[str, float]:
    x = np.asarray(arr, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "p95": 0.0}
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x)),
        "p95": float(np.percentile(x, 95)),
    }


def _resolve_scale(
    base_stats: dict[str, float],
    *,
    target_mean_m: float,
    target_median_m: float,
    target_std_m: float,
    scale_mode: str,
    weight_mean: float,
    weight_median: float,
    weight_std: float,
) -> float:
    bm = max(1e-12, float(base_stats["mean"]))
    bmed = max(1e-12, float(base_stats["median"]))
    bstd = max(1e-12, float(base_stats["std"]))
    tm = max(1e-12, float(target_mean_m))
    tmed = max(1e-12, float(target_median_m))
    ts = max(1e-12, float(target_std_m))

    mode = str(scale_mode).strip().lower()
    if mode == "mean":
        return float(tm / bm)
    if mode == "median":
        return float(tmed / bmed)

    wm = max(0.0, float(weight_mean))
    wmed = max(0.0, float(weight_median))
    ws = max(0.0, float(weight_std))
    denom = wm * bm * bm + wmed * bmed * bmed + ws * bstd * bstd
    if denom <= 0.0:
        return float(tm / bm)
    numer = wm * bm * tm + wmed * bmed * tmed + ws * bstd * ts
    return float(numer / denom)


def _make_sigma_factors(
    n: int,
    *,
    sigma_cv: float,
    ar_rho: float,
    sigma_factor_min: float,
    sigma_factor_max: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.float64)
    cv = max(0.0, float(sigma_cv))
    if cv <= 0.0:
        return np.ones((n,), dtype=np.float64)
    rho = float(np.clip(ar_rho, -0.999, 0.999))
    smin = max(1e-6, float(sigma_factor_min))
    smax = max(smin, float(sigma_factor_max))

    sigma_ln = float(np.sqrt(np.log(1.0 + cv * cv)))
    x = np.zeros((n,), dtype=np.float64)
    x[0] = rng.normal(0.0, 1.0)
    white_scale = np.sqrt(max(1e-12, 1.0 - rho * rho))
    for i in range(1, n):
        x[i] = rho * x[i - 1] + white_scale * rng.normal(0.0, 1.0)
    factors = np.exp(sigma_ln * x)
    factors = np.clip(factors, smin, smax)
    return factors


def _base_noise_hetero_gaussian(
    n: int,
    *,
    sigma_cv: float,
    ar_rho: float,
    sigma_factor_min: float,
    sigma_factor_max: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    factors = _make_sigma_factors(
        n,
        sigma_cv=sigma_cv,
        ar_rho=ar_rho,
        sigma_factor_min=sigma_factor_min,
        sigma_factor_max=sigma_factor_max,
        rng=rng,
    )
    z_e = rng.normal(0.0, 1.0, n)
    z_n = rng.normal(0.0, 1.0, n)
    de = factors * z_e
    dn = factors * z_n
    return de, dn, {
        "sigma_cv": float(sigma_cv),
        "ar_rho": float(ar_rho),
        "sigma_factor_min": float(sigma_factor_min),
        "sigma_factor_max": float(sigma_factor_max),
    }


def _base_noise_mixture_switching(
    n: int,
    *,
    p_stay_low: float,
    p_stay_high: float,
    sigma_low: float,
    sigma_high: float,
    init_state: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if n <= 0:
        return (
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            {
                "p_stay_low": float(p_stay_low),
                "p_stay_high": float(p_stay_high),
                "sigma_low": float(sigma_low),
                "sigma_high": float(sigma_high),
                "init_state": int(init_state),
                "state_ratio_low": 0.0,
                "state_ratio_high": 0.0,
            },
        )
    p0 = float(np.clip(p_stay_low, 0.0, 1.0))
    p1 = float(np.clip(p_stay_high, 0.0, 1.0))
    s0 = max(1e-9, float(sigma_low))
    s1 = max(1e-9, float(sigma_high))
    state = int(0 if int(init_state) <= 0 else 1)
    states = np.zeros((n,), dtype=np.int8)
    states[0] = state
    for i in range(1, n):
        if state == 0:
            state = 0 if rng.random() < p0 else 1
        else:
            state = 1 if rng.random() < p1 else 0
        states[i] = state
    sigmas = np.where(states == 0, s0, s1).astype(np.float64, copy=False)
    de = sigmas * rng.normal(0.0, 1.0, n)
    dn = sigmas * rng.normal(0.0, 1.0, n)
    return de, dn, {
        "p_stay_low": float(p0),
        "p_stay_high": float(p1),
        "sigma_low": float(s0),
        "sigma_high": float(s1),
        "init_state": int(init_state),
        "state_ratio_low": float(np.mean(states == 0)),
        "state_ratio_high": float(np.mean(states == 1)),
    }


def _parse_segment_length_choices(raw: str) -> list[int]:
    vals: list[int] = []
    for tok in str(raw).split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(int(tok))
    vals = sorted(set(v for v in vals if v > 0))
    if not vals:
        raise ValueError("PiecewiseBiasJitter segment length choices must be positive integers.")
    return vals


def _base_noise_piecewise_bias_jitter(
    n: int,
    *,
    segment_length_choices: list[int],
    bias_mag_median: float,
    bias_mag_ln_sigma: float,
    bias_mag_min: float,
    bias_mag_max: float,
    jitter_sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if n <= 0:
        return (
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            {
                "segment_length_choices": list(segment_length_choices),
                "bias_mag_median": float(bias_mag_median),
                "bias_mag_ln_sigma": float(bias_mag_ln_sigma),
                "bias_mag_min": float(bias_mag_min),
                "bias_mag_max": float(bias_mag_max),
                "jitter_sigma": float(jitter_sigma),
                "n_segments": 0,
                "mean_segment_len": 0.0,
                "mean_bias_mag": 0.0,
            },
        )

    choices = [int(v) for v in segment_length_choices if int(v) > 0]
    if not choices:
        raise ValueError("PiecewiseBiasJitter requires at least one positive segment length choice.")
    med = max(1e-9, float(bias_mag_median))
    ln_sigma = max(1e-9, float(bias_mag_ln_sigma))
    mag_min = max(1e-9, float(bias_mag_min))
    mag_max = max(mag_min, float(bias_mag_max))
    jit = max(1e-9, float(jitter_sigma))

    de = np.zeros((n,), dtype=np.float64)
    dn = np.zeros((n,), dtype=np.float64)
    seg_lengths: list[int] = []
    seg_bias_mags: list[float] = []

    start = 0
    while start < n:
        seg_len = int(rng.choice(choices))
        seg_len = max(1, min(seg_len, n - start))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        mag = float(np.exp(rng.normal(np.log(med), ln_sigma)))
        mag = float(np.clip(mag, mag_min, mag_max))
        bias_e = mag * np.cos(theta)
        bias_n = mag * np.sin(theta)

        stop = start + seg_len
        de[start:stop] = bias_e + rng.normal(0.0, jit, seg_len)
        dn[start:stop] = bias_n + rng.normal(0.0, jit, seg_len)
        seg_lengths.append(int(seg_len))
        seg_bias_mags.append(float(mag))
        start = stop

    return de, dn, {
        "segment_length_choices": list(choices),
        "bias_mag_median": float(med),
        "bias_mag_ln_sigma": float(ln_sigma),
        "bias_mag_min": float(mag_min),
        "bias_mag_max": float(mag_max),
        "jitter_sigma": float(jit),
        "n_segments": int(len(seg_lengths)),
        "mean_segment_len": float(np.mean(seg_lengths)),
        "mean_bias_mag": float(np.mean(seg_bias_mags)),
    }


def _base_noise_ou(
    n: int,
    *,
    alpha: float,
    white_sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if n <= 0:
        return (
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            {"alpha": float(alpha), "white_sigma": float(white_sigma)},
        )
    a = float(np.clip(alpha, 1e-6, 0.999999))
    s = max(1e-12, float(white_sigma))
    w_e = rng.normal(0.0, s, n)
    w_n = rng.normal(0.0, s, n)
    de = np.zeros((n,), dtype=np.float64)
    dn = np.zeros((n,), dtype=np.float64)
    de[0] = w_e[0]
    dn[0] = w_n[0]
    for i in range(1, n):
        de[i] = de[i - 1] * (1.0 - a) + w_e[i]
        dn[i] = dn[i - 1] * (1.0 - a) + w_n[i]
    return de, dn, {"alpha": float(a), "white_sigma": float(s)}


def _normalize_noise_type(raw: str) -> str:
    token = str(raw).strip().lower().replace("-", "_")
    if token in {"heterogaussian", "hetero_gaussian", "heteroscedastic", "heteroscedastic_gaussian"}:
        return "HeteroGaussian"
    if token in {"mixtureswitching", "mixture_switching", "switching", "mixture"}:
        return "MixtureSwitching"
    if token in {
        "piecewisebiasjitter",
        "piecewise_bias_jitter",
        "piecewise_bias",
        "biasjitter",
        "bias_jitter",
        "pbj",
    }:
        return "PiecewiseBiasJitter"
    if token in {"ou", "ou_correlated", "ornstein_uhlenbeck", "pol"}:
        return "OU"
    raise ValueError(f"Unsupported noise type: {raw}")


def inject_noise_robotcar_pt(
    input_pt: str,
    *,
    output_pt: str,
    noise_type: str,
    seed: int = 11,
    target_mean_m: float = 15.0,
    target_median_m: float = 15.0,
    target_std_m: float = 9.0,
    scale_mode: str = "weighted",
    weight_mean: float = 1.0,
    weight_median: float = 4.0,
    weight_std: float = 1.0,
    hetero_sigma_cv: float = 0.10,
    hetero_ar_rho: float = 0.96,
    hetero_sigma_factor_min: float = 0.35,
    hetero_sigma_factor_max: float = 3.00,
    mix_p_stay_low: float = 0.99,
    mix_p_stay_high: float = 0.80,
    mix_sigma_low: float = 1.0,
    mix_sigma_high: float = 1.2,
    mix_init_state: int = 0,
    pbj_segment_length_choices: str = "8,12,16,20,24,32",
    pbj_bias_mag_median: float = 1.0,
    pbj_bias_mag_ln_sigma: float = 0.50,
    pbj_bias_mag_min: float = 0.25,
    pbj_bias_mag_max: float = 3.50,
    pbj_jitter_sigma: float = 0.26,
    pol_alpha: float = 0.10,
    pol_white_sigma: float = 1.0,
) -> dict[str, Any]:
    in_path = Path(input_pt).resolve()
    out_path = Path(output_pt).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input PT not found: {in_path}")

    payload = torch.load(in_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Input PT payload must be a dict.")
    trajectories = payload.get("trajectories")
    if not isinstance(trajectories, list):
        raise ValueError("Input PT payload missing list key: trajectories")

    model = _normalize_noise_type(noise_type)
    rng = np.random.default_rng(int(seed))

    clean_lonlat: list[np.ndarray] = []
    base_noise_e: list[np.ndarray] = []
    base_noise_n: list[np.ndarray] = []
    model_meta_rows: list[dict[str, Any]] = []
    base_dist_list: list[np.ndarray] = []

    for i, row in enumerate(trajectories):
        if not isinstance(row, dict):
            raise ValueError(f"Trajectory[{i}] must be dict.")
        clean = row.get("label", row.get("data"))
        if clean is None:
            raise ValueError(f"Trajectory[{i}] missing label/data.")
        arr = _to_np(clean).astype(np.float64, copy=False)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Trajectory[{i}] clean shape invalid: {arr.shape}")
        n = int(arr.shape[0])
        clean_lonlat.append(arr.copy())

        if model == "HeteroGaussian":
            de, dn, one_meta = _base_noise_hetero_gaussian(
                n,
                sigma_cv=float(hetero_sigma_cv),
                ar_rho=float(hetero_ar_rho),
                sigma_factor_min=float(hetero_sigma_factor_min),
                sigma_factor_max=float(hetero_sigma_factor_max),
                rng=rng,
            )
        elif model == "MixtureSwitching":
            de, dn, one_meta = _base_noise_mixture_switching(
                n,
                p_stay_low=float(mix_p_stay_low),
                p_stay_high=float(mix_p_stay_high),
                sigma_low=float(mix_sigma_low),
                sigma_high=float(mix_sigma_high),
                init_state=int(mix_init_state),
                rng=rng,
            )
        elif model == "PiecewiseBiasJitter":
            de, dn, one_meta = _base_noise_piecewise_bias_jitter(
                n,
                segment_length_choices=_parse_segment_length_choices(pbj_segment_length_choices),
                bias_mag_median=float(pbj_bias_mag_median),
                bias_mag_ln_sigma=float(pbj_bias_mag_ln_sigma),
                bias_mag_min=float(pbj_bias_mag_min),
                bias_mag_max=float(pbj_bias_mag_max),
                jitter_sigma=float(pbj_jitter_sigma),
                rng=rng,
            )
        else:
            de, dn, one_meta = _base_noise_ou(
                n,
                alpha=float(pol_alpha),
                white_sigma=float(pol_white_sigma),
                rng=rng,
            )

        base_noise_e.append(de)
        base_noise_n.append(dn)
        one_dist = np.sqrt(de * de + dn * dn)
        base_dist_list.append(one_dist)
        model_meta_rows.append(one_meta)

    base_dist = np.concatenate(base_dist_list, axis=0) if base_dist_list else np.zeros((0,), dtype=np.float64)
    base_stats = _stats(base_dist)
    scale = _resolve_scale(
        base_stats,
        target_mean_m=float(target_mean_m),
        target_median_m=float(target_median_m),
        target_std_m=float(target_std_m),
        scale_mode=str(scale_mode),
        weight_mean=float(weight_mean),
        weight_median=float(weight_median),
        weight_std=float(weight_std),
    )

    out_trajs: list[dict[str, Any]] = []
    realized_dist_list: list[np.ndarray] = []
    points_total = 0
    for i, row in enumerate(trajectories):
        one = dict(row)
        clean = clean_lonlat[i]
        n = int(clean.shape[0])
        if n <= 0:
            one["data_clean"] = torch.empty((0, 2), dtype=torch.float32)
            one["data"] = torch.empty((0, 2), dtype=torch.float32)
            out_trajs.append(one)
            continue

        lon = clean[:, 0]
        lat = clean[:, 1]
        ref_lon = float(lon[0])
        ref_lat = float(lat[0])
        e_clean, n_clean, _ = geodetic2enu(lat, lon, 0.0, ref_lat, ref_lon, 0.0)
        e_clean = np.asarray(e_clean, dtype=np.float64)
        n_clean = np.asarray(n_clean, dtype=np.float64)

        de = float(scale) * base_noise_e[i]
        dn = float(scale) * base_noise_n[i]
        e_noisy = e_clean + de
        n_noisy = n_clean + dn

        lat_noisy, lon_noisy, _ = enu2geodetic(e_noisy, n_noisy, 0.0, ref_lat, ref_lon, 0.0)
        noisy_lonlat = np.column_stack([lon_noisy, lat_noisy]).astype(np.float32, copy=False)

        one["data_clean"] = torch.from_numpy(clean.astype(np.float32, copy=False))
        one["data"] = torch.from_numpy(noisy_lonlat).to(torch.float32)
        if "label" not in one:
            one["label"] = torch.from_numpy(clean.astype(np.float32, copy=False))
        out_trajs.append(one)

        realized_dist_list.append(np.sqrt(de * de + dn * dn))
        points_total += int(n)

    realized_dist = (
        np.concatenate(realized_dist_list, axis=0) if realized_dist_list else np.zeros((0,), dtype=np.float64)
    )
    realized_stats = _stats(realized_dist)

    noise_meta: dict[str, Any] = {
        "type": str(model),
        "coord_space": "ENU_meters",
        "seed": int(seed),
        "target_error_stats_m": {
            "mean": float(target_mean_m),
            "median": float(target_median_m),
            "std": float(target_std_m),
        },
        "scale_mode": str(scale_mode),
        "weights": {
            "mean": float(weight_mean),
            "median": float(weight_median),
            "std": float(weight_std),
        },
        "base_error_stats_m": base_stats,
        "global_scale_m": float(scale),
        "realized_error_stats_m": realized_stats,
        "notes": "data contains noisy GPS; label remains reference clean GPS; time_id preserved.",
    }
    if model == "HeteroGaussian":
        noise_meta["model"] = {
            "sigma_cv": float(hetero_sigma_cv),
            "ar_rho": float(hetero_ar_rho),
            "sigma_factor_min": float(hetero_sigma_factor_min),
            "sigma_factor_max": float(hetero_sigma_factor_max),
        }
    elif model == "MixtureSwitching":
        noise_meta["model"] = {
            "p_stay_low": float(mix_p_stay_low),
            "p_stay_high": float(mix_p_stay_high),
            "sigma_low": float(mix_sigma_low),
            "sigma_high": float(mix_sigma_high),
            "init_state": int(mix_init_state),
            "state_ratio_low_mean": float(np.mean([m.get("state_ratio_low", 0.0) for m in model_meta_rows])),
            "state_ratio_high_mean": float(np.mean([m.get("state_ratio_high", 0.0) for m in model_meta_rows])),
        }
    elif model == "PiecewiseBiasJitter":
        noise_meta["model"] = {
            "segment_length_choices": _parse_segment_length_choices(pbj_segment_length_choices),
            "bias_mag_median": float(pbj_bias_mag_median),
            "bias_mag_ln_sigma": float(pbj_bias_mag_ln_sigma),
            "bias_mag_min": float(pbj_bias_mag_min),
            "bias_mag_max": float(pbj_bias_mag_max),
            "jitter_sigma": float(pbj_jitter_sigma),
            "mean_segments_per_trajectory": float(np.mean([m.get("n_segments", 0.0) for m in model_meta_rows])),
            "mean_segment_len": float(np.mean([m.get("mean_segment_len", 0.0) for m in model_meta_rows])),
            "mean_bias_mag": float(np.mean([m.get("mean_bias_mag", 0.0) for m in model_meta_rows])),
        }
    else:
        noise_meta["model"] = {
            "alpha": float(pol_alpha),
            "white_sigma": float(pol_white_sigma),
        }

    out_payload = dict(payload)
    out_payload["trajectories"] = out_trajs
    out_payload["noise"] = noise_meta
    out_payload["source_input_pt"] = str(in_path)
    out_payload["output_pt"] = str(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_payload, out_path)

    return {
        "status": "completed",
        "input_pt": str(in_path),
        "output_pt": str(out_path),
        "noise_type": str(model),
        "n_trajectories": int(len(out_trajs)),
        "total_points": int(points_total),
        "target_error_stats_m": noise_meta["target_error_stats_m"],
        "realized_error_stats_m": realized_stats,
        "global_scale_m": float(scale),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inject RobotCar trajectory noise.")
    parser.add_argument("--input-pt", type=str, required=True, help="Input extracted RobotCar pt file.")
    parser.add_argument("--output-pt", type=str, required=True, help="Output noisy pt file path.")
    parser.add_argument(
        "--noise-type",
        type=str,
        required=True,
        choices=["HeteroGaussian", "MixtureSwitching", "PiecewiseBiasJitter", "OU", "POL"],
        help="Noise model. POL is accepted as an alias of OU.",
    )
    parser.add_argument("--seed", type=int, default=11, help="Random seed.")

    parser.add_argument("--target-mean-m", type=float, default=15.0)
    parser.add_argument("--target-median-m", type=float, default=15.0)
    parser.add_argument("--target-std-m", type=float, default=9.0)
    parser.add_argument(
        "--scale-mode",
        type=str,
        default="weighted",
        choices=["mean", "median", "weighted"],
    )
    parser.add_argument("--weight-mean", type=float, default=1.0)
    parser.add_argument("--weight-median", type=float, default=4.0)
    parser.add_argument("--weight-std", type=float, default=1.0)

    parser.add_argument("--hetero-sigma-cv", type=float, default=0.10)
    parser.add_argument("--hetero-ar-rho", type=float, default=0.96)
    parser.add_argument("--hetero-sigma-factor-min", type=float, default=0.35)
    parser.add_argument("--hetero-sigma-factor-max", type=float, default=3.00)

    parser.add_argument("--mix-p-stay-low", type=float, default=0.99)
    parser.add_argument("--mix-p-stay-high", type=float, default=0.80)
    parser.add_argument("--mix-sigma-low", type=float, default=1.0)
    parser.add_argument("--mix-sigma-high", type=float, default=1.2)
    parser.add_argument("--mix-init-state", type=int, default=0)

    parser.add_argument("--pbj-segment-length-choices", type=str, default="8,12,16,20,24,32")
    parser.add_argument("--pbj-bias-mag-median", type=float, default=1.0)
    parser.add_argument("--pbj-bias-mag-ln-sigma", type=float, default=0.50)
    parser.add_argument("--pbj-bias-mag-min", type=float, default=0.25)
    parser.add_argument("--pbj-bias-mag-max", type=float, default=3.50)
    parser.add_argument("--pbj-jitter-sigma", type=float, default=0.26)

    parser.add_argument("--pol-alpha", type=float, default=0.10)
    parser.add_argument("--pol-white-sigma", type=float, default=1.0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = inject_noise_robotcar_pt(
        input_pt=str(args.input_pt),
        output_pt=str(args.output_pt),
        noise_type=str(args.noise_type),
        seed=int(args.seed),
        target_mean_m=float(args.target_mean_m),
        target_median_m=float(args.target_median_m),
        target_std_m=float(args.target_std_m),
        scale_mode=str(args.scale_mode),
        weight_mean=float(args.weight_mean),
        weight_median=float(args.weight_median),
        weight_std=float(args.weight_std),
        hetero_sigma_cv=float(args.hetero_sigma_cv),
        hetero_ar_rho=float(args.hetero_ar_rho),
        hetero_sigma_factor_min=float(args.hetero_sigma_factor_min),
        hetero_sigma_factor_max=float(args.hetero_sigma_factor_max),
        mix_p_stay_low=float(args.mix_p_stay_low),
        mix_p_stay_high=float(args.mix_p_stay_high),
        mix_sigma_low=float(args.mix_sigma_low),
        mix_sigma_high=float(args.mix_sigma_high),
        mix_init_state=int(args.mix_init_state),
        pbj_segment_length_choices=str(args.pbj_segment_length_choices),
        pbj_bias_mag_median=float(args.pbj_bias_mag_median),
        pbj_bias_mag_ln_sigma=float(args.pbj_bias_mag_ln_sigma),
        pbj_bias_mag_min=float(args.pbj_bias_mag_min),
        pbj_bias_mag_max=float(args.pbj_bias_mag_max),
        pbj_jitter_sigma=float(args.pbj_jitter_sigma),
        pol_alpha=float(args.pol_alpha),
        pol_white_sigma=float(args.pol_white_sigma),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
