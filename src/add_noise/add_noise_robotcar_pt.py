#!/usr/bin/env python3
"""
Add correlated OU noise to RobotCar extracted trajectory PT files.

Expected input schema:
- top-level dict containing key "trajectories" (list)
- each trajectory has key "data" with shape (N, 2):
  [longitude, latitude]
- timestamp is provided in "timestamp" (N,)
- bounded test uses "error_range" (meters)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add OU correlated GPS noise to RobotCar PT trajectories.")
    parser.add_argument(
        "--input-pt",
        type=str,
        required=True,
        help="Path to extracted RobotCar .pt file.",
    )
    parser.add_argument(
        "--output-pt",
        type=str,
        default=None,
        help="Optional output .pt path. Defaults to <input_stem>_ou_noisy.pt.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="OU drift factor. Higher means faster correction.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.0001,
        help="Noise magnitude in degrees (same convention as src/add_noise/add_noise.py).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducible noise.",
    )
    return parser


def _add_correlated_noise(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    *,
    alpha: float,
    sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(latitudes.shape[0])
    lat_noise = np.zeros(n, dtype=np.float64)
    lon_noise = np.zeros(n, dtype=np.float64)

    white_lat = rng.normal(0.0, float(sigma), n)
    white_lon = rng.normal(0.0, float(sigma), n)

    for t in range(1, n):
        lat_noise[t] = lat_noise[t - 1] * (1.0 - float(alpha)) + white_lat[t]
        lon_noise[t] = lon_noise[t - 1] * (1.0 - float(alpha)) + white_lon[t]

    return latitudes + lat_noise, longitudes + lon_noise


def add_noise_to_robotcar_pt(
    input_pt: str,
    *,
    output_pt: str | None = None,
    alpha: float = 0.1,
    sigma: float = 0.0001,
    seed: int = 7,
) -> dict[str, Any]:
    in_path = Path(input_pt).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input PT not found: {in_path}")

    if output_pt:
        out_path = Path(output_pt).resolve()
    else:
        out_path = in_path.with_name(f"{in_path.stem}_ou_noisy.pt")

    payload = torch.load(in_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Input PT must contain a dict payload.")
    trajectories = payload.get("trajectories")
    if not isinstance(trajectories, list):
        raise ValueError("Input PT payload missing list key: trajectories")

    rng = np.random.default_rng(int(seed))
    out_trajs = []
    total_points = 0

    for i, traj in enumerate(trajectories):
        if not isinstance(traj, dict):
            raise ValueError(f"Trajectory[{i}] must be dict.")
        data = traj.get("data")
        if data is None:
            raise ValueError(f"Trajectory[{i}] missing key: data")
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Trajectory[{i}] data must be shape (N,2) in [lon,lat], got {arr.shape}")

        lon = arr[:, 0].copy()
        lat = arr[:, 1].copy()
        lat_n, lon_n = _add_correlated_noise(
            lat,
            lon,
            alpha=float(alpha),
            sigma=float(sigma),
            rng=rng,
        )
        noisy_lonlat = np.column_stack([lon_n, lat_n])

        one = dict(traj)
        one["data_clean"] = torch.from_numpy(np.column_stack([lon, lat])).to(torch.float32)
        one["data"] = torch.from_numpy(noisy_lonlat).to(torch.float32)
        out_trajs.append(one)
        total_points += int(noisy_lonlat.shape[0])

    out_payload = dict(payload)
    out_payload["trajectories"] = out_trajs
    out_payload["noise"] = {
        "type": "ornstein_uhlenbeck_correlated_noise",
        "alpha": float(alpha),
        "sigma_degrees": float(sigma),
        "seed": int(seed),
        "columns": ["longitude", "latitude"],
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
        "alpha": float(alpha),
        "sigma_degrees": float(sigma),
        "seed": int(seed),
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    result = add_noise_to_robotcar_pt(
        input_pt=str(args.input_pt),
        output_pt=str(args.output_pt) if args.output_pt else None,
        alpha=float(args.alpha),
        sigma=float(args.sigma),
        seed=int(args.seed),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
