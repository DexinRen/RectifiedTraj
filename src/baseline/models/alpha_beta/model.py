from __future__ import annotations

import numpy as np
from pymap3d import enu2geodetic, geodetic2enu

from ... import classic as classic_baseline
from ...base import BaselineModel


class AlphaBetaBaselineModel(BaselineModel):
    def __init__(
        self,
        *,
        dataset_name: str | None = None,
        use_timestamps: bool = True,
        fallback_dataset: str = "NUMOSIM_Kanto",
    ) -> None:
        super().__init__(
            method_name="alpha_beta",
            dataset_name=dataset_name,
            use_timestamps=use_timestamps,
            max_predict_points=None,
        )
        self.fallback_dataset = str(fallback_dataset)
        self.params = classic_baseline.AlphaBetaParams()

    def calibrate(
        self,
        calibration_file: str | None = None,
    ) -> dict:
        entry = classic_baseline.resolve_alpha_beta_params_entry_from_state(
            dataset_name_hint=self.dataset_name,
            fallback_dataset=self.fallback_dataset,
        )
        params = classic_baseline._extract_alpha_beta_params_from_calibration_index_entry(entry)
        if params is not None:
            self.params = params
            return {
                "status": "ok",
                "mode": "calib_json",
                "params": {
                    "alpha": float(self.params.alpha),
                    "beta": float(self.params.beta),
                },
            }

        if calibration_file:
            seed_kalman = classic_baseline.load_kalman_params_from_state(
                dataset_name_hint=self.dataset_name,
                fallback_dataset=self.fallback_dataset,
            )
            self.params, summary = classic_baseline.estimate_alpha_beta_params_from_calibration_file(
                calibration_file,
                default_params=self.params,
                seed_kalman_params=seed_kalman,
            )
            return {"status": "ok", "mode": "artifact", **summary}

        return {
            "status": "default",
            "mode": "default",
            "params": {
                "alpha": float(self.params.alpha),
                "beta": float(self.params.beta),
            },
        }

    def _predict_block(self, seq_latlon_t: np.ndarray) -> np.ndarray:
        lat = seq_latlon_t[:, 0]
        lon = seq_latlon_t[:, 1]
        ts = seq_latlon_t[:, 2] if self.use_timestamps else None
        ref_lat = float(lat[0])
        ref_lon = float(lon[0])
        e, n, _ = geodetic2enu(lat, lon, 0.0, ref_lat, ref_lon, 0.0)
        enu_noisy = np.stack([e, n], axis=1)
        den_enu = classic_baseline.alpha_beta_filter(
            enu_noisy,
            timestamps=ts,
            params=self.params,
        )
        den_lat, den_lon, _ = enu2geodetic(
            den_enu[:, 0],
            den_enu[:, 1],
            0.0,
            ref_lat,
            ref_lon,
            0.0,
        )
        return np.stack([den_lat, den_lon], axis=1)

    def predict_enu(
        self,
        positions_enu: np.ndarray,
        timestamps: np.ndarray | None = None,
    ) -> np.ndarray:
        enu = np.asarray(positions_enu, dtype=float)
        if enu.ndim != 2 or enu.shape[1] != 2:
            raise ValueError("positions_enu must be shape (N,2)")
        ts = np.asarray(timestamps, dtype=float).reshape(-1) if timestamps is not None else None
        if ts is not None and ts.size != enu.shape[0]:
            raise ValueError(f"timestamps length {ts.size} != n {enu.shape[0]}")
        return classic_baseline.alpha_beta_filter(
            enu,
            timestamps=ts if self.use_timestamps else None,
            params=self.params,
        )
