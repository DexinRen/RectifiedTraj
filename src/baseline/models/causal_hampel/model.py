from __future__ import annotations

import numpy as np
from pymap3d import enu2geodetic, geodetic2enu

from ... import classic as classic_baseline
from ...base import BaselineModel


class CausalHampelBaselineModel(BaselineModel):
    def __init__(
        self,
        *,
        dataset_name: str | None = None,
        use_timestamps: bool = True,
        fallback_dataset: str = "NUMOSIM_Kanto",
    ) -> None:
        super().__init__(
            method_name="causal_hampel",
            dataset_name=dataset_name,
            use_timestamps=use_timestamps,
            max_predict_points=None,
        )
        self.fallback_dataset = str(fallback_dataset)
        self.params = classic_baseline.CausalHampelParams()

    def calibrate(
        self,
        calibration_file: str | None = None,
    ) -> dict:
        entry = classic_baseline.resolve_causal_hampel_params_entry_from_state(
            dataset_name_hint=self.dataset_name,
            fallback_dataset=self.fallback_dataset,
        )
        params = classic_baseline._extract_causal_hampel_params_from_calibration_index_entry(entry)
        if params is not None:
            self.params = params
            return {
                "status": "ok",
                "mode": "state_json",
                "params": {
                    "window_size": int(self.params.window_size),
                    "n_sigma": float(self.params.n_sigma),
                },
            }

        if calibration_file:
            self.params, summary = classic_baseline.estimate_causal_hampel_params_from_calibration_file(
                calibration_file,
                default_params=self.params,
            )
            return {"status": "ok", "mode": "artifact", **summary}

        return {
            "status": "default",
            "mode": "default",
            "params": {
                "window_size": int(self.params.window_size),
                "n_sigma": float(self.params.n_sigma),
            },
        }

    def _predict_block(self, seq_latlon_t: np.ndarray) -> np.ndarray:
        lat = seq_latlon_t[:, 0]
        lon = seq_latlon_t[:, 1]
        ref_lat = float(lat[0])
        ref_lon = float(lon[0])
        e, n, _ = geodetic2enu(lat, lon, 0.0, ref_lat, ref_lon, 0.0)
        enu_noisy = np.stack([e, n], axis=1)
        den_enu = classic_baseline.causal_hampel_filter(
            enu_noisy,
            window_size=self.params.window_size,
            n_sigma=self.params.n_sigma,
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
        del timestamps
        enu = np.asarray(positions_enu, dtype=float)
        if enu.ndim != 2 or enu.shape[1] != 2:
            raise ValueError("positions_enu must be shape (N,2)")
        return classic_baseline.causal_hampel_filter(
            enu,
            window_size=self.params.window_size,
            n_sigma=self.params.n_sigma,
        )
