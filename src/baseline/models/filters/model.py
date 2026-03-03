from __future__ import annotations

import numpy as np
from pymap3d import enu2geodetic, geodetic2enu

from ... import classic as classic_baseline
from ...base import BaselineModel


# ================================================================
# === EuclideanFilterBaselineModel
# ================================================================
class EuclideanFilterBaselineModel(BaselineModel):
    def __init__(
        self,
        *,
        method_name: str,
        dataset_name: str | None = None,
        use_timestamps: bool = True,
    ) -> None:
        super().__init__(
            method_name=method_name,
            dataset_name=dataset_name,
            use_timestamps=use_timestamps,
            max_predict_points=None,
        )

        # ------------------------------------------------------------
        # Map registry method names to classic filter implementations.
        # ------------------------------------------------------------
        key = str(method_name).strip().lower()
        if key == "hampel":
            self._filter_key = "hampel"
        elif key == "savgol":
            self._filter_key = "savgol"
        elif key == "spline":
            self._filter_key = "spline"
        elif key == "raw":
            self._filter_key = "raw"
        else:
            raise ValueError(f"Unsupported Euclidean filter baseline: {method_name}")

    def _predict_enu(self, enu: np.ndarray, ts: np.ndarray | None) -> np.ndarray:
        # ------------------------------------------------------------
        # Dispatch stage: one branch per concrete filter backend.
        # ------------------------------------------------------------
        if self._filter_key == "hampel":
            return classic_baseline.hampel_filter(enu)
        if self._filter_key == "savgol":
            return classic_baseline.savitzky_golay_filter(enu)
        if self._filter_key == "spline":
            return classic_baseline.smoothing_spline(
                enu,
                timestamps=ts if self.use_timestamps else None,
            )
        if self._filter_key == "raw":
            return classic_baseline.raw_baseline(enu)
        raise RuntimeError(f"Unhandled filter key: {self._filter_key}")

    def _predict_block(self, seq_latlon_t: np.ndarray) -> np.ndarray:
        # ------------------------------------------------------------
        # 1) Unpack [lat, lon, t] and build local ENU frame
        # ------------------------------------------------------------
        lat = seq_latlon_t[:, 0]
        lon = seq_latlon_t[:, 1]
        ts = seq_latlon_t[:, 2] if self.use_timestamps else None
        ref_lat = float(lat[0])
        ref_lon = float(lon[0])
        e, n, _ = geodetic2enu(lat, lon, 0.0, ref_lat, ref_lon, 0.0)
        enu_noisy = np.stack([e, n], axis=1)

        # ------------------------------------------------------------
        # 2) Apply selected smoother in ENU coordinates
        # ------------------------------------------------------------
        den_enu = self._predict_enu(enu_noisy, ts=ts)

        # ------------------------------------------------------------
        # 3) Convert filtered ENU back to [lat, lon]
        # ------------------------------------------------------------
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
        return self._predict_enu(
            enu,
            ts if self.use_timestamps else None,
        )
