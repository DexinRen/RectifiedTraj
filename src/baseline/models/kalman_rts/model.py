from __future__ import annotations

import numpy as np

from ... import classic as classic_baseline
from ...base import BaselineModel


# ================================================================
# === KalmanRTSBaselineModel
# ================================================================
class KalmanRTSBaselineModel(BaselineModel):
    requires_calibration = True

    def __init__(
        self,
        *,
        dataset_name: str | None = None,
        use_timestamps: bool = True,
        fallback_dataset: str = "NUMOSIM_Kanto",
    ) -> None:
        super().__init__(
            method_name="kalman_rts",
            dataset_name=dataset_name,
            use_timestamps=use_timestamps,
            max_predict_points=None,
        )
        self.fallback_dataset = str(fallback_dataset)
        self._impl: classic_baseline.KalmanRTS | None = None

    def calibrate(
        self,
        calibration_file: str | None = None,
    ) -> dict:
        # ------------------------------------------------------------
        # Build calibrated Kalman-RTS engine once during initialize().
        # ------------------------------------------------------------
        self._impl = classic_baseline.KalmanRTS(
            calibration_file=calibration_file,
            dataset_name_hint=self.dataset_name,
            fallback_dataset=self.fallback_dataset,
            calibrate=True,
        )
        summary = dict(getattr(self._impl, "calibration_summary", {}) or {})
        summary.setdefault("status", "unknown")
        summary.setdefault("mode", "artifact")
        return summary

    def _predict_block(self, seq_latlon_t: np.ndarray) -> np.ndarray:
        # ------------------------------------------------------------
        # 1) Validate lifecycle + unpack unified [lat, lon, t] input
        # ------------------------------------------------------------
        if self._impl is None:
            raise RuntimeError("KalmanRTSBaselineModel is not initialized.")
        lat = seq_latlon_t[:, 0]
        lon = seq_latlon_t[:, 1]
        ts = seq_latlon_t[:, 2] if self.use_timestamps else None

        # ------------------------------------------------------------
        # 2) Convert coordinate order for classic backend [lon, lat]
        # ------------------------------------------------------------
        noisy_lonlat = np.stack([lon, lat], axis=1)
        den_lonlat = self._impl.denoise_gps(
            noisy_lonlat,
            timestamps=ts,
            use_timestamps=self.use_timestamps,
            ref_lat=float(lat[0]),
            ref_lon=float(lon[0]),
        )

        # ------------------------------------------------------------
        # 3) Convert back to public output contract [lat, lon]
        # ------------------------------------------------------------
        return np.stack([den_lonlat[:, 1], den_lonlat[:, 0]], axis=1)

    def predict_enu(
        self,
        positions_enu: np.ndarray,
        timestamps: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._impl is None:
            raise RuntimeError("KalmanRTSBaselineModel is not initialized.")
        enu = np.asarray(positions_enu, dtype=float)
        if enu.ndim != 2 or enu.shape[1] != 2:
            raise ValueError("positions_enu must be shape (N,2)")
        ts = np.asarray(timestamps, dtype=float).reshape(-1) if timestamps is not None else None
        if ts is not None and ts.size != enu.shape[0]:
            raise ValueError(f"timestamps length {ts.size} != n {enu.shape[0]}")
        return self._impl.denoise_enu(
            enu,
            timestamps=ts if self.use_timestamps else None,
            use_timestamps=self.use_timestamps,
        )

    def deconst(self) -> None:
        self._impl = None
