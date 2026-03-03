from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


# ================================================================
# === Input Normalization Helpers
# ================================================================
def ensure_lat_lon_timestamp_sequence(data_seq: np.ndarray) -> np.ndarray:
    """
    Normalize prediction input sequence to shape (N,3) in [lat, lon, timestamp].
    """
    # ------------------------------------------------------------
    # 1) Normalize dtype / base shape
    # ------------------------------------------------------------
    arr = np.asarray(data_seq, dtype=float)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError("data_seq must be shape (N,2|3) in [lat, lon, (timestamp)] format")

    # ------------------------------------------------------------
    # 2) Fast path: empty sequence
    # ------------------------------------------------------------
    if arr.shape[0] == 0:
        return np.empty((0, 3), dtype=float)

    # ------------------------------------------------------------
    # 3) Fill timestamps if caller provides only [lat, lon]
    # ------------------------------------------------------------
    if arr.shape[1] == 2:
        ts = np.arange(arr.shape[0], dtype=float)
        arr = np.concatenate([arr, ts[:, None]], axis=1)
    return arr


def build_lat_lon_timestamp_sequence_from_lonlat(
    noisy_lonlat: np.ndarray,
    timestamps: np.ndarray | None = None,
) -> np.ndarray:
    """
    Convert [lon, lat] (+ optional timestamps) into [lat, lon, timestamp].
    """
    # ------------------------------------------------------------
    # 1) Validate input coordinates
    # ------------------------------------------------------------
    gps = np.asarray(noisy_lonlat, dtype=float)
    if gps.ndim != 2 or gps.shape[1] != 2:
        raise ValueError("noisy_lonlat must be shape (N,2) in [lon,lat] order")
    n = gps.shape[0]

    # ------------------------------------------------------------
    # 2) Fast path: empty sequence
    # ------------------------------------------------------------
    if n == 0:
        return np.empty((0, 3), dtype=float)

    # ------------------------------------------------------------
    # 3) Resolve timestamps
    # ------------------------------------------------------------
    if timestamps is None:
        ts = np.arange(n, dtype=float)
    else:
        ts = np.asarray(timestamps, dtype=float).reshape(-1)
        if ts.size != n:
            raise ValueError(f"timestamps length {ts.size} != n {n}")

    # ------------------------------------------------------------
    # 4) Reorder to baseline contract [lat, lon, t]
    # ------------------------------------------------------------
    lat = gps[:, 1]
    lon = gps[:, 0]
    return np.stack([lat, lon, ts], axis=1)


def latlon_to_lonlat(latlon: np.ndarray) -> np.ndarray:
    # ------------------------------------------------------------
    # Canonical reorder helper used by adapters.
    # ------------------------------------------------------------
    arr = np.asarray(latlon, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("latlon must be shape (N,2) in [lat,lon] order")
    if arr.shape[0] == 0:
        return arr.copy()
    return np.stack([arr[:, 1], arr[:, 0]], axis=1)


def _predict_chunked(
    seq_latlon_t: np.ndarray,
    predict_block_fn: Callable[[np.ndarray], np.ndarray],
    chunk_size: int,
    overlap: int,
) -> np.ndarray:
    """
    Chunk large trajectories and average overlap regions for stable stitching.
    """
    # ------------------------------------------------------------
    # 1) Validate size / derive chunk schedule
    # ------------------------------------------------------------
    n = int(seq_latlon_t.shape[0])
    if n <= 0:
        return np.empty((0, 2), dtype=float)
    chunk = max(1, int(chunk_size))
    ov = max(0, int(overlap))
    if chunk <= ov:
        ov = max(0, chunk - 1)
    step = max(1, chunk - ov)

    # ------------------------------------------------------------
    # 2) Run chunk predictions and accumulate overlap votes
    # ------------------------------------------------------------
    out_sum = np.zeros((n, 2), dtype=float)
    out_cnt = np.zeros(n, dtype=float)
    start = 0
    while start < n:
        end = min(n, start + chunk)
        pred = np.asarray(predict_block_fn(seq_latlon_t[start:end]), dtype=float)
        if pred.shape != (end - start, 2):
            raise ValueError(
                f"Chunk predictor returned invalid shape {pred.shape}, expected {(end - start, 2)}"
            )
        out_sum[start:end] += pred
        out_cnt[start:end] += 1.0
        if end >= n:
            break
        start += step

    # ------------------------------------------------------------
    # 3) Average overlap regions
    # ------------------------------------------------------------
    out_cnt = np.maximum(out_cnt, 1.0)
    return out_sum / out_cnt[:, None]


# ================================================================
# === BaselineModel (Universal Lifecycle Interface)
# ================================================================
class BaselineModel(ABC):
    """
    Universal baseline interface:
    - initialize()
    - calibrate()
    - predict()
    - deconst()
    """

    requires_calibration: bool = False
    requires_map: bool = False

    def __init__(
        self,
        *,
        method_name: str,
        dataset_name: str | None = None,
        use_timestamps: bool = True,
        max_predict_points: int | None = None,
        chunk_overlap: int = 32,
    ) -> None:
        self.method_name = str(method_name)
        self.dataset_name = dataset_name
        self.use_timestamps = bool(use_timestamps)
        self.max_predict_points = int(max_predict_points) if max_predict_points else None
        self.chunk_overlap = int(chunk_overlap)
        self.calibration_file: str | None = None
        self.map_file: str | None = None
        self.calibration_summary: dict = {"status": "not_initialized"}

    def initialize(
        self,
        calibration_file: str | None = None,
        map_file: str | None = None,
    ) -> dict:
        # ------------------------------------------------------------
        # Setup phase (not timed): store artifacts + run calibration.
        # ------------------------------------------------------------
        # Keep all expensive setup in initialize(), outside timed predict().
        self.calibration_file = calibration_file
        self.map_file = map_file
        self.calibration_summary = self.calibrate(
            calibration_file=calibration_file,
            map_file=map_file,
        )
        return self.calibration_summary

    def calibrate(
        self,
        calibration_file: str | None = None,
        map_file: str | None = None,
    ) -> dict:
        del calibration_file, map_file
        return {"status": "noop"}

    def predict(self, data_seq: np.ndarray) -> np.ndarray:
        # ------------------------------------------------------------
        # 1) Normalize caller input to [lat, lon, t]
        # ------------------------------------------------------------
        # Canonicalize input first so all baselines receive the same schema.
        seq = ensure_lat_lon_timestamp_sequence(data_seq)
        if seq.shape[0] == 0:
            return np.empty((0, 2), dtype=float)

        # ------------------------------------------------------------
        # 2) Optional chunked path for long trajectories
        # ------------------------------------------------------------
        # Large-trajectory handling is centralized in the base class.
        if self.max_predict_points is not None and seq.shape[0] > int(self.max_predict_points):
            return _predict_chunked(
                seq,
                self._predict_block,
                chunk_size=int(self.max_predict_points),
                overlap=int(self.chunk_overlap),
            )
        # ------------------------------------------------------------
        # 3) Direct single-block prediction
        # ------------------------------------------------------------
        return self._predict_block(seq)

    def predict_enu(
        self,
        positions_enu: np.ndarray,
        timestamps: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Optional ENU prediction path for chunk evaluators.

        Baselines that naturally operate in ENU should override this.
        """
        del positions_enu, timestamps
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support direct ENU prediction."
        )

    @abstractmethod
    def _predict_block(self, seq_latlon_t: np.ndarray) -> np.ndarray:
        """
        Predict denoised GPS positions in [lat, lon] for a single sequence block.
        """

    def deconst(self) -> None:
        """
        Baseline memory/server cleanup hook.
        """
        return None


__all__ = [
    "BaselineModel",
    "ensure_lat_lon_timestamp_sequence",
    "build_lat_lon_timestamp_sequence_from_lonlat",
    "latlon_to_lonlat",
]
