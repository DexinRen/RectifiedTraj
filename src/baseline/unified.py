from __future__ import annotations

"""
Compatibility shim.

New structure lives in:
- baseline.base
- baseline.artifacts
- baseline.registry
- baseline.models.*
"""

from .artifacts import (
    BaselineArtifacts,
    resolve_baseline_artifacts_from_state,
)
from .base import (
    BaselineModel,
    build_lat_lon_timestamp_sequence_from_lonlat,
    ensure_lat_lon_timestamp_sequence,
    latlon_to_lonlat,
)
from .models import (
    EuclideanFilterBaselineModel,
    FMMBaselineModel,
    KalmanRTSBaselineModel,
)
from .registry import create_baseline_model

__all__ = [
    "BaselineArtifacts",
    "BaselineModel",
    "KalmanRTSBaselineModel",
    "EuclideanFilterBaselineModel",
    "FMMBaselineModel",
    "resolve_baseline_artifacts_from_state",
    "ensure_lat_lon_timestamp_sequence",
    "build_lat_lon_timestamp_sequence_from_lonlat",
    "latlon_to_lonlat",
    "create_baseline_model",
]
