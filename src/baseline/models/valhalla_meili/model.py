from __future__ import annotations

import logging
from collections import Counter

import numpy as np

from ...base import BaselineModel, ensure_lat_lon_timestamp_sequence
from .client import request_trace_attributes


LOGGER = logging.getLogger(__name__)


def validate_valhalla_config(config: dict) -> dict:
    """
    Purpose:
        Validate the explicit Valhalla Meili benchmark configuration.
    Parameters:
        config (dict), complete configuration supplied by the benchmark job.
    Return Dict:
        "error_code": int, 0 for valid configuration.
        "config": dict, normalized configuration values.
    Usage:
        ValhallaMeiliBaselineModel validates configuration at construction.
    TODO:
        1) Require every reproducibility-critical field.
        2) Validate numeric window and timeout constraints.
        3) Return normalized values without environment fallbacks.
    """

    # 1. Require Reproducibility-Critical Fields
    if not isinstance(config, dict):
        raise TypeError("valhalla_meili configuration must be a JSON object.")
    required = {
        "base_url",
        "costing",
        "shape_match",
        "window_points",
        "overlap_points",
        "timeout_sec",
        "auto_start",
        "map_id",
        "compose_file",
        "processed_map_root",
        "port",
        "startup_timeout_sec",
        "build_timeout_sec",
    }
    missing = sorted(required - set(config.keys()))
    if missing:
        raise ValueError(f"valhalla_meili configuration is missing: {', '.join(missing)}")
    if str(config["shape_match"]).strip() != "map_snap":
        raise ValueError("valhalla_meili.shape_match must be 'map_snap'.")
    if not str(config["costing"]).strip():
        raise ValueError("valhalla_meili.costing must be explicit.")
    if not str(config["map_id"]).strip():
        raise ValueError("valhalla_meili.map_id must be explicit.")

    # 2. Validate Window And Timeout Constraints
    window_points = int(config["window_points"])
    overlap_points = int(config["overlap_points"])
    timeout_sec = float(config["timeout_sec"])
    port = int(config["port"])
    startup_timeout_sec = float(config["startup_timeout_sec"])
    build_timeout_sec = float(config["build_timeout_sec"])
    if window_points < 2:
        raise ValueError("valhalla_meili.window_points must be >= 2.")
    if overlap_points < 0 or overlap_points >= window_points:
        raise ValueError("valhalla_meili.overlap_points must satisfy 0 <= overlap < window.")
    if timeout_sec <= 0.0:
        raise ValueError("valhalla_meili.timeout_sec must be positive.")
    if port < 1 or port > 65535:
        raise ValueError("valhalla_meili.port must be between 1 and 65535.")
    if startup_timeout_sec <= 0.0 or build_timeout_sec <= 0.0:
        raise ValueError("Valhalla startup/build timeouts must be positive.")

    # 3. Return Normalized Values
    normalized = dict(config)
    normalized["base_url"] = str(config["base_url"]).rstrip("/")
    normalized["costing"] = str(config["costing"]).strip()
    normalized["shape_match"] = "map_snap"
    normalized["window_points"] = window_points
    normalized["overlap_points"] = overlap_points
    normalized["timeout_sec"] = timeout_sec
    normalized["auto_start"] = bool(config["auto_start"])
    normalized["map_id"] = str(config["map_id"]).strip()
    normalized["compose_file"] = str(config["compose_file"])
    normalized["processed_map_root"] = str(config["processed_map_root"])
    normalized["port"] = port
    normalized["startup_timeout_sec"] = startup_timeout_sec
    normalized["build_timeout_sec"] = build_timeout_sec
    return {"error_code": 0, "config": normalized}


def build_request_windows(n_points: int, window_points: int, overlap_points: int) -> dict:
    """
    Purpose:
        Split one trajectory into deterministic fixed-size overlapping windows.
    Parameters:
        n_points (int), trajectory length, must be >= 0.
        window_points (int), maximum request size, must be >= 2.
        overlap_points (int), overlap, must satisfy 0 <= overlap < window.
    Return Dict:
        "error_code": int, 0 for valid schedule.
        "windows": list[dict], each with start and end indices.
    Usage:
        ValhallaMeiliBaselineModel uses the schedule for long trajectories.
    TODO:
        1) Validate sizes.
        2) Build fixed-stride windows.
        3) Ensure the final point is covered exactly by the last window.
    """

    # 1. Validate Sizes
    n_value = int(n_points)
    window_value = int(window_points)
    overlap_value = int(overlap_points)
    if n_value < 0:
        raise ValueError("n_points must be nonnegative.")
    if window_value < 2:
        raise ValueError("window_points must be >= 2.")
    if overlap_value < 0 or overlap_value >= window_value:
        raise ValueError("overlap_points must satisfy 0 <= overlap < window.")
    if n_value == 0:
        return {"error_code": 0, "windows": []}
    if n_value <= window_value:
        return {"error_code": 0, "windows": [{"start": 0, "end": n_value}]}

    # 2. Build Fixed-Stride Windows
    stride = window_value - overlap_value
    windows: list[dict] = []
    start = 0
    while start < n_value:
        end = min(start + window_value, n_value)
        windows.append({"start": int(start), "end": int(end)})
        if end == n_value:
            break
        start += stride

    # 3. Verify Full Coverage
    coverage = np.zeros(n_value, dtype=bool)
    for window in windows:
        coverage[window["start"] : window["end"]] = True
    if not bool(np.all(coverage)):
        raise RuntimeError("Valhalla request window schedule failed to cover the trajectory.")
    return {"error_code": 0, "windows": windows}


def merge_counter(target: Counter[str], source: dict) -> dict:
    """
    Purpose:
        Add one string-keyed diagnostic counter into an aggregate counter.
    Parameters:
        target (Counter[str]), mutable aggregate counter.
        source (dict), string keys with integer counts.
    Return Dict:
        "error_code": int, 0 after merge.
        "counter": Counter[str], updated target object.
    Usage:
        Valhalla prediction aggregation combines request diagnostics.
    TODO:
        1) Validate source counts.
        2) Add counts to the target.
        3) Return the updated counter packet.
    """

    # 1. Validate Source Counts
    if not isinstance(source, dict):
        raise TypeError("Diagnostic counter source must be a dict.")

    # 2. Add Counts
    for key, value in source.items():
        count = int(value)
        if count < 0:
            raise ValueError("Diagnostic counts must be nonnegative.")
        target[str(key)] += count

    # 3. Return Updated Counter
    return {"error_code": 0, "counter": target}


class ValhallaMeiliBaselineModel(BaselineModel):
    """Point-aligned Docker-backed Valhalla Meili baseline."""

    requires_map = True
    requires_calibration = False

    def __init__(self, *, dataset_name: str, config: dict) -> None:
        """
        Purpose:
            Construct an uninitialized Valhalla Meili baseline adapter.
        Parameters:
            dataset_name (str), explicit processed dataset identifier.
            config (dict), complete validated Valhalla configuration.
        Return Dict:
            Constructor; no return packet.
        Usage:
            baseline.registry creates this adapter for valhalla_meili jobs.
        TODO:
            1) Validate configuration.
            2) Initialize the shared baseline contract.
            3) Create an empty diagnostic summary.
        """

        # 1. Validate Configuration
        validated = validate_valhalla_config(config)
        self.config = validated["config"]

        # 2. Initialize Shared Contract
        super().__init__(
            method_name="valhalla_meili",
            dataset_name=str(dataset_name),
            use_timestamps=True,
        )

        # 3. Create Empty Diagnostic Summary
        self.last_prediction_summary: dict = {}
        self._service_started = False
        self._service_pid: int | None = None

    def initialize(self, calibration_file: str | None = None) -> dict:
        """
        Purpose:
            Start and verify the dataset-specific Docker Valhalla service.
        Parameters:
            calibration_file (str | None), must be None; Meili is not calibrated.
        Return Dict:
            "error_code": int, 0 when the Docker service is ready.
            "status": str, "ok" on success.
            "service": dict, Docker lifecycle and status details.
        Usage:
            baseline.registry calls initialize outside timed prediction.
        TODO:
            1) Reject calibration artifacts.
            2) Start or verify the Docker service.
            3) Return initialization telemetry.
        """

        # 1. Reject Calibration Artifacts
        if calibration_file is not None:
            raise ValueError("valhalla_meili does not accept calibration_file.")

        # 2. Start Or Verify Docker Service
        from .service import ensure_valhalla_service

        service_packet = ensure_valhalla_service(self.config)
        if int(service_packet["error_code"]) != 0:
            raise RuntimeError(str(service_packet["message"]))
        self._service_started = True
        self._service_pid = int(service_packet["container_pid"])

        # 3. Return Initialization Telemetry
        self.calibration_summary = {
            "error_code": 0,
            "status": "ok",
            "mode": "docker_map_matching",
            "service": service_packet,
        }
        return self.calibration_summary

    def resource_usage_roots(self) -> dict:
        """
        Purpose:
            Expose the initialized Valhalla container process for RSS sampling.
        Parameters:
            None.
        Return Dict:
            "error_code": int, 0 when the service PID is available.
            "pids": list[int], host PIDs whose process trees belong to this model.
        Usage:
            Benchmark RSS monitors include the Docker service after initialization.
        TODO:
            1) Require an initialized service PID.
            2) Return the PID as an explicit process-tree root.
        """

        # 1. Require Initialized Service PID
        if not self._service_started or self._service_pid is None:
            raise RuntimeError("Valhalla resource roots requested before initialization.")

        # 2. Return Explicit Process Root
        return {"error_code": 0, "pids": [int(self._service_pid)]}

    def deconst(self) -> dict:
        """
        Purpose:
            Dispose the per-testing-item Valhalla Docker service.
        Parameters:
            None.
        Return Dict:
            "error_code": int, 0 after cleanup or when already disposed.
            "status": str, "disposed" or "already_disposed".
            "service": dict | None, Docker cleanup packet.
        Usage:
            Evaluator finally blocks call this after timing and RSS sampling end.
        TODO:
            1) Return when this instance does not own a running service.
            2) Stop and remove the deterministic Compose project.
            3) Clear ownership state and fail on incomplete cleanup.
        """

        # 1. Return Without Owned Service
        if not self._service_started:
            return {
                "error_code": 0,
                "status": "already_disposed",
                "service": None,
            }

        # 2. Stop And Remove Compose Project
        from .service import stop_valhalla_service

        stop_packet = stop_valhalla_service(self.config)

        # 3. Clear Ownership And Validate Cleanup
        self._service_started = False
        self._service_pid = None
        if int(stop_packet["error_code"]) != 0:
            raise RuntimeError(str(stop_packet["message"]))
        return {
            "error_code": 0,
            "status": "disposed",
            "service": stop_packet,
        }

    def predict_packet(self, data_seq: np.ndarray) -> dict:
        """
        Purpose:
            Map-match a trajectory through fixed overlapping HTTP requests.
        Parameters:
            data_seq (np.ndarray), shape (N,2|3), [lat, lon, optional timestamp].
        Return Dict:
            "error_code": int, 0 complete, 1 partial, 2 rejected.
            "positions_latlon": np.ndarray, shape (N,2), Meili positions with
                rejected points preserved at their raw noisy coordinates.
            "accepted_mask": np.ndarray, shape (N,), boolean.
            "complete": bool.
            "diagnostics": dict, request, status, error, and coverage counts.
        Usage:
            Baseline evaluators call this method to preserve rejection evidence.
        TODO:
            1) Normalize and validate the trajectory.
            2) Build deterministic overlapping windows.
            3) Request one point-aligned match per window.
            4) Select the most central accepted result for overlap points.
            5) Preserve rejected points at their raw noisy coordinates.
            6) Aggregate strict coverage and error diagnostics.
        """

        # 1. Normalize And Validate Trajectory
        seq = ensure_lat_lon_timestamp_sequence(data_seq)
        n_points = int(seq.shape[0])
        if n_points < 2:
            positions = np.asarray(seq[:, :2], dtype=float).copy()
            accepted = np.zeros(n_points, dtype=bool)
            diagnostics = {
                "attempted_requests": 0,
                "accepted_requests": 0,
                "rejected_requests": 0,
                "request_rejection_rate": 0.0,
                "http_status_counts": {},
                "valhalla_error_code_counts": {},
                "adapter_error_code_counts": {"-4": 1},
                "transport_error_counts": {},
                "point_type_counts": {},
                "attempted_points": n_points,
                "accepted_points": 0,
                "rejected_points": n_points,
                "fallback_points": n_points,
                "fallback_policy": "raw_input",
                "point_rejection_rate": 1.0 if n_points else 0.0,
                "request_records": [],
            }
            self.last_prediction_summary = diagnostics
            return {
                "error_code": 2,
                "positions_latlon": positions,
                "accepted_mask": accepted,
                "complete": False,
                "diagnostics": diagnostics,
            }

        # 2. Build Deterministic Overlapping Windows
        schedule = build_request_windows(
            n_points,
            self.config["window_points"],
            self.config["overlap_points"],
        )
        windows = schedule["windows"]
        positions = np.asarray(seq[:, :2], dtype=float).copy()
        accepted = np.zeros(n_points, dtype=bool)
        selected_margin = np.full(n_points, -1, dtype=int)
        http_status_counts: Counter[str] = Counter()
        valhalla_error_counts: Counter[str] = Counter()
        adapter_error_counts: Counter[str] = Counter()
        transport_error_counts: Counter[str] = Counter()
        point_type_counts: Counter[str] = Counter()
        request_records: list[dict] = []

        # 3. Request One Point-Aligned Match Per Window
        accepted_requests = 0
        for request_index, window in enumerate(windows):
            start = int(window["start"])
            end = int(window["end"])
            packet = request_trace_attributes(
                seq[start:end],
                base_url=self.config["base_url"],
                costing=self.config["costing"],
                shape_match=self.config["shape_match"],
                timeout_sec=self.config["timeout_sec"],
            )
            http_status = packet["http_status"]
            http_status_counts[str(http_status) if http_status is not None else "transport"] += 1
            packet_error_code = int(packet["error_code"])
            valhalla_error_code = packet["valhalla_error_code"]
            adapter_error_code = int(packet["adapter_error_code"])
            if valhalla_error_code is not None:
                valhalla_error_counts[str(int(valhalla_error_code))] += 1
            if adapter_error_code != 0:
                adapter_error_counts[str(adapter_error_code)] += 1
            transport_error = packet["diagnostics"].get("transport_error")
            if transport_error:
                transport_error_counts[str(transport_error)] += 1
            merge_counter(point_type_counts, packet["diagnostics"].get("point_type_counts", {}))
            window_mask = np.asarray(packet["accepted_mask"], dtype=bool)
            window_positions = np.asarray(packet["positions_latlon"], dtype=float)
            request_complete = bool(window_mask.size == end - start and np.all(window_mask))
            if request_complete:
                accepted_requests += 1
            request_records.append(
                {
                    "request_index": int(request_index),
                    "start": start,
                    "end": end,
                    "n_points": end - start,
                    "http_status": http_status,
                    "error_code": packet_error_code,
                    "valhalla_error_code": valhalla_error_code,
                    "adapter_error_code": adapter_error_code,
                    "accepted_points": int(np.count_nonzero(window_mask)),
                    "unmatched_points": int(packet["diagnostics"]["unmatched_points"]),
                    "discontinuity_points": int(packet["diagnostics"]["discontinuity_points"]),
                    "invalid_response": bool(packet["diagnostics"]["invalid_response"]),
                }
            )

            # 4. Select Most Central Accepted Results In Overlaps
            for local_index in range(end - start):
                if not bool(window_mask[local_index]):
                    continue
                global_index = start + local_index
                margin = min(local_index, (end - start - 1) - local_index)
                if margin <= int(selected_margin[global_index]):
                    continue
                positions[global_index] = window_positions[local_index]
                accepted[global_index] = True
                selected_margin[global_index] = int(margin)

        # 5. Preserve Rejected Points At Raw Noisy Coordinates
        if not bool(np.all(np.isfinite(positions))):
            raise RuntimeError("Valhalla raw-fallback output contains non-finite coordinates.")

        # 6. Aggregate Coverage And Error Diagnostics
        accepted_points = int(np.count_nonzero(accepted))
        rejected_points = n_points - accepted_points
        complete = rejected_points == 0
        rejected_requests = len(windows) - accepted_requests
        diagnostics = {
            "attempted_requests": int(len(windows)),
            "accepted_requests": int(accepted_requests),
            "rejected_requests": int(rejected_requests),
            "request_rejection_rate": (
                float(rejected_requests) / float(len(windows)) if windows else 0.0
            ),
            "http_status_counts": dict(sorted(http_status_counts.items())),
            "valhalla_error_code_counts": dict(sorted(valhalla_error_counts.items())),
            "adapter_error_code_counts": dict(sorted(adapter_error_counts.items())),
            "transport_error_counts": dict(sorted(transport_error_counts.items())),
            "point_type_counts": dict(sorted(point_type_counts.items())),
            "attempted_points": n_points,
            "accepted_points": accepted_points,
            "rejected_points": rejected_points,
            "fallback_points": rejected_points,
            "fallback_policy": "raw_input",
            "point_rejection_rate": float(rejected_points) / float(n_points),
            "request_records": request_records,
        }
        self.last_prediction_summary = diagnostics
        outcome_code = 0 if complete else (1 if accepted_points > 0 else 2)
        return {
            "error_code": outcome_code,
            "positions_latlon": positions,
            "accepted_mask": accepted,
            "complete": complete,
            "diagnostics": diagnostics,
        }

    def _predict_block(self, seq_latlon_t: np.ndarray) -> np.ndarray:
        """
        Purpose:
            Preserve the legacy ndarray prediction contract for direct callers.
        Parameters:
            seq_latlon_t (np.ndarray), shape (N,3), [lat, lon, timestamp].
        Return Dict:
            Legacy override returns np.ndarray because BaselineModel requires it.
        Usage:
            BaselineModel.predict delegates to this method. Evaluation code uses
            predict_packet so rejection masks are not lost.
        TODO:
            1) Run the structured prediction.
            2) Return its coordinate payload unchanged.
        """

        # 1. Run Structured Prediction
        packet = self.predict_packet(seq_latlon_t)

        # 2. Return Coordinate Payload
        return np.asarray(packet["positions_latlon"], dtype=float)


__all__ = [
    "ValhallaMeiliBaselineModel",
    "build_request_windows",
    "merge_counter",
    "validate_valhalla_config",
]
