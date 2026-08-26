from __future__ import annotations

import logging
from collections import Counter

import numpy as np
import requests


LOGGER = logging.getLogger(__name__)

MATCHED_POINT_ATTRIBUTES = [
    "edge.id",
    "matched.point",
    "matched.type",
    "matched.edge_index",
    "matched.begin_route_discontinuity",
    "matched.end_route_discontinuity",
    "matched.distance_along_edge",
    "matched.distance_from_trace_point",
]


def build_trace_attributes_payload(
    seq_latlon_t: np.ndarray,
    *,
    costing: str,
    shape_match: str,
) -> dict:
    """
    Purpose:
        Build one Valhalla trace_attributes JSON request packet.
    Parameters:
        seq_latlon_t (np.ndarray), shape (N,3), [lat, lon, timestamp].
        costing (str), explicit Valhalla costing model.
        shape_match (str), must be "map_snap" for benchmark GPS traces.
    Return Dict:
        "error_code": int, 0 for a valid packet.
        "payload": dict, JSON-serializable HTTP request body.
    Usage:
        request_trace_attributes calls this before making an HTTP request.
    TODO:
        1) Validate coordinates and configuration.
        2) Serialize shape points.
        3) Attach the minimal matched-point filter.
    """

    # 1. Validate Coordinates And Configuration
    seq = np.asarray(seq_latlon_t, dtype=float)
    if seq.ndim != 2 or seq.shape[1] != 3 or seq.shape[0] < 2:
        raise ValueError("Valhalla trace input must have shape (N,3) with N >= 2.")
    if not np.isfinite(seq).all():
        raise ValueError("Valhalla trace coordinates and timestamps must be finite.")
    if np.any(seq[:, 0] < -90.0) or np.any(seq[:, 0] > 90.0):
        raise ValueError("Valhalla trace latitudes must be within [-90, 90].")
    if np.any(seq[:, 1] < -180.0) or np.any(seq[:, 1] > 180.0):
        raise ValueError("Valhalla trace longitudes must be within [-180, 180].")
    if not str(costing).strip():
        raise ValueError("Valhalla costing must be explicit and non-empty.")
    if str(shape_match).strip() != "map_snap":
        raise ValueError("Published Valhalla evaluation requires shape_match='map_snap'.")

    # 2. Serialize Shape Points
    shape = [
        {
            "lat": float(row[0]),
            "lon": float(row[1]),
            "time": int(round(float(row[2]))),
        }
        for row in seq
    ]

    # 3. Attach Matched-Point Filter
    payload = {
        "shape": shape,
        "costing": str(costing).strip(),
        "shape_match": "map_snap",
        "filters": {
            "action": "include",
            "attributes": list(MATCHED_POINT_ATTRIBUTES),
        },
    }
    return {"error_code": 0, "payload": payload}


def parse_trace_attributes_response(response_payload: dict, expected_points: int) -> dict:
    """
    Purpose:
        Convert a successful trace_attributes response into point-aligned data.
    Parameters:
        response_payload (dict), parsed Valhalla JSON response.
        expected_points (int), number of input points in the request window.
    Return Dict:
        "error_code": int, 0 for a structurally valid response.
        "positions_latlon": np.ndarray, shape (N,2), NaN for rejected points.
        "accepted_mask": np.ndarray, shape (N,), boolean.
        "point_type_counts": dict[str,int].
        "unmatched_points": int.
        "discontinuity_points": int.
        "invalid_response": bool.
    Usage:
        request_trace_attributes parses HTTP 200 response bodies with this function.
    TODO:
        1) Validate response structure and cardinality.
        2) Parse matched coordinates and point types.
        3) Reject unmatched, discontinuous, or edge-less results.
        4) Return aligned arrays and diagnostic counts.
    """

    # 1. Validate Response Structure And Cardinality
    if not isinstance(response_payload, dict):
        raise TypeError("Valhalla response payload must be a JSON object.")
    matched_points = response_payload.get("matched_points")
    if not isinstance(matched_points, list) or len(matched_points) != int(expected_points):
        return {
            "error_code": 1,
            "positions_latlon": np.full((int(expected_points), 2), np.nan, dtype=float),
            "accepted_mask": np.zeros(int(expected_points), dtype=bool),
            "point_type_counts": {},
            "unmatched_points": int(expected_points),
            "discontinuity_points": 0,
            "invalid_response": True,
        }

    # 2. Parse Matched Coordinates And Point Types
    positions = np.full((int(expected_points), 2), np.nan, dtype=float)
    accepted = np.zeros(int(expected_points), dtype=bool)
    type_counts: Counter[str] = Counter()
    discontinuity_points = 0
    edges = response_payload.get("edges")
    edges_are_valid = isinstance(edges, list) and len(edges) > 0

    for index, item in enumerate(matched_points):
        if not isinstance(item, dict):
            type_counts["invalid"] += 1
            continue
        point_type = str(item.get("type", "invalid")).strip().lower()
        type_counts[point_type] += 1
        has_discontinuity = bool(item.get("begin_route_discontinuity", False)) or bool(
            item.get("end_route_discontinuity", False)
        )
        if has_discontinuity:
            discontinuity_points += 1
        edge_index = item.get("edge_index")
        edge_is_valid = isinstance(edge_index, int) and edges_are_valid and 0 <= edge_index < len(edges)
        lat_value = item.get("lat")
        lon_value = item.get("lon")
        coordinate_is_valid = (
            isinstance(lat_value, (int, float))
            and isinstance(lon_value, (int, float))
            and bool(np.isfinite(float(lat_value)))
            and bool(np.isfinite(float(lon_value)))
            and -90.0 <= float(lat_value) <= 90.0
            and -180.0 <= float(lon_value) <= 180.0
        )
        point_is_accepted = (
            point_type in {"matched", "interpolated"}
            and not has_discontinuity
            and edge_is_valid
            and coordinate_is_valid
        )
        if not point_is_accepted:
            continue
        positions[index] = [float(item["lat"]), float(item["lon"])]
        accepted[index] = True

    # 3. Reject Edge-Less Results
    invalid_response = not edges_are_valid
    if invalid_response:
        positions[:] = np.nan
        accepted[:] = False

    # 4. Return Aligned Packet
    unmatched_points = int(expected_points) - int(np.count_nonzero(accepted))
    return {
        "error_code": 0 if not invalid_response else 2,
        "positions_latlon": positions,
        "accepted_mask": accepted,
        "point_type_counts": dict(sorted(type_counts.items())),
        "unmatched_points": unmatched_points,
        "discontinuity_points": int(discontinuity_points),
        "invalid_response": bool(invalid_response),
    }


def request_trace_attributes(
    seq_latlon_t: np.ndarray,
    *,
    base_url: str,
    costing: str,
    shape_match: str,
    timeout_sec: float,
) -> dict:
    """
    Purpose:
        Send one HTTP JSON request to Docker-hosted Valhalla Meili.
    Parameters:
        seq_latlon_t (np.ndarray), shape (N,3), one request window.
        base_url (str), local Valhalla service origin.
        costing (str), explicit Valhalla costing model.
        shape_match (str), must be "map_snap".
        timeout_sec (float), positive HTTP timeout in seconds.
    Return Dict:
        "error_code": int, Valhalla code, 0 success, -1 transport, -2 invalid JSON.
        "valhalla_error_code": int | None, code returned by Valhalla itself.
        "adapter_error_code": int, local transport/parse/alignment code.
        "http_status": int | None.
        "positions_latlon": np.ndarray, shape (N,2).
        "accepted_mask": np.ndarray, shape (N,), boolean.
        "diagnostics": dict, status and response counters.
    Usage:
        ValhallaMeiliBaselineModel calls this once per deterministic window.
    TODO:
        1) Build and validate the request body.
        2) Send the HTTP request.
        3) Parse deterministic Valhalla errors.
        4) Parse successful matched points.
    """

    # 1. Build And Validate Request Body
    if float(timeout_sec) <= 0.0:
        raise ValueError("timeout_sec must be positive.")
    request_packet = build_trace_attributes_payload(
        seq_latlon_t,
        costing=costing,
        shape_match=shape_match,
    )
    payload = request_packet["payload"]
    url = str(base_url).rstrip("/") + "/trace_attributes"
    n_points = int(np.asarray(seq_latlon_t).shape[0])
    empty_positions = np.full((n_points, 2), np.nan, dtype=float)
    empty_mask = np.zeros(n_points, dtype=bool)

    # 2. Send HTTP Request
    try:
        response = requests.post(url, json=payload, timeout=float(timeout_sec))
    except requests.RequestException as exc:
        LOGGER.warning("Valhalla transport failure: %s", exc)
        return {
            "error_code": -1,
            "valhalla_error_code": None,
            "adapter_error_code": -1,
            "http_status": None,
            "positions_latlon": empty_positions,
            "accepted_mask": empty_mask,
            "diagnostics": {
                "transport_error": exc.__class__.__name__,
                "point_type_counts": {},
                "unmatched_points": n_points,
                "discontinuity_points": 0,
                "invalid_response": False,
            },
        }

    # 3. Parse Deterministic Valhalla Errors
    try:
        response_payload = response.json()
    except requests.JSONDecodeError:
        return {
            "error_code": -2,
            "valhalla_error_code": None,
            "adapter_error_code": -2,
            "http_status": int(response.status_code),
            "positions_latlon": empty_positions,
            "accepted_mask": empty_mask,
            "diagnostics": {
                "transport_error": None,
                "point_type_counts": {},
                "unmatched_points": n_points,
                "discontinuity_points": 0,
                "invalid_response": True,
            },
        }
    if int(response.status_code) != 200:
        valhalla_code = response_payload.get("error_code")
        if not isinstance(valhalla_code, int):
            valhalla_code = -3
        return {
            "error_code": int(valhalla_code),
            "valhalla_error_code": int(valhalla_code),
            "adapter_error_code": 0,
            "http_status": int(response.status_code),
            "positions_latlon": empty_positions,
            "accepted_mask": empty_mask,
            "diagnostics": {
                "transport_error": None,
                "valhalla_error": str(response_payload.get("error", "")),
                "point_type_counts": {},
                "unmatched_points": n_points,
                "discontinuity_points": 0,
                "invalid_response": False,
            },
        }

    # 4. Parse Successful Matched Points
    parsed = parse_trace_attributes_response(response_payload, n_points)
    return {
        "error_code": int(parsed["error_code"]),
        "valhalla_error_code": None,
        "adapter_error_code": int(parsed["error_code"]),
        "http_status": int(response.status_code),
        "positions_latlon": parsed["positions_latlon"],
        "accepted_mask": parsed["accepted_mask"],
        "diagnostics": {
            "transport_error": None,
            "point_type_counts": parsed["point_type_counts"],
            "unmatched_points": int(parsed["unmatched_points"]),
            "discontinuity_points": int(parsed["discontinuity_points"]),
            "invalid_response": bool(parsed["invalid_response"]),
        },
    }
