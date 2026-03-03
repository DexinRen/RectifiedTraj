from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests

from ...base import BaselineModel


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_token(text: str | None) -> str:
    s = str(text or "default").strip().lower()
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    token = "".join(out).strip("_")
    return token or "default"


def _normalize_base_url(raw: str) -> str:
    url = str(raw).strip()
    if "://" not in url:
        url = f"http://{url}"
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid Valhalla URL: {raw}")
    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")


def _extract_lat_lon(point: dict) -> tuple[float, float] | None:
    if not isinstance(point, dict):
        return None
    key_pairs = [
        ("lat", "lon"),
        ("lat", "lng"),
        ("latitude", "longitude"),
        ("y", "x"),
    ]
    for lat_k, lon_k in key_pairs:
        lat = point.get(lat_k)
        lon = point.get(lon_k)
        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except Exception:
            continue
        if np.isfinite(lat_f) and np.isfinite(lon_f):
            return lat_f, lon_f

    ll = point.get("ll")
    if isinstance(ll, dict):
        lat = ll.get("lat")
        lon = ll.get("lon", ll.get("lng"))
        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except Exception:
            return None
        if np.isfinite(lat_f) and np.isfinite(lon_f):
            return lat_f, lon_f
    return None


class ValhallaMeiliBaselineModel(BaselineModel):
    requires_map = True

    def __init__(self, *, dataset_name: str | None = None, use_timestamps: bool = True) -> None:
        max_points = max(1, _env_int("VALHALLA_MAX_POINTS", 1024))
        overlap = max(0, _env_int("VALHALLA_CHUNK_OVERLAP", 32))
        super().__init__(
            method_name="valhalla_meili",
            dataset_name=dataset_name,
            use_timestamps=use_timestamps,
            max_predict_points=max_points,
            chunk_overlap=overlap,
        )
        self._logger = logging.getLogger(__name__)
        self._session = requests.Session()

        self._auto_start = _env_bool("VALHALLA_AUTO_START", True)
        self._auto_stop = _env_bool("VALHALLA_AUTO_STOP", True)
        self._startup_timeout_sec = _env_float("VALHALLA_STARTUP_TIMEOUT_SEC", 900.0)
        self._startup_poll_sec = max(0.5, _env_float("VALHALLA_STARTUP_POLL_SEC", 2.0))
        self._request_timeout_sec = max(1.0, _env_float("VALHALLA_REQUEST_TIMEOUT_SEC", 30.0))

        self._costing = os.getenv("VALHALLA_COSTING", "auto").strip() or "auto"
        self._shape_match = os.getenv("VALHALLA_SHAPE_MATCH", "map_snap").strip() or "map_snap"
        self._search_radius = _env_float("VALHALLA_SEARCH_RADIUS", 50.0)
        self._gps_accuracy = _env_float("VALHALLA_GPS_ACCURACY", 5.0)
        self._breakage_distance = _env_float("VALHALLA_BREAKAGE_DISTANCE", 2000.0)
        self._send_timestamps = _env_bool("VALHALLA_SEND_TIMESTAMPS", bool(self.use_timestamps))
        self._retry_relaxed = _env_bool("VALHALLA_RETRY_RELAXED", True)
        # Defaults are aligned to common valhalla.json service_limits.trace values.
        self._trace_limit_search_radius = 100.0
        self._trace_limit_gps_accuracy = 100.0
        self._trace_limit_breakage_distance = 2000.0
        # Segment long/discontinuous traces so one bad span does not fail the whole trajectory.
        self._segment_max_path_m = max(0.0, _env_float("VALHALLA_SEGMENT_MAX_PATH_M", 60000.0))
        self._segment_max_step_m = max(0.0, _env_float("VALHALLA_SEGMENT_MAX_STEP_M", 2000.0))
        self._split_on_failure = _env_bool("VALHALLA_SPLIT_ON_FAILURE", True)
        self._split_min_points = max(8, _env_int("VALHALLA_SPLIT_MIN_POINTS", 16))
        self._split_max_depth = max(0, _env_int("VALHALLA_SPLIT_MAX_DEPTH", 6))

        self._docker_image = (
            os.getenv("VALHALLA_DOCKER_IMAGE", "ghcr.io/valhalla/valhalla-scripted:latest").strip()
            or "ghcr.io/valhalla/valhalla-scripted:latest"
        )
        self._docker_container_port = _env_int("VALHALLA_CONTAINER_PORT", 8002)
        dataset_token = _safe_token(dataset_name)
        self._container_name = (
            os.getenv("VALHALLA_DOCKER_CONTAINER", f"valhalla_meili_{dataset_token}").strip()
            or f"valhalla_meili_{dataset_token}"
        )
        self._docker_extra_args = os.getenv("VALHALLA_DOCKER_EXTRA_ARGS", "")

        raw_base_url = os.getenv("VALHALLA_URL", "").strip()
        if raw_base_url:
            self._base_url = _normalize_base_url(raw_base_url)
        else:
            self._base_url = "http://127.0.0.1:8002"
        parsed = urlparse(self._base_url)
        default_port = parsed.port or 8002
        self._docker_host_port = _env_int("VALHALLA_PORT", default_port)
        if not raw_base_url:
            self._base_url = f"http://127.0.0.1:{self._docker_host_port}"
            parsed = urlparse(self._base_url)
        host = (parsed.hostname or "").strip().lower()
        self._is_local_endpoint = host in {"", "127.0.0.1", "localhost", "0.0.0.0"}
        if self._auto_start and not self._is_local_endpoint:
            self._logger.warning("VALHALLA_URL points to non-local host; disabling docker auto-start.")
            self._auto_start = False

        self._status_url = f"{self._base_url}/status"
        self._trace_url = f"{self._base_url}/trace_attributes"

        default_runtime_root = Path(__file__).resolve().parent / "runtime"
        runtime_root = os.getenv("VALHALLA_RUNTIME_DIR", str(default_runtime_root))
        self._runtime_dir = Path(runtime_root) / dataset_token
        self._started_by_me = False
        self._diag: dict[str, object] = {
            "predict_calls": 0,
            "predict_calls_with_failures": 0,
            "segments_total": 0,
            "segments_succeeded": 0,
            "segments_failed": 0,
            "segments_fallback_points": 0,
            "primary_attempts": 0,
            "primary_success": 0,
            "retry_relaxed_attempts": 0,
            "retry_relaxed_success": 0,
            "retry_relaxed_failures": 0,
            "retry_defaults_attempts": 0,
            "retry_defaults_success": 0,
            "retry_defaults_failures": 0,
            "split_events": 0,
            "split_leaf_failures": 0,
            "split_max_depth_seen": 0,
            "error_code_171_count": 0,
            "error_code_444_count": 0,
            "error_code_154_count": 0,
            "error_code_158_count": 0,
            "error_code_other_count": 0,
            "bounds_error_count": 0,
            "max_distance_error_count": 0,
            "first_error": "",
            "last_error": "",
        }

    def calibrate(
        self,
        calibration_file: str | None = None,
        map_file: str | None = None,
    ) -> dict:
        del calibration_file
        source_map = self._resolve_map_file(map_file)
        if self._is_server_ready():
            if self._runtime_map_matches_source(source_map):
                self._refresh_runtime_trace_limits()
                return {
                    "status": "ok",
                    "server": "already_running",
                    "base_url": self._base_url,
                    "map_file": str(source_map),
                }
            if not self._auto_start:
                raise RuntimeError(
                    "Valhalla service is already running, but runtime map hash does not match the "
                    f"expected sliced map ({source_map}). Enable VALHALLA_AUTO_START or restart "
                    f"container '{self._container_name}' manually."
                )
            self._logger.warning(
                "Valhalla runtime map hash mismatch detected; restarting container '%s' to rebuild tiles.",
                self._container_name,
            )
            action = self._restart_container(source_map)
            self._wait_until_ready()
            self._refresh_runtime_trace_limits()
            return {
                "status": "ok",
                "server": action,
                "base_url": self._base_url,
                "container_name": self._container_name,
                "map_file": str(source_map),
            }

        if not self._auto_start:
            raise RuntimeError(
                f"Valhalla service is not reachable at {self._base_url} and auto-start is disabled."
            )

        action = self._ensure_docker_container(source_map)
        self._wait_until_ready()
        self._refresh_runtime_trace_limits()
        return {
            "status": "ok",
            "server": action,
            "base_url": self._base_url,
            "container_name": self._container_name,
            "map_file": str(source_map),
        }

    @staticmethod
    def _sha256_file(path: Path) -> str | None:
        try:
            h = hashlib.sha256()
            with path.open("rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    h.update(chunk)
            return h.hexdigest().lower()
        except Exception:
            return None

    @staticmethod
    def _extract_hex64_token(text: str) -> str | None:
        for tok in str(text or "").replace("\n", " ").split():
            t = tok.strip().lower()
            if len(t) != 64:
                continue
            if all(ch in "0123456789abcdef" for ch in t):
                return t
        return None

    def _runtime_map_hash(self) -> str | None:
        hashes = self._runtime_dir / "file_hashes.txt"
        if not hashes.exists():
            return None
        try:
            raw = hashes.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None
        return self._extract_hex64_token(raw)

    def _runtime_map_matches_source(self, source_map: Path) -> bool:
        source_hash = self._sha256_file(source_map)
        runtime_hash = self._runtime_map_hash()
        if not source_hash or not runtime_hash:
            return False
        return source_hash == runtime_hash

    @staticmethod
    def _safe_positive_float(raw: object, default: float) -> float:
        try:
            val = float(raw)
        except Exception:
            return float(default)
        if not np.isfinite(val) or val <= 0.0:
            return float(default)
        return float(val)

    @staticmethod
    def _safe_non_negative_float(raw: object, default: float) -> float:
        try:
            val = float(raw)
        except Exception:
            return float(default)
        if not np.isfinite(val):
            return float(default)
        return max(0.0, float(val))

    def _refresh_runtime_trace_limits(self) -> None:
        cfg_path = self._runtime_dir / "valhalla.json"
        if not cfg_path.exists():
            return
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return
        service_trace = (cfg.get("service_limits") or {}).get("trace") or {}
        meili_default = (cfg.get("meili") or {}).get("default") or {}

        search_limit = self._safe_positive_float(
            service_trace.get("max_search_radius", meili_default.get("max_search_radius")),
            self._trace_limit_search_radius,
        )
        gps_limit = self._safe_positive_float(
            service_trace.get("max_gps_accuracy", 100.0),
            self._trace_limit_gps_accuracy,
        )
        breakage_limit = self._safe_positive_float(
            meili_default.get("breakage_distance", self._trace_limit_breakage_distance),
            self._trace_limit_breakage_distance,
        )
        self._trace_limit_search_radius = search_limit
        self._trace_limit_gps_accuracy = gps_limit
        self._trace_limit_breakage_distance = breakage_limit

    def _resolve_map_file(self, map_file: str | None) -> Path:
        if map_file:
            provided = Path(str(map_file))
            if not provided.is_absolute():
                provided = (Path.cwd() / provided).resolve()
            if not provided.exists():
                raise FileNotFoundError(f"Provided Valhalla map_file does not exist: {provided}")
            if not provided.is_file():
                raise FileNotFoundError(f"Provided Valhalla map_file is not a regular file: {provided}")
            if not str(provided.name).lower().endswith(".pbf"):
                raise ValueError(
                    f"Valhalla map_file must be a .pbf file, got: {provided.name}"
                )
            return provided

        ds_name = str(self.dataset_name or "").strip()
        if not ds_name:
            raise RuntimeError(
                "Valhalla Meili requires map_file or dataset_name to resolve a sliced map "
                "(expected map_file=<...>.pbf or dataset/map_processed/map_<dataset>.osm.pbf)."
            )

        expected = (Path.cwd() / "dataset" / "map_processed" / f"map_{ds_name}.osm.pbf").resolve()
        if not expected.exists():
            raise FileNotFoundError(
                f"Sliced map missing for dataset '{ds_name}': {expected}. "
                "Run parquet_processor map slicing first."
            )
        return expected

    def _run_cmd(self, args: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
        proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if check and proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"Command failed: {' '.join(args)} | {err[-400:]}")
        return proc

    def _docker_state(self) -> str:
        proc = self._run_cmd(["docker", "inspect", "-f", "{{.State.Running}}", self._container_name])
        if proc.returncode != 0:
            return "missing"
        out = (proc.stdout or "").strip().lower()
        if out == "true":
            return "running"
        if out == "false":
            return "stopped"
        return "unknown"

    def _stage_map_for_container(self, source_map: Path, *, force: bool = False) -> Path:
        self._runtime_dir.mkdir(parents=True, exist_ok=True)
        staged = self._runtime_dir / "input.osm.pbf"
        same = False
        if staged.exists() and not force:
            try:
                same = (
                    staged.stat().st_size == source_map.stat().st_size
                    and int(staged.stat().st_mtime) >= int(source_map.stat().st_mtime)
                )
            except Exception:
                same = False
        if same:
            return staged

        if staged.exists():
            staged.unlink()
        # Use copy instead of hard-link to avoid aliasing map identities across folders.
        shutil.copy2(source_map, staged)
        return staged

    def _restart_container(self, source_map: Path) -> str:
        if shutil.which("docker") is None:
            raise RuntimeError("docker command not found in PATH.")
        # Force clean re-create so valhalla-scripted rebuilds tiles from current input map.
        self._run_cmd(["docker", "rm", "-f", self._container_name], check=False)
        self._started_by_me = True
        action = self._ensure_docker_container(source_map)
        return f"restarted_container_for_map_refresh:{action}"

    def _ensure_docker_container(self, source_map: Path) -> str:
        if shutil.which("docker") is None:
            raise RuntimeError("docker command not found in PATH.")
        self._stage_map_for_container(source_map)

        state = self._docker_state()
        runtime_match = self._runtime_map_matches_source(source_map)
        if state in {"running", "stopped"} and not runtime_match:
            # Source map changed: force staged map refresh before container recreation.
            self._stage_map_for_container(source_map, force=True)
            self._logger.warning(
                "Valhalla runtime map hash mismatch (or missing hash); recreating container '%s'.",
                self._container_name,
            )
            self._run_cmd(["docker", "rm", "-f", self._container_name], check=False)
            state = "missing"

        if state == "running":
            return "reused_running_container"
        if state == "stopped":
            self._run_cmd(["docker", "start", self._container_name], check=True)
            self._started_by_me = True
            return "started_existing_container"

        args = [
            "docker",
            "run",
            "-d",
            "--name",
            self._container_name,
            "-p",
            f"{self._docker_host_port}:{self._docker_container_port}",
            "-v",
            f"{self._runtime_dir.resolve()}:/custom_files",
        ]
        if self._docker_extra_args.strip():
            args.extend(shlex.split(self._docker_extra_args))
        args.append(self._docker_image)
        self._run_cmd(args, check=True)
        self._started_by_me = True
        return "created_new_container"

    def _is_server_ready(self) -> bool:
        try:
            resp = self._session.get(self._status_url, timeout=self._request_timeout_sec)
            return 200 <= int(resp.status_code) < 300
        except Exception:
            return False

    def _wait_until_ready(self) -> None:
        deadline = time.time() + float(self._startup_timeout_sec)
        while time.time() < deadline:
            if self._is_server_ready():
                return
            time.sleep(self._startup_poll_sec)
        raise RuntimeError(
            f"Valhalla did not become ready at {self._status_url} within {self._startup_timeout_sec:.1f}s."
        )

    def _build_payload(
        self,
        seq_latlon_t: np.ndarray,
        *,
        shape_match: str | None = None,
        search_radius: float | None = None,
        gps_accuracy: float | None = None,
        breakage_distance: float | None = None,
        send_timestamps: bool | None = None,
        include_trace_options: bool = True,
    ) -> dict:
        n = int(seq_latlon_t.shape[0])
        lat = seq_latlon_t[:, 0]
        lon = seq_latlon_t[:, 1]
        ts = seq_latlon_t[:, 2]
        shape_match_mode = str(shape_match or self._shape_match)
        use_time = bool(self._send_timestamps if send_timestamps is None else send_timestamps)
        use_time = bool(use_time and self.use_timestamps)
        if use_time:
            use_time = bool(np.all(np.isfinite(ts)))
            if use_time and n >= 2:
                use_time = bool(np.all(np.diff(ts) >= 0.0))

        shape = []
        t0 = float(ts[0]) if use_time and n > 0 else 0.0
        for i in range(n):
            one = {"lat": float(lat[i]), "lon": float(lon[i])}
            if shape_match_mode == "map_snap":
                one["type"] = "break" if i in {0, n - 1} else "via"
            if use_time:
                one["time"] = max(0, int(round(float(ts[i] - t0))))
            shape.append(one)

        payload = {
            "shape": shape,
            "costing": self._costing,
            "shape_match": shape_match_mode,
        }
        if include_trace_options:
            gps_val = self._safe_non_negative_float(
                self._gps_accuracy if gps_accuracy is None else gps_accuracy,
                self._gps_accuracy,
            )
            search_val = self._safe_non_negative_float(
                self._search_radius if search_radius is None else search_radius,
                self._search_radius,
            )
            breakage_val = self._safe_non_negative_float(
                self._breakage_distance if breakage_distance is None else breakage_distance,
                self._breakage_distance,
            )
            payload["trace_options"] = {
                "gps_accuracy": min(gps_val, float(self._trace_limit_gps_accuracy)),
                "search_radius": min(search_val, float(self._trace_limit_search_radius)),
                "breakage_distance": min(breakage_val, float(self._trace_limit_breakage_distance)),
            }
        return payload

    def _post_trace(self, payload: dict) -> dict:
        resp = self._session.post(
            self._trace_url,
            json=payload,
            timeout=self._request_timeout_sec,
        )
        if 200 <= int(resp.status_code) < 300:
            return resp.json()
        text = (resp.text or "").strip()
        raise RuntimeError(f"HTTP {resp.status_code}: {text[-400:]}")

    def _extract_matched_points(self, response: dict, seq_latlon_t: np.ndarray) -> np.ndarray:
        n = int(seq_latlon_t.shape[0])
        points = None
        for key in ("matched_points", "tracepoints", "shape"):
            val = response.get(key)
            if isinstance(val, list):
                points = val
                break
        if points is None and isinstance(response.get("trip"), dict):
            val = response["trip"].get("matched_points")
            if isinstance(val, list):
                points = val

        if not points:
            raise RuntimeError(f"Valhalla response missing matched points keys: {sorted(response.keys())}")

        out = np.empty((n, 2), dtype=float)
        out[:, 0] = seq_latlon_t[:, 0]
        out[:, 1] = seq_latlon_t[:, 1]

        limit = min(n, len(points))
        for i in range(limit):
            ll = _extract_lat_lon(points[i])
            if ll is None:
                continue
            out[i, 0] = ll[0]
            out[i, 1] = ll[1]
        return out

    @staticmethod
    def _is_trace_option_bounds_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return "error_code\":158" in text or "input trace option is out of bounds" in text

    @staticmethod
    def _is_trace_max_distance_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return "error_code\":154" in text or "max distance limit" in text

    @staticmethod
    def _extract_error_code(exc: Exception) -> int | None:
        text = str(exc)
        m = re.search(r'"error_code"\s*:\s*(\d+)', text)
        if m is None:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def _record_error(self, exc: Exception) -> None:
        text = str(exc)
        if not self._diag.get("first_error"):
            self._diag["first_error"] = text
        self._diag["last_error"] = text

        code = self._extract_error_code(exc)
        if code == 171:
            self._diag["error_code_171_count"] = int(self._diag["error_code_171_count"]) + 1
        elif code == 444:
            self._diag["error_code_444_count"] = int(self._diag["error_code_444_count"]) + 1
        elif code == 154:
            self._diag["error_code_154_count"] = int(self._diag["error_code_154_count"]) + 1
        elif code == 158:
            self._diag["error_code_158_count"] = int(self._diag["error_code_158_count"]) + 1
        else:
            self._diag["error_code_other_count"] = int(self._diag["error_code_other_count"]) + 1

        if self._is_trace_option_bounds_error(exc):
            self._diag["bounds_error_count"] = int(self._diag["bounds_error_count"]) + 1
        if self._is_trace_max_distance_error(exc):
            self._diag["max_distance_error_count"] = int(self._diag["max_distance_error_count"]) + 1

    @staticmethod
    def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6371000.0
        p1 = np.deg2rad(float(lat1))
        p2 = np.deg2rad(float(lat2))
        dp = p2 - p1
        dl = np.deg2rad(float(lon2) - float(lon1))
        a = np.sin(dp * 0.5) ** 2 + np.cos(p1) * np.cos(p2) * (np.sin(dl * 0.5) ** 2)
        c = 2.0 * np.arcsin(np.sqrt(a))
        return float(r * c)

    def _segment_ranges(self, seq_latlon_t: np.ndarray) -> list[tuple[int, int]]:
        n = int(seq_latlon_t.shape[0])
        if n <= 1:
            return [(0, n)]

        max_path = float(self._segment_max_path_m)
        max_step = float(self._segment_max_step_m)
        ranges: list[tuple[int, int]] = []
        start = 0
        acc = 0.0
        lat = seq_latlon_t[:, 0]
        lon = seq_latlon_t[:, 1]

        for i in range(1, n):
            step_m = self._haversine_m(float(lat[i - 1]), float(lon[i - 1]), float(lat[i]), float(lon[i]))
            seg_len = i - start
            cut_for_step = bool(max_step > 0.0 and step_m > max_step and seg_len >= 1)
            cut_for_path = bool(max_path > 0.0 and (acc + step_m) > max_path and seg_len >= 2)
            if cut_for_step or cut_for_path:
                ranges.append((start, i))
                start = i
                acc = 0.0
                continue
            acc += step_m

        ranges.append((start, n))
        return ranges

    def _predict_noisy_passthrough(self, seq_latlon_t: np.ndarray) -> np.ndarray:
        return np.stack([seq_latlon_t[:, 0], seq_latlon_t[:, 1]], axis=1)

    def _match_with_retries(self, seq_latlon_t: np.ndarray) -> np.ndarray:
        payload = self._build_payload(seq_latlon_t)
        errors: list[str] = []
        self._diag["primary_attempts"] = int(self._diag["primary_attempts"]) + 1
        try:
            data = self._post_trace(payload)
            self._diag["primary_success"] = int(self._diag["primary_success"]) + 1
            return self._extract_matched_points(data, seq_latlon_t)
        except Exception as exc:
            self._record_error(exc)
            errors.append(f"map_snap={exc}")

        if not self._retry_relaxed:
            raise RuntimeError("; ".join(errors))

        self._diag["retry_relaxed_attempts"] = int(self._diag["retry_relaxed_attempts"]) + 1
        try:
            relaxed_search_radius = float(self._trace_limit_search_radius)
            relaxed_gps_accuracy = min(
                float(self._trace_limit_gps_accuracy),
                max(20.0, self._safe_non_negative_float(self._gps_accuracy, 20.0)),
            )
            relaxed_breakage_distance = float(self._trace_limit_breakage_distance)
            relaxed = self._build_payload(
                seq_latlon_t,
                shape_match="walk_or_snap",
                search_radius=relaxed_search_radius,
                gps_accuracy=relaxed_gps_accuracy,
                breakage_distance=relaxed_breakage_distance,
                send_timestamps=False,
            )
            data2 = self._post_trace(relaxed)
            self._diag["retry_relaxed_success"] = int(self._diag["retry_relaxed_success"]) + 1
            return self._extract_matched_points(data2, seq_latlon_t)
        except Exception as exc2:
            self._record_error(exc2)
            self._diag["retry_relaxed_failures"] = int(self._diag["retry_relaxed_failures"]) + 1
            errors.append(f"walk_or_snap={exc2}")
            if self._is_trace_option_bounds_error(exc2):
                self._diag["retry_defaults_attempts"] = int(self._diag["retry_defaults_attempts"]) + 1
                try:
                    defaults_retry = self._build_payload(
                        seq_latlon_t,
                        shape_match="walk_or_snap",
                        send_timestamps=False,
                        include_trace_options=False,
                    )
                    data3 = self._post_trace(defaults_retry)
                    self._diag["retry_defaults_success"] = int(self._diag["retry_defaults_success"]) + 1
                    return self._extract_matched_points(data3, seq_latlon_t)
                except Exception as exc3:
                    self._record_error(exc3)
                    self._diag["retry_defaults_failures"] = int(self._diag["retry_defaults_failures"]) + 1
                    errors.append(f"walk_or_snap_defaults={exc3}")

        raise RuntimeError("; ".join(errors))

    def _predict_segment_with_split(self, seq_latlon_t: np.ndarray, depth: int = 0) -> np.ndarray:
        self._diag["split_max_depth_seen"] = max(int(self._diag["split_max_depth_seen"]), int(depth))
        n = int(seq_latlon_t.shape[0])
        if n <= 1:
            return self._predict_noisy_passthrough(seq_latlon_t)

        try:
            return self._match_with_retries(seq_latlon_t)
        except Exception as exc:
            can_split = (
                bool(self._split_on_failure)
                and depth < int(self._split_max_depth)
                and n >= int(self._split_min_points) * 2
            )
            if can_split:
                self._diag["split_events"] = int(self._diag["split_events"]) + 1
                mid = n // 2
                left = self._predict_segment_with_split(seq_latlon_t[:mid], depth=depth + 1)
                right = self._predict_segment_with_split(seq_latlon_t[mid:], depth=depth + 1)
                return np.concatenate([left, right], axis=0)
            self._diag["split_leaf_failures"] = int(self._diag["split_leaf_failures"]) + 1
            raise exc

    def _predict_block(self, seq_latlon_t: np.ndarray) -> np.ndarray:
        if not self._is_server_ready():
            raise RuntimeError(f"Valhalla service is not ready at {self._base_url}.")
        self._diag["predict_calls"] = int(self._diag["predict_calls"]) + 1
        ranges = self._segment_ranges(seq_latlon_t)
        self._diag["segments_total"] = int(self._diag["segments_total"]) + int(len(ranges))
        out = np.empty((int(seq_latlon_t.shape[0]), 2), dtype=float)
        failures = 0
        first_error = ""
        for start, end in ranges:
            seg = seq_latlon_t[start:end]
            try:
                pred = self._predict_segment_with_split(seg, depth=0)
                out[start:end] = pred
                self._diag["segments_succeeded"] = int(self._diag["segments_succeeded"]) + 1
            except Exception as exc:
                failures += 1
                self._diag["segments_failed"] = int(self._diag["segments_failed"]) + 1
                self._diag["segments_fallback_points"] = int(self._diag["segments_fallback_points"]) + int(
                    max(0, end - start)
                )
                if not first_error:
                    first_error = str(exc)
                out[start:end] = self._predict_noisy_passthrough(seg)
        if failures > 0:
            self._diag["predict_calls_with_failures"] = int(self._diag["predict_calls_with_failures"]) + 1
            self._logger.warning(
                "Valhalla map-matching failed on %d/%d segments (fallback to noisy). first_error=%s",
                failures,
                len(ranges),
                first_error,
            )
        return out

    def diagnostics_snapshot(self) -> dict[str, object]:
        out: dict[str, object] = {
            "valhalla_base_url": self._base_url,
            "valhalla_container_name": self._container_name,
            "valhalla_costing": self._costing,
            "valhalla_shape_match": self._shape_match,
            "valhalla_retry_relaxed_enabled": bool(self._retry_relaxed),
            "valhalla_send_timestamps_enabled": bool(self._send_timestamps and self.use_timestamps),
            "valhalla_config_search_radius": float(self._search_radius),
            "valhalla_config_gps_accuracy": float(self._gps_accuracy),
            "valhalla_config_breakage_distance": float(self._breakage_distance),
            "valhalla_limit_search_radius": float(self._trace_limit_search_radius),
            "valhalla_limit_gps_accuracy": float(self._trace_limit_gps_accuracy),
            "valhalla_limit_breakage_distance": float(self._trace_limit_breakage_distance),
            "valhalla_segment_max_path_m": float(self._segment_max_path_m),
            "valhalla_segment_max_step_m": float(self._segment_max_step_m),
            "valhalla_split_on_failure": bool(self._split_on_failure),
            "valhalla_split_min_points": int(self._split_min_points),
            "valhalla_split_max_depth": int(self._split_max_depth),
        }
        for key, value in self._diag.items():
            out[f"valhalla_{key}"] = value
        return out

    def deconst(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass
        if not (self._auto_stop and self._started_by_me):
            return
        if shutil.which("docker") is None:
            return
        self._run_cmd(["docker", "rm", "-f", self._container_name], check=False)
