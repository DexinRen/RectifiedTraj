"""
Parquet Processor for Trajectory Denoising Pipeline

This module processes large parquet datasets containing GPS trajectory data,
dicing them into overlapping chunks for rectified flow model training.

Key features:
- File-by-file processing with garbage collection
- Lazy loading with Polars
- ENU coordinate transformation
- Chunk generation with overlap (buckle points)

TODO list:
    - Implement each stage as standalone function.
    - Maintain logger injection.
    - Follow return-dict pattern with explicit keys per code.style.txt.
"""

import os
import gc
import json
import math
import argparse
import logging
import hashlib
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import polars as pl
import torch
import glob
from pymap3d import geodetic2enu
try:
    from .traj_suite_runner import (
        run_traj_extraction_suites_isolated,
        TRAJ_ALLOW_SHORTER,
    )
    from .traj_extractor import (
        extract_native_traj as _extract_native_traj_for_calibration,
        build_traj_extraction_context as _build_traj_context_for_calibration,
    )
except ImportError:
    from traj_suite_runner import (
        run_traj_extraction_suites_isolated,
        TRAJ_ALLOW_SHORTER,
    )
    from traj_extractor import (
        extract_native_traj as _extract_native_traj_for_calibration,
        build_traj_extraction_context as _build_traj_context_for_calibration,
    )


# Configure logger
logger = logging.getLogger(__name__)
RAW_ROOT = Path("./dataset/raw")
DEFAULT_RAW_DATASET = "NUMOSIM_Kanto"
DEFAULT_RAW_DS_PATH = str(RAW_ROOT / DEFAULT_RAW_DATASET)
BLOGWATCHER_DATASET = "BlogWatcher"


def _empty_boundary_accumulator() -> dict:
    return {
        "min_lat": None,
        "max_lat": None,
        "min_lon": None,
        "max_lon": None,
    }


def _update_boundary_from_lonlat(boundary: dict, lon_values, lat_values) -> None:
    lon = np.asarray(lon_values, dtype=float).reshape(-1)
    lat = np.asarray(lat_values, dtype=float).reshape(-1)
    if lon.size == 0 or lat.size == 0:
        return
    n = min(int(lon.size), int(lat.size))
    if n <= 0:
        return
    lon = lon[:n]
    lat = lat[:n]
    mask = np.isfinite(lon) & np.isfinite(lat)
    if not np.any(mask):
        return
    lon = lon[mask]
    lat = lat[mask]
    min_lat = float(np.min(lat))
    max_lat = float(np.max(lat))
    min_lon = float(np.min(lon))
    max_lon = float(np.max(lon))
    if boundary["min_lat"] is None or min_lat < boundary["min_lat"]:
        boundary["min_lat"] = min_lat
    if boundary["max_lat"] is None or max_lat > boundary["max_lat"]:
        boundary["max_lat"] = max_lat
    if boundary["min_lon"] is None or min_lon < boundary["min_lon"]:
        boundary["min_lon"] = min_lon
    if boundary["max_lon"] is None or max_lon > boundary["max_lon"]:
        boundary["max_lon"] = max_lon


def _update_boundary_from_noisy_chunks(boundary: dict, raw_chunks: list[dict]) -> None:
    for chunk in raw_chunks:
        x1 = np.asarray(chunk.get("X1", []), dtype=float)
        if x1.size == 0 or x1.ndim != 2 or x1.shape[1] < 2:
            continue
        _update_boundary_from_lonlat(boundary, x1[:, 0], x1[:, 1])


def _update_boundary_from_saved_trajectories(boundary: dict, trajectories: list[dict]) -> int:
    used = 0
    for one in trajectories:
        if not isinstance(one, dict):
            continue
        data = one.get("data")
        if data is None:
            continue
        arr = np.asarray(data, dtype=float)
        if arr.size == 0 or arr.ndim != 2 or arr.shape[1] < 2:
            continue
        _update_boundary_from_lonlat(boundary, arr[:, 0], arr[:, 1])
        used += 1
    return int(used)


def _collect_completed_traj_output_files(traj_extraction: Optional[dict]) -> list[str]:
    if not isinstance(traj_extraction, dict):
        return []
    if str(traj_extraction.get("status", "")) != "completed":
        return []
    out: list[str] = []
    seen: set[str] = set()
    for suite_name in ("full", "debug"):
        suite = traj_extraction.get(suite_name, {})
        if not isinstance(suite, dict):
            continue
        runs = suite.get("runs", {})
        if not isinstance(runs, dict):
            continue
        for rec in runs.values():
            if not isinstance(rec, dict):
                continue
            if str(rec.get("status", "")) != "completed":
                continue
            output_file = rec.get("output_file")
            if not output_file:
                continue
            key = str(output_file)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def _update_boundary_from_traj_output_files(boundary: dict, output_files: list[str]) -> dict:
    stats = {
        "files_total": int(len(output_files)),
        "files_processed": 0,
        "files_failed": 0,
        "trajectories_used": 0,
    }
    for raw_path in output_files:
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            stats["files_failed"] = int(stats["files_failed"]) + 1
            logger.warning("Trajectory boundary source missing: %s", path)
            continue
        try:
            blob = torch.load(path, map_location="cpu")
            trajectories = blob.get("trajectories", []) if isinstance(blob, dict) else []
            if not isinstance(trajectories, list):
                trajectories = []
            used = _update_boundary_from_saved_trajectories(boundary, trajectories)
            stats["files_processed"] = int(stats["files_processed"]) + 1
            stats["trajectories_used"] = int(stats["trajectories_used"]) + int(used)
        except Exception as exc:
            stats["files_failed"] = int(stats["files_failed"]) + 1
            logger.warning("Failed reading trajectory boundary source %s: %s", path, exc)
    return stats


def _clone_boundary(boundary: dict) -> dict:
    return {
        "min_lat": boundary.get("min_lat"),
        "max_lat": boundary.get("max_lat"),
        "min_lon": boundary.get("min_lon"),
        "max_lon": boundary.get("max_lon"),
    }


def _boundary_to_four_corners(boundary: dict) -> dict | None:
    if (
        boundary["min_lat"] is None
        or boundary["max_lat"] is None
        or boundary["min_lon"] is None
        or boundary["max_lon"] is None
    ):
        return None
    return {
        "max_lat_min_lon": [boundary["max_lat"], boundary["min_lon"]],
        "max_lat_max_lon": [boundary["max_lat"], boundary["max_lon"]],
        "min_lat_min_lon": [boundary["min_lat"], boundary["min_lon"]],
        "min_lat_max_lon": [boundary["min_lat"], boundary["max_lon"]],
    }


def _boundary_corners_to_bbox(boundary_corners: dict | None) -> dict | None:
    if not isinstance(boundary_corners, dict):
        return None
    try:
        max_lat = float(boundary_corners["max_lat_min_lon"][0])
        min_lon = float(boundary_corners["max_lat_min_lon"][1])
        min_lat = float(boundary_corners["min_lat_min_lon"][0])
        max_lon = float(boundary_corners["max_lat_max_lon"][1])
    except Exception:
        return None
    if not (np.isfinite(min_lat) and np.isfinite(max_lat) and np.isfinite(min_lon) and np.isfinite(max_lon)):
        return None
    if min_lat > max_lat or min_lon > max_lon:
        return None
    return {
        "min_lon": min_lon,
        "min_lat": min_lat,
        "max_lon": max_lon,
        "max_lat": max_lat,
    }


def _pad_bbox_km(bbox: dict, padding_km: float) -> dict:
    km = float(max(0.0, padding_km))
    lat_pad = km / 111.32
    mean_lat = 0.5 * (float(bbox["min_lat"]) + float(bbox["max_lat"]))
    lon_scale = max(1e-6, abs(math.cos(math.radians(mean_lat))))
    lon_pad = km / (111.32 * lon_scale)
    out = {
        "min_lon": max(-180.0, float(bbox["min_lon"]) - lon_pad),
        "min_lat": max(-90.0, float(bbox["min_lat"]) - lat_pad),
        "max_lon": min(180.0, float(bbox["max_lon"]) + lon_pad),
        "max_lat": min(90.0, float(bbox["max_lat"]) + lat_pad),
    }
    return out


def _clip_bbox_to_bounds(bbox: dict, bounds: dict) -> dict | None:
    out = {
        "min_lon": max(float(bbox["min_lon"]), float(bounds["min_lon"])),
        "min_lat": max(float(bbox["min_lat"]), float(bounds["min_lat"])),
        "max_lon": min(float(bbox["max_lon"]), float(bounds["max_lon"])),
        "max_lat": min(float(bbox["max_lat"]), float(bounds["max_lat"])),
    }
    if out["min_lon"] > out["max_lon"] or out["min_lat"] > out["max_lat"]:
        return None
    return out


def _bbox_almost_equal(a: dict, b: dict, eps: float = 1e-9) -> bool:
    keys = ("min_lon", "min_lat", "max_lon", "max_lat")
    return all(abs(float(a[k]) - float(b[k])) <= float(eps) for k in keys)


def _bbox_contains(container: dict, inner: dict, eps: float = 1e-9) -> bool:
    return (
        float(container["min_lon"]) <= float(inner["min_lon"]) + float(eps)
        and float(container["min_lat"]) <= float(inner["min_lat"]) + float(eps)
        and float(container["max_lon"]) >= float(inner["max_lon"]) - float(eps)
        and float(container["max_lat"]) >= float(inner["max_lat"]) - float(eps)
    )


def _read_osm_bbox_with_osmium(osm_path: Path) -> dict | None:
    if shutil.which("osmium") is None:
        return None
    proc = subprocess.run(
        ["osmium", "fileinfo", "-e", str(osm_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        return None
    pat = re.compile(
        r"Bounding box:\s*\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)"
    )
    for line in proc.stdout.splitlines():
        m = pat.search(line)
        if not m:
            continue
        try:
            min_lon = float(m.group(1))
            min_lat = float(m.group(2))
            max_lon = float(m.group(3))
            max_lat = float(m.group(4))
        except Exception:
            continue
        if not (np.isfinite(min_lon) and np.isfinite(min_lat) and np.isfinite(max_lon) and np.isfinite(max_lat)):
            continue
        if min_lon > max_lon or min_lat > max_lat:
            continue
        return {
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat,
        }
    return None


def _dataset_map_aliases(raw_ds_path: str) -> list[str]:
    ds = Path(raw_ds_path).name.strip().lower()
    if ds == "numosim_kanto":
        return ["numosim", "kanto", "japan", "tokyo"]
    if ds == "blogwatcher":
        return ["blogwatcher", "japan", "tokyo", "kanto"]
    if ds == "pol":
        return ["pol", "georgia", "atlanta"]
    return [ds] if ds else []


def _score_raw_map_candidate(path: Path, aliases: list[str]) -> tuple[int, int]:
    name = path.name.lower()
    stem = path.stem.lower()
    score = 0
    if name.endswith(".osm.pbf") or name.endswith(".pbf"):
        score += 100
    if "tiny" in stem or "mini" in stem or "debug" in stem:
        score -= 80
    for alias in aliases:
        if stem == f"map_{alias}" or stem == alias or stem == f"{alias}-latest":
            score += 120
        elif f"map_{alias}" in stem or f"{alias}-latest" in stem:
            score += 80
        elif alias in stem:
            score += 30
    try:
        size = int(path.stat().st_size)
    except Exception:
        size = 0
    return score, size


_RAW_NAME_MIN_BYTES = {
    # Guardrails to catch accidental mislabeling (for example tiny slice named as whole-country map).
    "japan": 500_000_000,
    "georgia": 50_000_000,
    "kanto": 50_000_000,
}


def _processed_map_inode_keys() -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    roots = [Path("./dataset/map_processed"), Path("./dataset/map")]
    patterns = ("*.osm.pbf", "*.pbf")
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for pat in patterns:
            for p in root.glob(pat):
                try:
                    st = p.stat()
                    out.add((int(st.st_dev), int(st.st_ino)))
                except Exception:
                    continue
    return out


def _raw_map_name_scope_issue(path: Path, size_bytes: int) -> str | None:
    name = path.name.lower()
    for token, min_bytes in _RAW_NAME_MIN_BYTES.items():
        if token in name and int(size_bytes) < int(min_bytes):
            return (
                f"filename suggests '{token}' but file is too small "
                f"({int(size_bytes)} bytes < {int(min_bytes)} bytes threshold)"
            )
    return None


def _validate_raw_map_candidate(
    path: Path,
    processed_inode_keys: set[tuple[int, int]],
) -> tuple[bool, str | None]:
    try:
        st = path.stat()
    except Exception as exc:
        return False, f"stat failed: {exc}"
    inode_key = (int(st.st_dev), int(st.st_ino))
    if inode_key in processed_inode_keys:
        return False, "raw-map file is hard-linked with processed map artifacts"
    scope_issue = _raw_map_name_scope_issue(path, int(st.st_size))
    if scope_issue:
        return False, scope_issue
    return True, None


def _resolve_raw_pbf_candidates_for_dataset(
    raw_ds_path: str,
    raw_map_path: Optional[str] = None,
    raw_map_dir: str = "./dataset/raw_map",
) -> list[Path]:
    processed_inodes = _processed_map_inode_keys()

    if raw_map_path:
        p = Path(str(raw_map_path))
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if p.exists() and p.is_file() and p.name.lower().endswith(".pbf"):
            ok, reason = _validate_raw_map_candidate(p, processed_inodes)
            if ok:
                return [p]
            logger.warning("Rejected raw_map_path %s: %s", p, reason)
        return []

    root = Path(raw_map_dir)
    if not root.exists() or not root.is_dir():
        return []

    aliases = _dataset_map_aliases(raw_ds_path)
    candidates = sorted(set(list(root.glob("*.osm.pbf")) + list(root.glob("*.pbf"))))
    if not candidates:
        return []
    valid_candidates: list[Path] = []
    for cand in candidates:
        ok, reason = _validate_raw_map_candidate(cand, processed_inodes)
        if ok:
            valid_candidates.append(cand)
        else:
            logger.warning("Rejected raw-map candidate %s: %s", cand.name, reason)
    if not valid_candidates:
        return []

    ranked = sorted(valid_candidates, key=lambda p: _score_raw_map_candidate(p, aliases), reverse=True)
    return ranked


def _resolve_raw_pbf_for_dataset(
    raw_ds_path: str,
    raw_map_path: Optional[str] = None,
    raw_map_dir: str = "./dataset/raw_map",
) -> Path | None:
    ranked = _resolve_raw_pbf_candidates_for_dataset(
        raw_ds_path=raw_ds_path,
        raw_map_path=raw_map_path,
        raw_map_dir=raw_map_dir,
    )
    if not ranked:
        return None
    return ranked[0]


def _select_raw_pbf_for_boundary(
    raw_ds_path: str,
    *,
    bbox_unpadded: dict,
    bbox_padded: dict,
    raw_map_path: Optional[str] = None,
    raw_map_dir: str = "./dataset/raw_map",
) -> tuple[Path | None, dict | None, dict]:
    """
    Raw-map selection policy:
    1) Prefer candidate maps that fully contain padded bbox.
    2) If none, fallback to candidate maps that fully contain unpadded bbox.
    3) If none contains unpadded bbox, fail hard via caller.
    """
    candidates = _resolve_raw_pbf_candidates_for_dataset(
        raw_ds_path=raw_ds_path,
        raw_map_path=raw_map_path,
        raw_map_dir=raw_map_dir,
    )
    meta: dict = {
        "candidate_count": int(len(candidates)),
        "padded_cover_count": 0,
        "unpadded_cover_count": 0,
        "selected_mode": None,
        "selected_candidate": None,
    }
    if not candidates:
        meta["status"] = "failed_missing_raw_pbf"
        meta["error"] = (
            "No source PBF found. Provide --raw-map-path or place a .pbf in ./dataset/raw_map."
        )
        return None, None, meta

    selected_padded: tuple[Path, dict] | None = None
    selected_unpadded: tuple[Path, dict] | None = None
    checked_names: list[str] = []

    for cand in candidates:
        checked_names.append(cand.name)
        source_bbox = _read_osm_bbox_with_osmium(cand)
        if source_bbox is None:
            continue
        contains_unpadded = _bbox_contains(source_bbox, bbox_unpadded)
        contains_padded = _bbox_contains(source_bbox, bbox_padded)
        if contains_unpadded:
            meta["unpadded_cover_count"] = int(meta["unpadded_cover_count"]) + 1
            if selected_unpadded is None:
                selected_unpadded = (cand, source_bbox)
        if contains_padded:
            meta["padded_cover_count"] = int(meta["padded_cover_count"]) + 1
            if selected_padded is None:
                selected_padded = (cand, source_bbox)

    if selected_padded is not None:
        sel_path, sel_bbox = selected_padded
        meta["status"] = "ok"
        meta["selected_mode"] = "padded_bbox"
        meta["selected_candidate"] = sel_path.name
        return sel_path, sel_bbox, meta

    if selected_unpadded is not None:
        sel_path, sel_bbox = selected_unpadded
        meta["status"] = "ok"
        meta["selected_mode"] = "unpadded_bbox_fallback"
        meta["selected_candidate"] = sel_path.name
        return sel_path, sel_bbox, meta

    req = (
        f"{float(bbox_unpadded['min_lon']):.7f},{float(bbox_unpadded['min_lat']):.7f},"
        f"{float(bbox_unpadded['max_lon']):.7f},{float(bbox_unpadded['max_lat']):.7f}"
    )
    cand_list = ", ".join(checked_names) if checked_names else "(none)"
    meta["status"] = "failed_no_map_contains_unpadded_bbox"
    meta["error"] = (
        "No raw map candidate fully contains unpadded test bbox. "
        f"required_bbox={req}; candidates={cand_list}"
    )
    return None, None, meta


def _path_for_state(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path.resolve())


def _slice_map_pbf_to_boundary(
    raw_ds_path: str,
    boundary_corners: dict | None,
    *,
    map_padding_km: float = 5.0,
    raw_map_path: Optional[str] = None,
    run_map_slice: bool = True,
    raw_map_dir: str = "./dataset/raw_map",
) -> dict:
    meta: dict = {
        "provider": "local_raw_map",
        "mode": "slice_bbox",
        "format": "osm.pbf",
        "padding_km": float(map_padding_km),
        "raw_map_dir": str(raw_map_dir),
        "processed_at": datetime.now().isoformat(),
    }
    if not run_map_slice:
        meta["status"] = "skipped_disabled"
        return meta

    bbox = _boundary_corners_to_bbox(boundary_corners)
    if not bbox:
        meta["status"] = "skipped_no_boundary"
        return meta
    meta["bbox_unpadded"] = bbox

    if shutil.which("osmium") is None:
        meta["status"] = "failed_missing_osmium"
        meta["error"] = "osmium command is not available in PATH."
        return meta

    bbox_padded_requested = _pad_bbox_km(bbox, float(map_padding_km))
    source_path, source_bbox, source_selection = _select_raw_pbf_for_boundary(
        raw_ds_path=raw_ds_path,
        bbox_unpadded=bbox,
        bbox_padded=bbox_padded_requested,
        raw_map_path=raw_map_path,
        raw_map_dir=raw_map_dir,
    )
    meta["bbox_padded_requested"] = bbox_padded_requested
    if isinstance(source_selection, dict):
        meta["source_selection"] = source_selection
    if source_path is None:
        status = str((source_selection or {}).get("status", "failed_missing_raw_pbf"))
        error = str(
            (source_selection or {}).get(
                "error",
                "No source PBF found. Provide --raw-map-path or place a .pbf in ./dataset/raw_map.",
            )
        )
        meta["status"] = status
        meta["error"] = error
        if status == "failed_no_map_contains_unpadded_bbox":
            raise RuntimeError(error)
        return meta

    ds_name = _dataset_name_from_raw_ds_path(raw_ds_path)
    out_dir = Path("./dataset/map_processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"map_{ds_name}.osm.pbf"
    out_tmp_path = out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")

    def _invalidate_output_files() -> None:
        try:
            if out_tmp_path.exists():
                out_tmp_path.unlink()
        except Exception:
            pass
        try:
            if out_path.exists():
                out_path.unlink()
        except Exception:
            pass

    if out_tmp_path.exists():
        try:
            out_tmp_path.unlink()
        except Exception:
            pass

    bbox_padded = bbox_padded_requested
    clipped_by_source_bounds = False
    if source_bbox is not None:
        meta["source_bbox"] = source_bbox
        source_contains_test_bbox = _bbox_contains(source_bbox, bbox)
        meta["source_contains_test_bbox"] = bool(source_contains_test_bbox)
        if not source_contains_test_bbox:
            meta["status"] = "failed_test_bbox_not_fully_in_source"
            meta["bbox_padded_requested"] = bbox_padded_requested
            meta["error"] = "Source map bbox does not fully contain test data bbox."
            _invalidate_output_files()
            return meta

        clipped = _clip_bbox_to_bounds(bbox_padded_requested, source_bbox)
        if clipped is None:
            meta["status"] = "failed_bbox_outside_source"
            meta["bbox_padded_requested"] = bbox_padded_requested
            meta["error"] = "Requested bbox does not intersect source map bbox."
            _invalidate_output_files()
            return meta
        clipped_by_source_bounds = not _bbox_almost_equal(clipped, bbox_padded_requested)
        bbox_padded = clipped
    meta["bbox_padded_requested"] = bbox_padded_requested
    meta["bbox_padded"] = bbox_padded
    meta["padding_mode"] = "soft_clip_to_source_bbox"
    meta["clipped_by_source_bounds"] = bool(clipped_by_source_bounds)

    bbox_arg = (
        f"{bbox_padded['min_lon']:.7f},"
        f"{bbox_padded['min_lat']:.7f},"
        f"{bbox_padded['max_lon']:.7f},"
        f"{bbox_padded['max_lat']:.7f}"
    )
    cmd = [
        "osmium",
        "extract",
        "-b",
        bbox_arg,
        "-O",
        "-f",
        "pbf",
        "-o",
        str(out_tmp_path),
        str(source_path),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    meta["source_path"] = _path_for_state(source_path)
    meta["source_filename"] = source_path.name
    meta["path"] = _path_for_state(out_path)
    meta["filename"] = out_path.name
    meta["required_test_bbox"] = bbox

    if proc.returncode != 0:
        meta["status"] = "failed"
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        meta["error"] = stderr[-500:] if stderr else (stdout[-500:] if stdout else "osmium extract failed")
        _invalidate_output_files()
        return meta

    output_bbox = _read_osm_bbox_with_osmium(out_tmp_path)
    meta["output_bbox"] = output_bbox
    if output_bbox is None:
        meta["status"] = "failed_output_bbox_missing"
        meta["error"] = "Failed to read output map bbox after slicing."
        _invalidate_output_files()
        return meta

    output_contains_test_bbox = _bbox_contains(output_bbox, bbox)
    meta["output_contains_test_bbox"] = bool(output_contains_test_bbox)
    if not output_contains_test_bbox:
        meta["status"] = "failed_output_missing_test_bbox"
        meta["error"] = "Sliced map bbox does not fully contain test data bbox."
        _invalidate_output_files()
        return meta

    try:
        out_tmp_path.replace(out_path)
    except Exception as exc:
        meta["status"] = "failed_output_finalize"
        meta["error"] = f"Failed to finalize sliced map: {exc}"
        _invalidate_output_files()
        return meta

    try:
        out_size = int(out_path.stat().st_size)
    except Exception:
        out_size = 0
    meta["slice_stats"] = {"bytes": out_size}
    meta["status"] = "ok"
    return meta


def _best_effort_memory_cleanup() -> None:
    """Try to release Python and allocator-retained memory between heavy stages."""
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _dataset_name_from_raw_ds_path(raw_ds_path: str) -> str:
    return Path(raw_ds_path).name


def _processed_output_root(raw_ds_path: str) -> Path:
    return Path("./dataset/processed") / _dataset_name_from_raw_ds_path(raw_ds_path)


def _split_output_dir(raw_ds_path: str, split: str) -> Path:
    split_name = "chunk_test" if str(split) == "test" else str(split)
    return _processed_output_root(raw_ds_path) / split_name


def _chunk_test_debug_dir(raw_ds_path: str) -> Path:
    return _processed_output_root(raw_ds_path) / "chunk_test_debug"


def _native_chunk_filename(n_chunks: int) -> str:
    return f"chunk_native_{int(n_chunks)}.pt"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    i = 2
    while True:
        cand = parent / f"{stem}_{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1


def _normalize_coord_space_token(token: object) -> str:
    text = str(token or "").strip().upper()
    return text if text else "UNKNOWN"


def _infer_chunk_coord_space_from_xy(chunk_xy: object) -> str:
    try:
        arr = np.asarray(chunk_xy, dtype=float)
    except Exception:
        return "UNKNOWN"
    if arr.ndim == 3 and arr.shape[-1] >= 2:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.ndim != 2 or arr.shape[1] < 2:
        return "UNKNOWN"
    lon = arr[:, 0].reshape(-1)
    lat = arr[:, 1].reshape(-1)
    if lon.size == 0 or lat.size == 0:
        return "UNKNOWN"
    mask = np.isfinite(lon) & np.isfinite(lat)
    if not np.any(mask):
        return "UNKNOWN"
    lon = lon[mask]
    lat = lat[mask]
    if np.all((lon >= -180.0) & (lon <= 180.0) & (lat >= -90.0) & (lat <= 90.0)):
        return "GPS"
    if np.any(np.abs(lon) > 180.0) or np.any(np.abs(lat) > 90.0):
        return "ENU"
    return "UNKNOWN"


def _take_first_test_chunks(pack: dict, n_take: int) -> dict:
    n = int(max(0, n_take))
    if n <= 0:
        return {}
    out = {
        "X0": pack["X0"][:n].clone(),
        "X1": pack["X1"][:n].clone(),
    }
    coord_space = _normalize_coord_space_token(pack.get("coord_space"))
    if coord_space == "UNKNOWN":
        coord_space = _infer_chunk_coord_space_from_xy(out["X1"])
    if coord_space != "UNKNOWN":
        out["coord_space"] = coord_space
    if "accuracy" in pack:
        out["accuracy"] = pack["accuracy"][:n].clone()
    if "error_range" in pack:
        out["error_range"] = pack["error_range"][:n].clone()
    return out


def _append_test_chunk_debug(debug_pack: Optional[dict], pack: dict, max_chunks: int = 2) -> dict:
    current = 0
    if isinstance(debug_pack, dict) and "X1" in debug_pack:
        current = int(debug_pack["X1"].shape[0])
    remain = int(max_chunks) - int(current)
    if remain <= 0:
        return debug_pack if isinstance(debug_pack, dict) else {}

    take_n = min(int(remain), int(pack["X1"].shape[0]))
    if take_n <= 0:
        return debug_pack if isinstance(debug_pack, dict) else {}
    one = _take_first_test_chunks(pack, take_n)
    if not debug_pack:
        return one

    out = {
        "X0": torch.cat([debug_pack["X0"], one["X0"]], dim=0),
        "X1": torch.cat([debug_pack["X1"], one["X1"]], dim=0),
    }
    left_space = _normalize_coord_space_token(debug_pack.get("coord_space"))
    if left_space == "UNKNOWN":
        left_space = _infer_chunk_coord_space_from_xy(debug_pack.get("X1"))
    right_space = _normalize_coord_space_token(one.get("coord_space"))
    if right_space == "UNKNOWN":
        right_space = _infer_chunk_coord_space_from_xy(one.get("X1"))
    merged_space = left_space if left_space != "UNKNOWN" else right_space
    if (
        left_space != "UNKNOWN"
        and right_space != "UNKNOWN"
        and left_space != right_space
    ):
        logger.warning(
            "Mixed coord_space while building chunk_test_debug (%s vs %s); marking as UNKNOWN.",
            left_space,
            right_space,
        )
        merged_space = "UNKNOWN"
    if merged_space != "UNKNOWN":
        out["coord_space"] = merged_space
    if "accuracy" in debug_pack and "accuracy" in one:
        out["accuracy"] = torch.cat([debug_pack["accuracy"], one["accuracy"]], dim=0)
    if "error_range" in debug_pack and "error_range" in one:
        out["error_range"] = torch.cat([debug_pack["error_range"], one["error_range"]], dim=0)
    return out


def _save_chunk_test_debug(debug_pack: Optional[dict], raw_ds_path: str) -> Optional[str]:
    if not debug_pack or "X1" not in debug_pack or int(debug_pack["X1"].shape[0]) == 0:
        return None
    coord_space = _normalize_coord_space_token(debug_pack.get("coord_space"))
    if coord_space == "UNKNOWN":
        inferred = _infer_chunk_coord_space_from_xy(debug_pack["X1"])
        if inferred != "UNKNOWN":
            debug_pack = dict(debug_pack)
            debug_pack["coord_space"] = inferred
    out_dir = _chunk_test_debug_dir(raw_ds_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = _unique_path(out_dir / _native_chunk_filename(int(debug_pack["X1"].shape[0])))
    torch.save(debug_pack, out_file)
    with open(f"{out_file}.len", "w") as f:
        f.write(str(int(debug_pack["X1"].shape[0])))
    return str(out_file)


def _state_path_from_raw_ds_path(raw_ds_path: str) -> Path:
    ds_folder = Path(raw_ds_path).name
    return Path("./dataset/state") / f"state_{ds_folder}.json"


def _extraction_cursor_path_from_raw_ds_path(raw_ds_path: str) -> Path:
    ds_folder = Path(raw_ds_path).name
    return Path("./dataset/state") / f"ds_extraction_end_{ds_folder}.json"


def _is_blogwatcher_dataset(raw_ds_path: str) -> bool:
    return Path(raw_ds_path).name.lower() == BLOGWATCHER_DATASET.lower()


def _canonicalize_columns(df: pl.DataFrame, include_error_range: bool = False) -> pl.DataFrame:
    """
    Normalize dataset-specific parquet schemas to canonical columns used here.
    """
    cols = set(df.columns)
    aliases = {
        "agent": ["agent", "uuid"],
        "timestamp": ["timestamp", "datetime"],
        "longitude_n": ["longitude_n", "longitude_noisy"],
        "latitude_n": ["latitude_n", "latitude_noisy"],
        "longitude": ["longitude", "longitude_anonymous"],
        "latitude": ["latitude", "latitude_anonymous"],
    }
    if include_error_range:
        aliases["error_range"] = ["error_range", "accuracy"]

    rename_map = {}
    missing = []
    for canonical, cands in aliases.items():
        found = next((c for c in cands if c in cols), None)
        if found is None:
            missing.append(canonical)
            continue
        if found != canonical:
            rename_map[found] = canonical
    if missing:
        raise ValueError(
            f"Missing required columns after schema normalization: {missing}; available={sorted(cols)}"
        )
    if rename_map:
        df = df.rename(rename_map)
    return df


def _migrate_legacy_state_files(raw_ds_path: str) -> None:
    """
    Best-effort migration from legacy state files into state_<dataset>.json.
    Legacy files are kept in place; data is merged into dataset-scoped state.
    """
    target_state_path = _state_path_from_raw_ds_path(raw_ds_path)
    legacy_states_path = Path("./dataset/state/states.json")
    legacy_corrupted_path = Path("./dataset/state/corrupted_files.json")
    legacy_ds_records_path = Path("./dataset/state/ds_records.jsonl")

    payload = {}

    if legacy_states_path.exists():
        with open(legacy_states_path, "r") as f:
            payload.update(json.load(f))

    if legacy_corrupted_path.exists():
        with open(legacy_corrupted_path, "r") as f:
            payload["corrupted_files"] = json.load(f)

    if legacy_ds_records_path.exists():
        cursor_path = _extraction_cursor_path_from_raw_ds_path(raw_ds_path)
        cursor_path.parent.mkdir(parents=True, exist_ok=True)

        cursor_data = {"updated_at": datetime.now().isoformat(), "files": {}}
        if cursor_path.exists():
            with open(cursor_path, "r") as f:
                cursor_data = json.load(f)
            if "files" not in cursor_data or not isinstance(cursor_data["files"], dict):
                cursor_data["files"] = {}

        with open(legacy_ds_records_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                ds_name = rec.get("ds_name")
                users = rec.get("users")
                if ds_name is None or not isinstance(users, dict):
                    continue
                existing_users = (
                    cursor_data["files"].get(ds_name, {}).get("users", {})
                    if isinstance(cursor_data["files"].get(ds_name, {}), dict)
                    else {}
                )
                merged_users = dict(existing_users)
                for u, end_row in users.items():
                    if u not in merged_users or int(end_row) > int(merged_users[u]):
                        merged_users[u] = int(end_row)
                cursor_data["files"][ds_name] = {"users": merged_users}

        cursor_data["updated_at"] = datetime.now().isoformat()
        with open(cursor_path, "w") as f:
            json.dump(cursor_data, f, indent=2)

    if payload:
        payload["updated_at"] = datetime.now().isoformat()
        _upsert_states_json(payload, state_path=str(target_state_path))


def ds_dicer(
    ds_entry: dict,
    K: int = 256,
    Q: int = 1,
    extraction_cursor_path: Optional[str] = None,
    allowed_agents: Optional[set[str]] = None,
    max_users: int = 200,
    max_chunks: int = 10000,
) -> Tuple[dict, dict]:
    """
    Purpose:
        Randomly sample users in one dataset and dice their full
        trajectories into K-point chunks with overlap Q. Stop when a
        configured max chunk count is generated.

    Parameters:
        ds_entry (dict): {"name": str, "ds": polars.LazyFrame}
        K (int): chunk size (default 256)
        Q (int): overlap size (default 1)
        max_users (int): max sampled users in this file (default 200)
        max_chunks (int): max chunks generated in this file (default 10000)

    Return:
        out_ds_dicer (dict): {usr_id: [(chunk_id, row_start, row_end), ...]}
        ds_record (dict): {"ds_name": str, "users": {usr_id: end_row, ...}}

    Notes:
        - First Q points of each chunk = last Q points of previous chunk
        - For first chunk of a user: duplicate first point Q times
        - Updates dataset-scoped extraction cursor file (ds_extraction_end_<dataset>.json)
    """
    logger.info(f"Dicing dataset: {ds_entry['name']}")
    
    usr_num = int(max_users)
    chunk_num = int(max_chunks)
    ds_name = ds_entry['name']
    ds = ds_entry['ds']
    
    cursor_file = (
        Path(extraction_cursor_path)
        if extraction_cursor_path
        else _extraction_cursor_path_from_raw_ds_path(DEFAULT_RAW_DS_PATH)
    )
    cursor_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing per-file extraction cursors to resume from prior end rows.
    cursor_data = {"files": {}}
    if cursor_file.exists():
        with open(cursor_file, "r") as f:
            cursor_data = json.load(f)
        if "files" not in cursor_data or not isinstance(cursor_data["files"], dict):
            cursor_data["files"] = {}

    existing_entry = cursor_data["files"].get(ds_name, {})
    existing_records = existing_entry.get("users", {}) if isinstance(existing_entry, dict) else {}
    
    # Collect the data (lazy -> eager) with error handling
    try:
        df = ds.collect()
        df = _canonicalize_columns(df, include_error_range=False)
    except Exception as e:
        logger.error(f"Failed to collect data from {ds_name}: {e}")
        logger.warning(f"Skipping corrupted file: {ds_name}")
        # Return empty results to skip this file
        return {}, {"ds_name": ds_name, "users": {}}
    
    # Get unique user IDs
    all_users = df['agent'].unique().to_list()
    if allowed_agents is not None:
        all_users = [u for u in all_users if str(u) in allowed_agents]
    logger.info(f"Found {len(all_users)} unique users in dataset")
    
    # Randomly sample users
    n_users = min(usr_num, len(all_users))
    if n_users == 0:
        logger.warning("No eligible users found after agent filtering for %s", ds_name)
        return {}, {"ds_name": ds_name, "users": {}}
    sampled_users = np.random.choice(all_users, size=n_users, replace=False)
    logger.info(f"Sampled {n_users} users for processing")
    
    out_ds_dicer = {}
    total_chunks = 0
    chunk_counter = 0
    ds_record_users = {}
    chunk_error_batches = []
    
    # Process each sampled user
    for usr_id in sampled_users:
        if total_chunks >= chunk_num:
            break
        
        # Filter data for this user, remove NaN/null, and sort by timestamp
        # Note: Polars distinguishes between null and NaN - we need to filter both
        n_points_before = len(df.filter(pl.col('agent') == usr_id))
        
        user_df = df.filter(
            (pl.col('agent') == usr_id) &
            pl.col('longitude_n').is_not_null() &
            pl.col('latitude_n').is_not_null() &
            pl.col('longitude').is_not_null() &
            pl.col('latitude').is_not_null() &
            pl.col('longitude_n').is_finite() &
            pl.col('latitude_n').is_finite() &
            pl.col('longitude').is_finite() &
            pl.col('latitude').is_finite()
        ).sort('timestamp')
        
        n_points = len(user_df)
        
        if n_points < n_points_before:
            logger.info(f"User {usr_id}: Filtered out {n_points_before - n_points} NaN/null points, {n_points} valid points remaining")
        
        # CRITICAL: Verify no NaN made it through
        if n_points > 0:
            lon_n_array = user_df['longitude_n'].to_numpy()
            lat_n_array = user_df['latitude_n'].to_numpy()
            lon_array = user_df['longitude'].to_numpy()
            lat_array = user_df['latitude'].to_numpy()
            
            has_nan = (np.any(~np.isfinite(lon_n_array)) or np.any(~np.isfinite(lat_n_array)) or
                      np.any(~np.isfinite(lon_array)) or np.any(~np.isfinite(lat_array)))
            
            if has_nan:
                logger.error(f"❌ BUG: User {usr_id} still has NaN after Polars filter!")
                logger.error(f"   lon_n NaN count: {np.sum(~np.isfinite(lon_n_array))}")
                logger.error(f"   lat_n NaN count: {np.sum(~np.isfinite(lat_n_array))}")
                logger.error(f"   lon NaN count: {np.sum(~np.isfinite(lon_array))}")
                logger.error(f"   lat NaN count: {np.sum(~np.isfinite(lat_array))}")
                logger.error(f"   Polars filter is NOT working correctly!")
                continue
        
        # Check if user has enough valid points
        if n_points < K:
            logger.debug(f"User {usr_id}: Only {n_points} valid points (need {K}), skipping user")
            continue

        # Build per-point noisy-clean distance (meters) in ENU for stats.
        lon_n_full = user_df["longitude_n"].to_numpy()
        lat_n_full = user_df["latitude_n"].to_numpy()
        lon_c_full = user_df["longitude"].to_numpy()
        lat_c_full = user_df["latitude"].to_numpy()
        e_n, n_n, _ = geodetic2enu(lat_n_full, lon_n_full, 0.0, lat_c_full[0], lon_c_full[0], 0.0)
        e_c, n_c, _ = geodetic2enu(lat_c_full, lon_c_full, 0.0, lat_c_full[0], lon_c_full[0], 0.0)
        point_err = np.sqrt((e_n - e_c) ** 2 + (n_n - n_c) ** 2)
        
        # Determine starting row
        start_row = existing_records.get(str(usr_id), 0)
        
        if start_row >= n_points - K:
            logger.debug(f"User {usr_id} already fully processed, skipping")
            continue
        
        chunks_for_user = []
        current_row = start_row
        
        # Generate chunks with overlap
        # First chunk starts at start_row
        # Subsequent chunks overlap by Q points with previous chunk
        while current_row + K <= n_points and total_chunks < chunk_num:
            row_start = current_row
            row_end = current_row + K - 1  # inclusive
            
            chunks_for_user.append((chunk_counter, row_start, row_end))
            chunk_error_batches.append(point_err[row_start:row_end + 1])
            chunk_counter += 1
            total_chunks += 1
            
            # Move to next chunk: overlap by Q points
            # Next chunk starts at (current_row + K - Q)
            current_row += (K - Q)
        
        if chunks_for_user:
            out_ds_dicer[usr_id] = chunks_for_user
            # Record the last row index that was included in a chunk
            ds_record_users[str(usr_id)] = chunks_for_user[-1][2]  # last row_end
            logger.debug(f"User {usr_id}: generated {len(chunks_for_user)} chunks")
    
    logger.info(f"Total chunks generated: {total_chunks}")
    
    # Prepare ds_record
    if chunk_error_batches:
        all_chunk_err = np.concatenate(chunk_error_batches)
        error_stats = {
            "avg_error": float(np.mean(all_chunk_err)),
            "median_error": float(np.median(all_chunk_err)),
            "std_error": float(np.std(all_chunk_err)),
            "num_points": int(all_chunk_err.size),
        }
    else:
        error_stats = {
            "avg_error": None,
            "median_error": None,
            "std_error": None,
            "num_points": 0,
        }

    ds_record = {
        "ds_name": ds_name,
        "users": ds_record_users,
        "error_stats": error_stats,
    }
    
    # Merge with existing record: preserve largest end_row per user.
    if existing_records:
        for usr_id, end_row in existing_records.items():
            if usr_id not in ds_record_users or end_row > ds_record_users.get(usr_id, -1):
                ds_record_users[usr_id] = end_row

    ds_record = {"ds_name": ds_name, "users": ds_record_users, "error_stats": error_stats}
    cursor_data["files"][ds_name] = {"users": ds_record_users, "error_stats": error_stats}
    cursor_data["updated_at"] = datetime.now().isoformat()
    with open(cursor_file, "w") as f:
        json.dump(cursor_data, f, indent=2)
    logger.info("Updated extraction cursor in %s for %s", cursor_file.name, ds_name)
    
    return out_ds_dicer, ds_record


def ds_assemble(ds_entry: dict,
                usr_chunks: dict,
                K: int = 256,
                Q: int = 1,
                include_error_range: bool = False) -> List[dict]:
    """
    Purpose:
        Build chunk pairs from diced trajectory segments.
        Extract [longitude_n, latitude_n, longitude, latitude, timestamp] and mark buckle points.

    Parameters:
        ds_entry (dict): {"name": str, "ds": polars.LazyFrame}
        usr_chunks (dict): {usr_id: [(chunk_id, row_start, row_end), ...]}
        K (int): chunk size (default 256)
        Q (int): overlap size (default 1)

    Return:
        out_ds_assemble (list[dict]): [
            {
                "usr_id": int,
                "chunk_id": int,
                "X1": [[longitude_n, latitude_n, timestamp, is_start], ...],
                "X0": [[longitude, latitude, timestamp, is_start], ...]
            },
            ...
        ]

    Notes:
        - First Q points have is_start = True
        - For first chunk only: duplicate first point Q times
        - Other chunks: use natural overlap from ds_dicer
    """
    logger.info(f"Assembling chunks from {ds_entry['name']}")
    
    ds_name = ds_entry['name']
    ds = ds_entry['ds']
    
    # Collect the full dataframe with error handling
    try:
        df = ds.collect()
        df = _canonicalize_columns(df, include_error_range=include_error_range)
    except Exception as e:
        logger.error(f"Failed to collect data from {ds_name}: {e}")
        logger.warning(f"Skipping corrupted file in assembly: {ds_name}")
        return []  # Return empty list to skip this file
    
    out_ds_assemble = []
    
    for usr_id, chunks in usr_chunks.items():
        # Get all data for this user
        user_filter = (
            (pl.col('agent') == usr_id) &
            pl.col('longitude_n').is_not_null() &
            pl.col('latitude_n').is_not_null() &
            pl.col('longitude').is_not_null() &
            pl.col('latitude').is_not_null() &
            pl.col('longitude_n').is_finite() &
            pl.col('latitude_n').is_finite() &
            pl.col('longitude').is_finite() &
            pl.col('latitude').is_finite()
        )
        if include_error_range:
            user_filter = (
                user_filter
                & pl.col("error_range").is_not_null()
                & pl.col("error_range").is_finite()
            )
        user_df = df.filter(user_filter).sort('timestamp')
        
        for idx, (chunk_id, row_start, row_end) in enumerate(chunks):
            is_first_chunk = (idx == 0)
            
            # Extract the chunk slice (inclusive of row_end)
            chunk_df = user_df[row_start:row_end + 1]
            
            # Verify we have K points
            if len(chunk_df) != K:
                logger.warning(f"Chunk {chunk_id} has {len(chunk_df)} points, expected {K}. Skipping.")
                continue
            
            # Extract columns and make copies (Polars arrays are read-only)
            longitude_n = chunk_df['longitude_n'].to_numpy().copy()
            latitude_n = chunk_df['latitude_n'].to_numpy().copy()
            longitude = chunk_df['longitude'].to_numpy().copy()
            latitude = chunk_df['latitude'].to_numpy().copy()
            accuracy = None
            if include_error_range:
                accuracy = chunk_df["error_range"].to_numpy().astype(np.float32, copy=True)
            
            # Convert timestamp to Unix timestamp (seconds since epoch as float)
            timestamp_dt = chunk_df['timestamp'].to_numpy()
            timestamp = timestamp_dt.astype('datetime64[s]').astype(float)
            
            # Handle first chunk: duplicate the first point Q times
            if is_first_chunk:
                # Replace first Q points with duplicates of the very first point
                longitude_n[:Q] = longitude_n[0]
                latitude_n[:Q] = latitude_n[0]
                longitude[:Q] = longitude[0]
                latitude[:Q] = latitude[0]
                timestamp[:Q] = timestamp[0]
                if accuracy is not None:
                    accuracy[:Q] = accuracy[0]
            
            # Mark buckle points (first Q points have is_start = True)
            is_start = np.zeros(K, dtype=bool)
            is_start[:Q] = True
            
            # Build X1 and X0 arrays
            X1 = np.stack([longitude_n, latitude_n, timestamp, is_start.astype(float)], axis=1)
            X0 = np.stack([longitude, latitude, timestamp, is_start.astype(float)], axis=1)
            
            one_raw_chunk = {
                "usr_id": usr_id,
                "chunk_id": chunk_id,
                "X1": X1.tolist(),
                "X0": X0.tolist()
            }
            if accuracy is not None:
                one_raw_chunk["accuracy"] = accuracy.tolist()
            
            out_ds_assemble.append(one_raw_chunk)
    
    logger.info(f"Assembled {len(out_ds_assemble)} chunks")
    return out_ds_assemble


def enu_transform(one_raw_chunk: dict) -> dict:
    """
    Purpose:
        Transform GPS coordinates (longitude, latitude) to local ENU coordinates.
        Use the first point as the origin (0, 0) for both X0 and X1.

    Parameters:
        one_raw_chunk (dict): {
            "usr_id": int,
            "chunk_id": int,
            "X1": [[longitude_n, latitude_n, timestamp, is_start], ...],
            "X0": [[longitude, latitude, timestamp, is_start], ...]
        }

    Return:
        out_enu_transform (dict): {
            "usr_id": int,
            "chunk_id": int,
            "z": {"lon0": float, "lat0": float, "lon1": float, "lat1": float},
            "chunk_enu": {
                "X1": [[e, n, timestamp, is_start], ...],
                "X0": [[e, n, timestamp, is_start], ...]
            }
        }

    Notes:
        - z stores GPS reference points for recovery
        - X0 and X1 are transformed independently
    """
    X1 = np.array(one_raw_chunk['X1'])
    X0 = np.array(one_raw_chunk['X0'])
    
    # Extract coordinates
    lon1, lat1 = X1[:, 0], X1[:, 1]
    lon0, lat0 = X0[:, 0], X0[:, 1]
    
    # Reference points (first point of each)
    lon1_ref, lat1_ref = lon1[0], lat1[0]
    lon0_ref, lat0_ref = lon0[0], lat0[0]
    
    # Transform to ENU (assuming h=0 for all points)
    # 46 is the altitude above ellipsoid in downtown LA
    # this is the height-projection adjustment on the 2D distance 
    e1, n1, _ = geodetic2enu(lat1, lon1, 46, lat1_ref, lon1_ref, 0)
    e0, n0, _ = geodetic2enu(lat0, lon0, 46, lat1_ref, lon1_ref, 0)
    # we use X1's first point as both X1 and X0 (Y)'s enu reference point
    # so they are in the same map, without adding extra noise caused by 
    # two different coordination system centered by two different point
    # and also avoid expose label info (since we use X1 as reference)
    
    # CRITICAL: Check if ENU transformation produced NaN/Inf
    
    # Build transformed arrays (keep timestamp and is_start)
    X1_enu = np.stack([e1, n1, X1[:, 2], X1[:, 3]], axis=1)
    X0_enu = np.stack([e0, n0, X0[:, 2], X0[:, 3]], axis=1)
    
    out_enu_transform = {
        "usr_id": one_raw_chunk['usr_id'],
        "chunk_id": one_raw_chunk['chunk_id'],
        "z": {
            "lon0": float(lon0_ref),
            "lat0": float(lat0_ref),
            "lon1": float(lon1_ref),
            "lat1": float(lat1_ref)
        },
        "chunk_enu": {
            "X1": X1_enu.tolist(),
            "X0": X0_enu.tolist()
        }
    }
    if "accuracy" in one_raw_chunk:
        out_enu_transform["accuracy"] = list(one_raw_chunk["accuracy"])
    
    return out_enu_transform


def _build_test_pair_tensor_pack(
    out_raw_chunks: List[dict],
    *,
    keep_timestamps_float64: bool = True,
) -> tuple[dict, int]:
    """
    Build direct test pairs from raw GPS chunks.

    Returns tensor pack:
    - X1: noisy GPS + [timestamp, is_start]
    - X0: reference GPS + [timestamp, is_start]
    - accuracy (optional): per-point accuracy/error_range if present
    """
    if not out_raw_chunks:
        return {}, 0

    n_chunks = len(out_raw_chunks)
    k_local = len(out_raw_chunks[0]["X0"])
    coord_dtype = torch.float64 if keep_timestamps_float64 else torch.float32

    X0 = torch.empty((n_chunks, k_local, 4), dtype=coord_dtype)
    X1 = torch.empty((n_chunks, k_local, 4), dtype=coord_dtype)

    has_accuracy = all("accuracy" in chunk for chunk in out_raw_chunks)
    accuracy = torch.empty((n_chunks, k_local), dtype=torch.float32) if has_accuracy else None

    for i, chunk in enumerate(out_raw_chunks):
        X0[i] = torch.tensor(chunk["X0"], dtype=coord_dtype)
        X1[i] = torch.tensor(chunk["X1"], dtype=coord_dtype)
        if has_accuracy and accuracy is not None:
            accuracy[i] = torch.tensor(chunk["accuracy"], dtype=torch.float32)

    pack = {"X0": X0, "X1": X1, "coord_space": "GPS"}
    if accuracy is not None:
        # Keep both names for downstream compatibility.
        pack["accuracy"] = accuracy
        pack["error_range"] = accuracy
    return pack, int(n_chunks)


def v_labelizer(enu_chunk: dict) -> dict:
    """
    Purpose:
        Compute velocity field label V = X1 - X0 (coordinates only).

    Parameters:
        enu_chunk (dict): {
            "usr_id": int,
            "chunk_id": int,
            "z": {...},
            "chunk_enu": {
                "X1": [[e, n, timestamp, is_start], ...],
                "X0": [[e, n, timestamp, is_start], ...]
            }
        }

    Return:
        out_v_labelizer (dict): {
            "usr_id": int,
            "chunk_id": int,
            "z": {...},
            "chunk_enu": {
                "X0": [[e, n, timestamp, is_start], ...],
                "V": [[ve, vn], ...]  # coordinates only
            }
        }
    """
    X1 = np.array(enu_chunk['chunk_enu']['X1'])
    X0 = np.array(enu_chunk['chunk_enu']['X0'])
    
    # Compute velocity (coordinates only: e and n)
    V = X1[:, :2] - X0[:, :2]
    
    out_v_labelizer = {
        "usr_id": enu_chunk['usr_id'],
        "chunk_id": enu_chunk['chunk_id'],
        "z": enu_chunk['z'],
        "chunk_enu": {
            "X0": X0.tolist(),
            "V": V.tolist()
        }
    }
    
    return out_v_labelizer

def t_sampler(v_labelizer_chunk: dict, r: int = 5) -> List[dict]:
    """
    Purpose:
        Sample r independent time values t_i ∈ [0, 1) and create
        training records {X_t, t, V}.

    Parameters:
        v_labelizer_chunk (dict): {
            "usr_id": int,
            "chunk_id": int,
            "z": {...},
            "chunk_enu": {
                "X0": [[e, n, timestamp, is_start], ...],
                "V": [[ve, vn], ...]
            }
        }
        r (int): number of time samples per chunk (default 5)

    Return:
        training_samples (list[dict]): [
            {
                "usr_id": int,
                "chunk_id": int,
                "z": {...},
                "X_t": [[e, n, timestamp, is_start], ...],
                "t": float,
                "V": [[ve, vn], ...]
            },
            ...
        ]
    """
    X0 = np.array(v_labelizer_chunk['chunk_enu']['X0'])
    V = np.array(v_labelizer_chunk['chunk_enu']['V'])
    
    training_samples = []
    
    # Sample r time values
    t_values = np.random.uniform(0, 1, size=r)
    
    for t in t_values:
        # Compute X_t = X0 + t * V (only for coordinates)
        X_t_coords = X0[:, :2] + t * V
        
        # Reconstruct X_t with timestamp and is_start
        X_t = np.concatenate([X_t_coords, X0[:, 2:]], axis=1)
        
        sample = {
            "usr_id": v_labelizer_chunk['usr_id'],
            "chunk_id": v_labelizer_chunk['chunk_id'],
            "z": v_labelizer_chunk['z'],
            "X_t": X_t.tolist(),
            "t": float(t),
            "V": V.tolist()
        }
        
        training_samples.append(sample)
    
    return training_samples


def ds_reader(ds_path_list) -> dict:
    """
    Purpose:
        Lazily read multiple Parquet datasets using the Polars package
        and return their pointers as a list for later chunk processing.

    Parameters:
        ds_path_list (list[str]):
            List of dataset file paths under a dataset folder such as ./dataset/raw/NUMOSIM_Kanto.

    Return Dict:
        "datasets": [{"name": str (filename), "ds": polars.LazyFrame pointer}, ...]
    """
    logger.info(f"Loading {len(ds_path_list)} datasets with lazy evaluation")
    datasets = []
    for ds_path in ds_path_list:
        ds = pl.scan_parquet(ds_path)
        datasets.append({"name": Path(ds_path).name, "ds": ds})
    return {"datasets": datasets}


def estimate_uncertainty_dataset_stats_from_validation(
    raw_ds_path: str = "./dataset/raw/BlogWatcher",
    file_paths: Optional[List[str]] = None,
    max_agents_per_file: Optional[int] = 200,
    random_seed: int = 42,
) -> dict:
    """
    Estimate uncertainty dataset stats (dataset-intrinsic only).

    Uses noisy-vs-reference ENU distance and accuracy tiers.
    Does NOT store signed-margin or model-inference metrics.
    """
    raw_path = Path(raw_ds_path)
    if file_paths is not None:
        val_files = [Path(p) for p in file_paths]
    else:
        val_files = sorted(raw_path.glob("*.parquet"))
    if not val_files:
        raise FileNotFoundError(f"No validation parquet files found under {raw_ds_path}")

    rng = np.random.default_rng(random_seed)

    tier_defs = [
        ("tier4_all", None),
        ("tier3_acc_leq_30", 30.0),
        ("tier2_acc_leq_15", 15.0),
        ("tier1_acc_leq_10", 10.0),
        ("tier0_acc_leq_5", 5.0),
    ]
    tier_dist = {name: [] for name, _ in tier_defs}
    tier_acc = {name: [] for name, _ in tier_defs}
    total_points = 0
    n_agents_used = 0

    for file_path in val_files:
        df = pl.scan_parquet(str(file_path)).collect()
        df = _canonicalize_columns(df, include_error_range=True)
        agent_ids = df["agent"].unique().to_list()
        if max_agents_per_file is not None and len(agent_ids) > max_agents_per_file:
            agent_ids = rng.choice(agent_ids, size=max_agents_per_file, replace=False).tolist()

        for agent_id in agent_ids:
            user_df = (
                df.filter(
                    (pl.col("agent") == agent_id)
                    & pl.col("longitude_n").is_not_null()
                    & pl.col("latitude_n").is_not_null()
                    & pl.col("longitude").is_not_null()
                    & pl.col("latitude").is_not_null()
                    & pl.col("error_range").is_not_null()
                    & pl.col("longitude_n").is_finite()
                    & pl.col("latitude_n").is_finite()
                    & pl.col("longitude").is_finite()
                    & pl.col("latitude").is_finite()
                    & pl.col("error_range").is_finite()
                )
                .sort("timestamp")
            )
            if user_df.height == 0:
                continue

            lon_n = user_df["longitude_n"].to_numpy()
            lat_n = user_df["latitude_n"].to_numpy()
            lon_c = user_df["longitude"].to_numpy()
            lat_c = user_df["latitude"].to_numpy()
            acc = user_df["error_range"].to_numpy().astype(float)

            e_n, n_n, _ = geodetic2enu(lat_n, lon_n, 0.0, lat_c[0], lon_c[0], 0.0)
            e_c, n_c, _ = geodetic2enu(lat_c, lon_c, 0.0, lat_c[0], lon_c[0], 0.0)
            dist = np.sqrt((e_n - e_c) ** 2 + (n_n - n_c) ** 2)

            total_points += int(dist.size)
            n_agents_used += 1

            for name, thr in tier_defs:
                if thr is None:
                    mask = np.ones_like(acc, dtype=bool)
                else:
                    mask = acc <= thr
                if mask.any():
                    tier_dist[name].append(dist[mask])
                    tier_acc[name].append(acc[mask])

    def _summary(arr_list: List[np.ndarray]) -> dict:
        if not arr_list:
            return {"avg": 0.0, "median": 0.0, "std": 0.0, "count": 0}
        arr = np.concatenate(arr_list)
        return {
            "avg": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "count": int(arr.size),
        }

    out = {
        "overall_distance": _summary(tier_dist["tier4_all"]),
        "tiers": {},
        "meta": {
            "raw_ds_path": str(raw_path),
            "val_files": [p.name for p in val_files],
            "n_agents_used": int(n_agents_used),
            "n_points_used": int(total_points),
            "max_agents_per_file": None if max_agents_per_file is None else int(max_agents_per_file),
        },
    }
    for name, _ in tier_defs:
        out["tiers"][name] = {
            "dist2ref": _summary(tier_dist[name]),
            "accuracy": _summary(tier_acc[name]),
        }
    return out


def _upsert_states_json(
    state_updates: dict,
    state_path: str = "./dataset/state/state_NUMOSIM_Kanto.json",
) -> dict:
    """
    Merge and persist processor state under dataset/state/state_<dataset>.json.
    Existing keys are preserved unless overwritten by state_updates.
    """
    def _deep_merge(dst: dict, src: dict) -> dict:
        for k, v in src.items():
            if k == "uncertainty_dataset_stats":
                dst[k] = v
                continue
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                _deep_merge(dst[k], v)
            else:
                dst[k] = v
        return dst

    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    current = {}
    if path.exists():
        with open(path, "r") as f:
            current = json.load(f)

    current = _deep_merge(current, state_updates)
    with open(path, "w") as f:
        json.dump(current, f, indent=2)
    return current


def _build_traj_extraction_state_append(traj_extraction: Optional[dict]) -> dict:
    """
    Build append-only trajectory extraction summary keyed by extracted .pt filename.
    """
    if not isinstance(traj_extraction, dict):
        return {}
    if str(traj_extraction.get("status", "")) != "completed":
        return {}

    out: dict = {}
    for suite_name in ("full", "debug"):
        suite = traj_extraction.get(suite_name, {})
        if not isinstance(suite, dict):
            continue
        runs = suite.get("runs", {})
        if not isinstance(runs, dict):
            continue
        for class_name, rec in runs.items():
            if not isinstance(rec, dict):
                continue
            if str(rec.get("status", "")) != "completed":
                continue
            output_file = rec.get("output_file")
            if not output_file:
                continue
            ds_name = Path(str(output_file)).name
            quality = rec.get("quality_stats", {}) if isinstance(rec.get("quality_stats"), dict) else {}
            one = {
                "suite": str(suite_name),
                "sample_time_class": str(class_name),
                "sample_time_per_point_sec": rec.get("interval_stats_sec", {}),
                "dist2ref": quality.get("dist2ref", {}),
                "output_file": str(output_file),
            }
            if "accuracy" in quality:
                one["accuracy"] = quality.get("accuracy", {})
            if "tiers" in quality:
                one["tiers"] = quality.get("tiers", {})
            out[ds_name] = one
    return out


def _detect_agent_column(raw_ds_path: str) -> str:
    parquet_files = sorted(Path(raw_ds_path).glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {raw_ds_path}")
    schema_names = set(pl.scan_parquet(str(parquet_files[0])).collect_schema().names())
    for cand in ("agent", "uuid"):
        if cand in schema_names:
            return cand
    raise ValueError(f"Missing agent column in {parquet_files[0].name}; schema={sorted(schema_names)}")


def _list_test_split_agents(raw_ds_path: str) -> list[str]:
    parquet_files = sorted(Path(raw_ds_path).glob("*.parquet"))
    if not parquet_files:
        return []
    test_files = parquet_files[-1:]
    agent_col = _detect_agent_column(raw_ds_path)
    agents: set[str] = set()
    for fp in test_files:
        try:
            one = (
                pl.scan_parquet(str(fp))
                .select(pl.col(agent_col).drop_nulls().cast(pl.Utf8).unique())
                .collect()
                .get_column(agent_col)
                .to_list()
            )
            agents.update(str(a) for a in one)
        except Exception as exc:
            logger.warning("Test-split agent scan skipped for %s: %s", fp.name, exc)
    return sorted(agents)


def _find_native_output_file_from_traj_extraction(traj_extraction: Optional[dict]) -> Optional[str]:
    if not isinstance(traj_extraction, dict):
        return None
    if str(traj_extraction.get("status", "")) != "completed":
        return None

    full_suite = traj_extraction.get("full", {})
    if isinstance(full_suite, dict):
        runs = full_suite.get("runs", {})
        if isinstance(runs, dict):
            native_rec = runs.get("native", {})
            if isinstance(native_rec, dict) and str(native_rec.get("status", "")) == "completed":
                out = native_rec.get("output_file")
                if out:
                    return str(out)
            for rec in runs.values():
                if not isinstance(rec, dict):
                    continue
                if str(rec.get("status", "")) != "completed":
                    continue
                if str(rec.get("sample_time_label", "")) != "native":
                    continue
                out = rec.get("output_file")
                if out:
                    return str(out)
    return None


def _load_saved_native_user_ids(native_traj_file: Optional[str]) -> list[str]:
    if not native_traj_file:
        raise ValueError("Missing native trajectory output file path from trajectory extraction.")
    path = Path(str(native_traj_file))
    if not path.exists():
        raise FileNotFoundError(f"Native trajectory output file not found: {path}")
    blob = torch.load(path, map_location="cpu")
    rows = blob.get("trajectories", []) if isinstance(blob, dict) else []
    out = []
    for one in rows:
        if not isinstance(one, dict):
            continue
        aid = one.get("agent_id")
        if aid is None:
            continue
        out.append(str(aid))
    return sorted(set(out))


def _build_native_calibration_set(
    raw_ds_path: str,
    excluded_user_ids: list[str],
    calibration_ratio: float = 0.02,
    calibration_target_users: Optional[int] = None,
    calibration_base_user_count: Optional[int] = None,
    allow_shorter: Optional[bool] = None,
) -> dict:
    ds_name = _dataset_name_from_raw_ds_path(raw_ds_path)
    out_dir = _processed_output_root(raw_ds_path) / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"cal_{ds_name}_native.pt"

    excluded = set(str(x) for x in excluded_user_ids)
    if calibration_base_user_count is None:
        test_size = int(len(excluded))
    else:
        test_size = int(calibration_base_user_count)
    if test_size <= 0:
        raise ValueError("No saved native test users available for calibration ratio baseline.")

    ratio = float(calibration_ratio)
    explicit_target = (
        int(calibration_target_users)
        if calibration_target_users is not None and int(calibration_target_users) > 0
        else None
    )
    if explicit_target is not None:
        target_users = int(explicit_target)
        target_mode = "explicit_users"
    else:
        target_users = max(1, int(round(float(test_size) * ratio)))
        target_mode = "ratio"
    ctx = _build_traj_context_for_calibration(
        raw_ds_path,
        shuffle_seed=42,
        sort_users_by_entries=True,
    )
    all_agents = list(ctx.get("ordered_agents", []))

    candidate_agents = [a for a in all_agents if str(a) not in excluded]
    if not candidate_agents:
        raise RuntimeError(
            "No non-overlapping calibration candidates available after excluding native test users."
        )

    if target_users > len(candidate_agents):
        logger.warning(
            "Calibration target users reduced from %d to %d due to non-overlap constraint.",
            int(target_users),
            int(len(candidate_agents)),
        )
        target_users = int(len(candidate_agents))

    ordered_agents = list(candidate_agents)
    if allow_shorter is None:
        allow_shorter_flag = bool(TRAJ_ALLOW_SHORTER)
    else:
        allow_shorter_flag = bool(allow_shorter)

    result = _extract_native_traj_for_calibration(
        parquet_dir=raw_ds_path,
        output_dir=str(out_dir),
        m=int(target_users),
        n_points=5000,
        allow_size_override=True,
        allow_shorter=allow_shorter_flag,
        include_error_range=_is_blogwatcher_dataset(raw_ds_path),
        ordered_agents=ordered_agents,
    )
    temp_path = Path(str(result.get("output_path", "")))
    if not temp_path.exists():
        raise FileNotFoundError(f"Native calibration temporary output missing: {temp_path}")

    if temp_path.resolve() != out_file.resolve():
        # Keep a single canonical calibration artifact name and avoid duplicate native files.
        try:
            temp_path.replace(out_file)
        except Exception:
            shutil.copy2(temp_path, out_file)
            try:
                temp_path.unlink()
            except Exception:
                pass
    else:
        out_file = temp_path

    # Backward-compat cleanup: remove stale temp-named native files in calibration dir.
    for stale in out_dir.glob("traj_native_*.pt"):
        try:
            if stale.resolve() == out_file.resolve():
                continue
            stale.unlink()
        except Exception:
            pass

    cal_blob = torch.load(out_file, map_location="cpu")
    cal_users = sorted(
        set(
            str(one.get("agent_id"))
            for one in cal_blob.get("trajectories", [])
            if isinstance(one, dict) and one.get("agent_id") is not None
        )
    )
    overlap = sorted(set(cal_users) & excluded)
    if overlap:
        raise RuntimeError(f"Calibration overlap detected with test users: {overlap[:20]}")

    return {
        "status": "ok",
        "path": str(out_file),
        "filename": out_file.name,
        "target_users": int(target_users),
        "target_mode": target_mode,
        "test_size": int(test_size),
        "ratio": float(ratio),
        "excluded_users": int(len(excluded)),
        "calibration_base_user_count": int(test_size),
        "candidate_users": int(len(candidate_agents)),
        "saved_users": int(len(cal_users)),
        "saved_user_ids": cal_users,
        "overlap_user_ids": [],
        "native_source_output": str(out_file),
        "n_trajectories": int(result.get("n_trajectories", 0)),
        "avg_length": int(result.get("avg_length", 0)),
        "median_length": int(result.get("median_length", 0)),
    }


def _build_calibration_debug_subset(
    raw_ds_path: str,
    calibration_native: Optional[dict],
    debug_users: int = 4,
) -> Optional[dict]:
    if not isinstance(calibration_native, dict):
        return None
    if calibration_native.get("status") != "ok":
        return None

    source_path = Path(str(calibration_native.get("path", "")))
    if not source_path.exists():
        logger.warning("Calibration debug subset skipped: source file missing: %s", source_path)
        return None

    debug_user_limit = max(1, int(debug_users))
    blob = torch.load(source_path, map_location="cpu")
    trajectories = blob.get("trajectories", [])
    if not isinstance(trajectories, list) or not trajectories:
        logger.warning("Calibration debug subset skipped: no trajectories in %s", source_path)
        return None

    selected_user_ids = []
    selected_trajs = []
    seen = set()
    for one in trajectories:
        if not isinstance(one, dict):
            continue
        aid = one.get("agent_id")
        if aid is None:
            continue
        aid_str = str(aid)
        if aid_str not in seen:
            if len(seen) >= debug_user_limit:
                continue
            seen.add(aid_str)
            selected_user_ids.append(aid_str)
        if aid_str in seen:
            selected_trajs.append(one)
        if len(seen) >= debug_user_limit and len(selected_trajs) >= len(seen):
            # One trajectory per user is expected here; keep generic behavior.
            pass

    if not selected_trajs:
        logger.warning("Calibration debug subset skipped: failed to select trajectories from %s", source_path)
        return None

    ds_name = _dataset_name_from_raw_ds_path(raw_ds_path)
    out_dir = _processed_output_root(raw_ds_path) / "calibration_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"cal_{ds_name}_native_debug.pt"

    out_blob = dict(blob)
    out_blob["trajectories"] = selected_trajs
    metadata = out_blob.get("metadata", {})
    if isinstance(metadata, dict):
        meta = dict(metadata)
    else:
        meta = {}
    lengths = [
        int(one.get("n_points", 0))
        for one in selected_trajs
        if isinstance(one, dict) and one.get("n_points") is not None
    ]
    if lengths:
        avg_len = int(round(float(np.mean(np.asarray(lengths, dtype=float)))))
        med_len = int(np.median(np.asarray(lengths, dtype=float)))
    else:
        avg_len = 0
        med_len = 0
    meta["n_trajectories"] = int(len(selected_trajs))
    meta["avg_length"] = int(avg_len)
    meta["median_length"] = int(med_len)
    meta["debug_subset_of"] = str(source_path)
    out_blob["metadata"] = meta

    torch.save(out_blob, out_file)

    return {
        "status": "ok",
        "path": str(out_file),
        "filename": out_file.name,
        "source_path": str(source_path),
        "target_users": int(debug_user_limit),
        "saved_users": int(len(set(selected_user_ids))),
        "saved_user_ids": selected_user_ids,
        "n_trajectories": int(len(selected_trajs)),
        "avg_length": int(avg_len),
        "median_length": int(med_len),
    }


# NOTE ABOUT SINGLE-FILE DATASETS AND LEAKAGE PREVENTION
# ------------------------------------------------------
# For one-file datasets, splitting by rows/chunks can leak trajectory identity
# between validation (calibration) and test.
# We split by AGENT ID deterministically so val-only and test-only remain
# disjoint even when run independently in separate commands.
def _stable_bucket_for_agent(agent_id: str, seed: int = 42) -> int:
    h = hashlib.sha1(f"{seed}:{agent_id}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 10000


def _build_single_file_agent_split(
    raw_ds_path: str,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    parquet_files = sorted(Path(raw_ds_path).glob("*.parquet"))
    if len(parquet_files) != 1:
        return {}
    file_path = parquet_files[0]
    agent_col = _detect_agent_column(raw_ds_path)
    agents = (
        pl.scan_parquet(str(file_path))
        .select(pl.col(agent_col).drop_nulls().cast(pl.Utf8).unique().sort())
        .collect()
        .get_column(agent_col)
        .to_list()
    )
    threshold = max(1, min(9999, int(round(float(val_ratio) * 10000))))
    val_agents = []
    test_agents = []
    for a in agents:
        if _stable_bucket_for_agent(str(a), seed=seed) < threshold:
            val_agents.append(str(a))
        else:
            test_agents.append(str(a))
    if not test_agents and val_agents:
        test_agents.append(val_agents.pop())
    if not val_agents and test_agents:
        val_agents.append(test_agents.pop())
    return {
        "file": file_path.name,
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "val_agents": set(val_agents),
        "test_agents": set(test_agents),
        "val_agent_count": int(len(val_agents)),
        "test_agent_count": int(len(test_agents)),
    }


def parquet_processor(K: int = 256, 
                     Q: int = 1, 
                     r: int = 5,
                     raw_ds_path: str = DEFAULT_RAW_DS_PATH,
                     kalman_max_agents_per_file: Optional[int] = 200,
                     run_traj_extraction: bool = True,
                     calibration_ratio: float = 0.02,
                     calibration_target_users: int = 100,
                     calibration_debug_users: int = 4,
                     map_padding_km: float = 5.0,
                     raw_map_path: Optional[str] = None,
                     run_map_slice: bool = True) -> dict:
   
    raw_path = Path(raw_ds_path)
    state_path = _state_path_from_raw_ds_path(raw_ds_path)
    _migrate_legacy_state_files(raw_ds_path)
    split_output_dirs = {s: _split_output_dir(raw_ds_path, s) for s in ["train", "val", "test"]}
    for split_dir in split_output_dirs.values():
        split_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all parquet files and sort by name
    parquet_files = sorted(raw_path.glob("*.parquet"))
    logger.info(f"Found {len(parquet_files)} parquet files")
    
    # Define splits
    train_files = parquet_files[:26]
    val_files = parquet_files[26:29]
    test_files = parquet_files[29:32]
    
    logger.info(f"Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    # Statistics
    chunk_counts = {
        "train": 0,
        "val": 0,
        "test": 0
    }
    split_boundaries = {
        "train": _empty_boundary_accumulator(),
        "val": _empty_boundary_accumulator(),
        "test": _empty_boundary_accumulator(),
    }
    split_used_users = {
        "train": set(),
        "val": set(),
        "test": set(),
    }
    
    corrupted_files = []  # Track corrupted files
    test_debug_pack = None
    
    # Process each split
    for split_name, file_list in [("train", train_files), 
                                    ("val", val_files), 
                                    ("test", test_files)]:
        logger.info(f"Processing {split_name} split ({len(file_list)} files)")
        
        # Process each file individually
        for file_idx, file_path in enumerate(file_list):
            logger.info(f"Processing file {file_idx + 1}/{len(file_list)}: {file_path.name}")
            
            # Step 1: Read single file
            ds_result = ds_reader([str(file_path)])
            ds_entry = ds_result['datasets'][0]
            
            # Step 2: Dice trajectories
            out_ds_dicer, ds_record = ds_dicer(
                ds_entry,
                K=K,
                Q=Q,
                extraction_cursor_path=str(_extraction_cursor_path_from_raw_ds_path(raw_ds_path)),
            )
            
            if not out_ds_dicer:
                logger.warning(f"No chunks generated from {file_path.name}")
                # Check if this was due to corruption by seeing if ds_record is empty
                if not ds_record.get('users'):
                    corrupted_files.append(file_path.name)
                del out_ds_dicer, ds_result, ds_entry
                _best_effort_memory_cleanup()
                continue
            split_used_users[split_name].update(str(u) for u in out_ds_dicer.keys())
            
            # Step 3: Assemble chunks
            out_ds_assemble = ds_assemble(
                ds_entry,
                out_ds_dicer,
                K=K,
                Q=Q,
                include_error_range=_is_blogwatcher_dataset(raw_ds_path),
            )
            logger.info(f"Assembled {len(out_ds_assemble)} chunks from {file_path.name}")
            _update_boundary_from_noisy_chunks(split_boundaries[split_name], out_ds_assemble)
            
            if len(out_ds_assemble) == 0:
                logger.warning(f"No valid chunks assembled from {file_path.name} (all skipped due to NaN GPS data)")
                del out_ds_assemble, out_ds_dicer, ds_result, ds_entry
                _best_effort_memory_cleanup()
                continue
            
            if split_name == "test":
                tensor_pack, n_saved = _build_test_pair_tensor_pack(
                    out_ds_assemble,
                    keep_timestamps_float64=False,
                )
                if n_saved == 0:
                    logger.warning(f"No test pairs to save for {file_path.name}")
                    del tensor_pack, out_ds_assemble, out_ds_dicer, ds_result, ds_entry
                    _best_effort_memory_cleanup()
                    continue
            else:
                # Step 4: ENU transform
                out_enu_transform = [enu_transform(chunk) for chunk in out_ds_assemble]
                logger.info(f"ENU transformed {len(out_enu_transform)} chunks")

                # Step 5: V labelizer
                out_v_labelizer = [v_labelizer(chunk) for chunk in out_enu_transform]

                # Step 6: Time sampling
                all_training_samples = []
                for chunk in out_v_labelizer:
                    samples = t_sampler(chunk, r=r)
                    all_training_samples.extend(samples)

                # === Step 7: Save tensorized dataset ===
                N = len(all_training_samples)
                if N == 0:
                    logger.warning(f"No training samples to save for {file_path.name}")
                    del all_training_samples, out_v_labelizer, out_enu_transform, out_ds_assemble, out_ds_dicer, ds_result, ds_entry
                    _best_effort_memory_cleanup()
                    continue

                K_local = len(all_training_samples[0]["X_t"])  # usually 256

                # allocate tensors
                X_t = torch.empty((N, K_local, 4), dtype=torch.float32)
                V = torch.empty((N, K_local, 2), dtype=torch.float32)
                t = torch.empty((N, 1), dtype=torch.float32)

                # fill tensors
                for i, s in enumerate(all_training_samples):
                    X_t[i] = torch.tensor(s["X_t"], dtype=torch.float32)
                    V[i] = torch.tensor(s["V"], dtype=torch.float32)
                    t[i, 0] = s["t"]

                tensor_pack = {"X_t": X_t, "V": V, "t": t}
                n_saved = int(N)

            if split_name == "test":
                output_file = _unique_path(
                    split_output_dirs[split_name] / _native_chunk_filename(int(n_saved))
                )
            else:
                output_file = split_output_dirs[split_name] / f"chunks_{ds_entry['name'].replace('.parquet', '')}.pt"
            torch.save(tensor_pack, output_file)
            with open(f"{output_file}.len", "w") as f:
                f.write(str(n_saved))

            chunk_counts[split_name] += int(n_saved)
            if "X_t" in tensor_pack:
                logger.info(f"Saved tensor dataset {tensor_pack['X_t'].shape} -> {output_file}")
                del tensor_pack, X_t, V, t, all_training_samples, out_v_labelizer, out_enu_transform, out_ds_assemble, out_ds_dicer, ds_result, ds_entry
            else:
                test_debug_pack = _append_test_chunk_debug(
                    test_debug_pack,
                    tensor_pack,
                    max_chunks=2,
                )
                logger.info(
                    "Saved test pair dataset %s -> %s",
                    tuple(tensor_pack["X1"].shape),
                    output_file,
                )
                del tensor_pack, out_ds_assemble, out_ds_dicer, ds_result, ds_entry
            _best_effort_memory_cleanup()
        
    # Final summary
    out_parquet_processor = {
        "status": "completed",
        "train_files": len(train_files),
        "val_files": len(val_files),
        "test_files": len(test_files),
        "total_chunks": chunk_counts,
        "corrupted_files": corrupted_files
    }
    
    logger.info("Parquet processing pipeline completed successfully")
    logger.info(f"Total chunks - Train: {chunk_counts['train']}, "
                f"Val: {chunk_counts['val']}, Test: {chunk_counts['test']}")
    
    if corrupted_files:
        logger.warning(f"Skipped {len(corrupted_files)} corrupted files: {corrupted_files}")

    traj_extraction = None
    native_test_user_ids = []
    calibration_excluded_user_ids = sorted(split_used_users["test"])
    calibration_native = None
    calibration_native_debug = None
    if run_traj_extraction:
        _best_effort_memory_cleanup()
        logger.info("Running trajectory extraction suites for %s", raw_ds_path)
        traj_extraction = run_traj_extraction_suites_isolated(raw_ds_path)
        native_out = _find_native_output_file_from_traj_extraction(traj_extraction)
        native_test_user_ids = _load_saved_native_user_ids(native_out)
        calibration_excluded_user_ids = sorted(
            set(calibration_excluded_user_ids) | set(native_test_user_ids)
        )
        calibration_native = _build_native_calibration_set(
            raw_ds_path=raw_ds_path,
            excluded_user_ids=calibration_excluded_user_ids,
            calibration_ratio=calibration_ratio,
            calibration_target_users=calibration_target_users,
            calibration_base_user_count=len(native_test_user_ids),
        )
        calibration_native_debug = _build_calibration_debug_subset(
            raw_ds_path=raw_ds_path,
            calibration_native=calibration_native,
            debug_users=calibration_debug_users,
        )
    chunk_test_debug_file = _save_chunk_test_debug(test_debug_pack, raw_ds_path)

    dataset_noisy_boundary_corners = _boundary_to_four_corners(split_boundaries["test"])
    map_process = _slice_map_pbf_to_boundary(
        raw_ds_path=raw_ds_path,
        boundary_corners=dataset_noisy_boundary_corners,
        map_padding_km=map_padding_km,
        raw_map_path=raw_map_path,
        run_map_slice=run_map_slice,
    )

    state_payload = {
        "updated_at": datetime.now().isoformat(),
        # Explicitly clear legacy Kalman params so downstream falls back to defaults.
        "kalman_rts_params": None,
        "parquet_processor": {
            "raw_ds_path": str(raw_ds_path),
            "train_files": len(train_files),
            "val_files": len(val_files),
            "test_files": len(test_files),
            "total_chunks": chunk_counts,
            "corrupted_files": corrupted_files,
            "boundary_noisy_corners": {
                "train": _boundary_to_four_corners(split_boundaries["train"]),
                "val": _boundary_to_four_corners(split_boundaries["val"]),
                "test": _boundary_to_four_corners(split_boundaries["test"]),
            },
            "dataset_noisy_boundary_corners": dataset_noisy_boundary_corners,
            "map_process": map_process,
            "trajectory_extraction": traj_extraction,
            "test_used_user_ids_chunk": sorted(split_used_users["test"]),
            "test_used_user_ids_traj_native": native_test_user_ids,
            "calibration_excluded_user_ids": calibration_excluded_user_ids,
            "calibration_native": calibration_native,
            "calibration_native_debug": calibration_native_debug,
            "chunk_test_debug_file": chunk_test_debug_file,
            "trajectory_extraction_datasets": _build_traj_extraction_state_append(traj_extraction),
        },
    }
    state_payload["corrupted_files"] = {
        "total_corrupted": len(corrupted_files),
        "files": corrupted_files,
    }
    _upsert_states_json(state_payload, state_path=str(state_path))
    out_parquet_processor["trajectory_extraction"] = traj_extraction
    out_parquet_processor["chunk_test_debug_file"] = chunk_test_debug_file
    out_parquet_processor["map_process"] = map_process
    
    return out_parquet_processor


def parquet_processor_test_only(
    K: int = 256,
    Q: int = 1,
    raw_ds_path: str = DEFAULT_RAW_DS_PATH,
    test_files: list[str] | None = None,
    single_file_val_ratio: float = 0.1,
    single_file_split_seed: int = 42,
    run_traj_extraction: bool = True,
    calibration_ratio: float = 0.02,
    calibration_target_users: int = 100,
    calibration_debug_users: int = 4,
    map_padding_km: float = 5.0,
    raw_map_path: Optional[str] = None,
    run_map_slice: bool = True,
) -> dict:
    """
    Generate test-only chunk datasets as direct noisy/clean chunk pairs.

    Args:
        test_files: list of parquet filenames (or full paths). If None, use the last parquet file in raw_ds_path.
    """
    raw_path = Path(raw_ds_path)
    state_path = _state_path_from_raw_ds_path(raw_ds_path)
    _migrate_legacy_state_files(raw_ds_path)
    processed_path = _split_output_dir(raw_ds_path, "test")
    processed_path.mkdir(parents=True, exist_ok=True)

    if test_files:
        file_list = [Path(p) for p in test_files]
        file_list = [p if p.is_absolute() else raw_path / p for p in file_list]
    else:
        parquet_files = sorted(raw_path.glob("*.parquet"))
        file_list = parquet_files[-1:]
    single_file_split = {}
    allowed_agents = None
    if test_files is None:
        single_file_split = _build_single_file_agent_split(
            raw_ds_path=raw_ds_path,
            val_ratio=single_file_val_ratio,
            seed=single_file_split_seed,
        )
        if single_file_split:
            file_list = [raw_path / single_file_split["file"]]
            allowed_agents = set(single_file_split["test_agents"])
            logger.info(
                "Single-file agent split active (test-only): val_agents=%d test_agents=%d",
                single_file_split["val_agent_count"],
                single_file_split["test_agent_count"],
            )

    logger.info(f"Processing test-only split ({len(file_list)} files)")

    chunk_count = 0
    corrupted_files = []
    test_boundary_chunk = _empty_boundary_accumulator()
    test_debug_pack = None
    used_user_ids_chunk = set()

    for file_idx, file_path in enumerate(file_list):
        logger.info(f"Processing file {file_idx + 1}/{len(file_list)}: {file_path.name}")

        ds_result = ds_reader([str(file_path)])
        ds_entry = ds_result['datasets'][0]

        out_ds_dicer, ds_record = ds_dicer(
            ds_entry,
            K=K,
            Q=Q,
            extraction_cursor_path=str(_extraction_cursor_path_from_raw_ds_path(raw_ds_path)),
            allowed_agents=allowed_agents,
        )
        if not out_ds_dicer:
            logger.warning(f"No chunks generated from {file_path.name}")
            if not ds_record.get('users'):
                corrupted_files.append(file_path.name)
            del out_ds_dicer, ds_result, ds_entry
            _best_effort_memory_cleanup()
            continue
        used_user_ids_chunk.update(str(u) for u in out_ds_dicer.keys())

        out_ds_assemble = ds_assemble(
            ds_entry,
            out_ds_dicer,
            K=K,
            Q=Q,
            include_error_range=_is_blogwatcher_dataset(raw_ds_path),
        )
        logger.info(f"Assembled {len(out_ds_assemble)} chunks from {file_path.name}")
        _update_boundary_from_noisy_chunks(test_boundary_chunk, out_ds_assemble)

        if len(out_ds_assemble) == 0:
            logger.warning(f"No valid chunks assembled from {file_path.name} (all skipped due to NaN GPS data)")
            del out_ds_assemble, out_ds_dicer, ds_result, ds_entry
            _best_effort_memory_cleanup()
            continue

        tensor_pack, n_saved = _build_test_pair_tensor_pack(
            out_ds_assemble,
            keep_timestamps_float64=True,
        )
        if n_saved == 0:
            logger.warning(f"No test pairs to save for {file_path.name}")
            del tensor_pack, out_ds_assemble, out_ds_dicer, ds_result, ds_entry
            _best_effort_memory_cleanup()
            continue

        output_file = _unique_path(processed_path / _native_chunk_filename(int(n_saved)))
        torch.save(tensor_pack, output_file)
        with open(f"{output_file}.len", "w") as f:
            f.write(str(n_saved))
        test_debug_pack = _append_test_chunk_debug(
            test_debug_pack,
            tensor_pack,
            max_chunks=2,
        )

        chunk_count += int(n_saved)
        logger.info(
            "Saved test pair dataset %s -> %s",
            tuple(tensor_pack["X1"].shape),
            output_file,
        )
        del tensor_pack, out_ds_assemble, out_ds_dicer, ds_result, ds_entry
        _best_effort_memory_cleanup()

    out_parquet_processor = {
        "status": "completed",
        "test_files": len(file_list),
        "total_chunks": {"test": chunk_count},
        "corrupted_files": corrupted_files,
    }

    if corrupted_files:
        logger.warning(f"Skipped {len(corrupted_files)} corrupted files: {corrupted_files}")

    traj_extraction = None
    native_test_user_ids = []
    calibration_excluded_user_ids = sorted(used_user_ids_chunk)
    calibration_native = None
    calibration_native_debug = None
    if run_traj_extraction:
        _best_effort_memory_cleanup()
        logger.info("Running trajectory extraction suites for %s", raw_ds_path)
        traj_extraction = run_traj_extraction_suites_isolated(raw_ds_path)
        native_out = _find_native_output_file_from_traj_extraction(traj_extraction)
        native_test_user_ids = _load_saved_native_user_ids(native_out)
        calibration_excluded_user_ids = sorted(
            set(calibration_excluded_user_ids) | set(native_test_user_ids)
        )
        calibration_native = _build_native_calibration_set(
            raw_ds_path=raw_ds_path,
            excluded_user_ids=calibration_excluded_user_ids,
            calibration_ratio=calibration_ratio,
            calibration_target_users=calibration_target_users,
            calibration_base_user_count=len(native_test_user_ids),
        )
        calibration_native_debug = _build_calibration_debug_subset(
            raw_ds_path=raw_ds_path,
            calibration_native=calibration_native,
            debug_users=calibration_debug_users,
        )
    chunk_test_debug_file = _save_chunk_test_debug(test_debug_pack, raw_ds_path)

    # Map slice boundary for test-only mode:
    # union chunk_test boundary + all trajectory-suite test outputs (full/debug), excluding calibration.
    test_only_boundary = _clone_boundary(test_boundary_chunk)
    traj_output_files = _collect_completed_traj_output_files(traj_extraction)
    traj_boundary_stats = _update_boundary_from_traj_output_files(test_only_boundary, traj_output_files)
    if int(traj_boundary_stats.get("files_total", 0)) > 0:
        logger.info(
            "Test-only map boundary sources: chunk_test + traj_suite files=%d processed=%d failed=%d trajectories=%d",
            int(traj_boundary_stats.get("files_total", 0)),
            int(traj_boundary_stats.get("files_processed", 0)),
            int(traj_boundary_stats.get("files_failed", 0)),
            int(traj_boundary_stats.get("trajectories_used", 0)),
        )

    test_chunk_boundary_corners = _boundary_to_four_corners(test_boundary_chunk)
    dataset_noisy_boundary_corners = _boundary_to_four_corners(test_only_boundary)
    map_process = _slice_map_pbf_to_boundary(
        raw_ds_path=raw_ds_path,
        boundary_corners=dataset_noisy_boundary_corners,
        map_padding_km=map_padding_km,
        raw_map_path=raw_map_path,
        run_map_slice=run_map_slice,
    )

    state_payload = {
        "updated_at": datetime.now().isoformat(),
        # Explicitly clear legacy Kalman params so downstream falls back to defaults.
        "kalman_rts_params": None,
        "parquet_processor": {
            "raw_ds_path": str(raw_ds_path),
            "test_only": {
                "test_files": [p.name for p in file_list],
                "total_chunks": {"test": int(chunk_count)},
                "corrupted_files": corrupted_files,
                "boundary_noisy_corners": test_chunk_boundary_corners,
                "boundary_noisy_corners_all_test_only": dataset_noisy_boundary_corners,
                "map_boundary_sources": {
                    "chunk_test_boundary": test_chunk_boundary_corners,
                    "traj_suite_output_files": traj_output_files,
                    "traj_suite_boundary_stats": traj_boundary_stats,
                },
            },
            "dataset_noisy_boundary_corners": dataset_noisy_boundary_corners,
            "map_process": map_process,
            "trajectory_extraction": traj_extraction,
            "used_user_ids_chunk": sorted(used_user_ids_chunk),
            "used_user_ids_traj_native": native_test_user_ids,
            "calibration_excluded_user_ids": calibration_excluded_user_ids,
            "calibration_native": calibration_native,
            "calibration_native_debug": calibration_native_debug,
            "chunk_test_debug_file": chunk_test_debug_file,
            "trajectory_extraction_datasets": _build_traj_extraction_state_append(traj_extraction),
            "single_file_agent_split": {
                "enabled": bool(single_file_split),
                "file": single_file_split.get("file"),
                "seed": single_file_split.get("seed"),
                "val_ratio": single_file_split.get("val_ratio"),
                "val_agent_count": single_file_split.get("val_agent_count"),
                "test_agent_count": single_file_split.get("test_agent_count"),
            },
        },
        "corrupted_files": {
            "total_corrupted": len(corrupted_files),
            "files": corrupted_files,
        },
    }
    _upsert_states_json(state_payload, state_path=str(state_path))
    out_parquet_processor["trajectory_extraction"] = traj_extraction
    out_parquet_processor["chunk_test_debug_file"] = chunk_test_debug_file
    out_parquet_processor["dataset_noisy_boundary_corners"] = dataset_noisy_boundary_corners
    out_parquet_processor["map_process"] = map_process

    return out_parquet_processor


def parquet_processor_val_only(
    K: int = 256,
    Q: int = 1,
    r: int = 5,
    raw_ds_path: str = DEFAULT_RAW_DS_PATH,
    val_files: list[str] | None = None,
    kalman_max_agents_per_file: Optional[int] = 200,
    single_file_val_ratio: float = 0.1,
    single_file_split_seed: int = 42,
    map_padding_km: float = 5.0,
    raw_map_path: Optional[str] = None,
    run_map_slice: bool = True,
) -> dict:
    """
    Generate validation-only chunk datasets.

    This updates dataset-scoped state_<dataset>.json via merge.
    """
    raw_path = Path(raw_ds_path)
    _migrate_legacy_state_files(raw_ds_path)
    processed_path = _split_output_dir(raw_ds_path, "val")
    processed_path.mkdir(parents=True, exist_ok=True)

    if val_files:
        file_list = [Path(p) for p in val_files]
        file_list = [p if p.is_absolute() else raw_path / p for p in file_list]
    else:
        parquet_files = sorted(raw_path.glob("*.parquet"))
        file_list = parquet_files[26:29]
    single_file_split = {}
    allowed_agents = None
    if val_files is None:
        single_file_split = _build_single_file_agent_split(
            raw_ds_path=raw_ds_path,
            val_ratio=single_file_val_ratio,
            seed=single_file_split_seed,
        )
        if single_file_split:
            file_list = [raw_path / single_file_split["file"]]
            allowed_agents = set(single_file_split["val_agents"])
            logger.info(
                "Single-file agent split active (val-only): val_agents=%d test_agents=%d",
                single_file_split["val_agent_count"],
                single_file_split["test_agent_count"],
            )

    if _is_blogwatcher_dataset(raw_ds_path):
        logger.info("BlogWatcher dataset detected: using uncertainty-bound schema normalization.")
    logger.info(f"Processing val-only split ({len(file_list)} files)")

    chunk_count = 0
    corrupted_files = []
    val_boundary = _empty_boundary_accumulator()

    for file_idx, file_path in enumerate(file_list):
        logger.info(f"Processing file {file_idx + 1}/{len(file_list)}: {file_path.name}")

        ds_result = ds_reader([str(file_path)])
        ds_entry = ds_result["datasets"][0]

        out_ds_dicer, ds_record = ds_dicer(
            ds_entry,
            K=K,
            Q=Q,
            extraction_cursor_path=str(_extraction_cursor_path_from_raw_ds_path(raw_ds_path)),
            allowed_agents=allowed_agents,
        )
        if not out_ds_dicer:
            logger.warning(f"No chunks generated from {file_path.name}")
            if not ds_record.get("users"):
                corrupted_files.append(file_path.name)
            del out_ds_dicer, ds_result, ds_entry
            _best_effort_memory_cleanup()
            continue

        out_ds_assemble = ds_assemble(ds_entry, out_ds_dicer, K=K, Q=Q)
        logger.info(f"Assembled {len(out_ds_assemble)} chunks from {file_path.name}")
        _update_boundary_from_noisy_chunks(val_boundary, out_ds_assemble)

        if len(out_ds_assemble) == 0:
            logger.warning(f"No valid chunks assembled from {file_path.name} (all skipped due to NaN GPS data)")
            del out_ds_assemble, out_ds_dicer, ds_result, ds_entry
            _best_effort_memory_cleanup()
            continue

        out_enu_transform = [enu_transform(chunk) for chunk in out_ds_assemble]
        out_v_labelizer = [v_labelizer(chunk) for chunk in out_enu_transform]

        all_training_samples = []
        for chunk in out_v_labelizer:
            samples = t_sampler(chunk, r=r)
            all_training_samples.extend(samples)

        N = len(all_training_samples)
        if N == 0:
            logger.warning(f"No validation samples to save for {file_path.name}")
            del all_training_samples, out_v_labelizer, out_enu_transform, out_ds_assemble, out_ds_dicer, ds_result, ds_entry
            _best_effort_memory_cleanup()
            continue

        K_local = len(all_training_samples[0]["X_t"])
        X_t = torch.empty((N, K_local, 4), dtype=torch.float32)
        V = torch.empty((N, K_local, 2), dtype=torch.float32)
        t = torch.empty((N, 1), dtype=torch.float32)

        for i, s in enumerate(all_training_samples):
            X_t[i] = torch.tensor(s["X_t"], dtype=torch.float32)
            V[i] = torch.tensor(s["V"], dtype=torch.float32)
            t[i, 0] = s["t"]

        tensor_pack = {"X_t": X_t, "V": V, "t": t}
        output_file = processed_path / f"chunks_{ds_entry['name'].replace('.parquet', '')}.pt"
        torch.save(tensor_pack, output_file)
        with open(f"{output_file}.len", "w") as f:
            f.write(str(N))

        chunk_count += int(N)
        logger.info(f"Saved tensor dataset {tensor_pack['X_t'].shape} -> {output_file}")
        del tensor_pack, X_t, V, t, all_training_samples, out_v_labelizer, out_enu_transform, out_ds_assemble, out_ds_dicer, ds_result, ds_entry
        _best_effort_memory_cleanup()

    uncertainty_dataset_stats = None
    if _is_blogwatcher_dataset(raw_ds_path):
        try:
            uncertainty_dataset_stats = estimate_uncertainty_dataset_stats_from_validation(
                raw_ds_path=raw_ds_path,
                file_paths=[str(p) for p in file_list],
                max_agents_per_file=kalman_max_agents_per_file,
            )
        except Exception as e:
            logger.warning("Uncertainty dataset stats estimation skipped: %s", e)

    dataset_noisy_boundary_corners = _boundary_to_four_corners(val_boundary)
    map_process = _slice_map_pbf_to_boundary(
        raw_ds_path=raw_ds_path,
        boundary_corners=dataset_noisy_boundary_corners,
        map_padding_km=map_padding_km,
        raw_map_path=raw_map_path,
        run_map_slice=run_map_slice,
    )

    state_payload = {
        "updated_at": datetime.now().isoformat(),
        # Explicitly clear legacy Kalman params so downstream falls back to defaults.
        "kalman_rts_params": None,
        "parquet_processor": {
            "raw_ds_path": str(raw_ds_path),
            "val_only": {
                "val_files": [p.name for p in file_list],
                "total_chunks": {"val": int(chunk_count)},
                "corrupted_files": corrupted_files,
                "boundary_noisy_corners": _boundary_to_four_corners(val_boundary),
            },
            "dataset_noisy_boundary_corners": dataset_noisy_boundary_corners,
            "map_process": map_process,
            "single_file_agent_split": {
                "enabled": bool(single_file_split),
                "file": single_file_split.get("file"),
                "seed": single_file_split.get("seed"),
                "val_ratio": single_file_split.get("val_ratio"),
                "val_agent_count": single_file_split.get("val_agent_count"),
                "test_agent_count": single_file_split.get("test_agent_count"),
            },
        },
    }
    if uncertainty_dataset_stats is not None:
        state_payload["uncertainty_dataset_stats"] = uncertainty_dataset_stats
    state_payload["corrupted_files"] = {
        "total_corrupted": len(corrupted_files),
        "files": corrupted_files,
    }
    _upsert_states_json(
        state_payload,
        state_path=str(_state_path_from_raw_ds_path(raw_ds_path)),
    )

    out_parquet_processor = {
        "status": "completed",
        "val_files": len(file_list),
        "total_chunks": {"val": int(chunk_count)},
        "corrupted_files": corrupted_files,
        "uncertainty_dataset_stats": uncertainty_dataset_stats,
        "map_process": map_process,
    }
    return out_parquet_processor


def parquet_processor_calibration_only(
    K: int = 256,
    Q: int = 1,
    r: int = 1,
    raw_ds_path: str = DEFAULT_RAW_DS_PATH,
    calibration_files: list[str] | None = None,
    output_dir: str | None = None,
    kalman_max_agents_per_file: Optional[int] = 200,
    single_file_val_ratio: float = 0.02,
    single_file_split_seed: int = 42,
    calibration_max_users: int = 16,
    calibration_max_chunks: int = 256,
) -> dict:
    """
    Calibration generation is disabled in parquet processor.
    Kept only for backward compatibility with older callers.
    """
    logger.warning(
        "Calibration split generation is disabled in parquet processor; returning fallback-only status."
    )
    return {
        "status": "disabled",
        "calibration_files": 0,
        "total_chunks": {"calibration": 0},
        "corrupted_files": [],
        "kalman_rts_params": None,
    }


def parquet_processor_uncertainty_bound_val_only(
    K: int = 256,
    Q: int = 1,
    r: int = 5,
    raw_ds_path: str = "./dataset/raw/BlogWatcher",
    val_files: list[str] | None = None,
    kalman_max_agents_per_file: Optional[int] = 200,
    single_file_val_ratio: float = 0.1,
    single_file_split_seed: int = 42,
    map_padding_km: float = 5.0,
    raw_map_path: Optional[str] = None,
    run_map_slice: bool = True,
) -> dict:
    """
    Uncertainty-bound val-only processor path (BlogWatcher schema).
    This switches processing route by directory without touching parquet files.
    """
    logger.info("Using uncertainty-bound val-only processor path for %s", raw_ds_path)
    return parquet_processor_val_only(
        K=K,
        Q=Q,
        r=r,
        raw_ds_path=raw_ds_path,
        val_files=val_files,
        kalman_max_agents_per_file=kalman_max_agents_per_file,
        single_file_val_ratio=single_file_val_ratio,
        single_file_split_seed=single_file_split_seed,
        map_padding_km=map_padding_km,
        raw_map_path=raw_map_path,
        run_map_slice=run_map_slice,
    )


def _hash_tensor_en(x: torch.Tensor) -> bytes:
    x = x.contiguous()
    return hashlib.sha1(x.numpy().tobytes()).digest()


def _sample_is_valid(X_t: torch.Tensor, V: torch.Tensor, t: torch.Tensor) -> bool:
    if torch.isnan(X_t).any() or torch.isinf(X_t).any():
        return False
    if torch.isnan(V).any() or torch.isinf(V).any():
        return False
    if torch.isnan(t).any() or torch.isinf(t).any():
        return False
    is_start_vals = torch.unique(X_t[:, 3])
    for v in is_start_vals:
        if v.item() not in (0.0, 1.0):
            return False
    return True


def build_quick_val_sets(
    val_dir: str | None = None,
    output_dir: str | None = None,
    raw_ds_path: str = DEFAULT_RAW_DS_PATH,
    small_size=10_000,
    mid_size=50_000,
    big_size=90_000,
    seed=42,
):
    """
    Build three non-overlapping quick-val sets from val shards.
    Files are named:
      - quick_val_chunk_10k.pt
      - quick_val_chunk_50k.pt
      - quick_val_chunk_90k.pt
    """
    resolved_val_dir = val_dir or str(_split_output_dir(raw_ds_path, "val"))
    resolved_output_dir = output_dir or resolved_val_dir

    val_files = [f for f in glob.glob(os.path.join(resolved_val_dir, "*.pt")) if not f.endswith(".len")]
    if not val_files:
        raise FileNotFoundError(f"[build_quick_val_sets] No .pt files found in {resolved_val_dir}")

    X_t_all, V_all, t_all = [], [], []
    for f in sorted(val_files):
        data = torch.load(f, map_location="cpu")
        X_t_all.append(data["X_t"])
        V_all.append(data["V"])
        t_all.append(data["t"])

    X_t = torch.cat(X_t_all, dim=0)
    V = torch.cat(V_all, dim=0)
    t = torch.cat(t_all, dim=0)

    n = len(X_t)
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()

    targets = [
        ("small", small_size),
        ("mid", mid_size),
        ("big", big_size),
    ]
    selected = {name: [] for name, _ in targets}
    hash_set = set()

    for idx in perm:
        X_s = X_t[idx]
        V_s = V[idx]
        t_s = t[idx]

        if not _sample_is_valid(X_s, V_s, t_s):
            continue

        h = _hash_tensor_en(X_s[:, :2])
        if h in hash_set:
            continue

        for name, size in targets:
            if len(selected[name]) < size:
                selected[name].append(idx)
                hash_set.add(h)
                break

        if all(len(selected[name]) >= size for name, size in targets):
            break

    for name, size in targets:
        if len(selected[name]) < size:
            logger.warning(
                f"[build_quick_val_sets] Insufficient samples for {name}: {len(selected[name])} < {size}. "
                f"Using available samples."
            )

    def _save(name: str, size: int):
        idx_tensor = torch.tensor(selected[name], dtype=torch.long)
        pack = {
            "X_t": X_t.index_select(0, idx_tensor),
            "V": V.index_select(0, idx_tensor),
            "t": t.index_select(0, idx_tensor),
        }
        out_count = int(idx_tensor.numel())
        out_name = f"quick_val_chunk_{out_count // 1000}k.pt"
        out_path = os.path.join(resolved_output_dir, out_name)
        torch.save(pack, out_path)
        print(f"[build_quick_val_sets] Saved {name} set: {out_path} (N={len(idx_tensor)})")

    _save("small", small_size)
    _save("mid", mid_size)
    _save("big", big_size)


def shuffle_train_pt_pairwise(train_dir="./dataset/processed/NUMOSIM_Kanto/train", seed=42):
    """
    RAM-safe shuffle: load two train files at a time, shuffle, split back.
    Pairing uses a "least-mated" policy and runs for 2 * ceil(log2(N)) rounds.
    NOTE: Does NOT touch val/test directories.
    """
    train_files = [f for f in glob.glob(os.path.join(train_dir, "*.pt")) if not f.endswith(".len")]
    if not train_files:
        raise FileNotFoundError(f"[shuffle_train_pt_pairwise] No .pt files found in {train_dir}")

    files = sorted(train_files)
    n_files = len(files)
    rounds = int(2 * math.ceil(math.log2(n_files))) if n_files > 1 else 0

    # Track how often each pair has been mixed.
    mate_counts = {f: {} for f in files}

    def _mate_count(a, b):
        return mate_counts[a].get(b, 0)

    def _record_mate(a, b):
        mate_counts[a][b] = _mate_count(a, b) + 1
        mate_counts[b][a] = _mate_count(b, a) + 1

    # Greedy "least-mated" pairing with deterministic tie-break.
    for r in range(rounds):
        remaining = set(files)
        if len(remaining) % 2 == 1:
            # Rotate the "bye" file to avoid starving one file of mixing.
            total_mates = {f: sum(mate_counts[f].values()) for f in remaining}
            bye = max(sorted(remaining), key=lambda x: (total_mates[x], x))
            remaining.remove(bye)

        while len(remaining) > 1:
            a = sorted(remaining)[0]
            remaining.remove(a)

            # Choose b with the smallest mate count with a (tie-break by name).
            b = min(remaining, key=lambda x: (_mate_count(a, x), x))
            remaining.remove(b)

            d1 = torch.load(a, map_location="cpu")
            d2 = torch.load(b, map_location="cpu")

            n1 = d1["X_t"].shape[0]
            n2 = d2["X_t"].shape[0]

            X_t = torch.cat([d1["X_t"], d2["X_t"]], dim=0)
            V = torch.cat([d1["V"], d2["V"]], dim=0)
            t = torch.cat([d1["t"], d2["t"]], dim=0)

            g_pair = torch.Generator()
            g_pair.manual_seed(seed + r * 1000 + n1 + n2)
            perm = torch.randperm(n1 + n2, generator=g_pair)

            X_t = X_t[perm]
            V = V[perm]
            t = t[perm]

            pack1 = {"X_t": X_t[:n1], "V": V[:n1], "t": t[:n1]}
            pack2 = {"X_t": X_t[n1:], "V": V[n1:], "t": t[n1:]}

            torch.save(pack1, a)
            torch.save(pack2, b)
            with open(f"{a}.len", "w") as lf1:
                lf1.write(str(n1))
            with open(f"{b}.len", "w") as lf2:
                lf2.write(str(n2))

            _record_mate(a, b)

        # If odd file count, last file is left as-is for this round.

    print(f"[shuffle_train_pt_pairwise] Shuffled train files with {rounds} rounds (seed={seed})")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parquet processor CLI")
    parser.add_argument(
        "--mode",
        choices=["full", "val-only", "test-only"],
        default="full",
        help="Processing mode.",
    )
    parser.add_argument("--K", type=int, default=256, help="Chunk size.")
    parser.add_argument("--Q", type=int, default=1, help="Overlap size.")
    parser.add_argument("--r", type=int, default=5, help="Number of t samples per chunk.")
    parser.add_argument(
        "--raw-ds-path",
        type=str,
        default=DEFAULT_RAW_DS_PATH,
        help="Raw parquet dataset directory (e.g., ./dataset/raw/NUMOSIM_Kanto).",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Optional parquet files for val-only/test-only (names or paths).",
    )
    parser.add_argument(
        "--skip-shuffle",
        action="store_true",
        help="Skip train shard shuffling after full mode.",
    )
    parser.add_argument(
        "--skip-quick-val",
        action="store_true",
        help="Skip quick val set build after full mode.",
    )
    parser.add_argument(
        "--skip-traj-extraction",
        action="store_true",
        help="Skip trajectory extraction suite generation (full + debug).",
    )
    parser.add_argument(
        "--skip-map-slice",
        action="store_true",
        help="Skip automatic map slicing from raw PBF into dataset/map_processed.",
    )
    parser.add_argument(
        "--raw-map-path",
        type=str,
        default=None,
        help="Optional raw .pbf map path to use for slicing. If omitted, auto-resolve from dataset/raw_map.",
    )
    parser.add_argument(
        "--map-padding-km",
        type=float,
        default=5.0,
        help="Padding in kilometers applied to noisy GPS bbox before slicing map (default: 5.0).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for shuffle/quick-val.")
    parser.add_argument(
        "--kalman-max-agents",
        type=int,
        default=200,
        help="Deprecated/ignored: parquet processor no longer estimates Kalman-RTS parameters.",
    )
    parser.add_argument(
        "--single-file-calibration-ratio",
        "--single-file-val-ratio",
        dest="single_file_val_ratio",
        type=float,
        default=0.02,
        help=(
            "Deprecated alias; kept for compatibility with existing commands."
        ),
    )
    parser.add_argument(
        "--single-file-split-seed",
        type=int,
        default=42,
        help="Deterministic seed for single-file agent split (still used by test-only logic).",
    )
    parser.add_argument(
        "--calibration-ratio",
        type=float,
        default=0.02,
        help="Fallback calibration ratio against native test users when --calibration-target-users <= 0.",
    )
    parser.add_argument(
        "--calibration-target-users",
        type=int,
        default=100,
        help="Calibration native target user count (default: 100). Set <=0 to use --calibration-ratio.",
    )
    parser.add_argument(
        "--calibration-debug-users",
        type=int,
        default=4,
        help="Debug calibration subset user count copied from real calibration set (default: 4).",
    )
    return parser


def _run_cli(args: argparse.Namespace) -> dict:
    if not (0.0 < float(args.single_file_val_ratio) < 1.0):
        raise ValueError("--single-file-calibration-ratio/--single-file-val-ratio must be in (0, 1)")
    if int(args.calibration_target_users) <= 0 and not (0.0 < float(args.calibration_ratio) < 1.0):
        raise ValueError("--calibration-ratio must be in (0, 1)")
    if float(args.map_padding_km) < 0.0:
        raise ValueError("--map-padding-km must be >= 0")

    if args.mode == "full":
        result = parquet_processor(
            K=args.K,
            Q=args.Q,
            r=args.r,
            raw_ds_path=args.raw_ds_path,
            kalman_max_agents_per_file=args.kalman_max_agents,
            run_traj_extraction=not args.skip_traj_extraction,
            calibration_ratio=args.calibration_ratio,
            calibration_target_users=args.calibration_target_users,
            calibration_debug_users=args.calibration_debug_users,
            map_padding_km=args.map_padding_km,
            raw_map_path=args.raw_map_path,
            run_map_slice=not args.skip_map_slice,
        )
        if not args.skip_shuffle:
            shuffle_train_pt_pairwise(
                train_dir=str(_split_output_dir(args.raw_ds_path, "train")),
                seed=args.seed,
            )
        if not args.skip_quick_val:
            build_quick_val_sets(raw_ds_path=args.raw_ds_path, seed=args.seed)
        return result

    if args.mode == "val-only":
        if _is_blogwatcher_dataset(args.raw_ds_path):
            return parquet_processor_uncertainty_bound_val_only(
                K=args.K,
                Q=args.Q,
                r=args.r,
                raw_ds_path=args.raw_ds_path,
                val_files=args.files,
                kalman_max_agents_per_file=args.kalman_max_agents,
                single_file_val_ratio=args.single_file_val_ratio,
                single_file_split_seed=args.single_file_split_seed,
                map_padding_km=args.map_padding_km,
                raw_map_path=args.raw_map_path,
                run_map_slice=not args.skip_map_slice,
            )
        return parquet_processor_val_only(
            K=args.K,
            Q=args.Q,
            r=args.r,
            raw_ds_path=args.raw_ds_path,
            val_files=args.files,
            kalman_max_agents_per_file=args.kalman_max_agents,
            single_file_val_ratio=args.single_file_val_ratio,
            single_file_split_seed=args.single_file_split_seed,
            map_padding_km=args.map_padding_km,
            raw_map_path=args.raw_map_path,
            run_map_slice=not args.skip_map_slice,
        )

    return parquet_processor_test_only(
        K=args.K,
        Q=args.Q,
        raw_ds_path=args.raw_ds_path,
        test_files=args.files,
        single_file_val_ratio=args.single_file_val_ratio,
        single_file_split_seed=args.single_file_split_seed,
        run_traj_extraction=not args.skip_traj_extraction,
        calibration_ratio=args.calibration_ratio,
        calibration_target_users=args.calibration_target_users,
        calibration_debug_users=args.calibration_debug_users,
        map_padding_km=args.map_padding_km,
        raw_map_path=args.raw_map_path,
        run_map_slice=not args.skip_map_slice,
    )



if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Backward compatibility: no args behaves like legacy full pipeline.
    parser = _build_arg_parser()
    cli_args = parser.parse_args()
    result = _run_cli(cli_args)

    print("\n" + "="*50)
    print("Pipeline completed!")
    print(f"Status: {result['status']}")
    if "train_files" in result:
        print(
            f"Files processed: {result['train_files']} train, "
            f"{result['val_files']} val, {result['test_files']} test"
        )
    elif "calibration_files" in result and "test_files" in result:
        print(f"Files processed: {result['calibration_files']} calibration, {result['test_files']} test")
    elif "val_files" in result and "test_files" in result:
        print(f"Files processed: {result['val_files']} val, {result['test_files']} test")
    elif "calibration_files" in result:
        print(f"Files processed: {result['calibration_files']} calibration")
    elif "val_files" in result:
        print(f"Files processed: {result['val_files']} val")
    elif "test_files" in result:
        print(f"Files processed: {result['test_files']} test")
    print(f"Total chunks: {result['total_chunks']}")
    if result.get('corrupted_files'):
        print(f"Corrupted files skipped: {len(result['corrupted_files'])}")
        print(f"  {result['corrupted_files']}")
    print("="*50)

    
