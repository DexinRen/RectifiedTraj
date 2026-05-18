#!/usr/bin/env python3
"""
Capture region-focused NUMOSIM trajectories into native trajectory PT format.

This script scans all raw NUMOSIM parquet files, ranks agents by how many points
fall inside a lon/lat box, clips the selected trajectories to only the in-region
points, and saves the result in the same native schema used by trajectory test
datasets so denoising models and baselines can consume it directly.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np
import polars as pl

warnings.filterwarnings(
    "ignore",
    message="the `streaming` parameter was deprecated",
    category=DeprecationWarning,
)

try:
    from .traj_extractor import (
        FIXED_TRAJ_POINTS,
        _assert_not_calibration_or_processed_source,
        _detect_column_map,
        _load_agent_arrays_for_sampling,
        _save_processed_trajectory_dataset,
        data_processor,
    )
except ImportError:
    from traj_extractor import (
        FIXED_TRAJ_POINTS,
        _assert_not_calibration_or_processed_source,
        _detect_column_map,
        _load_agent_arrays_for_sampling,
        _save_processed_trajectory_dataset,
        data_processor,
    )


LOGGER = logging.getLogger(__name__)
DEFAULT_RAW_DS_PATH = "./dataset/raw/NUMOSIM_Kanto"
DEFAULT_OUTPUT_DIR = "./dataset/processed/mini_map"
DEFAULT_TARGET_AGENTS = 200
DEFAULT_TEST_SPLIT_PARQUET_COUNT = 3
DEFAULT_MIN_LON = 139.4
DEFAULT_MAX_LON = 139.6
DEFAULT_MIN_LAT = 35.4
DEFAULT_MAX_LAT = 35.6
DEFAULT_SAVE_MODE = "native-from-first-in-region"


def _selection_mode_name(*, stop_when_full: bool) -> str:
    return "stop_when_full" if bool(stop_when_full) else "exact_top_k"


def _output_name_stem(*, save_mode: str, stop_when_full: bool, prefix: str) -> str:
    save_mode_token = str(save_mode).replace("-", "_")
    if bool(stop_when_full):
        return f"{prefix}_{save_mode_token}_stop_when_full"
    return f"{prefix}_{save_mode_token}"


def _bar(done: int, total: int, width: int = 28) -> str:
    total_i = max(1, int(total))
    done_i = max(0, min(int(done), total_i))
    filled = int(round((done_i / total_i) * int(width)))
    return "[" + ("#" * filled) + ("-" * (int(width) - filled)) + "]"


class _LiveProgressBar:
    def __init__(self, *, total: int, label: str) -> None:
        self.total = int(max(1, total))
        self.label = str(label)
        self._started = False

    def update(
        self,
        *,
        done: int,
        saved: int,
        skipped: int,
        last_inside_points: int,
        last_saved_points: int,
        elapsed_sec: float,
    ) -> None:
        line = (
            f"[{self.label}] {_bar(done, self.total)} "
            f"{int(done)}/{int(self.total)} "
            f"saved={int(saved)} skipped={int(skipped)} "
            f"last_inside={int(last_inside_points)} "
            f"last_saved={int(last_saved_points)} "
            f"elapsed={float(elapsed_sec):.1f}s"
        )
        if not self._started:
            sys.stdout.write(line)
            self._started = True
        else:
            sys.stdout.write("\r" + line)
        sys.stdout.flush()

    def finish(self) -> None:
        if self._started:
            sys.stdout.write("\n")
            sys.stdout.flush()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture the top-K NUMOSIM agents by point count inside a lon/lat "
            "region and save native trajectories for denoising evaluation."
        )
    )
    parser.add_argument(
        "--raw-ds-path",
        type=str,
        default=DEFAULT_RAW_DS_PATH,
        help="Raw NUMOSIM parquet directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the captured PT and JSON summary.",
    )
    parser.add_argument(
        "--target-agents",
        type=int,
        default=DEFAULT_TARGET_AGENTS,
        help="Maximum number of agents to keep in the capture pool.",
    )
    parser.add_argument(
        "--min-lon",
        type=float,
        default=DEFAULT_MIN_LON,
        help="Region minimum longitude.",
    )
    parser.add_argument(
        "--max-lon",
        type=float,
        default=DEFAULT_MAX_LON,
        help="Region maximum longitude.",
    )
    parser.add_argument(
        "--min-lat",
        type=float,
        default=DEFAULT_MIN_LAT,
        help="Region minimum latitude.",
    )
    parser.add_argument(
        "--max-lat",
        type=float,
        default=DEFAULT_MAX_LAT,
        help="Region maximum latitude.",
    )
    parser.add_argument(
        "--region-field",
        choices=["noisy", "clean"],
        default="noisy",
        help="Field used to count whether points fall inside the bbox.",
    )
    parser.add_argument(
        "--save-mode",
        choices=["native-from-first-in-region", "from-first-in-region", "full", "inside-region-only"],
        default=DEFAULT_SAVE_MODE,
        help=(
            "How to save selected agent trajectories. "
            f"'native-from-first-in-region' starts at the first in-region point and keeps the "
            f"next {int(FIXED_TRAJ_POINTS)} consecutive points; "
            "'from-first-in-region' starts at the first in-region point and keeps the "
            "continuous suffix; "
            "'full' keeps the continuous native trajectory from its beginning; "
            "'inside-region-only' saves only points inside the bbox and can splice visits."
        ),
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=2,
        help="Minimum clipped points required to keep a saved trajectory.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=0,
        help="Optional cap within the selected test split; 0 means use all selected test files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable per-file progress logging.",
    )
    parser.add_argument(
        "--stop-when-full",
        action="store_true",
        help=(
            "Stop immediately once the pool first reaches --target-agents, "
            "instead of continuing the exact top-K replacement pass."
        ),
    )
    return parser


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalize_agent_id(agent_id: Any) -> Any:
    agent_id = _python_scalar(agent_id)
    if isinstance(agent_id, str):
        text = agent_id.strip()
        if text.isdigit():
            try:
                return int(text)
            except Exception:
                return text
        return text
    if isinstance(agent_id, (int, np.integer)):
        return int(agent_id)
    return agent_id


def _bbox_slug(min_lon: float, max_lon: float, min_lat: float, max_lat: float) -> str:
    return (
        f"lon_{min_lon:.5f}_{max_lon:.5f}_lat_{min_lat:.5f}_{max_lat:.5f}"
        .replace("-", "m")
        .replace(".", "p")
    )


def _region_mask(
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
) -> np.ndarray:
    return (
        np.isfinite(lon)
        & np.isfinite(lat)
        & (lon >= float(min_lon))
        & (lon <= float(max_lon))
        & (lat >= float(min_lat))
        & (lat <= float(max_lat))
    )


def _scan_file_region_hits(
    parquet_path: Path,
    *,
    column_map: Dict[str, str],
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    region_field: str,
) -> Dict[Any, int]:
    agent_col = column_map["agent"]
    if region_field == "clean":
        lon_col = column_map["longitude"]
        lat_col = column_map["latitude"]
    else:
        lon_col = column_map["longitude_n"]
        lat_col = column_map["latitude_n"]

    lazy = (
        pl.scan_parquet(str(parquet_path))
        .filter(
            pl.col(agent_col).is_not_null()
            & pl.col(lon_col).is_not_null()
            & pl.col(lat_col).is_not_null()
            & pl.col(lon_col).is_finite()
            & pl.col(lat_col).is_finite()
            & (pl.col(lon_col) >= float(min_lon))
            & (pl.col(lon_col) <= float(max_lon))
            & (pl.col(lat_col) >= float(min_lat))
            & (pl.col(lat_col) <= float(max_lat))
        )
        .group_by(agent_col)
        .agg(pl.len().alias("inside_points"))
        .sort("inside_points", descending=True)
    )
    try:
        try:
            df = lazy.collect(streaming=True)
        except TypeError:
            df = lazy.collect()
    except Exception as exc:
        LOGGER.warning("[map_capture] skipping unreadable parquet %s: %s", parquet_path.name, exc)
        raise

    out: Dict[Any, int] = {}
    if len(df) == 0:
        return out
    for row in df.iter_rows(named=True):
        agent_id = _normalize_agent_id(row[agent_col])
        inside_points = int(row["inside_points"])
        if inside_points <= 0:
            continue
        out[agent_id] = inside_points
    return out


def _collect_candidates(
    *,
    raw_ds_path: str,
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    region_field: str,
    limit_files: int,
    verbose: bool,
) -> tuple[dict[Any, dict[str, Any]], list[Path], Dict[str, str], dict[str, Any]]:
    _assert_not_calibration_or_processed_source(raw_ds_path)
    column_map = _detect_column_map(raw_ds_path)
    all_parquet_paths = sorted(Path(raw_ds_path).glob("*.parquet"))
    test_split_paths = all_parquet_paths[-int(DEFAULT_TEST_SPLIT_PARQUET_COUNT):]
    if int(limit_files) > 0:
        parquet_paths = test_split_paths[-int(limit_files):]
    else:
        parquet_paths = test_split_paths
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {raw_ds_path}")
    LOGGER.info(
        "[map_capture] using test split files=%d/%d: %s",
        len(parquet_paths),
        int(DEFAULT_TEST_SPLIT_PARQUET_COUNT),
        ", ".join(path.name for path in parquet_paths),
    )

    candidates: dict[Any, dict[str, Any]] = {}
    first_seen_counter = 0
    corrupted_files: list[str] = []
    files_processed = 0

    for idx, parquet_path in enumerate(parquet_paths, start=1):
        try:
            file_hits = _scan_file_region_hits(
                parquet_path,
                column_map=column_map,
                min_lon=min_lon,
                max_lon=max_lon,
                min_lat=min_lat,
                max_lat=max_lat,
                region_field=region_field,
            )
            files_processed += 1
        except Exception:
            corrupted_files.append(parquet_path.name)
            continue
        if verbose or idx == 1 or idx == len(parquet_paths) or (idx % 25 == 0):
            LOGGER.info(
                "[map_capture] scan file %d/%d %s matched_agents=%d",
                idx,
                len(parquet_paths),
                parquet_path.name,
                len(file_hits),
            )
        for agent_id, inside_points in file_hits.items():
            rec = candidates.get(agent_id)
            if rec is None:
                rec = {
                    "agent_id": agent_id,
                    "inside_points": 0,
                    "hit_files": [],
                    "first_seen_rank": first_seen_counter,
                }
                candidates[agent_id] = rec
                first_seen_counter += 1
            rec["inside_points"] = int(rec["inside_points"]) + int(inside_points)
            rec["hit_files"].append(str(parquet_path))

    scan_stats = {
        "files_total": int(len(parquet_paths)),
        "files_processed": int(files_processed),
        "files_failed": int(len(corrupted_files)),
        "corrupted_files": corrupted_files,
        "test_split_file_count": int(DEFAULT_TEST_SPLIT_PARQUET_COUNT),
        "selected_test_files": [path.name for path in parquet_paths],
    }
    if corrupted_files:
        LOGGER.warning(
            "[map_capture] skipped %d unreadable parquet files",
            len(corrupted_files),
        )
    return candidates, parquet_paths, column_map, scan_stats


def _select_top_agents(
    candidates: dict[Any, dict[str, Any]],
    *,
    target_agents: int,
) -> list[dict[str, Any]]:
    selected = sorted(
        candidates.values(),
        key=lambda rec: (
            -int(rec["inside_points"]),
            int(rec["first_seen_rank"]),
            str(rec["agent_id"]),
        ),
    )
    return selected[: max(0, int(target_agents))]


def _sort_candidate_agents(candidates: dict[Any, dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        candidates.values(),
        key=lambda rec: (
            -int(rec["inside_points"]),
            int(rec["first_seen_rank"]),
            str(rec["agent_id"]),
        ),
    )


def _insert_into_region_pool(
    pool: list[dict[str, Any]],
    candidate: dict[str, Any],
    *,
    target_agents: int,
) -> bool:
    def _pool_key(rec: dict[str, Any]) -> tuple[int, int, str]:
        return (
            int(rec["inside_points_segment"]),
            int(rec["saved_points"]),
            str(rec["agent_id"]),
        )

    if len(pool) < int(target_agents):
        pool.append(candidate)
        pool.sort(key=_pool_key, reverse=True)
        return True

    if _pool_key(candidate) <= _pool_key(pool[-1]):
        return False

    pool[-1] = candidate
    pool.sort(key=_pool_key, reverse=True)
    return True


def _extract_clipped_processed_trajectory(
    *,
    agent_id: Any,
    hit_files: list[str],
    region_field: str,
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    column_map: Dict[str, str],
    min_points: int,
    save_mode: str,
    target_points: int,
) -> dict[str, Any] | None:
    agent_data = _load_agent_arrays_for_sampling(
        agent_id=agent_id,
        parquet_paths=hit_files,
        max_rows=None,
        column_map=column_map,
        include_error_range=False,
        verbose_progress=False,
    )
    if agent_data is None:
        return None

    if region_field == "clean":
        mask = _region_mask(
            np.asarray(agent_data["longitude"], dtype=np.float64),
            np.asarray(agent_data["latitude"], dtype=np.float64),
            min_lon=min_lon,
            max_lon=max_lon,
            min_lat=min_lat,
            max_lat=max_lat,
        )
    else:
        mask = _region_mask(
            np.asarray(agent_data["longitude_n"], dtype=np.float64),
            np.asarray(agent_data["latitude_n"], dtype=np.float64),
            min_lon=min_lon,
            max_lon=max_lon,
            min_lat=min_lat,
            max_lat=max_lat,
        )

    inside_points = int(np.sum(mask))
    if inside_points < int(min_points):
        return None

    hit_idxs = np.flatnonzero(mask)
    start_idx = int(hit_idxs[0])
    target_points_i = int(target_points)

    if str(save_mode) == "native-from-first-in-region":
        end_idx = int(start_idx + target_points_i)
        if end_idx > int(len(agent_data["timestamp"])):
            return None
        seg_mask = mask[start_idx:end_idx]
        extracted = {
            "agent_id": _normalize_agent_id(agent_id),
            "n_points": int(target_points_i),
            "longitude_n": np.asarray(agent_data["longitude_n"])[start_idx:end_idx],
            "latitude_n": np.asarray(agent_data["latitude_n"])[start_idx:end_idx],
            "longitude": np.asarray(agent_data["longitude"])[start_idx:end_idx],
            "latitude": np.asarray(agent_data["latitude"])[start_idx:end_idx],
            "timestamp": np.asarray(agent_data["timestamp"])[start_idx:end_idx],
        }
        processed = data_processor(extracted)
        processed["inside_points"] = int(np.sum(seg_mask))
        processed["inside_points_total"] = int(inside_points)
        processed["first_inside_index"] = int(start_idx)
        processed["saved_points"] = int(processed["n_points"])
        return processed

    if str(save_mode) == "inside-region-only":
        extracted = {
            "agent_id": _normalize_agent_id(agent_id),
            "n_points": int(inside_points),
            "longitude_n": np.asarray(agent_data["longitude_n"])[mask],
            "latitude_n": np.asarray(agent_data["latitude_n"])[mask],
            "longitude": np.asarray(agent_data["longitude"])[mask],
            "latitude": np.asarray(agent_data["latitude"])[mask],
            "timestamp": np.asarray(agent_data["timestamp"])[mask],
        }
    elif str(save_mode) == "from-first-in-region":
        extracted = {
            "agent_id": _normalize_agent_id(agent_id),
            "n_points": int(len(agent_data["timestamp"]) - start_idx),
            "longitude_n": np.asarray(agent_data["longitude_n"])[start_idx:],
            "latitude_n": np.asarray(agent_data["latitude_n"])[start_idx:],
            "longitude": np.asarray(agent_data["longitude"])[start_idx:],
            "latitude": np.asarray(agent_data["latitude"])[start_idx:],
            "timestamp": np.asarray(agent_data["timestamp"])[start_idx:],
        }
    else:
        extracted = {
            "agent_id": _normalize_agent_id(agent_id),
            "n_points": int(len(agent_data["timestamp"])),
            "longitude_n": np.asarray(agent_data["longitude_n"]),
            "latitude_n": np.asarray(agent_data["latitude_n"]),
            "longitude": np.asarray(agent_data["longitude"]),
            "latitude": np.asarray(agent_data["latitude"]),
            "timestamp": np.asarray(agent_data["timestamp"]),
        }
    processed = data_processor(extracted)
    processed["inside_points"] = int(inside_points)
    processed["inside_points_total"] = int(inside_points)
    processed["first_inside_index"] = int(start_idx)
    processed["saved_points"] = int(processed["n_points"])
    return processed


def run_map_capture(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.target_agents) <= 0:
        raise ValueError("--target-agents must be > 0")
    if int(args.min_points) <= 0:
        raise ValueError("--min-points must be > 0")
    if str(args.save_mode) not in {
        "native-from-first-in-region",
        "from-first-in-region",
        "full",
        "inside-region-only",
    }:
        raise ValueError(
            "--save-mode must be one of: native-from-first-in-region, from-first-in-region, "
            "full, inside-region-only"
        )
    if float(args.min_lon) >= float(args.max_lon):
        raise ValueError("--min-lon must be smaller than --max-lon")
    if float(args.min_lat) >= float(args.max_lat):
        raise ValueError("--min-lat must be smaller than --max-lat")

    candidates, parquet_paths, column_map, scan_stats = _collect_candidates(
        raw_ds_path=str(args.raw_ds_path),
        min_lon=float(args.min_lon),
        max_lon=float(args.max_lon),
        min_lat=float(args.min_lat),
        max_lat=float(args.max_lat),
        region_field=str(args.region_field),
        limit_files=int(args.limit_files),
        verbose=bool(args.verbose),
    )
    candidate_rows = _sort_candidate_agents(candidates)
    if not candidate_rows:
        raise RuntimeError("No agents contain points inside the requested region")

    selection_mode = _selection_mode_name(stop_when_full=bool(args.stop_when_full))

    LOGGER.info(
        "[map_capture] candidates=%d target_pool=%d selection_mode=%s raw_files_scanned=%d/%d",
        len(candidates),
        int(args.target_agents),
        selection_mode,
        int(scan_stats["files_processed"]),
        int(scan_stats["files_total"]),
    )

    processed_pool: list[dict[str, Any]] = []
    extract_started_at = time.time()
    live = _LiveProgressBar(total=len(candidate_rows), label="map_capture:extract")
    skipped_after_selection = 0
    last_inside_points = 0
    last_saved_points = 0
    live.update(
        done=0,
        saved=0,
        skipped=0,
        last_inside_points=0,
        last_saved_points=0,
        elapsed_sec=0.0,
    )
    for rank, rec in enumerate(candidate_rows, start=1):
        if len(processed_pool) >= int(args.target_agents):
            if bool(args.stop_when_full):
                LOGGER.info(
                    "[map_capture] early stop: pool filled in stop-when-full mode (%d)",
                    int(args.target_agents),
                )
                break
            worst_kept = int(processed_pool[-1]["inside_points_segment"])
            if str(args.save_mode) == "native-from-first-in-region" and worst_kept >= int(FIXED_TRAJ_POINTS):
                LOGGER.info(
                    "[map_capture] early stop: pool floor reached native maximum (%d)",
                    int(FIXED_TRAJ_POINTS),
                )
                break
            current_upper_bound = int(rec["inside_points"])
            if current_upper_bound <= worst_kept:
                LOGGER.info(
                    "[map_capture] early stop: remaining candidates cannot exceed pool floor "
                    "(upper_bound=%d floor=%d)",
                    current_upper_bound,
                    worst_kept,
                )
                break
        processed = _extract_clipped_processed_trajectory(
            agent_id=rec["agent_id"],
            hit_files=list(rec["hit_files"]),
            region_field=str(args.region_field),
            min_lon=float(args.min_lon),
            max_lon=float(args.max_lon),
            min_lat=float(args.min_lat),
            max_lat=float(args.max_lat),
            column_map=column_map,
            min_points=int(args.min_points),
            save_mode=str(args.save_mode),
            target_points=int(FIXED_TRAJ_POINTS),
        )
        if processed is None:
            skipped_after_selection += 1
            last_inside_points = int(rec["inside_points"])
            last_saved_points = 0
            LOGGER.warning(
                "[map_capture] skipping agent=%s after clip; insufficient valid points",
                rec["agent_id"],
            )
            live.update(
                done=rank,
                saved=len(processed_pool),
                skipped=skipped_after_selection,
                last_inside_points=last_inside_points,
                last_saved_points=last_saved_points,
                elapsed_sec=time.time() - extract_started_at,
            )
            continue
        inserted = _insert_into_region_pool(
            processed_pool,
            {
                "agent_id": rec["agent_id"],
                "inside_points_segment": int(processed["inside_points"]),
                "inside_points_total": int(rec["inside_points"]),
                "saved_points": int(processed["saved_points"]),
                "first_inside_index": int(processed["first_inside_index"]),
                "hit_files": int(len(rec["hit_files"])),
                "trajectory": processed,
            },
            target_agents=int(args.target_agents),
        )
        if not inserted:
            skipped_after_selection += 1
        last_inside_points = int(processed["inside_points"])
        last_saved_points = int(processed["saved_points"])
        live.update(
            done=rank,
            saved=len(processed_pool),
            skipped=skipped_after_selection,
            last_inside_points=last_inside_points,
            last_saved_points=last_saved_points,
            elapsed_sec=time.time() - extract_started_at,
        )

    live.finish()

    if not processed_pool:
        raise RuntimeError("No clipped trajectories met the save criteria")

    processed_rows = []
    selected_summary = []
    for out_rank, rec in enumerate(processed_pool, start=1):
        processed_rows.append(rec["trajectory"])
        selected_summary.append(
            {
                "rank": int(out_rank),
                "agent_id": rec["agent_id"],
                "inside_points": int(rec["inside_points_segment"]),
                "inside_points_total": int(rec["inside_points_total"]),
                "saved_points": int(rec["saved_points"]),
                "first_inside_index": int(rec["first_inside_index"]),
                "hit_files": int(rec["hit_files"]),
            }
        )

    lengths = [int(one["n_points"]) for one in processed_rows]
    avg_length = int(round(float(np.mean(lengths)))) if lengths else 0
    median_length = int(np.median(lengths)) if lengths else 0
    out_dir = Path(str(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    slug = _bbox_slug(
        float(args.min_lon),
        float(args.max_lon),
        float(args.min_lat),
        float(args.max_lat),
    )

    metadata_extra = {
        "sampler": "map_capture",
        "sample_time_label": "map_capture",
        "source_dataset": "NUMOSIM_Kanto",
        "source_raw_path": str(args.raw_ds_path),
        "region_field": str(args.region_field),
        "selection_metric": "inside_region_point_count",
        "selection_mode": selection_mode,
        "save_mode": str(args.save_mode),
        "target_points_per_trajectory": int(FIXED_TRAJ_POINTS)
        if str(args.save_mode) == "native-from-first-in-region"
        else None,
        "avg_length": int(avg_length),
        "scan_files": int(scan_stats["files_processed"]),
        "scan_files_total_requested": int(scan_stats["files_total"]),
        "scan_files_failed": int(scan_stats["files_failed"]),
        "corrupted_files": list(scan_stats["corrupted_files"]),
        "test_split_file_count": int(scan_stats["test_split_file_count"]),
        "selected_test_files": list(scan_stats["selected_test_files"]),
        "candidate_agents": int(len(candidates)),
        "region_bbox": {
            "min_lon": float(args.min_lon),
            "max_lon": float(args.max_lon),
            "min_lat": float(args.min_lat),
            "max_lat": float(args.max_lat),
        },
    }
    output_path = _save_processed_trajectory_dataset(
        processed_trajs=processed_rows,
        output_dir=str(out_dir),
        prefix="traj_map_capture",
        target_m=int(args.target_agents),
        n_points=int(median_length),
        metadata_extra=metadata_extra,
        filename_override=(
            f"{_output_name_stem(save_mode=str(args.save_mode), stop_when_full=bool(args.stop_when_full), prefix='traj_map_capture')}_"
            f"{len(processed_rows)}_{median_length}_{slug}.pt"
        ),
    )

    summary_path = out_dir / (
        f"{_output_name_stem(save_mode=str(args.save_mode), stop_when_full=bool(args.stop_when_full), prefix='map_capture_summary')}_{slug}.json"
    )
    summary = {
        "status": "completed",
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "source_dataset": "NUMOSIM_Kanto",
        "source_raw_path": str(args.raw_ds_path),
        "output_dir": str(out_dir),
        "region_field": str(args.region_field),
        "selection_mode": selection_mode,
        "save_mode": str(args.save_mode),
        "region_bbox": metadata_extra["region_bbox"],
        "target_agents": int(args.target_agents),
        "min_points": int(args.min_points),
        "raw_files_scanned": int(scan_stats["files_processed"]),
        "raw_files_total_requested": int(scan_stats["files_total"]),
        "raw_files_failed": int(scan_stats["files_failed"]),
        "corrupted_files": list(scan_stats["corrupted_files"]),
        "test_split_file_count": int(scan_stats["test_split_file_count"]),
        "selected_test_files": list(scan_stats["selected_test_files"]),
        "candidate_agents": int(len(candidates)),
        "selected_agents": int(len(processed_rows)),
        "saved_trajectories": int(len(processed_rows)),
        "target_points_per_trajectory": int(FIXED_TRAJ_POINTS)
        if str(args.save_mode) == "native-from-first-in-region"
        else None,
        "avg_length": int(avg_length),
        "median_length": int(median_length),
        "min_length": int(min(lengths)) if lengths else 0,
        "max_length": int(max(lengths)) if lengths else 0,
        "selected": selected_summary,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    LOGGER.info("[map_capture] saved trajectories -> %s", output_path)
    LOGGER.info("[map_capture] saved summary -> %s", summary_path)
    return summary


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    result = run_map_capture(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
