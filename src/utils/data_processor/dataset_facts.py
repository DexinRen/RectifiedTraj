#!/usr/bin/env python3
"""
Summarize processed trajectory dataset facts.

This helper describes the generated dataset files themselves. It does not use
benchmark caps such as M or N; every trajectory and point stored in each input
file is included in the reported facts.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from pymap3d import geodetic2enu


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STATE_FACT_PATH = REPO_ROOT / "dataset" / "state" / "fact.json"


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(path)


def _finite_1d(values: Iterable[Any]) -> np.ndarray:
    if isinstance(values, np.ndarray):
        arr = values.astype(np.float64, copy=False).reshape(-1)
    else:
        arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _stats(values: Iterable[Any]) -> dict[str, float | int | None]:
    arr = _finite_1d(values)
    if arr.size == 0:
        return {
            "n": 0,
            "min": None,
            "med": None,
            "avg": None,
            "p95": None,
            "max": None,
        }
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "med": float(np.median(arr)),
        "avg": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _stats_from_counter(counter: Counter) -> dict[str, float | int | None]:
    if not counter:
        return _stats([])
    n_total = int(sum(int(c) for c in counter.values()))
    if n_total <= 0:
        return _stats([])

    values_sorted = sorted((float(v), int(c)) for v, c in counter.items() if int(c) > 0)
    total_sum = float(sum(v * c for v, c in values_sorted))

    def value_at(index: int) -> float:
        target = int(max(0, min(index, n_total - 1)))
        acc = 0
        for value, count in values_sorted:
            acc += count
            if acc > target:
                return float(value)
        return float(values_sorted[-1][0])

    def percentile(q: float) -> float:
        pos = (n_total - 1) * float(q) / 100.0
        lo = int(np.floor(pos))
        hi = int(np.ceil(pos))
        vlo = value_at(lo)
        vhi = value_at(hi)
        return float(vlo + (vhi - vlo) * (pos - lo))

    return {
        "n": n_total,
        "min": float(values_sorted[0][0]),
        "med": percentile(50.0),
        "avg": float(total_sum / n_total),
        "p95": percentile(95.0),
        "max": float(values_sorted[-1][0]),
    }


def _partial_sample_time_stats_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    raw = metadata.get("sample_interval_stats_sec")
    out = _stats([])
    if not isinstance(raw, dict):
        out["source"] = "missing"
        return out
    out["n"] = int(float(raw.get("n", 0) or 0))
    if raw.get("median") is not None:
        out["med"] = float(raw["median"])
    if raw.get("mean") is not None:
        out["avg"] = float(raw["mean"])
    out["source"] = "metadata_partial"
    return out


def _as_numpy_2d(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    return arr[:, :2].astype(np.float64, copy=False)


def _as_numpy_1d(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.ndim == 0:
        return None
    return arr.reshape(-1).astype(np.float64, copy=False)


def _first_present(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return None


def _get_noisy_lonlat(record: dict[str, Any]) -> np.ndarray | None:
    return _as_numpy_2d(
        _first_present(record, ("data", "noisy_lonlat", "noisy_gps", "X1"))
    )


def _get_ref_lonlat(record: dict[str, Any]) -> np.ndarray | None:
    return _as_numpy_2d(
        _first_present(record, ("label", "clean_lonlat", "clean_gps", "ref_gps", "X0"))
    )


def _get_timestamps(record: dict[str, Any]) -> np.ndarray | None:
    return _as_numpy_1d(_first_present(record, ("timestamp", "timestamps")))


def _get_dt_sec(record: dict[str, Any]) -> np.ndarray | None:
    direct = _as_numpy_1d(_first_present(record, ("dt_sec", "time_gap_sec", "sample_dt_sec")))
    if direct is not None:
        return direct
    for key in ("data", "X1", "X0", "X_t"):
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy()
        else:
            arr = np.asarray(value)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return np.asarray(arr[:, 2], dtype=np.float64).reshape(-1)
    return None


def _get_radius(record: dict[str, Any]) -> np.ndarray | None:
    return _as_numpy_1d(_first_present(record, ("error_range", "radius", "accuracy")))


def _l1_lonlat_error_m(noisy_lonlat: np.ndarray, ref_lonlat: np.ndarray) -> np.ndarray:
    n = min(int(noisy_lonlat.shape[0]), int(ref_lonlat.shape[0]))
    if n <= 0:
        return np.empty((0,), dtype=np.float64)
    noisy = noisy_lonlat[:n]
    ref = ref_lonlat[:n]
    ref_lon = float(ref[0, 0])
    ref_lat = float(ref[0, 1])
    e_noisy, n_noisy, _ = geodetic2enu(
        noisy[:, 1],
        noisy[:, 0],
        np.zeros(n, dtype=np.float64),
        ref_lat,
        ref_lon,
        0.0,
    )
    e_ref, n_ref, _ = geodetic2enu(
        ref[:, 1],
        ref[:, 0],
        np.zeros(n, dtype=np.float64),
        ref_lat,
        ref_lon,
        0.0,
    )
    return np.abs(np.asarray(e_noisy) - np.asarray(e_ref)) + np.abs(
        np.asarray(n_noisy) - np.asarray(n_ref)
    )


def _positive_timestamp_diffs_sec(timestamps: np.ndarray) -> np.ndarray:
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    ts = ts[np.isfinite(ts)]
    if ts.size < 2:
        return np.empty((0,), dtype=np.float64)
    diffs = np.diff(ts)
    return diffs[np.isfinite(diffs) & (diffs > 0)]


def _load_trajectory_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and isinstance(payload.get("trajectories"), list):
        return payload["trajectories"], payload.get("metadata", {}) or {}
    if isinstance(payload, list):
        records = [x for x in payload if isinstance(x, dict)]
        return records, {}
    raise ValueError(f"Unsupported trajectory file format: {path}")


def _update_counter_from_tensor_values(
    counter: Counter,
    tensor: torch.Tensor,
    *,
    batch_rows: int = 1024,
) -> None:
    if tensor.ndim < 1:
        return
    rows = int(tensor.shape[0]) if tensor.ndim >= 2 else 1
    if tensor.ndim == 1:
        batches = [tensor]
    else:
        batches = [tensor[start : start + batch_rows] for start in range(0, rows, batch_rows)]
    for batch in batches:
        arr = batch.detach().cpu().reshape(-1).numpy().astype(np.float64, copy=False)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if arr.size == 0:
            continue
        values, counts = np.unique(np.round(arr, 6), return_counts=True)
        counter.update(dict(zip(values.tolist(), counts.astype(int).tolist())))


def summarize_train_chunk_file(path: str | Path) -> tuple[str, dict[str, Any]]:
    path = Path(path)
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported train chunk file format: {path}")

    source = payload.get("X_t")
    source_name = "X_t"
    if source is None:
        source = payload.get("X1")
        source_name = "X1"
    if not isinstance(source, torch.Tensor) or source.ndim != 3 or source.shape[-1] < 3:
        raise ValueError(f"Train chunk file has no 3D tensor with dt channel: {path}")

    dt_counter: Counter = Counter()
    _update_counter_from_tensor_values(dt_counter, source[:, :, 2])
    n_samples = int(source.shape[0])
    chunk_len = int(source.shape[1])
    summary = {
        "source_dataset": path.parts[path.parts.index("processed") + 1]
        if "processed" in path.parts and path.parts.index("processed") + 1 < len(path.parts)
        else None,
        "file": _repo_relative(path),
        "kind": "train_chunks",
        "dt_channel_source": source_name,
        "sample_time": {
            **_stats_from_counter(dt_counter),
            "source": "dt_sec_channel",
        },
        "num_point": int(n_samples * chunk_len),
        "num_record": n_samples,
        "chunk_len": chunk_len,
        "point_per_record": _stats(np.full((n_samples,), chunk_len, dtype=np.float64)),
    }
    return "train_chunks", summary


def summarize_trajectory_file(path: str | Path, *, kind: str = "auto") -> tuple[str, dict[str, Any]]:
    path = Path(path)
    records, metadata = _load_trajectory_records(path)

    point_counts: list[int] = []
    sample_diffs: list[np.ndarray] = []
    exact_errors: list[np.ndarray] = []
    uncertainty_distances: list[np.ndarray] = []
    radii: list[np.ndarray] = []
    missing_timestamp_traj = 0
    missing_ref_traj = 0
    has_radius = False
    has_dt_samples = False
    has_timestamp_samples = False

    for record in records:
        noisy = _get_noisy_lonlat(record)
        ref = _get_ref_lonlat(record)
        dt_sec = _get_dt_sec(record)
        ts = _get_timestamps(record)
        radius = _get_radius(record)

        n_candidates = []
        if noisy is not None:
            n_candidates.append(int(noisy.shape[0]))
        if ref is not None:
            n_candidates.append(int(ref.shape[0]))
        if dt_sec is not None:
            n_candidates.append(int(dt_sec.shape[0]))
        if ts is not None:
            n_candidates.append(int(ts.shape[0]))
        if radius is not None:
            n_candidates.append(int(radius.shape[0]))
            has_radius = True
        if record.get("n_points") is not None:
            n_candidates.append(int(record["n_points"]))
        n = min(n_candidates) if n_candidates else 0
        if n <= 0:
            continue
        point_counts.append(int(n))

        if dt_sec is not None:
            dt_arr = np.asarray(dt_sec[:n], dtype=np.float64).reshape(-1)
            sample_diffs.append(dt_arr[np.isfinite(dt_arr) & (dt_arr > 0)])
            has_dt_samples = True
        elif ts is None:
            missing_timestamp_traj += 1
        else:
            sample_diffs.append(_positive_timestamp_diffs_sec(ts[:n]))
            has_timestamp_samples = True

        if ref is None or noisy is None:
            missing_ref_traj += 1
            continue

        err = _l1_lonlat_error_m(noisy[:n], ref[:n])
        if radius is not None:
            uncertainty_distances.append(err)
            radii.append(radius[: min(n, int(radius.shape[0]))])
        else:
            exact_errors.append(err)

    sample_values = np.concatenate(sample_diffs) if sample_diffs else np.empty((0,))
    if sample_values.size > 0:
        sample_time = _stats(sample_values)
        if has_dt_samples and has_timestamp_samples:
            sample_time["source"] = "dt_sec_and_timestamps"
        elif has_dt_samples:
            sample_time["source"] = "dt_sec"
        else:
            sample_time["source"] = "timestamps"
    else:
        sample_time = _partial_sample_time_stats_from_metadata(metadata)

    inferred_kind = "uncertainty" if has_radius else "exact"
    out_kind = inferred_kind if kind == "auto" else str(kind)
    summary: dict[str, Any] = {
        "source_dataset": path.parts[path.parts.index("processed") + 1]
        if "processed" in path.parts and path.parts.index("processed") + 1 < len(path.parts)
        else None,
        "file": _repo_relative(path),
        "kind": out_kind,
        "metadata": {
            "sample_time_label": metadata.get("sample_time_label")
            or metadata.get("interval_label")
            or metadata.get("sampler"),
            "n_trajectories": metadata.get("n_trajectories"),
            "total_points": metadata.get("total_points"),
        },
        "sample_time": sample_time,
        "num_point": int(sum(point_counts)),
        "num_traj": int(len(point_counts)),
        "point_per_traj": _stats(point_counts),
        "missing_timestamp_traj": int(missing_timestamp_traj),
        "missing_ref_traj": int(missing_ref_traj),
    }

    if out_kind == "uncertainty":
        distance_values = (
            np.concatenate(uncertainty_distances)
            if uncertainty_distances
            else np.empty((0,), dtype=np.float64)
        )
        radius_values = (
            np.concatenate(radii) if radii else np.empty((0,), dtype=np.float64)
        )
        summary["distance_to_ref_l1"] = _stats(distance_values)
        summary["radius"] = _stats(radius_values)
        if distance_values.size and radius_values.size:
            n_cov = min(distance_values.size, radius_values.size)
            summary["coverage_rate_l1_le_radius"] = float(
                np.mean(distance_values[:n_cov] <= radius_values[:n_cov])
            )
    else:
        error_values = (
            np.concatenate(exact_errors)
            if exact_errors
            else np.empty((0,), dtype=np.float64)
        )
        summary["error_per_point_l1"] = _stats(error_values)

    return out_kind, summary


def discover_dataset_files(dataset: str | Path, *, include_debug: bool = False) -> list[Path]:
    root = Path(dataset)
    if root.is_file():
        return [root]

    if root.name == "train" and root.is_dir():
        search_root = root
    elif (root / "test" / "traj_test").is_dir():
        search_root = root / "test" / "traj_test"
    else:
        search_root = root

    files = sorted(search_root.rglob("*.pt"))
    if include_debug:
        return files
    return [p for p in files if "debug" not in {part.lower() for part in p.parts}]


def build_dataset_facts(dataset: str | Path, *, include_debug: bool = False) -> dict[str, Any]:
    facts: dict[str, Any] = {
        "version": 1,
        "generated_at": datetime.now().isoformat(),
        "dataset": _repo_relative(Path(dataset)),
        "units": {
            "sample_time": "seconds",
            "point_per_traj": "points",
            "error_per_point_l1": "meters",
            "distance_to_ref_l1": "meters",
            "radius": "meters",
        },
        "exact": {},
        "uncertainty": {},
        "train_chunks": {},
    }

    for path in discover_dataset_files(dataset, include_debug=include_debug):
        if path.name.startswith("chunks_") or path.parent.name == "train":
            kind, summary = summarize_train_chunk_file(path)
        else:
            kind, summary = summarize_trajectory_file(path, kind="auto")
        facts[kind][_repo_relative(path)] = summary
    return facts


def merge_into_state_fact(new_facts: dict[str, Any], state_path: str | Path = DEFAULT_STATE_FACT_PATH) -> dict[str, Any]:
    path = Path(state_path)
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except Exception:
            existing = {}

    merged = {
        "version": 1,
        "updated_at": datetime.now().isoformat(),
        "units": new_facts.get("units", existing.get("units", {})),
        "exact": dict(existing.get("exact", {}) if isinstance(existing.get("exact"), dict) else {}),
        "uncertainty": dict(
            existing.get("uncertainty", {}) if isinstance(existing.get("uncertainty"), dict) else {}
        ),
        "train_chunks": dict(
            existing.get("train_chunks", {}) if isinstance(existing.get("train_chunks"), dict) else {}
        ),
    }
    for kind in ("exact", "uncertainty", "train_chunks"):
        entries = new_facts.get(kind, {})
        if isinstance(entries, dict):
            merged[kind].update(entries)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return merged


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report processed dataset facts.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Processed dataset root, traj_test directory, or one trajectory .pt file.",
    )
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument(
        "--update-state",
        action="store_true",
        help="Merge facts into dataset/state/fact.json.",
    )
    parser.add_argument(
        "--state-path",
        default=str(DEFAULT_STATE_FACT_PATH),
        help="State fact path used with --update-state.",
    )
    parser.add_argument(
        "--include-debug",
        action="store_true",
        help="Include debug trajectory files.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    facts = build_dataset_facts(args.dataset, include_debug=bool(args.include_debug))
    text = json.dumps(facts, indent=2, sort_keys=True) + "\n"
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")

    if args.update_state:
        merge_into_state_fact(facts, args.state_path)


if __name__ == "__main__":
    main()
