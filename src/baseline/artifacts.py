from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re


# ================================================================
# === State File Resolution Helpers
# ================================================================
def _safe_dataset_token(name: str | None) -> str | None:
    if not name:
        return None
    token = str(name).strip()
    if not token:
        return None
    for sep in ["/", "\\", ":", "."]:
        token = token.replace(sep, "_")
    token = "_".join(part for part in token.split("_") if part)
    return token or None


def _state_candidate_files(
    dataset_name_hint: str | None,
    state_dir: str,
    fallback_dataset: str,
    strict_dataset_hint: bool = False,
) -> list[Path]:
    """
    Candidate order:
    1) exact state_<dataset>.json
    2) fuzzy matches containing dataset token
    3) fallback dataset state
    4) all remaining state files
    """
    root = Path(state_dir)
    all_states = sorted(root.glob("state_*.json"))
    out: list[Path] = []

    def _push(path: Path) -> None:
        if path not in out:
            out.append(path)

    hint = _safe_dataset_token(dataset_name_hint)
    if hint:
        _push(root / f"state_{hint}.json")
        hint_lower = hint.lower()
        for path in all_states:
            if hint_lower in path.stem.lower():
                _push(path)
        if strict_dataset_hint:
            return out

    fallback = _safe_dataset_token(fallback_dataset)
    if fallback:
        _push(root / f"state_{fallback}.json")

    for path in all_states:
        _push(path)
    return out


def _resolve_path(raw_path: str | None) -> str | None:
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        return None
    return str(path)


def _tokenize(text: str | None) -> list[str]:
    if not text:
        return []
    return [t for t in re.split(r"[^a-z0-9]+", str(text).lower()) if t]


def _dataset_alias_tokens(dataset_name_hint: str | None, fallback_dataset: str) -> list[str]:
    base = _safe_dataset_token(dataset_name_hint) or _safe_dataset_token(fallback_dataset) or ""
    tokens = _tokenize(base)
    aliases = set(tokens)
    if base:
        aliases.add(base.lower())

    # Dataset-specific aliases used by current project datasets.
    key = base.lower()
    if key == "numosim_kanto":
        aliases.update({"japan", "kanto", "tokyo", "numosim"})
    elif key == "blogwatcher":
        aliases.update({"japan", "tokyo", "kanto", "blogwatcher"})
    elif key == "pol":
        aliases.update({"georgia", "atlanta", "pol"})
    return sorted(a for a in aliases if a)


def _score_map_candidate(path: Path, aliases: list[str]) -> tuple[int, int, int]:
    """
    Return a sortable score tuple:
    - higher primary score is better
    - larger file size is better
    - stable name tie-breaker
    """
    stem = path.stem.lower()
    name = path.name.lower()
    score = 0

    # Prefer PBF first for map-matching stacks (Valhalla/OSRM ingest pbf natively).
    if name.endswith(".osm.pbf") or name.endswith(".pbf"):
        score += 40
    elif name.endswith(".osm.xml") or name.endswith(".xml"):
        score += 20

    for alias in aliases:
        if stem == f"map_{alias}" or stem == alias or stem == f"{alias}-latest":
            score += 120
        elif f"map_{alias}" in stem or f"{alias}-latest" in stem:
            score += 80
        elif alias in stem:
            score += 30

    # Avoid tiny/debug maps unless they are the only option.
    if any(flag in stem for flag in ("tiny", "mini", "debug", "sample")):
        score -= 50

    try:
        size = int(path.stat().st_size)
    except Exception:
        size = -1
    return (score, size, -len(path.name))


def _discover_map_file(
    dataset_name_hint: str | None,
    fallback_dataset: str,
    preferred_raw_map_dir: str | None = None,
) -> str | None:
    aliases = _dataset_alias_tokens(dataset_name_hint, fallback_dataset)
    exts = ("*.osm.pbf", "*.pbf", "*.osm.xml", "*.xml")

    def _scan_dirs(dirs: list[Path]) -> str | None:
        seen: set[Path] = set()
        scored: list[tuple[tuple[int, int, int], Path]] = []
        for d in dirs:
            if not d.exists() or not d.is_dir():
                continue
            for pattern in exts:
                for p in d.glob(pattern):
                    rp = p.resolve()
                    if rp in seen:
                        continue
                    seen.add(rp)
                    scored.append((_score_map_candidate(rp, aliases), rp))
        if not scored:
            return None
        scored.sort(key=lambda x: x[0], reverse=True)
        return str(scored[0][1])

    # 1) Prefer processed dataset maps first when a sliced PBF exists.
    processed_dirs = [Path("./dataset/map_processed"), Path("./dataset/map")]
    best_processed = _scan_dirs(processed_dirs)
    if best_processed and str(best_processed).lower().endswith(".pbf"):
        return best_processed

    # 2) Then fallback to raw map inventory.
    preferred_dirs = []
    if preferred_raw_map_dir:
        preferred_dirs.append(Path(preferred_raw_map_dir))
    preferred_dirs.append(Path("./dataset/raw_map"))
    best_raw = _scan_dirs(preferred_dirs)
    if best_raw:
        return best_raw

    # 3) If only non-PBF processed map exists, use it as last fallback.
    if best_processed:
        return best_processed

    # 4) Last resort: scan any explicitly provided dir.
    for d in [Path(preferred_raw_map_dir)] if preferred_raw_map_dir else []:
        if not d.exists() or not d.is_dir():
            continue
        best = _scan_dirs([d])
        if best:
            return best
    return None


@dataclass
class BaselineArtifacts:
    """
    Resolved artifact bundle used by baseline initialization.
    """
    state_file: str | None = None
    calibration_file: str | None = None
    map_file: str | None = None


# ================================================================
# === Public Artifact Resolver
# ================================================================
def resolve_baseline_artifacts_from_state(
    dataset_name_hint: str | None = None,
    state_dir: str = "./dataset/state",
    fallback_dataset: str = "NUMOSIM_Kanto",
    strict_dataset_hint: bool = False,
) -> BaselineArtifacts:
    """
    Read dataset state metadata and resolve absolute file paths for:
    - calibration artifact
    """
    artifacts = BaselineArtifacts()

    # ------------------------------------------------------------
    # Scan candidate state files in priority order.
    # ------------------------------------------------------------
    for state_path in _state_candidate_files(
        dataset_name_hint,
        state_dir,
        fallback_dataset,
        strict_dataset_hint=bool(strict_dataset_hint),
    ):
        if not state_path.exists():
            continue
        try:
            with open(state_path, "r") as f:
                payload = json.load(f)
            parquet = payload.get("parquet_processor", {}) if isinstance(payload, dict) else {}
            if not isinstance(parquet, dict):
                continue

            # Keep traceability for diagnostics.
            artifacts.state_file = str(state_path.resolve())

            # ------------------------------------------------------------
            # Resolve calibration set artifact
            # ------------------------------------------------------------
            cal = parquet.get("calibration_native", {})
            if isinstance(cal, dict):
                raw_cal = cal.get("path") or cal.get("native_source_output")
                resolved_cal = _resolve_path(raw_cal)
                if resolved_cal and artifacts.calibration_file is None:
                    artifacts.calibration_file = resolved_cal

            # ------------------------------------------------------------
            # Resolve map artifact
            # ------------------------------------------------------------
            preferred_raw_map_dir = None
            map_process = parquet.get("map_process", {})
            if isinstance(map_process, dict):
                preferred_raw_map_dir = str(map_process.get("raw_map_dir", "") or "").strip() or None
                # Prefer processed/sliced map artifact first; source_path is the upstream raw map.
                for raw_map in (
                    map_process.get("path"),
                    map_process.get("source_path"),
                ):
                    resolved_map = _resolve_path(raw_map)
                    if resolved_map and artifacts.map_file is None:
                        artifacts.map_file = resolved_map
            map_download = parquet.get("map_download", {})
            if isinstance(map_download, dict):
                resolved_download = _resolve_path(map_download.get("path"))
                if resolved_download and artifacts.map_file is None:
                    artifacts.map_file = resolved_download

            discovered_map = _discover_map_file(
                dataset_name_hint=dataset_name_hint,
                fallback_dataset=fallback_dataset,
                preferred_raw_map_dir=preferred_raw_map_dir,
            )
            if discovered_map and (
                artifacts.map_file is None or not str(artifacts.map_file).lower().endswith(".pbf")
            ):
                artifacts.map_file = discovered_map

            # Stop once both required artifacts are found.
            if artifacts.calibration_file is not None and artifacts.map_file is not None:
                break
        except Exception:
            continue
    return artifacts


__all__ = [
    "BaselineArtifacts",
    "resolve_baseline_artifacts_from_state",
]
