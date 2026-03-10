from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


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
        parts = list(path.parts)
        for idx, part in enumerate(parts):
            if str(part).lower() != "processed":
                continue
            if idx + 2 >= len(parts):
                continue
            split = str(parts[idx + 2]).lower()
            if split not in {
                "calibration",
                "calibration_debug",
                "chunk_test",
                "chunk_test_debug",
                "traj_test",
                "traj_test_debug",
            }:
                continue
            migrated = Path(*parts[: idx + 2], "test", parts[idx + 2], *parts[idx + 3 :])
            if migrated.exists():
                return str(migrated.resolve())
        return None
    return str(path)


@dataclass
class BaselineArtifacts:
    """
    Resolved artifact bundle used by baseline initialization.
    """

    state_file: str | None = None
    calibration_file: str | None = None


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

            artifacts.state_file = str(state_path.resolve())
            cal = parquet.get("calibration_native", {})
            if isinstance(cal, dict):
                raw_cal = cal.get("path") or cal.get("native_source_output")
                resolved_cal = _resolve_path(raw_cal)
                if resolved_cal:
                    artifacts.calibration_file = resolved_cal
                    break
        except Exception:
            continue
    return artifacts


__all__ = [
    "BaselineArtifacts",
    "resolve_baseline_artifacts_from_state",
]
