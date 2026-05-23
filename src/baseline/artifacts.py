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


CALIBRATION_INDEX_FILENAME = "calib.json"


def _calibration_key_candidates(
    dataset_name_hint: str | None,
    fallback_dataset: str | None,
    *,
    strict_dataset_hint: bool = False,
) -> list[str]:
    out: list[str] = []

    def _push(raw: str | None) -> None:
        key = _safe_dataset_token(raw)
        if key and key not in out:
            out.append(key)

    _push(dataset_name_hint)
    hint = _safe_dataset_token(dataset_name_hint)
    if hint and "_traj_" in hint:
        base, suffix = hint.split("_traj_", 1)
        parts = suffix.split("_")
        if len(parts) > 1:
            _push(f"{base}_traj_native_{'_'.join(parts[1:])}")
        _push(base)
    if not strict_dataset_hint:
        _push(fallback_dataset)
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
    Read dataset/state/calib.json and resolve absolute file paths for:
    - calibration artifact

    The function name is retained for compatibility with older call sites; it
    no longer scans state_<dataset>.json fallback metadata.
    """
    artifacts = BaselineArtifacts()
    calib_path = Path(state_dir) / CALIBRATION_INDEX_FILENAME
    if not calib_path.exists():
        return artifacts

    try:
        with open(calib_path, "r") as f:
            payload = json.load(f)
    except Exception:
        return artifacts
    if not isinstance(payload, dict):
        return artifacts

    for key in _calibration_key_candidates(
        dataset_name_hint,
        fallback_dataset,
        strict_dataset_hint=bool(strict_dataset_hint),
    ):
        entry = payload.get(key)
        if not isinstance(entry, dict):
            continue
        artifacts.state_file = str(calib_path.resolve())
        resolved_cal = _resolve_path(entry.get("calibration_file"))
        if resolved_cal:
            artifacts.calibration_file = resolved_cal
            break
    return artifacts


__all__ = [
    "BaselineArtifacts",
    "resolve_baseline_artifacts_from_state",
]
