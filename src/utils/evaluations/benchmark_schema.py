"""Benchmark job-schema helpers for run_benchmarks.

Purpose:
    Keep eval_joblist parsing, normalization, and validation out of the
    top-level benchmark entry point.

Logic Chain:
    1. Normalize JSON list-like values.
    2. Validate learned-model hypothesis and root consistency.
    3. Expand classic baseline specs.
    4. Normalize legacy/new eval_joblist schemas into one internal packet.
"""

from __future__ import annotations

from pathlib import Path


DEFAULT_CLASSIC_BASELINES = [
    "alpha_beta",
    "causal_hampel",
    "kalman_filter",
    "kalman_rts",
    "hampel",
    "savgol",
    "raw",
]
ALLOWED_CLASSIC_BASELINES = list(DEFAULT_CLASSIC_BASELINES)
ALLOWED_CLASSIC_BASELINES.append("valhalla_meili")


# ================================================================
# === Basic Normalization
# ================================================================
def as_list(value) -> list:
    """Normalize list-like config fields into a Python list."""
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [value]


def normalize_denoise_steps(raw) -> list[int | None]:
    """Normalize optional RF integration step counts."""
    values = as_list(raw)
    if not values:
        return [None]
    out: list[int | None] = []
    for raw_value in values:
        if raw_value is None:
            out.append(None)
            continue
        step_count = int(raw_value)
        if step_count <= 0:
            raise ValueError(f"denoise_steps values must be positive integers, got {raw_value!r}.")
        out.append(step_count)
    return out


def normalize_sample_steps(raw) -> list[int | None]:
    """Normalize optional diffusion reverse-sampling step counts."""
    values = as_list(raw)
    if not values:
        return [None]
    out: list[int | None] = []
    for raw_value in values:
        if raw_value is None:
            out.append(None)
            continue
        step_count = int(raw_value)
        if step_count <= 0:
            raise ValueError(f"sample_steps values must be positive integers, got {raw_value!r}.")
        out.append(step_count)
    return out


def dedupe_keep_order(values: list[str]) -> list[str]:
    """Remove duplicates while preserving input order."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        item = str(raw).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


# ================================================================
# === Learned Model Schema
# ================================================================
def normalize_data_hypothesis(raw, default: str = "RectifiedTraj") -> str:
    """Validate canonical learned-model family token."""
    if raw is None:
        return str(default)
    value = str(raw).strip()
    if value in {"RectifiedTraj", "ResidualReg", "Diffusion"}:
        return value
    raise ValueError(
        f"Unsupported data_hypothesis={raw!r}. "
        "Recognized values: RectifiedTraj, ResidualReg, Diffusion."
    )


def validate_supported_data_hypothesis(data_hypothesis: str, *, context: str) -> None:
    """Reject unsupported learned-model family tokens."""
    if data_hypothesis in {"RectifiedTraj", "ResidualReg", "Diffusion"}:
        return
    raise ValueError(
        f"{context} has unsupported data_hypothesis={data_hypothesis!r}. "
        "Supported values: RectifiedTraj, ResidualReg, Diffusion."
    )


def validate_model_root_matches_hypothesis(
    model_root: str,
    data_hypothesis: str,
    *,
    context: str,
) -> None:
    """Ensure explicit hypothesis leaf folders match the configured family."""
    root_name = Path(model_root).name.strip().lower()
    explicit_roots = {
        "rectifiedtraj": "RectifiedTraj",
        "rectifiedtraj_online": "RectifiedTraj",
        "residualreg": "ResidualReg",
        "residualreg_online": "ResidualReg",
        "diffusion": "Diffusion",
        "diffusion_online": "Diffusion",
    }
    expected = explicit_roots.get(root_name)
    if expected is not None and expected != data_hypothesis:
        raise ValueError(
            f"{context} has model_root={model_root!r} but data_hypothesis={data_hypothesis!r}. "
            "model_root hypothesis folder must match data_hypothesis."
        )


def default_model_root_for_hypothesis(data_hypothesis: str) -> str:
    """Return the canonical model root for one learned-model family."""
    return str(Path("./bin/model") / normalize_data_hypothesis(data_hypothesis))


def normalize_model_root_with_hypothesis(model_root_value, data_hypothesis: str) -> str:
    """Resolve empty/default model roots against the configured hypothesis."""
    model_root_text = str(model_root_value).strip()
    if not model_root_text:
        return default_model_root_for_hypothesis(data_hypothesis)
    base = Path(model_root_text)
    hypothesis = normalize_data_hypothesis(data_hypothesis)
    if base.as_posix().rstrip("/") in {"./bin/model", "bin/model"}:
        return str(base / hypothesis)
    return str(base)


def reject_step_fields(scope: dict | None, *, context: str) -> None:
    """Reject step/t_delta fields from eval_joblist."""
    if not isinstance(scope, dict):
        return
    for key in ("delta_t", "t_delta", "step", "method", "methods", "denoise_method", "denoise_methods"):
        if key in scope:
            raise ValueError(
                f"{context}.{key} is not supported. "
                "Trajectory evaluation uses the fixed chunk_stitch denoiser."
            )


def reject_rolling_window(scope: dict | None, *, context: str) -> None:
    """Reject rolling-window trajectory evaluation."""
    if not isinstance(scope, dict):
        return
    if "rolling_window" in scope:
        raise ValueError(
            f"{context}.rolling_window is no longer supported. "
            "Rolling-window evaluation was removed because it is too expensive; "
            "delete the key from eval_joblist."
        )


def normalize_model_group_schema_entry(
    raw_group: dict,
    *,
    default_group: dict | None,
    context: str,
) -> dict:
    """Normalize one model-group block from eval_joblist."""
    if not isinstance(raw_group, dict):
        raise ValueError(f"{context} must be a JSON object.")
    reject_step_fields(raw_group, context=context)
    reject_rolling_window(raw_group, context=context)

    if isinstance(default_group, dict):
        default_hypothesis = str(default_group.get("data_hypothesis", "RectifiedTraj"))
    else:
        default_hypothesis = "RectifiedTraj"
    data_hypothesis = normalize_data_hypothesis(
        raw_group.get("data_hypothesis", default_hypothesis)
    )
    validate_supported_data_hypothesis(data_hypothesis, context=context)

    if "model_root" in raw_group:
        model_root = normalize_model_root_with_hypothesis(
            raw_group.get("model_root"),
            data_hypothesis,
        )
    else:
        model_root = default_model_root_for_hypothesis(data_hypothesis)
    validate_model_root_matches_hypothesis(
        model_root,
        data_hypothesis,
        context=context,
    )

    if "models" in raw_group or "model_names" in raw_group:
        raw_models = raw_group.get("models", raw_group.get("model_names"))
    elif isinstance(default_group, dict):
        raw_models = default_group.get("model_names")
    else:
        raw_models = None
    model_names = None if raw_models is None else as_list(raw_models)

    if isinstance(default_group, dict):
        default_q1 = default_group.get("Q1", [1])
        default_q2 = default_group.get("Q2", [12])
        default_steps = default_group.get("denoise_steps", [None])
        default_sample_steps = default_group.get("sample_steps", [None])
    else:
        default_q1 = [1]
        default_q2 = [12]
        default_steps = [None]
        default_sample_steps = [None]
    raw_q1 = raw_group.get("Q1", default_q1)
    raw_q2 = raw_group.get("Q2", default_q2)
    raw_steps = raw_group.get("denoise_steps", default_steps)
    raw_sample_steps = raw_group.get("sample_steps", default_sample_steps)

    return {
        "data_hypothesis": data_hypothesis,
        "model_root": model_root,
        "model_names": model_names,
        "Q1": as_list(raw_q1) or [1],
        "Q2": as_list(raw_q2) or [12],
        "denoise_steps": normalize_denoise_steps(raw_steps),
        "sample_steps": normalize_sample_steps(raw_sample_steps),
    }


def build_primary_model_group_from_job(job: dict) -> dict:
    """Build the canonical primary model-group packet from normalized job fields."""
    data_hypothesis = normalize_data_hypothesis(job.get("data_hypothesis", "RectifiedTraj"))
    return {
        "data_hypothesis": data_hypothesis,
        "model_root": str(
            job.get("model_root", default_model_root_for_hypothesis(data_hypothesis))
        ),
        "model_names": job.get("model_names"),
        "Q1": as_list(job.get("Q1")) or [1],
        "Q2": as_list(job.get("Q2")) or [12],
        "denoise_steps": normalize_denoise_steps(job.get("denoise_steps")),
        "sample_steps": normalize_sample_steps(job.get("sample_steps")),
    }


def dedupe_model_groups(groups: list[dict]) -> list[dict]:
    """De-duplicate learned-model groups by family and model root."""
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for group in groups:
        key = (
            str(group.get("data_hypothesis", "")).strip(),
            str(group.get("model_root", "")).strip(),
        )
        if not key[0] or not key[1] or key in seen:
            continue
        seen.add(key)
        out.append(group)
    return out


# ================================================================
# === Classic Baseline Schema
# ================================================================
def normalize_kalman_calibration_mode_token(raw: str | None) -> str:
    """Validate canonical Kalman calibration token."""
    token = str(raw or "dataset").strip()
    if token in {"numosim_kanto", "dataset"}:
        return token
    raise ValueError(
        f"Unsupported kalman_rts calibration mode={raw!r}. "
        "Recognized values: numosim_kanto, dataset."
    )


def split_baseline_spec(spec: str) -> tuple[str, str | None, str]:
    """Split baseline spec into base-name, mode, and display token."""
    token = str(spec).strip()
    if not token:
        return "", None, ""
    if "@" in token:
        base, mode = token.split("@", 1)
    else:
        base, mode = token, None
    base_name = str(base).strip().lower()
    mode_name = None
    if base_name == "kalman_rts" and mode is not None:
        mode_name = normalize_kalman_calibration_mode_token(mode)
    display = (
        f"{base_name}@{mode_name}"
        if base_name == "kalman_rts" and mode_name
        else base_name
    )
    return base_name, mode_name, display


def expand_baseline_specs(baseline_models, calibration_cfg) -> list[str]:
    """Expand baseline config into canonical baseline spec list."""
    models = as_list(baseline_models)
    if not models:
        models = list(DEFAULT_CLASSIC_BASELINES)
    if not isinstance(calibration_cfg, dict):
        calibration_cfg = {}

    out: list[str] = []
    for raw_model in models:
        model = str(raw_model).strip().lower()
        if not model:
            continue
        if model == "kalman_rts":
            for mode in as_list(calibration_cfg.get("kalman_rts")):
                mode_name = normalize_kalman_calibration_mode_token(str(mode))
                if mode_name != "dataset":
                    raise ValueError(
                        "kalman_rts calibration variants are no longer supported in eval_joblist. "
                        "Use baseline.models=[\"kalman_rts\"] and dataset calibration from calib.json."
                    )
        out.append(model)
    return dedupe_keep_order(out)


def _normalize_legacy_kalman_rts_spec(base_name: str, mode_name: str | None, item: str) -> str:
    if base_name != "kalman_rts":
        return item
    if mode_name in {None, "dataset"}:
        return "kalman_rts"
    raise ValueError(
        "kalman_rts calibration variants are no longer supported. "
        "Use kalman_rts with dataset calibration from calib.json."
    )


def resolve_classic_baselines(job: dict) -> list[str]:
    """Validate the classic baseline list selected by eval_joblist."""
    raw = job.get("classic_baselines")
    if raw is None:
        return list(DEFAULT_CLASSIC_BASELINES)
    if not isinstance(raw, (str, list, tuple)):
        raise ValueError("classic_baselines must be a list of names or comma-separated string.")

    selected: list[str] = []
    allowed = set(ALLOWED_CLASSIC_BASELINES)
    for item in as_list(raw):
        base_name, mode_name, display = split_baseline_spec(str(item))
        if not base_name:
            continue
        if base_name not in allowed:
            raise ValueError(
                f"Unsupported classic baseline={item!r}. "
                "Recognized values: alpha_beta, causal_hampel, kalman_filter, "
                "kalman_rts, hampel, savgol, raw, valhalla_meili."
            )
        selected.append(_normalize_legacy_kalman_rts_spec(base_name, mode_name, display or base_name))
    return dedupe_keep_order(selected)


# ================================================================
# === Test File Schema
# ================================================================
def extract_test_dirs(test_files) -> tuple[list[str], list[str], str | None]:
    """Extract canonical test-file paths from eval_joblist."""
    if test_files is None:
        return [], [], None
    if not isinstance(test_files, dict):
        raise ValueError("test_files must be a JSON object.")

    traj_dirs: list[str] = []
    chunk_dirs: list[str] = []
    uncertainty_path: str | None = None

    allowed_keys = {"traj_files", "chunk_files", "uncertainty_path"}
    unknown_keys = sorted(set(test_files.keys()) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            "Unsupported test_files keys: %s. Recognized keys: traj_files, chunk_files, uncertainty_path."
            % ", ".join(unknown_keys)
        )

    for value in as_list(test_files.get("traj_files")):
        path_value = str(value).strip()
        if path_value:
            traj_dirs.append(path_value)
    for value in as_list(test_files.get("chunk_files")):
        path_value = str(value).strip()
        if path_value:
            chunk_dirs.append(path_value)

    raw_uncertainty = test_files.get("uncertainty_path")
    if isinstance(raw_uncertainty, str) and raw_uncertainty.strip():
        uncertainty_path = str(raw_uncertainty).strip()

    return dedupe_keep_order(traj_dirs), dedupe_keep_order(chunk_dirs), uncertainty_path


# ================================================================
# === Runtime Schema
# ================================================================
def apply_runtime_defaults(runtime_cfg: dict | None) -> dict:
    """Attach canonical runtime defaults."""
    cfg = dict(runtime_cfg) if isinstance(runtime_cfg, dict) else {}
    cfg.setdefault("device", "cuda")
    cfg.setdefault("strict_init", True)
    cfg.setdefault("traj_parallel", 4)
    return cfg


def runtime_cfg(job: dict) -> dict:
    """Read runtime config packet from normalized job."""
    cfg = job.get("runtime", {})
    return cfg if isinstance(cfg, dict) else {}


def normalize_runtime_device_token(raw) -> str:
    """Validate canonical runtime device token."""
    token = str(raw or "cuda").strip()
    if token == "cuda":
        return "cuda"
    if token == "cpu":
        return "cpu"
    raise ValueError(
        f"Unsupported runtime.device={raw!r}. Recognized values: cuda, cpu."
    )


def normalize_device_label(raw) -> str:
    """Collapse runtime device labels into short canonical values."""
    token = str(raw or "").strip().lower()
    if token.startswith("cuda"):
        return "cuda"
    if token == "cpu":
        return "cpu"
    return token or "unknown"


def runtime_device_requested(job: dict) -> str:
    """Return the requested runtime device from the job packet."""
    return normalize_runtime_device_token(runtime_cfg(job).get("device", "cuda"))


def runtime_device_for_defaults(job: dict) -> str:
    """Return effective runtime device when a previous phase already resolved it."""
    cfg = runtime_cfg(job)
    effective = cfg.get("device_effective")
    if effective is not None:
        return normalize_runtime_device_token(effective)
    return runtime_device_requested(job)


# ================================================================
# === Full Job Schema
# ================================================================
def normalize_job_schema(raw_job: dict) -> dict:
    """Normalize legacy/new eval_joblist formats into one benchmark job packet."""
    if not isinstance(raw_job, dict):
        raise ValueError("eval_joblist.json must be a JSON object.")

    has_new_schema = any(
        key in raw_job
        for key in (
            "test_type",
            "test_items",
            "test_files",
            "rectifiedtraj",
            "baseline",
            "baselines",
            "data_source",
            "model_groups",
        )
    )

    if not has_new_schema:
        job = dict(raw_job)
        reject_rolling_window(raw_job, context="eval_joblist")
        job.setdefault("test_type", "exact")
        job.setdefault("traj_test", True)
        reject_step_fields(job, context="eval_joblist")
        job["denoise_steps"] = normalize_denoise_steps(raw_job.get("denoise_steps"))
        job["sample_steps"] = normalize_sample_steps(raw_job.get("sample_steps"))

        data_hypothesis = normalize_data_hypothesis(job.get("data_hypothesis", "RectifiedTraj"))
        validate_supported_data_hypothesis(
            data_hypothesis,
            context="eval_joblist (legacy schema)",
        )
        job["data_hypothesis"] = data_hypothesis

        if "model_root" in job:
            job["model_root"] = normalize_model_root_with_hypothesis(
                job.get("model_root"),
                data_hypothesis,
            )
        else:
            job["model_root"] = default_model_root_for_hypothesis(data_hypothesis)
        validate_model_root_matches_hypothesis(
            job["model_root"],
            data_hypothesis,
            context="eval_joblist (legacy schema)",
        )

        job["runtime"] = apply_runtime_defaults(raw_job.get("runtime", {}))
        baseline_options = raw_job.get("baseline_options", {})
        if not isinstance(baseline_options, dict):
            raise ValueError("baseline_options must be a JSON object.")
        job["baseline_options"] = dict(baseline_options)

        raw_dataset_dir = str(job.get("raw_dataset_dir", "") or "").strip()
        if raw_dataset_dir:
            job["raw_dataset_dir"] = raw_dataset_dir
            job["data_source"] = {
                "raw_dataset_dir": raw_dataset_dir,
                "raw_test_files": None,
            }

        job["model_groups"] = [build_primary_model_group_from_job(job)]
        return job

    job = dict(raw_job)
    reject_rolling_window(raw_job, context="eval_joblist")

    test_type = str(raw_job.get("test_type", "exact")).strip().lower()
    if test_type not in {"exact", "uncertainty", "tuning"}:
        raise ValueError("test_type must be one of: exact, uncertainty, tuning")

    test_items = raw_job.get("test_items", {})
    if not isinstance(test_items, dict):
        raise ValueError("test_items must be a JSON object.")
    rectified = raw_job.get("rectifiedtraj", {})
    if not isinstance(rectified, dict):
        raise ValueError("rectifiedtraj must be a JSON object.")
    baseline_cfg = raw_job.get("baseline", raw_job.get("baselines", {}))
    if not isinstance(baseline_cfg, dict):
        raise ValueError("baseline must be a JSON object.")
    data_source = raw_job.get("data_source", {})
    if not isinstance(data_source, dict):
        raise ValueError("data_source must be a JSON object.")

    reject_step_fields(raw_job, context="eval_joblist")
    reject_step_fields(rectified, context="eval_joblist.rectifiedtraj")

    traj_dirs, chunk_dirs, uncertainty_path = extract_test_dirs(raw_job.get("test_files"))

    data_hypothesis = normalize_data_hypothesis(
        rectified.get("data_hypothesis", raw_job.get("data_hypothesis", "RectifiedTraj"))
    )
    validate_supported_data_hypothesis(
        data_hypothesis,
        context="eval_joblist (new schema)",
    )
    job["data_hypothesis"] = data_hypothesis

    model_root_raw = rectified.get("model_root", raw_job.get("model_root"))
    if model_root_raw is None:
        job["model_root"] = default_model_root_for_hypothesis(data_hypothesis)
    else:
        job["model_root"] = normalize_model_root_with_hypothesis(
            model_root_raw,
            data_hypothesis,
        )
    validate_model_root_matches_hypothesis(
        job["model_root"],
        data_hypothesis,
        context="eval_joblist (new schema)",
    )

    models = rectified.get("models", raw_job.get("model_names"))
    job["model_names"] = None if models is None else as_list(models)
    job["Q1"] = as_list(rectified.get("Q1", raw_job.get("Q1", [1]))) or [1]
    job["Q2"] = as_list(rectified.get("Q2", raw_job.get("Q2", [1, 12, 24]))) or [1, 12, 24]
    job["denoise_steps"] = normalize_denoise_steps(
        rectified.get("denoise_steps", raw_job.get("denoise_steps"))
    )
    job["sample_steps"] = normalize_sample_steps(
        rectified.get("sample_steps", raw_job.get("sample_steps"))
    )

    if traj_dirs:
        job["traj_dirs"] = traj_dirs
        job["traj_paths"] = {"full_traj": traj_dirs[0]}
    if chunk_dirs:
        job["chunk_dirs"] = chunk_dirs
        job["chunk_test_dir"] = chunk_dirs[0]

    baseline_models = baseline_cfg.get("models", raw_job.get("classic_baselines"))
    baseline_calibration = baseline_cfg.get("calibration", {})
    job["classic_baselines"] = expand_baseline_specs(
        baseline_models,
        baseline_calibration,
    )
    baseline_options = baseline_cfg.get("options", raw_job.get("baseline_options", {}))
    if not isinstance(baseline_options, dict):
        raise ValueError("baseline.options must be a JSON object.")
    job["baseline_options"] = dict(baseline_options)

    raw_dataset_dir_raw = data_source.get("raw_dataset_dir", raw_job.get("raw_dataset_dir", ""))
    if raw_dataset_dir_raw is None:
        raw_dataset_dir = ""
    else:
        raw_dataset_dir = str(raw_dataset_dir_raw).strip()
        if raw_dataset_dir.lower() in {"none", "null"}:
            raw_dataset_dir = ""
    if raw_dataset_dir:
        job["raw_dataset_dir"] = raw_dataset_dir

    raw_test_files = data_source.get("raw_test_files", raw_job.get("raw_test_files"))
    if isinstance(raw_test_files, str):
        raw_test_files = [raw_test_files]
    elif isinstance(raw_test_files, tuple):
        raw_test_files = list(raw_test_files)
    if raw_test_files is not None:
        raw_test_files = [str(value).strip() for value in raw_test_files if str(value).strip()]
        if not raw_test_files:
            raw_test_files = None
    job["data_source"] = {
        "raw_dataset_dir": raw_dataset_dir or None,
        "raw_test_files": raw_test_files,
    }

    traj_default = test_type == "exact"
    chunk_default = False
    uncertainty_default = test_type == "uncertainty"
    run_baseline_default = test_type != "tuning"
    if test_type == "tuning":
        traj_default = False
        chunk_default = True
        uncertainty_default = False

    job["test_type"] = test_type
    job["traj_test"] = bool(test_items.get("traj_test", raw_job.get("traj_test", traj_default)))
    job["chunk_test"] = bool(test_items.get("chunk_test", raw_job.get("chunk_test", chunk_default)))
    job["range_test"] = bool(
        test_items.get(
            "uncertainty_test",
            raw_job.get("range_test", uncertainty_default),
        )
    )
    job["run_baseline"] = bool(raw_job.get("run_baseline", run_baseline_default))

    if test_type == "tuning":
        quick_val_path = str(
            raw_job.get(
                "quick_val_path",
                "./dataset/processed/NUMOSIM_Kanto/val/quick_val_chunk_50k.pt",
            )
        ).strip()
        job["quick_val_path"] = quick_val_path
        if not job.get("chunk_test_dir"):
            job["chunk_test_dir"] = quick_val_path
        job["run_baseline"] = False
        job["classic_baselines"] = []
        job["chunk_grid_search"] = True

    if uncertainty_path:
        job["test_data_path"] = uncertainty_path
        job["test_data_paths"] = [uncertainty_path]
    elif job.get("range_test") and traj_dirs:
        job["test_data_paths"] = list(traj_dirs)
        job["test_data_path"] = traj_dirs[0]

    job["runtime"] = apply_runtime_defaults(raw_job.get("runtime", {}))

    raw_model_groups = raw_job.get("model_groups")
    if raw_model_groups is not None and not isinstance(raw_model_groups, list):
        raise ValueError("model_groups must be a list of objects.")
    model_groups: list[dict] = []

    if isinstance(raw_model_groups, list) and raw_model_groups:
        for idx, raw_group in enumerate(raw_model_groups):
            model_groups.append(
                normalize_model_group_schema_entry(
                    raw_group,
                    default_group=None,
                    context=f"eval_joblist.model_groups[{idx}]",
                )
            )
    else:
        rectified_has_group_fields = any(
            key in rectified for key in ("models", "model_names", "model_root", "Q1", "Q2", "data_hypothesis")
        )
        if rectified_has_group_fields:
            primary_group = build_primary_model_group_from_job(job)
            model_groups.append(primary_group)

    residualreg_block = raw_job.get("residualreg")
    if isinstance(residualreg_block, dict):
        default_group = model_groups[0] if model_groups else None
        model_groups.append(
            normalize_model_group_schema_entry(
                residualreg_block,
                default_group=default_group,
                context="eval_joblist.residualreg",
            )
        )

    job["model_groups"] = dedupe_model_groups(model_groups)
    return job


def build_job_list_from_group(group: dict) -> dict:
    """Build the compact trajectory-grid packet for one learned-model group."""
    job_list = {
        "Q1": group.get("Q1"),
        "Q2": group.get("Q2"),
        "denoise_steps": group.get("denoise_steps") or [None],
        "sample_steps": group.get("sample_steps") or [None],
    }
    if not job_list["Q1"] or not job_list["Q2"]:
        raise ValueError("Each model group must include non-empty Q1 and Q2 lists.")
    return job_list
