"""Range-test benchmark runner helpers.

Purpose:
    Keep bounded/range evaluation orchestration out of run_benchmarks.

Logic Chain:
    1. Resolve dataset entries from explicit uncertainty input paths.
    2. Expand Q1/Q2 config grid.
    3. Apply optional CPU caps.
    4. Dispatch each uncertainty evaluation run through TestManager.
"""

from __future__ import annotations

import logging

from .benchmark_inputs import build_bounded_manual_configs, resolve_bounded_dataset_entries
from .benchmark_runtime import cpu_shrink_enabled, resolve_cpu_traj_caps


def run_bounded_eval(
    manager,
    job: dict,
    job_list: dict,
    model_root: str,
    model_names: list | None,
    classic_baselines: list[str],
    model_tag: str,
    run_baselines: bool,
) -> None:
    """Run one bounded/range evaluation phase for one learned-model group."""
    dataset_entries = resolve_bounded_dataset_entries(job)
    manual_configs = build_bounded_manual_configs(job_list)
    resolved_model_names = model_names or manager._discover_models(model_root)
    baseline_count = len(list(classic_baselines or [])) if run_baselines else 0
    learned_job_count = len(list(resolved_model_names or []))

    explicit_m = int(job["M"]) if "M" in job else None
    explicit_n = int(job["N"]) if "N" in job else None
    cpu_cap_m = None
    cpu_cap_n = None
    if cpu_shrink_enabled(job):
        cpu_cap_m, cpu_cap_n = resolve_cpu_traj_caps(job)

    planned_runs: list[dict] = []
    progress_total_units = 0
    for entry in dataset_entries:
        if explicit_m is not None:
            m_value = explicit_m
        elif cpu_cap_m is not None:
            m_value = min(int(entry["M"]), int(cpu_cap_m))
        else:
            m_value = int(entry["M"])

        if explicit_n is not None:
            n_value = explicit_n
        elif cpu_cap_n is not None:
            n_value = min(int(entry["N"]), int(cpu_cap_n))
        else:
            n_value = int(entry["N"])

        traj_count = int(m_value)
        for config_idx, manual_config in enumerate(manual_configs):
            local_units = traj_count * learned_job_count
            if run_baselines and config_idx == 0:
                local_units += traj_count * baseline_count
            planned_runs.append(
                {
                    "entry": entry,
                    "m_value": int(m_value),
                    "n_value": int(n_value),
                    "manual_config": manual_config,
                    "config_idx": int(config_idx),
                    "unit_offset": int(progress_total_units),
                }
            )
            progress_total_units += int(local_units)

    for plan in planned_runs:
        entry = plan["entry"]
        m_value = int(plan["m_value"])
        n_value = int(plan["n_value"])
        manual_config = dict(plan["manual_config"])
        config_idx = int(plan["config_idx"])
        unit_offset = int(plan["unit_offset"])
        q1_value = int(manual_config["Q1"])
        q2_value = int(manual_config["Q2"])

        logging.debug(
            "Range eval start | dataset=%s M=%d N=%d Q1=%d Q2=%d",
            entry["name"],
            m_value,
            n_value,
            q1_value,
            q2_value,
        )
        manager.run_uncertainty_band_test(
            model_names=resolved_model_names,
            model_root=model_root,
            test_data_path=str(entry["path"]),
            M=m_value,
            N=n_value,
            model_tag=model_tag,
            run_baselines=bool(run_baselines and config_idx == 0),
            baseline_methods=classic_baselines,
            manual_config=manual_config,
            progress_unit_offset=unit_offset,
            progress_total_units=progress_total_units,
        )
