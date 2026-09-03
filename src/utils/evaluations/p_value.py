"""Pairwise p-value helpers for merged benchmark sample-level CSVs."""

from __future__ import annotations

import csv
import logging
import re
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

from .result_io import write_rows_to_csv


LOGGER = logging.getLogger(__name__)


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    """
    Purpose:
        Read one merged sample-level CSV into memory as plain row dicts.
    Parameters:
        csv_path (Path), existing UTF-8 CSV file.
    Return Dict:
        Not used. Returns one list[dict[str, str]] row list.
    Usage:
        Called by generate_pairwise_p_value_report before grouping rows.
    TODO:
        1) Open CSV file.
        2) Read rows through DictReader.
        3) Return list payload.
    """
    with csv_path.open("r", newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def _safe_float(value: str) -> float:
    """
    Purpose:
        Convert one stored CSV scalar into float.
    Parameters:
        value (str), non-empty numeric cell from sample-level CSV.
    Return Dict:
        Not used. Returns one float scalar.
    Usage:
        Called by pairwise builders when parsing metric columns.
    TODO:
        1) Normalize string token.
        2) Convert token into float.
        3) Return parsed value.
    """
    token = str(value).strip()
    if token in {"", "NA", "nan", "NaN", "None"}:
        return float("nan")
    return float(token)


def _group_key(row: dict[str, str], key_columns: list[str]) -> tuple[str, ...]:
    """
    Purpose:
        Build one stable grouping key from selected row columns.
    Parameters:
        row (dict[str, str]), one sample-level CSV row.
        key_columns (list[str]), grouping columns that must exist in output rows.
    Return Dict:
        Not used. Returns tuple[str, ...] group key.
    Usage:
        Called by generate_pairwise_p_value_report while partitioning rows.
    TODO:
        1) Read each requested column.
        2) Normalize values into stripped strings.
        3) Return tuple key.
    """
    return tuple(str(row.get(column, "")).strip() for column in key_columns)


def _sample_key(row: dict[str, str]) -> tuple[str, str, str]:
    """
    Purpose:
        Build one sample identity key that aligns the same sample across models.
    Parameters:
        row (dict[str, str]), one sample-level CSV row.
    Return Dict:
        Not used. Returns tuple[str, str, str] sample key.
    Usage:
        Called by generate_pairwise_p_value_report while matching paired rows.
    TODO:
        1) Read explicit sample index.
        2) Include timestamps to strengthen identity when available.
        3) Return tuple key.
    """
    return (
        str(row["sample_index"]).strip(),
        str(row.get("first_timestamp", "")).strip(),
        str(row.get("last_timestamp", "")).strip(),
    )


def _paired_metric_arrays(
    rows_a: list[dict[str, str]],
    rows_b: list[dict[str, str]],
    metric_column: str,
) -> dict:
    """
    Purpose:
        Align two model row sets on shared samples and extract paired metric arrays.
    Parameters:
        rows_a (list[dict[str, str]]), sample rows for model A within one group.
        rows_b (list[dict[str, str]]), sample rows for model B within one group.
        metric_column (str), numeric metric column to compare.
    Return Dict:
        "error_code": 0 | -1
        "values_a": np.ndarray
        "values_b": np.ndarray
        "n_common": int
    Usage:
        Called by generate_pairwise_p_value_report for each model pair.
    TODO:
        1) Build sample-key maps.
        2) Intersect shared keys.
        3) Extract paired metric values.
        4) Return payload.
    """
    map_a = {_sample_key(row): row for row in rows_a}
    map_b = {_sample_key(row): row for row in rows_b}
    common_keys = sorted(set(map_a.keys()) & set(map_b.keys()))
    if not common_keys:
        return {
            "error_code": -1,
            "values_a": np.asarray([], dtype=float),
            "values_b": np.asarray([], dtype=float),
            "n_common": 0,
        }

    values_a = np.asarray([_safe_float(map_a[key][metric_column]) for key in common_keys], dtype=float)
    values_b = np.asarray([_safe_float(map_b[key][metric_column]) for key in common_keys], dtype=float)
    valid_mask = np.isfinite(values_a) & np.isfinite(values_b)
    values_a = values_a[valid_mask]
    values_b = values_b[valid_mask]
    return {
        "error_code": 0,
        "values_a": values_a,
        "values_b": values_b,
        "n_common": int(values_a.size),
    }


def _paired_test_packet(values_a: np.ndarray, values_b: np.ndarray) -> dict:
    """
    Purpose:
        Compute paired significance summaries for one matched metric vector pair.
    Parameters:
        values_a (np.ndarray), finite paired metric values for model A.
        values_b (np.ndarray), finite paired metric values for model B.
    Return Dict:
        "error_code": 0
        "n_common": int
        "mean_a": float
        "mean_b": float
        "mean_delta_a_minus_b": float
        "relative_improvement_pct_a_vs_b": float
        "paired_t_pvalue": float
        "wilcoxon_pvalue": float
    Usage:
        Called by generate_pairwise_p_value_report after aligned arrays are built.
    TODO:
        1) Compute descriptive means.
        2) Compute relative improvement.
        3) Compute paired t-test p-value.
        4) Compute Wilcoxon p-value.
        5) Return payload.
    """
    assert values_a.shape == values_b.shape
    assert values_a.ndim == 1
    n_common = int(values_a.size)
    mean_a = float(np.mean(values_a))
    mean_b = float(np.mean(values_b))
    mean_delta = float(np.mean(values_a - values_b))
    if mean_b == 0.0:
        relative_improvement = float("nan")
    else:
        relative_improvement = float((mean_b - mean_a) / mean_b * 100.0)

    paired_t = ttest_rel(values_a, values_b, alternative="two-sided")
    diff = values_a - values_b
    if np.allclose(diff, 0.0):
        wilcoxon_pvalue = 1.0
        wilcoxon_stat = 0.0
    else:
        wilcoxon_result = wilcoxon(values_a, values_b, zero_method="zsplit", alternative="two-sided", method="auto")
        wilcoxon_pvalue = float(wilcoxon_result.pvalue)
        wilcoxon_stat = float(wilcoxon_result.statistic)

    return {
        "error_code": 0,
        "n_common": n_common,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_delta_a_minus_b": mean_delta,
        "relative_improvement_pct_a_vs_b": relative_improvement,
        "paired_t_pvalue": float(paired_t.pvalue),
        "paired_t_statistic": float(paired_t.statistic),
        "wilcoxon_pvalue": wilcoxon_pvalue,
        "wilcoxon_statistic": wilcoxon_stat,
    }


def _normalize_text_token(value: str) -> str:
    """
    Purpose:
        Normalize one arbitrary text token into a safe filename fragment.
    Parameters:
        value (str), raw label fragment.
    Return Dict:
        Not used. Returns one sanitized string token.
    Usage:
        Called by matrix filename builders.
    TODO:
        1) Strip whitespace.
        2) Replace non-alnum spans with underscores.
        3) Return cleaned token.
    """
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    token = re.sub(r"_+", "_", token).strip("._-")
    return token or "NA"


def _format_q_token(row: dict[str, str]) -> str:
    """
    Purpose:
        Build one compact Q token from one sample row.
    Parameters:
        row (dict[str, str]), one sample-level CSV row.
    Return Dict:
        Not used. Returns one compact Q string or empty string.
    Usage:
        Called by model label builders and filename builders.
    TODO:
        1) Read Q1 token.
        2) Normalize blank cases.
        3) Return compact Q string.
    """
    q1 = str(row.get("Q1", "")).strip()
    if not q1:
        return ""
    return f"Q={q1}"


def _format_step_token(row: dict[str, str]) -> str:
    """Build one compact denoising-step token from one sample row."""
    steps = str(row.get("denoise_steps", "")).strip()
    if not steps or steps in {"NA", "None", "nan"}:
        return ""
    return f"S={steps}"


def _format_sample_step_token(row: dict[str, str]) -> str:
    """Build one compact diffusion-sampling-step token from one sample row."""
    steps = str(row.get("sample_steps", "")).strip()
    if not steps or steps in {"NA", "None", "nan"}:
        return ""
    return f"D={steps}"


def _baseline_label(model_name: str) -> str:
    """
    Purpose:
        Convert one baseline internal name into one paper-friendly display label.
    Parameters:
        model_name (str), baseline method identifier from CSV rows.
    Return Dict:
        Not used. Returns one display label string.
    Usage:
        Called by _model_display_label for baseline rows.
    TODO:
        1) Match known baseline names.
        2) Return explicit display label.
        3) Fall back to raw token only if unknown.
    """
    baseline_map = {
        "alpha_beta": "AlphaBeta",
        "causal_hampel": "Causal Hampel",
        "hampel": "Hampel",
        "kalman_filter": "Kalman",
        "kalman_rts@dataset": "Kalman RTS",
        "raw": "Raw",
        "savgol": "SavGol",
    }
    if model_name in baseline_map:
        return baseline_map[model_name]
    return str(model_name)


def _model_display_label(row: dict[str, str]) -> str:
    """
    Purpose:
        Convert one model row into one compact paper-style matrix label.
    Parameters:
        row (dict[str, str]), one sample-level CSV row.
    Return Dict:
        Not used. Returns one display label string.
    Usage:
        Called by generate_pairwise_p_value_report while naming matrix axes.
    TODO:
        1) Detect baseline rows.
        2) Build short family prefix.
        3) Build short architecture label.
        4) Append Q token when present.
    """
    model_tag = str(row["model_tag"]).strip()
    model_name = str(row["model_name"]).strip()
    if model_tag == "Baseline":
        return _baseline_label(model_name)

    tag_prefix_map = {
        "RectifiedTraj": "RT",
        "DirectReg": "DR",
        "ResidualReg": "DR",
        "Diffusion": "Diff.",
    }
    arch_map = {
        "cnn": "CNN",
        "hybrid": "Hybrid",
        "transformer": "Trans.",
        "mlp": "MLP",
    }
    family = tag_prefix_map.get(model_tag, model_tag)
    arch_token = model_name.split("_", 1)[0]
    architecture = arch_map.get(arch_token, arch_token)
    tokens = [
        token
        for token in (
            _format_q_token(row),
            _format_step_token(row),
            _format_sample_step_token(row),
        )
        if token
    ]
    suffix = " ".join(tokens)
    if suffix:
        label = f"{family} {architecture} {suffix}"
    else:
        label = f"{family} {architecture}"

    model_full_name = str(row.get("model_full_name", "") or "").strip()
    if model_full_name and model_full_name != model_name:
        return f"{label} [{model_full_name}]"
    return label


def _model_full_name(row: dict[str, str]) -> str:
    """Return the full model run name when present, otherwise the display model name."""
    full_name = str(row.get("model_full_name", "") or "").strip()
    if full_name:
        return full_name
    return str(row.get("model_name", "") or "").strip()


def _group_filename(group_key: tuple[str, ...], sample_type: str, report_label: str) -> str:
    """
    Purpose:
        Build one stable per-group matrix filename.
    Parameters:
        group_key (tuple[str, ...]), one dataset grouping key tuple.
        sample_type (str), trajectory|chunk|uncertainty_trajectory label.
        report_label (str), short metric/report suffix such as mean_error.
    Return Dict:
        Not used. Returns one CSV filename string.
    Usage:
        Called by generate_pairwise_p_value_report while writing matrices.
    TODO:
        1) Encode dataset name.
        2) Append sample-type suffix.
        3) Append report-label suffix.
    """
    dataset_name = group_key[0]
    parts = [_normalize_text_token(dataset_name)]
    parts.append(_normalize_text_token(sample_type))
    parts.append(_normalize_text_token(report_label))
    return "__".join(parts) + ".csv"


def _write_matrix_csv(
    output_csv_path: Path,
    labels: list[str],
    pair_map: dict[tuple[str, str], float],
) -> dict:
    """
    Purpose:
        Write one symmetric model-vs-model p-value matrix CSV.
    Parameters:
        output_csv_path (Path), destination CSV path.
        labels (list[str]), ordered model labels for rows and columns.
        pair_map (dict[tuple[str, str], float]), symmetric pairwise p-values.
    Return Dict:
        "error_code": 0
        "output_csv_path": str
        "n_labels": int
    Usage:
        Called by generate_pairwise_p_value_report once per dataset/config group.
    TODO:
        1) Create parent directory.
        2) Write header row.
        3) Fill diagonal with 1.0.
        4) Fill off-diagonal cells from pair map.
    """
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["model"] + labels)
        for row_label in labels:
            row_values: list[str] = [row_label]
            for col_label in labels:
                if row_label == col_label:
                    row_values.append("1.0")
                    continue
                pvalue = pair_map.get((row_label, col_label))
                if pvalue is None:
                    row_values.append("")
                    continue
                row_values.append(f"{float(pvalue):.12g}")
            writer.writerow(row_values)
    return {
        "error_code": 0,
        "output_csv_path": str(output_csv_path),
        "n_labels": len(labels),
    }


def generate_pairwise_p_value_report(
    input_csv_path: str | Path,
    output_dir_path: str | Path,
    *,
    sample_type: str,
    metric_column: str,
    report_label: str | None = None,
) -> dict:
    """
    Purpose:
        Build one per-dataset model-vs-model p-value matrix folder from merged sample rows.
    Parameters:
        input_csv_path (str|Path), merged per-sample CSV such as traj_p_val.csv.
        output_dir_path (str|Path), destination folder for matrix CSV files.
        sample_type (str), label such as "trajectory" or "chunk".
        metric_column (str), numeric metric column used for paired comparison.
        report_label (str|None), short suffix used in output filenames.
    Return Dict:
        "error_code": 0 | -1
        "output_dir_path": str
        "manifest_csv_path": str
        "n_groups": int
        "n_pairs": int
    Usage:
        Called by batch aggregators after merging per-job p-value source CSVs.
    TODO:
        1) Validate input path and create output folder.
        2) Read merged sample rows.
        3) Group rows by dataset and config.
        4) Compute pairwise tests and one symmetric p-value matrix per group.
        5) Write manifest and long-form pair CSV for debugging.
    """
    input_path = Path(input_csv_path)
    output_dir = Path(output_dir_path)
    report_name = str(report_label or metric_column).strip()
    manifest_csv = output_dir / f"index__{_normalize_text_token(report_name)}.csv"
    pairs_csv = output_dir / f"all_pairs__{_normalize_text_token(report_name)}.csv"

    # 1. Validate input and prepare output folder.
    if not input_path.exists():
        LOGGER.info("P-value input is missing; skipping %s", input_path)
        return {
            "error_code": -1,
            "output_dir_path": str(output_dir),
            "manifest_csv_path": str(manifest_csv),
            "n_groups": 0,
            "n_pairs": 0,
        }
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Read merged sample rows.
    rows = _read_csv_rows(input_path)
    if not rows:
        write_rows_to_csv([], manifest_csv)
        write_rows_to_csv([], pairs_csv)
        return {
            "error_code": 0,
            "output_dir_path": str(output_dir),
            "manifest_csv_path": str(manifest_csv),
            "n_groups": 0,
            "n_pairs": 0,
        }

    # 3. Group rows by dataset and config.
    key_columns = ["dataset_name"]
    grouped: dict[tuple[str, ...], dict[str, list[dict[str, str]]]] = {}
    label_row_map: dict[tuple[str, ...], dict[str, dict[str, str]]] = {}
    for row in rows:
        group_key = _group_key(row, key_columns)
        model_label = _model_display_label(row)
        grouped.setdefault(group_key, {})
        grouped[group_key].setdefault(model_label, [])
        grouped[group_key][model_label].append(row)
        label_row_map.setdefault(group_key, {})
        if model_label not in label_row_map[group_key]:
            label_row_map[group_key][model_label] = row

    # 4. Compute pairwise p-values and write one matrix per group.
    manifest_rows: list[dict] = []
    pair_rows: list[dict] = []
    n_groups_written = 0
    for group_key, model_map in sorted(grouped.items()):
        model_labels = sorted(model_map.keys())
        if len(model_labels) < 2:
            continue

        pair_map: dict[tuple[str, str], float] = {}
        n_pairs_group = 0
        for model_a, model_b in combinations(model_labels, 2):
            paired_packet = _paired_metric_arrays(model_map[model_a], model_map[model_b], metric_column)
            if paired_packet["n_common"] < 2:
                continue
            test_packet = _paired_test_packet(
                paired_packet["values_a"],
                paired_packet["values_b"],
            )
            pair_map[(model_a, model_b)] = float(test_packet["wilcoxon_pvalue"])
            pair_map[(model_b, model_a)] = float(test_packet["wilcoxon_pvalue"])
            pair_rows.append(
                {
                    "sample_type": sample_type,
                    "dataset_name": group_key[0],
                    "metric_column": metric_column,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_common": test_packet["n_common"],
                    "mean_a": test_packet["mean_a"],
                    "mean_b": test_packet["mean_b"],
                    "mean_delta_a_minus_b": test_packet["mean_delta_a_minus_b"],
                    "relative_improvement_pct_a_vs_b": test_packet["relative_improvement_pct_a_vs_b"],
                    "paired_t_statistic": test_packet["paired_t_statistic"],
                    "paired_t_pvalue": test_packet["paired_t_pvalue"],
                    "wilcoxon_statistic": test_packet["wilcoxon_statistic"],
                    "wilcoxon_pvalue": test_packet["wilcoxon_pvalue"],
                    "model_a_full_name": _model_full_name(label_row_map[group_key].get(model_a, {})),
                    "model_b_full_name": _model_full_name(label_row_map[group_key].get(model_b, {})),
                }
            )
            n_pairs_group += 1

        if not pair_map:
            continue

        matrix_filename = _group_filename(group_key, sample_type, report_name)
        matrix_path = output_dir / matrix_filename
        matrix_packet = _write_matrix_csv(
            matrix_path,
            model_labels,
            pair_map,
        )
        assert matrix_packet["error_code"] == 0
        manifest_rows.append(
            {
                "sample_type": sample_type,
                "dataset_name": group_key[0],
                "metric_column": metric_column,
                "n_models": len(model_labels),
                "n_pairs": n_pairs_group,
                "matrix_csv_path": str(matrix_path),
                "labels": " | ".join(model_labels),
                "model_full_names": " | ".join(
                    _model_full_name(label_row_map[group_key].get(label, {})) for label in model_labels
                ),
            }
        )
        n_groups_written += 1

    # 5. Write manifest and long-form pair CSV for debugging.
    write_rows_to_csv(
        manifest_rows,
        manifest_csv,
        field_order=[
            "sample_type",
            "dataset_name",
            "metric_column",
            "n_models",
            "n_pairs",
            "matrix_csv_path",
            "labels",
            "model_full_names",
        ],
    )
    write_rows_to_csv(
        pair_rows,
        pairs_csv,
        field_order=[
            "sample_type",
            "dataset_name",
            "metric_column",
            "model_a",
            "model_b",
            "n_common",
            "mean_a",
            "mean_b",
            "mean_delta_a_minus_b",
            "relative_improvement_pct_a_vs_b",
            "paired_t_statistic",
            "paired_t_pvalue",
            "wilcoxon_statistic",
            "wilcoxon_pvalue",
            "model_a_full_name",
            "model_b_full_name",
        ],
    )
    return {
        "error_code": 0,
        "output_dir_path": str(output_dir),
        "manifest_csv_path": str(manifest_csv),
        "n_groups": n_groups_written,
        "n_pairs": len(pair_rows),
    }
