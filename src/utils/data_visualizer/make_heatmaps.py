#!/usr/bin/env python3
"""Generate benchmark heatmaps from aggregated summary CSVs.

This module is the canonical post-benchmark heatmap pass used by
``src/run_benchmarks.py`` after all evaluation summaries are fully aggregated.
It emits dataset-scoped figures under:

    <run_dir>/dataset_heatmaps/<dataset_name>/

Supported inputs inside ``run_dir``:
    - trajectory_pointwise_summary.csv
    - chunk_pointwise_summary.csv
    - chunk_bytewise_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from pathlib import Path

import numpy as np


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def _safe_name(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._@-]+", "_", str(value))
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "dataset"


def _clean_model_token(name: str) -> str:
    token = str(name or "").strip()
    token = re.sub(r"_\d{8}_\d{6}$", "", token)
    return token


def _family_code(model_tag: str) -> str:
    token = str(model_tag or "").strip().lower()
    if token.startswith("rectifiedtraj"):
        return "RT"
    if token.startswith("directreg"):
        return "DR"
    if token.startswith("residualreg"):
        return "DR"
    if token.startswith("baseline"):
        return "BL"
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", token) if p]
    if len(parts) >= 2:
        return (parts[0][0] + parts[1][0]).upper()
    if parts:
        return parts[0][:2].upper()
    return "NA"


def _arch_label(model_name: str) -> str:
    token = _clean_model_token(str(model_name or ""))
    lower = token.lower()
    if lower.startswith("hybrid"):
        return "Hybrid"
    if lower.startswith("cnn"):
        return "CNN"
    if lower.startswith("transformer"):
        return "Trans."
    if lower.startswith("mlp"):
        return "MLP"
    if token:
        return token.split("_", 1)[0]
    return "NA"


def _format_q_line(q_value: str) -> str:
    q_text = str(q_value or "").strip()
    if not q_text:
        q_text = "NA"
    return f"Q = {q_text}"


def _trajectory_display_label(row: dict[str, str]) -> str:
    raw_label = str(row.get("model_dir", "") or "").strip()
    q_value = str(row.get("Q1", "") or "").strip()
    if raw_label.startswith("kalman_rts"):
        return raw_label
    if raw_label == "test_data":
        return raw_label

    if "/" in raw_label:
        family, model_name = raw_label.split("/", 1)
        base = f"{_family_code(family)} {_arch_label(model_name)}"
    else:
        base = raw_label
    return f"{base}\n{_format_q_line(q_value)}"


def _chunk_display_label(row: dict[str, str]) -> str:
    model_name = str(row.get("model_name", "") or "NA").strip()
    model_tag = str(row.get("model_tag", "") or "").strip()
    q_value = str(row.get("Q1", "") or "").strip()

    if model_tag.lower() == "baseline" or model_name.startswith("kalman_rts"):
        return model_name

    base = f"{_family_code(model_tag)} {_arch_label(model_name)}"
    return f"{base}\n{_format_q_line(q_value)}"


def _to_float(raw: str) -> float:
    text = str(raw or "").strip()
    if text in {"", "NA", "nan", "None"}:
        return float("nan")
    return float(text)


def _read_grouped_summary(
    csv_path: Path,
    *,
    value_prefix: str,
    label_builder,
) -> dict[str, list[tuple[str, np.ndarray]]]:
    if not csv_path.exists():
        return {}

    grouped: dict[str, list[tuple[str, np.ndarray]]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        value_cols = [c for c in fieldnames if c.startswith(f"{value_prefix}_")]
        value_cols = sorted(value_cols, key=lambda c: int(c.split("_", 1)[1]))
        if not value_cols:
            return {}

        for row in reader:
            dataset_name = str(row.get("dataset_name", "") or "").strip()
            if not dataset_name:
                continue
            values = np.asarray([_to_float(row.get(col, "")) for col in value_cols], dtype=float)
            if values.size == 0 or np.all(~np.isfinite(values)):
                continue
            label = label_builder(row)
            grouped.setdefault(dataset_name, []).append((label, values))
    return grouped


def _normalize_global_log(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float).copy()
    arr[~np.isfinite(arr)] = np.nan
    arr[arr <= 0] = np.nan
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.full_like(arr, np.nan)

    log_arr = np.log10(arr)
    lo = np.nanmin(log_arr)
    hi = np.nanmax(log_arr)
    if not math.isfinite(lo) or not math.isfinite(hi) or abs(hi - lo) < 1e-12:
        return np.where(np.isnan(log_arr), np.nan, 0.0)
    return (log_arr - lo) / (hi - lo)


def _to_matrix(rows: list[np.ndarray]) -> np.ndarray:
    if not rows:
        return np.zeros((0, 0), dtype=float)
    max_len = max(len(row) for row in rows)
    out = []
    for row in rows:
        arr = np.asarray(row, dtype=float)
        if arr.shape[0] < max_len:
            pad = np.full((max_len - arr.shape[0],), np.nan)
            arr = np.concatenate([arr, pad])
        out.append(arr)
    return np.vstack(out)


def _plot_heatmap(
    *,
    data: np.ndarray,
    labels: list[str],
    x_label: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    font_size = 12
    masked = np.ma.masked_invalid(data)
    fig_h = max(2.5, 0.46 * len(labels))
    fig_w = max(5.2, min(7.2, 0.012 * max(1, data.shape[1]) + 4.6))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = LinearSegmentedColormap.from_list(
        "rectifiedtraj_report_heatmap",
        ["white", "#32b44a", "#1f5fff"],
    )
    cmap.set_bad(color="white")
    im = ax.imshow(
        masked,
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=(0, data.shape[1], len(labels), 0),
    )
    ax.set_yticks([i + 0.5 for i in range(len(labels))])
    ax.set_yticklabels(labels, fontsize=font_size)
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Log normalized error", fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _render_grouped_heatmaps(
    grouped: dict[str, list[tuple[str, np.ndarray]]],
    *,
    output_root: Path,
    output_name: str,
    x_label: str,
) -> int:
    rendered = 0
    for dataset_name, entries in sorted(grouped.items()):
        if not entries:
            continue
        labels = [label for label, _ in entries]
        matrix = _to_matrix([values for _, values in entries])
        matrix = _normalize_global_log(matrix)
        if matrix.size == 0 or np.all(np.isnan(matrix)):
            continue
        target_dir = output_root / _safe_name(dataset_name)
        _plot_heatmap(
            data=matrix,
            labels=labels,
            x_label=x_label,
            output_path=target_dir / output_name,
        )
        rendered += 1
    return rendered


def generate_run_heatmaps(run_dir: Path) -> dict[str, int]:
    run_dir = Path(run_dir)
    output_root = run_dir / "dataset_heatmaps"

    trajectory_grouped = _read_grouped_summary(
        run_dir / "trajectory_pointwise_summary.csv",
        value_prefix="point",
        label_builder=_trajectory_display_label,
    )
    chunk_point_grouped = _read_grouped_summary(
        run_dir / "chunk_pointwise_summary.csv",
        value_prefix="point",
        label_builder=_chunk_display_label,
    )
    chunk_byte_grouped = _read_grouped_summary(
        run_dir / "chunk_bytewise_summary.csv",
        value_prefix="byte",
        label_builder=_chunk_display_label,
    )

    counts = {
        "trajectory_pointwise": _render_grouped_heatmaps(
            trajectory_grouped,
            output_root=output_root,
            output_name="trajectory_pointwise_heatmap.png",
            x_label="Point index",
        ),
        "chunk_pointwise": _render_grouped_heatmaps(
            chunk_point_grouped,
            output_root=output_root,
            output_name="chunk_pointwise_heatmap.png",
            x_label="Point index",
        ),
        "chunk_bytewise": _render_grouped_heatmaps(
            chunk_byte_grouped,
            output_root=output_root,
            output_name="chunk_bytewise_heatmap.png",
            x_label="Byte index",
        ),
    }
    return counts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate dataset heatmaps from aggregated benchmark summaries.")
    parser.add_argument("--run-dir", required=True, type=Path, help="Benchmark run directory under bin/test_results.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    counts = generate_run_heatmaps(args.run_dir)
    total = sum(counts.values())
    print(f"[HEATMAPS] Generated {total} dataset heatmap sets under {Path(args.run_dir) / 'dataset_heatmaps'}")
    for key, value in counts.items():
        print(f"[HEATMAPS] {key}: {value}")


if __name__ == "__main__":
    main()
