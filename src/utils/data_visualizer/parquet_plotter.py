#!/usr/bin/env python3
"""
Plot heatmaps directly from saved parquet results.

Two entry modes are provided:
1) benchmark   -> trajectory benchmark parquet (avg_l2_err_pw_profile / avg_l2_err_cw)
2) uncertainty -> uncertainty parquet aggregates (pass_rate / excess / distance)
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Iterable


def _load_parquet_tree(input_dir: Path) -> pd.DataFrame:
    import pandas as pd

    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        files = sorted(input_dir.rglob("*.parquet"))
    if not files:
        return pd.DataFrame()

    frames = []
    for path in files:
        try:
            frames.append(pd.read_parquet(path))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _clean_model_name(name: str) -> str:
    parts = str(name).split("_")
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        if len(parts[-2]) == 8 and len(parts[-1]) == 6:
            return "_".join(parts[:-2])
    return str(name)


def _normalize_baseline_name(name: str) -> str | None:
    key = str(name).strip().lower().replace(" ", "_")
    if key in {"test_data"}:
        return None
    return str(name)


def _display_name_from_row(row: pd.Series) -> str:
    model_name = _clean_model_name(str(row.get("model_name", "NA")))
    model_tag = str(row.get("model_tag", ""))
    if model_tag == "Baseline":
        return model_name.replace("_", " ")
    return model_name.replace("_", " ")


def _normalize_row(values: np.ndarray) -> np.ndarray:
    import numpy as np

    arr = np.asarray(values, dtype=float).copy()
    arr[arr <= 0] = np.nan
    if np.all(np.isnan(arr)):
        return np.full_like(arr, np.nan)
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)
    if abs(hi - lo) < 1e-12:
        return np.where(np.isnan(arr), np.nan, 0.0)
    return (arr - lo) / (hi - lo)


def _normalize_global_log(matrix: np.ndarray) -> np.ndarray:
    import numpy as np

    arr = np.asarray(matrix, dtype=float).copy()
    arr[arr <= 0] = np.nan
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.full_like(arr, np.nan)

    log_arr = np.log10(arr)
    lo = np.nanmin(log_arr)
    hi = np.nanmax(log_arr)
    if abs(hi - lo) < 1e-12:
        return np.where(np.isnan(log_arr), np.nan, 0.0)
    return (log_arr - lo) / (hi - lo)


def _chunk_row_order_key(model_tag: str, model_name: str, fallback_idx: int) -> tuple:
    key_name = str(model_name).strip().lower()
    if str(model_tag) == "Baseline":
        bucket = 0
    elif "mlp" in key_name:
        bucket = 2
    else:
        bucket = 1
    return (bucket, key_name, fallback_idx)


def _plot_heatmap(
    data: np.ndarray,
    labels: list[str],
    x_label: str,
    title: str,
    output_path: Path,
    cmap_name: str = "Greys",
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    masked = np.ma.masked_invalid(data)
    fig_h = max(2.5, 0.5 * len(labels))
    fig, ax = plt.subplots(figsize=(14, fig_h))

    cmap = plt.get_cmap(cmap_name).copy()
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
    ax.set_yticklabels(labels)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Value")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_chunk_from_summary_csv(input_dir: Path, output_dir: Path) -> bool:
    import pandas as pd
    import numpy as np

    summary_specs = [
        (
            "chunk_pointwise_summary.csv",
            "point_",
            "Point index",
            "Chunk Point-wise Error (from chunk summary, raw -> global log-normalized)",
        ),
        (
            "chunk_bytewise_summary.csv",
            "byte_",
            "Byte index",
            "Chunk Byte-wise Error (from chunk summary, raw -> global log-normalized)",
        ),
    ]

    for csv_name, column_prefix, x_label, title in summary_specs:
        csv_path = input_dir / csv_name
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        value_cols = [c for c in df.columns if c.startswith(column_prefix)]
        if not value_cols:
            continue

        value_cols = sorted(value_cols, key=lambda c: int(c.split("_")[1]))

        labels = []
        rows = []
        tags = []
        names = []
        for _, row in df.iterrows():
            raw_name = str(row.get("model_name", "NA"))
            norm_name = _normalize_baseline_name(raw_name)
            if norm_name is None:
                continue
            vals = pd.to_numeric(row[value_cols], errors="coerce").to_numpy(dtype=float)
            if vals.size == 0:
                continue
            vals = np.asarray(vals, dtype=float)
            rows.append(vals)
            tags.append(str(row.get("model_tag", "")))
            names.append(norm_name)
            labels.append(str(norm_name).replace("_", " "))

        if not rows:
            continue

        order = sorted(
            range(len(rows)),
            key=lambda i: _chunk_row_order_key(tags[i], names[i], i),
        )
        rows = [rows[i] for i in order]
        labels = [labels[i] for i in order]

        matrix = _normalize_global_log(np.vstack(rows))
        _plot_heatmap(
            data=matrix,
            labels=labels,
            x_label=x_label,
            title=title,
            output_path=output_dir / "chunkwise_heatmap.png",
            cmap_name="Greys",
        )
        return True

    return False


def _to_list_like(value) -> list[float]:
    import numpy as np

    if isinstance(value, list):
        return [float(v) for v in value]
    if isinstance(value, np.ndarray):
        return [float(v) for v in value.tolist()]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [float(v) for v in parsed]
        except Exception:
            return []
    return []


def _plot_benchmark_from_parquet(df: pd.DataFrame, output_dir: Path, field: str, output_name: str, x_label: str) -> bool:
    import numpy as np

    if df.empty or field not in df.columns:
        return False

    work = df.copy()
    if {"K", "Q1", "Q2"}.issubset(work.columns):
        model_mask = (
            (work["K"] == 256)
            & (work["Q1"] == 1)
            & (work["Q2"] == 12)
        )
        baseline_mask = (
            work.get("model_tag", np.full((len(work),), "", dtype=object)) == "Baseline"
        )
        work = work[model_mask | baseline_mask].copy()

    if "model_name" in work.columns:
        work["model_name"] = work["model_name"].astype(str).map(_normalize_baseline_name)
        work = work[work["model_name"].notna()]

    labels: list[str] = []
    rows: list[np.ndarray] = []
    tags: list[str] = []
    names: list[str] = []
    for _, row in work.iterrows():
        values = _to_list_like(row.get(field, []))
        if not values:
            continue
        vec = np.asarray(values, dtype=float)
        rows.append(vec)
        tags.append(str(row.get("model_tag", "")))
        names.append(str(row.get("model_name", "NA")))
        labels.append(_display_name_from_row(row))

    if not rows:
        return False

    order = sorted(range(len(rows)), key=lambda i: _chunk_row_order_key(tags[i], names[i], i))
    rows = [rows[i] for i in order]
    labels = [labels[i] for i in order]
    matrix = _normalize_global_log(_to_matrix(rows))
    _plot_heatmap(
        data=matrix,
        labels=labels,
        x_label=x_label,
        title=f"{x_label} Heatmap (global log-normalized)",
        output_path=output_dir / output_name,
        cmap_name="Greys",
    )
    return True


def run_benchmark(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _load_parquet_tree(input_dir)
    used_traj_point = _plot_benchmark_from_parquet(
        df=df,
        output_dir=output_dir,
        field="avg_l2_err_pw_profile",
        output_name="pointwise_heatmap.png",
        x_label="Point index",
    )
    if not used_traj_point:
        used_traj_point = _plot_benchmark_from_parquet(
            df=df,
            output_dir=output_dir,
            field="avg_l2_err_bw",
            output_name="bytewise_heatmap.png",
            x_label="Byte index",
        )

    used_chunk_csv = _plot_chunk_from_summary_csv(input_dir=input_dir, output_dir=output_dir)
    used_traj_chunk = False
    if not used_chunk_csv:
        used_traj_chunk = _plot_benchmark_from_parquet(
            df=df,
            output_dir=output_dir,
            field="avg_l2_err_cw",
            output_name="chunkwise_heatmap.png",
            x_label="Chunk index",
        )
    if not used_chunk_csv and not used_traj_point and not used_traj_chunk:
        raise RuntimeError(
            f"No readable benchmark heatmap sources under: {input_dir}"
        )


def _apply_filters(
    df: pd.DataFrame,
    aggregate_type: str,
    metric: str,
    k: int | None,
    q1: int | None,
    q2: int | None,
) -> pd.DataFrame:
    import pandas as pd

    out = df.copy()
    if "aggregate_type" in out.columns:
        out = out[out["aggregate_type"] == aggregate_type]
    if metric not in out.columns:
        raise ValueError(f"Metric '{metric}' not found in parquet columns.")
    if k is not None and "K" in out.columns:
        out = out[pd.to_numeric(out["K"], errors="coerce") == float(k)]
    if q1 is not None and "Q1" in out.columns:
        out = out[pd.to_numeric(out["Q1"], errors="coerce") == float(q1)]
    if q2 is not None and "Q2" in out.columns:
        out = out[pd.to_numeric(out["Q2"], errors="coerce") == float(q2)]
    return out


def _group_keys(df: pd.DataFrame) -> list[str]:
    preferred = [
        "model_name",
        "model_tag",
        "aggregate_type",
        "K",
        "Q1",
        "Q2",
        "test_timestamp",
    ]
    return [col for col in preferred if col in df.columns]


def _to_matrix(vectors: Iterable[np.ndarray]) -> np.ndarray:
    import numpy as np

    rows = list(vectors)
    if not rows:
        return np.zeros((0, 0), dtype=float)
    max_len = max(len(x) for x in rows)
    out = []
    for v in rows:
        vv = np.asarray(v, dtype=float)
        if len(vv) < max_len:
            pad = np.full((max_len - len(vv),), np.nan)
            vv = np.concatenate([vv, pad])
        out.append(vv)
    return np.vstack(out)


def run_uncertainty(
    input_dir: Path,
    output_dir: Path,
    aggregate_type: str,
    metric: str,
    normalize: str,
    k: int | None,
    q1: int | None,
    q2: int | None,
) -> None:
    import pandas as pd

    df = _load_parquet_tree(input_dir)
    if df.empty:
        raise RuntimeError(f"No readable parquet files found under: {input_dir}")

    df = _apply_filters(
        df=df,
        aggregate_type=aggregate_type,
        metric=metric,
        k=k,
        q1=q1,
        q2=q2,
    )
    if df.empty:
        raise RuntimeError("No rows left after uncertainty filters.")

    if "model_name" in df.columns:
        df["model_name"] = df["model_name"].astype(str).map(_normalize_baseline_name)
        df = df[df["model_name"].notna()]
    if "position_index" not in df.columns:
        raise ValueError("Uncertainty parquet must include 'position_index'.")
    if df.empty:
        raise RuntimeError("No rows left after baseline normalization.")

    labels = []
    vectors = []
    key_cols = _group_keys(df)
    for _, group in df.groupby(key_cols, dropna=False):
        grp = group.sort_values("position_index")
        values = pd.to_numeric(grp[metric], errors="coerce").to_numpy(dtype=float)
        if values.size == 0:
            continue
        if normalize == "row":
            values = _normalize_row(values)
        label = _display_name_from_row(grp.iloc[0])
        labels.append(label)
        vectors.append(values)

    if not vectors:
        raise RuntimeError("No valid uncertainty vectors to plot.")

    matrix = _to_matrix(vectors)
    x_label = "Position index"
    title = f"Uncertainty {aggregate_type} {metric}"
    if normalize == "row":
        title += " (row-normalized)"
    out_name = f"uncertainty_{aggregate_type}_{metric}_heatmap.png"
    _plot_heatmap(
        data=matrix,
        labels=labels,
        x_label=x_label,
        title=title,
        output_path=output_dir / out_name,
        cmap_name="Greys",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate heatmaps from parquet outputs for benchmark-test or uncertainty-test."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_bench = sub.add_parser("benchmark", help="Plot benchmark point/chunk heatmaps from parquet.")
    p_bench.add_argument("--input-dir", required=True, type=Path, help="Directory containing benchmark parquet files.")
    p_bench.add_argument("--output-dir", type=Path, default=None, help="Output directory. Default: <input-dir>/figures")

    p_unc = sub.add_parser("uncertainty", help="Plot uncertainty aggregate heatmap from parquet.")
    p_unc.add_argument("--input-dir", required=True, type=Path, help="Directory containing uncertainty parquet files.")
    p_unc.add_argument("--output-dir", type=Path, default=None, help="Output directory. Default: <input-dir>/figures")
    p_unc.add_argument(
        "--aggregate-type",
        default="chunk_point_avg",
        choices=["chunk_point_avg", "trajectory_point_avg"],
        help="Which uncertainty aggregate to visualize.",
    )
    p_unc.add_argument(
        "--metric",
        default="pass_rate",
        choices=["pass_rate", "mean_excess", "mean_distance", "mean_accuracy"],
        help="Metric column to plot.",
    )
    p_unc.add_argument(
        "--normalize",
        default="none",
        choices=["none", "row"],
        help="Normalization strategy for uncertainty mode.",
    )
    p_unc.add_argument("--K", type=int, default=None, help="Optional K filter.")
    p_unc.add_argument("--Q1", type=int, default=None, help="Optional Q1 filter.")
    p_unc.add_argument("--Q2", type=int, default=None, help="Optional Q2 filter.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    output_dir = args.output_dir or (args.input_dir / "figures")

    if args.mode == "benchmark":
        run_benchmark(input_dir=args.input_dir, output_dir=output_dir)
        return

    run_uncertainty(
        input_dir=args.input_dir,
        output_dir=output_dir,
        aggregate_type=args.aggregate_type,
        metric=args.metric,
        normalize=args.normalize,
        k=args.K,
        q1=args.Q1,
        q2=args.Q2,
    )


if __name__ == "__main__":
    main()
