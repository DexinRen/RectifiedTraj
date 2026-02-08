#!/usr/bin/env python3
"""
Plot heatmaps directly from saved parquet results.

Two entry modes are provided:
1) benchmark   -> trajectory benchmark parquet (avg_l2_err_bw / avg_l2_err_cw)
2) uncertainty -> uncertainty parquet aggregates (pass_rate / margin / distance)
"""

from __future__ import annotations

import argparse
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
    if key == "kalman_rts_ts":
        return None
    if key == "kalman_rts_notime":
        return "kalman_rts"
    if key in {"test_data"}:
        return None
    return str(name)


def _display_name_from_row(row: pd.Series) -> str:
    model_name = _clean_model_name(str(row.get("model_name", "NA")))
    method = str(row.get("denoise_method", "N/A"))
    model_tag = str(row.get("model_tag", ""))
    if model_tag == "Baseline" or method in {"N/A", "Baseline"}:
        return model_name.replace("_", " ")
    return f"{model_name}_{method}".replace("_", " ")


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

    csv_path = input_dir / "chunk_bytewise_summary.csv"
    if not csv_path.exists():
        return False

    df = pd.read_csv(csv_path)
    byte_cols = [c for c in df.columns if c.startswith("byte_")]
    if not byte_cols:
        return False

    # Ensure byte order is numeric.
    byte_cols = sorted(byte_cols, key=lambda c: int(c.split("_")[1]))

    labels = []
    rows = []
    tags = []
    names = []
    for _, row in df.iterrows():
        raw_name = str(row.get("model_name", "NA"))
        norm_name = _normalize_baseline_name(raw_name)
        if norm_name is None:
            continue
        vals = pd.to_numeric(row[byte_cols], errors="coerce").to_numpy(dtype=float)
        if vals.size == 0:
            continue
        vals = _normalize_row(vals)
        rows.append(vals)
        tags.append(str(row.get("model_tag", "")))
        names.append(norm_name)
        labels.append(str(norm_name).replace("_", " "))

    if not rows:
        return False

    order = sorted(
        range(len(rows)),
        key=lambda i: _chunk_row_order_key(tags[i], names[i], i),
    )
    rows = [rows[i] for i in order]
    labels = [labels[i] for i in order]

    matrix = np.vstack(rows)
    _plot_heatmap(
        data=matrix,
        labels=labels,
        x_label="Byte index",
        title="Chunk Byte-wise Error (from chunk summary, row-normalized)",
        output_path=output_dir / "chunkwise_heatmap.png",
        cmap_name="Greys",
    )
    return True


def run_benchmark(input_dir: Path, output_dir: Path) -> None:
    # Reuse existing benchmark plotting pipeline from visualizer_traj_test.py.
    from visualizer_traj_test import plot_bytewise_heatmap, plot_chunkwise_heatmap

    output_dir.mkdir(parents=True, exist_ok=True)
    df = _load_parquet_tree(input_dir)
    if not df.empty:
        plot_bytewise_heatmap(df, output_dir, show_buckles=True)

    used_chunk_csv = _plot_chunk_from_summary_csv(input_dir=input_dir, output_dir=output_dir)
    if not used_chunk_csv:
        if df.empty:
            raise RuntimeError(
                f"No readable parquet files and no chunk summary CSV under: {input_dir}"
            )
        plot_chunkwise_heatmap(df, output_dir)


def _apply_filters(
    df: pd.DataFrame,
    aggregate_type: str,
    metric: str,
    k: int | None,
    q1: int | None,
    q2: int | None,
    n_steps: int | None,
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
    if n_steps is not None and "N_steps" in out.columns:
        out = out[pd.to_numeric(out["N_steps"], errors="coerce") == float(n_steps)]
    return out


def _group_keys(df: pd.DataFrame) -> list[str]:
    preferred = [
        "model_name",
        "model_tag",
        "denoise_method",
        "aggregate_type",
        "K",
        "Q1",
        "Q2",
        "t_delta",
        "N_steps",
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
    n_steps: int | None,
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
        n_steps=n_steps,
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

    p_bench = sub.add_parser("benchmark", help="Plot benchmark byte/chunk heatmaps from parquet.")
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
        choices=["pass_rate", "mean_signed_margin", "mean_distance", "mean_accuracy"],
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
    p_unc.add_argument("--N-steps", type=int, default=None, help="Optional N_steps filter.")
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
        n_steps=args.N_steps,
    )


if __name__ == "__main__":
    main()
