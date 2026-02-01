"""
Visualization script for trajectory evaluation results.

USAGE:
    python src/visualize_results.py

Generates:
    - Byte-wise heatmap: all models (BF/DF) as horizontal strips, row-normalized
    - Chunk-wise heatmap: all models (BF/DF) as horizontal strips, row-normalized
    - Model comparison: avg vs median error (best config per model+method)
    - Q1/Q2 vs avg error per model+method
    - Step size summary table (avg error vs runtime)
"""

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


# ================================================================
# CONFIGURATION
# ================================================================
CSV_PATH = "./bin/test_results/trajectory_evaluation_summary.csv"
PARQUET_DIR = "./bin/test_results/trajectory_evaluation_results"
OUTPUT_DIR = "./bin/test_results/figures"
ARCHIVED_CSV_PATH = "./archived/archived.csv"

# Plot styling
plt.style.use('default')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


# ================================================================
# LOAD DATA
# ================================================================
def load_summary_csv(csv_path: str) -> pd.DataFrame:
    """Load summary CSV produced by EvaluationManager."""
    df = pd.read_csv(csv_path)
    df = df.sort_values(["model_name", "denoise_method", "test_timestamp"])
    print(f"Loaded {len(df)} summary rows")
    return df


def load_parquet_results(parquet_dir: str) -> pd.DataFrame:
    """Load detailed parquet results (byte/chunk lists)."""
    parquet_path = Path(parquet_dir)
    if not parquet_path.exists():
        print(f"Parquet directory not found: {parquet_dir}")
        return pd.DataFrame()

    files = sorted(parquet_path.glob("*.parquet"))
    if not files:
        print(f"No parquet files found in: {parquet_dir}")
        return pd.DataFrame()

    frames = []
    for fpath in files:
        try:
            frames.append(pd.read_parquet(fpath))
        except Exception as exc:
            print(f"Failed to read {fpath}: {exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _to_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return []
    return []


def _clean_model_name(name: str) -> str:
    parts = str(name).split("_")
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        if len(parts[-2]) == 8 and len(parts[-1]) == 6:
            return "_".join(parts[:-2])
    return str(name)


def _display_name(row) -> str:
    return f"{_clean_model_name(row['model_name'])}_{row['denoise_method']}"


def _normalize_row(values: list) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr[arr <= 0] = np.nan
    if np.all(np.isnan(arr)):
        return np.full_like(arr, np.nan)
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    if abs(vmax - vmin) < 1e-12:
        return np.where(np.isnan(arr), np.nan, 0.0)
    return (arr - vmin) / (vmax - vmin)


def _buckle_positions_bytes(row, num_bytes: int) -> list:
    if any(k not in row for k in ["K", "Q1", "Q2"]):
        return []
    try:
        k_points = int(row["K"])
        q1_points = int(row["Q1"]) * 8
        q2_points = int(row["Q2"]) * 8
    except Exception:
        return []

    stride = k_points - q1_points - q2_points
    if stride <= 0:
        return []

    max_points = num_bytes * 8
    positions = []
    chunk_idx = 0
    while True:
        start = chunk_idx * stride
        end = start + k_points
        if start >= max_points:
            break
        head_buckle = start + q1_points
        tail_buckle = end - q2_points
        for pos in (head_buckle, tail_buckle):
            if 0 <= pos < max_points:
                positions.append(int(round(pos / 8.0)))
        chunk_idx += 1

    positions = sorted(set([p for p in positions if 0 <= p < num_bytes]))
    return positions


# ================================================================
# PLOT 1: Byte-wise heatmap (row-normalized)
# ================================================================
def plot_bytewise_heatmap(df: pd.DataFrame, output_dir: Path, show_buckles: bool = True):
    if df.empty:
        print("No parquet data available for byte-wise heatmap.")
        return

    rows = []
    metas = []
    df = df[
        (df["K"] == 256)
        & (df["Q1"] == 1)
        & (df["Q2"] == 12)
        & (df["N_steps"] == 1)
    ]
    for _, row in df.iterrows():
        values = _to_list(row.get("avg_l2_err_bw", []))
        if not values:
            continue
        rows.append(values)
        metas.append(row)

    if not rows:
        print("No byte-wise data found in parquet.")
        return

    max_len = max(len(r) for r in rows)
    norm_rows = []
    for r in rows:
        norm = _normalize_row(r)
        if len(norm) < max_len:
            pad = np.full((max_len - len(norm),), np.nan)
            norm = np.concatenate([norm, pad])
        norm_rows.append(norm)

    data = np.vstack(norm_rows)
    masked = np.ma.masked_invalid(data)

    fig_h = max(2.5, 0.5 * len(norm_rows))
    fig = plt.figure(figsize=(14, fig_h))
    ax = fig.add_subplot(1, 1, 1)

    cmap = plt.cm.Greys.copy()
    cmap.set_bad(color="white")

    im = ax.imshow(
        masked,
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=(0, max_len, len(norm_rows), 0),
    )

    ax.set_yticks([i + 0.5 for i in range(len(norm_rows))])
    ax.set_yticklabels([_display_name(m) for m in metas])
    ax.set_xlabel("Byte index")
    title = "Byte-wise Avg Error (row-normalized, black = higher error)"
    if metas:
        meta0 = metas[0]
        title += f"\nK={int(meta0['K'])}  Q1={int(meta0['Q1'])}  Q2={int(meta0['Q2'])}  tΔ={meta0['t_delta']:.4f}  N={int(meta0['N_steps'])}"
    ax.set_title(title)

    if show_buckles:
        for row_idx, meta in enumerate(metas):
            buckle_positions = _buckle_positions_bytes(meta, len(_to_list(meta.get("avg_l2_err_bw", []))))
            if len(buckle_positions) > 500:
                step = max(1, len(buckle_positions) // 500)
                buckle_positions = buckle_positions[::step]
            if buckle_positions:
                ys = np.full((len(buckle_positions),), row_idx + 0.92)
                ax.scatter(
                    buckle_positions,
                    ys,
                    marker="v",
                    s=8,
                    color="black",
                    alpha=0.6,
                    linewidths=0,
                )

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Row-normalized avg error")

    output_path = output_dir / "bytewise_heatmap.png"
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


# ================================================================
# PLOT 2: Chunk-wise heatmap (row-normalized)
# ================================================================
def plot_chunkwise_heatmap(df: pd.DataFrame, output_dir: Path):
    if df.empty:
        print("No parquet data available for chunk-wise heatmap.")
        return

    df = df[
        (df["Q1"] == 1)
        & (df["Q2"] == 12)
        & (df["N_steps"] == 1)
    ]
    if df.empty:
        print("No chunk-wise data for Q1=1, Q2=12, N_steps=1.")
        return

    rows = []
    metas = []
    for _, row in df.iterrows():
        values = _to_list(row.get("avg_l2_err_cw", []))
        if not values:
            continue
        rows.append(values)
        metas.append(row)

    if not rows:
        print("No chunk-wise data found in parquet.")
        return

    max_len = max(len(r) for r in rows)
    norm_rows = []
    for r in rows:
        norm = _normalize_row(r)
        if len(norm) < max_len:
            pad = np.full((max_len - len(norm),), np.nan)
            norm = np.concatenate([norm, pad])
        norm_rows.append(norm)

    data = np.vstack(norm_rows)
    masked = np.ma.masked_invalid(data)

    fig_h = max(2.5, 0.5 * len(norm_rows))
    fig = plt.figure(figsize=(14, fig_h))
    ax = fig.add_subplot(1, 1, 1)

    cmap = plt.cm.Greys.copy()
    cmap.set_bad(color="white")

    im = ax.imshow(
        masked,
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=(0, max_len, len(norm_rows), 0),
    )

    ax.set_yticks([i + 0.5 for i in range(len(norm_rows))])
    ax.set_yticklabels([_display_name(m) for m in metas])
    ax.set_xlabel("Chunk index")
    title = "Chunk-wise Avg Error (row-normalized, black = higher error)"
    if metas:
        meta0 = metas[0]
        title += f"\nK={int(meta0['K'])}  Q1={int(meta0['Q1'])}  Q2={int(meta0['Q2'])}  tΔ={meta0['t_delta']:.4f}  N={int(meta0['N_steps'])}"
    ax.set_title(title)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Row-normalized avg error")

    output_path = output_dir / "chunkwise_heatmap.png"
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


# ================================================================
# PLOT 3: Model comparison (BF - DF) for error and runtime
# ================================================================
def plot_model_bf_df_diffs(df: pd.DataFrame, output_dir: Path):
    if df.empty:
        print("No summary data available for BF-DF comparison.")
        return

    df = df.copy()
    df = df[
        (df["Q1"] == 1)
        & (df["Q2"] == 12)
        & (df["N_steps"] == 1)
    ]
    if df.empty:
        print("No summary rows for Q1=1, Q2=12, N_steps=1.")
        return

    df["model_base"] = df["model_name"].apply(_clean_model_name)

    agg = (
        df.groupby(["model_base", "denoise_method"], as_index=False)
        .agg(
            avg_l2_err_pw=("avg_l2_err_pw", "mean"),
            med_l2_err_pw=("med_l2_err_pw", "mean"),
            avg_denoise_time_sec=("avg_denoise_time_sec", "mean"),
            n_rows=("avg_l2_err_pw", "size"),
        )
    )

    diff_rows = []
    for model_base, group in agg.groupby("model_base"):
        methods = {row["denoise_method"]: row for _, row in group.iterrows()}
        if "BF" not in methods or "DF" not in methods:
            continue
        bf = methods["BF"]
        dfm = methods["DF"]
        diff_rows.append(
            {
                "model_base": model_base,
                "avg_err_diff": bf["avg_l2_err_pw"] - dfm["avg_l2_err_pw"],
                "med_err_diff": bf["med_l2_err_pw"] - dfm["med_l2_err_pw"],
                "runtime_diff": bf["avg_denoise_time_sec"] - dfm["avg_denoise_time_sec"],
                "bf_rows": bf["n_rows"],
                "df_rows": dfm["n_rows"],
            }
        )

    diff_df = pd.DataFrame(diff_rows)
    if diff_df.empty:
        print("No models with both BF and DF results for Q1=1, Q2=12, N_steps=1.")
        return

    diff_df = diff_df.sort_values("avg_err_diff")

    # Error diff chart (avg vs median)
    x = np.arange(len(diff_df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, diff_df["avg_err_diff"], width, label="Avg Error (BF-DF)", color="slategray")
    ax.bar(x + width / 2, diff_df["med_err_diff"], width, label="Median Error (BF-DF)", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(diff_df["model_base"], rotation=20, ha="right")
    ax.set_ylabel("L2 Error (m)")
    ax.set_title("BF - DF Error (Q1=1, Q2=12, N_steps=1)")
    ax.axhline(0, color="black", linewidth=1)
    max_abs_err = np.nanmax(np.abs(diff_df[["avg_err_diff", "med_err_diff"]].to_numpy()))
    if np.isfinite(max_abs_err) and max_abs_err > 0:
        ax.set_ylim(-1.1 * max_abs_err, 1.1 * max_abs_err)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    output_path = output_dir / "model_error_bf_minus_df.png"
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()

    # Runtime diff chart
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, diff_df["runtime_diff"], color="seagreen", label="Runtime (BF-DF)")
    ax.set_xticks(x)
    ax.set_xticklabels(diff_df["model_base"], rotation=20, ha="right")
    ax.set_ylabel("Runtime (sec)")
    ax.set_title("BF - DF Runtime (Q1=1, Q2=12, N_steps=1)")
    ax.axhline(0, color="black", linewidth=1)
    max_abs_rt = np.nanmax(np.abs(diff_df["runtime_diff"].to_numpy()))
    if np.isfinite(max_abs_rt) and max_abs_rt > 0:
        ax.set_ylim(-1.1 * max_abs_rt, 1.1 * max_abs_rt)
    ax.grid(axis="y", alpha=0.3)
    output_path = output_dir / "model_runtime_bf_minus_df.png"
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()

    diff_df.to_csv(output_dir / "model_bf_minus_df_summary.csv", index=False)


# ================================================================
# PLOT: Archived N_steps=10 BF-DF comparison (error + runtime)
# ================================================================
def plot_archived_nsteps10_bf_df(df: pd.DataFrame, output_dir: Path):
    if df.empty:
        print("No archived data available for N_steps=10 comparison.")
        return

    df = df.copy()
    df = df[df["N_steps"] == 10]
    if df.empty:
        print("No archived rows for N_steps=10.")
        return

    df["model_base"] = df["model_name"].apply(_clean_model_name)

    agg = (
        df.groupby(["model_base", "denoise_method"], as_index=False)
        .agg(
            avg_l2_err_pw=("avg_l2_err_pw", "mean"),
            med_l2_err_pw=("med_l2_err_pw", "mean"),
            avg_denoise_time_sec=("avg_denoise_time_sec", "mean"),
            n_rows=("avg_l2_err_pw", "size"),
        )
    )

    diff_rows = []
    for model_base, group in agg.groupby("model_base"):
        methods = {row["denoise_method"]: row for _, row in group.iterrows()}
        if "BF" not in methods or "DF" not in methods:
            continue
        bf = methods["BF"]
        dfm = methods["DF"]
        diff_rows.append(
            {
                "model_base": model_base,
                "avg_err_diff": bf["avg_l2_err_pw"] - dfm["avg_l2_err_pw"],
                "med_err_diff": bf["med_l2_err_pw"] - dfm["med_l2_err_pw"],
                "runtime_diff": bf["avg_denoise_time_sec"] - dfm["avg_denoise_time_sec"],
                "bf_rows": bf["n_rows"],
                "df_rows": dfm["n_rows"],
            }
        )

    diff_df = pd.DataFrame(diff_rows)
    if diff_df.empty:
        print("No archived models with both BF and DF results for N_steps=10.")
        return

    diff_df = diff_df.sort_values("avg_err_diff")

    x = np.arange(len(diff_df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, diff_df["avg_err_diff"], width, label="Avg Error (BF-DF)", color="slategray")
    ax.bar(x + width / 2, diff_df["med_err_diff"], width, label="Median Error (BF-DF)", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(diff_df["model_base"], rotation=20, ha="right")
    ax.set_ylabel("L2 Error (m)")
    ax.set_title("Archived BF - DF Error (N_steps=10)")
    ax.axhline(0, color="black", linewidth=1)
    max_abs_err = np.nanmax(np.abs(diff_df[["avg_err_diff", "med_err_diff"]].to_numpy()))
    if np.isfinite(max_abs_err) and max_abs_err > 0:
        ax.set_ylim(-1.1 * max_abs_err, 1.1 * max_abs_err)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    output_path = output_dir / "archived_n10_error_bf_minus_df.png"
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, diff_df["runtime_diff"], color="seagreen", label="Runtime (BF-DF)")
    ax.set_xticks(x)
    ax.set_xticklabels(diff_df["model_base"], rotation=20, ha="right")
    ax.set_ylabel("Runtime (sec)")
    ax.set_title("Archived BF - DF Runtime (N_steps=10)")
    ax.axhline(0, color="black", linewidth=1)
    max_abs_rt = np.nanmax(np.abs(diff_df["runtime_diff"].to_numpy()))
    if np.isfinite(max_abs_rt) and max_abs_rt > 0:
        ax.set_ylim(-1.1 * max_abs_rt, 1.1 * max_abs_rt)
    ax.grid(axis="y", alpha=0.3)
    output_path = output_dir / "archived_n10_runtime_bf_minus_df.png"
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()

    diff_df.to_csv(output_dir / "archived_n10_bf_minus_df_summary.csv", index=False)


# ================================================================
# PLOT 4: Q1/Q2 vs avg error per model
# ================================================================
def plot_q1_q2_vs_error(df: pd.DataFrame, output_dir: Path):
    if df.empty:
        print("No summary data available for Q1/Q2 plots.")
        return

    df = df.copy()
    df["model_base"] = df["model_name"].apply(_clean_model_name)

    for model_base, group in df.groupby("model_base"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        for method, mgroup in group.groupby("denoise_method"):
            q1_df = (
                mgroup.groupby("Q1", as_index=False)["avg_l2_err_pw"]
                .mean()
                .sort_values("Q1")
            )
            q2_df = (
                mgroup.groupby("Q2", as_index=False)["avg_l2_err_pw"]
                .mean()
                .sort_values("Q2")
            )

            axes[0].plot(q1_df["Q1"], q1_df["avg_l2_err_pw"], marker="o", label=method)
            axes[1].plot(q2_df["Q2"], q2_df["avg_l2_err_pw"], marker="o", label=method)

        axes[0].set_xlabel("Q1 (head bytes)")
        axes[0].set_ylabel("Avg L2 Error (m)")
        axes[0].set_title("Q1 vs Avg Error")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(title="Method")

        axes[1].set_xlabel("Q2 (tail bytes)")
        axes[1].set_ylabel("Avg L2 Error (m)")
        axes[1].set_title("Q2 vs Avg Error")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(title="Method")

        plt.suptitle(f"{model_base} (avg over other params)", fontsize=12)
        plt.tight_layout()

        output_path = output_dir / f"q1_q2_vs_error_{model_base}.png"
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved: {output_path}")
        plt.close()


# ================================================================
# TABLE: Step size vs accuracy/runtime
# ================================================================
def save_step_size_table(df: pd.DataFrame, output_dir: Path):
    if df.empty:
        print("No summary data available for step size table.")
        return

    df = df.copy()
    df["display_name"] = df.apply(_display_name, axis=1)

    summary = (
        df.groupby(["display_name", "t_delta", "N_steps"], as_index=False)
        .agg(
            avg_l2_err_pw=("avg_l2_err_pw", "mean"),
            med_l2_err_pw=("med_l2_err_pw", "mean"),
            avg_denoise_time_sec=("avg_denoise_time_sec", "mean"),
        )
        .sort_values(["display_name", "N_steps"])
    )

    output_path = output_dir / "step_size_summary.csv"
    summary.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


# ================================================================
# PLOT: Step size analysis per model
# ================================================================
def plot_step_size_per_model(df: pd.DataFrame, output_dir: Path):
    if df.empty:
        print("No summary data available for step size analysis.")
        return

    df = df.copy()
    df["display_name"] = df.apply(_display_name, axis=1)

    grouped = (
        df.groupby(["display_name", "N_steps"], as_index=False)
        .agg(
            avg_l2_err_pw=("avg_l2_err_pw", "mean"),
            avg_denoise_time_sec=("avg_denoise_time_sec", "mean"),
        )
        .sort_values(["display_name", "N_steps"])
    )

    for display_name, group in grouped.groupby("display_name"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(
            group["N_steps"],
            group["avg_l2_err_pw"],
            marker="o",
            color="steelblue",
        )
        axes[0].set_xlabel("N_steps")
        axes[0].set_ylabel("Avg L2 Error (m)")
        axes[0].set_title("Accuracy vs Step Size")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(
            group["N_steps"],
            group["avg_denoise_time_sec"],
            marker="o",
            color="seagreen",
        )
        axes[1].set_xlabel("N_steps")
        axes[1].set_ylabel("Avg Runtime (sec)")
        axes[1].set_title("Runtime vs Step Size")
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f"Step Size Analysis: {display_name}")
        plt.tight_layout()

        output_path = output_dir / f"step_size_{display_name}.png"
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved: {output_path}")
        plt.close()


# ================================================================
# MAIN
# ================================================================
def main():
    """Run requested visualizations."""

    print("="*80)
    print("TRAJECTORY EVALUATION VISUALIZATION")
    print("="*80)

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    summary_df = load_summary_csv(CSV_PATH)
    parquet_df = load_parquet_results(PARQUET_DIR)
    archived_path = Path(ARCHIVED_CSV_PATH)
    archived_df = pd.DataFrame()
    if archived_path.exists():
        archived_df = load_summary_csv(str(archived_path))
    else:
        print(f"Archived CSV not found, skipping: {ARCHIVED_CSV_PATH}")

    if len(summary_df) == 0 and len(parquet_df) == 0:
        print("ERROR: No data found in CSV or parquet")
        return

    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80 + "\n")

    # Generate all plots
    if len(parquet_df) > 0:
        plot_bytewise_heatmap(parquet_df, output_dir, show_buckles=True)
        plot_chunkwise_heatmap(parquet_df, output_dir)

    if len(summary_df) > 0:
        plot_model_bf_df_diffs(summary_df, output_dir)
        plot_q1_q2_vs_error(summary_df, output_dir)
        save_step_size_table(summary_df, output_dir)
        plot_step_size_per_model(summary_df, output_dir)

    if len(archived_df) > 0:
        plot_archived_nsteps10_bf_df(archived_df, output_dir)

    print("\n" + "="*80)
    print(f"ALL VISUALIZATIONS SAVED TO: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
