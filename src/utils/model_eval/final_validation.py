# THIS TEST SCRIPT ONLY USE DATA FROM VALIDATION FOLD
# THIS TEST SCRIPT ONLY TEST CHUNK-WISE ONETIME PREDICTION ACC
# IN NORMAL RUN, RECTIFIED FLOW MODEL SHOULD BE USED TO MAKE SMALL STEP PREDICTION ITERATIVELY
import re
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from theta_model import (
    thetaMLP,
    thetaTransformer,
    thetaCNN1D,
    thetaHybridCNNTransformer,
    # add future classes here automatically supported
)


# ================================================================
# === Load checkpoint into model
# ================================================================
def load_ckpt(model: torch.nn.Module, ckpt_path: Path, device: torch.device):
    """
    Load training snapshot checkpoint:
        {
            "model_state_dict": ...,
            "optimizer_state_dict": ...,
            "scheduler_state_dict": ...,
            "epoch": ...,
            "global_step": ...,
            ...
        }
    """
    blob = torch.load(ckpt_path, map_location=device)

    if "model_state_dict" not in blob:
        raise KeyError(f"Checkpoint missing model_state_dict: {ckpt_path}")

    sd = blob["model_state_dict"]
    model.load_state_dict(sd)
    return model


# ================================================================
# === Core L2 evaluation on big val set
# ================================================================
@torch.no_grad()
def large_scale_eval(model, device, big_path, K=256, Q=1, batch_size=64):

    blob = torch.load(big_path, map_location="cpu")

    X_t = blob["X_t"][:, :, :2].to(device)
    V   = blob["V"].to(device)
    t   = blob["t"].to(device)
    N   = X_t.shape[0]

    byte_sum = torch.zeros(32, dtype=torch.float32, device=device)
    byte_cnt = torch.zeros(32, dtype=torch.float32, device=device)

    global_err = []

    model.eval()

    for i in range(0, N, batch_size):
        xb = X_t[i:i+batch_size]
        vb = V[i:i+batch_size]
        tb = t[i:i+batch_size]

        pred = model(xb, tb)

        diff = pred - vb
        l2 = torch.sqrt((diff ** 2).sum(dim=-1))  # (batch, K)

        # Sample-level error: mean over all K indices
        sample_err = l2.mean(dim=1)               # (batch,)

        global_err.extend(sample_err.cpu().tolist())

        # Byte errors: mean over each group of 8
        for b in range(32):
            s = b * 8
            e = s + 8
            seg = l2[:, s:e]                     # (batch, 8)
            byte_sum[b] += seg.sum().item()
            byte_cnt[b] += seg.numel()

    # ------------------------------
    # Device-aligned byte_mean
    # ------------------------------
    byte_mean = torch.zeros(32, dtype=torch.float32, device=device)
    nonzero = byte_cnt > 0
    byte_mean[nonzero] = byte_sum[nonzero] / byte_cnt[nonzero]

    byte_mean = byte_mean.cpu().numpy()
    global_err = np.array(global_err)

    return {
        "mean": float(global_err.mean()),
        "median": float(np.median(global_err)),
        "std": float(global_err.std()),
        "byte_mean": byte_mean,
    }


# ================================================================
# === Buckle detection (Q1, Q2)
# ================================================================
def detect_buckles(byte_mean: np.ndarray):
    """
    Given 32-length byte_mean error vector, compute:
        - Q1: prefix buckle length (bytes) above threshold
        - Q2: suffix buckle length (bytes) above threshold
    Threshold = Q3 + 1.5 * IQR over all 32 bytes.
    """

    e = np.asarray(byte_mean)
    if e.shape[0] != 32:
        raise ValueError(f"byte_mean must have length 32, got {e.shape[0]}")

    Q1p = np.percentile(e, 25)
    Q3p = np.percentile(e, 75)
    IQR = Q3p - Q1p
    T = Q3p + 1.5 * IQR

    # prefix buckle
    front = 0
    for i in range(32):
        if e[i] > T:
            front += 1
        else:
            break

    # suffix buckle
    tail = 0
    for i in range(31, -1, -1):
        if e[i] > T:
            tail += 1
        else:
            break

    # interior check: ensure no interior > T after trimming
    while front + tail < 32:
        mid = e[front:32 - tail]
        if np.any(mid > T):
            # shrink the side with larger error
            if e[front] > e[31 - tail]:
                front += 1
            else:
                tail += 1
        else:
            break

    return {"Q1": int(front), "Q2": int(tail)}

def plot_all_ckpt_heatmaps(model_name: str, results: dict, out_path: Path):
    """
    results: dict ckpt_name -> { "byte_mean": np.array of shape (32,) }

    Produces one big heatmap:
        rows = ckpts
        cols = bytes 0..31
        colormap: white -> red -> purple
        step numbers shown at left of rows
    """

    # sort ckpts by step number
    def step_of(name):
        m = re.search(r"_s(\d+)", name)
        return int(m.group(1)) if m else -1

    ordered = sorted(results.items(), key=lambda kv: step_of(kv[0]))

    # assemble matrix
    heat_matrix = np.vstack([s["byte_mean"] for _, s in ordered])
    steps = [step_of(k) for k, _ in ordered]

    # custom colormap (white -> red -> purple)
    cmap = LinearSegmentedColormap.from_list(
        "errmap",
        ["white", "red", "purple"]
    )

    plt.figure(figsize=(16, max(3, 0.4 * len(ordered))))
    plt.imshow(heat_matrix, cmap=cmap, aspect="auto")

    plt.colorbar(label="L2 error")

    # row labels = step numbers
    plt.yticks(ticks=range(len(ordered)), labels=steps)

    # columns = bytes
    plt.xticks(ticks=range(32), labels=[str(i) for i in range(32)])

    plt.xlabel("Byte index")
    plt.ylabel("Checkpoint step")
    plt.title(f"Byte-wise L2 Heatmap for {model_name}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

# ================================================================
# === CSV writer
# ================================================================
def save_final_csv(results: dict, base: Path):
    """
    Write:
        model/<name>/log/final_eval.csv

    Columns:
        ckpt_name, mean_l2, median_l2, std_l2, byte_0..byte_31
    """
    out = base / "log" / "final_eval.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="") as f:
        writer = csv.writer(f)

        header = ["ckpt_name", "mean_l2", "median_l2", "std_l2"]
        header.extend([f"byte_{i}" for i in range(32)])
        writer.writerow(header)

        for ckpt, s in results.items():
            row = [
                ckpt,
                s["mean"],
                s["median"],
                s["std"],
            ]
            row.extend(list(map(float, s["byte_mean"])))
            writer.writerow(row)

    print(f"[FinalEval] CSV written → {out}")



# ================================================================
# === Best checkpoint selection
# ================================================================
def select_best_ckpt(base: Path) -> str:
    """
    Select best ckpt from final_eval.csv by:
        1. smallest median_l2
        2. then smallest mean_l2
        3. then smallest std_l2
        4. then smallest step (parsed from _sXXXXX)
    """
    csv_path = base / "log" / "final_eval.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing final_eval.csv at {csv_path}")

    rows = []
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ck = r["ckpt_name"]
            m = re.search(r"_s(\d+)", ck)
            if not m:
                raise ValueError(f"Invalid ckpt name (no step): {ck}")
            step = int(m.group(1))

            rows.append({
                "ckpt": ck,
                "median": float(r["median_l2"]),
                "mean": float(r["mean_l2"]),
                "std": float(r["std_l2"]),
                "step": step,
            })

    if not rows:
        raise RuntimeError("final_eval.csv has no rows.")

    rows = sorted(
        rows,
        key=lambda r: (r["mean"], r["median"], r["std"], r["step"]),
    )
    best = rows[0]
    print("[Best] Selected best checkpoint:", best["ckpt"])
    print(
        f"        median={best['median']:.6f}, "
        f"mean={best['mean']:.6f}, "
        f"std={best['std']:.6f}, "
        f"step={best['step']}"
    )
    return best["ckpt"]


# ================================================================
# === Copy best checkpoint
# ================================================================
def export_best_ckpt(base: Path, ckpt: str):
    ckpt_dir = base / "ckpts"
    best_dir = base / "best_ckpt"
    best_dir.mkdir(exist_ok=True)

    # wipe old
    for f in best_dir.glob("*"):
        f.unlink()

    # copy safetensors
    src_safe = ckpt_dir / ckpt
    shutil.copy2(src_safe, best_dir / ckpt)

    # copy full pt
    full = ckpt.replace(".safetensors", "_full.pt")
    p = ckpt_dir / full
    if p.exists():
        shutil.copy2(p, best_dir / full)

    print("[Best] Exported best checkpoint →", best_dir)


# ================================================================
# === AUTO MODEL LOADER (from config.json)
# ================================================================
def load_model_from_config(base: Path, device: torch.device) -> torch.nn.Module:
    """
    Load model according to:
        model/<name>/log/config.json

    The config must contain:
        K, Q, coord_dim, model_type, hidden, layers, dropout, ...
    """
    cfg_path = base / "log" / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json at {cfg_path}")

    with cfg_path.open("r") as f:
        cfg = json.load(f)

    model_type = cfg["model_type"].lower()

    if model_type == "mlp":
        model = thetaMLP(
            K=cfg["K"],
            coord_dim=cfg["coord_dim"],
            hidden=cfg["hidden"],
            layers=cfg["layers"],
            dropout=cfg["dropout"],
        )

    elif model_type == "transformer":
        model = thetaTransformer(
            K=cfg["K"],
            coord_dim=cfg["coord_dim"],
            hidden=cfg["hidden"],
            layers=cfg["layers"],
            nhead=cfg["nhead"],
            dropout=cfg["dropout"],
        )

    elif model_type == "cnn":
        model = thetaCNN1D(
            K=cfg["K"],
            coord_dim=cfg["coord_dim"],
            hidden=cfg["hidden"],
            cnn_layers=cfg["cnn_layers"],
            kernel_size=cfg["kernel_size"],
            dropout=cfg["dropout"],
        )

    elif model_type in ["hybrid", "cnn_transformer", "cnn+transformer"]:
        model = thetaHybridCNNTransformer(
            K=cfg["K"],
            coord_dim=cfg["coord_dim"],
            hidden=cfg["hidden"],
            cnn_layers=cfg["cnn_layers"],
            transf_layers=cfg["layers"],
            nhead=cfg["nhead"],
            dropout=cfg["dropout"],
            noise_dim=cfg["noise_dim"],          # ← REQUIRED
            kernel_size=cfg["kernel_size"],      # ← if training used this
        )


    else:
        raise ValueError(f"Unsupported model_type={cfg['model_type']}")

    return model.to(device)


# ================================================================
# === CONTROL FUNCTION (PUBLIC ENTRY)
# ================================================================
def ckpt_audit(
    model_name: str,
    big_path: str | Path = "./dataset/quick_val_big.pt",
    device: str = "cuda",
):
    device = torch.device(device)
    base = Path("./model") / model_name
    ckpt_dir = base / "ckpts"
    log_dir = base / "log"

    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

    safes = sorted(ckpt_dir.glob("*.safetensors"))
    if not safes:
        raise RuntimeError(f"No .safetensors checkpoints found in {ckpt_dir}")

    results: dict[str, dict] = {}

    for sf in safes:
        ck = sf.name
        full = ck.replace(".safetensors", "_full.pt")
        full_path = ckpt_dir / full
        if not full_path.exists():
            print(f"[WARN] Missing full snapshot for {ck}, skipping.")
            continue

        print(f"\n=== Eval {ck} ===")

        # rebuild model + load weights
        model = load_model_from_config(base, device)
        load_ckpt(model, full_path, device)

        stats = large_scale_eval(model, device, big_path)
        # Q1/Q2 intentionally disabled for now

        # raw byte array
        np.save(ckpt_dir / f"{ck}.byte_err.npy", stats["byte_mean"])

        results[ck] = stats

        print(
            f"[DONE] median={stats['median']:.6f} "
            f"mean={stats['mean']:.6f} "
            f"std={stats['std']:.6f}"
        )

    # CSV + best selection
    save_final_csv(results, base)
    best = select_best_ckpt(base)
    export_best_ckpt(base, best)

    # ===== COMBINED HEATMAP HERE =====
    heatmap_out = log_dir / "byte_heatmap.png"
    plot_all_ckpt_heatmaps(model_name, results, heatmap_out)
    print(f"[HEATMAP] Saved → {heatmap_out}")

    print("\n[AUDIT COMPLETE]")
    return best


def audit_all_models(
    model_root: str | Path = "./model",
    big_path: str | Path = "./dataset/quick_val_big.pt",
    device: str = "cuda",
):
    """
    Run ckpt_audit() for every model directory under ./model/
    A valid model directory must contain a subdirectory: model/<name>/ckpts/

    Returns:
        results: dict mapping model_name -> best_checkpoint_name
    """
    root = Path(model_root)
    if not root.exists():
        raise FileNotFoundError(f"Model root not found: {root}")

    # discover model directories
    model_dirs = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if (p / "ckpts").exists():
            model_dirs.append(p.name)

    if not model_dirs:
        raise RuntimeError(f"No models with ckpts/ found under {root}")

    print("\n=== BEGIN AUDIT FOR ALL MODELS ===\n")
    results = {}

    for name in sorted(model_dirs):
        print(f"\n===============================")
        print(f"=== Auditing model: {name}")
        print(f"===============================")

        try:
            best = ckpt_audit(
                model_name=name,
                big_path=big_path,
                device=device
            )
            results[name] = best
        except Exception as e:
            print(f"[ERROR] Failed auditing model {name}: {e}")
            results[name] = None

    print("\n=== ALL MODEL AUDITS COMPLETE ===")
    for name, best in results.items():
        print(f"{name:30} -> {best}")

    # Global combined heatmap + CSV
    generate_global_best_heatmap(results)

    return results



def generate_global_best_heatmap(results: dict, model_root="./model", out_dir="./log"):
    """
    results: dict model_name -> best_ckpt_name (from audit_all_models)

    Produces:
        ./log/best_ckpt_heatmap.png
        ./log/best_ckpt_summary.csv

    Q1/Q2 REMOVED.
    """

    root = Path(model_root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []         # list of dicts with metrics
    matrices = []     # heatmap rows
    labels = []       # row labels

    for model_name, best_ckpt in results.items():
        if best_ckpt is None:
            continue

        model_dir = root / model_name
        ckpt_dir = model_dir / "ckpts"

        # load byte_mean vector
        npy_path = ckpt_dir / f"{best_ckpt}.byte_err.npy"
        if not npy_path.exists():
            print(f"[WARN] Missing byte_err.npy for {model_name}, skipping.")
            continue

        byte_mean = np.load(npy_path)

        # read performance data from final_eval.csv
        csv_path = model_dir / "log" / "final_eval.csv"
        best_row = None
        with csv_path.open("r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["ckpt_name"] == best_ckpt:
                    best_row = r
                    break

        if best_row is None:
            raise RuntimeError(f"Best ckpt {best_ckpt} not found in {csv_path}")

        # parse step
        m = re.search(r"_s(\d+)", best_ckpt)
        step = int(m.group(1)) if m else -1

        # row label = full model folder + step number
        labels.append(f"{model_name} (s{step})")
        matrices.append(byte_mean.copy())

        # store stats WITHOUT Q1/Q2
        rows.append({
            "model": model_name,
            "step": step,
            "mean": float(best_row["mean_l2"]),
            "median": float(best_row["median_l2"]),
            "std": float(best_row["std_l2"]),
            "byte_mean": byte_mean,
        })

    # ======================================================
    # SAVE CSV (Q1/Q2 removed)
    # ======================================================
    csv_out = out / "best_ckpt_summary.csv"
    with csv_out.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["model", "step", "mean", "median", "std"]
        header.extend([f"byte_{i}" for i in range(32)])
        writer.writerow(header)

        for r in rows:
            writer.writerow([
                r["model"],
                r["step"],
                r["mean"],
                r["median"],
                r["std"],
                *list(map(float, r["byte_mean"])),
            ])

    print(f"[GLOBAL] Summary CSV saved → {csv_out}")

    # ======================================================
    # GLOBAL HEATMAP
    # ======================================================
    if not matrices:
        print("[GLOBAL] No valid models, skipping heatmap.")
        return

    heat = np.vstack(matrices)

    cmap = LinearSegmentedColormap.from_list(
        "errmap", ["white", "red", "purple"]
    )

    plt.figure(figsize=(18, max(3, 0.5 * len(matrices))))
    plt.imshow(heat, cmap=cmap, aspect="auto")
    plt.colorbar(label="L2 error")

    plt.yticks(ticks=range(len(labels)), labels=labels)
    plt.xticks(ticks=range(32), labels=[str(i) for i in range(32)])

    plt.xlabel("Byte index")
    plt.ylabel("Model (architecture_size_date_time)")
    plt.title("Best Checkpoint Byte-wise L2 Comparison Across All Models")

    heatmap_out = out / "best_ckpt_heatmap.png"
    plt.savefig(heatmap_out, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[GLOBAL] Global best heatmap saved → {heatmap_out}")

# ================================================================
# === CLI ENTRY
# ================================================================
# if __name__ == "__main__":
#     import sys

#     if len(sys.argv) != 2:
#         print("Usage: python post_train_eval.py <model_name>")
#         sys.exit(1)

#     ckpt_audit(sys.argv[1])

# if __name__ == "__main__":
#     audit_all_models()