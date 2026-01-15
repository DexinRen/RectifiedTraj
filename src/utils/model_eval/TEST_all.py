# ================================================================
# CHUNK-WISE MODEL BENCHMARK SCRIPT
# ================================================================
# This script tests ENU→ENU denoising accuracy using EncoderDecoder.
#
# Benchmarks computed:
#   1. FULL CHUNK ERROR      (no buckle removal)
#   2. MIDDLE-ONLY ERROR     (remove Q1*8 head pts, Q2*8 tail pts)
#
# Outputs:
#   - console summary
#   - appends row to TEST_chunkwise.csv
#
# Uses ONLY the ./dataset/processed/test/ .pt files.
# Loads exactly 600,000 samples (200k from 3 files).
# ================================================================

import os
import csv
import glob
import torch
import numpy as np
from pathlib import Path

from encoder_decoder import EncoderDecoder
from theta_train import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================
# Compute X0 and X1 in ENU space
# ================================================================
def reconstruct_X0_X1(Xt, V, t):
    """
    Xt, V : (B,K,2)
    t     : (B,)
    Returns:
        X0 : (B,K,2)
        X1 : (B,K,2)
    """
    t = t.view(-1, 1, 1)
    X0 = Xt - V * t
    X1 = Xt + V * (1.0 - t)
    return X0, X1


# ================================================================
# Load from test fold
# ================================================================
def load_all_test_samples(runtime, test_dir):
    """
    Load ALL samples from ALL test .pt files.
    Each .pt file contributes up to 37,000 samples (because of DataLoader.max_steps).
    Returns:
        X0 : (N,K,2) clean ENU
        X1 : (N,K,2) noisy ENU
    """
    test_files = sorted(glob.glob(f"{test_dir}/*.pt"))
    assert test_files, f"No .pt files found under {test_dir}"

    X0_list = []
    X1_list = []

    for i, pt_file in enumerate(test_files):
        print(f"[INFO] Loading test file #{i+1}: {pt_file}")

        # clone runtime
        rt = dict(runtime)
        rt["config"] = dict(runtime["config"])
        rt["config"]["train_dir"] = test_dir

        dl = DataLoader(rt)
        dl.set(i)                    # load i-th file → always ~37k samples

        Xt = dl.X_t[:, :, :2]    # force ENU (E,N)
        V  = dl.V[:, :, :2]      # force ENU velocity (E_dot, N_dot)
        t  = dl.t

        X0, X1 = reconstruct_X0_X1(Xt, V, t)
        X0_list.append(X0.cpu())
        X1_list.append(X1.cpu())

    # Combine all files
    X0 = torch.cat(X0_list, dim=0)
    X1 = torch.cat(X1_list, dim=0)

    print(f"[INFO] Total test samples loaded: {X0.shape[0]}")
    return X0, X1

# ================================================================
# Append a benchmark row into TEST_chunkwise.csv
# ================================================================
def append_result_csv(csv_path,
                      model_arch, steps, Q1_bytes, Q2_bytes,
                      mean_full, median_full, std_full,
                      mean_mid,  median_mid,  std_mid):

    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not exists:
            writer.writerow([
                "model_arch",
                "steps",
                "Q1_bytes",
                "Q2_bytes",
                "err_mean_full",
                "err_median_full",
                "err_std_full",
                "err_mean_mid",
                "err_median_mid",
                "err_std_mid",
            ])

        writer.writerow([
            model_arch,
            steps,
            Q1_bytes,
            Q2_bytes,
            f"{mean_full:.9f}",
            f"{median_full:.9f}",
            f"{std_full:.9f}",
            f"{mean_mid:.9f}",
            f"{median_mid:.9f}",
            f"{std_mid:.9f}",
        ])


# ================================================================
# ENU-only model evaluation (full vs middle)
# ================================================================
def evaluate_enu(ed: EncoderDecoder, X0, X1):
    """
    ed : EncoderDecoder instance
    X0 : (N,K,2) clean ENU
    X1 : (N,K,2) noisy ENU input
    """
    N, K, _ = X0.shape
    N = 5000
    Q1p = ed.Q1         # buckle head, in points
    Q2p = ed.Q2         # buckle tail, in points

    errs_full = []
    errs_mid  = []

    print(f"[INFO] Evaluating {N} chunks (K={K})")
    print(f"[INFO] Buckle points: head={Q1p}, tail={Q2p}")

    for i in range(N):
        inp = X1[i].numpy()    # (K,2)
        gt  = X0[i].numpy()

        pred = ed.denoise_chunk_enu(inp)

        # -------- FULL error (no removal) -------------
        diff_full = pred - gt
        l2_full = np.sqrt((diff_full * diff_full).sum(axis=-1))  # (K,)
        errs_full.append(l2_full)

        # -------- MIDDLE error (remove buckle) --------
        if Q2p > 0:
            pred_mid = pred[Q1p:-Q2p]
            gt_mid   = gt[Q1p:-Q2p]
        else:
            pred_mid = pred[Q1p:]
            gt_mid   = gt[Q1p:]

        diff_mid = pred_mid - gt_mid
        l2_mid = np.sqrt((diff_mid * diff_mid).sum(axis=-1))     # (K_mid,)
        errs_mid.append(l2_mid)

        if i % 1000 == 0:
            print(f"[INFO] Processed: {i}/{N}")

    errs_full = np.stack(errs_full, axis=0)
    errs_mid  = np.stack(errs_mid,  axis=0)

    mean_full,  median_full,  std_full  = errs_full.mean(), np.median(errs_full), errs_full.std()
    mean_mid,   median_mid,   std_mid   = errs_mid.mean(),  np.median(errs_mid),  errs_mid.std()

    print("============== RF ENU Chunk-wise Report ==============")
    print(f"[FULL] Mean={mean_full:.6f}, Median={median_full:.6f}, Std={std_full:.6f}")
    print(f"[MIDL] Mean={mean_mid:.6f},  Median={median_mid:.6f},  Std={std_mid:.6f}")
    print("======================================================")

    return (
        errs_full, errs_mid,
        (mean_full, median_full, std_full),
        (mean_mid,  median_mid,  std_mid),
    )


# ================================================================
# RUN ALL MODELS WRAPPER (FUNCTION VERSION)
# ================================================================
def run_all_models(model_root="./model"):
    """
    Scan ./model/<model_name>/best_ckpt/
    If exists:
        pick checkpoint
        run evaluation
        append to TEST_chunkwise.csv
    Else:
        skip
    """

    model_root = Path(model_root)
    model_dirs = sorted([d for d in model_root.iterdir() if d.is_dir()])

    if not model_dirs:
        print("[ERROR] No model directories found under ./model/")
        return

    print("[INFO] Pre-loading 600k ENU test chunks (one time only)...")

    runtime_template = {
        "device": DEVICE,
        "config": {
            "batch_size": 64,
            "train_dir": "./dataset/processed/test",
            "K": 256,
        }
    }

    X0, X1 = load_all_test_samples(runtime_template, "./dataset/processed/test")


    # ------------------------------
    # helper: pick ckpt
    # ------------------------------
    def pick_ckpt(ckpt_dir: Path):
        safes = sorted(ckpt_dir.glob("*.safetensors"))
        pts   = sorted(ckpt_dir.glob("*.pt"))
        if safes:
            return safes[0]
        if pts:
            return pts[0]
        return None

    # ------------------------------
    # iterate over models
    # ------------------------------
    for model_dir in model_dirs:
        model_name = model_dir.name
        best_ckpt = model_dir / "best_ckpt"

        if not best_ckpt.exists() or not best_ckpt.is_dir():
            print(f"[SKIP] {model_name}: no best_ckpt/ folder")
            continue

        ckpt = pick_ckpt(best_ckpt)
        if ckpt is None:
            print(f"[SKIP] {model_name}: best_ckpt/ contains no checkpoints")
            continue

        print(f"\n[INFO] Evaluating model: {model_name}")
        print(f"[INFO] Using checkpoint: {ckpt}")

        # load model
        ED = EncoderDecoder(str(ckpt))

        # run eval
        (
            errs_full, errs_mid,
            stats_full, stats_mid
        ) = evaluate_enu(ED, X0, X1)

        mean_full, median_full, std_full = stats_full
        mean_mid, median_mid, std_mid    = stats_mid

        append_result_csv(
            "TEST_chunkwise.csv",
            model_arch=ED.cfg.get("model_type", model_name),
            steps=X0.shape[0],
            Q1_bytes=ED.Q1_bytes,
            Q2_bytes=ED.Q2_bytes,
            mean_full=mean_full,
            median_full=median_full,
            std_full=std_full,
            mean_mid=mean_mid,
            median_mid=median_mid,
            std_mid=std_mid,
        )

        print(f"[INFO] Done model: {model_name}")
        print("[INFO] Row appended to TEST_chunkwise.csv")

    print("\n[INFO] All eligible models processed.")

# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    # Uncomment ONE of the following two flows:

    # --------------------------------------------------
    # (1) Evaluate a single checkpoint
    # --------------------------------------------------
    # ckpt = "CKPT PATH"

    # runtime = {
    #     "device": DEVICE,
    #     "config": {
    #         "batch_size": 64,
    #         "train_dir": "./dataset/processed/test",
    #         "K": 256,
    #     }
    # }

    # print("[INFO] Loading EncoderDecoder...")
    # ED = EncoderDecoder(ckpt)

    # print("[INFO] Loading 600k ENU samples from test fold...")
    # X0, X1 = load_600k_samples(runtime, "./dataset/processed/test")

    # print("[INFO] Running ENU-only RF evaluation...")
    # (
    #     errs_full, errs_mid,
    #     stats_full, stats_mid
    # ) = evaluate_enu(ED, X0, X1)

    # # Unpack stats
    # mean_full, median_full, std_full = stats_full
    # mean_mid,  median_mid,  std_mid  = stats_mid

    # model_arch = ED.cfg.get("model_type", "unknown")
    # steps      = X0.shape[0]
    # Q1_bytes   = ED.Q1_bytes
    # Q2_bytes   = ED.Q2_bytes

    # append_result_csv(
    #     "TEST_chunkwise.csv",
    #     model_arch, steps, Q1_bytes, Q2_bytes,
    #     mean_full, median_full, std_full,
    #     mean_mid,  median_mid,  std_mid
    # )

    # print("[INFO] Done. Row appended to TEST_chunkwise.csv")

    # --------------------------------------------------
    # (2) Evaluate all models under ./model/
    # --------------------------------------------------
    run_all_models()
