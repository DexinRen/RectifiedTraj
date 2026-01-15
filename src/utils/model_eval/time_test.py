import time
import csv
import os
import json
import numpy as np
import torch
from pathlib import Path
from safetensors.torch import load_file
from theta_model import build_theta_model


# ============================================================
# Constants
# ============================================================
DEVICE = torch.device("cuda")
NPY_PATH = "./dataset/time_test/source_list.npy"
LOG_PATH = "./log/time_test.csv"
BATCH_SIZE = 64

# ============================================================
# Load first 32 chunks
# ============================================================
def load_batch(npy_path: str, batch_size: int = BATCH_SIZE):
    arr = np.load(npy_path)
    if arr.shape[0] < batch_size:
        raise ValueError(f"source_list.npy has only {arr.shape[0]} samples, need 32.")
    arr = arr[:batch_size]                          # (32, K, 2)
    Xt = torch.tensor(arr, dtype=torch.float32, device=DEVICE)
    return Xt


# ============================================================
# Build model, load checkpoint, extract model_name fields
# ============================================================
def test_init(config_path: str, ckpt_path: str):
    # Load config
    with open(config_path, "r") as f:
        cfg = json.load(f)

    runtime = {"config": cfg}

    # Build model
    model = build_theta_model(runtime).to(DEVICE)

    # Load checkpoint (pt or safetensors)
    blob = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = blob["model_state_dict"]
    model.load_state_dict(state)

    # Extract model_arch + model_size
    path = Path(ckpt_path)
    model_dir = path.parts[-3]     # e.g. hybrid_10M_20251205_011946
    parts = model_dir.split("_")

    model_arch = parts[0]
    model_size = parts[1]

    return model, model_arch, model_size, cfg


# ============================================================
# RF batched denoise (timed) — FIXED t SHAPE
# ============================================================
@torch.no_grad()
def run_test(model, delta_t: float, Xt: torch.Tensor):
    B = Xt.shape[0]

    # t must be shape (B,1) — REQUIRED FOR SINUSOIDAL EMBEDDING
    t = torch.ones((B, 1), device=DEVICE)

    # Warm-up
    _ = model(Xt, t)
    torch.cuda.synchronize()

    num_steps = int(1.0 / delta_t)

    # Timed loop
    start = time.time()
    for _ in range(num_steps):
        V = model(Xt, t)                # OK: t is (B,1)

        Xt = Xt - delta_t * V           # update Xt

        # update t while keeping shape (B,1)
        t = t - delta_t
        t = torch.clamp(t, min=0.0)
    torch.cuda.synchronize()
    end = time.time()

    return Xt, end - start, num_steps


# ============================================================
# CSV logging
# ============================================================
def append_time_log(
    model_arch, model_size, delta_t, num_steps,
    elapsed, batch_size, K, logfile
):
    Path(logfile).parent.mkdir(parents=True, exist_ok=True)

    file_exists = Path(logfile).exists()
    with open(logfile, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "model_arch",
                "model_size",
                "delta_t",
                "num_steps",
                "batch_size",
                "K",
                "time_total_sec",
                "time_per_step_sec",
                "throughput_chunks_per_sec",
            ])

        writer.writerow([
            model_arch,
            model_size,
            delta_t,
            num_steps,
            batch_size,
            K,
            f"{elapsed:.6f}",
            f"{elapsed/num_steps:.6f}",
            f"{batch_size/elapsed:.6f}",
        ])



# ============================================================
# Exposed entry point
# ============================================================
def run_time_test(config_path: str, ckpt_path: str, delta_t: float):

    # init model + metadata
    model, model_arch, model_size, cfg = test_init(config_path, ckpt_path)

    # Load exactly 32 chunks
    Xt0 = load_batch(NPY_PATH, batch_size=BATCH_SIZE)

    # Run timed RF test
    Xt_clean, elapsed, num_steps = run_test(model, delta_t, Xt0)

    # Report
    print("================================================")
    print("BATCHED RF TIME TEST")
    print("------------------------------------------------")
    print(f"Model arch        : {model_arch}")
    print(f"Model size        : {model_size}")
    print(f"Batch size        : {BATCH_SIZE}")
    print(f"K                 : {Xt0.shape[1]}")
    print(f"delta_t           : {delta_t}")
    print(f"Steps             : {num_steps}")
    print(f"Total time        : {elapsed:.6f} sec")
    print(f"Time per step     : {elapsed/num_steps:.6f} sec")
    print(f"Throughput        : {Xt0.shape[0] / elapsed:.2f} chunks/sec")
    print("================================================")

    # save record
    append_time_log(
        model_arch=model_arch,
        model_size=model_size,
        delta_t=delta_t,
        num_steps=num_steps,
        elapsed=elapsed,
        batch_size=Xt0.shape[0],
        K=Xt0.shape[1],
        logfile=LOG_PATH,
    )


    return Xt_clean, elapsed

def run_all_time_tests(
    model_root="./model",
    delta_list=(1.0, 0.2, 0.04, 0.01),
    config_name="config.json"
):
    """
    Scan ./model/<model_name>/best_ckpt/ and run time tests on each model.
    For each model, run delta_t ∈ {1.0, 0.2, 0.04, 0.01}.

    Requirements:
        - config at: ./model/<model_name>/log/config.json
        - checkpoint inside: ./model/<model_name>/best_ckpt/
    """

    model_root = Path(model_root)
    if not model_root.exists():
        raise FileNotFoundError(f"Model root not found: {model_root}")

    # Discover all model folders
    model_dirs = [p for p in model_root.iterdir() if p.is_dir()]
    if not model_dirs:
        print("[ERROR] No subdirectories found under ./model/")
        return

    print("===============================================")
    print(" BEGIN TIME EVALUATION FOR ALL MODELS ")
    print("===============================================")

    for model_dir in sorted(model_dirs):
        name = model_dir.name
        best_dir = model_dir / "best_ckpt"
        log_dir  = model_dir / "log"

        # -------------------------------------------------
        # Validate directory structure
        # -------------------------------------------------
        if not best_dir.exists():
            print(f"[SKIP] {name}: no best_ckpt/ directory")
            continue

        cfg_path = log_dir / "config.json"
        if not cfg_path.exists():
            print(f"[SKIP] {name}: missing config.json at {cfg_path}")
            continue

        # -------------------------------------------------
        # Find checkpoint
        # -------------------------------------------------
        ckpts = list(best_dir.glob("*.pt")) + list(best_dir.glob("*.safetensors"))
        if not ckpts:
            print(f"[SKIP] {name}: no .pt or .safetensors in best_ckpt/")
            continue

        # Prefer .pt → easier to debug corruption issues
        ckpts_sorted = sorted(ckpts, key=lambda p: p.suffix)
        ckpt = ckpts_sorted[0]

        print("\n-----------------------------------------------")
        print(f"MODEL: {name}")
        print(f"Config: {cfg_path}")
        print(f"Checkpoint: {ckpt}")
        print("-----------------------------------------------")

        # -------------------------------------------------
        # Run multiple delta_t evaluations
        # -------------------------------------------------
        for dt in delta_list:
            print(f"\n>>> Running delta_t = {dt}")
            try:
                run_time_test(
                    config_path=str(cfg_path),
                    ckpt_path=str(ckpt),
                    delta_t=dt
                )
            except Exception as e:
                print(f"[ERROR] Failed at delta_t={dt} for model {name}: {e}")
                continue

    print("\n===============================================")
    print("  ALL MODELS TIME TEST COMPLETE  ")
    print("===============================================")


# ============================================================
# DIRECT CALL
# ============================================================
# run_time_test(
#     config_path="./model/hybrid_10M_20251205_011946/log/config.json",
#     ckpt_path="./model/hybrid_10M_20251205_011946/best_ckpt/ckpt_e24_s860000_full.pt",
#     delta_t=1
# )

run_all_time_tests()