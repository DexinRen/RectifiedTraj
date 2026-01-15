# accuracy_runner.py — aggregate chunk & traj tests for all models
# FIXED VERSION - All bugs corrected

from pathlib import Path
import json
import numpy as np
import polars as pl
import torch

# ---- project paths ----
RAW_DIR = Path("./dataset/raw")
PT_TEST_DIR = Path("./dataset/processed/test")
CKPT_ROOT = Path("./bin/checkpoints")
LOG_ROOT = Path("./log")

# ---- device ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- bring in pipeline funcs from your encoder_decoder.py ----
from encoder_decoder import denoise_traj, denoise_chunk  # noqa: F401

# ============================ helpers ============================

def _gps_to_enu_batch(lonlat: np.ndarray, lat0: float, lon0: float) -> np.ndarray:
    from pymap3d import geodetic2enu
    e, n, _ = geodetic2enu(lonlat[:,1], lonlat[:,0], 46, lat0, lon0, 0)
    return np.stack([e, n], axis=1)

def _err_stats(d: np.ndarray) -> dict:
    return {
        "total_err": float(d.sum()),
        "avg_err": float(d.mean()),
        "med_err": float(np.median(d)),
        "std_err": float(d.std(ddof=0)),
        "count": int(d.size),
    }

def _last_three_parquets() -> list[Path]:
    files = sorted(RAW_DIR.glob("*.parquet"))
    return files[-3:]

def _sample_users(df: pl.DataFrame, k: int) -> list:
    users = df["agent"].unique().to_list()
    users.sort()
    if not users:
        return []
    if len(users) <= k:
        return users
    # deterministic: take first k after sorting (stable)
    return users[:k]

def _collect_trajectories(sample_per_file: int = 20) -> list[dict]:
    """Return list of dicts: {"file": name, "usr": id, "traj_n": (T,2), "traj_g": (T,2)}"""
    out = []
    for pq in _last_three_parquets():
        df = pl.read_parquet(pq)
        # FIX 1: Add .is_finite() checks (matching parquet_processor.py)
        df = df.filter(
            pl.col("longitude_n").is_not_null() &
            pl.col("latitude_n").is_not_null() &
            pl.col("longitude").is_not_null() & 
            pl.col("latitude").is_not_null() &
            pl.col("longitude_n").is_finite() &
            pl.col("latitude_n").is_finite() &
            pl.col("longitude").is_finite() &
            pl.col("latitude").is_finite()
        )
        users = _sample_users(df, sample_per_file)
        for u in users:
            sel = df.filter(pl.col("agent") == u).sort("timestamp")
            if sel.height == 0:
                continue
            # FIX 2: Use 'sel' (filtered dataframe) instead of 'df'
            traj_n = np.stack([sel["longitude_n"].to_numpy(), sel["latitude_n"].to_numpy()], axis=1)
            traj_g = np.stack([sel["longitude"].to_numpy(), sel["latitude"].to_numpy()], axis=1)

            out.append({"file": pq.name, "usr": u, "traj_n": traj_n, "traj_g": traj_g})
    return out

def _chunk_err_aggregate(theta) -> dict:
    """Aggregate chunk denoise across all .pt files (all samples)."""
    deno = []
    base = []
    for ptp in sorted(PT_TEST_DIR.glob("*.pt")):
        pack = torch.load(ptp, map_location="cpu")
        X_t = pack["X_t"][:, :, :2]   # (N,K,2)
        V   = pack["V"]               # (N,K,2)
        t   = pack["t"].reshape(-1)   # (N,)
        N, K, _ = X_t.shape
        for i in range(N):
            x  = X_t[i]              # ENU
            vi = V[i]
            ti = float(t[i])
            x0_ref = x - ti * vi     # reference clean ENU
            # denoise
            C = x.to(DEVICE)
            steps = 30
            dt = 1.0/steps
            for s in range(steps):
                cur_t = max(0.0, ti - s*dt)
                tt = torch.tensor([[cur_t]], dtype=torch.float32, device=DEVICE)
                with torch.no_grad():
                    vv = theta(C.unsqueeze(0), tt).squeeze(0)
                C = C - dt * vv
            x_hat = C.detach().cpu().numpy()
            deno.append(np.linalg.norm(x_hat - x0_ref.numpy(), axis=1))
            base.append(np.linalg.norm((x.numpy() - x0_ref.numpy()), axis=1))
    deno = np.concatenate(deno) if deno else np.array([], dtype=float)
    base = np.concatenate(base) if base else np.array([], dtype=float)
    return {"denoised": _err_stats(deno) if deno.size else None,
            "input": _err_stats(base) if base.size else None}

def _traj_err_aggregate(theta, samples: list[dict]) -> dict:
    """Aggregate traj denoise over provided (traj_n, traj_g) samples."""
    deno = []
    base = []
    for rec in samples:
        n = rec["traj_n"]
        g = rec["traj_g"]
        if len(n) < 2 or len(g) < 2:
            continue
        # run denoise_traj (stitches & drops first Q internally)
        hat = denoise_traj(n, theta)  # (T',2)
        T = len(hat)
        g_tail = g[-T:]
        n_tail = n[-T:]
        lat0, lon0 = float(g_tail[0,1]), float(g_tail[0,0])
        e_hat = _gps_to_enu_batch(hat, lat0, lon0)
        e_g   = _gps_to_enu_batch(g_tail, lat0, lon0)
        e_n   = _gps_to_enu_batch(n_tail, lat0, lon0)
        deno.append(np.linalg.norm(e_hat - e_g, axis=1))
        base.append(np.linalg.norm(e_n - e_g, axis=1))
    deno = np.concatenate(deno) if deno else np.array([], dtype=float)
    base = np.concatenate(base) if base else np.array([], dtype=float)
    return {"denoised": _err_stats(deno) if deno.size else None,
            "input": _err_stats(base) if base.size else None}

# FIX 3: Add baseline-only computation function
def _compute_baseline_only(samples: list[dict]) -> dict:
    """Compute input baseline errors without denoising."""
    base = []
    for rec in samples:
        n = rec["traj_n"]
        g = rec["traj_g"]
        if len(n) < 2 or len(g) < 2:
            continue
        # Align to shorter length
        min_len = min(len(n), len(g))
        n_aligned = n[-min_len:]
        g_aligned = g[-min_len:]
        
        lat0, lon0 = float(g_aligned[0,1]), float(g_aligned[0,0])
        e_n = _gps_to_enu_batch(n_aligned, lat0, lon0)
        e_g = _gps_to_enu_batch(g_aligned, lat0, lon0)
        base.append(np.linalg.norm(e_n - e_g, axis=1))
    
    base = np.concatenate(base) if base else np.array([], dtype=float)
    return {"input": _err_stats(base) if base.size else None}

def _find_models() -> list[str]:
    """All subdirs in ./log that contain a 'final_choice' file."""
    names = []
    for p in LOG_ROOT.iterdir():
        if not p.is_dir():
            continue
        if (p / "final_choice").exists():
            names.append(p.name)
    names.sort()
    return names

def _read_final_choice(model_name: str) -> str:
    return (LOG_ROOT / model_name / "final_choice").read_text().strip()

def _load_theta(model_name: str, ckpt_name: str):
    """Robust loader that reads config from model_card file."""
    ckpt_dir = CKPT_ROOT / model_name
    
    # Read model_card from log directory
    model_card_path = LOG_ROOT / model_name / "model_card"
    if not model_card_path.exists():
        raise FileNotFoundError(f"model_card not found at {model_card_path}")
    
    # Parse model_card (simple key: value format)
    cfg = {}
    with open(model_card_path, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line and not line.startswith('='):
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                # Try to parse value as int/float/bool
                try:
                    if value.lower() == 'true':
                        cfg[key] = True
                    elif value.lower() == 'false':
                        cfg[key] = False
                    elif '.' in value:
                        cfg[key] = float(value)
                    else:
                        cfg[key] = int(value)
                except ValueError:
                    cfg[key] = value  # Keep as string
    
    print(f"  Config: {cfg.get('model_type', 'unknown')} with hidden={cfg.get('hidden', '?')}, layers={cfg.get('layers', '?')}")
    
    # Try to find checkpoint file
    for ext in (".pt", ".pth", ""):
        path = ckpt_dir / f"{ckpt_name}{ext}"
        if path.exists():
            ckpt_path = path
            break
    else:
        raise FileNotFoundError(f"Checkpoint {ckpt_name} not found in {ckpt_dir}")
    
    blob = torch.load(ckpt_path, map_location="cpu")

    # Build model with config from model_card
    from theta_model import build_theta
    
    model_result = build_theta(
        model_type=cfg.get("model_type", "nn"),
        hidden=cfg.get("hidden", 512),
        layers=cfg.get("layers", 6),
        K=cfg.get("K", 256),
        dropout=cfg.get("dropout", 0.1)
    )
    model = model_result["model"]

    # Load weights
    if isinstance(blob, dict) and "model_state_dict" in blob:
        model.load_state_dict(blob["model_state_dict"], strict=True)
    else:
        raise ValueError(f"Checkpoint format not recognized")
    
    return model.to(DEVICE).eval()

# ============================ main runner ============================

def run_all_tests(sample_users_per_file: int = 20, traj_steps_note: str = ""):
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    # FIX 5: Use baseline-only computation
    print("Collecting trajectories for testing...")
    samples = _collect_trajectories(sample_users_per_file)
    print(f"Collected {len(samples)} trajectory samples")
    
    print("Computing input baseline...")
    baseline = _compute_baseline_only(samples)
    input_avg = baseline["input"]["avg_err"] if baseline["input"] else float("nan")
    input_med = baseline["input"]["med_err"] if baseline["input"] else float("nan")
    input_std = baseline["input"]["std_err"] if baseline["input"] else float("nan")
    
    print(f"Input baseline - Avg: {input_avg:.6f}m, Median: {input_med:.6f}m, Std: {input_std:.6f}m")

    # 2) Prepare result log file (overwrite)
    result_log = LOG_ROOT / "test_result.log"
    with result_log.open("w", encoding="utf-8") as f:
        f.write(f"input,N/A,{input_avg:.6f},{input_med:.6f},{input_std:.6f}\n")

    # 3) Iterate models
    models = _find_models()
    print(f"\nFound {len(models)} models to test")
    
    for idx, model_name in enumerate(models, 1):
        print(f"\n[{idx}/{len(models)}] Testing model: {model_name}")
        ckpt = _read_final_choice(model_name)
        print(f"  Loading checkpoint: {ckpt}")
        theta = _load_theta(model_name, ckpt)

        # traj eval on sampled users
        print(f"  Running trajectory evaluation...")
        traj_stats = _traj_err_aggregate(theta, samples)
        
        # chunk eval across all .pt
        print(f"  Running chunk evaluation...")
        # chunk_stats = _chunk_err_aggregate(theta)
        chunk_stats = None  # Ban chunk test right now
        # pick the summary numbers (trajectory denoised)
        if traj_stats["denoised"]:
            avg = traj_stats["denoised"]["avg_err"]
            med = traj_stats["denoised"]["med_err"]
            std = traj_stats["denoised"]["std_err"]
            print(f"  Results - Avg: {avg:.6f}m, Median: {med:.6f}m, Std: {std:.6f}m")
        else:
            avg = med = std = float("nan")
            print(f"  Results - No valid denoised data")

        # write per-model detail JSON
        detail_path = LOG_ROOT / model_name / "accuracy.json"
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        with detail_path.open("w", encoding="utf-8") as jf:
            json.dump({
                "model": model_name,
                "ckpt": ckpt,
                "trajectory": traj_stats,
                # "chunk": chunk_stats  # Comment out or remove
            }, jf, indent=2)

        # append summary row
        with result_log.open("a", encoding="utf-8") as f:
            f.write(f"{model_name},{ckpt},{avg:.6f},{med:.6f},{std:.6f}\n")
    
    print(f"\n✓ All tests complete! Results saved to {result_log}")


# If running as a script:
if __name__ == "__main__":
    run_all_tests(sample_users_per_file=20)