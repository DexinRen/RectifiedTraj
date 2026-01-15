# encoder_decoder.py  (GPU, protocol-preserving, includes data_extract; no inner defs)
# FIXED VERSION - Column name bug corrected

import numpy as np
import torch
from pathlib import Path
import polars as pl
from pymap3d import geodetic2enu, enu2geodetic

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RAW_DIR = Path("./dataset/raw")
PROC_TEST_DIR = Path("./dataset/processed/test")


# -------------------- helpers (top-level, no inner defs) --------------------

def ENUtoGPS(X_enu, z):
    e_n = X_enu.detach().cpu().numpy()
    lat0, lon0 = z
    lon_list, lat_list = [], []
    for e, n in e_n:
        lat, lon, _ = enu2geodetic(e, n, 46, lat0, lon0, 0)
        lon_list.append(lon)
        lat_list.append(lat)
    return np.stack([lon_list, lat_list], axis=1)

def merge(traj_list, part):
    traj_list.extend(part.tolist())
    return traj_list

def _sorted_test_parquets():
    files = sorted(RAW_DIR.glob("*.parquet"))
    return files[-3:]

def _sorted_test_pts():
    return sorted(PROC_TEST_DIR.glob("*.pt"))

def _next_in_list(items, last):
    if not items: return None
    if last is None: return items[0]
    try:
        i = items.index(last)
        return items[i+1] if i+1 < len(items) else None
    except ValueError:
        return items[0]

def _next_user_after(sorted_users, last_usr):
    if not sorted_users: return None
    if last_usr is None: return sorted_users[0]
    for u in sorted_users:
        if u > last_usr:
            return u
    return None

def load_pt_file(path, map_location="cpu"):
    return torch.load(path, map_location=map_location)

def chunk_from_pt(path, idx=0, map_location="cpu"):
    pack = load_pt_file(path, map_location)
    X = pack["X_t"][idx, :, :2].clone()   # (K,2) ENU coords only
    t = float(pack["t"][idx, 0])
    V = pack["V"][idx].clone()
    return X, t, V


# -------------------- data extractor (inside this file) ---------------------

def data_extract(type, info: dict):
    # info: {"last_file": str|None, "last_usr": int|None}
    last_file = info.get("last_file")
    last_usr = info.get("last_usr")

    if type == "traj":
        test_files = _sorted_test_parquets()
        file_names = [p.name for p in test_files]
        cur_name = _next_in_list(file_names, last_file)
        if cur_name is None:
            return None

        f = test_files[file_names.index(cur_name)]
        df = pl.read_parquet(f)

        users = df["agent"].unique().to_list()
        users.sort()

        usr = _next_user_after(users, last_usr)
        if usr is None:
            next_name = _next_in_list(file_names, cur_name)
            if next_name is None:
                return None
            f = test_files[file_names.index(next_name)]
            df = pl.read_parquet(f)
            users = df["agent"].unique().to_list()
            users.sort()
            usr = _next_user_after(users, None)
            if usr is None:
                return None
            cur_name = next_name

        sel = df.filter(
            (pl.col("agent") == usr) &
            pl.col("longitude_n").is_not_null() &
            pl.col("latitude_n").is_not_null() &
            pl.col("longitude_n").is_finite() &
            pl.col("latitude_n").is_finite()
        ).sort("timestamp")

        traj = np.stack(
            [sel["longitude_n"].to_numpy(), sel["latitude_n"].to_numpy()],
            axis=1
        )
        return {"type": "traj", "traj": traj, "file": cur_name, "usr": usr}

    if type == "chunk":
        pts = _sorted_test_pts()
        names = [p.name for p in pts]
        cur_name = _next_in_list(names, last_file)
        if cur_name is None:
            return None
        fpath = pts[names.index(cur_name)]
        return {"type": "chunk", "path": str(fpath), "file": cur_name, "loader": load_pt_file}

    return None


# -------------------- encoder / decoder / pipelines -------------------------

def chunk_dicer(traj,
                idx=None,
                buckle=None,
                conf=None):
    K, Q = conf["K"], conf["Q"]
    idx = 0 if idx is None else idx
    stride = (K - Q)
    start = idx * stride

    if start + K > len(traj):
        end = len(traj)  # REAL end (not padded)
        piece = traj[start:end]
        real_len = len(piece)  # Track actual data length
        if len(piece) < K:
            pad = np.repeat(piece[-1][None, :], K - len(piece), axis=0)
            piece = np.concatenate([piece, pad], axis=0)
    else:
        end = start + K
        piece = traj[start:end]
        real_len = K  # Full chunk

    if idx == 0 and Q > 0:
        piece[:Q] = piece[0]

    lon0, lat0 = piece[0, 0], piece[0, 1]
    e, n, _ = geodetic2enu(piece[:, 1], piece[:, 0], 46, lat0, lon0, 0)
    Xt = torch.tensor(np.stack([e, n], axis=1), dtype=torch.float32, device=DEVICE)

    return {
        "Xt": Xt,              # {xi}^K in ENU (Tensor[K,2] on GPU) - MAY BE PADDED
        "t": 1.0,
        "z": (lat0, lon0),
        "end": end,            # Original end index
        "real_len": real_len   # ADD THIS: actual data length (excluding padding)
    }

def pred(theta, chunk):
    Xt = chunk["Xt"].unsqueeze(0)                                   # (1,K,2)
    t = torch.tensor([[chunk["t"]]], dtype=torch.float32, device=DEVICE)  # (1,1)
    with torch.no_grad():
        Vt = theta(Xt, t).squeeze(0)                                 # (K,2)
    return Vt

def denoise_traj(traj, theta):
    K, Q = 256, 1
    t_delta = 0.05
    conf = {"K": K, "Q": Q}

    traj_denoised = []
    buckle = None

    stride = (K - Q)
    total = int(np.ceil(len(traj) / stride))

    for idx in range(total):
        chunk = chunk_dicer(traj, idx=idx, buckle=buckle, conf=conf)

        while chunk["t"] > 0.0:
            Vt = pred(theta, chunk)
            chunk["Xt"] = chunk["Xt"] - t_delta * Vt
            chunk["t"] = max(0.0, chunk["t"] - t_delta)

        # Convert back to GPS
        chunk["Xt"] = ENUtoGPS(chunk["Xt"], chunk["z"])
        buckle = chunk["Xt"][:Q]
        
        # FIX: Use real_len to exclude padded points
        real_len = chunk["real_len"]
        traj_denoised = merge(traj_denoised, chunk["Xt"][Q:real_len])

    return np.array(traj_denoised, dtype=float)

@torch.no_grad()
def denoise_chunk(theta, X_t, t=1.0, steps=30):
    theta = theta.to(DEVICE).eval()
    C = X_t.to(DEVICE).contiguous()
    dt = 1.0 / steps if steps > 0 else 1.0
    for i in range(steps):
        cur_t = max(0.0, t - i * dt)
        tt = torch.tensor([[cur_t]], dtype=torch.float32, device=DEVICE)
        V = theta(C.unsqueeze(0), tt).squeeze(0)
        C = C - dt * V
    return C.detach().cpu()

# === Accuracy tests (chunk & traj) ==========================================
# Error metric: per-point L2 distance in meters between aligned time points.

def _gps_to_enu_batch(gps_lonlat: np.ndarray, ref_lat: float, ref_lon: float) -> np.ndarray:
    # gps_lonlat: (N,2) -> columns [lon,lat]
    e, n, _ = geodetic2enu(gps_lonlat[:,1], gps_lonlat[:,0], 46, ref_lat, ref_lon, 0)
    return np.stack([e, n], axis=1)  # meters

def _err_stats(err_vec: np.ndarray) -> dict:
    return {
        "total_err": float(err_vec.sum()),
        "avg_err": float(err_vec.mean()),
        "med_err": float(np.median(err_vec)),
        "std_err": float(err_vec.std(ddof=0)),
        "count": int(err_vec.size),
    }

@torch.no_grad()
def test_chunk_accuracy(theta, pt_path: str, steps: int = 30, map_location: str = "cpu") -> dict:
    """
    Evaluate per-chunk accuracy on the processed .pt test file.
    Uses ENU distances (meters). Aggregates over all samples and all points.

    Returns:
      {
        "denoised": {total_err, avg_err, med_err, std_err, count},
        "input":    {total_err, avg_err, med_err, std_err, count}
      }
    """
    pack = torch.load(pt_path, map_location=map_location)
    # Expected: X_t: (N,K,4) [e,n,*,*] or at least [:,:2] ENU; V: (N,K,2); t: (N,1)
    X_t = pack["X_t"][:, :, :2]            # (N,K,2) ENU noisy
    Vref = pack["V"]                        # (N,K,2)
    tvec = pack["t"].reshape(-1)            # (N,)

    N, K, _ = X_t.shape

    theta = theta.eval().to(DEVICE)

    denoised_errs = []
    input_errs = []

    for i in range(N):
        x = X_t[i]                      # (K,2) ENU noisy at t_i
        v = Vref[i]                     # (K,2) reference velocity
        ti = float(tvec[i])

        # Reference clean ENU at t=0 from pack: X0_ref = X_t - t * V
        x0_ref = x - ti * v

        # Denoise with rectified-flow integration in ENU space
        C = x.to(DEVICE) if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32, device=DEVICE)
        dt = 1.0 / steps if steps > 0 else 1.0
        for s in range(steps):
            cur_t = max(0.0, ti - s * dt)
            tt = torch.tensor([[cur_t]], dtype=torch.float32, device=DEVICE)
            V = theta(C.unsqueeze(0), tt).squeeze(0)  # (K,2)
            C = C - dt * V
        x_hat = C.detach().cpu().numpy()  # (K,2) ENU

        # Errors in meters (already ENU)
        denoised_errs.append(np.linalg.norm(x_hat - x0_ref.numpy() if isinstance(x0_ref, torch.Tensor) else x0_ref, axis=1))
        input_errs.append(np.linalg.norm((x.numpy() if isinstance(x, torch.Tensor) else x) - (x0_ref.numpy() if isinstance(x0_ref, torch.Tensor) else x0_ref), axis=1))

    denoised_errs = np.concatenate(denoised_errs, axis=0)
    input_errs = np.concatenate(input_errs, axis=0)

    return {
        "denoised": _err_stats(denoised_errs),
        "input": _err_stats(input_errs),
    }

def test_traj_accuracy(theta, info: dict) -> dict:
    """
    Evaluate trajectory-wise accuracy (one user at a time) using parquet test split.
    Alignment: same timestamps; distances in meters via ENU.
    Uses data_extract('traj', info) to fetch the next user trajectory.

    Returns None if no more data. Otherwise:
      {
        "file": str,
        "usr": <id>,
        "denoised": {total_err, avg_err, med_err, std_err, count},
        "input":    {total_err, avg_err, med_err, std_err, count}
      }
    """
    # Pull noisy lon/lat and identifiers
    rec = data_extract("traj", info)
    if rec is None:
        return None

    traj_n = rec["traj"]          # (T,2) [lon_n, lat_n], sorted by time
    file_name = rec["file"]
    usr = rec["usr"]

    # FIX: Use correct column names (longitude, latitude) not (longitude_g, latitude_g)
    # Load parquet again to fetch the clean ground-truth for the same rows
    pq_path = RAW_DIR / file_name
    df = pl.read_parquet(pq_path).filter(
        (pl.col("agent") == usr) &
        pl.col("longitude_n").is_not_null() & pl.col("latitude_n").is_not_null() &
        pl.col("longitude").is_not_null() & pl.col("latitude").is_not_null() &
        pl.col("longitude_n").is_finite() & pl.col("latitude_n").is_finite() &
        pl.col("longitude").is_finite() & pl.col("latitude").is_finite()
    ).sort("timestamp")

    lng_g = df["longitude"].to_numpy() 
    lat_g = df["latitude"].to_numpy()  

    traj_g = np.stack([lng_g, lat_g], axis=1)  # (T,2)

    # Run denoising on the noisy GPS trajectory
    traj_hat = denoise_traj(traj_n, theta)     # (T',2) after stitch; first Q dropped by design

    # Align lengths (our decoder drops the first Q). Match to tail of ground-truth.
    T_hat = len(traj_hat)
    traj_g_aligned = traj_g[-T_hat:]
    traj_n_aligned = traj_n[-T_hat:]

    # Convert both to ENU meters using the first ground-truth point as origin
    ref_lat, ref_lon = float(traj_g_aligned[0,1]), float(traj_g_aligned[0,0])
    enu_hat = _gps_to_enu_batch(traj_hat, ref_lat, ref_lon)
    enu_g   = _gps_to_enu_batch(traj_g_aligned, ref_lat, ref_lon)
    enu_n   = _gps_to_enu_batch(traj_n_aligned, ref_lat, ref_lon)

    # L2 errors in meters
    err_hat = np.linalg.norm(enu_hat - enu_g, axis=1)
    err_inp = np.linalg.norm(enu_n - enu_g, axis=1)

    return {
        "file": file_name,
        "usr": usr,
        "denoised": _err_stats(err_hat),
        "input": _err_stats(err_inp),
    }