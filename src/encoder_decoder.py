import json
import torch
import numpy as np
from pathlib import Path
from pymap3d import geodetic2enu, enu2geodetic
from safetensors.torch import load_file as load_safetensors

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Small helper: remove NaNs from trajectory
# ============================================================
def remove_nan_rows(arr: np.ndarray) -> np.ndarray:
    """Remove any row that contains NaN."""
    mask = ~np.isnan(arr).any(axis=1)
    return arr[mask]


# ============================================================
# Helper: GPS → ENU and ENU → GPS (per-chunk), matching training
# ============================================================
def gps_to_enu(gps_chunk: np.ndarray):
    """
    gps_chunk : (K,2) lon,lat degrees
    Returns:
        Xt : (K,2) ENU coordinates in meters
        origin : (lat0, lon0)
    """
    lon0, lat0 = gps_chunk[0, 0], gps_chunk[0, 1]
    lons = gps_chunk[:, 0]
    lats = gps_chunk[:, 1]

    # Match training: h = 46, origin height 0
    e, n, _ = geodetic2enu(lats, lons, 46.0, lat0, lon0, 0.0)
    Xt = np.stack([e, n], axis=1)
    return Xt.astype(np.float32), (lat0, lon0)


def enu_to_gps(enu_chunk: np.ndarray, origin):
    """
    enu_chunk : (K,2)
    origin    : (lat0, lon0)
    Returns:
        gps : (K,2) lon,lat
    """
    e = enu_chunk[:, 0]
    n = enu_chunk[:, 1]
    lat0, lon0 = origin

    # Match training: h = 46, origin height 0
    lats, lons, _ = enu2geodetic(e, n, 46.0, lat0, lon0, 0.0)
    gps = np.stack([lons, lats], axis=1)
    return gps


# ============================================================
# Helper: load theta model & pred() wrapper
# ============================================================
def load_model_from_config(config_json_path: Path, ckpt_path: Path):
    """
    config_json_path : path/to/config.json
    ckpt_path        : *.ckpt or *.safetensors

    Returns:
        model : torch.nn.Module on DEVICE
        cfg   : dict
    """
    from theta_model import build_theta_model

    cfg = json.loads(Path(config_json_path).read_text())
    runtime = {"config": cfg}

    model = build_theta_model(runtime).to(DEVICE)

    ckpt_path = Path(ckpt_path)
    if ckpt_path.suffix == ".safetensors":
        sd = load_safetensors(str(ckpt_path))
    else:
        sd = torch.load(str(ckpt_path), map_location=DEVICE)

    model.load_state_dict(sd)
    model.eval()
    return model, cfg


@torch.no_grad()
def pred_chunk(model, Xt_tensor: torch.Tensor, t_tensor: torch.Tensor) -> torch.Tensor:
    """
    model      : theta model
    Xt_tensor  : (K,2)
    t_tensor   : scalar tensor
    Returns:
        Vt : (K,2)
    """
    Xt_b = Xt_tensor.unsqueeze(0)     # (1,K,2)
    t_b  = t_tensor.view(1, 1)        # (1,1)  — matches theta_model.forward
    Vt = model(Xt_b, t_b)             # (1,K,2)
    return Vt.squeeze(0)              # (K,2)


# ============================================================
# ================= ENCODER–DECODER CLASS =====================
# ============================================================

class EncoderDecoder:
    def __init__(self, ckpt_path: str):
        """
        - Load config.json from model directory
        - Build model
        - Extract (K, Q1, Q2) with defaults:
              K=256, Q1=1, Q2=0 if missing
        """
        ckpt_path = Path(ckpt_path)
        best_ckpt_dir = ckpt_path.parent
        model_dir = best_ckpt_dir.parent
        print("============================")
        print(ckpt_path)
        print(model_dir)
        print("============================")
        config_file = model_dir / "log" / "config.json"
        model, cfg = load_model_from_config(config_file, ckpt_path)

        self.model = model
        self.cfg   = cfg

        self.K  = cfg.get("K", 256)
        self.Q1_bytes = cfg.get("Q1", 1)   # number of bytes
        self.Q2_bytes = cfg.get("Q2", 0)

        self.Q1 = self.Q1_bytes * 8        # convert to points
        self.Q2 = self.Q2_bytes * 8


        assert self.K > self.Q1 + self.Q2, "Invalid buckle settings: K must be > Q1+Q2."

        self.stride  = self.K - (self.Q1 + self.Q2)
        self.t_delta = cfg.get("t_delta", 0.1)

    # ========================================================
    # Public: denoise a single chunk (GPS, shape (K,2))
    # ========================================================
    def denoise_chunk(self, gps_chunk: np.ndarray) -> np.ndarray:
        """
        GPS chunk (lon,lat) → ENU → RF clean → GPS
        """
        Xt_np, origin = gps_to_enu(gps_chunk)
        Xt_clean_np = self.denoise_chunk_enu(Xt_np)
        gps_clean = enu_to_gps(Xt_clean_np, origin)
        return gps_clean

    @torch.no_grad()
    def denoise_step(self, Xt: torch.Tensor, t: torch.Tensor):
        """
        Perform ONE RF Euler update and return:
            Xt_next, t_next, Vt
        """
        Vt = pred_chunk(self.model, Xt, t)   # (K,2)
        Xt_next = Xt - self.t_delta * Vt
        t_next  = torch.tensor(max(0.0, t.item() - self.t_delta), device=Xt.device)

        return Xt_next, t_next, Vt

    @torch.no_grad()
    def denoise_chunk_enu(self, Xt_np: np.ndarray) -> np.ndarray:
        """
        Input:
            Xt_np : (K,2) ENU noisy chunk

        Output:
            Xt_clean_np : (K,2) ENU cleaned chunk

        Performs ONLY RF integration in ENU space.
        No GPS conversion. No stitching. No padding logic.
        """
        Xt = torch.tensor(Xt_np, device=DEVICE)
        t  = torch.tensor(1.0, device=DEVICE)

        while t.item() > 0.0:
            Vt = pred_chunk(self.model, Xt, t)
            Xt = Xt - self.t_delta * Vt
            t  = torch.tensor(max(0.0, t.item() - self.t_delta), device=DEVICE)

        return Xt.detach().cpu().numpy()

    # ========================================================
    # Public: denoise an arbitrary-length GPS trajectory
    # ========================================================
    def denoise_traj(self, traj) -> np.ndarray:
        """
        traj : (T,2) noisy GPS lon,lat (may include NaN)

        Returns:
            clean_traj : (T',2) cleaned GPS traj (T' == T with NaN rows removed)
        """
        traj = np.asarray(traj, dtype=float)
        traj = remove_nan_rows(traj)
        T = len(traj)
        if T == 0:
            return np.zeros((0, 2), dtype=float)

        chunks_gps = []
        curr = 0

        # -------- Chunk 0 --------
        dup  = np.repeat(traj[0:1, :], self.Q1, axis=0)
        take = traj[: self.K - self.Q1]
        piece0 = np.concatenate([dup, take], axis=0)
        real_len0 = piece0.shape[0]

        if real_len0 < self.K:
            pad = np.repeat(piece0[-1:], self.K - real_len0, axis=0)
            piece0 = np.concatenate([piece0, pad], axis=0)

        chunks_gps.append((piece0, real_len0))
        curr = self.K - self.Q1

        # -------- Middle / last chunks --------
        while curr < T:
            prev_piece, _ = chunks_gps[-1]
            overlap = prev_piece[-(self.Q1 + self.Q2):] if (self.Q1 + self.Q2) > 0 else np.zeros((0, 2))

            remain = T - curr
            need   = self.stride

            if remain >= need:
                take = traj[curr: curr + need]
                curr += need
                piece = np.concatenate([overlap, take], axis=0)
                real_len = piece.shape[0]
            else:
                take = traj[curr:T]
                pad_len = need - take.shape[0]
                pad = np.repeat(traj[-1:], pad_len, axis=0)
                piece = np.concatenate([overlap, take, pad], axis=0)
                real_len = overlap.shape[0] + take.shape[0]
                curr = T

            if piece.shape[0] < self.K:
                pad2 = np.repeat(piece[-1:], self.K - piece.shape[0], axis=0)
                piece = np.concatenate([piece, pad2], axis=0)

            chunks_gps.append((piece, real_len))

        # -------- Denoise each chunk & stitch --------
        output = []

        for idx, (gps_chunk, real_len) in enumerate(chunks_gps):
            gps_clean = self.denoise_chunk(gps_chunk)

            is_last = (idx == len(chunks_gps) - 1)
            if is_last:
                end_mid = real_len
            else:
                end_mid = min(self.K - self.Q2, real_len)

            if end_mid > self.Q1:
                mid = gps_clean[self.Q1:end_mid]
                output.append(mid)

        return np.concatenate(output, axis=0)
