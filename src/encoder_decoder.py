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
    gps_chunk : (K,2) lon,lat degree
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
        blob = torch.load(str(ckpt_path), map_location=DEVICE)
        if isinstance(blob, dict) and "model_state_dict" in blob:
            sd = blob["model_state_dict"]
        else:
            sd = blob

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
    def __init__(self, ckpt_path: str, manual_config = None):
        """
        Purpose:
            Initialize EncoderDecoder with model and validate buckle configuration.
        
        Parameters:
            ckpt_path (str): Path to model checkpoint (.safetensors or .pt)
        
        Raises:
            AssertionError: If buckle configuration is illegal
        
        Validation Rules:
            1. K > Q1 + Q2 (basic sanity)
            2. payload_size >= Q1 (next chunk needs Q1 points from payload)
            3. Buckles are byte-aligned (Q1, Q2 are multiples of 8)
        
        TODO:
            1. Load model and config
            2. Extract K, Q1_bytes, Q2_bytes from config
            3. Convert bytes to points (multiply by 8)
            4. Validate buckle legality
            5. Compute stride and t_delta
        """
        ckpt_path = Path(ckpt_path)
        best_ckpt_dir = ckpt_path.parent
        model_dir = best_ckpt_dir.parent
        config_file = model_dir / "log" / "config.json"
        model, cfg = load_model_from_config(config_file, ckpt_path)

        self.model = model
        self.cfg   = cfg

        # ============================================================
        # 1. Extract buckle configuration (BYTE LEVEL)
        # ============================================================
        self.K = cfg.get("K", 256)
        # Defaults are fixed here; do NOT read Q1/Q2 from model config.
        self.Q1_bytes = 1   # number of BYTES (not points)
        self.Q2_bytes = 12  # number of BYTES (not points)

        # APPLY MANUAL OVERRIDE if provided
        if manual_config is not None:
            if "Q1" in manual_config:
                self.Q1_bytes = manual_config["Q1"]
            if "Q2" in manual_config:
                self.Q2_bytes = manual_config["Q2"]
                
        # ============================================================
        # 2. Convert bytes to points (8 points per byte)
        # ============================================================
        # Each byte represents 8 consecutive points in the chunk
        # This design is for regional accuracy (byte-aligned buckles)
        self.Q1 = self.Q1_bytes * 8        
        self.Q2 = self.Q2_bytes * 8

        # ============================================================
        # 3. Compute payload size
        # ============================================================
        payload_size = self.K - (self.Q1 + self.Q2)
        
        # ============================================================
        # 4. CRITICAL VALIDATION
        # ============================================================
        # Check 1: Basic sanity (K must be larger than total buckle size)
        assert self.K > self.Q1 + self.Q2, \
            f"Invalid buckle settings:\n" \
            f"  K={self.K} must be > Q1+Q2={self.Q1 + self.Q2}\n" \
            f"  (Q1_bytes={self.Q1_bytes}, Q2_bytes={self.Q2_bytes})"
        
        # Check 2: Payload legality
        # Next chunk needs Q1 points from previous chunk's payload as head buckle
        # Therefore: payload_size MUST be >= Q1
        assert payload_size >= self.Q1, \
            f"Illegal buckle configuration:\n" \
            f"  K={self.K}\n" \
            f"  Q1_bytes={self.Q1_bytes} → Q1={self.Q1} points\n" \
            f"  Q2_bytes={self.Q2_bytes} → Q2={self.Q2} points\n" \
            f"  payload_size = K - (Q1 + Q2) = {payload_size}\n" \
            f"  REQUIREMENT: payload_size >= Q1 (next chunk needs Q1 points from payload)\n" \
            f"  VIOLATION: {payload_size} < {self.Q1}\n" \
            f"  SOLUTION: Reduce Q1_bytes or Q2_bytes such that 2*Q1_bytes + Q2_bytes <= 32"

        # ============================================================
        # 5. Compute derived values
        # ============================================================
        self.stride  = payload_size
        self.t_delta = cfg.get("t_delta", 0.1)

        if manual_config is not None and "t_delta" in manual_config:
            self.t_delta = manual_config["t_delta"]

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
    def denoise_traj_DF(self, traj) -> np.ndarray:
        """
        traj : (T,2) noisy GPS lon,lat (may include NaN)

        Returns:
            clean_traj : (T',2) cleaned GPS traj (T' == T with NaN rows removed)
        """
        traj = np.asarray(traj, dtype=float)
        traj = remove_nan_rows(traj)
        N = len(traj)
        if N == 0:
            return np.zeros((0, 2), dtype=float)

        S = self.stride
        M = int(np.ceil(N / S))

        head = np.repeat(traj[0:1, :], self.Q1, axis=0) if self.Q1 > 0 else np.zeros((0, 2))
        payload_pad_len = M * S - N
        payload_pad = np.repeat(traj[-1:], payload_pad_len, axis=0) if payload_pad_len > 0 else np.zeros((0, 2))
        tail = np.repeat(traj[-1:], self.Q2, axis=0) if self.Q2 > 0 else np.zeros((0, 2))
        traj_padded = np.concatenate([head, traj, payload_pad, tail], axis=0)

        payloads = []
        for j in range(M):
            start = j * S
            end = start + self.K
            gps_chunk = traj_padded[start:end]
            gps_clean = self.denoise_chunk(gps_chunk)
            payload = gps_clean[self.Q1:self.Q1 + S]
            payloads.append(payload)

        out_full = np.concatenate(payloads, axis=0) if payloads else np.zeros((0, 2), dtype=float)
        out = out_full[:N]
        assert out.shape[0] == N, f"DF produced wrong length: out={out.shape[0]} != N={N}"
        return out

    def denoise_traj_BF(self, traj: np.ndarray) -> np.ndarray:
        """
        Purpose:
            ACCURATE trajectory denoising using BREADTH-FIRST traversal.
            Synchronize noise reduction across all chunks for maximum smoothness.
            Buckle sections receive context at matching noise levels.

        Parameters:
            traj (np.ndarray): (T, 2) GPS trajectory [lon, lat]
                - May contain NaN rows (will be removed)
                - T is arbitrary length >= K
            
        Return Dict:
            "error_code": 0 (success) | -1 (empty trajectory after NaN removal)
            "traj_clean": (T', 2) cleaned GPS trajectory (np.ndarray)
                - T' = T with NaN rows removed
                - dtype: float64

        Usage:
            Called by EncoderDecoder when user requests BF mode for highest quality.
            Recommended for offline processing where accuracy > speed.

        TODO:
            1. Remove NaN rows and validate trajectory length
            2. Calculate number of chunks needed
            3. Initialize trajectory storage at t=1.0
            4. Outer loop: iterate noise levels (t=1.0 → 0.0)
            5. Inner loop: denoise all chunks one step at current noise level
            6. Stitch payloads into full trajectory at each noise level
            7. Return final trajectory at t=0.0
        """
        
        # ================================================================
        # 1. Input validation and preprocessing
        # ================================================================
        traj = np.asarray(traj, dtype=float)
        traj = remove_nan_rows(traj)
        N = len(traj)
        
        if N == 0:
            return np.zeros((0, 2), dtype=float)
        
        # ================================================================
        # 2. Calculate chunk parameters
        # ================================================================
        S = self.stride
        M = int(np.ceil(N / S))
        
        
        # ================================================================
        # 3. Initialize trajectory storage
        # ================================================================
        # trajectories[t] = full GPS trajectory at noise level t
        # We only store the CLEAN trajectory points (no padding)
        trajectories = {1.0: traj.copy()}
        
        # ================================================================
        # 4. Outer loop: iterate over noise levels (y-axis)
        # ================================================================
        t_current = 1.0
        
        while t_current > 0.0:
            t_next = max(0.0, t_current - self.t_delta)
            
            # Storage for this iteration
            payloads = []  # Only payloads for final stitching
            
            # Current trajectory at t_current
            traj_at_t = trajectories[t_current]
            T_curr = len(traj_at_t)

            head = np.repeat(traj_at_t[0:1, :], self.Q1, axis=0) if self.Q1 > 0 else np.zeros((0, 2))
            payload_pad_len = M * S - T_curr
            payload_pad = np.repeat(traj_at_t[-1:], payload_pad_len, axis=0) if payload_pad_len > 0 else np.zeros((0, 2))
            tail = np.repeat(traj_at_t[-1:], self.Q2, axis=0) if self.Q2 > 0 else np.zeros((0, 2))
            traj_padded = np.concatenate([head, traj_at_t, payload_pad, tail], axis=0)
            
            # ============================================================
            # 5. Inner loop: denoise all chunks at t_current (x-axis)
            # ============================================================
            for j in range(M):
                # --------------------------------------------------------
                # 5a. Build chunk[j] at noise level t_current
                # --------------------------------------------------------
                start = j * S
                end = start + self.K
                chunk_gps = traj_padded[start:end]
                
                # --------------------------------------------------------
                # 5b. Transform GPS → ENU
                # --------------------------------------------------------
                chunk_enu, origin = gps_to_enu(chunk_gps)
                
                # --------------------------------------------------------
                # 5c. Denoise ONE STEP: t_current → t_next
                # --------------------------------------------------------
                Xt = torch.tensor(chunk_enu, device=DEVICE)
                t_tensor = torch.tensor(t_current, device=DEVICE)
                
                Xt_next, t_next_tensor, Vt = self.denoise_step(Xt, t_tensor)
                chunk_enu_next = Xt_next.detach().cpu().numpy()
                
                # --------------------------------------------------------
                # 5d. Transform ENU → GPS
                # --------------------------------------------------------
                chunk_gps_next = enu_to_gps(chunk_enu_next, origin)
                
                # --------------------------------------------------------
                # 5e. Store full chunk and extract payload
                # --------------------------------------------------------
                # Extract payload (strip Q1 head, Q2 tail)
                payload = chunk_gps_next[self.Q1:self.Q1 + S]
                payloads.append(payload)
            
            # ============================================================
            # 6. Stitch all payloads into full trajectory at t_next
            # ============================================================
            if len(payloads) > 0:
                stitched = np.concatenate(payloads, axis=0)
            else:
                stitched = np.zeros((0, 2), dtype=float)
            trajectories[t_next] = stitched[:N]
            
            # ============================================================
            # 7. Update noise level
            # ============================================================
            t_current = t_next
        
        # ================================================================
        # 8. Return final trajectory at t=0.0
        # ================================================================
        out = trajectories[0.0]
        assert out.shape[0] == N, f"BF produced wrong length: out={out.shape[0]} != N={N}"
        return out
