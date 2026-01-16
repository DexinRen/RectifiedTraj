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
        self.Q1_bytes = cfg.get("Q1", 1)   # number of BYTES (not points)
        self.Q2_bytes = cfg.get("Q2", 0)   # number of BYTES (not points)

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

    def denoise_traj_BF(self, traj: np.ndarray) -> dict:
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
        T = len(traj)
        
        if T == 0:
            return {
                "error_code": -1,
                "traj_clean": np.zeros((0, 2), dtype=float)
            }
        
        # ================================================================
        # 2. Calculate chunk parameters
        # ================================================================
        # Number of chunks needed to cover trajectory
        # First chunk: K-Q1 points from traj
        # Following chunks: stride points each
        num_chunks = 1  # first chunk
        remaining = T - (self.K - self.Q1)
        if remaining > 0:
            num_chunks += int(np.ceil(remaining / self.stride))
        
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
            denoised_chunks_full = []  # Full chunks (with buckles) for next chunk's head
            payloads = []              # Only payloads for final stitching
            
            # Current trajectory at t_current
            traj_at_t = trajectories[t_current]
            T_curr = len(traj_at_t)
            
            # ============================================================
            # 5. Inner loop: denoise all chunks at t_current (x-axis)
            # ============================================================
            curr_idx = 0  # Position in trajectory
            
            for chunk_i in range(num_chunks):
                
                # --------------------------------------------------------
                # 5a. Build chunk[i] at noise level t_current
                # --------------------------------------------------------
                if chunk_i == 0:
                    # First chunk: duplicate head buckle
                    dup_head = np.repeat(traj_at_t[0:1, :], self.Q1, axis=0)
                    take = traj_at_t[:self.K - self.Q1]
                    chunk_gps = np.concatenate([dup_head, take], axis=0)
                    real_len = chunk_gps.shape[0]
                    curr_idx = self.K - self.Q1
                    
                else:
                    # CRITICAL: Use previous chunk's denoised tail (at t_next)
                    prev_chunk_denoised = denoised_chunks_full[chunk_i - 1]
                    
                    # Extract head buckle: last Q1+Q2 points from prev chunk
                    # Last Q1 points before tail buckle
                    head_start = self.K - (self.Q1 + self.Q2)
                    head_end = self.K - self.Q2
                    head_buckle = prev_chunk_denoised[head_start:head_end, :]
                    
                    # Last Q2 points (tail buckle)
                    if self.Q2 > 0:
                        tail_buckle = prev_chunk_denoised[-self.Q2:, :]
                        buckle = np.concatenate([head_buckle, tail_buckle], axis=0)
                    else:
                        buckle = head_buckle
                    
                    # Remaining points from current trajectory at t_current
                    remain = T_curr - curr_idx
                    need = self.stride
                    
                    if remain >= need:
                        take = traj_at_t[curr_idx : curr_idx + need]
                        chunk_gps = np.concatenate([buckle, take], axis=0)
                        real_len = chunk_gps.shape[0]
                        curr_idx += need
                    else:
                        # Last chunk: pad with last point
                        take = traj_at_t[curr_idx:T_curr]
                        pad_len = need - take.shape[0]
                        pad = np.repeat(traj_at_t[-1:], pad_len, axis=0)
                        chunk_gps = np.concatenate([buckle, take, pad], axis=0)
                        real_len = buckle.shape[0] + take.shape[0]
                        curr_idx = T_curr
                
                # Pad to K if needed
                if chunk_gps.shape[0] < self.K:
                    pad2 = np.repeat(chunk_gps[-1:], self.K - chunk_gps.shape[0], axis=0)
                    chunk_gps = np.concatenate([chunk_gps, pad2], axis=0)
                
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
                denoised_chunks_full.append(chunk_gps_next)  # Keep for next chunk's buckle
                
                # Extract payload (strip Q1 head, Q2 tail)
                is_last = (chunk_i == num_chunks - 1)
                if is_last:
                    end_mid = min(self.K - self.Q2, real_len)
                else:
                    end_mid = self.K - self.Q2
                
                if end_mid > self.Q1:
                    payload = chunk_gps_next[self.Q1 : end_mid]
                    payloads.append(payload)
            
            # ============================================================
            # 6. Stitch all payloads into full trajectory at t_next
            # ============================================================
            if len(payloads) > 0:
                trajectories[t_next] = np.concatenate(payloads, axis=0)
            else:
                trajectories[t_next] = np.zeros((0, 2), dtype=float)
            
            # ============================================================
            # 7. Update noise level
            # ============================================================
            t_current = t_next
        
        # ================================================================
        # 8. Return final trajectory at t=0.0
        # ================================================================
        return {
            "error_code": 0,
            "traj_clean": trajectories[0.0]
        }