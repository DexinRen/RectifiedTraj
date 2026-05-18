import json
import os
import torch
import numpy as np
from pathlib import Path
from pymap3d import geodetic2enu, enu2geodetic
from safetensors.torch import load_file as load_safetensors

_DEFAULT_DEVICE = str(os.getenv("RECTIFIEDTRAJ_DEVICE", "cuda")).strip().lower()
if _DEFAULT_DEVICE.startswith("cpu"):
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _normalize_device_spec(device) -> str:
    token = str(device or "").strip().lower()
    if token.startswith("cuda"):
        return "cuda"
    if token == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported device spec: {device}. Use 'cuda' or 'cpu'.")


def _normalize_data_hypothesis(raw, default: str = "RectifiedTraj") -> str:
    """Normalize hypothesis aliases into canonical names."""
    token = str(raw if raw is not None else "").strip().lower().replace("-", "_")
    if token in {"", "rf", "rectified_flow", "rectified", "rectifiedtraj", "rectified_traj"}:
        return "RectifiedTraj"
    if token in {"rr", "residualreg", "residual_reg", "residual", "residual_regression"}:
        return "ResidualReg"
    text = str(raw).strip() if raw is not None else ""
    return text if text else str(default)


def set_runtime_device(device) -> torch.device:
    """
    Set the global runtime device used by EncoderDecoder and checkpoint loading.
    """
    global DEVICE
    target = _normalize_device_spec(device)
    if target == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("runtime.device=cuda requested but CUDA is unavailable.")
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
    return DEVICE


def get_runtime_device() -> str:
    return str(DEVICE)


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
    Xt_tensor  : (K,C)
    t_tensor   : scalar tensor
    Returns:
        Vt : (K,2)
    """
    Xt_b = Xt_tensor.unsqueeze(0)     # (1,K,C)
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
            1. K > Q1 + Q2 (positive payload stride)
            2. Buckles are byte-aligned (Q1, Q2 are multiples of 8)
        
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
        self.data_hypothesis = _normalize_data_hypothesis(
            cfg.get("data_hypothesis", cfg.get("data_hypothetis", "RectifiedTraj"))
        )
        self.input_coord_dim = int(cfg.get("input_coord_dim", cfg.get("coord_dim", 2)))

        # ============================================================
        # 1. Extract buckle configuration (BYTE LEVEL)
        # ============================================================
        self.K = cfg.get("K", 256)
        prediction_mode = str(cfg.get("prediction_mode", "offline")).strip().lower()
        if prediction_mode == "online":
            self.Q1_bytes = 0
            self.Q2_bytes = 0
        else:
            # Defaults are fixed here; do NOT read Q1/Q2 from model config.
            self.Q1_bytes = 1   # number of BYTES (not points)
            self.Q2_bytes = 12  # number of BYTES (not points)

        # APPLY MANUAL OVERRIDE if provided.
        # Q1/Q2 are eval-time byte-level assembly settings. Online checkpoints
        # default to no buckles, but trajectory eval may override them.
        if manual_config is not None:
            manual_q1 = int(manual_config.get("Q1", self.Q1_bytes))
            manual_q2 = int(manual_config.get("Q2", self.Q2_bytes))

            self.Q1_bytes = manual_q1
            self.Q2_bytes = manual_q2
                
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
        # Basic sanity: K must leave a positive payload stride.
        assert self.K > self.Q1 + self.Q2, \
            f"Invalid buckle settings:\n" \
            f"  K={self.K} must be > Q1+Q2={self.Q1 + self.Q2}\n" \
            f"  (Q1_bytes={self.Q1_bytes}, Q2_bytes={self.Q2_bytes})"

        # ============================================================
        # 5. Compute derived values
        # ============================================================
        self.stride  = payload_size
        self.t_delta = float(cfg.get("t_delta", 0.1))
        if manual_config is not None:
            if "denoise_steps" in manual_config and manual_config.get("denoise_steps") is not None:
                denoise_steps = int(manual_config["denoise_steps"])
                if denoise_steps <= 0:
                    raise ValueError(f"denoise_steps must be positive, got {denoise_steps}.")
                self.t_delta = 1.0 / float(denoise_steps)
            elif "t_delta" in manual_config and manual_config.get("t_delta") is not None:
                self.t_delta = float(manual_config["t_delta"])
        if not (0.0 < self.t_delta <= 1.0):
            raise ValueError(f"t_delta must be in (0, 1], got {self.t_delta}.")

    def _build_model_input(self, Xt: torch.Tensor, pad_count: int = 0, pad_mask=None) -> torch.Tensor:
        """
        Build model input tensor from ENU coordinates, adding is_pad if required.
        """
        if self.input_coord_dim <= 2:
            return Xt

        if self.input_coord_dim != 3:
            raise ValueError(
                f"Unsupported input_coord_dim={self.input_coord_dim}. "
                "EncoderDecoder currently supports 2 or 3 input channels."
            )

        K = int(Xt.shape[0])
        is_pad = torch.zeros((K, 1), dtype=Xt.dtype, device=Xt.device)
        if pad_mask is not None:
            mask = torch.as_tensor(pad_mask, dtype=Xt.dtype, device=Xt.device).reshape(-1)
            if int(mask.shape[0]) != K:
                raise ValueError(f"pad_mask length {int(mask.shape[0])} != chunk length {K}")
            is_pad[:, 0] = mask
        elif pad_count > 0:
            is_pad[:pad_count, 0] = 1.0
        return torch.cat([Xt, is_pad], dim=1)

    @torch.no_grad()
    def _pred_chunk(self, Xt: torch.Tensor, t: torch.Tensor, pad_count: int = 0, pad_mask=None) -> torch.Tensor:
        Xt_input = self._build_model_input(Xt, pad_count=pad_count, pad_mask=pad_mask)
        return pred_chunk(self.model, Xt_input, t)

    # ========================================================
    # Public: denoise a single chunk (GPS, shape (K,2))
    # ========================================================
    def denoise_chunk(self, gps_chunk: np.ndarray, pad_count: int = 0, pad_mask=None) -> np.ndarray:
        """
        GPS chunk (lon,lat) → ENU → RF clean → GPS
        """
        Xt_np, origin = gps_to_enu(gps_chunk)
        Xt_clean_np = self.denoise_chunk_enu(Xt_np, pad_count=pad_count, pad_mask=pad_mask)
        gps_clean = enu_to_gps(Xt_clean_np, origin)
        return gps_clean

    @torch.no_grad()
    def denoise_step(self, Xt: torch.Tensor, t: torch.Tensor, pad_count: int = 0, pad_mask=None):
        """
        Perform ONE RF Euler update and return:
            Xt_next, t_next, Vt
        """
        if self.data_hypothesis == "ResidualReg":
            x0_pred = self._pred_chunk(Xt, t, pad_count=pad_count, pad_mask=pad_mask)
            t_next = torch.tensor(0.0, device=Xt.device)
            return x0_pred, t_next, x0_pred

        v_pred = self._pred_chunk(Xt, t, pad_count=pad_count, pad_mask=pad_mask)   # (K,2)
        Xt_next = Xt - self.t_delta * v_pred
        t_next = torch.tensor(max(0.0, t.item() - self.t_delta), device=Xt.device)
        return Xt_next, t_next, v_pred

    @torch.no_grad()
    def denoise_chunk_enu(self, Xt_np: np.ndarray, pad_count: int = 0, pad_mask=None) -> np.ndarray:
        """
        Input:
            Xt_np : (K,2) ENU noisy chunk

        Output:
            Xt_clean_np : (K,2) ENU cleaned chunk

        Performs ONLY RF integration in ENU space.
        No GPS conversion. No stitching. No padding logic.
        """
        Xt = torch.tensor(Xt_np, device=DEVICE)
        if self.data_hypothesis == "ResidualReg":
            t = torch.tensor(1.0, device=DEVICE)
            x0_pred = self._pred_chunk(Xt, t, pad_count=pad_count, pad_mask=pad_mask)
            return x0_pred.detach().cpu().numpy()

        t  = torch.tensor(1.0, device=DEVICE)

        while t.item() > 0.0:
            Vt = self._pred_chunk(Xt, t, pad_count=pad_count, pad_mask=pad_mask)
            Xt = Xt - self.t_delta * Vt
            t  = torch.tensor(max(0.0, t.item() - self.t_delta), device=DEVICE)

        return Xt.detach().cpu().numpy()

    def build_padded_trajectory(self, traj) -> tuple[np.ndarray, np.ndarray, int, int]:
        """
        Build encoder-owned chunk padding and is_pad mask for trajectory evaluation.

        Returns:
            traj_padded: GPS trajectory with artificial head/payload/tail padding.
            pad_mask: 1 for artificial padding positions, 0 for real observations.
            n_chunks: number of K-sized chunks needed.
            n_points: number of real observations after NaN removal.
        """
        traj = np.asarray(traj, dtype=float)
        traj = remove_nan_rows(traj)
        n_points = int(len(traj))
        if n_points == 0:
            return (
                np.zeros((0, 2), dtype=float),
                np.zeros((0,), dtype=np.float32),
                0,
                0,
            )

        stride = int(self.stride)
        n_chunks = int(np.ceil(n_points / stride))

        head = np.repeat(traj[0:1, :], self.Q1, axis=0) if self.Q1 > 0 else np.zeros((0, 2))
        payload_pad_len = n_chunks * stride - n_points
        payload_pad = (
            np.repeat(traj[-1:], payload_pad_len, axis=0)
            if payload_pad_len > 0
            else np.zeros((0, 2))
        )
        tail = np.repeat(traj[-1:], self.Q2, axis=0) if self.Q2 > 0 else np.zeros((0, 2))
        traj_padded = np.concatenate([head, traj, payload_pad, tail], axis=0)
        pad_mask = np.concatenate(
            [
                np.ones((len(head),), dtype=np.float32),
                np.zeros((n_points,), dtype=np.float32),
                np.ones((len(payload_pad),), dtype=np.float32),
                np.ones((len(tail),), dtype=np.float32),
            ],
            axis=0,
        )
        return traj_padded, pad_mask, n_chunks, n_points

    # ========================================================
    # Public: denoise an arbitrary-length GPS trajectory
    # ========================================================
    def denoise_traj_DF(self, traj) -> np.ndarray:
        """
        traj : (T,2) noisy GPS lon,lat (may include NaN)

        Returns:
            clean_traj : (T',2) cleaned GPS traj (T' == T with NaN rows removed)
        """
        traj_padded, pad_mask_padded, M, N = self.build_padded_trajectory(traj)
        if N == 0:
            return np.zeros((0, 2), dtype=float)

        S = self.stride

        payloads = []
        for j in range(M):
            start = j * S
            end = start + self.K
            gps_chunk = traj_padded[start:end]
            chunk_pad_mask = pad_mask_padded[start:end]
            gps_clean = self.denoise_chunk(gps_chunk, pad_mask=chunk_pad_mask)
            payload = gps_clean[self.Q1:self.Q1 + S]
            payloads.append(payload)

        out_full = np.concatenate(payloads, axis=0) if payloads else np.zeros((0, 2), dtype=float)
        out = out_full[:N]
        assert out.shape[0] == N, f"chunk_stitch produced wrong length: out={out.shape[0]} != N={N}"
        return out
