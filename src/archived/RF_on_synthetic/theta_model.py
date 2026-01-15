"""
theta: Rectified Flow Model for GPS Trajectory Denoising

This module implements a unified neural network θ that predicts velocity field
v_θ(X_t, t) under the rectified flow framework for GPS trajectory denoising.

Architecture choices:
- 'nn': Multi-layer perceptron (MLP) - simple, efficient for local patterns
- 'cnn1d': 1D Convolutional network - captures local temporal patterns
- 'transformer': Self-attention based - captures global dependencies

The model learns a rectified flow field that drives noisy data X₁ toward
clean data X₀ by predicting v_target = X₁ - X₀, trained with Huber loss.

CRITICAL RECTIFIED FLOW CONCEPT:
    t ∈ [0, 1): NOISE LEVEL / INTERPOLATION PARAMETER, NOT time!
    
    - t=0: Clean data (X_0)
    - t=1: Noisy data (X_1)  
    - X_t = X_0 + t * V  (linear interpolation between clean and noisy)
    - Model learns: v_θ(X_t, t) ≈ V = X_1 - X_0
    
    Trajectory timestamps (actual GPS time) are in X_t[:, 2] (3rd column).

Loss function:
    L(θ) = Σ ρ(v_θ(X_t, t) - v_target)
    ρ(x) = { ½||x||², if ||x|| ≤ δ
           { δ||x|| - ½δ², otherwise

Target model size: ~100M parameters (preferably ~10M)
"""
from pathlib import Path
import time
import sys
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
logger = logging.getLogger(__name__)


# ============================================================================
# Model Architecture Components
# ============================================================================

class SinusoidalNoiseEmbedding(nn.Module):
    """
    Sinusoidal embedding for noise level conditioning in rectified flow.
    
    CRITICAL: t is the NOISE LEVEL / INTERPOLATION PARAMETER, NOT time!
    
    In rectified flow:
    - t ∈ [0, 1): Noise interpolation parameter
    - t=0: Clean data (X_0, no noise)
    - t=1: Noisy data (X_1, full noise)
    - X_t = X_0 + t * V  (linear interpolation)
    
    Trajectory timestamps are in X_t[:, 2] (3rd column).
    
    Maps scalar noise level t ∈ [0, 1) to a fixed-dimensional embedding.
    """
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B, 1) noise level values in [0, 1)
        Returns:
            (B, dim) noise embeddings
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class MLPBlock(nn.Module):
    """
    MLP block with LayerNorm, Linear, GELU activation, and residual connection.
    """
    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, K, dim)
        Returns:
            (B, K, dim)
        """
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class thetaMLP(nn.Module):
    """
    Multi-Layer Perceptron architecture for rectified flow.
    Processes each point independently with noise level conditioning.
    
    NOTE: Input X_t contains only coordinates (E, N). Trajectory metadata
    (timestamp, is_start) is NOT used in this simple MLP architecture since
    we process each point independently. For temporal modeling, use CNN1D or Transformer.
    
    Parameters:
        K: chunk length (default 256)
        coord_dim: coordinate dimension (default 2 for E, N)
        hidden: hidden layer width
        layers: number of MLP blocks
        noise_dim: noise level embedding dimension
    """
    def __init__(self, K: int = 256, coord_dim: int = 2, hidden: int = 512, 
                 layers: int = 6, noise_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.K = K
        self.coord_dim = coord_dim
        self.hidden = hidden
        self.layers = layers
        
        # Noise level embedding (for rectified flow parameter t)
        self.noise_embed = SinusoidalNoiseEmbedding(noise_dim)
        
        # Input projection: (B, K, 2) -> (B, K, hidden)
        self.input_proj = nn.Linear(coord_dim, hidden)
        
        # Noise level projection: (B, noise_dim) -> (B, hidden)
        self.noise_proj = nn.Linear(noise_dim, hidden)
        
        # MLP blocks
        self.blocks = nn.ModuleList([
            MLPBlock(hidden, hidden * 4, dropout) for _ in range(layers)
        ])
        
        # Output projection: (B, K, hidden) -> (B, K, 2)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, coord_dim)
        )
        
    def forward(self, X_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict velocity field v_θ(X_t, t).
        
        Args:
            X_t: (B, K, 2) trajectory at noise level t, coordinates [E, N]
            t: (B, 1) noise level in [0, 1) (rectified flow interpolation parameter)
        Returns:
            v_theta: (B, K, 2) predicted velocity field
        """
        B, K, C = X_t.shape
        
        # Embed noise level: (B, 1) -> (B, noise_dim) -> (B, hidden)
        t_emb = self.noise_embed(t)  # (B, noise_dim)
        t_emb = self.noise_proj(t_emb)  # (B, hidden)
        
        # Project input coordinates: (B, K, 2) -> (B, K, hidden)
        x = self.input_proj(X_t)  # (B, K, hidden)
        
        # Add noise level embedding to all points
        x = x + t_emb.unsqueeze(1)  # (B, K, hidden)
        
        # Apply MLP blocks
        for block in self.blocks:
            x = block(x)
        
        # Output projection
        v_theta = self.output_proj(x)  # (B, K, 2)
        
        return v_theta


class thetaCNN1D(nn.Module):
    """
    1D Convolutional architecture for rectified flow.
    Captures local temporal patterns in trajectory sequences.
    
    Parameters:
        K: chunk length (default 256)
        coord_dim: coordinate dimension (default 2)
        hidden: base channel width
        layers: number of conv blocks
        kernel_size: convolutional kernel size
        noise_dim: noise level embedding dimension
    """
    def __init__(self, K: int = 256, coord_dim: int = 2, hidden: int = 256,
                 layers: int = 8, kernel_size: int = 7, noise_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.K = K
        self.coord_dim = coord_dim
        self.hidden = hidden
        self.layers = layers
        
        # Noise level embedding
        self.noise_embed = SinusoidalNoiseEmbedding(noise_dim)
        self.noise_proj = nn.Linear(noise_dim, hidden)
        
        # Input projection: (B, 2, K) -> (B, hidden, K)
        self.input_proj = nn.Conv1d(coord_dim, hidden, kernel_size=1)
        
        # Convolutional blocks with residual connections
        self.conv_blocks = nn.ModuleList()
        for i in range(layers):
            self.conv_blocks.append(nn.ModuleList([
                nn.LayerNorm(hidden),
                nn.Conv1d(hidden, hidden, kernel_size=kernel_size, 
                         padding=kernel_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden, hidden, kernel_size=1),
                nn.Dropout(dropout)
            ]))
        
        # Output projection: (B, hidden, K) -> (B, 2, K)
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, coord_dim, kernel_size=1)
        )
        
    def forward(self, X_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict velocity field v_θ(X_t, t).
        
        Args:
            X_t: (B, K, 2) trajectory at noise level t
            t: (B, 1) noise level in [0, 1)
        Returns:
            v_theta: (B, K, 2) predicted velocity field
        """
        B, K, C = X_t.shape
        
        # Embed noise level: (B, 1) -> (B, hidden)
        t_emb = self.noise_embed(t)  # (B, noise_dim)
        t_emb = self.noise_proj(t_emb)  # (B, hidden)
        
        # Transpose for conv1d: (B, K, 2) -> (B, 2, K)
        x = X_t.transpose(1, 2)  # (B, 2, K)
        
        # Input projection: (B, 2, K) -> (B, hidden, K)
        x = self.input_proj(x)  # (B, hidden, K)
        
        # Add noise level embedding
        x = x + t_emb[:, :, None]  # (B, hidden, K)
        
        # Apply conv blocks with residual connections
        for norm, conv1, act, drop1, conv2, drop2 in self.conv_blocks:
            residual = x
            # LayerNorm operates on channel dimension
            x_norm = norm(x.transpose(1, 2)).transpose(1, 2)
            x = conv1(x_norm)
            x = act(x)
            x = drop1(x)
            x = conv2(x)
            x = drop2(x)
            x = x + residual
        
        # Output projection: (B, hidden, K) -> (B, 2, K)
        x = self.output_proj(x)  # (B, 2, K)
        
        # Transpose back: (B, 2, K) -> (B, K, 2)
        v_theta = x.transpose(1, 2)  # (B, K, 2)
        
        return v_theta


class thetaTransformer(nn.Module):
    """
    Transformer architecture for rectified flow.
    Captures global dependencies in trajectory sequences.
    
    Parameters:
        K: chunk length (default 256)
        coord_dim: coordinate dimension (default 2)
        hidden: model dimension (d_model)
        layers: number of transformer layers
        nhead: number of attention heads
        noise_dim: noise level embedding dimension
    """
    def __init__(self, K: int = 256, coord_dim: int = 2, hidden: int = 256,
                 layers: int = 6, nhead: int = 8, noise_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.K = K
        self.coord_dim = coord_dim
        self.hidden = hidden
        self.layers = layers
        
        # Noise level embedding
        self.noise_embed = SinusoidalNoiseEmbedding(noise_dim)
        self.noise_proj = nn.Linear(noise_dim, hidden)
        
        # Input projection: (B, K, 2) -> (B, K, hidden)
        self.input_proj = nn.Linear(coord_dim, hidden)
        
        # Positional encoding for sequence position
        self.pos_embed = nn.Parameter(torch.randn(1, K, hidden) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        
        # Output projection: (B, K, hidden) -> (B, K, 2)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, coord_dim)
        )
        
    def forward(self, X_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict velocity field v_θ(X_t, t).
        
        Args:
            X_t: (B, K, 2) trajectory at noise level t
            t: (B, 1) noise level in [0, 1)
        Returns:
            v_theta: (B, K, 2) predicted velocity field
        """
        B, K, C = X_t.shape
        
        # Embed noise level: (B, 1) -> (B, hidden)
        t_emb = self.noise_embed(t)  # (B, noise_dim)
        t_emb = self.noise_proj(t_emb)  # (B, hidden)
        
        # Project input: (B, K, 2) -> (B, K, hidden)
        x = self.input_proj(X_t)  # (B, K, hidden)
        
        # Add positional and noise level embeddings
        x = x + self.pos_embed  # Add position info
        x = x + t_emb.unsqueeze(1)  # Add noise level info
        
        # Apply transformer
        x = self.transformer(x)  # (B, K, hidden)
        
        # Output projection
        v_theta = self.output_proj(x)  # (B, K, 2)
        
        return v_theta


# ============================================================================
# Model Building Functions
# ============================================================================

def build_theta(model_type: str,
                coord_dim: int = 2,
                hidden: int = 512,
                layers: int = 6,
                K: int = 256,
                dropout: float = 0.1,
                **kwargs) -> dict:
    """
    Purpose:
        Build the neural network θ : (X_t, t) → v_θ(X_t, t)
        under one of three supported architectures.

    Parameters:
        model_type (str): "nn", "cnn1d", or "transformer"
        coord_dim (int): coordinate dimension (default 2)
        hidden (int): hidden layer width
        layers (int): number of layers
        K (int): chunk length (default 256)
        dropout (float): dropout rate
        **kwargs: additional model-specific arguments

    Return Dict:
        {
            "model": nn.Module,
            "model_type": str,
            "num_params": int
        }
    """
    if model_type.lower() == "nn":
        model = thetaMLP(
            K=K,
            coord_dim=coord_dim,
            hidden=hidden,
            layers=layers,
            dropout=dropout
        )
    elif model_type.lower() == "cnn1d":
        model = thetaCNN1D(
            K=K,
            coord_dim=coord_dim,
            hidden=hidden,
            layers=layers,
            dropout=dropout,
            kernel_size=kwargs.get('kernel_size', 7)
        )
    elif model_type.lower() == "transformer":
        model = thetaTransformer(
            K=K,
            coord_dim=coord_dim,
            hidden=hidden,
            layers=layers,
            dropout=dropout,
            nhead=kwargs.get('nhead', 8)
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. "
                        f"Choose from ['nn', 'cnn1d', 'transformer']")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Built {model_type.upper()} model with {num_params:,} parameters")
    
    return {
        "model": model,
        "model_type": model_type,
        "num_params": num_params
    }


def init_training_runtime(model: nn.Module,
                          device: str,
                          lr: float = 1e-4,
                          weight_decay: float = 0.01,
                          epochs: int = 10,
                          steps_per_epoch: int = 37500,
                          **kwargs) -> dict:
    """
    Purpose:
        Initialize the runtime environment for rectified-flow training.
        Triple the initial learning rate and extend the scheduler so
        the cosine cycle spans all epochs (total_steps = steps_per_epoch * epochs).

    Parameters:
        model (nn.Module): model θ
        device (str): "cuda" or "cpu"
        lr (float): base learning rate (will be tripled)
        weight_decay (float): AdamW weight decay
        epochs (int): number of epochs for scheduling
        steps_per_epoch (int): steps per epoch (for cosine schedule)
        **kwargs: optional overrides

    Return Dict:
        {
            "theta": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "device": device,
            "lr": float
        }
    """
    # === 1. Prepare model and optimizer ===
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=kwargs.get('betas', (0.9, 0.999))
    )

    # === 2. Setup warm-up + cosine scheduler ===
    warmup_steps = kwargs.get("warmup_steps", 1000)
    total_steps = steps_per_epoch * 2
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_steps
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )

    # === 3. Log and return ===
    logger.info(f"Initialized training runtime on {device}")
    logger.info(f"Learning rate (tripled): {lr:.2e}, Weight decay: {weight_decay}")
    logger.info(f"Scheduler total steps: {total_steps}, warmup: {warmup_steps}")
    return {
        "theta": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "device": device,
        "lr": lr
    }


def predict_velocity(model: nn.Module,
                     X_t: torch.Tensor,
                     t: torch.Tensor) -> dict:
    """
    Purpose:
        Perform forward pass to compute predicted velocity field v_θ(X_t, t).

    Parameters:
        model (nn.Module): model θ
        X_t (Tensor): trajectory at noise level t (B, K, 2)
        t (Tensor): noise level in [0, 1) (B, 1)

    Return Dict:
        {"v_theta": Tensor(B, K, 2)}
    """
    v_theta = model(X_t, t)
    return {"v_theta": v_theta}


def compute_velocity_target(X0: torch.Tensor,
                            X1: torch.Tensor) -> dict:
    """
    Purpose:
        Compute ground-truth velocity field label v_target = X₁ - X₀.

    Parameters:
        X0 (Tensor): clean coordinates (B, K, 2)
        X1 (Tensor): noisy coordinates (B, K, 2)

    Return Dict:
        {"v_target": Tensor(B, K, 2)}
    """
    v_target = X1 - X0
    return {"v_target": v_target}

@torch.no_grad()
def quick_acc_test(model, val_data, runtime, batch_size=32, device="cuda"):
    """
    Quick validation: compute per-sample L1 distance statistics (m).
    Ignores the first Q points of each chunk.

    Args:
        model: Trained model.
        val_data: dict containing {"X_t", "V", "t"}.
        runtime: dict containing "Q" or "config.Q".
        batch_size: mini-batch size for evaluation.
        device: compute device.

    Returns:
        dict(mean=float, median=float, std=float)
    """
    model.eval()
    X_t = val_data["X_t"].to(device)
    V   = val_data["V"].to(device)
    t   = val_data["t"].to(device)

    # --- Determine Q (ignore first Q points) ---
    Q = (
        getattr(runtime.get("config", {}), "Q", None)
        or runtime.get("Q", None)
        or 1
    )
    if not isinstance(Q, int) or Q < 0 or Q >= X_t.shape[1]:
        Q = 1

    # --- Collect per-sample errors ---
    all_errors = []

    for i in range(0, len(X_t), batch_size):
        xb = X_t[i:i+batch_size]
        vb = V[i:i+batch_size]
        tb = t[i:i+batch_size]

        # === Defensive channel fix ===
        # Some quick_val.pt files have 4 channels [lon, lat, ts, is_start]
        # Only the first two (EN coordinates) should go into the model.
        if xb.shape[2] == 4:
            xb = xb[:, :, :2]

        pred = model(xb, tb)

        # L1 distance per sample (mean over spatial dims, ignoring first Q)
        l1 = torch.abs(pred[:, Q:] - vb[:, Q:]).mean(dim=(1, 2))
        all_errors.append(l1.cpu())


    all_errors = torch.cat(all_errors)
    mean_err   = all_errors.mean().item()
    median_err = all_errors.median().item()
    std_err    = all_errors.std(unbiased=False).item()

    return {"mean": mean_err, "median": median_err, "std": std_err}


def compute_huber_loss(v_theta: torch.Tensor,
                       v_target: torch.Tensor,
                       delta: float = 1.0,
                       mask: torch.Tensor = None) -> dict:
    """
    Purpose:
        Compute robust Huber loss between predicted and target velocities.

    Parameters:
        v_theta (Tensor): predicted velocity (B, K, 2)
        v_target (Tensor): label velocity (B, K, 2)
        delta (float): transition threshold for linear/quadratic region
        mask (Tensor | None): optional weight mask (B, K, 1) or (B, K)

    Return Dict:
        {"loss": scalar Tensor, "mean_error": float}
    
    Notes:
        Huber loss: ρ(x) = { ½||x||², if ||x|| ≤ δ
                            { δ||x|| - ½δ², otherwise
    """
    # Compute per-point L2 error: ||v_θ - v_target||_2
    diff = v_theta - v_target  # (B, K, 2)
    error = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)  # (B, K)
    
    # Apply Huber loss formula
    huber = torch.where(
        error <= delta,
        0.5 * error ** 2,  # Quadratic for small errors
        delta * error - 0.5 * delta ** 2  # Linear for large errors
    )  # (B, K)
    
    # Apply mask if provided
    if mask is not None:
        if mask.dim() == 3:  # (B, K, 1)
            mask = mask.squeeze(-1)  # (B, K)
        huber = huber * mask
        loss = huber.sum() / (mask.sum() + 1e-8)
    else:
        loss = huber.mean()
    
    mean_error = error.mean().item()
    
    return {
        "loss": loss,
        "mean_error": mean_error
    }


def train_step(runtime: dict,
               batch: dict,
               delta: float = 1.0) -> dict:
    """
    Purpose:
        Perform one rectified-flow training iteration:
            1. Forward v_θ(X_t, t)
            2. Compute Huber loss
            3. Backpropagate and optimizer.step()

    Parameters:
        runtime (dict): {"theta", "optimizer", "device"}
        batch (dict): {"X_t", "t", "V", "mask" (optional)}
        delta (float): Huber threshold

    Return Dict:
        {"loss": float, "mean_error": float, "lr": float}
    """
    model = runtime["theta"]
    optimizer = runtime["optimizer"]
    device = runtime["device"]
    
    # Move batch to device
    X_t = batch["X_t"].to(device)  # (B, K, 2)
    t = batch["t"].to(device)  # (B, 1)
    V = batch["V"].to(device)  # (B, K, 2) - this is the velocity label
    mask = batch.get("mask", None)
    if mask is not None:
        mask = mask.to(device)
    
    # Forward pass
    model.train()
    optimizer.zero_grad()
    
    result = predict_velocity(model, X_t, t)
    v_theta = result["v_theta"]
    
    # Compute loss (V is already the velocity target from data)
    loss_result = compute_huber_loss(v_theta, V, delta=delta, mask=mask)
    loss = loss_result["loss"]
    mean_error = loss_result["mean_error"]
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping (prevent exploding gradients)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    if "scheduler" in runtime:
        runtime["scheduler"].step()
    return {
        "loss": loss.item(),
        "mean_error": mean_error,
        "lr": optimizer.param_groups[0]['lr']
    }


def train_loop(runtime: dict,
               dataloader,
               epoch: int,
               delta: float,
               save_dir: str,
               log_interval: int = 100,
               save_interval: int = 1000,
               model_name: str = "model") -> dict:
    """
    Purpose:
        Execute a full training epoch under rectified flow framework.
        Periodically log metrics, save checkpoints, and append validation stats.

    Return Dict:
        {"epoch": int, "total_loss": float, "avg_loss": float,
         "avg_error": float, "num_steps": int}
    """
    from pathlib import Path
    import time, sys, csv
    from datetime import datetime

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    log_dir = Path("./log") / model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "train_data.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time", "step (ckpt)", "avg_loss",
                "acc_mean", "acc_median", "acc_std",
                "lr", "huber_loss_delta"
            ])

    model = runtime["theta"]
    model.train()
    total_loss = total_error = 0.0
    num_steps = 0
    total_steps = len(dataloader)
    start_time = time.time()
    latest_checkpoint = "None"
    val_data = torch.load("./dataset/quick_val.pt")

    for step, batch in enumerate(dataloader, 1):
        result = train_step(runtime, batch, delta=delta)
        total_loss += result["loss"]
        total_error += result["mean_error"]
        num_steps += 1
        avg_loss = total_loss / num_steps
        avg_error = total_error / num_steps
        elapsed = time.time() - start_time
        steps_per_sec = num_steps / elapsed if elapsed > 0 else 0

        progress = step / total_steps
        filled = int(40 * progress)
        bar = "█" * filled + "░" * (40 - filled)
        sys.stdout.write("\r\033[K")
        sys.stdout.write(
            f"Epoch {epoch} | Step {step}/{total_steps} [{bar}] {progress*100:.1f}%\n"
        )
        sys.stdout.write("\r\033[K")
        sys.stdout.write(
            f"Loss: {result['loss']:.4f} | Avg: {avg_loss:.4f} | "
            f"Error: {avg_error:.4f} | Speed: {steps_per_sec:.1f} steps/s | "
            f"Checkpoint: {latest_checkpoint}"
        )
        sys.stdout.write("\033[F")
        sys.stdout.flush()

        if step % log_interval == 0:
            sys.stdout.write("\n\n")
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(
                f"{current_time} | Epoch {epoch} | Step {step}/{total_steps} | "
                f"Loss: {result['loss']:.4f} | Avg Loss: {avg_loss:.4f} | "
                f"Error: {result['mean_error']:.4f} | Avg Error: {avg_error:.4f} | "
                f"LR: {result['lr']:.6f} | Speed: {steps_per_sec:.2f} steps/s | "
                f"Checkpoint: {latest_checkpoint}"
            )

        # === Save checkpoint and run quick_acc_test ===
        if step % save_interval == 0:
            checkpoint_name = f"ckpt_{epoch}_{step}.pt"
            checkpoint_path = save_path / checkpoint_name
            metrics = quick_acc_test(model, val_data, runtime)
            torch.save({
                "epoch": epoch,
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": runtime["optimizer"].state_dict(),
                "loss": result["loss"],
                "avg_loss": avg_loss,
                "delta": delta,
                "model_name": model_name,
                "timestamp": datetime.now().isoformat()
            }, checkpoint_path)
            latest_checkpoint = checkpoint_name
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            log_line = (
                f"{current_time} | {checkpoint_name} | "
                f"avg_loss={avg_loss:.6f} | lr={result['lr']:.6e} | "
                f"acc_mean={metrics['mean']:.4f} | acc_med={metrics['median']:.4f} | "
                f"acc_std={metrics['std']:.4f} | step={step}"
            )
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    current_time,
                    checkpoint_name,
                    f"{avg_loss:.6f}",
                    f"{metrics['mean']:.6f}",
                    f"{metrics['median']:.6f}",
                    f"{metrics['std']:.6f}",
                    f"{result['lr']:.6f}",
                    f"{delta:.6f}"
                ])
            with open("./log/runtime.log", "a") as f:
                f.write(log_line + "\n")
            logger.info(log_line)

    sys.stdout.write("\n\n")
    avg_loss = total_loss / num_steps if num_steps else 0.0
    avg_error = total_error / num_steps if num_steps else 0.0
    checkpoint_name = f"ckpt_{epoch}_final.pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": runtime["optimizer"].state_dict(),
        "avg_loss": avg_loss,
        "avg_error": avg_error,
        "delta": delta
    }, save_path / checkpoint_name)
    logger.info(f"Saved epoch checkpoint: {save_path/checkpoint_name}")

    return {
        "epoch": epoch,
        "total_loss": total_loss,
        "avg_loss": avg_loss,
        "avg_error": avg_error,
        "num_steps": num_steps
    }


# ============================================================================
# Utility Functions
# ============================================================================

def count_parameters(model: nn.Module) -> dict:
    """
    Count total and trainable parameters in model.
    
    Returns:
        {
            "total": int,
            "trainable": int,
            "non_trainable": int
        }
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable
    }


if __name__ == "__main__":
    # Test model architectures
    logging.basicConfig(level=logging.INFO)
    
    K = 256
    batch_size = 4
    
    print("="*60)
    print("Testing theta Model Architectures")
    print("="*60)
    
    # Test data
    X_t = torch.randn(batch_size, K, 2)
    t = torch.rand(batch_size, 1)
    
    # Test each architecture
    for model_type in ["nn", "cnn1d", "transformer"]:
        print(f"\n{model_type.upper()} Architecture:")
        print("-"*60)
        
        # Build model
        result = build_theta(model_type, hidden=256, layers=4, K=K)
        model = result["model"]
        
        print(f"Parameters: {result['num_params']:,}")
        
        # Forward pass
        v_theta = model(X_t, t)
        print(f"Input shape: {X_t.shape}")
        print(f"Output shape: {v_theta.shape}")
        print(f"Output range: [{v_theta.min():.4f}, {v_theta.max():.4f}]")
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)