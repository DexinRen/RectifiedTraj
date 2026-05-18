import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ================================================================
# === SinusoidalNoiseEmbedding (low-level)
# ================================================================
class SinusoidalNoiseEmbedding(nn.Module):
    """
    Sinusoidal embedding for noise level conditioning in rectified flow.
    Maps scalar t ∈ [0, 1) to a fixed-dimensional embedding.
    """
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: (B, 1) noise level values in [0, 1)

        Returns:
            (B, dim) noise embeddings
        """
        device = t.device
        half_dim = self.dim // 2
        freq_exponent = math.log(10000.0) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -freq_exponent)
        angles = t * freqs[None, :]              # (B, half_dim)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb


class MLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class thetaMLP(nn.Module):
    def __init__(self, K=256, coord_dim=2, hidden=512,
                 layers=6, noise_dim=128, dropout=0.1):
        super().__init__()
        self.K = K
        self.input_norm = nn.LayerNorm(hidden)
        self.noise_embed = SinusoidalNoiseEmbedding(noise_dim)
        self.noise_proj  = nn.Linear(noise_dim, hidden)

        self.input_proj = nn.Linear(coord_dim, hidden)

        self.blocks = nn.ModuleList([
            MLPBlock(hidden, hidden * 4, dropout)
            for _ in range(layers)
        ])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, coord_dim),
        )

    def forward(self, X_t, t):
        # X_t: (B,K,2)
        # t:   (B,1)
        B, K, C = X_t.shape

        t_emb = embed_noise(t, self.noise_embed, self.noise_proj) # (B,hidden)
        x = self.input_proj(X_t)                                  # (B,K,hidden)
        x = self.input_norm(x)
        x = x + t_emb.unsqueeze(1)                                # broadcast

        for block in self.blocks:
            x = block(x)

        v = self.output_proj(x)
        return v   # (B,K,2)


class thetaCNN1D(nn.Module):
    def __init__(self, K=256, coord_dim=2, hidden=256,
                 layers=8, kernel_size=7, noise_dim=128, dropout=0.1):
        super().__init__()
        self.K = K

        self.noise_embed = SinusoidalNoiseEmbedding(noise_dim)
        self.noise_proj  = nn.Linear(noise_dim, hidden)

        self.input_proj = nn.Conv1d(coord_dim, hidden, kernel_size=1)

        self.cnn_blocks = nn.ModuleList()
        for _ in range(layers):
            self.cnn_blocks.append(
                nn.ModuleList([
                    nn.LayerNorm(hidden),
                    nn.Conv1d(hidden, hidden,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(hidden, hidden, kernel_size=1),
                    nn.Dropout(dropout),
                ])
            )

        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, coord_dim, kernel_size=1)
        )

    def forward(self, X_t, t):
        # X_t: (B,K,2)  → (B,2,K)
        B, K, C = X_t.shape

        t_emb = embed_noise(t, self.noise_embed, self.noise_proj) # (B,hidden)

        x = X_t.transpose(1,2)
        x = self.input_proj(x)                     # (B,hidden,K)
        x = x + t_emb[:, :, None]                  # inject noise

        for norm, conv1, act, drop1, conv2, drop2 in self.cnn_blocks:
            residual = x
            xn = norm(x.transpose(1,2)).transpose(1,2)
            x = conv1(xn)
            x = act(x)
            x = drop1(x)
            x = conv2(x)
            x = drop2(x)
            x = x + residual

        x = self.output_proj(x)                    # (B,2,K)
        return x.transpose(1,2)                    # (B,K,2)


class thetaTransformer(nn.Module):
    def __init__(self, K=256, coord_dim=2, hidden=256,
                 layers=6, nhead=8, noise_dim=128, dropout=0.1):
        super().__init__()
        self.K = K

        self.noise_embed = SinusoidalNoiseEmbedding(noise_dim)
        self.noise_proj  = nn.Linear(noise_dim, hidden)

        self.input_proj = nn.Linear(coord_dim, hidden)

        self.pos_embed = nn.Parameter(torch.randn(1, K, hidden) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, coord_dim)
        )

    def forward(self, X_t, t):
        B, K, C = X_t.shape

        t_emb = embed_noise(t, self.noise_embed, self.noise_proj) # (B,hidden)

        x = self.input_proj(X_t)                                 # (B,K,hidden)
        x = x + self.pos_embed[:, :K, :]
        x = x + t_emb.unsqueeze(1)

        x = self.transformer(x)
        v = self.output_proj(x)
        return v


class thetaHybridCNNTransformer(nn.Module):
    def __init__(self,
                 K=256, coord_dim=2, hidden=384,
                 cnn_layers=4, transf_layers=8,
                 nhead=8, noise_dim=128,
                 kernel_size=7, dropout=0.1):
        super().__init__()
        self.K = K

        self.noise_embed = SinusoidalNoiseEmbedding(noise_dim)
        self.noise_proj  = nn.Linear(noise_dim, hidden)

        # CNN front-end
        self.input_proj = nn.Conv1d(coord_dim, hidden, kernel_size=1)

        self.cnn_blocks = nn.ModuleList()
        for _ in range(cnn_layers):
            self.cnn_blocks.append(
                nn.ModuleList([
                    nn.LayerNorm(hidden),
                    nn.Conv1d(hidden, hidden,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(hidden, hidden, kernel_size=1),
                    nn.Dropout(dropout),
                ])
            )

        # Transformer back-end
        self.pos_embed = nn.Parameter(torch.randn(1, K, hidden) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=transf_layers)

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, coord_dim),
        )

    def forward(self, X_t, t):
        B, K, C = X_t.shape

        t_emb = embed_noise(t, self.noise_embed, self.noise_proj) # (B,hidden)

        # CNN stage
        x = X_t.transpose(1,2)
        x = self.input_proj(x)
        x = x + t_emb[:, :, None]

        for norm, conv1, act, drop1, conv2, drop2 in self.cnn_blocks:
            residual = x
            xn = norm(x.transpose(1,2)).transpose(1,2)
            x = conv1(xn)
            x = act(x)
            x = drop1(x)
            x = conv2(x)
            x = drop2(x)
            x = x + residual

        # Transformer stage
        x = x.transpose(1,2)
        x = x + self.pos_embed[:, :K, :]
        x = x + t_emb.unsqueeze(1)
        x = self.transformer(x)

        v = self.output_proj(x)
        return v


class thetaCNN1DOnline(nn.Module):
    def __init__(self,
                 K=256, coord_dim=2, input_coord_dim=None, hidden=384,
                 layers=8, kernel_size=7, noise_dim=128, dropout=0.1):
        super().__init__()
        self.K = K
        self.kernel_size = kernel_size
        self.input_coord_dim = coord_dim if input_coord_dim is None else int(input_coord_dim)

        self.noise_embed = SinusoidalNoiseEmbedding(noise_dim)
        self.noise_proj = nn.Linear(noise_dim, hidden)

        self.input_proj = nn.Conv1d(self.input_coord_dim, hidden, kernel_size=1)

        self.cnn_blocks = nn.ModuleList()
        for _ in range(layers):
            self.cnn_blocks.append(
                nn.ModuleList([
                    nn.LayerNorm(hidden),
                    nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=0),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(hidden, hidden, kernel_size=1),
                    nn.Dropout(dropout),
                ])
            )

        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, coord_dim, kernel_size=1),
        )

    def _causal_conv1d(self, conv: nn.Conv1d, x: Tensor) -> Tensor:
        left_pad = conv.kernel_size[0] - 1
        if left_pad > 0:
            x = F.pad(x, (left_pad, 0))
        return conv(x)

    def forward(self, X_t, t):
        B, K, C = X_t.shape

        t_emb = embed_noise(t, self.noise_embed, self.noise_proj)  # (B,hidden)

        x = X_t.transpose(1, 2)
        x = self.input_proj(x)
        x = x + t_emb[:, :, None]

        for norm, conv1, act, drop1, conv2, drop2 in self.cnn_blocks:
            residual = x
            xn = norm(x.transpose(1, 2)).transpose(1, 2)
            x = self._causal_conv1d(conv1, xn)
            x = act(x)
            x = drop1(x)
            x = conv2(x)
            x = drop2(x)
            x = x + residual

        x = self.output_proj(x)
        return x.transpose(1, 2)


class thetaTransformerOnline(nn.Module):
    def __init__(self,
                 K=256, coord_dim=2, input_coord_dim=None, hidden=256,
                 layers=6, nhead=8, noise_dim=128, dropout=0.1):
        super().__init__()
        self.K = K
        self.input_coord_dim = coord_dim if input_coord_dim is None else int(input_coord_dim)

        self.noise_embed = SinusoidalNoiseEmbedding(noise_dim)
        self.noise_proj = nn.Linear(noise_dim, hidden)

        self.input_proj = nn.Linear(self.input_coord_dim, hidden)
        self.pos_embed = nn.Parameter(torch.randn(1, K, hidden) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, coord_dim),
        )

    def _causal_attn_mask(self, K: int, device) -> Tensor:
        return torch.triu(
            torch.full((K, K), float("-inf"), device=device),
            diagonal=1,
        )

    def forward(self, X_t, t):
        B, K, C = X_t.shape

        t_emb = embed_noise(t, self.noise_embed, self.noise_proj)  # (B,hidden)

        x = self.input_proj(X_t)
        x = x + self.pos_embed[:, :K, :]
        x = x + t_emb.unsqueeze(1)
        attn_mask = self._causal_attn_mask(K, x.device)
        x = self.transformer(x, mask=attn_mask)

        v = self.output_proj(x)
        return v


class thetaHybridOnline(nn.Module):
    def __init__(self,
                 K=256, coord_dim=2, input_coord_dim=None, hidden=384,
                 cnn_layers=4, transf_layers=8,
                 nhead=8, noise_dim=128,
                 kernel_size=7, dropout=0.1):
        super().__init__()
        self.K = K
        self.kernel_size = kernel_size
        self.input_coord_dim = coord_dim if input_coord_dim is None else int(input_coord_dim)

        self.noise_embed = SinusoidalNoiseEmbedding(noise_dim)
        self.noise_proj = nn.Linear(noise_dim, hidden)

        # Causal CNN front-end
        self.input_proj = nn.Conv1d(self.input_coord_dim, hidden, kernel_size=1)

        self.cnn_blocks = nn.ModuleList()
        for _ in range(cnn_layers):
            self.cnn_blocks.append(
                nn.ModuleList([
                    nn.LayerNorm(hidden),
                    nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=0),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(hidden, hidden, kernel_size=1),
                    nn.Dropout(dropout),
                ])
            )

        # Expose both learned CNN features and the raw input to the transformer.
        self.pre_transformer_proj = nn.Linear(hidden + self.input_coord_dim, hidden)
        self.pos_embed = nn.Parameter(torch.randn(1, K, hidden) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=transf_layers)

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, coord_dim),
        )

    def _causal_conv1d(self, conv: nn.Conv1d, x: Tensor) -> Tensor:
        left_pad = conv.kernel_size[0] - 1
        if left_pad > 0:
            x = F.pad(x, (left_pad, 0))
        return conv(x)

    def _causal_attn_mask(self, K: int, device) -> Tensor:
        return torch.triu(
            torch.full((K, K), float("-inf"), device=device),
            diagonal=1,
        )

    def forward(self, X_t, t):
        B, K, C = X_t.shape

        t_emb = embed_noise(t, self.noise_embed, self.noise_proj)  # (B,hidden)

        # Causal CNN stage
        x = X_t.transpose(1, 2)
        x = self.input_proj(x)
        x = x + t_emb[:, :, None]

        for norm, conv1, act, drop1, conv2, drop2 in self.cnn_blocks:
            residual = x
            xn = norm(x.transpose(1, 2)).transpose(1, 2)
            x = self._causal_conv1d(conv1, xn)
            x = act(x)
            x = drop1(x)
            x = conv2(x)
            x = drop2(x)
            x = x + residual

        # Transformer stage with causal attention and raw-input skip concat
        x = x.transpose(1, 2)
        x = torch.cat([x, X_t], dim=-1)
        x = self.pre_transformer_proj(x)
        x = x + self.pos_embed[:, :K, :]
        x = x + t_emb.unsqueeze(1)
        attn_mask = self._causal_attn_mask(K, x.device)
        x = self.transformer(x, mask=attn_mask)

        v = self.output_proj(x)
        return v


class thetaHybridAlt(nn.Module):
    def __init__(self,
                 K=256, coord_dim=2, input_coord_dim=None, hidden=384,
                 cnn_layers=4, transf_layers=8,
                 nhead=8, noise_dim=128,
                 kernel_size=7, dropout=0.1):
        super().__init__()
        self.K = K
        self.kernel_size = kernel_size
        self.input_coord_dim = coord_dim if input_coord_dim is None else int(input_coord_dim)

        self.noise_embed = SinusoidalNoiseEmbedding(noise_dim)
        self.noise_proj = nn.Linear(noise_dim, hidden)

        # Causal CNN front-end
        self.input_proj = nn.Conv1d(self.input_coord_dim, hidden, kernel_size=1)

        self.cnn_blocks = nn.ModuleList()
        for _ in range(cnn_layers):
            self.cnn_blocks.append(
                nn.ModuleList([
                    nn.LayerNorm(hidden),
                    nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=0),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(hidden, hidden, kernel_size=1),
                    nn.Dropout(dropout),
                ])
            )

        self.pos_embed = nn.Parameter(torch.randn(1, K, hidden) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=transf_layers)

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, coord_dim),
        )

    def _causal_conv1d(self, conv: nn.Conv1d, x: Tensor) -> Tensor:
        left_pad = conv.kernel_size[0] - 1
        if left_pad > 0:
            x = F.pad(x, (left_pad, 0))
        return conv(x)

    def _causal_attn_mask(self, K: int, device) -> Tensor:
        return torch.triu(
            torch.full((K, K), float("-inf"), device=device),
            diagonal=1,
        )

    def forward(self, X_t, t):
        B, K, C = X_t.shape

        t_emb = embed_noise(t, self.noise_embed, self.noise_proj)  # (B,hidden)

        # Causal CNN stage
        x = X_t.transpose(1, 2)
        x = self.input_proj(x)
        x = x + t_emb[:, :, None]

        for norm, conv1, act, drop1, conv2, drop2 in self.cnn_blocks:
            residual = x
            xn = norm(x.transpose(1, 2)).transpose(1, 2)
            x = self._causal_conv1d(conv1, xn)
            x = act(x)
            x = drop1(x)
            x = conv2(x)
            x = drop2(x)
            x = x + residual

        # Transformer stage with causal attention on CNN features only.
        x = x.transpose(1, 2)
        x = x + self.pos_embed[:, :K, :]
        x = x + t_emb.unsqueeze(1)
        attn_mask = self._causal_attn_mask(K, x.device)
        x = self.transformer(x, mask=attn_mask)

        v = self.output_proj(x)
        return v


# ================================================================
# === build_theta_model(runtime)
# ================================================================
def build_theta_model(runtime) -> nn.Module:
    """
    Purpose:
        Factory that constructs a theta model using ONLY runtime.

    Required runtime keys:
        runtime["config"]["model_type"]
        runtime["config"]["K"]
        runtime["config"]["coord_dim"]
        runtime["config"]["hidden"]
        runtime["config"]["layers"]

    Optional runtime keys:
        kernel_size, nhead, noise_dim, dropout, cnn_layers
    """

    cfg = runtime["config"]
    mt  = cfg["model_type"].lower()

    K          = cfg["K"]
    coord_dim  = cfg["coord_dim"]
    input_coord_dim = cfg.get("input_coord_dim", coord_dim)
    hidden     = cfg["hidden"]
    layers     = cfg["layers"]

    # optional hyperparams
    noise_dim   = cfg.get("noise_dim", 128)
    dropout     = cfg.get("dropout", 0.1)
    kernel_size = cfg.get("kernel_size", 7)
    nhead       = cfg.get("nhead", 8)
    cnn_layers  = cfg.get("cnn_layers", 4)

    if mt == "mlp":
        return thetaMLP(
            K=K,
            coord_dim=coord_dim,
            hidden=hidden,
            layers=layers,
            noise_dim=noise_dim,
            dropout=dropout,
        )

    elif mt in ["cnn1d", "cnn"]:
        return thetaCNN1D(
            K=K,
            coord_dim=coord_dim,
            hidden=hidden,
            layers=layers,
            kernel_size=kernel_size,
            noise_dim=noise_dim,
            dropout=dropout,
        )

    elif mt in ["transformer", "transf"]:
        return thetaTransformer(
            K=K,
            coord_dim=coord_dim,
            hidden=hidden,
            layers=layers,
            nhead=nhead,
            noise_dim=noise_dim,
            dropout=dropout,
        )

    elif mt in ["hybrid", "cnn_transformer", "cnn+transformer"]:
        return thetaHybridCNNTransformer(
            K=K,
            coord_dim=coord_dim,
            hidden=hidden,
            cnn_layers=cnn_layers,
            transf_layers=layers,
            nhead=nhead,
            noise_dim=noise_dim,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    elif mt in ["cnn_online", "online_cnn", "causal_cnn"]:
        return thetaCNN1DOnline(
            K=K,
            coord_dim=coord_dim,
            input_coord_dim=input_coord_dim,
            hidden=hidden,
            layers=layers,
            kernel_size=kernel_size,
            noise_dim=noise_dim,
            dropout=dropout,
        )

    elif mt in ["transformer_online", "online_transformer", "causal_transformer"]:
        return thetaTransformerOnline(
            K=K,
            coord_dim=coord_dim,
            input_coord_dim=input_coord_dim,
            hidden=hidden,
            layers=layers,
            nhead=nhead,
            noise_dim=noise_dim,
            dropout=dropout,
        )

    elif mt in ["hybrid_online", "online_hybrid", "causal_hybrid"]:
        return thetaHybridOnline(
            K=K,
            coord_dim=coord_dim,
            input_coord_dim=input_coord_dim,
            hidden=hidden,
            cnn_layers=cnn_layers,
            transf_layers=layers,
            nhead=nhead,
            noise_dim=noise_dim,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    elif mt in ["hybrid_alt", "online_hybrid_alt", "causal_hybrid_alt"]:
        return thetaHybridAlt(
            K=K,
            coord_dim=coord_dim,
            input_coord_dim=input_coord_dim,
            hidden=hidden,
            cnn_layers=cnn_layers,
            transf_layers=layers,
            nhead=nhead,
            noise_dim=noise_dim,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    else:
        raise ValueError(f"[build_theta_model] Unknown model_type: {cfg['model_type']}")


# ================================================================
# === count_parameters
# ================================================================
def count_parameters(model: nn.Module) -> int:
    """
    Purpose:
        Count trainable parameters inside the model.

    Input:
        model: nn.Module

    Output:
        int: number of parameters requiring gradients
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ================================================================
# === Shared Noise Embedding Helper
# ================================================================
def embed_noise(t: torch.Tensor, noise_embed, noise_proj):
    t_emb = noise_embed(t)        # (B, noise_dim)
    t_emb = noise_proj(t_emb)     # (B, hidden)

    # --- Normalize to prevent overwhelming the features ---
    # Scale by sqrt(hidden) so magnitude matches typical token embeddings.
    t_emb = t_emb / math.sqrt(t_emb.shape[-1])

    return t_emb
