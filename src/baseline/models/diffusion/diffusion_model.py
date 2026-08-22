import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ================================================================
# === SinusoidalDiffusionEmbedding
# ================================================================
class SinusoidalDiffusionEmbedding(nn.Module):
    """
    Sinusoidal embedding for normalized diffusion steps.

    Input:
        t: Tensor(B,1), values in (0, 1]
    Output:
        Tensor(B, dim)
    """

    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: Tensor) -> Tensor:
        device = t.device
        half_dim = self.dim // 2
        freq_exponent = math.log(10000.0) / max(1, half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -freq_exponent)
        angles = t * freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb


# ================================================================
# === DiffusionHybridOnline
# ================================================================
class DiffusionHybridOnline(nn.Module):
    """
    Causal hybrid CNN/Transformer backbone for step-local diffusion.

    Input channel contract:
        X: Tensor(B,K,3) = [x_s(2), is_pad(1)]
        t: Tensor(B,1)   = normalized diffusion step

    Output:
        Tensor(B,K,2), predicted terminal-noise epsilon.
    """

    def __init__(
        self,
        K=256,
        coord_dim=2,
        input_coord_dim=3,
        hidden=384,
        cnn_layers=4,
        transf_layers=8,
        nhead=8,
        noise_dim=128,
        kernel_size=7,
        dropout=0.1,
    ):
        super().__init__()
        self.K = int(K)
        self.kernel_size = int(kernel_size)
        self.coord_dim = int(coord_dim)
        self.input_coord_dim = int(input_coord_dim)

        self.noise_embed = SinusoidalDiffusionEmbedding(noise_dim)
        self.noise_proj = nn.Linear(noise_dim, hidden)

        # Causal CNN front-end.
        self.input_proj = nn.Conv1d(self.input_coord_dim, hidden, kernel_size=1)

        self.cnn_blocks = nn.ModuleList()
        for _ in range(cnn_layers):
            self.cnn_blocks.append(
                nn.ModuleList(
                    [
                        nn.LayerNorm(hidden),
                        nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=0),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Conv1d(hidden, hidden, kernel_size=1),
                        nn.Dropout(dropout),
                    ]
                )
            )

        # Transformer stage sees both CNN features and raw diffusion input.
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

    def forward(self, X, t):
        B, K, C = X.shape
        del B
        if C != self.input_coord_dim:
            raise ValueError(
                f"DiffusionHybridOnline expected input_coord_dim={self.input_coord_dim}, got {C}"
            )

        t_emb = embed_diffusion_time(t, self.noise_embed, self.noise_proj)

        # Causal CNN stage.
        x = X.transpose(1, 2)
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

        # Causal Transformer stage with raw-input skip concat.
        x = x.transpose(1, 2)
        x = torch.cat([x, X], dim=-1)
        x = self.pre_transformer_proj(x)
        x = x + self.pos_embed[:, :K, :]
        x = x + t_emb.unsqueeze(1)
        attn_mask = self._causal_attn_mask(K, x.device)
        x = self.transformer(x, mask=attn_mask)

        return self.output_proj(x)


# ================================================================
# === Helpers
# ================================================================
def embed_diffusion_time(t: torch.Tensor, noise_embed, noise_proj):
    t_emb = noise_embed(t)
    t_emb = noise_proj(t_emb)
    t_emb = t_emb / math.sqrt(t_emb.shape[-1])
    return t_emb


def build_diffusion_model(runtime_or_config) -> nn.Module:
    """
    Build a diffusion baseline model from a runtime dict or raw config dict.
    """

    cfg = runtime_or_config.get("config", runtime_or_config)
    mt = str(cfg.get("model_type", "diffusion_hybrid_online")).strip().lower()

    if mt not in {
        "diffusion_hybrid_online",
        "hybrid_online",
        "online_hybrid",
        "causal_hybrid",
    }:
        raise ValueError(f"[build_diffusion_model] Unknown model_type: {cfg.get('model_type')}")

    return DiffusionHybridOnline(
        K=int(cfg.get("K", 256)),
        coord_dim=int(cfg.get("coord_dim", 2)),
        input_coord_dim=int(cfg.get("input_coord_dim", 3)),
        hidden=int(cfg.get("hidden", 384)),
        cnn_layers=int(cfg.get("cnn_layers", 4)),
        transf_layers=int(cfg.get("layers", cfg.get("transf_layers", 8))),
        nhead=int(cfg.get("nhead", 8)),
        noise_dim=int(cfg.get("noise_dim", 128)),
        kernel_size=int(cfg.get("kernel_size", 7)),
        dropout=float(cfg.get("dropout", 0.1)),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "DiffusionHybridOnline",
    "SinusoidalDiffusionEmbedding",
    "build_diffusion_model",
    "count_parameters",
]
