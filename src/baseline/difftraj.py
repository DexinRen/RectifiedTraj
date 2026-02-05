from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional, Union

import numpy as np


@dataclass
class DiffTrajPaths:
    repo_dir: Optional[str] = None
    checkpoint_path: Optional[str] = None


def _default_checkpoint_path() -> str:
    import os

    return os.path.join("bin", "baseline_model", "difftraj", "model.pt")


def _resolve_repo_dir(repo_dir: Optional[str]) -> str:
    import os

    if repo_dir:
        return repo_dir
    env = os.getenv("DIFFTRAJ_REPO")
    if env:
        return env
    raise FileNotFoundError(
        "DiffTraj repo dir not set. Provide DiffTrajPaths(repo_dir=...) or set DIFFTRAJ_REPO."
    )


def _resolve_checkpoint_path(checkpoint_path: Optional[str]) -> str:
    import os

    if checkpoint_path:
        return checkpoint_path
    default_path = _default_checkpoint_path()
    if os.path.exists(default_path):
        return default_path
    print(
        "DiffTraj checkpoint not found. Please download the model weight from GitHub and place it under "
        f"{default_path}."
    )
    print(
        "Download URL (example from official repo):\n"
        "https://github.com/Yasoz/DiffTraj/blob/main/model.pt"
    )
    raise FileNotFoundError("DiffTraj checkpoint not found.")


def load_difftraj_config(repo_dir: str) -> SimpleNamespace:
    import sys

    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    from utils.config import args  # type: ignore

    temp = {k: SimpleNamespace(**v) for k, v in args.items()}
    return SimpleNamespace(**temp)


def load_difftraj_model(config: SimpleNamespace, checkpoint_path: str, device: str):
    import torch

    from utils.Traj_UNet import Guide_UNet  # type: ignore

    model = Guide_UNet(config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _infer_head_dim(config: SimpleNamespace, fallback: int = 8) -> int:
    model_cfg = getattr(config, "model", None)
    for key in ("head_dim", "head_dims", "head", "guide_dim"):
        if model_cfg is not None and hasattr(model_cfg, key):
            value = getattr(model_cfg, key)
            if isinstance(value, int) and value > 0:
                return value
    return fallback


def _prepare_head(
    head: Optional[Union[np.ndarray, "torch.Tensor"]],
    batch: int,
    head_dim: int,
    device: str,
):
    import torch

    if head is None:
        return torch.ones(batch, head_dim, device=device)
    if isinstance(head, np.ndarray):
        head = torch.from_numpy(head)
    head = head.to(device=device, dtype=torch.float32)
    if head.ndim == 1:
        head = head.unsqueeze(0)
    if head.shape[0] != batch:
        if head.shape[0] == 1:
            head = head.repeat(batch, 1)
        else:
            raise ValueError(f"head batch {head.shape[0]} != {batch}")
    return head


def difftraj_denoise(
    positions: np.ndarray,
    paths: DiffTrajPaths,
    *,
    head: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    device: str = "cuda",
    timesteps: int = 100,
    final_steps: Optional[int] = None,
    eta: float = 0.0,
) -> np.ndarray:
    """Run DiffTraj denoising on 2D trajectories.

    positions: (T, 2) or (B, T, 2)
    paths.repo_dir: path to DiffTraj repo
    paths.checkpoint_path: model checkpoint (.pt)
    """
    import torch
    from utils.utils import compute_alpha  # type: ignore

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    repo_dir = _resolve_repo_dir(paths.repo_dir)
    checkpoint_path = _resolve_checkpoint_path(paths.checkpoint_path)

    config = load_difftraj_config(repo_dir)
    model = load_difftraj_model(config, checkpoint_path, device)

    input_ndim = np.asarray(positions).ndim
    pos = np.asarray(positions, dtype=np.float32)
    if pos.ndim == 2:
        pos = pos[None, ...]
    if pos.ndim != 3 or pos.shape[2] != 2:
        raise ValueError("positions must be shape (T,2) or (B,T,2)")

    x = torch.from_numpy(pos).to(device)
    x = x.transpose(1, 2)  # (B, 2, T)
    batch = x.shape[0]

    head_dim = _infer_head_dim(config)
    head_tensor = _prepare_head(head, batch, head_dim, device)

    n_steps = int(config.diffusion.num_diffusion_timesteps)
    beta = torch.linspace(
        float(config.diffusion.beta_start),
        float(config.diffusion.beta_end),
        n_steps,
        device=device,
    )

    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    skip = max(1, n_steps // timesteps)
    seq = list(range(0, n_steps, skip))
    seq_next = [-1] + list(seq[:-1])

    if final_steps is not None:
        seq = seq[:final_steps]
        seq_next = seq_next[:final_steps]

    with torch.no_grad():
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = torch.full((batch,), i, device=device)
            next_t = torch.full((batch,), j, device=device)
            pred_noise = model(x, t, head_tensor)
            at = compute_alpha(beta, t.long())
            at_next = compute_alpha(beta, next_t.long())
            x0_t = (x - pred_noise * (1 - at).sqrt()) / at.sqrt()
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1**2).sqrt()
            eps = torch.randn_like(x)
            x = at_next.sqrt() * x0_t + c1 * eps + c2 * pred_noise

    out = x.transpose(1, 2).cpu().numpy()
    return out[0] if input_ndim == 2 else out


def prepare_difftraj(
    paths: DiffTrajPaths,
    *,
    device: str = "cuda",
) -> tuple[SimpleNamespace, "torch.nn.Module", str]:
    """Load DiffTraj config/model once for repeated denoising."""
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    repo_dir = _resolve_repo_dir(paths.repo_dir)
    checkpoint_path = _resolve_checkpoint_path(paths.checkpoint_path)
    config = load_difftraj_config(repo_dir)
    model = load_difftraj_model(config, checkpoint_path, device)
    return config, model, device


def difftraj_denoise_with_model(
    positions: np.ndarray,
    *,
    config: SimpleNamespace,
    model,
    device: str,
    head: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    timesteps: int = 100,
    final_steps: Optional[int] = None,
    eta: float = 0.0,
) -> np.ndarray:
    """Run DiffTraj denoising with a preloaded config/model."""
    import torch
    from utils.utils import compute_alpha  # type: ignore

    input_ndim = np.asarray(positions).ndim
    pos = np.asarray(positions, dtype=np.float32)
    if pos.ndim == 2:
        pos = pos[None, ...]
    if pos.ndim != 3 or pos.shape[2] != 2:
        raise ValueError("positions must be shape (T,2) or (B,T,2)")

    x = torch.from_numpy(pos).to(device)
    x = x.transpose(1, 2)  # (B, 2, T)
    batch = x.shape[0]

    head_dim = _infer_head_dim(config)
    head_tensor = _prepare_head(head, batch, head_dim, device)

    n_steps = int(config.diffusion.num_diffusion_timesteps)
    beta = torch.linspace(
        float(config.diffusion.beta_start),
        float(config.diffusion.beta_end),
        n_steps,
        device=device,
    )

    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    skip = max(1, n_steps // timesteps)
    seq = list(range(0, n_steps, skip))
    seq_next = [-1] + list(seq[:-1])

    if final_steps is not None:
        seq = seq[:final_steps]
        seq_next = seq_next[:final_steps]

    with torch.no_grad():
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = torch.full((batch,), i, device=device)
            next_t = torch.full((batch,), j, device=device)
            pred_noise = model(x, t, head_tensor)
            at = compute_alpha(beta, t.long())
            at_next = compute_alpha(beta, next_t.long())
            x0_t = (x - pred_noise * (1 - at).sqrt()) / at.sqrt()
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1**2).sqrt()
            eps = torch.randn_like(x)
            x = at_next.sqrt() * x0_t + c1 * eps + c2 * pred_noise

    out = x.transpose(1, 2).cpu().numpy()
    return out[0] if input_ndim == 2 else out


__all__ = [
    "DiffTrajPaths",
    "load_difftraj_config",
    "load_difftraj_model",
    "difftraj_denoise",
    "prepare_difftraj",
    "difftraj_denoise_with_model",
]
