#!/usr/bin/env python3
"""
Diffusion data loader for the learned baseline family.

The existing NUMOSIM training shards store rectified-flow triplets:
    X_t, V, t

This loader recovers the paired denoising data:
    x0 = clean trajectory
    x1 = observed noisy trajectory

Then it treats the observed noisy endpoint as the terminal diffusion state and
builds step-local diffusion batches:
    model_input = [x_s, is_pad]
    target      = epsilon
    time        = normalized diffusion step in (0, 1]
"""

from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


# ================================================================
# === DiffusionBatch
# ================================================================
@dataclass
class DiffusionBatch:
    """
    Batch object returned by DiffusionDataLoader.get_batch().

    model_input:
        Tensor(B,K,3) with [x_s(2), is_pad(1)].
    target:
        Tensor(B,K,2), terminal-noise epsilon.
    diffusion_t:
        Tensor(B,1), normalized diffusion step.
    valid_mask:
        Tensor(B,K), 1 for real positions and 0 for artificial startup pad.
    """

    model_input: torch.Tensor
    target: torch.Tensor
    diffusion_t: torch.Tensor
    valid_mask: torch.Tensor
    x0: torch.Tensor
    x1: torch.Tensor
    x_s: torch.Tensor
    noise: torch.Tensor
    step_index: torch.Tensor


# ================================================================
# === Helpers: Normalization
# ================================================================
def _normalize_loader_mode(raw: object) -> str:
    token = str(raw if raw is not None else "").strip().lower()
    if token in {"train", "training"}:
        return "train"
    if token in {"eval", "val", "valid", "validation", "test"}:
        return "eval"
    raise ValueError(f"Unsupported diffusion loader mode: {raw!r}. Use train or eval.")


def _normalize_prediction_mode(raw: object, default: str = "online") -> str:
    token = str(raw if raw is not None else "").strip().lower().replace("-", "_")
    if token in {"", "online", "causal", "streaming"}:
        return "online"
    if token in {"offline", "batch", "global"}:
        return "offline"
    text = str(raw).strip() if raw is not None else ""
    return text if text else str(default)


def _normalize_prediction_type(raw: object, default: str = "epsilon") -> str:
    token = str(raw if raw is not None else "").strip().lower().replace("-", "_")
    if token in {"", "eps", "epsilon", "noise"}:
        return "epsilon"
    if raw is None:
        return str(default)
    raise ValueError(f"Diffusion baseline is epsilon-only; got prediction_type={raw!r}.")


# ================================================================
# === Helpers: Existing Shard Schema
# ================================================================
def recover_x0_x1_from_rf_triplets(
    x_t: torch.Tensor,
    v: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Recover clean/noisy endpoint pairs from stored rectified-flow tensors.

    Stored RF convention:
        X_t = (1 - t) * x0 + t * x1
        V   = x1 - x0

    Therefore:
        x0 = X_t - V * t
        x1 = X_t + V * (1 - t)
    """

    x_t_xy = x_t[:, :, :2].to(dtype=torch.float32)
    v_xy = v[:, :, :2].to(device=x_t_xy.device, dtype=x_t_xy.dtype)
    t_view = t.reshape(-1, 1, 1).to(device=x_t_xy.device, dtype=x_t_xy.dtype)

    x0 = x_t_xy - v_xy * t_view
    x1 = x_t_xy + v_xy * (1.0 - t_view)
    return x0, x1


def load_x0_x1_from_pack(pack: dict, file_path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load clean/noisy endpoint pairs from either:
        - explicit X0/X1 chunk-pair files
        - RF train-triplet files with X_t/V/t
    """

    if {"X0", "X1"}.issubset(pack.keys()):
        x0 = pack["X0"][:, :, :2].to(dtype=torch.float32)
        x1 = pack["X1"][:, :, :2].to(dtype=torch.float32)
        return x0, x1

    if {"X_t", "V", "t"}.issubset(pack.keys()):
        return recover_x0_x1_from_rf_triplets(
            pack["X_t"],
            pack["V"],
            pack["t"],
        )

    raise KeyError(
        f"Diffusion loader expects X0/X1 or X_t/V/t in {file_path}; "
        f"got keys={sorted(pack.keys())}"
    )


# ================================================================
# === Helpers: Window Construction
# ================================================================
def _build_window_pair(
    x0_raw: torch.Tensor,
    x1_raw: torch.Tensor,
    *,
    target_k: int,
    start_idx: torch.Tensor,
    real_len: torch.Tensor,
    startup_pad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build local online windows from clean/noisy endpoint pairs.

    Coordinates are rebased by the first noisy observation in the window. The
    same origin is applied to x0 and x1 so both tensors stay in one local ENU
    frame.
    """

    n_rows = int(x0_raw.shape[0])
    raw_k = int(x0_raw.shape[1])
    if target_k > raw_k:
        raise ValueError(f"target_k={target_k} exceeds raw_k={raw_k}")

    x0_xy = x0_raw[:, :, :2]
    x1_xy = x1_raw[:, :, :2].to(device=x0_xy.device, dtype=x0_xy.dtype)

    pos = torch.arange(target_k, dtype=torch.long, device=x0_xy.device).view(1, -1)
    pad_count = (target_k - real_len).view(-1, 1)
    rel = pos - pad_count
    rel = torch.clamp(rel, min=0)
    max_rel = (real_len - 1).view(-1, 1)
    rel = torch.minimum(rel, max_rel)
    gather_idx = start_idx.view(-1, 1) + rel

    gather_xy = gather_idx.unsqueeze(-1).expand(-1, -1, 2)
    x0_sub = torch.gather(x0_xy, 1, gather_xy)
    x1_sub = torch.gather(x1_xy, 1, gather_xy)

    row_idx = torch.arange(n_rows, device=x0_xy.device)
    origin = x1_xy[row_idx, start_idx, :].unsqueeze(1)
    x0_sub = x0_sub - origin
    x1_sub = x1_sub - origin

    if startup_pad:
        valid_mask = (pos >= pad_count).to(dtype=x0_xy.dtype)
    else:
        valid_mask = torch.ones((n_rows, target_k), dtype=x0_xy.dtype, device=x0_xy.device)
    is_pad = (1.0 - valid_mask).unsqueeze(-1)

    return x0_sub, x1_sub, is_pad, valid_mask


def build_diffusion_online_train_pairs(
    x0_raw: torch.Tensor,
    x1_raw: torch.Tensor,
    *,
    target_k: int,
    startup_pad_prob: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly sub-dice stored chunks into causal online training windows."""

    n_rows = int(x0_raw.shape[0])
    raw_k = int(x0_raw.shape[1])
    if target_k > raw_k:
        raise ValueError(f"target_k={target_k} exceeds raw_k={raw_k}")

    startup_mask = torch.rand(n_rows, device=x0_raw.device) < float(startup_pad_prob)
    real_len = torch.full((n_rows,), target_k, dtype=torch.long, device=x0_raw.device)
    if bool(startup_mask.any()):
        startup_count = int(startup_mask.sum().item())
        real_len[startup_mask] = torch.randint(
            low=1,
            high=target_k,
            size=(startup_count,),
            device=x0_raw.device,
        )

    start_full = torch.randint(
        low=0,
        high=max(1, raw_k - target_k + 1),
        size=(n_rows,),
        device=x0_raw.device,
    )
    max_start_pad = raw_k - real_len
    start_pad = torch.floor(
        torch.rand(n_rows, device=x0_raw.device)
        * (max_start_pad.to(dtype=torch.float32) + 1.0)
    ).to(dtype=torch.long)
    start_idx = torch.where(startup_mask, start_pad, start_full)

    return _build_window_pair(
        x0_raw,
        x1_raw,
        target_k=target_k,
        start_idx=start_idx,
        real_len=real_len,
        startup_pad=True,
    )


def build_diffusion_online_eval_pairs(
    x0_raw: torch.Tensor,
    x1_raw: torch.Tensor,
    *,
    target_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministically extract tail windows for quick validation."""

    n_rows = int(x0_raw.shape[0])
    raw_k = int(x0_raw.shape[1])
    if target_k > raw_k:
        raise ValueError(f"target_k={target_k} exceeds raw_k={raw_k}")

    start_idx = torch.full(
        (n_rows,),
        raw_k - target_k,
        dtype=torch.long,
        device=x0_raw.device,
    )
    real_len = torch.full(
        (n_rows,),
        target_k,
        dtype=torch.long,
        device=x0_raw.device,
    )

    return _build_window_pair(
        x0_raw,
        x1_raw,
        target_k=target_k,
        start_idx=start_idx,
        real_len=real_len,
        startup_pad=False,
    )


def build_diffusion_offline_pairs(
    x0_raw: torch.Tensor,
    x1_raw: torch.Tensor,
    *,
    target_k: int,
    train_mode: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build fixed-length non-causal windows.

    The diffusion baseline is expected to use online mode for fair comparison,
    but this helper keeps the loader usable for offline ablations.
    """

    n_rows = int(x0_raw.shape[0])
    raw_k = int(x0_raw.shape[1])
    if target_k > raw_k:
        raise ValueError(f"target_k={target_k} exceeds raw_k={raw_k}")

    if train_mode:
        start_idx = torch.randint(
            low=0,
            high=max(1, raw_k - target_k + 1),
            size=(n_rows,),
            device=x0_raw.device,
        )
    else:
        start_idx = torch.full(
            (n_rows,),
            raw_k - target_k,
            dtype=torch.long,
            device=x0_raw.device,
        )
    real_len = torch.full((n_rows,), target_k, dtype=torch.long, device=x0_raw.device)

    return _build_window_pair(
        x0_raw,
        x1_raw,
        target_k=target_k,
        start_idx=start_idx,
        real_len=real_len,
        startup_pad=False,
    )


# ================================================================
# === Helpers: Diffusion Schedule
# ================================================================
class DiffusionNoiseSchedule:
    """
    Precomputed DDPM noising schedule.

    step_index is zero-based. The model receives normalized diffusion_t:
        diffusion_t = (step_index + 1) / num_steps
    """

    def __init__(
        self,
        *,
        num_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule_type: str = "linear",
    ) -> None:
        self.num_steps = int(num_steps)
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")

        token = str(schedule_type).strip().lower()
        if token == "linear":
            betas = torch.linspace(float(beta_start), float(beta_end), self.num_steps)
        elif token == "cosine":
            betas = self._build_cosine_betas(self.num_steps)
        else:
            raise ValueError(f"Unsupported diffusion schedule_type={schedule_type!r}")

        betas = torch.clamp(betas.to(dtype=torch.float32), min=1e-8, max=0.999)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    @staticmethod
    def _build_cosine_betas(num_steps: int, offset: float = 0.008) -> torch.Tensor:
        steps = torch.arange(num_steps + 1, dtype=torch.float32)
        x = steps / float(num_steps)
        alpha_bar = torch.cos(((x + offset) / (1.0 + offset)) * torch.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
        return torch.clamp(betas, min=1e-8, max=0.999)

    def to(self, device: torch.device | str) -> "DiffusionNoiseSchedule":
        device = torch.device(device)
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.sqrt_alpha_bars = self.sqrt_alpha_bars.to(device)
        self.sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars.to(device)
        return self

    def sample_steps(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return torch.randint(
            low=0,
            high=self.num_steps,
            size=(int(batch_size),),
            device=torch.device(device),
        )

    def time_values(self, step_index: torch.Tensor) -> torch.Tensor:
        step_index = step_index.reshape(-1).to(dtype=torch.float32)
        return ((step_index + 1.0) / float(self.num_steps)).view(-1, 1)

    def _extract(self, values: torch.Tensor, step_index: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        step_index = step_index.reshape(-1).to(device=values.device, dtype=torch.long)
        out = values.gather(0, step_index).view(-1, 1, 1)
        return out.to(device=ref.device, dtype=ref.dtype)

    def q_sample(
        self,
        x0: torch.Tensor,
        step_index: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha = self._extract(self.sqrt_alpha_bars, step_index, x0)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alpha_bars, step_index, x0)
        x_s = sqrt_alpha * x0 + sqrt_one_minus * noise
        return x_s, noise

    def terminal_noise_from_observation(
        self,
        *,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Infer the terminal diffusion noise that makes the stored noisy endpoint
        equal to x_T under this schedule.

        The model does not receive x1 as a condition. x1 only anchors the
        forward path so training starts from the same kind of terminal state
        used by evaluation.
        """

        if x0.shape != x1.shape:
            raise ValueError(f"x0 shape {tuple(x0.shape)} != x1 shape {tuple(x1.shape)}")

        terminal_index = torch.full(
            (x0.shape[0],),
            self.num_steps - 1,
            dtype=torch.long,
            device=x0.device,
        )
        sqrt_alpha = self._extract(self.sqrt_alpha_bars, terminal_index, x0)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alpha_bars, terminal_index, x0)
        return (x1 - sqrt_alpha * x0) / sqrt_one_minus.clamp_min(1e-8)

    def training_target(
        self,
        *,
        prediction_type: str,
        x0: torch.Tensor,
        x_s: torch.Tensor,
        noise: torch.Tensor,
        step_index: torch.Tensor,
    ) -> torch.Tensor:
        _normalize_prediction_type(prediction_type)
        del x0, x_s, step_index
        return noise

    def predict_x0_from_output(
        self,
        *,
        prediction_type: str,
        model_output: torch.Tensor,
        x_s: torch.Tensor,
        step_index: torch.Tensor,
    ) -> torch.Tensor:
        _normalize_prediction_type(prediction_type)

        sqrt_alpha = self._extract(self.sqrt_alpha_bars, step_index, x_s)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alpha_bars, step_index, x_s)
        return (x_s - sqrt_one_minus * model_output) / sqrt_alpha.clamp_min(1e-8)


def build_diffusion_model_input(
    x_s: torch.Tensor,
    is_pad: torch.Tensor,
) -> torch.Tensor:
    """Build the three-channel diffusion backbone input."""

    if is_pad.shape[:2] != x_s.shape[:2] or is_pad.shape[-1] != 1:
        raise ValueError(
            f"is_pad shape {tuple(is_pad.shape)} is incompatible with x_s {tuple(x_s.shape)}"
        )
    return torch.cat([x_s, is_pad.to(dtype=x_s.dtype)], dim=-1)


# ================================================================
# === DiffusionDataLoader
# ================================================================
class DiffusionDataLoader:
    """
    Epoch-file loader for step-local diffusion training.

    Output contract:
        DiffusionBatch(
            model_input=[x_s, is_pad],
            target=epsilon,
            diffusion_t=(step+1)/T,
            valid_mask=real point mask,
        )
    """

    def __init__(
        self,
        *,
        mode: str,
        data_dir: str,
        schedule: DiffusionNoiseSchedule,
        batch_size: int = 64,
        device: str | torch.device = "cpu",
        data_per_epoch: int = 37000,
        file_pattern: str = "*.pt",
        shuffle: bool = True,
        prediction_mode: str = "online",
        target_k: Optional[int] = None,
        online_pad_prob: float = 0.10,
        prediction_type: str = "epsilon",
    ) -> None:
        self.mode = _normalize_loader_mode(mode)
        self.data_dir = str(data_dir)
        self.schedule = schedule
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.data_per_epoch = int(data_per_epoch)
        self.file_pattern = str(file_pattern)
        self.shuffle = bool(shuffle)
        self.prediction_mode = _normalize_prediction_mode(prediction_mode)
        self.target_k = int(target_k) if target_k is not None else None
        self.online_pad_prob = float(online_pad_prob)
        self.prediction_type = _normalize_prediction_type(prediction_type)

        self.file_list = sorted(glob.glob(str(Path(self.data_dir) / self.file_pattern)))
        if not self.file_list:
            raise FileNotFoundError(
                f"No .pt files found in {self.data_dir} matching {self.file_pattern}"
            )

        self.file_rows_raw: list[int] = []
        self.file_rows_used: list[int] = []
        for file_path in self.file_list:
            pack = torch.load(file_path, map_location="cpu")
            x0, _x1 = load_x0_x1_from_pack(pack, file_path)
            n_raw = int(x0.shape[0])
            n_div = (n_raw // 1000) * 1000
            n_used = min(self.data_per_epoch, n_div if n_div > 0 else n_raw)
            if n_used <= 0:
                raise ValueError(f"Computed epoch size is 0 for {file_path} (n_raw={n_raw}).")
            self.file_rows_raw.append(n_raw)
            self.file_rows_used.append(n_used)

        self.X0: Optional[torch.Tensor] = None
        self.X1: Optional[torch.Tensor] = None
        self.N: int = 0
        self.perm: Optional[torch.Tensor] = None
        self.idx: int = 0

    @property
    def epoch_count(self) -> int:
        return len(self.file_list)

    @property
    def batches_per_epoch(self) -> int:
        if self.N <= 0:
            raise RuntimeError("No epoch loaded. Call set(epoch_idx) first.")
        return (self.N + self.batch_size - 1) // self.batch_size

    # ------------------------------------------------------------
    # Epoch API
    # ------------------------------------------------------------
    def set(self, epoch_idx: int) -> None:
        files = self.file_list
        epoch_idx = int(epoch_idx) % len(files)
        file_path = files[epoch_idx]
        pack = torch.load(file_path, map_location="cpu")
        x0, x1 = load_x0_x1_from_pack(pack, file_path)

        n = self.file_rows_used[epoch_idx]
        self.X0 = x0[:n].to(self.device, dtype=torch.float32)
        self.X1 = x1[:n].to(self.device, dtype=torch.float32)
        self.N = int(n)
        self.perm = (
            torch.randperm(self.N, device=self.device)
            if self.shuffle and self.mode == "train"
            else torch.arange(self.N, device=self.device)
        )
        self.idx = 0

        if self.target_k is None:
            self.target_k = int(self.X0.shape[1])

    def get_batch(self) -> DiffusionBatch:
        if self.X0 is None or self.X1 is None or self.perm is None:
            raise RuntimeError("No epoch loaded. Call set(epoch_idx) first.")
        if self.idx >= self.N:
            raise RuntimeError("Epoch data exhausted. Call set(epoch_idx) for the next epoch.")

        end = min(self.idx + self.batch_size, self.N)
        idx_slice = self.perm[self.idx : end]
        self.idx = end

        x0_raw = self.X0[idx_slice]
        x1_raw = self.X1[idx_slice]
        target_k = int(self.target_k)

        if self.prediction_mode == "online":
            if self.mode == "train":
                x0, x1, is_pad, valid_mask = build_diffusion_online_train_pairs(
                    x0_raw,
                    x1_raw,
                    target_k=target_k,
                    startup_pad_prob=self.online_pad_prob,
                )
            else:
                x0, x1, is_pad, valid_mask = build_diffusion_online_eval_pairs(
                    x0_raw,
                    x1_raw,
                    target_k=target_k,
                )
        else:
            x0, x1, is_pad, valid_mask = build_diffusion_offline_pairs(
                x0_raw,
                x1_raw,
                target_k=target_k,
                train_mode=self.mode == "train",
            )

        step_index = self.schedule.sample_steps(x0.shape[0], x0.device)
        terminal_noise = self.schedule.terminal_noise_from_observation(x0=x0, x1=x1)
        x_s, noise = self.schedule.q_sample(x0, step_index, terminal_noise)
        target = self.schedule.training_target(
            prediction_type=self.prediction_type,
            x0=x0,
            x_s=x_s,
            noise=noise,
            step_index=step_index,
        )
        model_input = build_diffusion_model_input(x_s, is_pad)
        diffusion_t = self.schedule.time_values(step_index).to(device=x0.device, dtype=x0.dtype)

        return DiffusionBatch(
            model_input=model_input,
            target=target,
            diffusion_t=diffusion_t,
            valid_mask=valid_mask,
            x0=x0,
            x1=x1,
            x_s=x_s,
            noise=noise,
            step_index=step_index,
        )

    def next_epoch(self) -> None:
        return None


# ================================================================
# === CLI Smoke Test
# ================================================================
def _main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the diffusion data loader.")
    parser.add_argument("--data_dir", default="./dataset/processed/NUMOSIM_Kanto/train")
    parser.add_argument("--file_pattern", default="*.pt")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--data_per_epoch", type=int, default=1000)
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    schedule = DiffusionNoiseSchedule(num_steps=args.num_steps).to(args.device)
    loader = DiffusionDataLoader(
        mode="train",
        data_dir=args.data_dir,
        file_pattern=args.file_pattern,
        schedule=schedule,
        batch_size=args.batch_size,
        data_per_epoch=args.data_per_epoch,
        target_k=args.K,
        device=args.device,
    )
    loader.set(0)
    batch = loader.get_batch()
    print(
        "batch loaded: "
        f"model_input={tuple(batch.model_input.shape)} "
        f"target={tuple(batch.target.shape)} "
        f"diffusion_t={tuple(batch.diffusion_t.shape)} "
        f"valid_mask={tuple(batch.valid_mask.shape)}"
    )


if __name__ == "__main__":
    _main()
