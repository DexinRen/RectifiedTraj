"""
Evaluation-time decoder for the diffusion baseline.

The class mirrors encoder_decoder.EncoderDecoder's public surface so the
normal benchmark pipeline can evaluate diffusion checkpoints without a
separate evaluator.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors

import encoder_decoder

from .data_loader_diffusion import DiffusionNoiseSchedule, build_diffusion_model_input
from .diffusion_model import build_diffusion_model


# ================================================================
# === Helpers
# ================================================================
def _normalize_prediction_type(raw: object, default: str = "epsilon") -> str:
    token = str(raw if raw is not None else "").strip().lower().replace("-", "_")
    if token in {"", "eps", "epsilon", "noise"}:
        return "epsilon"
    if raw is None:
        return str(default)
    raise ValueError(f"Diffusion decoder is epsilon-only; got prediction_type={raw!r}.")


def _resolve_sample_steps(cfg: dict, manual_config: dict | None) -> int:
    raw_value = None
    if isinstance(manual_config, dict):
        raw_value = manual_config.get("sample_steps")
    if raw_value is None:
        raw_value = cfg.get("primary_sample_steps")
    if raw_value is None:
        raw_value = cfg.get("diffusion_steps", 500)

    sample_steps = int(raw_value)
    diffusion_steps = int(cfg.get("diffusion_steps", 500))
    if sample_steps <= 0:
        raise ValueError(f"sample_steps must be positive, got {sample_steps}.")
    if sample_steps > diffusion_steps:
        raise ValueError(
            f"sample_steps={sample_steps} exceeds trained diffusion_steps={diffusion_steps}."
        )
    return sample_steps


def _load_diffusion_checkpoint(config_path: Path, ckpt_path: Path) -> tuple[torch.nn.Module, dict]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    input_coord_dim = int(cfg.get("input_coord_dim", 3))
    if input_coord_dim != 3:
        raise ValueError(
            "This diffusion decoder expects a step-local checkpoint with "
            f"input_coord_dim=3 ([x_k, is_pad]); got input_coord_dim={input_coord_dim}. "
            "Retrain the diffusion baseline before evaluation."
        )

    device = encoder_decoder.DEVICE
    model = build_diffusion_model(cfg).to(device)

    if ckpt_path.suffix == ".safetensors":
        state_dict = load_safetensors(str(ckpt_path), device=str(device))
    else:
        blob = torch.load(str(ckpt_path), map_location=device)
        if isinstance(blob, dict) and "model_state_dict" in blob:
            state_dict = blob["model_state_dict"]
        else:
            state_dict = blob

    model.load_state_dict(state_dict)
    model.eval()
    return model, cfg


# ================================================================
# === DiffusionEncoderDecoder
# ================================================================
class DiffusionEncoderDecoder:
    """
    Step-local diffusion decoder with EncoderDecoder-compatible methods.

    Public methods:
        denoise_chunk_enu(Xt_np, pad_count=0, pad_mask=None)
        denoise_chunk(gps_chunk, pad_count=0, pad_mask=None)
        denoise_traj_DF(traj)
    """

    def __init__(self, ckpt_path: str, manual_config: dict | None = None):
        ckpt_path = Path(ckpt_path)
        model_dir = ckpt_path.parent.parent
        config_file = model_dir / "log" / "config.json"
        model, cfg = _load_diffusion_checkpoint(config_file, ckpt_path)

        self.model = model
        self.cfg = cfg
        self.K = int(cfg.get("K", 256))
        self.prediction_type = _normalize_prediction_type(cfg.get("prediction_type", "epsilon"))
        self.sample_steps = _resolve_sample_steps(cfg, manual_config)
        self.t_delta = None

        prediction_mode = str(cfg.get("prediction_mode", "online")).strip().lower()
        if prediction_mode == "online":
            self.Q1_bytes = 0
            self.Q2_bytes = 0
        else:
            self.Q1_bytes = 1
            self.Q2_bytes = 12

        if manual_config is not None:
            self.Q1_bytes = int(manual_config.get("Q1", self.Q1_bytes))
            self.Q2_bytes = int(manual_config.get("Q2", self.Q2_bytes))

        self.Q1 = encoder_decoder.q_config_to_points(self.Q1_bytes)
        self.Q2 = encoder_decoder.q_config_to_points(self.Q2_bytes)
        if self.K <= self.Q1 + self.Q2:
            raise ValueError(
                f"Invalid buckle settings: K={self.K}, Q1={self.Q1}, Q2={self.Q2}."
            )
        self.stride = self.K - (self.Q1 + self.Q2)

        self.schedule = DiffusionNoiseSchedule(
            num_steps=int(cfg.get("diffusion_steps", 500)),
            beta_start=float(cfg.get("beta_start", 1e-4)),
            beta_end=float(cfg.get("beta_end", 0.02)),
            schedule_type=str(cfg.get("diffusion_schedule", "linear")),
        ).to(encoder_decoder.DEVICE)
        self._sampling_indices = self._build_sampling_indices()

    def _build_sampling_indices(self) -> list[int]:
        total_steps = int(self.schedule.num_steps)
        if self.sample_steps == total_steps:
            return list(range(total_steps - 1, -1, -1))
        if self.sample_steps == 1:
            return [total_steps - 1]

        values = np.linspace(total_steps - 1, 0, num=self.sample_steps)
        indices = [int(round(float(value))) for value in values]
        out: list[int] = []
        seen: set[int] = set()
        for index in indices:
            index = max(0, min(total_steps - 1, int(index)))
            if index in seen:
                continue
            seen.add(index)
            out.append(index)
        if out[-1] != 0:
            out[-1] = 0
        return sorted(out, reverse=True)

    def _pad_channel(self, K: int, *, pad_count: int = 0, pad_mask=None) -> torch.Tensor:
        is_pad = torch.zeros((1, K, 1), dtype=torch.float32, device=encoder_decoder.DEVICE)
        if pad_mask is not None:
            mask = torch.as_tensor(
                pad_mask,
                dtype=torch.float32,
                device=encoder_decoder.DEVICE,
            ).reshape(1, K, 1)
            is_pad.copy_(mask)
        elif pad_count > 0:
            is_pad[:, : int(pad_count), :] = 1.0
        return is_pad

    def _extract(self, values: torch.Tensor, step_index: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        step_index = step_index.reshape(-1).to(device=values.device, dtype=torch.long)
        out = values.gather(0, step_index).view(-1, 1, 1)
        return out.to(device=ref.device, dtype=ref.dtype)

    def _ddim_next(
        self,
        *,
        pred_x0: torch.Tensor,
        pred_epsilon: torch.Tensor,
        next_index: int,
    ) -> torch.Tensor:
        if next_index < 0:
            return pred_x0

        index_tensor = torch.tensor([next_index], dtype=torch.long, device=encoder_decoder.DEVICE)
        sqrt_alpha_next = self._extract(self.schedule.sqrt_alpha_bars, index_tensor, pred_x0)
        sqrt_one_minus_next = self._extract(
            self.schedule.sqrt_one_minus_alpha_bars,
            index_tensor,
            pred_x0,
        )
        return sqrt_alpha_next * pred_x0 + sqrt_one_minus_next * pred_epsilon

    @torch.no_grad()
    def denoise_chunk_enu(self, Xt_np: np.ndarray, pad_count: int = 0, pad_mask=None) -> np.ndarray:
        """
        Denoise one ENU chunk. Xt_np is the observed noisy chunk in meters.
        """

        x_s = torch.as_tensor(
            Xt_np,
            dtype=torch.float32,
            device=encoder_decoder.DEVICE,
        ).reshape(1, self.K, 2)
        is_pad = self._pad_channel(self.K, pad_count=pad_count, pad_mask=pad_mask)

        for pos, step_index_value in enumerate(self._sampling_indices):
            step_index = torch.tensor(
                [step_index_value],
                dtype=torch.long,
                device=encoder_decoder.DEVICE,
            )
            diffusion_t = self.schedule.time_values(step_index).to(
                device=encoder_decoder.DEVICE,
                dtype=x_s.dtype,
            )
            model_input = build_diffusion_model_input(x_s, is_pad)
            model_output = self.model(model_input, diffusion_t)
            next_index = self._sampling_indices[pos + 1] if pos + 1 < len(self._sampling_indices) else -1

            pred_x0 = self.schedule.predict_x0_from_output(
                prediction_type=self.prediction_type,
                model_output=model_output,
                x_s=x_s,
                step_index=step_index,
            )
            x_s = self._ddim_next(
                pred_x0=pred_x0,
                pred_epsilon=model_output,
                next_index=next_index,
            )

        return x_s.squeeze(0).detach().cpu().numpy()

    def denoise_chunk(self, gps_chunk: np.ndarray, pad_count: int = 0, pad_mask=None) -> np.ndarray:
        Xt_np, origin = encoder_decoder.gps_to_enu(gps_chunk)
        clean_enu = self.denoise_chunk_enu(Xt_np, pad_count=pad_count, pad_mask=pad_mask)
        return encoder_decoder.enu_to_gps(clean_enu, origin)

    def build_padded_trajectory(self, traj) -> tuple[np.ndarray, np.ndarray, int, int]:
        traj = np.asarray(traj, dtype=float)
        traj = encoder_decoder.remove_nan_rows(traj)
        n_points = int(len(traj))
        if n_points == 0:
            return (
                np.zeros((0, 2), dtype=float),
                np.zeros((0,), dtype=np.float32),
                0,
                0,
            )

        n_chunks = int(np.ceil(n_points / self.stride))
        head = np.repeat(traj[0:1, :], self.Q1, axis=0) if self.Q1 > 0 else np.zeros((0, 2))
        payload_pad_len = n_chunks * self.stride - n_points
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

    def denoise_traj_DF(self, traj) -> np.ndarray:
        traj_padded, pad_mask_padded, n_chunks, n_points = self.build_padded_trajectory(traj)
        if n_points == 0:
            return np.zeros((0, 2), dtype=float)

        payloads = []
        for chunk_idx in range(n_chunks):
            start = chunk_idx * self.stride
            end = start + self.K
            gps_chunk = traj_padded[start:end]
            chunk_pad_mask = pad_mask_padded[start:end]
            gps_clean = self.denoise_chunk(gps_chunk, pad_mask=chunk_pad_mask)
            payloads.append(gps_clean[self.Q1:self.Q1 + self.stride])

        out_full = np.concatenate(payloads, axis=0) if payloads else np.zeros((0, 2), dtype=float)
        out = out_full[:n_points]
        if out.shape[0] != n_points:
            raise RuntimeError(
                f"chunk_stitch produced wrong length: out={out.shape[0]} != N={n_points}"
            )
        return out


__all__ = ["DiffusionEncoderDecoder"]
