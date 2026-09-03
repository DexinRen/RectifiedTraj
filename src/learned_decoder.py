"""
Factory for learned denoising decoders used by the benchmark pipeline.

RectifiedTraj and DirectReg use encoder_decoder.EncoderDecoder.
Diffusion checkpoints use the diffusion baseline decoder with the same public
methods.
"""

from __future__ import annotations

import json
from pathlib import Path

from encoder_decoder import EncoderDecoder


# ================================================================
# === Helpers
# ================================================================
def _normalize_data_hypothesis(raw, default: str = "RectifiedTraj") -> str:
    token = str(raw if raw is not None else "").strip().lower().replace("-", "_")
    if token in {"", "rf", "rectified", "rectified_flow", "rectifiedtraj", "rectified_traj"}:
        return "RectifiedTraj"
    if token in {
        "dr", "directreg", "direct_reg", "direct_regression",
        "rr", "residual", "residualreg", "residual_reg", "residual_regression",
    }:
        return "DirectReg"
    if token in {"diffusion", "ddpm", "diffusion_baseline"}:
        return "Diffusion"
    text = str(raw).strip() if raw is not None else ""
    return text if text else str(default)


def _load_checkpoint_config(ckpt_path: str | Path) -> dict:
    ckpt = Path(ckpt_path)
    config_path = ckpt.parent.parent / "log" / "config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


# ================================================================
# === Public Factory
# ================================================================
def build_learned_decoder(ckpt_path: str | Path, manual_config: dict | None = None):
    cfg = _load_checkpoint_config(ckpt_path)
    data_hypothesis = _normalize_data_hypothesis(
        cfg.get("data_hypothesis", cfg.get("data_hypothetis", "RectifiedTraj"))
    )

    if data_hypothesis == "Diffusion":
        from baseline.models.diffusion import DiffusionEncoderDecoder

        return DiffusionEncoderDecoder(str(ckpt_path), manual_config=manual_config)

    return EncoderDecoder(str(ckpt_path), manual_config=manual_config)


__all__ = ["build_learned_decoder"]
