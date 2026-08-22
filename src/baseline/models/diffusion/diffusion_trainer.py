#!/usr/bin/env python3
"""
Trainer for the step-local diffusion learned baseline.

This script intentionally mirrors the artifact shape of theta_train.py:
    bin/model/Diffusion_online/<run_name>/
        ckpts/
        best_ckpt/
        log/config.json
        log/config_init.json
        log/train_data.csv
        fig/
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from time import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import torch
from safetensors.torch import save_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from baseline.models.diffusion.data_loader_diffusion import (  # noqa: E402
    DiffusionBatch,
    DiffusionDataLoader,
    DiffusionNoiseSchedule,
)
from baseline.models.diffusion.diffusion_model import (  # noqa: E402
    build_diffusion_model,
    count_parameters,
)


# ================================================================
# === Helpers: Configuration
# ================================================================
def default_diffusion_config() -> dict:
    return {
        "data_hypothesis": "Diffusion",
        "prediction_mode": "online",
        "model_type": "diffusion_hybrid_online",
        "model_root": "./bin/model",
        "train_dir": "./dataset/processed/NUMOSIM_Kanto/train",
        "quick_val_path": "./dataset/processed/NUMOSIM_Kanto/val/quick_val_chunk_50k.pt",
        "K": 256,
        "coord_dim": 2,
        "input_coord_dim": 3,
        "hidden": 384,
        "layers": 8,
        "cnn_layers": 4,
        "nhead": 8,
        "noise_dim": 128,
        "kernel_size": 7,
        "dropout": 0.1,
        "diffusion_steps": 1000,
        "diffusion_schedule": "linear",
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "prediction_type": "epsilon",
        "batch_size": 64,
        "data_per_epoch": 37000,
        "epochs": 50,
        "lr": 1e-4,
        "lr_floor": 5e-6,
        "warmup_steps": 1000,
        "weight_decay": 0.0,
        "save_every_rows": 37000,
        "quick_val_batches": 0,
        "keep_best_checkpoints": 3,
        "keep_last_checkpoints": 3,
        "online_pad_prob": 0.10,
        "terminal_print": True,
        "device": "",
        "cpu": False,
    }


def load_config(config_path: str | None) -> dict:
    cfg = default_diffusion_config()
    if config_path is None:
        return normalize_config(cfg)

    path = Path(config_path)
    text = path.read_text(encoding="utf-8").strip()
    if text:
        user_cfg = json.loads(text)
        cfg.update(user_cfg)
    return normalize_config(cfg)


def normalize_config(config: dict) -> dict:
    config["data_hypothesis"] = "Diffusion"
    config["prediction_mode"] = str(config.get("prediction_mode", "online")).strip().lower()
    config["input_coord_dim"] = int(config.get("input_coord_dim", 3))
    config["coord_dim"] = int(config.get("coord_dim", 2))
    config["K"] = int(config.get("K", 256))
    config["batch_size"] = int(config.get("batch_size", 64))
    config["data_per_epoch"] = int(config.get("data_per_epoch", 37000))
    config["epochs"] = int(config.get("epochs", 50))
    config["diffusion_steps"] = int(config.get("diffusion_steps", 1000))
    if config["input_coord_dim"] != 3:
        raise ValueError("Diffusion trainer expects input_coord_dim=3 for [x_s, is_pad].")
    return config


def resolve_model_root_dir(config: dict) -> Path:
    base = Path(str(config.get("model_root", "./bin/model")).strip() or "./bin/model")
    if base.as_posix().rstrip("/") in {"./bin/model", "bin/model"}:
        if str(config.get("prediction_mode", "online")).strip().lower() == "online":
            return base / "Diffusion_online"
        return base / "Diffusion"
    return base


def resolve_training_device(config: dict) -> torch.device:
    if bool(config.get("cpu", False)):
        return torch.device("cpu")

    token = str(config.get("device", "")).strip().lower()
    if token.startswith("cpu"):
        return torch.device("cpu")
    if token.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("config.device=cuda requested but CUDA is unavailable.")
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================
# === Helpers: Logging and Run Directories
# ================================================================
def build_logger(log_path: str, runtime: dict) -> logging.Logger:
    logger = logging.getLogger("diffusion_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    if bool(runtime["config"].get("terminal_print", True)):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

    return logger


def make_model_name(config: dict) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = str(config.get("model_type", "diffusion_hybrid_online"))
    model = build_diffusion_model(config)
    size_tag = parameter_size_tag(count_parameters(model))
    return f"{model_type}_{size_tag}_{timestamp}"


def parameter_size_tag(param_count: int) -> str:
    count = int(param_count)
    if count >= 1_000_000:
        return f"{round(count / 1_000_000):.0f}M"
    if count >= 1_000:
        return f"{round(count / 1_000):.0f}K"
    return str(count)


def model_house_builder(runtime: dict) -> dict:
    cfg = runtime["config"]
    model_root = Path(runtime["model_root_dir"])
    model_name = runtime["model_name"]
    base = model_root / model_name

    (base / "ckpts").mkdir(parents=True, exist_ok=True)
    (base / "best_ckpt").mkdir(parents=True, exist_ok=True)
    (base / "log").mkdir(parents=True, exist_ok=True)
    (base / "fig").mkdir(parents=True, exist_ok=True)

    runtime["model_dir"] = str(base)
    runtime["ckpt_dir"] = str(base / "ckpts")
    runtime["best_ckpt_dir"] = str(base / "best_ckpt")
    runtime["log_dir"] = str(base / "log")
    runtime["fig_dir"] = str(base / "fig")
    runtime["config_path"] = str(base / "log" / "config.json")
    runtime["config_init_path"] = str(base / "log" / "config_init.json")
    runtime["train_log"] = str(base / "log" / "train_data.csv")

    init_cfg_path = Path(runtime["config_init_path"])
    if not init_cfg_path.exists():
        init_cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    Path(runtime["config_path"]).write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    train_log = Path(runtime["train_log"])
    if not train_log.exists():
        train_log.write_text(
            "ckpt_name,epoch,step,loss_train,loss_val,x0_err_val,lr\n",
            encoding="utf-8",
        )

    return runtime


# ================================================================
# === Helpers: Optimizer and Scheduler
# ================================================================
def optimizer_steps_per_epoch_from_config(config: dict) -> int:
    rows_per_epoch = int(config["data_per_epoch"])
    batch_size = int(config["batch_size"])
    if rows_per_epoch <= 0 or batch_size <= 0:
        raise ValueError("data_per_epoch and batch_size must be > 0.")
    return max(1, math.ceil(rows_per_epoch / batch_size))


def build_scheduler(optimizer, config: dict, total_steps_override: int | None = None):
    if total_steps_override is None:
        total_steps = optimizer_steps_per_epoch_from_config(config) * int(config["epochs"])
    else:
        total_steps = int(total_steps_override)
    warmup_steps = int(config.get("warmup_steps", 1000))

    if total_steps <= 0:
        raise ValueError("Total training steps must be > 0.")
    if total_steps == 1:
        warmup_steps = 0
    else:
        warmup_steps = max(0, min(warmup_steps, total_steps - 1))

    if warmup_steps > 0:
        main_steps = max(1, total_steps - warmup_steps)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=main_steps,
            eta_min=float(config["lr"]) * 0.1,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps),
        eta_min=float(config["lr"]) * 0.1,
    )


# ================================================================
# === Helpers: Loss and Metrics
# ================================================================
def reduce_masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    point_mse = ((pred - target) ** 2).mean(dim=-1)
    valid_mask = valid_mask.to(device=point_mse.device, dtype=point_mse.dtype)
    return (point_mse * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)


def reduce_x0_error(
    pred_x0: torch.Tensor,
    true_x0: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    point_l2 = torch.sqrt(((pred_x0 - true_x0) ** 2).sum(dim=-1) + 1e-8)
    valid_mask = valid_mask.to(device=point_l2.device, dtype=point_l2.dtype)
    return (point_l2 * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)


# ================================================================
# === Initialization
# ================================================================
def build_noise_schedule(config: dict, device: torch.device) -> DiffusionNoiseSchedule:
    return DiffusionNoiseSchedule(
        num_steps=int(config.get("diffusion_steps", 1000)),
        beta_start=float(config.get("beta_start", 1e-4)),
        beta_end=float(config.get("beta_end", 0.02)),
        schedule_type=str(config.get("diffusion_schedule", "linear")),
    ).to(device)


def training_initializer(runtime: dict) -> dict:
    config = runtime["config"]
    device = runtime["device"]

    model = build_diffusion_model(config).to(device)
    model.train()
    runtime["model"] = model

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    runtime["optimizer"] = optimizer
    runtime["scheduler"] = build_scheduler(optimizer, config)

    resume_path = runtime.get("resume_ckpt_path")
    if resume_path:
        checkpoint = torch.load(str(resume_path), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        runtime["scheduler"].load_state_dict(checkpoint["scheduler_state_dict"])
        runtime["logger"].info(f"[Resume] Loaded checkpoint: {resume_path}")

    runtime["best_val_loss"] = runtime.get("best_val_loss", None)
    runtime["last_saved_step"] = int(runtime.get("last_saved_step", 0))
    return runtime


def build_train_loader(runtime: dict) -> DiffusionDataLoader:
    config = runtime["config"]
    return DiffusionDataLoader(
        mode="train",
        data_dir=config["train_dir"],
        schedule=runtime["diffusion_schedule"],
        batch_size=int(config["batch_size"]),
        device=runtime["device"],
        data_per_epoch=int(config["data_per_epoch"]),
        shuffle=True,
        prediction_mode=str(config.get("prediction_mode", "online")),
        target_k=int(config["K"]),
        online_pad_prob=float(config.get("online_pad_prob", 0.10)),
        prediction_type=str(config.get("prediction_type", "epsilon")),
    )


def build_val_loader(runtime: dict) -> DiffusionDataLoader:
    config = runtime["config"]
    quick_val_path = Path(config["quick_val_path"])
    return DiffusionDataLoader(
        mode="eval",
        data_dir=str(quick_val_path.parent),
        file_pattern=quick_val_path.name,
        schedule=runtime["diffusion_schedule"],
        batch_size=int(config["batch_size"]),
        device=runtime["device"],
        data_per_epoch=int(config.get("quick_val_rows", config["data_per_epoch"])),
        shuffle=False,
        prediction_mode=str(config.get("prediction_mode", "online")),
        target_k=int(config["K"]),
        online_pad_prob=0.0,
        prediction_type=str(config.get("prediction_type", "epsilon")),
    )


# ================================================================
# === Train and Validation Steps
# ================================================================
def train_step(runtime: dict, batch: DiffusionBatch) -> dict:
    model = runtime["model"]
    optimizer = runtime["optimizer"]
    scheduler = runtime["scheduler"]
    noise_schedule = runtime["diffusion_schedule"]
    config = runtime["config"]

    model.train()
    pred = model(batch.model_input, batch.diffusion_t)
    loss = reduce_masked_mse(pred, batch.target, batch.valid_mask)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()

    lr_floor = float(config.get("lr_floor", 5e-6))
    for group in optimizer.param_groups:
        if group["lr"] < lr_floor:
            group["lr"] = lr_floor

    pred_x0 = noise_schedule.predict_x0_from_output(
        prediction_type=str(config.get("prediction_type", "epsilon")),
        model_output=pred.detach(),
        x_s=batch.x_s,
        step_index=batch.step_index,
    )
    x0_err = reduce_x0_error(pred_x0, batch.x0, batch.valid_mask)

    return {
        "loss": float(loss.item()),
        "x0_err": float(x0_err.item()),
        "lr": float(optimizer.param_groups[0]["lr"]),
    }


@torch.no_grad()
def quick_diffusion_val(runtime: dict) -> tuple[float, float]:
    model = runtime["model"]
    loader = runtime["val_loader"]
    noise_schedule = runtime["diffusion_schedule"]
    config = runtime["config"]
    prediction_type = str(config.get("prediction_type", "epsilon"))
    max_batches = int(config.get("quick_val_batches", 0))

    model.eval()
    loader.set(0)

    total_loss = 0.0
    total_x0_err = 0.0
    count = 0
    steps = loader.batches_per_epoch
    if max_batches > 0:
        steps = min(steps, max_batches)

    for _idx in range(steps):
        batch = loader.get_batch()
        pred = model(batch.model_input, batch.diffusion_t)
        loss = reduce_masked_mse(pred, batch.target, batch.valid_mask)
        pred_x0 = noise_schedule.predict_x0_from_output(
            prediction_type=prediction_type,
            model_output=pred,
            x_s=batch.x_s,
            step_index=batch.step_index,
        )
        x0_err = reduce_x0_error(pred_x0, batch.x0, batch.valid_mask)

        total_loss += float(loss.item())
        total_x0_err += float(x0_err.item())
        count += 1

    if count == 0:
        raise RuntimeError("quick_diffusion_val received zero validation batches.")
    model.train()
    return total_loss / count, total_x0_err / count


# ================================================================
# === Checkpoints and Logs
# ================================================================
def save_checkpoint_and_log(
    runtime: dict,
    *,
    avg_train_loss: float,
    val_loss: float,
    x0_err_val: float,
) -> None:
    model = runtime["model"]
    optimizer = runtime["optimizer"]
    scheduler = runtime["scheduler"]
    epoch = int(runtime["epoch"])
    global_step = int(runtime["global_step"])

    ckpt_name = f"ckpt_e{epoch}_s{global_step}"
    ckpt_path = Path(runtime["ckpt_dir"]) / f"{ckpt_name}.safetensors"
    full_path = Path(runtime["ckpt_dir"]) / f"{ckpt_name}_full.pt"

    save_file(model.state_dict(), str(ckpt_path))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": runtime["config"],
        },
        str(full_path),
    )

    with Path(runtime["train_log"]).open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ckpt_name",
                "epoch",
                "step",
                "loss_train",
                "loss_val",
                "x0_err_val",
                "lr",
            ],
        )
        writer.writerow(
            {
                "ckpt_name": ckpt_name,
                "epoch": epoch,
                "step": global_step,
                "loss_train": f"{avg_train_loss:.8f}",
                "loss_val": f"{val_loss:.8f}",
                "x0_err_val": f"{x0_err_val:.8f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.10g}",
            }
        )

    best_val = runtime.get("best_val_loss")
    if best_val is None or float(val_loss) < float(best_val):
        runtime["best_val_loss"] = float(val_loss)
        best_dir = Path(runtime["best_ckpt_dir"])
        for path in best_dir.glob("*"):
            if path.is_file():
                path.unlink()
        shutil.copy2(ckpt_path, best_dir / ckpt_path.name)
        shutil.copy2(full_path, best_dir / full_path.name)

    runtime["last_saved_step"] = global_step
    plot_training_curves(runtime)
    runtime["logger"].info(
        "[Checkpoint] %s | train=%.6f val=%.6f x0_err=%.6f",
        ckpt_name,
        avg_train_loss,
        val_loss,
        x0_err_val,
    )
    trim_checkpoints(runtime)


def trim_checkpoints(runtime: dict) -> None:
    """
    Trim regular checkpoint files while preserving best_ckpt copies.

    Keep policy:
        - best N by validation diffusion loss
        - best N by validation one-step x0 error
        - last M by global step
    """

    ckpt_dir = Path(runtime["ckpt_dir"])
    csv_path = Path(runtime["train_log"])
    if not ckpt_dir.exists() or not csv_path.exists():
        return

    keep_best = int(runtime["config"].get("keep_best_checkpoints", 3))
    keep_last = int(runtime["config"].get("keep_last_checkpoints", 3))

    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ckpt_name = str(row.get("ckpt_name", "")).strip()
            if not ckpt_name:
                continue
            rows.append(
                {
                    "ckpt": ckpt_name,
                    "step": int(row["step"]),
                    "loss_val": float(row["loss_val"]),
                    "x0_err_val": float(row["x0_err_val"]),
                }
            )

    if not rows:
        return

    rows_by_loss = sorted(rows, key=lambda item: item["loss_val"])
    rows_by_x0 = sorted(rows, key=lambda item: item["x0_err_val"])
    rows_by_step = sorted(rows, key=lambda item: item["step"])

    keep_set = set()
    for item in rows_by_loss[:keep_best]:
        keep_set.add(item["ckpt"])
    for item in rows_by_x0[:keep_best]:
        keep_set.add(item["ckpt"])
    for item in rows_by_step[-keep_last:]:
        keep_set.add(item["ckpt"])

    removed = 0
    for ckpt_file in ckpt_dir.glob("*.safetensors"):
        if ckpt_file.stem in keep_set:
            continue
        full_file = ckpt_dir / ckpt_file.name.replace(".safetensors", "_full.pt")
        if ckpt_file.exists():
            ckpt_file.unlink()
            removed += 1
        if full_file.exists():
            full_file.unlink()
            removed += 1

    runtime["logger"].info(
        "[Trim] Checkpoints trimmed | kept=%d removed_files=%d",
        len(keep_set),
        removed,
    )


def plot_training_curves(runtime: dict) -> None:
    train_log = Path(runtime["train_log"])
    if not train_log.exists():
        return

    steps = []
    train_loss = []
    val_loss = []
    x0_err = []
    with train_log.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("step"):
                continue
            steps.append(int(row["step"]))
            train_loss.append(float(row["loss_train"]))
            val_loss.append(float(row["loss_val"]))
            x0_err.append(float(row["x0_err_val"]))

    if not steps:
        return

    fig_dir = Path(runtime["fig_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(steps, train_loss, label="train_loss")
    plt.plot(steps, val_loss, label="val_loss")
    plt.xlabel("step")
    plt.ylabel("diffusion mse")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "loss_vs_step.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(steps, x0_err, label="x0_err_val")
    plt.xlabel("step")
    plt.ylabel("one-step x0 error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "x0_err_vs_step.png", dpi=160)
    plt.close()


# ================================================================
# === Progress Display
# ================================================================
def write_live_progress(
    *,
    epoch: int,
    epochs: int,
    step_idx: int,
    steps_per_ep: int,
    result: dict,
    avg_loss: float,
    avg_x0_err: float,
    speed: float,
) -> None:
    """
    Render one compact two-line progress display.

    The cursor is returned to the first line, so repeated calls overwrite the
    same two terminal lines instead of growing the output.
    """

    progress = step_idx / max(1, steps_per_ep)
    width = 24
    filled = int(width * progress)
    bar = "#" * filled + "-" * (width - filled)

    sys.stdout.write("\r\033[K")
    sys.stdout.write(
        f"Ep {epoch}/{epochs} | {step_idx}/{steps_per_ep} "
        f"[{bar}] {progress * 100.0:5.1f}%\n"
    )
    sys.stdout.write("\r\033[K")
    sys.stdout.write(
        f"loss {result['loss']:.4f} | avg {avg_loss:.4f} | "
        f"x0 {avg_x0_err:.4f} | lr {result['lr']:.2e} | {speed:.1f}/s"
    )
    sys.stdout.write("\033[F")
    sys.stdout.flush()


def clear_live_progress() -> None:
    """
    Clear the two-line live display and leave the cursor on a fresh line.
    """

    sys.stdout.write("\r\033[K\033[B\r\033[K\n")
    sys.stdout.flush()


# ================================================================
# === Training Loop
# ================================================================
def training_manager(runtime: dict) -> None:
    config = runtime["config"]
    logger = runtime["logger"]
    dataloader = runtime["dataloader"]

    batch_size = int(config["batch_size"])
    save_every_rows = int(config.get("save_every_rows", config["data_per_epoch"]))
    save_every = max(1, save_every_rows // batch_size)
    start_epoch = int(runtime.get("start_epoch", 1))
    epochs = int(config["epochs"])

    logger.info(
        "Training start | model=%s params=%d device=%s save_every_steps=%d",
        runtime["model_name"],
        count_parameters(runtime["model"]),
        runtime["device"],
        save_every,
    )

    total_loss = 0.0
    total_x0_err = 0.0
    total_steps = 0
    start_time = time()

    for epoch in range(start_epoch, epochs + 1):
        runtime["epoch"] = epoch
        dataloader.set(epoch - 1)
        steps_per_ep = dataloader.batches_per_epoch

        for step_idx in range(1, steps_per_ep + 1):
            batch = dataloader.get_batch()
            result = train_step(runtime, batch)
            runtime["global_step"] += 1

            total_steps += 1
            total_loss += result["loss"]
            total_x0_err += result["x0_err"]
            avg_loss = total_loss / total_steps
            avg_x0_err = total_x0_err / total_steps

            if bool(config.get("terminal_print", True)):
                elapsed = max(time() - start_time, 1e-8)
                speed = total_steps / elapsed
                write_live_progress(
                    epoch=epoch,
                    epochs=epochs,
                    step_idx=step_idx,
                    steps_per_ep=steps_per_ep,
                    result=result,
                    avg_loss=avg_loss,
                    avg_x0_err=avg_x0_err,
                    speed=speed,
                )

            if runtime["global_step"] % save_every == 0:
                if bool(config.get("terminal_print", True)):
                    clear_live_progress()
                val_loss, x0_err_val = quick_diffusion_val(runtime)
                save_checkpoint_and_log(
                    runtime,
                    avg_train_loss=avg_loss,
                    val_loss=val_loss,
                    x0_err_val=x0_err_val,
                )

        if bool(config.get("terminal_print", True)):
            clear_live_progress()

    if int(runtime.get("last_saved_step", 0)) != int(runtime["global_step"]):
        val_loss, x0_err_val = quick_diffusion_val(runtime)
        save_checkpoint_and_log(
            runtime,
            avg_train_loss=total_loss / max(1, total_steps),
            val_loss=val_loss,
            x0_err_val=x0_err_val,
        )


# ================================================================
# === Resume and Runtime Assembly
# ================================================================
def resolve_resume_full_checkpoint(path: str | Path) -> Path:
    ckpt_path = Path(path)
    if ckpt_path.suffix == ".safetensors":
        full_path = ckpt_path.with_name(f"{ckpt_path.stem}_full.pt")
        if full_path.exists():
            return full_path
    if ckpt_path.exists():
        return ckpt_path
    raise FileNotFoundError(f"Resume checkpoint not found: {path}")


def build_runtime(args: argparse.Namespace) -> dict:
    runtime: dict = {}

    if args.resume:
        resume_path = resolve_resume_full_checkpoint(args.resume)
        model_dir = resume_path.parent.parent
        config = load_config(str(model_dir / "log" / "config.json"))
        if args.cpu:
            config["cpu"] = True

        checkpoint = torch.load(str(resume_path), map_location="cpu")
        runtime["resume_ckpt_path"] = str(resume_path)
        runtime["start_epoch"] = int(checkpoint.get("epoch", 0)) + 1
        runtime["global_step"] = int(checkpoint.get("global_step", 0))
        runtime["model_root_dir"] = str(model_dir.parent)
        runtime["model_name"] = model_dir.name
    else:
        config = load_config(args.config)
        if args.cpu:
            config["cpu"] = True
        runtime["start_epoch"] = 1
        runtime["global_step"] = 0
        runtime["model_root_dir"] = str(resolve_model_root_dir(config))
        runtime["model_name"] = args.model_name or make_model_name(config)

    runtime["config"] = config
    model_house_builder(runtime)
    log_root = Path("./bin/log")
    log_root.mkdir(parents=True, exist_ok=True)
    runtime["logger"] = build_logger(str(log_root / "diffusion_train.log"), runtime)
    runtime["device"] = resolve_training_device(config)
    runtime["diffusion_schedule"] = build_noise_schedule(config, runtime["device"])
    runtime["dataloader"] = build_train_loader(runtime)
    runtime["val_loader"] = build_val_loader(runtime)
    training_initializer(runtime)
    return runtime


# ================================================================
# === Cleanup and Main
# ================================================================
def cleanup_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the diffusion learned baseline.")
    parser.add_argument("--config", default=None, help="Path to diffusion config JSON.")
    parser.add_argument("--resume", default=None, help="Path to a *_full.pt checkpoint.")
    parser.add_argument("--model-name", default=None, help="Optional run folder name for new training.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training.")
    args = parser.parse_args()

    cleanup_memory()
    runtime = build_runtime(args)
    training_manager(runtime)
    runtime["logger"].info("Training finished | model_dir=%s", runtime["model_dir"])


if __name__ == "__main__":
    main()
