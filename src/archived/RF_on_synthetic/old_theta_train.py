"""
Training Script for theta Rectified Flow Model

This script orchestrates the full training pipeline:
1. Load processed training data from parquet_processor
2. Create dataloaders with buckle point masking
3. Initialize theta model
4. Train with Huber loss and rectified flow framework
5. Save checkpoints and log metrics

RECTIFIED FLOW: t is the noise level parameter ∈ [0, 1)
    - t=0: clean data (X_0)
    - t=1: noisy data (X_1)
    - X_t = X_0 + t * V (linear interpolation)
    - Model predicts: v_θ(X_t, t) ≈ V = X_1 - X_0

Usage:
    python theta_train.py --model_type nn --hidden 512 --layers 6 --epochs 10
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch, random
from torch.utils.data import Dataset, DataLoader
import sys
import csv
import statistics
import re
from datetime import datetime

# Import our modules
from theta_model import (
    build_theta,
    init_training_runtime,
    train_loop,
    count_parameters
)

# Setup logger
logger = logging.getLogger(__name__)


# ============================================================================
# Dataset
# ============================================================================

class TrajectoryDataset(Dataset):
    """
    Streaming dataset that loads one tensorized .pt file at a time.

    Each .pt file now contains:
        {
            "X_t": FloatTensor [N, K, 4],
            "V":   FloatTensor [N, K, 2],
            "t":   FloatTensor [N, 1]
        }
    """

    def __init__(self, data_dir: str, K: int = 256, Q: int = 1):
        self.data_dir = Path(data_dir)
        self.data_files = sorted(self.data_dir.glob("*.pt"))
        if not self.data_files:
            raise ValueError(f"No .pt files found in {data_dir}")
        self.K = K
        self.Q = Q

        # cache state
        self._cached_idx = -1
        self._cached_data = None
        self._cached_len = 0

        # pre-read file lengths from .len sidecars if they exist
        self.file_lengths = {}
        for fp in self.data_files:
            len_path = fp.with_suffix(fp.suffix + ".len")
            if len_path.exists():
                with open(len_path) as f:
                    self.file_lengths[fp.name] = int(f.read().strip())

    def _load_file(self, idx: int):
        """Load one tensorized .pt file lazily."""
        if idx == self._cached_idx:
            return
        torch.cuda.empty_cache()
        data = torch.load(self.data_files[idx], map_location="cpu", weights_only=False)
        self._cached_data = data
        self._cached_len = data["X_t"].shape[0]
        self._cached_idx = idx

    def __len__(self):
        if self.file_lengths:
            return sum(self.file_lengths.values())
        if self._cached_len > 0:
            return self._cached_len * len(self.data_files)
        return 1000 * len(self.data_files)  # fallback estimate

    def __getitem__(self, idx):
        file_idx = idx % len(self.data_files)
        self._load_file(file_idx)

        i = random.randint(0, self._cached_len - 1)
        data = self._cached_data

        X_t = data["X_t"][i]       # [K, 4]
        V   = data["V"][i]         # [K, 2]
        t   = data["t"][i]         # [1]

        mask = (X_t[:, 3] < 0.5).float().unsqueeze(-1)  # 0 for is_start=True
        return {
            "X_t": X_t[:, :2],  # keep only EN coordinates
            "t": t,
            "V": V,
            "mask": mask,
        }


def create_dataloaders(train_dir: str, val_dir: str = None, batch_size: int = 32,
                      num_workers: int = 4, K: int = 256, Q: int = 1) -> Dict:
    """
    Create train and validation dataloaders.
    
    Args:
        train_dir: path to training data
        val_dir: path to validation data (optional)
        batch_size: batch size
        num_workers: number of dataloader workers
        K: chunk length
        Q: overlap length
    
    Returns:
        {
            "train": DataLoader,
            "val": DataLoader (if val_dir provided)
        }
    """
    train_dataset = TrajectoryDataset(train_dir, K=K, Q=Q)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,  # keeps workers alive between epochs
        prefetch_factor=4          # each worker preloads 4 batches ahead
    )
    
    result = {"train": train_loader}
    
    if val_dir is not None:
        val_dataset = TrajectoryDataset(val_dir, K=K, Q=Q)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4
        )

        result["val"] = val_loader
    
    logger.info(f"Train batches: {len(train_loader)}")
    if val_dir:
        logger.info(f"Val batches: {len(result['val'])}")
    
    return result


# ============================================================================
# Validation
# ============================================================================

def validate(runtime: dict, val_loader: DataLoader, delta: float) -> dict:
    """
    Validate model on validation set.
    
    Args:
        runtime: training runtime dict
        val_loader: validation dataloader
        delta: Huber loss threshold
    
    Returns:
        {
            "val_loss": float,
            "val_error": float
        }
    """
    from theta_model import predict_velocity, compute_huber_loss
    
    model = runtime["theta"]
    device = runtime["device"]
    
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            X_t = batch["X_t"].to(device)
            t = batch["t"].to(device)
            V = batch["V"].to(device)
            mask = batch.get("mask", None)
            if mask is not None:
                mask = mask.to(device)
            
            # Forward pass
            result = predict_velocity(model, X_t, t)
            v_theta = result["v_theta"]
            
            # Compute loss
            loss_result = compute_huber_loss(v_theta, V, delta=delta, mask=mask)
            
            total_loss += loss_result["loss"].item()
            total_error += loss_result["mean_error"]
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_error = total_error / num_batches if num_batches > 0 else 0.0
    
    logger.info(f"Validation - Loss: {avg_loss:.4f}, Error: {avg_error:.4f}")
    
    return {
        "val_loss": avg_loss,
        "val_error": avg_error
    }


# ============================================================================
# Main Training Loop
# ============================================================================

def main(args):
    # Create timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{args.model_type}_{timestamp}"
    
    # Create output directories with timestamp
    checkpoint_dir = Path("./bin/checkpoints") / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path("./log")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{model_name}.log"
    
    # Setup file logger with config header
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # === Write configuration header (new run) ===
    with open(log_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("theta Model Training Configuration\n")
        f.write("=" * 60 + "\n")
        for key, value in sorted(vars(args).items()):
            f.write(f"{key}: {value}\n")
        f.write("=" * 60 + "\n\n")
    
    logger.info(f"Training started: {model_name}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Log file: {log_file}")
    
    # Save training config to checkpoint dir
    config = vars(args)
    config['timestamp'] = timestamp
    config['model_name'] = model_name
    with open(checkpoint_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved config to {checkpoint_dir / 'config.json'}")
    
    # Load Huber delta
    delta_path = Path(args.huber_delta_path)
    if delta_path.exists():
        with open(delta_path, 'r') as f:
            delta_data = json.load(f)
            delta = delta_data["delta"]
        logger.info(f"Loaded Huber delta: {delta:.6f}")
    else:
        delta = args.delta_fallback
        logger.warning(f"Huber delta file not found, using fallback: {delta:.6f}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    dataloaders = create_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir if args.val_dir else None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        K=args.K,
        Q=args.Q
    )
    train_loader = dataloaders["train"]
    val_loader = dataloaders.get("val", None)
    
    # Calculate checkpoint interval in steps (>= 1000 chunks worth of data)
    samples_per_step = args.batch_size
    steps_per_1000_chunks = max(1, int(1000 * 5 / samples_per_step))  # 5 samples per chunk
    checkpoint_interval = max(steps_per_1000_chunks, args.save_interval)
    logger.info(f"Checkpoint interval: {checkpoint_interval} steps (~{checkpoint_interval * samples_per_step / 5:.0f} chunks)")
    
    # Build model
    logger.info(f"Building {args.model_type.upper()} model...")
    model_result = build_theta(
        model_type=args.model_type,
        coord_dim=2,
        hidden=args.hidden,
        layers=args.layers,
        K=args.K,
        dropout=args.dropout
    )
    model = model_result["model"]
    num_params = model_result["num_params"]
    
    logger.info(f"Model has {num_params:,} parameters")
    
    # Initialize training runtime
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    logger.info(f"Using device: {device}")
    
    runtime = init_training_runtime(
        model=model,
        device=device,
        lr=args.lr, 
        weight_decay=args.weight_decay,
        epochs=args.epochs
    )
    
    # Training loop
    logger.info("Starting training...")
    logger.info("="*60)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        logger.info("-"*60)
        
        # Train
        train_result = train_loop(
            runtime=runtime,
            dataloader=train_loader,
            epoch=epoch,
            delta=delta,
            save_dir=str(checkpoint_dir),
            log_interval=args.log_interval,
            save_interval=checkpoint_interval,  # Use calculated interval
            model_name=model_name
        )
        
        logger.info(
            f"Epoch {epoch} Training - "
            f"Avg Loss: {train_result['avg_loss']:.4f}, "
            f"Avg Error: {train_result['avg_error']:.4f}"
        )
    logger.info("\n" + "="*60)
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info("="*60)


# ============================================================================
# Resume Training Function
# ============================================================================

def resume_training(args, ckpt_path: str):
    logger.info(f"Resuming training from checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Recover epoch and step from filename
    import re
    match = re.search(r"ckpt_(\d+)_(\d+)", ckpt_path)
    start_epoch = int(match.group(1)) if match else ckpt.get("epoch", 0)
    start_step = int(match.group(2)) if match else ckpt.get("step", 0)
    logger.info(f"Resuming from epoch={start_epoch}, step={start_step}")

    # === Warmup skip logic based on resume step ===
    extra_args = {}
    if start_step >= 1000:
        extra_args["warmup_steps"] = 0
        logger.info("Skipping warm-up phase (already past 1000 steps).")
    else:
        extra_args["warmup_steps"] = 1000
        logger.info("Applying standard warm-up schedule (<=1000 steps).")

    # Use original model directory
    resume_dir = Path(ckpt_path).parent
    with open(resume_dir / "config.json", "r") as f:
        old_config = json.load(f)

    # Keep using old model directory and log file
    log_file = Path("./log") / f"{old_config['model_name']}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # === Append new runtime arguments (resume mode) ===
    with open(log_file, "a") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write("Resumed Training Configuration\n")
        f.write("=" * 60 + "\n")
        for key, value in sorted(vars(args).items()):
            f.write(f"{key}: {value}\n")
        f.write("=" * 60 + "\n\n")


    # Load huber delta
    delta_path = Path(args.huber_delta_path)
    if delta_path.exists():
        with open(delta_path, "r") as f:
            delta = json.load(f)["delta"]
    else:
        delta = args.delta_fallback
        logger.warning(f"Huber delta file not found, fallback={delta:.6f}")

    # Create dataloaders
    dataloaders = create_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir if args.val_dir else None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        K=args.K,
        Q=args.Q,
    )
    train_loader = dataloaders["train"]
    val_loader = dataloaders.get("val", None)

    # Rebuild model
    model_result = build_theta(
        model_type=old_config["model_type"],
        coord_dim=2,
        hidden=old_config["hidden"],
        layers=old_config["layers"],
        K=old_config["K"],
        dropout=old_config.get("dropout", 0.1),
    )
    model = model_result["model"]

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    runtime = init_training_runtime(
        model=model,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        **extra_args
    )

    # Load checkpoint states
    model.load_state_dict(ckpt["model_state_dict"])
    runtime["optimizer"].load_state_dict(ckpt["optimizer_state_dict"])
    logger.info("Checkpoint weights and optimizer state loaded successfully")

    # === Force reset LR to tripled runtime value ===
    for g in runtime["optimizer"].param_groups:
        g["lr"] = runtime["lr"]
    logger.info(f"Learning rate forcibly reset to tripled value: {runtime['lr']:.6e}")

    # Continue training from next epoch
    best_val_loss = float("inf")
    for epoch in range(start_epoch + 1, args.epochs + 1):
        logger.info(f"\nResumed Epoch {epoch}/{args.epochs}")
        train_result = train_loop(
            runtime=runtime,
            dataloader=train_loader,
            epoch=epoch,
            delta=delta,
            save_dir=str(resume_dir),
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            model_name=old_config["model_name"],
        )

    logger.info("Resumed training completed.")
    return {"status": "resumed", "start_epoch": start_epoch, "start_step": start_step}

# ============================================================================
# Command Line Interface
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train theta Rectified Flow Model")
    
    # Data
    parser.add_argument("--train_dir", type=str, default="./dataset/processed/train",
                       help="Path to training data directory")
    parser.add_argument("--val_dir", type=str, default="./dataset/processed/val",
                       help="Path to validation data directory")
    parser.add_argument("--K", type=int, default=256,
                       help="Chunk length")
    parser.add_argument("--Q", type=int, default=1,
                       help="Overlap length (buckle points)")
    
    # Model
    parser.add_argument("--model_type", type=str, default="nn",
                       choices=["nn", "cnn1d", "transformer"],
                       help="Model architecture")
    parser.add_argument("--hidden", type=int, default=512,
                       help="Hidden dimension size")
    parser.add_argument("--layers", type=int, default=6,
                       help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate")
    
    # Training
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--huber_delta_path", type=str, default="./dataset/state/huber_delta.json",
                       help="Path to Huber delta JSON file")
    parser.add_argument("--delta_fallback", type=float, default=38.13,
                       help="Fallback Huber delta if file not found")
    
    # System
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU training (ignore CUDA)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of dataloader workers")
    parser.add_argument("--log_interval", type=int, default=100,
                       help="Logging interval (steps)")
    parser.add_argument("--save_interval", type=int, default=1000,
                       help="Checkpoint save interval (steps, or auto-calculated for >=1000 chunks)")
    
    return parser.parse_args()



if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    args = parse_args()

    print("=" * 60)
    print("theta Model Training Interface")
    print("=" * 60)
    print("1. Train new model")
    print("2. Resume previous training")
    print("=" * 60)
    choice = input("Select mode (1 or 2): ").strip()

    if choice == "1":
        main(args)
    elif choice == "2":
        ckpt_path = input("Enter checkpoint path (e.g., ./bin/checkpoints/nn_20251026_111116/ckpt_1_30000.pt): ").strip()
        if not Path(ckpt_path).exists():
            print("❌ Checkpoint path not found.")
            sys.exit(1)
        resume_training(args, ckpt_path)
    else:
        print("Invalid selection. Exiting.")