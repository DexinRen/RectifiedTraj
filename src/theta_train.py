import gc
import math
import sys
import csv
import glob
import json
import torch
import shutil
import logging
import matplotlib
from time import time
from pathlib import Path
from logging import Logger
from datetime import datetime
import matplotlib.pyplot as plt
from safetensors.torch import save_file
from theta_model import count_parameters
from theta_model import build_theta_model
from utils.evaluations.validation import ckpt_audit
from utils.helpers.model_size_check import size_abbrv
from utils.evaluations.validation import quick_acc_test
from utils.evaluations.validation import reduce_point_error
from utils.data_loader_standalone import DataLoader
from utils.data_loader_standalone import build_online_eval_triplets

matplotlib.use('Agg')

# ================================================================
# === build_loss_mask
# ================================================================
def _normalize_loss_mask_policy(raw, default: str = "Q1=8pt") -> str:
    """Normalize and validate loss-mask policy token.

    Supported canonical values:
      - Q1=8pt: keep current head-masked policy.
      - Q1=0pt: disable head masking (all points weighted equally).
    """
    token = str(raw if raw is not None else "").strip().lower().replace(" ", "")
    if token in {"", "q1=8pt", "q1=8"}:
        return "Q1=8pt"
    if token in {"q1=0pt", "q1=0", "none", "nomask", "no_mask"}:
        return "Q1=0pt"
    raise ValueError(
        f"Unsupported loss_mask_policy={raw!r}. "
        "Supported values: Q1=8pt, Q1=0pt."
    )


def build_loss_mask(K: int, loss_mask_policy: str):
    """
    Purpose:
        Create the per-point loss mask for training based on policy.

        Policy: Q1=8pt
            HEAD (0..7):
                weight = 0.0  # all head points are attachment buckle

            BODY (8..K-1):
                weight = 1.0

        Policy: Q1=0pt
            all points weight = 1.0 (no loss masking)

    Inputs:
        K : int, chunk size (default = 256)

    Output:
        mask : Tensor(K,)

    Notes:
        - Mask applied inside train_step().
        - Stored in runtime["loss_mask"].
    """

    policy = _normalize_loss_mask_policy(loss_mask_policy)

    # allocate full mask
    mask = torch.ones(K, dtype=torch.float32)

    # ------------------------------------------------------------
    # Policy: no mask
    # ------------------------------------------------------------
    if policy == "Q1=0pt":
        return mask

    # ------------------------------------------------------------
    # Policy: Q1=8pt
    # HEAD REGION: p = 0..7 (all zero)
    # ------------------------------------------------------------
    mask[:8] = 0.0

    # ------------------------------------------------------------
    # BODY REGION: p = 8..K-1 (fully weighted = 1.0)
    # Tail masking was removed; only the head attachment buckle stays masked.
    # No operation needed since initialized to ones

    return mask


def _normalize_data_hypothesis(raw, default: str = "RectifiedTraj") -> str:
    token = str(raw if raw is not None else "").strip().lower().replace("-", "_")
    if token in {"", "rf", "rectified_flow", "rectified", "rectifiedtraj", "rectified_traj"}:
        return "RectifiedTraj"
    if token in {"rr", "residualreg", "residual_reg", "residual", "residual_regression"}:
        return "ResidualReg"
    text = str(raw).strip() if raw is not None else ""
    return text if text else str(default)


def _normalize_prediction_mode(raw, default: str = "offline") -> str:
    token = str(raw if raw is not None else "").strip().lower().replace("-", "_")
    if token in {"", "offline", "batch", "global"}:
        return "offline"
    if token in {"online", "causal", "streaming"}:
        return "online"
    raise ValueError(
        f"Unsupported prediction_mode={raw!r}. "
        "Supported values: offline, online."
    )


def _resolve_model_root_dir(config: dict) -> Path:
    base_text = str(config.get("model_root", "./bin/model")).strip() or "./bin/model"
    base = Path(base_text)
    data_hypothesis = _normalize_data_hypothesis(
        config.get("data_hypothesis", config.get("data_hypothetis", "RectifiedTraj"))
    )
    config["data_hypothesis"] = data_hypothesis
    prediction_mode = _normalize_prediction_mode(
        config.get(
            "prediction_mode",
            "online" if str(config.get("model_type", "")).strip().lower() in {"hybrid_online", "online_hybrid", "causal_hybrid"} else "offline",
        )
    )
    config["prediction_mode"] = prediction_mode
    loss_mask_policy = _resolve_loss_mask_policy(config)
    config["loss_mask_policy"] = loss_mask_policy

    # Keep explicit hypothesis/custom leaf roots as-is.
    if base.name.lower() in {"rectifiedtraj", "residualreg", "rectifiedtraj_no_chunk"}:
        return base

    # Default root policy routing.
    if base.as_posix().rstrip("/") in {"./bin/model", "bin/model"}:
        if prediction_mode == "online":
            return base / f"{data_hypothesis}_online"
        if data_hypothesis == "RectifiedTraj" and loss_mask_policy == "Q1=0pt":
            return base / "RectifiedTraj_no_chunk"
        return base / data_hypothesis

    # Caller supplied a custom non-default root.
    return base


def _resolve_loss_mask_policy(config: dict) -> str:
    """Resolve and persist canonical loss-mask policy in config."""
    prediction_mode = _normalize_prediction_mode(
        config.get(
            "prediction_mode",
            "online" if str(config.get("model_type", "")).strip().lower() in {"hybrid_online", "online_hybrid", "causal_hybrid"} else "offline",
        )
    )
    config["prediction_mode"] = prediction_mode
    if prediction_mode == "online":
        policy = "Q1=0pt"
    else:
        policy = _normalize_loss_mask_policy(config.get("loss_mask_policy", "Q1=8pt"))
    config["loss_mask_policy"] = policy
    return policy


def _resolve_input_coord_dim(config: dict) -> int:
    """Resolve model input channel count from config."""
    prediction_mode = _normalize_prediction_mode(
        config.get(
            "prediction_mode",
            "online" if str(config.get("model_type", "")).strip().lower() in {"hybrid_online", "online_hybrid", "causal_hybrid"} else "offline",
        )
    )
    config["prediction_mode"] = prediction_mode
    coord_dim = int(config["coord_dim"])
    input_coord_dim = int(config.get("input_coord_dim", coord_dim))
    if prediction_mode == "online":
        input_coord_dim = 3
    config["input_coord_dim"] = input_coord_dim
    return input_coord_dim


def _build_model_input(runtime: dict, x_t: torch.Tensor) -> torch.Tensor:
    """Build model input tensor according to active prediction mode."""
    input_coord_dim = int(runtime["config"].get("input_coord_dim", runtime["config"]["coord_dim"]))
    return x_t[:, :, :input_coord_dim].to(dtype=torch.float32)


def _ensure_train_log_schema(train_log_path: Path) -> None:
    """Upgrade train_data.csv to the current schema if needed."""
    if not train_log_path.exists():
        return

    with train_log_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    required_fields = {
        "ckpt_name",
        "epoch",
        "step",
        "loss_train",
        "loss_val",
        "acc_mean",
        "acc_median",
        "acc_std",
        "acc_tail",
        "lr",
    }
    if required_fields.issubset(set(fieldnames)):
        return

    new_fieldnames = [
        "ckpt_name",
        "epoch",
        "step",
        "loss_train",
        "loss_val",
        "acc_mean",
        "acc_median",
        "acc_std",
        "acc_tail",
        "lr",
    ]
    with train_log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        for row in rows:
            row["loss_train"] = row.get("loss_train", "")
            row["loss_val"] = row.get("loss_val", "")
            row["acc_tail"] = row.get("acc_tail", "")
            writer.writerow({key: row.get(key, "") for key in new_fieldnames})


# ================================================================
# === model_house_builder
# ================================================================
def model_house_builder(runtime):
    """
    Purpose:
        Create directory structure for model and logging.
        Save immutable config_init.json only once.
        Prepare all resolved paths inside runtime.
    """

    cfg = runtime["config"]
    model_name = runtime["model_name"]
    model_root_dir = runtime.get("model_root_dir")
    if model_root_dir is None:
        model_root = _resolve_model_root_dir(cfg)
    else:
        model_root = Path(str(model_root_dir))
    runtime["model_root_dir"] = str(model_root)
    base = model_root / model_name

    # create structure
    (base / "ckpts").mkdir(parents=True, exist_ok=True)
    (base / "log").mkdir(parents=True, exist_ok=True)

    runtime["ckpt_dir"]  = str(base / "ckpts")
    runtime["log_dir"]   = str(base / "log")
    runtime["config_path"]      = str(base / "log" / "config.json")
    runtime["config_init_path"] = str(base / "log" / "config_init.json")
    runtime["train_log"]        = str(base / "log" / "train_data.csv")

    # Save config_init.json ONCE
    init_cfg_path = Path(runtime["config_init_path"])
    if not init_cfg_path.exists():
        with open(init_cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

    # Always save current config.json (mutable)
    with open(runtime["config_path"], "w") as f:
        json.dump(cfg, f, indent=2)

    # Create empty train_data.csv if missing
    train_log_path = Path(runtime["train_log"])
    if not train_log_path.exists():
        with open(train_log_path, "w") as f:
            f.write(
                "ckpt_name,epoch,step,loss_train,loss_val,acc_mean,acc_median,acc_std,acc_tail,lr\n"
            )
    _ensure_train_log_schema(train_log_path)

    return runtime


def _resolve_training_device(config: dict) -> torch.device:
    """Resolve training device from config in a fail-fast way."""
    if bool(config.get("cpu", False)):
        return torch.device("cpu")

    device_token = str(config.get("device", "")).strip().lower()
    if device_token.startswith("cpu"):
        return torch.device("cpu")
    if device_token.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("config.device=cuda requested but CUDA is unavailable.")
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================
# === config_solver()
# ================================================================
def config_solver(runtime):
    """
    Purpose:
        Parse user choice (NEW or RESUME), load config.json,
        build all training directories, and populate runtime.

    Behavior:
        - NEW (choice=1):
            * Read ./src/config.json
            * Create model directory structure
            * Save config_init.json
            * runtime["start_epoch"] = 1

        - RESUME (choice=2):
            * User provides checkpoint path
            * Infer model directory
            * Load existing config.json inside model/log/
            * Parse train_data.csv to get last epoch
            * runtime["start_epoch"] = last_epoch + 1

    Returns:
        config (dict)
    """

    print("1) New training")
    print("2) Resume training")
    choice = input().strip()
    # ------------------------------------------------------------
    # NEW TRAINING
    # ------------------------------------------------------------
    if choice == "1":
        config_path = "./src/config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        data_hypothesis = _normalize_data_hypothesis(
            config.get("data_hypothesis", config.get("data_hypothetis", "RectifiedTraj"))
        )
        config["data_hypothesis"] = data_hypothesis
        _resolve_input_coord_dim(config)
        runtime["data_hypothesis"] = data_hypothesis
        runtime["loss_mask_policy"] = _resolve_loss_mask_policy(config)
        runtime["model_root_dir"] = str(_resolve_model_root_dir(config))
        
        # Generate model name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = config.get("model_type", "model")
        model_size = size_abbrv(config_path)
        runtime["model_name"] = f"{model_type}_{model_size}_{timestamp}"
        runtime["config"] = config
        
        # build full model house and update runtime
        model_house_builder(runtime)

        # NEW starts at epoch 1
        runtime["start_epoch"] = 1
        runtime["global_step"] = 0
        runtime["resume"] = False

    # ------------------------------------------------------------
    # RESUME TRAINING
    # ------------------------------------------------------------
    elif choice == "2":
        ckpt_path = input("Enter the path of checkpoint you want to resume (pt file): ").strip()
        ckpt_path = Path(ckpt_path)

        # Save checkpoint path for loading weights
        runtime["resume_ckpt_path"] = str(ckpt_path)

        # model dir = parent of ckpt folder
        ckpt_dir = ckpt_path.parent
        model_dir = ckpt_dir.parent
        log_dir = model_dir / "log"

        # Load config stored inside model/log/config.json
        config_path = log_dir / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        data_hypothesis = _normalize_data_hypothesis(
            config.get("data_hypothesis", config.get("data_hypothetis", "RectifiedTraj"))
        )
        config["data_hypothesis"] = data_hypothesis
        _resolve_input_coord_dim(config)
        runtime["data_hypothesis"] = data_hypothesis
        runtime["loss_mask_policy"] = _resolve_loss_mask_policy(config)
        
        runtime["config_path"] = config_path
        runtime["config_init_path"] = str(model_dir / "log" / "config_init.json")
        runtime["config"] = config
        runtime["model_dir"] = str(model_dir)
        runtime["model_root_dir"] = str(model_dir.parent)
        runtime["model_name"] = model_dir.name
        runtime["ckpt_dir"]  = str(ckpt_dir)
        runtime["log_dir"]   = str(log_dir)
        runtime["train_log"] = str(model_dir / "log" / "train_data.csv")
        # Parse last row of train_data.csv
        train_csv = log_dir / "train_data.csv"
        assert train_csv.exists(), "Missing train_data.csv for resume"

        last = parse_train_data_csv(train_csv)
        last_epoch = last["last_epoch"]
        last_epoch_step  = last["last_step"]

        # Next epoch (resume always starts new epoch)
        runtime["start_epoch"] = last_epoch + 1
        runtime["global_step"] = last_epoch_step 
        runtime["resume"] = True

    # ------------------------------------------------------------
    # Invalid input
    # ------------------------------------------------------------
    else:
        print("Invalid choice.")
        sys.exit(1)

    log_root = Path("./bin/log")
    log_root.mkdir(parents=True, exist_ok=True)
    runtime["logger"] = build_logger(str(log_root / "theta_train.log"), runtime)
    return config

# =============================================================
# === training_initializer(runtime)
# =============================================================
def _build_scheduler(optimizer, config: dict, total_steps_override: int | None = None):
    """Build warmup + cosine scheduler from config.

    Args:
        optimizer: AdamW optimizer instance.
        config: Runtime config dict.
        total_steps_override: Optional explicit total step count.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: scheduler object.
    """
    # ------------------------------------------------------------
    # Resolve schedule length
    # ------------------------------------------------------------
    if total_steps_override is None:
        total_steps = _optimizer_steps_per_epoch_from_config(config) * int(config["epochs"])
    else:
        total_steps = int(total_steps_override)
    warmup_steps = int(config.get("warmup_steps", 1000))

    if total_steps <= 0:
        raise ValueError("Total training steps must be > 0.")

    if total_steps == 1:
        warmup_steps = 0
    else:
        warmup_steps = max(0, min(warmup_steps, total_steps - 1))

    # ------------------------------------------------------------
    # Build scheduler
    # ------------------------------------------------------------
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
            eta_min=config["lr"] * 0.1,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps),
            eta_min=config["lr"] * 0.1,
        )

    return scheduler


def _scheduler_state_is_exhausted(scheduler_state: dict) -> bool:
    """Return True if loaded cosine schedule has reached/passed its T_max."""
    sub_states = scheduler_state.get("_schedulers", None)
    if isinstance(sub_states, list):
        for sub in sub_states:
            if isinstance(sub, dict) and "T_max" in sub and "last_epoch" in sub:
                return int(sub["last_epoch"]) >= int(sub["T_max"])

    if "T_max" in scheduler_state and "last_epoch" in scheduler_state:
        return int(scheduler_state["last_epoch"]) >= int(scheduler_state["T_max"])

    return False


def _reset_optimizer_lr(optimizer, lr: float) -> None:
    """Reset optimizer lr/initial_lr for every param group."""
    for group in optimizer.param_groups:
        group["lr"] = float(lr)
        group["initial_lr"] = float(lr)


def _optimizer_steps_per_epoch_from_config(config: dict) -> int:
    """Interpret config data_per_epoch as loaded rows and convert to batch updates."""
    rows_per_epoch = int(config["data_per_epoch"])
    batch_size = int(config["batch_size"])
    if rows_per_epoch <= 0 or batch_size <= 0:
        raise ValueError("data_per_epoch and batch_size must be > 0.")
    return max(1, math.ceil(rows_per_epoch / batch_size))


def training_initializer(runtime):
    """
    Purpose:
        Initialize model, optimizer, scheduler, loss mask.
        
    Logic:
        1. ALWAYS create model, optimizer, scheduler (for both new and resume)
        2. If resuming: load checkpoint states to restore training state
        3. If new: use fresh initialized states
    """
    
    config = runtime["config"]
    device = runtime["device"]
    
    # ================================================================
    # Block 1: Build model (ALWAYS - for both new and resume)
    # ================================================================
    model = build_theta_model(runtime)
    model = model.to(device)
    model.train()
    runtime["model"] = model
    
    # ================================================================
    # Block 2: Create optimizer (ALWAYS - for both new and resume)
    # ================================================================
    lr = config["lr"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    runtime["optimizer"] = optimizer
    
    # ================================================================
    # Block 3: Create scheduler (ALWAYS - for both new and resume)
    # ================================================================
    scheduler = _build_scheduler(optimizer, config)
    
    runtime["scheduler"] = scheduler
    
    # ================================================================
    # Block 4: Load checkpoint if resuming (ONLY if resume=True)
    # ================================================================
    if runtime.get("resume", False) and "resume_ckpt_path" in runtime:
        ckpt_path = runtime["resume_ckpt_path"]
        
        # Ensure we use the _full.pt file (not .safetensors)
        if ckpt_path.endswith(".safetensors"):
            ckpt_path = ckpt_path.replace(".safetensors", "_full.pt")
        
        runtime["logger"].info(f"[Resume] Loading checkpoint: {ckpt_path}")
        
        # Load checkpoint file
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Restore model + optimizer states
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler_state = checkpoint["scheduler_state_dict"]

        restart_lr = config.get("restart_lr", None)
        if restart_lr is not None:
            restart_lr = float(restart_lr)
            if restart_lr <= 0.0:
                raise ValueError(f"restart_lr must be > 0, got {restart_lr}")

        # ------------------------------------------------------------
        # Rebuild remaining schedule if:
        #   1. caller explicitly requests a restart_lr override, or
        #   2. old scheduler is already exhausted.
        # ------------------------------------------------------------
        if restart_lr is not None or _scheduler_state_is_exhausted(scheduler_state):
            resume_epochs = max(1, int(config["epochs"]) - int(runtime["start_epoch"]) + 1)
            resume_steps = _optimizer_steps_per_epoch_from_config(config) * resume_epochs
            if restart_lr is not None:
                resume_base_lr = restart_lr
            else:
                # Continue from checkpoint LR instead of restarting at base config LR.
                # This avoids a sharp LR jump when extending training after schedule end.
                resume_base_lr = float(optimizer.param_groups[0].get("lr", config["lr"]))
                if resume_base_lr <= 0.0:
                    resume_base_lr = float(config["lr"])
            _reset_optimizer_lr(optimizer, resume_base_lr)

            resume_config = dict(config)
            resume_config["lr"] = resume_base_lr
            resume_config["warmup_steps"] = 0

            scheduler = _build_scheduler(
                optimizer,
                resume_config,
                total_steps_override=resume_steps,
            )
            runtime["scheduler"] = scheduler
            if restart_lr is not None:
                runtime["logger"].info(
                    "[Resume] restart_lr override detected. "
                    f"Reset optimizer LR to {resume_base_lr:.8f} and rebuilt schedule "
                    f"for remaining_steps={resume_steps}."
                )
            else:
                runtime["logger"].info(
                    "[Resume] Scheduler state is exhausted. "
                    f"Rebuilt schedule from checkpoint LR={resume_base_lr:.8f} "
                    f"for remaining_steps={resume_steps}."
                )
        else:
            scheduler.load_state_dict(scheduler_state)
            runtime["logger"].info(f"[Resume] Scheduler state loaded (LR schedule restored)")
        
        # Restore counters from checkpoint so resume target follows selected ckpt.
        runtime["global_step"] = int(checkpoint.get("global_step", runtime.get("global_step", 0)))
        ckpt_epoch = checkpoint.get("epoch", None)
        if ckpt_epoch is not None:
            ckpt_start_epoch = int(ckpt_epoch) + 1
            if ckpt_start_epoch != int(runtime.get("start_epoch", ckpt_start_epoch)):
                runtime["logger"].warning(
                    "[Resume] start_epoch from train_data.csv does not match selected checkpoint. "
                    f"Using checkpoint-derived start_epoch={ckpt_start_epoch}."
                )
            runtime["start_epoch"] = ckpt_start_epoch
        
        runtime["logger"].info(f"[Resume] Model weights loaded")
        runtime["logger"].info(f"[Resume] Optimizer state loaded (momentum restored)")
        runtime["logger"].info(
            f"[Resume] Resuming from epoch={runtime.get('start_epoch', 1)} "
            f"global_step={runtime['global_step']}"
        )
    
    # ================================================================
    # Block 5: Loss mask (ALWAYS)
    # ================================================================
    K = config["K"]
    loss_mask_policy = runtime.get("loss_mask_policy", _resolve_loss_mask_policy(config))
    runtime["loss_mask_policy"] = loss_mask_policy
    if loss_mask_policy == "Q1=0pt":
        runtime["loss_mask"] = None
    else:
        loss_mask = build_loss_mask(K, loss_mask_policy).to(device)
        runtime["loss_mask"] = loss_mask
    
    # ================================================================
    # Block 6: Step & epoch counters (ALWAYS)
    # ================================================================
    if runtime.get("resume", False):
        runtime["step"] = 0
        runtime["epoch"] = runtime["start_epoch"]
    else:
        runtime["step"] = runtime["global_step"]
        runtime["epoch"] = 1

    runtime["best_val_loss"] = runtime.get("best_val_loss", None)
    runtime["bad_val_count"] = int(runtime.get("bad_val_count", 0))
    runtime["early_stop_triggered"] = False
    
    return runtime


# ================================================================
# === train_step
# ================================================================
def train_step(runtime, batch):
    """
    Purpose:
        Execute one training step:
            - Forward pass
            - Masked point-wise L2 loss
            - Backprop + optimizer update
            - Scheduler update

    Returns:
        {
            "loss": loss.item(),
            "mean_error": mean_error,
            "lr": optimizer.param_groups[0]["lr"]
        }

    Notes:
        - Batch MUST be provided by training_manager().
        - Target semantics depend on runtime["data_hypothesis"]:
            RectifiedTraj -> target V
            ResidualReg  -> target X0
        - No defensive programming. Fail-fast.
    """
    model      = runtime["model"]
    optimizer  = runtime["optimizer"]
    scheduler  = runtime["scheduler"]
    loss_mask  = runtime["loss_mask"]          # (K,) or (B,K)

    device = runtime["device"]
    model.train()

    valid_mask = None
    if len(batch) == 4:
        X_t, y_true, t, valid_mask = batch
    else:
        X_t, y_true, t = batch
    X_t = X_t.to(device, dtype=torch.float32)
    y_true = y_true.to(device, dtype=torch.float32)
    t = t.to(device, dtype=torch.float32)
    if valid_mask is not None:
        valid_mask = valid_mask.to(device)

    X_t_input = _build_model_input(runtime, X_t)

    # Forward
    y_pred = model(X_t_input, t)

    # Loss (torch API)
    diff = y_pred - y_true
    error = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)  # (B, K)

    loss, mean_error, _ = reduce_point_error(
        error,
        loss_mask=loss_mask,
        valid_mask=valid_mask,
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()

    lr_floor = float(runtime["config"].get("lr_floor", 5e-6))

    for group in optimizer.param_groups:
        if group["lr"] < lr_floor:
            group["lr"] = lr_floor
        
    lr_now = optimizer.param_groups[0]["lr"]

    return {
        "loss": loss.item(),
        "mean_error": mean_error.item(),
        "lr": lr_now,
    }


# ================================================================
# === save_checkpoint_and_log
# ================================================================
def save_checkpoint_and_log(runtime, avg_loss):
    """
    Purpose:
        Save checkpoint, update logs, and generate plots.
        Called every N steps during training.
        
    Saves TWO files:
        1. .safetensors - Model weights only (for inference/distribution)
        2. _full.pt     - Complete training state (for resuming)
        
    Steps:
        1. Save model weights as .safetensors
        2. Save full checkpoint as _full.pt (with optimizer/scheduler)
        3. Append metrics to train_data.csv
        4. Update config.json
        5. Generate all plots by calling plot_training_metrics()
        
    Inputs:
        runtime: dict with all training state
        avg_loss: float, average loss for this checkpoint
        
    Returns:
        ckpt_name: str, name of saved checkpoint
    """
    
    config = runtime["config"]
    model = runtime["model"]
    optimizer = runtime["optimizer"]
    scheduler = runtime["scheduler"]
    
    epoch = runtime["current_epoch"]
    global_step = runtime["global_step"]
    ckpt_dir = Path(runtime["ckpt_dir"])
    log_dir = Path(runtime["log_dir"])
    
    # ================================================================
    # STEP 1: Save model weights as safetensors
    # ================================================================
    ckpt_name = f"ckpt_e{epoch}_s{global_step}.safetensors"
    ckpt_path = ckpt_dir / ckpt_name
    
    # Save ONLY model weights (safetensors format)
    # Used for: inference, distribution, model sharing
    save_file(model.state_dict(), str(ckpt_path))
    
    # ================================================================
    # STEP 2: Save full checkpoint with optimizer/scheduler
    # ================================================================
    full_ckpt_name = f"ckpt_e{epoch}_s{global_step}_full.pt"
    full_ckpt_path = ckpt_dir / full_ckpt_name
    
    # Save COMPLETE training state (PyTorch pickle format)
    # Used for: resuming training
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
        "lr": optimizer.param_groups[0]["lr"],
    }, str(full_ckpt_path))
    
    # ================================================================
    # STEP 3: Append to train_data.csv
    # ================================================================
    train_log_path = Path(runtime["train_log"])
    
    acc_mean = runtime.get("acc_mean", None)
    acc_median = runtime.get("acc_median", None)
    acc_std = runtime.get("acc_std", None)
    acc_tail = runtime.get("acc_tail", None)
    loss_val = runtime.get("loss_val", None)
    lr = optimizer.param_groups[0]["lr"]

    def _acc_to_csv(v):
        return "" if v is None else f"{v:.6f}"

    def _acc_to_str(v):
        return "N/A" if v is None else f"{v:.6f}"
    
    with open(train_log_path, "a", newline="") as f:
        f.write(
            f"{ckpt_name},{epoch},{global_step},{avg_loss:.6f},{_acc_to_csv(loss_val)},"
            f"{_acc_to_csv(acc_mean)},{_acc_to_csv(acc_median)},{_acc_to_csv(acc_std)},"
            f"{_acc_to_csv(acc_tail)},"
            f"{lr:.8f}\n"
        )
    
    # ================================================================
    # STEP 4: Update config.json (keep latest state)
    # ================================================================
    config_path = Path(runtime["config_path"])
    config["last_checkpoint"] = ckpt_name
    config["last_global_step"] = global_step
    config["last_epoch"] = epoch
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    # ================================================================
    # STEP 5: Generate plots (automatically called here!)
    # ================================================================
    plot_training_metrics(runtime)
    runtime["logger"].info(
        f"[CKPT SAVE] ckpt_e{epoch}_s{global_step} \n\t| Train Loss: {avg_loss:.6f} | "
        f"Val Loss: {_acc_to_str(loss_val)} | "
        f"LR: {lr:.8f} | acc_mean: {_acc_to_str(acc_mean)} | "
        f"acc_med: {_acc_to_str(acc_median)} | acc_std: {_acc_to_str(acc_std)} | "
        f"acc_tail: {_acc_to_str(acc_tail)}"
    )
    
    return ckpt_name


# ================================================================
# === converge_detector
# ================================================================
def converge_detector(runtime, val_loss):
    """
    Purpose:
        Validation-loss patience early stopping.

    Returns:
        True  -> stop training
        False -> continue training
    """
    if val_loss is None:
        return False

    patience = int(runtime["config"].get("early_stop_patience", 20))
    min_delta = float(runtime["config"].get("early_stop_min_delta", 0.01))

    best_val_loss = runtime.get("best_val_loss", None)
    if best_val_loss is None or val_loss < (best_val_loss - min_delta):
        runtime["best_val_loss"] = float(val_loss)
        runtime["bad_val_count"] = 0
        runtime["logger"].info(
            f"[Early Stop] Validation improved: val_loss={val_loss:.6f} "
            f"(best={runtime['best_val_loss']:.6f}, min_delta={min_delta:.6f})"
        )
        return False

    runtime["bad_val_count"] = int(runtime.get("bad_val_count", 0)) + 1
    bad_val_count = runtime["bad_val_count"]
    runtime["logger"].info(
        f"[Early Stop] No significant val_loss improvement: "
        f"current={val_loss:.6f} best={best_val_loss:.6f} "
        f"count={bad_val_count}/{patience} min_delta={min_delta:.6f}"
    )

    if bad_val_count >= patience:
        runtime["early_stop_triggered"] = True
        runtime["logger"].info(
            f"[Early Stop] Triggered after {bad_val_count} consecutive "
            f"non-improving validation checkpoints."
        )
        return True

    return False


# ================================================================
# === training_manager  (with progress display + accuracy per ckpt)
# ================================================================
def training_manager(runtime):
    config     = runtime["config"]
    dataloader = runtime["dataloader"]

    num_epochs     = config["epochs"]
    batch_size     = int(config["batch_size"])
    save_every_rows = int(config["save_every"])
    save_every     = max(1, save_every_rows // batch_size)

    # Resume always starts new epoch, step=0
    start_epoch      = runtime.get("start_epoch", 1)
    runtime["step"]  = runtime["global_step"]

    num_trainable = count_parameters(runtime["model"]) 
    if runtime["config"]["terminal_print"] == True:
        message = runtime["config"]["model_type"]
        print(f"Model: {message}")
        print(f"Total Trainable Parameters: {num_trainable:,}")

    should_stop = False
    for epoch in range(start_epoch, num_epochs + 1):
        total_loss  = 0.0
        total_error = 0.0
        num_steps   = 0

        # dynamic epoch slice
        dataloader.set(epoch-1)
        steps_per_ep = dataloader.batches_per_epoch

        start_time = time()

        runtime["current_epoch"] = epoch
        
        # ------------------------------------------------------------
        # Training steps
        # ------------------------------------------------------------
        for step_idx in range(1, steps_per_ep + 1):

            batch  = dataloader.get_batch()
            result = train_step(runtime, batch)

            runtime["global_step"] += 1
            loss       = result["loss"]
            mean_error = result["mean_error"]
            lr_now     = result["lr"]


            # aggregates
            num_steps   += 1
            total_loss  += loss
            total_error += mean_error

            avg_loss  = total_loss / num_steps
            avg_error = total_error / num_steps

            elapsed   = time() - start_time
            speed     = num_steps / elapsed if elapsed > 0 else 0.0

            progress   = step_idx / steps_per_ep
            filled     = int(40 * progress)
            bar        = "█" * filled + "░" * (40 - filled)

            # ---- 2-line live display ----
            if runtime["config"]["terminal_print"] == True:
                sys.stdout.write("\r\033[K")
                sys.stdout.write(
                    f"Epoch {epoch} | Step {step_idx}/{steps_per_ep} "
                    f"[{bar}] {progress*100:.1f}%\n"
                )
                sys.stdout.write("\r\033[K")
                sys.stdout.write(
                    f"Loss {loss:.4f} | Avg {avg_loss:.4f} | "
                    f"Err {avg_error:.4f} | LR {lr_now:.3e} | "
                    f"Speed {speed:.1f} steps/s"
                )
                sys.stdout.write("\033[F")
                sys.stdout.flush()

            # ------------------------------------------------------------
            # Checkpoint + validation condition
            # ------------------------------------------------------------
            if runtime["global_step"] % save_every == 0 and runtime["global_step"] > 0:
                if runtime["config"]["terminal_print"] == True:
                    # Clear the 2-line progress display before logging/validation output.
                    sys.stdout.write("\r\033[K\033[B\r\033[K")
                    sys.stdout.flush()

                # --- quick validation BEFORE saving ---
                val_loss, acc_mean, acc_median, acc_std, acc_tail = quick_acc_test(
                    runtime,
                    epoch,
                    runtime["global_step"],
                )
                runtime["loss_val"]    = val_loss
                runtime["acc_mean"]   = acc_mean
                runtime["acc_median"] = acc_median
                runtime["acc_std"]    = acc_std
                runtime["acc_tail"]   = acc_tail
                runtime["model"].train()
                # --- save checkpoint with accurate metrics ---
                save_checkpoint_and_log(runtime, avg_loss)
                trim_checkpoints(runtime)
                if converge_detector(runtime, val_loss):
                    should_stop = True
                    break
        # cleanup newlines after epoch
        if runtime["config"]["terminal_print"] == True:
            sys.stdout.write("\r\033[K\n\033[K")
        if should_stop:
            break


# ================================================================
# === HELPER FUNCTIONS
# ================================================================
def cleanup_memory():
    """
    Purpose:
        - Clear CPU + GPU memory at program start and between heavy phases.

    Behavior:
        - torch.cuda.empty_cache()
        - torch.cuda.ipc_collect()
        - gc.collect()

    Error handling:
        - If anything fails, write the error into runtime["logger"]
          and re-raise immediately (fail-fast).
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def read_config(config_path):
    """
    Purpose:
        Load a JSON configuration file and return it as a dict.

    Inputs:
        config_path : str or Path

    Returns:
        config : dict

    Behavior:
        - No defaults, no fallbacks.
        - Raise error if file missing or JSON invalid.
    """
    with Path(config_path).open("r") as f:
        return json.load(f)


def parse_train_data_csv(csv_path):
    """
    Minimal CSV parser.
    Reads ONLY the last row and returns:
        last_step, last_epoch

    No defensive checks. If something is wrong, let it fail.
    """

    last_row = None
    with Path(csv_path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last_row = row
    
    if last_row is None:
        raise ValueError(f"Empty CSV file: {csv_path}")
    
    return {
        "last_step": int(last_row["step"]),
        "last_epoch": int(last_row["epoch"]),
    }


def build_logger(log_file: str, runtime) -> Logger:
    """
    Purpose:
        Build a logger that writes to both console and file.

    Behavior:
        - Overwrites existing log file
        - No defensive programming
    """
    logger = logging.getLogger("theta_train")
    logger.setLevel(logging.INFO)

    # Remove old handlers if reinitializing
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    # File handler
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.INFO)

    # Console handler
    if runtime["config"]["terminal_print"] == True:    
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

    # Formatter
    fmt = logging.Formatter("%(asctime)s — %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if runtime["config"]["terminal_print"] == True:
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger


def plot_training_metrics(runtime):
    """
    Purpose:
        Generate training plots showing optimization progress.
        Called automatically by save_checkpoint_and_log().
        
    Plots generated:
        1. loss_vs_step.png - Training loss curve
        2. acc_mean_vs_step.png - Validation accuracy mean
        3. acc_std_vs_step.png - Validation accuracy std dev
        4. acc_combined_vs_step.png - All accuracy metrics together
        
    Each plot includes:
        - X-axis: Global Step (diagonal labels)
        - Y-axis: Metric value
        - Model info box: architecture, parameters
        - Grid for readability
        
    Saves to: ./bin/model/<model_name>/fig/*.png
    """
    
    # ================================================================
    # Parse CSV to get all historical data
    # ================================================================
    train_log_path = Path(runtime["train_log"])
    
    if not train_log_path.exists():
        return  # No data to plot yet
    
    # Read all rows from CSV
    global_steps = []
    train_losses = []
    val_losses = []
    acc_means = []
    acc_medians = []
    acc_stds = []
    acc_tails = []
    with open(train_log_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            global_steps.append(int(row["step"]))
            loss_train_text = str(row["loss_train"]).strip()
            train_losses.append(float(loss_train_text))
            val_loss_text = str(row["loss_val"]).strip()
            val_losses.append(float(val_loss_text) if val_loss_text else float("nan"))
            acc_means.append(float(row["acc_mean"]))
            acc_medians.append(float(row["acc_median"]))
            acc_stds.append(float(row["acc_std"]))
            acc_tails.append(float(row.get("acc_tail", row["acc_median"])))
    
    if len(global_steps) == 0:
        return  # No data to plot
    
    # ================================================================
    # Get model info for info box
    # ================================================================
    model = runtime["model"]
    config = runtime["config"]
    
    model_type = config["model_type"]
    hidden = config["hidden"]
    layers = config["layers"]
    num_params = count_parameters(model)
    
    # Format parameters (e.g., 1.2M, 45.3K)
    if num_params >= 1_000_000:
        param_str = f"{num_params / 1_000_000:.2f}M"
    elif num_params >= 1_000:
        param_str = f"{num_params / 1_000:.1f}K"
    else:
        param_str = f"{num_params}"
    
    model_info_text = f"Model: {model_type}\nHidden: {hidden}\nLayers: {layers}\nParams: {param_str}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    
    # ================================================================
    # Create figure directory
    # ================================================================
    fig_dir = Path(runtime["log_dir"]).parent / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # ================================================================
    # PLOT 1: Loss vs Global Step
    # ================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(global_steps, train_losses, marker='o', linewidth=2, markersize=4, color='#E74C3C', label='Train Loss')
    ax.plot(global_steps, val_losses, marker='s', linewidth=2, markersize=4, color='#2980B9', label='Validation Loss')
    ax.set_xlabel('Global Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Train and Validation Loss vs Global Step', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.tick_params(axis='x', rotation=45)
    
    # Add model info box (top right)
    ax.text(0.98, 0.98, model_info_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=props, family='monospace')
    
    plt.tight_layout()
    plt.savefig(fig_dir / "loss_vs_step.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # ================================================================
    # PLOT 2: Accuracy Mean vs Global Step
    # ================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(global_steps, acc_means, marker='o', linewidth=2, markersize=4, color='#3498DB')
    ax.set_xlabel('Global Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy Mean (L1 Error)', fontsize=12, fontweight='bold')
    ax.set_title('Validation Accuracy Mean vs Global Step', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='x', rotation=45)
    
    ax.text(0.98, 0.98, model_info_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=props, family='monospace')
    
    plt.tight_layout()
    plt.savefig(fig_dir / "acc_mean_vs_step.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # ================================================================
    # PLOT 3: Accuracy Std vs Global Step
    # ================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(global_steps, acc_stds, marker='o', linewidth=2, markersize=4, color='#9B59B6')
    ax.set_xlabel('Global Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy Std Dev', fontsize=12, fontweight='bold')
    ax.set_title('Validation Accuracy Std Dev vs Global Step', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='x', rotation=45)
    
    ax.text(0.98, 0.98, model_info_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=props, family='monospace')
    
    plt.tight_layout()
    plt.savefig(fig_dir / "acc_std_vs_step.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # ================================================================
    # PLOT 4: Combined Accuracy Metrics
    # ================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(global_steps, acc_means, marker='o', linewidth=2, markersize=4, 
            color='#3498DB', label='Mean')
    ax.plot(global_steps, acc_medians, marker='s', linewidth=2, markersize=4, 
            color='#2ECC71', label='Median')
    ax.plot(global_steps, acc_stds, marker='^', linewidth=2, markersize=4, 
            color='#9B59B6', label='Std Dev')
    ax.plot(global_steps, acc_tails, marker='D', linewidth=2, markersize=4,
            color='#F39C12', label='Tail')
    
    ax.set_xlabel('Global Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy Metrics', fontsize=12, fontweight='bold')
    ax.set_title('Validation Accuracy Metrics vs Global Step', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.tick_params(axis='x', rotation=45)
    
    # Model info box (moved to upper left to avoid legend)
    ax.text(0.02, 0.98, model_info_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', horizontalalignment='left',
            bbox=props, family='monospace')
    
    plt.tight_layout()
    plt.savefig(fig_dir / "acc_combined_vs_step.png", dpi=150, bbox_inches='tight')
    plt.close()




def trim_checkpoints(runtime, keep_last=None):
    """
    Trim checkpoint directory to keep:
        - best 2 by acc_mean (lowest)
        - best 2 by acc_median (lowest)
        - last N checkpoints (keep_last)
    Total target: 11 checkpoints.

    Inputs:
        runtime: dict containing:
            runtime["ckpt_dir"]
            runtime["train_log"]

    Behavior:
        - Reads train_data.csv
        - Determines which checkpoints to keep
        - Deletes all others (both .safetensors and _full.pt pairs)
    """

    ckpt_dir = Path(runtime["ckpt_dir"])
    csv_path = Path(runtime["train_log"])

    # ------------------------------------------------------------
    # Parse CSV
    # ------------------------------------------------------------
    prediction_mode = _normalize_prediction_mode(runtime["config"].get("prediction_mode", "offline"))
    if keep_last is None:
        keep_last = 3 if prediction_mode == "online" else 7

    rows = []
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "ckpt": row["ckpt_name"],
                "step": int(row["step"]),
                "mean": float(row["acc_mean"]),
                "median": float(row["acc_median"]),
                "tail": float(row.get("acc_tail", row["acc_median"])),
            })

    if len(rows) == 0:
        return  # nothing to trim

    # ------------------------------------------------------------
    # Sort for best metrics
    # ------------------------------------------------------------
    rows_by_mean = sorted(rows, key=lambda r: r["mean"])
    rows_by_aux = sorted(rows, key=lambda r: r["tail" if prediction_mode == "online" else "median"])
    rows_by_step = sorted(rows, key=lambda r: r["step"])

    keep_set = set()

    # keep the 1st, 2nd and 3rd best mean, med
    for i in range(3):
        if len(rows_by_mean) > i:
            keep_set.add(rows_by_mean[i]["ckpt"])
        if len(rows_by_aux) > i:
            keep_set.add(rows_by_aux[i]["ckpt"])

    # last N checkpoints
    for r in rows_by_step[-keep_last:]:
        keep_set.add(r["ckpt"])

    # ------------------------------------------------------------
    # Map ckpt.safetensors → ckpt_full.pt
    # ------------------------------------------------------------
    def paired_files(ckpt_name):
        """
        Given:
            ckpt_e3_s12000.safetensors
        Return:
            (safetensors_path, full_pt_path)
        """
        p = ckpt_dir / ckpt_name
        full = ckpt_name.replace(".safetensors", "_full.pt")
        return p, ckpt_dir / full

    # ------------------------------------------------------------
    # Delete everything NOT in keep_set
    # ------------------------------------------------------------
    all_ckpts = list(ckpt_dir.glob("*.safetensors"))
    for ckpt_file in all_ckpts:
        if ckpt_file.name not in keep_set:
            safepath, fullpath = paired_files(ckpt_file.name)

            if safepath.exists():
                safepath.unlink()
            if fullpath.exists():
                fullpath.unlink()

    runtime["logger"].info(f"[Trim] Checkpoints trimmed. Kept {len(keep_set)} checkpoints.")


def _resolve_existing_model_dir(model_name: str, model_root: str | None = None) -> Path:
    if model_root:
        root = Path(str(model_root))
        candidate = root / model_name
        if candidate.exists():
            return candidate
    roots = [
        Path("./bin/model/RectifiedTraj_online"),
        Path("./bin/model/ResidualReg_online"),
        Path("./bin/model/RectifiedTraj"),
        Path("./bin/model/RectifiedTraj_no_chunk"),
        Path("./bin/model/ResidualReg"),
        Path("./bin/model"),
    ]
    for root in roots:
        candidate = root / model_name
        if candidate.exists():
            return candidate
    if model_root:
        return Path(str(model_root)) / model_name
    return Path("./bin/model/RectifiedTraj") / model_name


def pick_best_checkpoint(model_name: str, model_root: str | None = None):
    model_dir = _resolve_existing_model_dir(model_name, model_root=model_root)
    csv_path = model_dir / "log" / "train_data.csv"
    cfg_path = model_dir / "log" / "config.json"
    rows = []

    with cfg_path.open("r") as f:
        cfg = json.load(f)

    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "ckpt": r["ckpt_name"],
                "median": float(r["acc_median"]),
                "mean": float(r["acc_mean"]),
                "tail": float(r.get("acc_tail", r["acc_median"])),
                "step": int(r["step"]),
            })

    if not rows:
        raise RuntimeError(f"No checkpoint records found for {model_name}")

    prediction_mode = _normalize_prediction_mode(cfg.get("prediction_mode", "offline"))
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            r["tail"] if prediction_mode == "online" else r["median"],
            r["mean"],
            -r["step"],
        )
    )

    best = rows_sorted[0]["ckpt"]
    return best.replace(".safetensors", "_full.pt")


def export_best_checkpoint(model_name: str, ckpt_full_name: str, model_root: str | None = None):
    """
    Create ./bin/model//<model_name>/best_ckpt/
    Clear existing files
    Copy best _full.pt and matching .safetensors
    """

    model_dir = _resolve_existing_model_dir(model_name, model_root=model_root)
    ckpt_dir  = model_dir / "ckpts"
    best_dir  = model_dir / "best_ckpt"

    # 1. Create best_ckpt dir if missing
    best_dir.mkdir(parents=True, exist_ok=True)

    # 2. Clear old files
    for f in best_dir.iterdir():
        if f.is_file():
            f.unlink()

    # 3. Copy _full.pt
    src_full = ckpt_dir / ckpt_full_name
    shutil.copy2(src_full, best_dir / ckpt_full_name)

    # 4. Copy the safetensors companion file
    safetensors_name = ckpt_full_name.replace("_full.pt", ".safetensors")
    src_safe = ckpt_dir / safetensors_name
    if src_safe.exists():
        shutil.copy2(src_safe, best_dir / safetensors_name)


# ================================================================
# === main()
# ================================================================
def main():
    """
    Purpose:
        Entry point for the theta training system.
        Implements STEP 11 of the protocol.

        Responsibilities:
            1. cleanup_memory()
            2. Initialize logger
            3. Resolve configuration via config_solver()
            4. Construct runtime dict
            5. Build loss mask
            6. Initialize model, optimizer, scheduler, dataloader
            7. Load quick validation set
            8. Run training_manager()
            9. Exit cleanly

        Notes:
            - Fail-fast behavior: any error aborts execution.
            - No defensive programming.
    """

    # ------------------------------------------------------------
    # Block 1: Memory cleanup at program start
    # ------------------------------------------------------------
    cleanup_memory()
    runtime = {}

    # ------------------------------------------------------------
    # Block 3: Resolve configuration (NEW or RESUME)
    # ------------------------------------------------------------
    # config_solver() will:
    #   - read config.json
    #   - build model directory
    #   - set runtime paths

    config_solver(runtime)   # runtime is filled inside

    # ------------------------------------------------------------
    # Block 4: Device
    # ------------------------------------------------------------
    device = _resolve_training_device(runtime["config"])
    runtime["device"] = device

    # ------------------------------------------------------------
    # Block 6: Initialize training modules
    # ------------------------------------------------------------
    # training_initializer():
    #   - build model
    #   - build optimizer
    #   - build scheduler
    #   - build dataloader
    #   - fill runtime state
    runtime["dataloader"] = DataLoader(
        mode="train",
        data_dir=runtime["config"]["train_dir"],
        batch_size=runtime["config"]["batch_size"],
        device=runtime["device"],
        data_per_epoch=runtime["config"]["data_per_epoch"],
        shuffle=True,
        data_hypothesis=runtime.get(
            "data_hypothesis",
            runtime["config"].get("data_hypothesis", runtime["config"].get("data_hypothetis", "RectifiedTraj")),
        ),
        prediction_mode=runtime["config"].get("prediction_mode", "offline"),
        target_k=runtime["config"]["K"],
        online_pad_prob=runtime["config"].get("online_pad_prob", 0.10),
    )
    training_initializer(runtime)

    # ------------------------------------------------------------
    # Block 7: Load quick validation set
    # ------------------------------------------------------------
    # config["quick_val_path"] must be present
    val_path = runtime["config"]["quick_val_path"]

    val_blob  = torch.load(val_path, map_location="cpu")
    if _normalize_prediction_mode(runtime["config"].get("prediction_mode", "offline")) == "online":
        X_t_val, V_val, t_val, valid_mask_val = build_online_eval_triplets(
            val_blob["X_t"].to(dtype=torch.float32),
            val_blob["V"].to(dtype=torch.float32),
            val_blob["t"].to(dtype=torch.float32),
            target_k=runtime["config"]["K"],
            data_hypothesis=runtime["data_hypothesis"],
        )
    else:
        x_t_val_raw = val_blob["X_t"].to(dtype=torch.float32)
        v_val_raw = val_blob["V"].to(dtype=torch.float32)
        t_val_raw = val_blob["t"].to(dtype=torch.float32)
        if runtime["data_hypothesis"] == "ResidualReg":
            t_view = t_val_raw.reshape(-1, 1, 1).to(dtype=x_t_val_raw.dtype)
            x0_val = x_t_val_raw[:, :, :2] - v_val_raw[:, :, :2] * t_view
            x1_val = x_t_val_raw[:, :, :2] + v_val_raw[:, :, :2] * (1.0 - t_view)
            X_t_val = x1_val
            V_val = x0_val
            t_val = torch.ones((t_val_raw.shape[0], 1), dtype=t_val_raw.dtype)
        else:
            X_t_val = x_t_val_raw[:, :, :2]   # keep only EN coords
            V_val = v_val_raw
            t_val = t_val_raw
        valid_mask_val = None

    runtime["val_data"] = {
        "X_t": X_t_val,
        "V":   V_val,
        "t":   t_val,
        "valid_mask": valid_mask_val,
    }

    # ------------------------------------------------------------
    # Block 8: Start training loop
    # ------------------------------------------------------------
    training_manager(runtime)

    # ---- Post-train Eval ----
    ckpt_audit(
        runtime["model_name"],
        big_path="./dataset/processed/NUMOSIM_Kanto/val/quick_val_chunk_90k.pt",
        device=str(runtime["device"]),
        model_root=runtime.get("model_root_dir", "./bin/model/RectifiedTraj"),
    )
    # ------------------------------------------------------------
    # Block 9: Final message
    # ------------------------------------------------------------
    runtime["logger"].info("Training complete. Exiting.")

if __name__ == "__main__":
    main()
