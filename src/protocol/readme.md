# 1. FILE STRUCTURE
---
./src/config.json (for initializing new model, not used if resume training):
```json
    {
        K: 256 (defualt)
        Q1: 1 (defualt)
        Q2: 0 (defualt)
        batch_size: 32 (defualt)
        cpu: False
        dropout: 0.1 (defualt)
        epochs: 10 (defualt)
        hidden: 384
        huber_delta_path: ./dataset/state/huber_delta.json (defualt)
        layers: 8 
        log_interval: 100 (defualt)
        lr: 0.0001 (defualt)
        model_type: hybrid 
        num_workers: 4 (defualt)
        save_interval: 1000 (defualt)
        train_dir: ./dataset/processed/train (defualt)
        val_dir: ./dataset/processed/val (defualt)
        weight_decay: 0.01 (defualt)
    }
```

./models/\<model\_name\>/log/config.json:
```json
    {
        K: model's original config
        Q1: model's original config
        Q2: model's original config
        batch_size: model's original config
        cpu: model's original config
        delta_curr: latest delta
        dropout: model's original config
        epochs: model's original config
        hidden: model's original config
        huber_delta_path: ./dataset/state/huber_delta.json
        layers: model's original config
        log_interval: model's original config
        lr: model's latest
        model_type: model's original config
        num_workers: model's original config
        save_interval: model's original config
        train_dir: model's original config
        val_dir: model's original config
        weight_decay: model's original config
    }
```

./models/\<model\_name\>/log/train_data.csv:
```csv
ckpt_name, epoch, step, avg_loss, acc_mean, acc_median, acc_std, lr, huber_loss_delta
```

./models/\<model\_name\>/log/runtime.log:
```
ALL MESSAGE PRINTED SHOULD ALSO BE DUMPPED IN HERE AFTER IT GET PRINT OUT
[ TIME ][ LEVEL ] <MESSAGE>
```

./models/\<model\_name\>/log/byte_wise_err.csv:
byte_1_mean is the average error of the first 8 point in all tested chunk
This is to check which region of the chunk contribute to the std the most
Q1 is the starting buckle length, Q2 is the ending buckle length 
```
ckpt_name, Q1, Q2, buckle_head, byte_1_mean, ..., byte_<K/8>_mean
```

./models/\<model\_name\>/log/loss-ckpt.png: loss vs ckpt chart, update per ckpt save
./models/\<model\_name\>/log/acc-ckpt.png:  acc_mean & acc_med & acc_std vs ckpt chart, update per ckpt save

# 2. Program Protocol
## 2.1 theta_model.py: 
define the model class, const, reconst, embedding, buid, and model class essential helpers
```python
"""
theta_model_protocol.py

FINAL ARCHITECTURE-ONLY INTERFACE SPECIFICATION
------------------------------------------------
This file describes the required structure of theta_model.py.

This is a header/protocol file:
    - NO implementations
    - Only function/class signatures
    - Comments describe precise required behavior

theta_model MUST contain ONLY model architecture and utilities.
theta_train owns all training logic, optimization, scheduling, metrics,
loss computation, and dataloader interactions.
"""


# ================================================================
# === SinusoidalNoiseEmbedding
# ================================================================
class SinusoidalNoiseEmbedding:
    """
    Purpose:
        Convert scalar noise level t into a higher-dimensional sinusoidal
        embedding, similar to positional encodings.

    Inputs:
        t : Tensor of shape (batch,) or (batch, 1)

    Behavior:
        - Produce embedding of shape (batch, D)
        - Deterministic, stateless
        - No training-time dependencies
    """
    pass


# ================================================================
# === Model Architecture Classes
# ================================================================
class ThetaMLP:
    """
    Purpose:
        Pure MLP architecture for predicting velocity v_hat from (X_t, t).

    Inputs:
        - X_t : (batch, K, 2)
        - t   : (batch, 1) or embedded version
    Output:
        - v_hat : (batch, K, 2)

    Notes:
        - Forward pass ONLY.
        - No loss, no optimizer, no training logic.
    """
    def forward(self, X_t, t):
        pass


class ThetaCNN1D:
    """
    Purpose:
        1D CNN-based architecture operating along the chunk dimension.

    Inputs/Outputs:
        Same as ThetaMLP.

    Behavior:
        - Use convolution layers over the K dimension.
        - Incorporate noise embedding if needed.
    """
    def forward(self, X_t, t):
        pass


class ThetaTransformer:
    """
    Purpose:
        Transformer encoder-style architecture over the K sequence.

    Inputs/Outputs:
        Same as ThetaMLP.

    Notes:
        - Multi-head self-attention over sequence length K.
        - t embedding added to sequence representation.
    """
    def forward(self, X_t, t):
        pass


class ThetaHybrid:
    """
    Purpose:
        Combined CNN + Transformer + MLP model.
        Typically:
            CNN → Transformer → MLP head

    Inputs/Outputs:
        Same as ThetaMLP.

    Notes:
        - Hybrid architecture from previous project.
        - Pure forward-only model.
    """
    def forward(self, X_t, t):
        pass


# ================================================================
# === build_theta
# ================================================================
def build_theta(model_type: str,
                coord_dim: int,
                hidden: int,
                layers: int,
                K: int,
                dropout: float):
    """
    Purpose:
        Construct the correct model architecture based on config.

    Inputs:
        model_type : one of {"nn", "cnn1d", "transformer", "hybrid", ...}
        coord_dim  : always 2
        hidden     : hidden dimension
        layers     : number of layers
        K          : chunk length (256)
        dropout    : dropout rate

    Behavior:
        - Instantiate correct model class.
        - Initialize weights if needed.
        - Count trainable parameters.

    Outputs:
        return {
            "model":       model_instance,
            "num_params":  int
        }
    """
    pass


# ================================================================
# === count_parameters
# ================================================================
def count_parameters(model):
    """
    Purpose:
        Count trainable parameters inside the model.

    Input:
        model : nn.Module

    Output:
        int : number of parameters requiring gradients

    Notes:
        - Stateless utility.
        - Used only for logging/model summary.
    """
    pass


# ================================================================
# === model_const
# ================================================================
def model_const(model_dir: str):
    """
    Purpose:
        Load a *fully constructed* model from a model directory.

    Inputs:
        model_dir : path to model folder containing:
            - config.json  
                {
                    "model_type": ...,
                    "coord_dim": ...,
                    "hidden": ...,
                    "layers": ...,
                    "K": ...,
                    "dropout": ...,
                    ... other fields ignored by this function ...
                }

            - weights.safetensors OR model.safetensors  
              (exact file name to be defined when implementing)

    Behavior:
        1. Read config.json from model_dir/log/config.json
        2. Extract ONLY the architecture fields needed for build_theta:
               model_type, coord_dim, hidden, layers, K, dropout
        3. Call:
               build_theta(...)
        4. Load state_dict from .safetensors file into the constructed model
        5. Return the ready-to-use model instance (nn.Module)

    Output:
        model : nn.Module

    Notes:
        - No optimizer or scheduler is loaded here
        - Device transfer is left to the caller
        - This function does NOT modify training runtime
        - This function does NOT perform training
    """
    pass


# ================================================================
# END OF ARCHITECTURE-ONLY PROTOCOL
# ================================================================
```

## 2.2 theta_train.py
Define the training logic (new training or resume training), training monitor (halt when converged), ckpt saving, ckpt puring, quick testing, info & stats logging

```python
"""
theta_train_protocol.py

FINAL HEADER / INTERFACE SPECIFICATION
--------------------------------------

This file defines the authoritative interfaces, responsibilities, and required
behavior for the new theta training system. This is the C++-style header file:
    - No implementation
    - Only function/class signatures
    - Comments describe exact required behavior

This protocol overrides ALL previous drafts.
"""


# ================================================================
# === cleanup_memory
# ================================================================
def cleanup_memory():
    """
    Purpose:
        - Clear CPU + GPU memory at program start and between heavy phases.
    Behavior:
        - If CUDA available: torch.cuda.empty_cache() + torch.cuda.ipc_collect()
        - Always call gc.collect()
    Constraints:
        - Must NOT modify runtime or training state.
    """
    pass


# ================================================================
# === config_solver
# ================================================================
def config_solver(runtime: dict):
    """
    Purpose:
        Resolve new vs resume mode and fully populate the runtime dict.

    Inputs:
        runtime : dict (initially empty)

    Outputs (runtime keys REQUIRED after return):
        runtime["config"]         : dict, the complete training config
        runtime["model_name"]     : str, auto-generated or derived from resume
        runtime["model_root_dir"] : Path("./models/<model_name>/")
        runtime["log_dir"]        : Path("./models/<model_name>/log/")
        runtime["ckpt_dir"]       : Path("./models/<model_name>/ckpt/")
        runtime["start_epoch"]    : int  (0 for new, ckpt_epoch+1 for resume)
        runtime["resume_ckpt"]    : None or ckpt path string
        runtime["huber_delta"]    : latest huber delta
        runtime["K"]              : chunk size
        runtime["Q1"]             : buckle start length
        runtime["Q2"]             : buckle end length
        runtime["stop_flag"]      : bool, initialize to False

    NEW TRAINING (choice == "1"):
        - Load ./src/config.json as base.
        - Generate:
              model_name = f"{model_type}_{timestamp}"
        - model_root_dir = ./models/<model_name>/
        - start_epoch = 0
        - resume_ckpt = None
        - Load huber_delta from config["huber_delta_path"] else fallback.
        - Copy config.json into model_root_dir/log/

    RESUME TRAINING (choice == "2"):
        - User enters ckpt_path.
        - Extract model_name from:
              ./models/<model_name>/ckpt/ckpt_epoch<E>_step<S>.pt
        - Load existing config:
              ./models/<model_name>/log/config.json
        - Parse E from ckpt filename → start_epoch = E + 1
        - resume_ckpt = ckpt_path
        - Load huber_delta from config["delta_curr"] (latest)
        - If start_epoch >= number of .pt files, exit.

    Notes:
        - This function handles ALL branching logic.
        - Other modules ONLY read runtime.
    """
    pass


# ================================================================
# === model_house_builder
# ================================================================
def model_house_builder(runtime: dict):
    """
    Purpose:
        Construct or verify the directory layout under ./models/<model_name>/.

    Required structure:
        ./models/<model_name>/
        ./models/<model_name>/log/
        ./models/<model_name>/ckpt/
        ./models/<model_name>/fig/   (optional: for loss/acc plots)

    NEW TRAINING:
        - Create directories.
        - Write config.json (initial training config).

    RESUME TRAINING:
        - Directories must already exist.
        - Do NOT overwrite config.json here; that is done at checkpoint time.

    Constraints:
        - Must NOT load model or create optimizer.
        - Must NOT start training.
    """
    pass


# ================================================================
# === DataLoader (file-per-epoch)
# ================================================================
class DataLoader:
    """
    Purpose:
        Control loading of epoch data (X,V,t) from sorted .pt files.
        Provide per-batch access.

    Notes:
        - Only this class reads .pt files.
        - Uses fixed epoch size = min(37000, file length).
        - No global shuffle — only shuffle per epoch.
    """

    def __init__(self, runtime: dict):
        """
        Inputs:
            runtime["config"]["train_dir"]  : path to *.pt files

        Behavior:
            - Scan train_dir for "*.pt".
            - Sort deterministically.
            - Store:
                  self.file_list
                  self.epoch_count
                  self.X, self.V, self.t = None
            - DO NOT load tensor data yet.
        """
        pass

    def set(self, epoch_idx: int):
        """
        Purpose:
            Load the i-th training file into memory.

        Behavior:
            - Free previously loaded tensors.
            - torch.load(file_list[epoch_idx], map_location="cpu")
            - Expect keys: "X_t", "V", "t"
            - LIMIT = min(37000, len(X_t))
            - Store:
                  self.X = X_t[:LIMIT]
                  self.V = V[:LIMIT]
                  self.t = t[:LIMIT]
            - No batching logic here.
        """
        pass

    def get_batch(self, batch_idx: int):
        """
        Purpose:
            Retrieve batch #batch_idx from the current epoch.

        Behavior:
            - Create or reuse a torch.utils.data.DataLoader with EXACT settings:
                  shuffle=True
                  drop_last=True
                  pin_memory=True
                  persistent_workers=False
                  collate_fn=None
                  batch_size = runtime["config"]["batch_size"]

            - t batch is handled naturally by tensor slicing.

        Returns:
            (X_batch, V_batch, t_batch)
        """
        pass

    def next_epoch(self, runtime: dict):
        """
        Placeholder for future streaming interface.
        Currently UNUSED.
        """
        pass

    def chunk_const(self, runtime: dict):
        """
        Placeholder for K/Q1/Q2 on-the-fly chunk construction.
        Currently UNUSED.
        """
        pass


# ================================================================
# === training_initializer
# ================================================================
def training_initializer(runtime: dict):
    """
    Purpose:
        Build model, optimizer, scheduler, and move model to device.

    Inputs (runtime["config"]):
        model_type, hidden, layers, K, dropout
        lr, weight_decay
        cpu flag

    Behavior:
        - Create device = "cuda" unless cpu=True.
        - Build model using theta_model.build_theta.
        - Move model to device.
        - Create optimizer.
        - Create scheduler using OLD PROJECT LOGIC:
              warmup_steps = 1000 for new training
              warmup_steps = 0 for resume if previous step >= 1000
        - If runtime["resume_ckpt"] is not None:
              load model_state_dict
              load optimizer_state_dict
              load scheduler_state_dict

    Outputs (runtime writes):
        runtime["theta"]
        runtime["optimizer"]
        runtime["scheduler"]
        runtime["device"]
    """
    pass


# ================================================================
# === converge_detector
# ================================================================
def converge_detector(runtime: dict) -> bool:
    """
    Purpose:
        Decide whether training should stop early based on train_data.csv.

    Behavior:
        - Read:
              ./models/<model_name>/log/train_data.csv
        - Analyze:
              loss, acc_mean, acc_median, acc_std
        - CURRENT BEHAVIOR:
              return False   (placeholder)

    Future:
        - Determine best ckpt.
        - Possibly remove extra ckpts.
        - Implement plateau detection.
    """
    pass


# ================================================================
# === save_checkpoint_and_log
# ================================================================
def save_checkpoint_and_log(runtime: dict,
                            avg_loss: float,
                            acc_mean: float,
                            acc_median: float,
                            acc_std: float,
                            step: int):
    """
    Purpose:
        Save model state at checkpoint interval and update all logs.

    Checkpoint Naming Format (FINAL):
        ckpt_epoch_<E>_step_<S>.pt

        Where:
            <E> = epoch index (runtime["epoch_idx"])
            <S> = step index inside epoch (resets each epoch)

        Example:
            ckpt_epoch0_step0.pt
            ckpt_epoch0_step200.pt
            ckpt_epoch1_step0.pt

        No ckpt_final. No global step.

    Behavior:
        1. Save checkpoint file into runtime["ckpt_dir"]:
              model_state_dict
              optimizer_state_dict
              scheduler_state_dict
              epoch_idx
              step
              huber_delta
              lr

        2. Append CSV row to train_data.csv:
              ckpt_name, epoch, step, avg_loss,
              acc_mean, acc_median, acc_std,
              lr, huber_loss_delta

        3. Overwrite config.json with updated values:
              delta_curr
              lr
              epoch
              (other runtime-dependent fields)

        4. Regenerate:
              loss-ckpt.png
              acc-ckpt.png

        5. DO NOT decide convergence.
           That is converge_detector()'s responsibility.
    """
    pass


# ================================================================
# === training_manager
# ================================================================
def training_manager(runtime: dict):
    """
    Purpose:
        Orchestrate epoch-level training.

    Behavior:
        for epoch_idx in range(runtime["start_epoch"], DataLoader.epoch_count):

            - runtime["epoch_idx"] = epoch_idx

            - runtime["DataLoader"].set(epoch_idx)
                -> load that epoch's X,V,t

            - Call train_loop(...) (from theta_model):
                * loops over batches
                * computes L2/Huber loss
                * updates model
                * every save_interval steps calls save_checkpoint_and_log()

            - After save operations:
                if converge_detector(runtime) == True:
                    runtime["stop_flag"] = True

            - If runtime["stop_flag"]:
                    break

    Notes:
        - No validation here.
        - No full test here.
        - Quick validation is a separate script using quick_val.pt.
    """
    pass


# ================================================================
# === main
# ================================================================
def main():
    """
    Purpose:
        High-level entry point.

    Behavior:
        - cleanup_memory()
        - runtime = {}
        - config_solver(runtime)
        - model_house_builder(runtime)
        - runtime["DataLoader"] = DataLoader(runtime)
        - training_initializer(runtime)
        - training_manager(runtime)

    Terminal Output Policy:
        - Minimal printing:
            * initializing sections
            * epoch summary (single-line dynamic)
        - runtime.log (file) contains:
            * initial config
            * resume config
            * checkpoint summaries
            * error messages
    """
    pass

```