# RectifiedTraj

RectifiedTraj is a trajectory denoising and benchmark workspace. The current
tree contains:

- learned RectifiedTraj and ResidualReg online model checkpoints under
  `bin/model/`,
- benchmark runners for trajectory, chunk, and uncertainty-band evaluation,
- parquet-to-processed-dataset utilities under `src/utils/data_processor/`,
- plotting and map utilities under `src/utils/data_visualizer/`.

Large checkpoint files are tracked with Git LFS.

## Setup

Use Python 3.11.

```bash
python3.11 -m venv env_RectifiedTraj
source env_RectifiedTraj/bin/activate
pip install -r requirements.txt
```

After cloning, make sure Git LFS objects are present:

```bash
git lfs pull
```

Most raw and processed datasets are local working artifacts. The repository
tracks placeholders such as `dataset/raw/BlogWatcher/.gitkeep`, calibration
state in `dataset/state/calib.json`, and model checkpoints under `bin/model/`.

## Tracked Models

Current tracked model roots:

- `bin/model/RectifiedTraj_online`
  - `cnn_online_1M_20260518_181708`
  - `hybrid_online_1M_20260518_181215`
  - `transformer_online_1M_20260518_181440`
- `bin/model/ResidualReg_online`
  - `hybrid_online_1M_20260523_180637`

Use `data_hypothesis: "RectifiedTraj"` for `RectifiedTraj_online` models and
`data_hypothesis: "ResidualReg"` for `ResidualReg_online` models.

## Evaluation

The unified runner is:

```bash
python src/run_benchmarks.py
```

It reads `src/eval_joblist.json`. The checked-in file is intentionally blank,
so write a valid JSON object there before running. Each run copies the exact
joblist into the result directory.

Useful flags:

- `-test`: use the debug mini trajectory dataset where supported.
- `--wandb`: upload the completed result directory to Weights & Biases.
- `--wandb_project`, `--wandb_entity`, `--wandb_run_name`: W&B metadata.

### Current Joblist Schema

Top-level fields used by the current runner:

- `test_type`: one of `exact`, `uncertainty`, or `tuning`.
- `test_items`:
  - `traj_test`: run trajectory evaluation.
  - `chunk_test`: run chunk evaluation.
  - `uncertainty_test`: run uncertainty-band evaluation.
- `test_files`:
  - `traj_files`: list of processed trajectory `.pt` files or directories.
  - `chunk_files`: list of processed chunk `.pt` files or directories.
  - `uncertainty_path`: processed uncertainty/trajectory test file or directory.
- `model_groups`: list of learned model groups. Each group supports:
  - `model_root`: folder containing model run directories.
  - `models`: model run names. Omit or set to `null` to discover all models in
    the root.
  - `data_hypothesis`: `RectifiedTraj` or `ResidualReg`.
  - `Q1`, `Q2`: byte settings for the evaluation grid.
  - `denoise_steps`: optional positive integer or list of integers. Omit for
    the default single-step behavior.
- `baseline.models`: classic baselines to run. Supported values are
  `alpha_beta`, `causal_hampel`, `kalman_filter`, `kalman_rts`, `hampel`,
  `savgol`, and `raw`.
- `runtime.device`: `cuda` or `cpu`.
- `runtime.strict_init`: fail fast on incompatible checkpoint/config loading.
- `runtime.traj_parallel`: trajectory/chunk worker count.
- `run_baseline`: run classic baselines when the selected phase supports them.
- `data_source.raw_dataset_dir`: raw parquet directory used when generating
  processed evaluation inputs.
- `data_source.raw_test_files`: optional raw parquet file list.

Deprecated joblist fields from older README versions, such as `test_item`,
`benchmark`, `grid`, `trajectory`, `bounded`, and `time`, are not the current
configuration surface.

The runner rejects `t_delta`, `delta_t`, `step`, `method`, `methods`,
`denoise_method`, `denoise_methods`, and `rolling_window` in `eval_joblist.json`.
Trajectory evaluation uses the fixed chunk-stitch denoiser.

### Exact Evaluation Example

```json
{
  "test_type": "exact",
  "test_items": {
    "traj_test": true,
    "chunk_test": false,
    "uncertainty_test": false
  },
  "test_files": {
    "traj_files": [
      "./dataset/processed/NUMOSIM_Kanto/test/traj_test"
    ],
    "chunk_files": []
  },
  "data_source": {
    "raw_dataset_dir": null,
    "raw_test_files": null
  },
  "model_groups": [
    {
      "model_root": "./bin/model/RectifiedTraj_online",
      "models": [
        "cnn_online_1M_20260518_181708",
        "hybrid_online_1M_20260518_181215",
        "transformer_online_1M_20260518_181440"
      ],
      "Q1": [-1],
      "Q2": [0],
      "data_hypothesis": "RectifiedTraj"
    }
  ],
  "baseline": {
    "models": []
  },
  "runtime": {
    "device": "cuda",
    "strict_init": true,
    "traj_parallel": 4
  },
  "run_baseline": false,
  "baseline_progress": false,
  "brief_summary": true,
  "brief_visualizer": false,
  "log_level": "INFO"
}
```

`Q1: [-1]` is the strict-online endpoint sentinel used by recent online
trajectory runs. Use the values required by the experiment you are reproducing.

### BlogWatcher Uncertainty Example

```json
{
  "test_type": "uncertainty",
  "test_items": {
    "traj_test": false,
    "chunk_test": false,
    "uncertainty_test": true
  },
  "test_files": {
    "traj_files": [],
    "chunk_files": [],
    "uncertainty_path": "./dataset/processed/BlogWatcher/test/traj_test"
  },
  "data_source": {
    "raw_dataset_dir": "./dataset/raw/BlogWatcher",
    "raw_test_files": null
  },
  "model_groups": [
    {
      "model_root": "./bin/model/RectifiedTraj_online",
      "models": [
        "cnn_online_1M_20260518_181708",
        "hybrid_online_1M_20260518_181215",
        "transformer_online_1M_20260518_181440"
      ],
      "Q1": [0],
      "Q2": [0],
      "data_hypothesis": "RectifiedTraj"
    },
    {
      "model_root": "./bin/model/ResidualReg_online",
      "models": [
        "hybrid_online_1M_20260523_180637"
      ],
      "Q1": [0],
      "Q2": [0],
      "data_hypothesis": "ResidualReg"
    }
  ],
  "baseline": {
    "models": [
      "alpha_beta",
      "causal_hampel",
      "kalman_filter",
      "kalman_rts",
      "hampel",
      "savgol",
      "raw"
    ]
  },
  "runtime": {
    "device": "cuda",
    "strict_init": true,
    "traj_parallel": 4
  },
  "run_baseline": true,
  "baseline_progress": true,
  "brief_summary": true,
  "brief_visualizer": false,
  "log_level": "INFO"
}
```

### Outputs

Benchmark outputs are written under:

```text
bin/test_results/<run_name>/
```

Typical files include:

- `eval_joblist.json`: snapshot of the input joblist.
- `system_info.json`: runtime device and system snapshot.
- `fact_used_dataset.json`: dataset facts for inputs used by the run.
- `trajectory_evaluation_summary.csv`
- `trajectory_pointwise_summary.csv`
- `chunk_evaluation_summary.csv`
- `chunk_pointwise_summary.csv`
- `chunk_bytewise_summary.csv`
- `uncertainty_band_summary.csv`
- generated heatmaps when the relevant summaries are present.

## Training

Training uses:

```bash
python src/theta_train.py
```

`src/theta_train.py` reads `src/config.json` for new runs. The checked-in file
is blank, so populate it with the intended training config before launching.
Model artifacts are written under the configured `model_root`, with each run
containing:

- `log/config.json`
- `log/config_init.json`
- `best_ckpt/*.safetensors`
- `best_ckpt/*_full.pt`
- `fig/*.png`

## BlogWatcher All-In-One

The BlogWatcher helper runs parquet processing and then `src/run_benchmarks.py`.

1. Put a BlogWatcher parquet file under `dataset/raw/BlogWatcher/`.
2. Populate `src/eval_joblist.json` with a valid BlogWatcher joblist.
3. Optionally smoke-test prerequisites:

```bash
python src/utils/evaluations/BlogWatcher_All_in_one.py \
  --smoke-test \
  --check-git-tracked \
  --file <name>.parquet
```

4. Run the full flow:

```bash
python src/utils/evaluations/BlogWatcher_All_in_one.py --wandb
```

Notes:

- The parquet filename can be arbitrary. The dataset identity comes from the
  folder name `dataset/raw/BlogWatcher`.
- If multiple parquet files are present, the script uses the newest one.
  Use `--file <name>.parquet` to force a specific file.
- `--wandb` uploads only the generated result directory under
  `bin/test_results/`, not the raw dataset.
- Omit `--wandb` for local-only benchmark output.

## Data And Plot Utilities

### Region Trajectory Map

Use `src/utils/data_visualizer/region_traj_mapper.py` to:

- select trajectories that pass through a lon/lat bounding box,
- save matched full trajectories as a new trajectory `.pt`,
- save a window-clipped trajectory `.pt` for local plotting,
- optionally add a `denoised` field with model output,
- render a map-style PNG for noisy, clean, and denoised tracks.

Example:

```bash
env_RectifiedTraj/bin/python src/utils/data_visualizer/region_traj_mapper.py \
  --input ./dataset/processed/NUMOSIM_Kanto/test/traj_test/traj_native_200_5000.pt \
  --min-lon 139.590 --max-lon 139.600 \
  --min-lat 35.333 --max-lat 35.343 \
  --pad-lon 0.01 --pad-lat 0.01 \
  --output-dir ./bin/region_maps/demo
```

Add denoising output:

```bash
env_RectifiedTraj/bin/python src/utils/data_visualizer/region_traj_mapper.py \
  --input ./bin/region_maps/demo/region_window_lon_139p59000_139p60000_lat_35p33300_35p34300.pt \
  --min-lon 139.590 --max-lon 139.600 \
  --min-lat 35.333 --max-lat 35.343 \
  --checkpoint ./bin/model/RectifiedTraj_online/hybrid_online_1M_20260518_181215/best_ckpt/ckpt_e408_s637296_full.pt \
  --denoise-method DF \
  --device cpu \
  --output-dir ./bin/region_maps/demo_window_denoised
```

Outputs:

- `region_full_*.pt`
- `region_window_*.pt`
- `region_map_*.png`
- `region_summary_*.json`

### Raw Density Survey

Use `src/utils/data_visualizer/raw_density_survey.py` to find point-dense
lon/lat regions directly from raw parquet before choosing a plotting bbox.

```bash
env_RectifiedTraj/bin/python src/utils/data_visualizer/raw_density_survey.py \
  --input ./dataset/raw/NUMOSIM_Kanto/part-00469-0c4ec7f6-0818-42c2-8fa5-cd7fa3706b9d.c000.zstd.parquet \
  --coord-field clean \
  --top-k 10 \
  --row-stride 50 \
  --output-dir ./bin/density_survey/part_00469
```

Outputs:

- `density_summary.json`
- `density_top_cell_*.csv`

The default `row-stride` is `10`. Use `--row-stride 1` only when full
resolution is worth the cost.

### Whole Raw Parquet Plot

Use `src/utils/data_visualizer/raw_parquet_plot.py` to render an entire raw
parquet file or parquet directory as a lon/lat density heatmap.

```bash
env_RectifiedTraj/bin/python src/utils/data_visualizer/raw_parquet_plot.py \
  --input ./dataset/raw/NUMOSIM_Kanto/part-00469-0c4ec7f6-0818-42c2-8fa5-cd7fa3706b9d.c000.zstd.parquet \
  --coord-field clean \
  --cell-size 0.005 \
  --row-stride 50 \
  --output-dir ./bin/raw_parquet_plots/part_00469
```

Crop around a center point:

```bash
env_RectifiedTraj/bin/python src/utils/data_visualizer/raw_parquet_plot.py \
  --input ./dataset/raw/NUMOSIM_Kanto \
  --coord-field clean \
  --cell-size 0.002 \
  --row-stride 50 \
  --center-lon 139.5426585 \
  --center-lat 35.51860427856445 \
  --radius-miles 10 \
  --output-dir ./bin/raw_parquet_plots/numosim_center_10mi
```

Outputs:

- `*.png`: density heatmap.
- `*.json`: extent, grid shape, and top bins.

Start with `--row-stride 50` on large directories.
