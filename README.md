# RectifiedTraj

## Evaluation

### Runner
Use the unified runner:

```bash
python src/run_benchmarks.py
```

This reads `src/eval_joblist.json` and runs tests according to `test_item`.

### Joblist (`src/eval_joblist.json`)

All evaluation configuration is driven by this file.

#### Common fields

- `test_item`: one of `benchmark`, `grid`, `trajectory`, `bounded`, `time`.
  - `benchmark` runs **all** tests in a fixed order: grid on 3 datasets, trajectory eval, bounded eval, then time tests.
  - `grid` runs only grid search on a single dataset.
  - `trajectory` runs trajectory evaluation on a single dataset.
  - `bounded` runs uncertainty-band evaluation on a single dataset.
  - `time` runs only the time test (single config).
- `model_root`: root directory containing model folders (default: `./bin/model/RectifiedTraj`).
- `data_hypothesis`: model family routing token (`RectifiedTraj` or `ResidualReg`).
- `model_names`: list of model folder names to test. Set to `null` or `[]` to test **all** models.

#### Dataset fields (used by `grid` / `trajectory` / `bounded`)

- `test_data_path`: dataset path.
  - For `bounded`, this should point to `./dataset/processed/<dataset>/test/traj_test/full_traj_range`.
- `M`: number of trajectories.
- `D`: days per trajectory (optional).
- `N`: points per trajectory (optional). If `N` is `null`, then `N = D * 8640`.

#### Grid-search fields (required for `grid` / `benchmark`)

- `Q1`: list of Q1 values (bytes).
- `Q2`: list of Q2 values (bytes).
- `denoise_steps`: optional positive integer or list of positive integers for
  multi-step RF integration. Leave it unset or `null` for the default single
  denoising step.

`Q1` and `Q2` must be non-empty lists for `grid` / `benchmark`. When
`denoise_steps` is not provided, the evaluation pipeline leaves `t_delta`
unset and the model defaults to `t_delta = 1.0`. If `denoise_steps = n` is
provided, the runner uses `t_delta = 1.0 / n`. Do not put `t_delta`, `delta_t`,
`step`, `method`, or `methods` in `src/eval_joblist.json`; those keys are
rejected by the schema.

#### Time-test fields

- `npy_path`: input `.npy` file for timing (default: `./dataset/time_test/source_list.npy`).
- `time_log_path`: CSV output for timing results (default: `./bin/log/time_test.csv`).
- `batch_size`: batch size for timing (default: `64`).
- `device`: `cuda` or `cpu` (default: `cuda`).

### Example joblist

```json
{
  "test_item": "benchmark",
  "model_root": "./bin/model/RectifiedTraj",
  "data_hypothesis": "RectifiedTraj",
  "model_names": null,

  "test_data_path": "./dataset/processed/NUMOSIM_Kanto/test/traj_test",
  "M": 100,
  "D": null,
  "N": 432,

  "Q1": [1],
  "Q2": [12],
  "denoise_steps": null,

  "npy_path": "./dataset/time_test/source_list.npy",
  "time_log_path": "./bin/log/time_test.csv",
  "batch_size": 64,
  "device": "cuda"
}
```

### Output locations

- Trajectory/grid results: under `./bin/test_results/`.
- Timing results: `./bin/log/time_test.csv` (or `time_log_path`).

### Region trajectory map utility

Use `src/utils/data_visualizer/region_traj_mapper.py` to:

- select any trajectory that passes through a lon/lat bounding box,
- save the matched full trajectories as a new trajectory `.pt`,
- save a window-clipped trajectory `.pt` for local plotting,
- optionally add a `denoised` field with model output,
- render a map-style PNG for noisy / clean / denoised tracks.

Example from an existing NUMOSIM trajectory suite:

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
  --checkpoint ./bin/model/RectifiedTraj/hybrid_5M_20260201_231921/best_ckpt/ckpt_e24_s874000_full.pt \
  --denoise-method DF \
  --device cpu \
  --output-dir ./bin/region_maps/demo_window_denoised
```

Outputs:

- `region_full_*.pt`: full pass-through trajectories with the native trajectory schema.
- `region_window_*.pt`: view-window-clipped trajectories for plotting.
- `region_map_*.png`: rendered figure.
- `region_summary_*.json`: bbox, counts, and output paths.

### Raw density survey utility

Use `src/utils/data_visualizer/raw_density_survey.py` to find point-dense lon/lat regions directly from raw parquet before choosing a plotting bbox.

Example from the project parent directory:

```bash
RectifiedTraj/env_RectifiedTraj/bin/python RectifiedTraj/src/utils/data_visualizer/raw_density_survey.py \
  --input RectifiedTraj/dataset/raw/NUMOSIM_Kanto/part-00469-0c4ec7f6-0818-42c2-8fa5-cd7fa3706b9d.c000.zstd.parquet \
  --coord-field clean \
  --top-k 10 \
  --row-stride 50 \
  --output-dir RectifiedTraj/bin/density_survey/part_00469
```

Outputs:

- `density_summary.json`: extent plus top bins for each cell size.
- `density_top_cell_*.csv`: ranked hotspot bins per grid size.
- The default `row-stride` is `10` for lower-impact previews. Use `--row-stride 1` only when full resolution is worth the cost.

### Whole raw parquet plot utility

Use `src/utils/data_visualizer/raw_parquet_plot.py` to render an entire raw parquet file or parquet directory as a lon/lat density heatmap.

Example from the project parent directory:

```bash
RectifiedTraj/env_RectifiedTraj/bin/python RectifiedTraj/src/utils/data_visualizer/raw_parquet_plot.py \
  --input RectifiedTraj/dataset/raw/NUMOSIM_Kanto/part-00469-0c4ec7f6-0818-42c2-8fa5-cd7fa3706b9d.c000.zstd.parquet \
  --coord-field clean \
  --cell-size 0.005 \
  --row-stride 50 \
  --output-dir RectifiedTraj/bin/raw_parquet_plots/part_00469
```

You can also crop around a center point with a radius:

```bash
RectifiedTraj/env_RectifiedTraj/bin/python RectifiedTraj/src/utils/data_visualizer/raw_parquet_plot.py \
  --input RectifiedTraj/dataset/raw/NUMOSIM_Kanto \
  --coord-field clean \
  --cell-size 0.002 \
  --row-stride 50 \
  --center-lon 139.5426585 \
  --center-lat 35.51860427856445 \
  --radius-miles 10 \
  --output-dir RectifiedTraj/bin/raw_parquet_plots/numosim_center_10mi
```

Outputs:

- `*.png`: full-file density heatmap.
- `*.json`: extent, grid shape, and top bins.
- The default `row-stride` is `10` for lower-impact previews. Start with `--row-stride 50` on large directories.

### BlogWatcher All-in-one

Quick run:

1. Create and activate a Python 3.11 environment, then install dependencies:
   ```bash
   python3.11 -m venv env_RectifiedTraj
   source env_RectifiedTraj/bin/activate
   pip install -r requirements.txt
   ```
2. Log in to Weights & Biases (required when using `--wandb`):
   ```bash
   wandb login
   ```
3. From the repo root, place a BlogWatcher parquet file under
   `./dataset/raw/BlogWatcher/`, then run:
   ```bash
   python src/utils/evaluations/BlogWatcher_All_in_one.py --wandb
   ```
   The script auto-detects the parquet filename inside that folder, runs
   `parquet_processor --mode test-only`, then runs `src/run_benchmarks.py`.
   When `--wandb` is enabled, it uploads only the generated benchmark result
   directory under `./bin/test_results/`, not the raw dataset.

Behavior notes:

- The parquet filename can be arbitrary; the dataset identity comes from the
  folder name `dataset/raw/BlogWatcher`.
- If multiple parquet files are present, the script uses the newest one.
  Use `--file <name>.parquet` to force a specific file.
- Run `python src/utils/evaluations/BlogWatcher_All_in_one.py --smoke-test --check-git-tracked --file <name>.parquet`
  to validate runner prerequisites and confirm required source files are
  git-tracked before launching the benchmark. Raw datasets and model outputs
  under `bin/model/` are checked for existence but not required to be tracked.
- Omit `--wandb` if you only want the local benchmark outputs.
- Results are written under `./bin/test_results/<timestamped_run>/`.
