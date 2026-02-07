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
- `model_root`: root directory containing model folders (default: `./bin/model`).
- `model_names`: list of model folder names to test. Set to `null` or `[]` to test **all** models.
- `methods`: list of denoise methods, e.g. `["BF", "DF"]`.

#### Dataset fields (used by `grid` / `trajectory` / `bounded`)

- `test_data_path`: dataset path.
  - For `bounded`, this should point to `./dataset/processed/full_traj_range`.
- `M`: number of trajectories.
- `D`: days per trajectory (optional).
- `N`: points per trajectory (optional). If `N` is `null`, then `N = D * 8640`.

#### Grid-search fields (required for `grid` / `benchmark`)

- `Q1`: list of Q1 values (bytes).
- `Q2`: list of Q2 values (bytes).
- `t_delta`: list of step sizes.

All three must be non-empty lists for `grid` / `benchmark`.

#### Time-test fields

- `npy_path`: input `.npy` file for timing (default: `./dataset/time_test/source_list.npy`).
- `time_log_path`: CSV output for timing results (default: `./bin/log/time_test.csv`).
- `batch_size`: batch size for timing (default: `64`).
- `device`: `cuda` or `cpu` (default: `cuda`).

### Example joblist

```json
{
  "test_item": "benchmark",
  "model_root": "./bin/model",
  "model_names": null,
  "methods": ["BF", "DF"],

  "test_data_path": "./dataset/processed/full_traj_10min",
  "M": 100,
  "D": null,
  "N": 432,

  "Q1": [1],
  "Q2": [12],
  "t_delta": [1],

  "npy_path": "./dataset/time_test/source_list.npy",
  "time_log_path": "./bin/log/time_test.csv",
  "batch_size": 64,
  "device": "cuda"
}
```

### Output locations

- Trajectory/grid results: under `./bin/test_results/`.
- Timing results: `./bin/log/time_test.csv` (or `time_log_path`).

### Uncertainty Test (UTokyo)

Quick run (UTokyo researchers):

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
3. Place UTokyo parquet data under `./dataset/UTokyo/`, then run:
   ```bash
   python src/utils/evaluations/UTokyo_test.py --wandb
   ```
   Optional: add `-csv` to save detailed per-point aggregates as CSV instead of parquet
   (W&B uploads CSV artifacts in that case).

For UTokyo datasets, the uncertainty-band run writes a CSV named
`uncertainty_band_summary.csv` under a timestamped folder (e.g.,
`./bin/test_result_UTokyo/test_YYYYMMDD_HHMMSS/`). Columns:

- `model_name`: model or baseline name.
- `denoise_method`: `BF`, `DF`, or `Baseline`.
- `K`, `Q1`, `Q2`: model config (may be `NA` for baselines).
- `t_delta`, `N_steps`: fixed to `1.0` and `1` for UTokyo runs.
- `pass_rate_points`: overall point pass rate (distance <= accuracy).
- `pass_rate_trajectories`: average of per-trajectory pass rates.
- `avg_outside_error`: mean of (distance - accuracy) for failed points only.
- `data_avg_sample_time_sec`, `data_median_sample_time_sec`, `data_std_sample_time_sec`:
  dataset sample time stats from timestamps (seconds).
- `mean_distance_all`, `mean_signed_margin_all`:
  mean distance and mean (distance - accuracy) over all points.
- `tier4_*_all`: all points (no accuracy filter).
- `tier3_*_acc_leq_30`: points with `accuracy <= 30`.
- `tier2_*_acc_leq_15`: points with `accuracy <= 15`.
- `tier1_*_acc_leq_10`: points with `accuracy <= 10`.
- `tier0_*_acc_leq_5`: points with `accuracy <= 5`.
  Each tier reports:
  `points`, `pass_rate_points`, `pass_rate_trajectories`,
  `mean_distance`, `mean_signed_margin`.
- `num_tested_trajectories`, `num_tested_points`, `longest_trajectory_length`.
- `test_timestamp`: ISO timestamp for the run row.
