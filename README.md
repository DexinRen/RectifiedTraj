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
- `methods`: list of denoise methods, e.g. `["BF", "DF"]`.

#### Dataset fields (used by `grid` / `trajectory` / `bounded`)

- `test_data_path`: dataset path.
  - For `bounded`, this should point to `./dataset/processed/<dataset>/test/traj_test/full_traj_range`.
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
  "model_root": "./bin/model/RectifiedTraj",
  "data_hypothesis": "RectifiedTraj",
  "model_names": null,
  "methods": ["BF", "DF"],

  "test_data_path": "./dataset/processed/NUMOSIM_Kanto/test/traj_test",
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

### Uncertainty Test (External)

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
3. Place a BlogWatcher parquet file under `./dataset/raw/BlogWatcher/`, then run:
   ```bash
   python src/utils/evaluations/BlogWatcher_All_in_one.py --wandb
   ```
   The script auto-detects the parquet filename inside that folder, runs
   `parquet_processor --mode test-only`, then runs `src/run_benchmarks.py`.
   When `--wandb` is enabled, it uploads only the generated benchmark result
   directory under `./bin/test_results/`, not the raw dataset.
- `t_delta`, `N_steps`: fixed to `1.0` and `1` for these runs.
- `pass_rate_points`: overall point pass rate (distance <= accuracy).
- `pass_rate_trajectories`: average of per-trajectory pass rates.
- `avg_outside_error`: mean of (distance - accuracy) for failed points only.
- `mean_exceed_m`, `p95_exceed_m`: failed-point-only exceedance magnitude summaries.
- `data_avg_sample_time_sec`, `data_median_sample_time_sec`, `data_std_sample_time_sec`:
  dataset sample time stats from timestamps (seconds).
- `mean_distance_all`, `p50_distance_all`, `p95_distance_all`:
  point-to-reference L2 distance summaries over all points.
- `mean_signed_margin_all`:
  mean (distance - accuracy) over all points.
- `mean_normalized_distance_all`, `p50_normalized_distance_all`,
  `p95_normalized_distance_all`, `pass_rate_normalized_distance_leq_1`:
  scalar normalized error summaries using `distance / accuracy`.
- `mean_excess_m`, `p95_excess_m`:
  summaries of `max(distance - accuracy, 0)` over all points.
- `mean_anisotropic_z_all`, `p95_anisotropic_z_all`,
  `pass_rate_anisotropic_z_leq_1`:
  optional anisotropic normalized error summaries using per-axis sigmas when
  trajectory payloads include `longitude_sigma` and `latitude_sigma`;
  otherwise the CSV shows `NA`.
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
