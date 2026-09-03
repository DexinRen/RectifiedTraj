# RectifiedTraj

RectifiedTraj is a trajectory denoising and benchmark workspace. The current
tree contains:

- learned RectifiedTraj, DirectReg, and Diffusion online model checkpoints under
  `bin/model/`,
- benchmark runners for trajectory, chunk, and uncertainty-band evaluation,
- parquet-to-processed-dataset utilities under `src/utils/data_processor/`,
- plotting and map utilities under `src/utils/data_visualizer/`.

Large checkpoint files are tracked with Git LFS.

## Setup

Use Python 3.11.
Docker Engine with the Compose plugin is also required for the Valhalla Meili
baseline and map cutter.

```bash
python3.11 -m venv env_RectifiedTraj
source env_RectifiedTraj/bin/activate
pip install -r requirements.txt
```

After cloning, make sure Git LFS objects are present:

```bash
git lfs pull
```

The processed PoL_5s dataset and its metadata are public and included in this
repository.

## Tracked Models

Current tracked model roots:

- `bin/model/RectifiedTraj_online`
  - `cnn_online_1M_20260518_181708`
  - `hybrid_online_1M_20260518_181215`
  - `transformer_online_1M_20260518_181440`
- `bin/model/DirectReg_online`
  - `hybrid_online_1M_20260523_180637`
  - `causal_mlp_1M_20260825_134854`
- `bin/model/Diffusion_online`
  - `diffusion_hybrid_online_1M_20260809_161937`

Use `data_hypothesis: "RectifiedTraj"` for `RectifiedTraj_online` models and
`data_hypothesis: "DirectReg"` for `DirectReg_online` models. Use
`data_hypothesis: "Diffusion"` for `Diffusion_online` models.

## Evaluation

The unified runner is:

```bash
python src/run_benchmarks.py
```

It reads `src/eval_joblist.json`. The checked-in file is a Diffusion evaluation
example over NUMOSIM and the public PoL pack. Each run copies the exact joblist
into the result directory.

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
  - `data_hypothesis`: `RectifiedTraj`, `DirectReg`, or `Diffusion`.
  - `Q1`, `Q2`: byte settings for the evaluation grid.
  - `denoise_steps`: optional positive integer or list of integers. Omit for
    the default single-step behavior.
  - `sample_steps`: optional positive integer or list of integers for Diffusion
    reverse sampling. It cannot exceed the checkpoint's trained diffusion-step
    count.
- `baseline.models`: classic baselines to run. Supported values are
  `alpha_beta`, `causal_hampel`, `kalman_filter`, `kalman_rts`, `hampel`,
  `savgol`, `raw`, and `valhalla_meili`.
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
      "model_root": "./bin/model/DirectReg_online",
      "models": [
        "hybrid_online_1M_20260523_180637"
      ],
      "Q1": [0],
      "Q2": [0],
      "data_hypothesis": "DirectReg"
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
- `valhalla_meili_diagnostics/`: per-dataset Meili coverage, error-code, and
  sanitized request records when that baseline is selected.
- generated heatmaps when the relevant summaries are present.

### Valhalla Meili Map Matching

Valhalla Meili runs in Docker. The map workflow never reads trajectory data to
choose its boundary. It accepts only
`parquet_processor.dataset_noisy_boundary_corners` from a state JSON generated
by the parquet processor.

Download the fixed Japan and Georgia (Atlanta) Geofabrik extracts into
`dataset/raw/map/`:

```bash
PYTHONPATH=src env_RectifiedTraj/bin/python \
  -m baseline.models.valhalla_meili.manage download \
  --source japan --source georgia
```

Cut a dataset-specific map into `dataset/processed/map/<dataset>/`. The buffer
is soft: it is at most 1 km on each side and is clipped at the source-map
boundary.

```bash
PYTHONPATH=src env_RectifiedTraj/bin/python \
  -m baseline.models.valhalla_meili.manage cut \
  --dataset PoL_5s \
  --state-file dataset/state/state_PoL_5s.json \
  --source georgia \
  --buffer-km 1
```

For BlogWatcher at the University of Tokyo, first download the complete Japan
Geofabrik map into `dataset/raw/map/` with the `download --source japan`
command above. Keep the real processor-generated state file under
`dataset/state/`. The all-in-one workflow regenerates
`dataset/state/state_BlogWatcher.json` from the real UTokyo parquet before
launching the benchmark. Do not manually tailor a map from the repository's
fake BlogWatcher fixture or its current test metadata.

Build and manage the dataset-specific Valhalla tiles:

```bash
PYTHONPATH=src env_RectifiedTraj/bin/python \
  -m baseline.models.valhalla_meili.manage build --dataset PoL_5s --port 8002

PYTHONPATH=src env_RectifiedTraj/bin/python \
  -m baseline.models.valhalla_meili.manage status --dataset PoL_5s --port 8002

PYTHONPATH=src env_RectifiedTraj/bin/python \
  -m baseline.models.valhalla_meili.manage down --dataset PoL_5s --port 8002
```

The Docker service is pinned to Valhalla `3.8.3`, binds only to localhost, and
is limited to 2 CPUs and 8 GB RAM. Tile construction uses one Valhalla build
thread to reduce peak memory.

Enable Meili in `eval_joblist.json` with one explicit profile per dataset
family. Profile keys match the dataset name or its underscore-delimited prefix.
The `source` field selects the already-downloaded complete Geofabrik extract;
it never guesses from trajectory coordinates. Use different ports when
different map services may run concurrently.

```json
{
  "baseline": {
    "models": ["valhalla_meili"],
    "options": {
      "valhalla_meili": {
        "profiles": {
          "PoL_5s": {
            "map_id": "PoL_5s",
            "source": "georgia",
            "costing": "auto",
            "port": 8002
          },
          "NUMOSIM_Kanto": {
            "map_id": "NUMOSIM_Kanto",
            "source": "japan",
            "costing": "auto",
            "port": 8003
          },
          "BlogWatcher": {
            "map_id": "BlogWatcher",
            "source": "japan",
            "costing": "auto",
            "port": 8004
          }
        }
      }
    }
  },
  "run_baseline": true
}
```

When `run_benchmarks.py` prepares a Valhalla testing item, it checks for
`dataset/processed/map/<map_id>/<map_id>.osm.pbf`. If that file is missing, it
automatically cuts it from the configured raw Geofabrik map using only
`dataset/state/state_<map_id>.json`, with the 1 km soft buffer. Initialization
then builds missing or stale Valhalla tiles and waits for a fresh Docker server.
Benchmark timing and RSS sampling begin only after the server is ready. At the
end of that testing item, sampling stops and the Docker container is removed,
including when evaluation fails.

Long trajectories are sent as fixed 500-point requests with a 50-point
overlap. Overlap results are selected by distance from the request edge; routes
are never resampled to conceal unmatched points. The reported
`valhalla_meili_raw_fallback` benchmark uses the Meili coordinate for each
accepted point and preserves the original noisy coordinate for each rejected
point. Accuracy therefore covers every trajectory while the strict Meili
acceptance mask remains unchanged. `rejected_points` is also the number of raw
fallback points. Every Meili evaluation writes
`valhalla_meili_summary.json`, `valhalla_meili_error_codes.csv`, and
`valhalla_meili_requests.jsonl` with the fallback policy, coverage, HTTP
status, Valhalla error codes, and rejection evidence.

## Training

Training uses:

```bash
python src/theta_train.py
```

`src/theta_train.py` reads `src/config.json` for new runs. It is currently
configured for the causal DirectReg MLP; the reusable copy is
`src/baseline/models/mlp/config_causal_mlp_online_1m.json`.
Model artifacts are written under the configured `model_root`, with each run
containing:

- `log/config.json`
- `log/config_init.json`
- `best_ckpt/*.safetensors`
- `best_ckpt/*_full.pt`
- `fig/*.png`

Train the Diffusion baseline with its dedicated trainer and checked-in config:

```bash
PYTHONPATH=src python src/baseline/models/diffusion/diffusion_trainer.py \
  --config src/baseline/models/diffusion/config_diffusion_hybrid_online_1m.json
```

## University of Tokyo: BlogWatcher All-In-One

This workflow is for authorized University of Tokyo personnel with access to
the real BlogWatcher parquet dataset. Run every command from the repository
root. The checked-in BlogWatcher material is only a synthetic development
fixture; do not use it for the research evaluation or map boundary.

The all-in-one script performs the complete sequence:

1. Process the selected real parquet in test-only mode.
2. Generate `dataset/state/state_BlogWatcher.json`, including
   `parquet_processor.dataset_noisy_boundary_corners`.
3. Run `src/run_benchmarks.py` with `src/eval_joblist.json`.
4. Automatically tailor the Japan map when the processed BlogWatcher map is
   missing.
5. Build Valhalla tiles, run one Docker server per testing item, and remove the
   server after measurement.

### 1. Download the full Japan map first

This step is required before running BlogWatcher processing or evaluation. The
map comes from the official [Geofabrik Japan download
page](https://download.geofabrik.de/asia/japan.html). This repository pins the
2026-08-15 snapshot for reproducibility; do not substitute a regional extract
or `japan-latest.osm.pbf`.

The required source files are:

- `https://download.geofabrik.de/asia/japan-260815.osm.pbf`
- `https://download.geofabrik.de/asia/japan-260815.osm.pbf.md5`
- `https://download.geofabrik.de/asia/japan.poly`

The automatic downloader is the recommended method. Run it from the repository
root:

```bash
PYTHONPATH=src env_RectifiedTraj/bin/python \
  -m baseline.models.valhalla_meili.manage download \
  --source japan
```

It downloads and verifies these exact destination files:

```text
dataset/raw/map/japan-260815.osm.pbf
dataset/raw/map/japan.poly
```

If the Python downloader cannot be used, download the same pinned files
manually:

```bash
mkdir -p dataset/raw/map

curl --fail --location --continue-at - \
  --output dataset/raw/map/japan-260815.osm.pbf \
  https://download.geofabrik.de/asia/japan-260815.osm.pbf

curl --fail --location \
  --output dataset/raw/map/japan-260815.osm.pbf.md5 \
  https://download.geofabrik.de/asia/japan-260815.osm.pbf.md5

curl --fail --location \
  --output dataset/raw/map/japan.poly \
  https://download.geofabrik.de/asia/japan.poly

cd dataset/raw/map
md5sum --check japan-260815.osm.pbf.md5
cd ../../..
```

The checksum command must print `japan-260815.osm.pbf: OK`. Verify the files
before continuing:

```bash
ls -lh dataset/raw/map/japan-260815.osm.pbf dataset/raw/map/japan.poly
```

The PBF is approximately 2.5 GB. Keep at least 10 GiB free for the raw PBF,
tailored map, Valhalla tiles, and temporary build files. Docker is required for
the later cutting and Valhalla initialization steps, but not for the download
itself. Do not manually crop the raw map.

### 2. Install the real BlogWatcher parquet

The BlogWatcher research dataset is private and is not downloadable from this
repository or from Geofabrik. Obtain the authorized parquet file from the
University of Tokyo project storage or the project data custodian. Do not use
`dataset/raw/BlogWatcher/BG_big.parquet`; that file is synthetic test data.

Copy the authorized parquet into the required directory:

```bash
mkdir -p dataset/raw/BlogWatcher
cp /absolute/path/to/<real-blogwatcher-file>.parquet dataset/raw/BlogWatcher/
```

The resulting path must be:

```text
dataset/raw/BlogWatcher/<real-blogwatcher-file>.parquet
```

Confirm the intended file before running anything:

```bash
ls -lh dataset/raw/BlogWatcher/*.parquet
```

The filename may be arbitrary. The directory name `BlogWatcher` establishes
the dataset identity and causes the processor to write
`dataset/state/state_BlogWatcher.json`. If the directory contains multiple
parquet files, the `--file` argument in step 4 is mandatory.

### 3. Configure the BlogWatcher benchmark

Set `src/eval_joblist.json` to use the processed BlogWatcher trajectory output
and enable its Valhalla profile. The following is a baseline-only exact-test
example; add the required learned-model groups when evaluating trained models.

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
      "./dataset/processed/BlogWatcher/test/traj_test"
    ],
    "chunk_files": []
  },
  "data_source": {
    "raw_dataset_dir": null,
    "raw_test_files": null
  },
  "model_groups": [],
  "baseline": {
    "models": ["valhalla_meili"],
    "options": {
      "valhalla_meili": {
        "profiles": {
          "BlogWatcher": {
            "map_id": "BlogWatcher",
            "source": "japan",
            "costing": "auto",
            "port": 8004
          }
        }
      }
    }
  },
  "runtime": {
    "device": "cpu",
    "strict_init": true,
    "traj_parallel": 1
  },
  "run_baseline": true,
  "baseline_progress": true,
  "brief_summary": true,
  "brief_visualizer": false,
  "log_level": "INFO"
}
```

Do not manually create the BlogWatcher boundary metadata. The all-in-one run
regenerates it from the selected real parquet before `run_benchmarks.py` starts.
The map cutter reads only that processor metadata; it does not scan trajectory
files to choose a boundary.

### 4. Run the complete workflow

Run against an explicit parquet filename:

```bash
PYTHONPATH=src env_RectifiedTraj/bin/python \
  src/utils/evaluations/BlogWatcher_All_in_one.py \
  --file <name>.parquet
```

To upload the resulting benchmark directory to Weights & Biases:

```bash
PYTHONPATH=src env_RectifiedTraj/bin/python \
  src/utils/evaluations/BlogWatcher_All_in_one.py \
  --file <name>.parquet \
  --wandb
```

`--wandb` uploads only the generated result directory under
`bin/test_results/`; it does not upload the raw BlogWatcher parquet. Omit
`--wandb` for local-only output.

### 5. Verify the generated artifacts

After a successful first run, verify these files and directories:

- `dataset/state/state_BlogWatcher.json`: processor-generated metadata.
- `dataset/processed/BlogWatcher/test/`: processed evaluation data.
- `dataset/processed/map/BlogWatcher/BlogWatcher.osm.pbf`: automatically
  tailored map with a source-clipped soft buffer of at most 1 km.
- `dataset/processed/map/BlogWatcher/map_manifest.json`: map source, metadata
  hash, bounds, buffer, and output checksum.
- `dataset/processed/map/BlogWatcher/valhalla_tiles.tar`: Valhalla tiles.
- `bin/test_results/<run>/valhalla_meili_diagnostics/`: acceptance, rejection,
  HTTP status, Valhalla error-code, and adapter error-code evidence.

Valhalla initialization and tile construction are excluded from prediction
timing and RSS measurement. RSS sampling includes the running Valhalla
container. The testing-item container is removed after measurement; the
tailored map and tile archive remain for future runs.

The optional `--smoke-test` mode checks repository and joblist prerequisites
without reading or processing the parquet, cutting a map, starting Docker, or
running a benchmark. It is only a static prerequisite check and is not an
end-to-end validation. On a fresh installation, processed BlogWatcher paths do
not exist yet, so run the complete workflow first or omit the smoke test for
the initial run.

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
