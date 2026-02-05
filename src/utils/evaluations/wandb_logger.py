from __future__ import annotations

from pathlib import Path
from typing import Optional


def log_run_to_wandb(
    run_dir: str,
    project: str,
    entity: Optional[str] = None,
    run_name: Optional[str] = None,
) -> None:
    try:
        import wandb
    except Exception as exc:
        raise RuntimeError(f"wandb not available: {exc}") from exc

    run_path = Path(run_dir)
    if not run_path.exists():
        raise FileNotFoundError(run_path)

    run = wandb.init(project=project, entity=entity, name=run_name)

    try:
        # Log summary CSV if present
        csv_files = sorted(run_path.glob("uncertainty_band_summary*.csv"))
        if csv_files:
            import pandas as pd
            csv_path = csv_files[0]
            df = pd.read_csv(csv_path)
            run.log({"uncertainty_summary": wandb.Table(dataframe=df)})

        # Upload uncertainty-band aggregates if present
        detail_dir = run_path / "uncertainty_band_traj_test_result"
        if detail_dir.exists():
            parquet_files = list(detail_dir.glob("*.parquet"))
            csv_files = list(detail_dir.glob("*.csv"))
            if parquet_files:
                parquet_artifact = wandb.Artifact(
                    name=f"{run_path.name}_uncertainty_parquet",
                    type="utokyo_parquet",
                )
                parquet_artifact.add_dir(str(detail_dir))
                run.log_artifact(parquet_artifact)
            elif csv_files:
                csv_artifact = wandb.Artifact(
                    name=f"{run_path.name}_uncertainty_csv",
                    type="utokyo_csv",
                )
                csv_artifact.add_dir(str(detail_dir))
                run.log_artifact(csv_artifact)

        # Upload entire run directory as an artifact
        artifact = wandb.Artifact(name=run_path.name, type="utokyo_run")
        artifact.add_dir(str(run_path))
        run.log_artifact(artifact)
    finally:
        run.finish()
