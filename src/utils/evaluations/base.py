import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import torch

from utils.data_processor.traj_extractor import (
    traj_extractor,
    traj_extractor_with_error_range,
)


class EvaluationManager:
    """
    Base class for evaluation managers.

    Responsibilities:
        - Model discovery
        - Config loading
        - Shared dataset utilities
    """

    def __init__(self, output_dir: str = "test_results"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = Path(output_dir)

    def _normalize_model_name(self, model_name: str) -> str:
        import re
        return re.sub(r"_\d{8}_\d{6}$", "", model_name)

    def _discover_models(self, model_root: str) -> List[str]:
        model_root = Path(model_root)

        if not model_root.exists():
            raise FileNotFoundError(f"Model root not found: {model_root}")

        model_names = []
        for model_dir in model_root.iterdir():
            if not model_dir.is_dir():
                continue

            has_checkpoints = False
            for ckpt_dir_name in ["best_ckpt", "ckpts"]:
                ckpt_dir = model_dir / ckpt_dir_name
                if ckpt_dir.exists() and any(ckpt_dir.glob("*_full.pt")):
                    has_checkpoints = True
                    break

            if has_checkpoints:
                model_names.append(model_dir.name)

        return sorted(model_names)

    def _find_best_checkpoint(self, model_dir: Path) -> Optional[str]:
        best_ckpt_dir = model_dir / "best_ckpt"
        if best_ckpt_dir.exists():
            best_ckpts = list(best_ckpt_dir.glob("*_full.pt"))
            if best_ckpts:
                return best_ckpts[0].name

        ckpts_dir = model_dir / "ckpts"
        if not ckpts_dir.exists():
            return None

        all_ckpts = sorted(ckpts_dir.glob("*_full.pt"), key=lambda p: p.stat().st_mtime)
        if all_ckpts:
            return all_ckpts[-1].name

        return None

    def _load_model_config(self, model_dir: Path) -> Dict:
        config_path = model_dir / "log" / "config.json"
        if not config_path.exists():
            self.logger.warning(f"Config not found: {config_path}")
            return {}

        with open(config_path, "r") as f:
            return json.load(f)

    def _load_or_generate_test_data(self, test_data_path: str, M: int, N: int) -> tuple:
        test_path = Path(test_data_path)
        if test_path.is_file() and test_path.suffix == ".pt":
            matching_file = test_path
        else:
            test_dir = test_path
            test_dir.mkdir(parents=True, exist_ok=True)

            target_median_min = int(N * 0.8)
            target_median_max = int(N * 1.2)

            matching_file = None
            for pt_file in test_dir.glob("fulltraj_*.pt"):
                parts = pt_file.stem.split("_")
                if len(parts) != 3:
                    continue

                try:
                    file_M = int(parts[1])
                    file_median = int(parts[2])

                    if file_M == M and target_median_min <= file_median <= target_median_max:
                        matching_file = pt_file
                        self.logger.info(
                            f"Found matching dataset: {pt_file.name} "
                            f"(M={file_M}, median={file_median}, target N={N})"
                        )
                        break
                except (ValueError, IndexError):
                    continue

            if matching_file is None:
                existing = sorted(test_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
                if existing:
                    matching_file = existing[0]
                    self.logger.info(
                        f"No matching dataset for M={M}, N={N}. "
                        f"Using existing dataset: {matching_file.name}"
                    )

            if matching_file is None:
                self.logger.info(
                    f"No matching dataset found for M={M}, N={N}. "
                    f"Generating new dataset..."
                )

                extractor_path = Path("./src/utils/data_processor")
                if str(extractor_path) not in sys.path:
                    sys.path.insert(0, str(extractor_path))

                result = traj_extractor(
                    parquet_dir="./dataset/raw",
                    M=M,
                    N=N,
                    output_dir=str(test_dir)
                )

                matching_file = Path(result["output_file"])
                self.logger.info(
                    f"Generated new dataset: {matching_file.name} "
                    f"({result['n_trajectories']} trajectories, "
                    f"{result['total_points']} total points)"
                )

        self.logger.info(f"Loading test trajectories from {matching_file.name}")
        data = torch.load(matching_file, map_location="cpu")
        raw_trajectories = data["trajectories"]

        trajectories = []
        for traj_dict in raw_trajectories:
            traj_obj = SimpleNamespace(
                agent_id=traj_dict["agent_id"],
                n_points=traj_dict["n_points"],
                noisy_gps=traj_dict["data"].numpy(),
                clean_gps=traj_dict["label"].numpy(),
            )
            trajectories.append(traj_obj)

        self.logger.info(f"Loaded {len(trajectories)} trajectories")
        dataset_name = matching_file.stem
        return trajectories, dataset_name

    def _load_or_generate_uncertainty_test_data(self, test_data_path: str, M: int, N: int) -> List:
        test_path = Path(test_data_path)
        if test_path.is_file() and test_path.suffix == ".pt":
            matching_file = test_path
        else:
            test_dir = test_path
            test_dir.mkdir(parents=True, exist_ok=True)

            target_median_min = int(N * 0.8)
            target_median_max = int(N * 1.2)

            matching_file = None
            for pt_file in test_dir.glob("fulltraj_range_*.pt"):
                parts = pt_file.stem.split("_")
                if len(parts) != 4:
                    continue

                try:
                    file_M = int(parts[2])
                    file_median = int(parts[3])
                    if file_M == M and target_median_min <= file_median <= target_median_max:
                        matching_file = pt_file
                        self.logger.info(
                            f"Found matching dataset: {pt_file.name} "
                            f"(M={file_M}, median={file_median}, target N={N})"
                        )
                        break
                except (ValueError, IndexError):
                    continue

            if matching_file is None:
                self.logger.info(
                    f"No matching dataset found for M={M}, N={N}. "
                    f"Generating new dataset (error_range)..."
                )

                extractor_path = Path("./src/utils/data_processor")
                if str(extractor_path) not in sys.path:
                    sys.path.insert(0, str(extractor_path))

                result = traj_extractor_with_error_range(
                    parquet_dir="./dataset/raw",
                    M=M,
                    N=N,
                    output_dir=str(test_dir)
                )

                matching_file = Path(result["output_file"])
                self.logger.info(
                    f"Generated new dataset: {matching_file.name} "
                    f"({result['n_trajectories']} trajectories, "
                    f"{result['total_points']} total points)"
                )

        self.logger.info(f"Loading test trajectories from {matching_file.name}")
        data = torch.load(matching_file, map_location="cpu")
        raw_trajectories = data["trajectories"]

        trajectories = []
        for traj_dict in raw_trajectories:
            traj_obj = SimpleNamespace(
                agent_id=traj_dict["agent_id"],
                n_points=traj_dict["n_points"],
                noisy_gps=traj_dict["data"].numpy(),
                ref_gps=traj_dict["label"].numpy(),
                error_range=traj_dict["error_range"].numpy(),
                timestamps=traj_dict.get("timestamp").numpy() if "timestamp" in traj_dict else None,
            )
            trajectories.append(traj_obj)

        self.logger.info(f"Loaded {len(trajectories)} trajectories (error_range)")
        return trajectories


class InTrainingEvaluator:
    """Quick validation during training (not implemented yet)."""

    pass


class RegionalEvaluator:
    """Q1/Q2 buckle region accuracy analysis (not implemented yet)."""

    pass
