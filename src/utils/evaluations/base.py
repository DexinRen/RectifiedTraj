import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch

from utils.data_loader_standalone import DataLoader

class EvaluationManager:
    """
    Base class for evaluation managers.

    Responsibilities:
        - Model discovery
        - Config loading
        - Shared dataset utilities
    """

    def __init__(self, output_dir: str = "./bin/test_results"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._traj_suite_roots = ["traj_test", "traj_test_debug"]

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
                if ckpt_dir.exists() and (
                    any(ckpt_dir.glob("*.safetensors")) or any(ckpt_dir.glob("*_full.pt"))
                ):
                    has_checkpoints = True
                    break

            if has_checkpoints:
                model_names.append(model_dir.name)

        return sorted(model_names)

    def _find_best_checkpoint(self, model_dir: Path) -> Optional[str]:
        best_ckpt_dir = model_dir / "best_ckpt"
        if best_ckpt_dir.exists():
            best_ckpts = sorted(best_ckpt_dir.glob("*.safetensors"))
            if best_ckpts:
                return best_ckpts[0].name
            best_ckpts = sorted(best_ckpt_dir.glob("*_full.pt"))
            if best_ckpts:
                return best_ckpts[0].name

        ckpts_dir = model_dir / "ckpts"
        if not ckpts_dir.exists():
            return None

        all_ckpts = sorted(ckpts_dir.glob("*.safetensors"), key=lambda p: p.stat().st_mtime)
        if all_ckpts:
            return all_ckpts[-1].name

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

    def _infer_dataset_name_from_test_path(self, test_path: Path) -> Optional[str]:
        parts = list(test_path.parts)
        for idx, part in enumerate(parts):
            if part.lower() == "processed" and idx + 1 < len(parts):
                return parts[idx + 1]
        return None

    def _resolve_dataset_display_name(
        self,
        source_path: str | Path,
        matched_file: Optional[Path] = None,
    ) -> str:
        source = Path(source_path)
        dataset_root = self._infer_dataset_name_from_test_path(
            matched_file if matched_file is not None else source
        )
        explicit_file = source.is_file() and source.suffix == ".pt"
        if explicit_file:
            file_path = matched_file if matched_file is not None else source
            stem = file_path.stem
            if not dataset_root:
                return stem
            root_norm = str(dataset_root).strip().lower()
            stem_norm = str(stem).strip().lower()
            if stem_norm == root_norm or stem_norm.startswith(root_norm + "_"):
                return stem
            return f"{dataset_root}_{stem}"
        if matched_file is not None and matched_file.is_file() and matched_file.suffix == ".pt":
            return dataset_root or matched_file.stem
        return dataset_root or source.stem or source.name or "unknown"

    def _is_numosim_dataset(self, dataset_name: str) -> bool:
        return str(dataset_name).lower().startswith("numosim")

    def _dataset_processed_root(self, dataset_name: str) -> Path:
        return Path("./dataset/processed") / dataset_name

    def _dataset_raw_root(self, dataset_name: str) -> Path:
        return Path("./dataset/raw") / dataset_name

    def _ensure_dataset_test_only_outputs(self, dataset_name: str) -> None:
        processed_root = self._dataset_processed_root(dataset_name)
        test_root = processed_root / "test"
        test_dir = test_root / "chunk_test"
        missing_test_chunks = (not test_dir.exists()) or (not any(test_dir.glob("*.pt")))

        missing = []
        if self._is_numosim_dataset(dataset_name):
            has_traj_suite = False
            for rel_dir in self._traj_suite_roots:
                suite_dir = test_root / rel_dir
                if suite_dir.exists() and any(suite_dir.rglob("*.pt")):
                    has_traj_suite = True
                    break
            if not has_traj_suite:
                missing.append(str(test_root / "traj_test"))

        if (not missing_test_chunks) and (not missing):
            return

        raw_root = self._dataset_raw_root(dataset_name)
        if not raw_root.exists():
            self.logger.warning(
                "Cannot auto-generate trajectory suite for %s: raw dataset not found at %s",
                dataset_name,
                raw_root,
            )
            return

        if missing:
            self.logger.info(
                "Missing trajectory suite files for %s (%d entries). Running parquet_processor test-only generation.",
                dataset_name,
                len(missing),
            )
        else:
            self.logger.info(
                "Missing test-only outputs for %s. Running parquet_processor test-only generation.",
                dataset_name,
            )
        try:
            from utils.data_processor.parquet_processor import parquet_processor_test_only
        except Exception as exc:
            self.logger.warning("Failed to import parquet_processor_test_only: %s", exc)
            return

        test_files = None
        if self._is_numosim_dataset(dataset_name):
            parquet_files = sorted(raw_root.glob("*.parquet"))
            test_files = [str(p) for p in parquet_files[-1:]]
            self.logger.info(
                "NUMOSIM test-only generation will use last %d parquet file(s).",
                len(test_files),
            )

        parquet_processor_test_only(
            raw_ds_path=str(raw_root),
            test_files=test_files,
            run_traj_extraction=True,
        )

    def _ensure_test_only_generated_if_needed(self, test_data_path: str) -> None:
        test_path = Path(test_data_path)
        dataset_name = self._infer_dataset_name_from_test_path(test_path)
        if dataset_name is None:
            return
        self._ensure_dataset_test_only_outputs(dataset_name)

    def _collect_candidate_pt_files(
        self,
        test_dir: Path,
        preferred_patterns: Optional[List[str]] = None,
    ) -> List[Path]:
        patterns = list(preferred_patterns or [])
        patterns.extend(["traj_*.pt", "fulltraj_*.pt", "*.pt"])
        seen = set()
        out: List[Path] = []
        for pattern in patterns:
            for path in sorted(test_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True):
                if not path.is_file():
                    continue
                if path in seen:
                    continue
                seen.add(path)
                out.append(path)
        return out

    @staticmethod
    def _extract_median_tokens_from_filename(path: Path) -> Optional[tuple[int, int]]:
        nums = [int(x) for x in re.findall(r"\d+", path.stem)]
        if len(nums) < 2:
            return None
        return int(nums[-2]), int(nums[-1])

    def _select_matching_test_file(
        self,
        test_data_path: str,
        M: int,
        N: int,
        preferred_patterns: Optional[List[str]] = None,
    ) -> Path:
        test_path = Path(test_data_path)
        if test_path.is_file() and test_path.suffix == ".pt":
            return test_path

        test_dir = test_path.parent if test_path.suffix == ".pt" else test_path
        test_dir.mkdir(parents=True, exist_ok=True)

        target_median_min = int(N * 0.8)
        target_median_max = int(N * 1.2)

        matching_file = None
        candidates = self._collect_candidate_pt_files(
            test_dir,
            preferred_patterns=preferred_patterns,
        )
        if (not candidates) and test_dir.parent.exists():
            parent_candidates = self._collect_candidate_pt_files(
                test_dir.parent,
                preferred_patterns=preferred_patterns,
            )
            if parent_candidates:
                self.logger.info(
                    "No .pt files found under %s; falling back to parent directory %s",
                    test_dir,
                    test_dir.parent,
                )
                candidates = parent_candidates
        for pt_file in candidates:
            parsed = self._extract_median_tokens_from_filename(pt_file)
            if parsed is None:
                continue
            file_M, file_median = parsed
            if file_M == int(M) and target_median_min <= file_median <= target_median_max:
                matching_file = pt_file
                self.logger.info(
                    f"Found matching dataset: {pt_file.name} "
                    f"(M={file_M}, median={file_median}, target N={N})"
                )
                break

        if matching_file is None and candidates:
            matching_file = candidates[0]
            self.logger.info(
                f"No matching dataset for M={M}, N={N}. "
                f"Using existing dataset: {matching_file.name}"
            )

        if matching_file is None:
            self._ensure_test_only_generated_if_needed(test_data_path)
            refreshed = self._collect_candidate_pt_files(
                test_dir,
                preferred_patterns=preferred_patterns,
            )
            if (not refreshed) and test_dir.parent.exists():
                refreshed = self._collect_candidate_pt_files(
                    test_dir.parent,
                    preferred_patterns=preferred_patterns,
                )
            if refreshed:
                matching_file = refreshed[0]

        if matching_file is None:
            raise FileNotFoundError(
                f"No usable trajectory .pt found in {test_dir} after test-only generation trigger."
            )
        return matching_file

    def _load_trajectories_via_dataloader(
        self,
        matching_file: Path,
        *,
        require_error_range: bool = False,
    ) -> List[SimpleNamespace]:
        loader = DataLoader(
            mode="test",
            data_dir=str(matching_file.parent),
            file_pattern=matching_file.name,
            shuffle=False,
        )

        trajectories: List[SimpleNamespace] = []
        for rec in loader.iter_trajectory_sequences():
            noisy = np.asarray(rec.get("noisy_lonlat"), dtype=float)
            clean = rec.get("clean_lonlat")
            if clean is None:
                continue
            clean = np.asarray(clean, dtype=float)
            if noisy.ndim != 2 or noisy.shape[1] != 2:
                continue
            if clean.ndim != 2 or clean.shape[1] != 2:
                continue

            ts = rec.get("timestamps")
            ts_arr = None if ts is None else np.asarray(ts, dtype=float).reshape(-1)
            err = rec.get("error_range")
            err_arr = None if err is None else np.asarray(err, dtype=float).reshape(-1)
            lat_sigma = rec.get("latitude_sigma")
            lat_sigma_arr = None if lat_sigma is None else np.asarray(lat_sigma, dtype=float).reshape(-1)
            lon_sigma = rec.get("longitude_sigma")
            lon_sigma_arr = None if lon_sigma is None else np.asarray(lon_sigma, dtype=float).reshape(-1)

            n = min(int(noisy.shape[0]), int(clean.shape[0]))
            if ts_arr is not None:
                n = min(n, int(ts_arr.shape[0]))
            if require_error_range:
                if err_arr is None:
                    continue
                n = min(n, int(err_arr.shape[0]))
            if lat_sigma_arr is not None:
                n = min(n, int(lat_sigma_arr.shape[0]))
            if lon_sigma_arr is not None:
                n = min(n, int(lon_sigma_arr.shape[0]))

            if n <= 0:
                continue

            if require_error_range:
                trajectories.append(
                    SimpleNamespace(
                        agent_id=rec.get("agent_id"),
                        source_file=str(rec.get("file_path", "") or ""),
                        file_index=rec.get("file_index"),
                        record_index=rec.get("record_index"),
                        n_points=int(n),
                        noisy_gps=noisy[:n],
                        ref_gps=clean[:n],
                        error_range=err_arr[:n],
                        timestamps=None if ts_arr is None else ts_arr[:n],
                        latitude_sigma=None if lat_sigma_arr is None else lat_sigma_arr[:n],
                        longitude_sigma=None if lon_sigma_arr is None else lon_sigma_arr[:n],
                    )
                )
            else:
                trajectories.append(
                    SimpleNamespace(
                        agent_id=rec.get("agent_id"),
                        source_file=str(rec.get("file_path", "") or ""),
                        file_index=rec.get("file_index"),
                        record_index=rec.get("record_index"),
                        n_points=int(n),
                        noisy_gps=noisy[:n],
                        clean_gps=clean[:n],
                        timestamps=None if ts_arr is None else ts_arr[:n],
                        latitude_sigma=None if lat_sigma_arr is None else lat_sigma_arr[:n],
                        longitude_sigma=None if lon_sigma_arr is None else lon_sigma_arr[:n],
                    )
                )
        return trajectories

    def _cap_loaded_trajectories(
        self,
        trajectories: List[SimpleNamespace],
        *,
        max_traj: int,
        max_points: int,
        require_error_range: bool,
    ) -> List[SimpleNamespace]:
        traj_cap = max(1, int(max_traj))
        point_cap = max(1, int(max_points))
        selected = trajectories[:traj_cap]
        out: List[SimpleNamespace] = []
        for traj in selected:
            n_cur = int(getattr(traj, "n_points", 0))
            n_use = min(n_cur, point_cap)
            if n_use <= 0:
                continue
            if require_error_range:
                out.append(
                    SimpleNamespace(
                        agent_id=getattr(traj, "agent_id", None),
                        source_file=str(getattr(traj, "source_file", "") or ""),
                        file_index=getattr(traj, "file_index", None),
                        record_index=getattr(traj, "record_index", None),
                        n_points=int(n_use),
                        noisy_gps=np.asarray(traj.noisy_gps)[:n_use],
                        ref_gps=np.asarray(traj.ref_gps)[:n_use],
                        error_range=np.asarray(traj.error_range)[:n_use],
                        timestamps=None
                        if getattr(traj, "timestamps", None) is None
                        else np.asarray(traj.timestamps)[:n_use],
                        latitude_sigma=None
                        if getattr(traj, "latitude_sigma", None) is None
                        else np.asarray(traj.latitude_sigma)[:n_use],
                        longitude_sigma=None
                        if getattr(traj, "longitude_sigma", None) is None
                        else np.asarray(traj.longitude_sigma)[:n_use],
                    )
                )
            else:
                out.append(
                    SimpleNamespace(
                        agent_id=getattr(traj, "agent_id", None),
                        source_file=str(getattr(traj, "source_file", "") or ""),
                        file_index=getattr(traj, "file_index", None),
                        record_index=getattr(traj, "record_index", None),
                        n_points=int(n_use),
                        noisy_gps=np.asarray(traj.noisy_gps)[:n_use],
                        clean_gps=np.asarray(traj.clean_gps)[:n_use],
                        timestamps=None
                        if getattr(traj, "timestamps", None) is None
                        else np.asarray(traj.timestamps)[:n_use],
                        latitude_sigma=None
                        if getattr(traj, "latitude_sigma", None) is None
                        else np.asarray(traj.latitude_sigma)[:n_use],
                        longitude_sigma=None
                        if getattr(traj, "longitude_sigma", None) is None
                        else np.asarray(traj.longitude_sigma)[:n_use],
                    )
                )
        return out

    @staticmethod
    def _to_cpu_tensor(x) -> torch.Tensor:
        if torch.is_tensor(x):
            return x.detach().cpu()
        return torch.as_tensor(x)

    def _build_test_loader(
        self,
        path_value: str | Path,
    ) -> DataLoader:
        path = Path(path_value)
        if path.is_file():
            data_dir = str(path.parent)
            pattern = path.name
        else:
            data_dir = str(path)
            pattern = "*.pt"
        return DataLoader(
            mode="test",
            data_dir=data_dir,
            file_pattern=pattern,
            shuffle=False,
        )

    @staticmethod
    def _normalize_coord_space_token(token: object) -> str:
        text = str(token or "").strip().upper()
        return text if text else "UNKNOWN"

    @staticmethod
    def _infer_coord_space_from_xy(xy: torch.Tensor) -> str:
        try:
            arr = np.asarray(xy, dtype=np.float64)
        except Exception:
            return "UNKNOWN"
        if arr.ndim != 2 or arr.shape[1] < 2:
            return "UNKNOWN"
        lon = arr[:, 0].reshape(-1)
        lat = arr[:, 1].reshape(-1)
        if lon.size <= 0 or lat.size <= 0:
            return "UNKNOWN"
        mask = np.isfinite(lon) & np.isfinite(lat)
        if not np.any(mask):
            return "UNKNOWN"
        lon = lon[mask]
        lat = lat[mask]
        if np.all((lon >= -180.0) & (lon <= 180.0) & (lat >= -90.0) & (lat <= 90.0)):
            return "GPS"
        if np.any(np.abs(lon) > 180.0) or np.any(np.abs(lat) > 90.0):
            return "ENU"
        return "UNKNOWN"

    def _load_chunk_pairs_via_dataloader(
        self,
        test_dir: str,
        *,
        max_chunks: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int, str]:
        loader = self._build_test_loader(test_dir)

        x0_list: List[torch.Tensor] = []
        x1_list: List[torch.Tensor] = []
        ts_list: List[torch.Tensor] = []
        coord_space_tokens: List[str] = []
        any_ts = False
        all_ts = True

        limit = None if max_chunks is None else max(0, int(max_chunks))
        for rec in loader.iter_test_records():
            payload = rec["payload"]
            rtype = rec["record_type"]
            x0 = None
            x1 = None
            ts = None

            if rtype == "chunk_pair":
                x0_t_raw = self._to_cpu_tensor(payload["X0"])
                x1_t_raw = self._to_cpu_tensor(payload["X1"])
                if x0_t_raw.ndim != 2 or x1_t_raw.ndim != 2 or x0_t_raw.shape[1] < 2 or x1_t_raw.shape[1] < 2:
                    continue
                x0_t = x0_t_raw.float()
                x1_t = x1_t_raw.float()
                x0 = x0_t[:, :2]
                x1 = x1_t[:, :2]
                if int(x1_t_raw.shape[1]) >= 3:
                    ts = torch.cumsum(x1_t_raw[:, 2].to(dtype=torch.float64), dim=0)
                token = self._normalize_coord_space_token(payload.get("coord_space"))
                if token == "UNKNOWN":
                    token = self._infer_coord_space_from_xy(x1)
                if token != "UNKNOWN":
                    coord_space_tokens.append(token)
            elif rtype == "train_triplet":
                xt = self._to_cpu_tensor(payload["X_t"]).float()
                v = self._to_cpu_tensor(payload["V"]).float()
                t_tensor = self._to_cpu_tensor(payload["t"]).float().reshape(-1)
                if xt.ndim != 2 or v.ndim != 2 or xt.shape[1] < 2 or v.shape[1] < 2:
                    continue
                if int(t_tensor.numel()) <= 0:
                    continue
                t_scalar = float(t_tensor[0].item())
                x0 = xt[:, :2] - v[:, :2] * t_scalar
                x1 = xt[:, :2] + v[:, :2] * (1.0 - t_scalar)
                ts = None
                coord_space_tokens.append("ENU")
            else:
                continue

            if x0 is None or x1 is None:
                continue
            if x0.shape != x1.shape:
                continue
            if x0.shape[0] <= 0:
                continue

            x0_list.append(x0)
            x1_list.append(x1)
            if ts is None:
                all_ts = False
            else:
                any_ts = True
                ts_list.append(ts)

            if limit is not None and limit > 0 and len(x0_list) >= limit:
                break

        if not x0_list:
            raise RuntimeError(f"No chunk-pair records found via DataLoader for {test_dir}")

        x0_out = torch.stack(x0_list, dim=0)
        x1_out = torch.stack(x1_list, dim=0)
        if any_ts and all_ts and len(ts_list) == len(x0_list):
            ts_out: Optional[torch.Tensor] = torch.stack(ts_list, dim=0)
        else:
            if any_ts and not all_ts:
                self.logger.warning(
                    "Mixed timestamp availability in %s; chunk baseline timestamp support disabled for this run.",
                    test_dir,
                )
            ts_out = None
        coord_tokens = sorted(set(t for t in coord_space_tokens if t))
        if not coord_tokens:
            coord_space = "UNKNOWN"
        elif len(coord_tokens) == 1:
            coord_space = coord_tokens[0]
        else:
            self.logger.warning(
                "Mixed chunk coord_space tokens in %s: %s; treating as UNKNOWN.",
                test_dir,
                coord_tokens,
            )
            coord_space = "UNKNOWN"
        return x0_out, x1_out, ts_out, int(loader.epoch_count), coord_space

    def _load_or_generate_test_data(self, test_data_path: str, M: int, N: int) -> tuple:
        matching_file = self._select_matching_test_file(
            test_data_path,
            M,
            N,
            preferred_patterns=["traj_*.pt", "fulltraj_*.pt"],
        )
        self.logger.debug(f"Loading test trajectories from {matching_file.name} via DataLoader")
        trajectories = self._load_trajectories_via_dataloader(
            matching_file,
            require_error_range=False,
        )
        trajectories = self._cap_loaded_trajectories(
            trajectories,
            max_traj=int(M),
            max_points=int(N),
            require_error_range=False,
        )
        if not trajectories:
            raise RuntimeError(f"No trajectory records found in {matching_file}")
        self.logger.debug(f"Loaded {len(trajectories)} trajectories")
        dataset_name = self._resolve_dataset_display_name(test_data_path, matching_file)
        return trajectories, dataset_name

    def _load_or_generate_uncertainty_test_data(self, test_data_path: str, M: int, N: int) -> List:
        matching_file = self._select_matching_test_file(
            test_data_path,
            M,
            N,
            preferred_patterns=["fulltraj_range_*.pt", "traj_range_*.pt", "traj_*.pt", "*.pt"],
        )
        self.logger.debug(
            f"Loading uncertainty test trajectories from {matching_file.name} via DataLoader"
        )
        trajectories = self._load_trajectories_via_dataloader(
            matching_file,
            require_error_range=True,
        )
        trajectories = self._cap_loaded_trajectories(
            trajectories,
            max_traj=int(M),
            max_points=int(N),
            require_error_range=True,
        )
        if not trajectories:
            raise RuntimeError(
                f"No uncertainty trajectories with error_range found in {matching_file}"
            )
        self.logger.debug(f"Loaded {len(trajectories)} trajectories (error_range)")
        return trajectories


class InTrainingEvaluator:
    """Quick validation during training (not implemented yet)."""

    pass


class RegionalEvaluator:
    """Q1/Q2 buckle region accuracy analysis (not implemented yet)."""

    pass
