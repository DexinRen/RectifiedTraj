#!/usr/bin/env python3
"""
Standalone data loader for RectifiedTraj.

This module intentionally duplicates key loading behavior used in training,
while adding a test-stream mode that emits records one-by-one from a test
directory with lazy file loading.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch


def _normalize_data_hypothesis(raw: object, default: str = "RectifiedTraj") -> str:
    """Normalize hypothesis aliases into canonical names.

    Args:
      raw: Raw hypothesis token from config/caller.
      default: Fallback value when token is empty.

    Returns:
      Canonical hypothesis name.
    """
    token = str(raw if raw is not None else "").strip().lower().replace("-", "_")
    if token in {"", "rf", "rectified_flow", "rectified", "rectifiedtraj", "rectified_traj"}:
        return "RectifiedTraj"
    if token in {"rr", "residualreg", "residual_reg", "residual", "residual_regression"}:
        return "ResidualReg"
    text = str(raw).strip() if raw is not None else ""
    return text if text else str(default)


class StandaloneDataLoader:
    """
    Standalone data loader with two modes:
    - train: epoch file loading + random batch sampling (theta_train-like).
    - test: lazy stream of records one-by-one across sorted .pt files.
    """

    def __init__(
        self,
        *,
        mode: str,
        data_dir: str,
        batch_size: int = 64,
        device: str = "cpu",
        max_steps: int = 37000,
        file_pattern: str = "*.pt",
        shuffle: bool = True,
        data_hypothesis: str = "RectifiedTraj",
    ):
        self.mode = str(mode).strip().lower()
        if self.mode not in {"train", "test"}:
            raise ValueError(f"Unsupported mode: {mode}. Expected one of: train, test.")

        self.data_dir = str(data_dir)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.max_steps = int(max_steps)
        self.file_pattern = str(file_pattern)
        self.shuffle = bool(shuffle)
        self.data_hypothesis = _normalize_data_hypothesis(data_hypothesis)

        self.file_list = sorted(glob.glob(str(Path(self.data_dir) / self.file_pattern)))
        if not self.file_list:
            raise FileNotFoundError(f"No .pt files found in {self.data_dir} matching {self.file_pattern}")

        # Train mode state (duplicated behavior from theta_train.DataLoader).
        self.X_t: Optional[torch.Tensor] = None
        self.V: Optional[torch.Tensor] = None
        self.t: Optional[torch.Tensor] = None
        self.N: int = 0
        self.perm: Optional[torch.Tensor] = None
        self.idx: int = 0

    @property
    def epoch_count(self) -> int:
        return len(self.file_list)

    # ------------------------------------------------------------------
    # Train mode API (theta_train-like)
    # ------------------------------------------------------------------
    def set(self, epoch_idx: int) -> None:
        if self.mode != "train":
            raise RuntimeError("set(epoch_idx) is only available in train mode.")
        files = self.file_list
        epoch_idx = int(epoch_idx) % len(files)
        file_path = files[epoch_idx]
        pack = torch.load(file_path, map_location="cpu")

        if not {"X_t", "V", "t"}.issubset(pack.keys()):
            raise KeyError(
                f"Train mode expects keys X_t/V/t in {file_path}; got {sorted(pack.keys())}"
            )

        x_t_raw = pack["X_t"]
        v_raw = pack["V"]
        t_raw = pack["t"]

        n_raw = int(x_t_raw.shape[0])
        n_div = (n_raw // 1000) * 1000
        n = min(self.max_steps, n_div)
        if n <= 0:
            raise ValueError(f"Computed epoch size is 0 for {file_path} (n_raw={n_raw}).")

        # ------------------------------------------------------------
        # Load canonical RF tensors first.
        # ------------------------------------------------------------
        x_t = x_t_raw[:n].to(self.device)
        v = v_raw[:n].to(self.device)
        t = t_raw[:n].to(self.device)

        # ------------------------------------------------------------
        # Hypothesis branch:
        #   RectifiedTraj: (X_t, V, t)
        #   ResidualReg : (X1, X0, t=1)
        # ------------------------------------------------------------
        if self.data_hypothesis == "ResidualReg":
            t_view = t.reshape(-1, 1, 1).to(dtype=x_t.dtype)
            one_minus_t = 1.0 - t_view
            x0 = x_t[:, :, :2] - v[:, :, :2] * t_view
            x1 = x_t[:, :, :2] + v[:, :, :2] * one_minus_t
            t_const = torch.ones((t.shape[0], 1), dtype=t.dtype, device=t.device)

            self.X_t = x1
            self.V = x0
            self.t = t_const
        else:
            self.X_t = x_t
            self.V = v
            self.t = t
        self.N = int(n)

        self.perm = (
            torch.randperm(self.N, device=self.device)
            if self.shuffle
            else torch.arange(self.N, device=self.device)
        )
        self.idx = 0

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.mode != "train":
            raise RuntimeError("get_batch() is only available in train mode.")
        if self.X_t is None or self.V is None or self.t is None or self.perm is None:
            raise RuntimeError("No epoch loaded. Call set(epoch_idx) first.")

        b = int(self.batch_size)
        if self.idx + b > self.N:
            self.perm = (
                torch.randperm(self.N, device=self.device)
                if self.shuffle
                else torch.arange(self.N, device=self.device)
            )
            self.idx = 0

        idx_slice = self.perm[self.idx : self.idx + b]
        self.idx += b

        return self.X_t[idx_slice], self.V[idx_slice], self.t[idx_slice]

    def next_epoch(self) -> None:
        return None

    def chunk_const(self, k: Optional[int] = None) -> Optional[int]:
        return None if k is None else int(k)

    # ------------------------------------------------------------------
    # Test mode API (new)
    # ------------------------------------------------------------------
    def iter_test_records(self) -> Iterator[Dict]:
        """
        Emit test records one-by-one with lazy file loading.

        Output schema:
        {
          "file_path": str,
          "file_index": int,
          "record_index": int,
          "record_type": "chunk_pair" | "trajectory" | "train_triplet",
          "payload": dict
        }
        """
        if self.mode != "test":
            raise RuntimeError("iter_test_records() is only available in test mode.")

        for fidx, file_path in enumerate(self.file_list):
            pack = torch.load(file_path, map_location="cpu")

            # Case A: chunk pair tensors (test chunk split): X0/X1
            if {"X0", "X1"}.issubset(pack.keys()):
                x0 = pack["X0"]
                x1 = pack["X1"]
                n = int(x0.shape[0])
                for ridx in range(n):
                    payload = {
                        "X0": x0[ridx],
                        "X1": x1[ridx],
                    }
                    if "accuracy" in pack:
                        payload["accuracy"] = pack["accuracy"][ridx]
                    if "error_range" in pack:
                        payload["error_range"] = pack["error_range"][ridx]
                    if "latitude_sigma" in pack:
                        payload["latitude_sigma"] = pack["latitude_sigma"][ridx]
                    if "longitude_sigma" in pack:
                        payload["longitude_sigma"] = pack["longitude_sigma"][ridx]
                    if "coord_space" in pack:
                        payload["coord_space"] = pack["coord_space"]
                    yield {
                        "file_path": file_path,
                        "file_index": fidx,
                        "record_index": ridx,
                        "record_type": "chunk_pair",
                        "payload": payload,
                    }
                continue

            # Case B: trajectory list files (fulltraj_*.pt / fulltraj_range_*.pt)
            if "trajectories" in pack and isinstance(pack["trajectories"], list):
                rows: List[dict] = pack["trajectories"]
                for ridx, traj in enumerate(rows):
                    yield {
                        "file_path": file_path,
                        "file_index": fidx,
                        "record_index": ridx,
                        "record_type": "trajectory",
                        "payload": traj,
                    }
                continue

            # Case C: train triplet tensors (X_t/V/t)
            if {"X_t", "V", "t"}.issubset(pack.keys()):
                x_t = pack["X_t"]
                v = pack["V"]
                t = pack["t"]
                n = int(x_t.shape[0])
                for ridx in range(n):
                    yield {
                        "file_path": file_path,
                        "file_index": fidx,
                        "record_index": ridx,
                        "record_type": "train_triplet",
                        "payload": {"X_t": x_t[ridx], "V": v[ridx], "t": t[ridx]},
                    }
                continue

            raise KeyError(
                f"Unsupported .pt schema in {file_path}; keys={sorted(pack.keys())}"
            )

    def iter_trajectory_sequences(self) -> Iterator[Dict]:
        """
        Emit trajectory-like inputs for baseline prediction.

        Output schema:
        {
          "file_path": str,
          "file_index": int,
          "record_index": int,
          "record_type": "trajectory" | "chunk_pair",
          "agent_id": Any | None,
          "noisy_lonlat": np.ndarray,   # (N,2) [lon, lat]
          "clean_lonlat": np.ndarray | None,  # (N,2) [lon, lat]
          "timestamps": np.ndarray | None,    # (N,)
          "error_range": np.ndarray | None
        }
        """

        def _to_np(x) -> np.ndarray:
            if isinstance(x, np.ndarray):
                return x
            if torch.is_tensor(x):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        if self.mode != "test":
            raise RuntimeError("iter_trajectory_sequences() is only available in test mode.")

        for rec in self.iter_test_records():
            payload = rec["payload"]
            rtype = rec["record_type"]

            if rtype == "trajectory":
                noisy = _to_np(payload.get("data"))
                clean = _to_np(payload.get("label")) if "label" in payload else None
                ts = _to_np(payload.get("timestamp")).reshape(-1) if "timestamp" in payload else None
                lat_sigma = (
                    _to_np(payload.get("latitude_sigma")).reshape(-1)
                    if "latitude_sigma" in payload
                    else None
                )
                lon_sigma = (
                    _to_np(payload.get("longitude_sigma")).reshape(-1)
                    if "longitude_sigma" in payload
                    else None
                )
                if "error_range" in payload:
                    err = _to_np(payload.get("error_range")).reshape(-1)
                elif "accuracy" in payload:
                    err = _to_np(payload.get("accuracy")).reshape(-1)
                else:
                    err = None
                if noisy.ndim != 2 or noisy.shape[1] != 2:
                    continue
                if clean is not None and (clean.ndim != 2 or clean.shape[1] != 2):
                    clean = None
                yield {
                    "file_path": rec["file_path"],
                    "file_index": rec["file_index"],
                    "record_index": rec["record_index"],
                    "record_type": "trajectory",
                    "agent_id": payload.get("agent_id"),
                    "noisy_lonlat": noisy.astype(float, copy=False),
                    "clean_lonlat": None if clean is None else clean.astype(float, copy=False),
                    "timestamps": None if ts is None else ts.astype(float, copy=False),
                    "error_range": None if err is None else err.astype(float, copy=False),
                    "latitude_sigma": None if lat_sigma is None else lat_sigma.astype(float, copy=False),
                    "longitude_sigma": None if lon_sigma is None else lon_sigma.astype(float, copy=False),
                }
                continue

            if rtype == "chunk_pair":
                x1 = _to_np(payload["X1"])
                x0 = _to_np(payload["X0"])
                if x1.ndim != 2 or x1.shape[1] < 2:
                    continue
                if x0.ndim != 2 or x0.shape[1] < 2:
                    continue
                ts = x1[:, 2] if x1.shape[1] >= 3 else None
                err = None
                lat_sigma = (
                    _to_np(payload.get("latitude_sigma")).reshape(-1)
                    if "latitude_sigma" in payload
                    else None
                )
                lon_sigma = (
                    _to_np(payload.get("longitude_sigma")).reshape(-1)
                    if "longitude_sigma" in payload
                    else None
                )
                if "error_range" in payload:
                    err = _to_np(payload["error_range"]).reshape(-1)
                elif "accuracy" in payload:
                    err = _to_np(payload["accuracy"]).reshape(-1)
                yield {
                    "file_path": rec["file_path"],
                    "file_index": rec["file_index"],
                    "record_index": rec["record_index"],
                    "record_type": "chunk_pair",
                    "agent_id": None,
                    "noisy_lonlat": x1[:, :2].astype(float, copy=False),
                    "clean_lonlat": x0[:, :2].astype(float, copy=False),
                    "timestamps": None if ts is None else np.asarray(ts, dtype=float),
                    "error_range": None if err is None else err.astype(float, copy=False),
                    "latitude_sigma": None if lat_sigma is None else lat_sigma.astype(float, copy=False),
                    "longitude_sigma": None if lon_sigma is None else lon_sigma.astype(float, copy=False),
                }
                continue


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standalone data loader (train/test)")
    p.add_argument("--mode", choices=["train", "test"], required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--pattern", default="*.pt")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max_steps", type=int, default=37000)
    p.add_argument("--data_hypothesis", default="RectifiedTraj")
    p.add_argument("--limit", type=int, default=10, help="Preview record count in test mode")
    p.add_argument("--epoch_idx", type=int, default=0, help="Epoch index in train mode")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    loader = StandaloneDataLoader(
        mode=args.mode,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        device=args.device,
        max_steps=args.max_steps,
        file_pattern=args.pattern,
        data_hypothesis=args.data_hypothesis,
    )

    if args.mode == "train":
        loader.set(args.epoch_idx)
        x_t, v, t = loader.get_batch()
        print(
            f"train batch loaded: X_t={tuple(x_t.shape)} V={tuple(v.shape)} "
            f"t={tuple(t.shape)} files={loader.epoch_count}"
        )
        return

    # test mode preview
    count = 0
    for rec in loader.iter_test_records():
        payload = rec["payload"]
        if rec["record_type"] == "chunk_pair":
            shape = tuple(payload["X1"].shape)
        elif rec["record_type"] == "trajectory":
            shape = tuple(payload.get("data", torch.empty(0)).shape)
        else:
            shape = tuple(payload["X_t"].shape)
        print(
            f"[{count}] type={rec['record_type']} file={Path(rec['file_path']).name} "
            f"idx={rec['record_index']} shape={shape}"
        )
        count += 1
        if count >= int(args.limit):
            break
    print(f"previewed={count} files={loader.epoch_count}")


if __name__ == "__main__":
    main()
