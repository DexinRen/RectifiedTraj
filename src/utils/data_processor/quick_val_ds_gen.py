"""
quick_val_ds_gen.py

Build two clean validation sets from processed val shards:

  1) SMALL set  -> ./dataset/quick_val.pt
  2) BIG set    -> ./dataset/quick_val_big.pt

Properties:
  - Samples come from multiple val shards (each shard = agent-group).
  - Each sample is a full (K,4) chunk + (K,2) V + (1,) t from parquet_processor.
  - Per-shard order is preserved (parquet_processor already sorts by timestamp per user).
  - NaN / Inf are rejected (mirrors parquet_processor filtering guarantees).
  - BIG set is strictly non-overlapping with SMALL set (based on EN coordinates).
"""

import glob
import hashlib
from pathlib import Path

import torch


# ============================================================
# === Config
# ============================================================
VAL_DIR    = "./dataset/processed/val"
SMALL_PATH = "./dataset/quick_val.pt"
BIG_PATH   = "./dataset/quick_val_big.pt"

SMALL_SIZE = 20000    # number of samples in small quick_val
BIG_SIZE   = 100000   # number of samples in big quick_val_big


# ============================================================
# === Helpers
# ============================================================
def hash_tensor_en(x: torch.Tensor) -> bytes:
    """
    Compute SHA1 hash of EN coordinates (K,2) for dedupe.

    x: (K, 2) float tensor (e, n)
    """
    x = x.contiguous()
    return hashlib.sha1(x.numpy().tobytes()).digest()


def sample_is_valid(X_t: torch.Tensor, V: torch.Tensor, t: torch.Tensor) -> bool:
    """
    Validate a single sample (X_t, V, t) with rules consistent with parquet_processor:

        - No NaN or Inf in X_t, V, or t.
        - is_start channel (X_t[:, 3]) must contain only {0, 1}.
    """
    if torch.isnan(X_t).any() or torch.isinf(X_t).any():
        return False
    if torch.isnan(V).any() or torch.isinf(V).any():
        return False
    if torch.isnan(t).any() or torch.isinf(t).any():
        return False

    is_start_vals = torch.unique(X_t[:, 3])
    for v in is_start_vals:
        if v.item() not in (0.0, 1.0):
            return False

    return True


# ============================================================
# === Loading val shards
# ============================================================
def load_val_shards():
    """
    Load all val .pt shards and return concatenated tensors plus shard-group IDs.

    Returns:
        X_all: (N, K, 4)
        V_all: (N, K, 2)
        t_all: (N, 1)
        agent_all: (N,) int32   # synthetic "agent-group" == shard index
    """
    print("=== [1] Scanning validation shard files ===")
    files = sorted(glob.glob(f"{VAL_DIR}/*.pt"))
    files = [f for f in files if not f.endswith(".len")]

    if not files:
        raise RuntimeError(f"No .pt files found in {VAL_DIR}")

    print(f"    Found {len(files)} shard files")

    X_all_list = []
    V_all_list = []
    t_all_list = []
    agent_all_list = []

    for shard_idx, fp in enumerate(files):
        blob = torch.load(fp, map_location="cpu")
        X = blob["X_t"]  # (N, K, 4)
        V = blob["V"]    # (N, K, 2)
        T = blob["t"]    # (N, 1)

        N_shard = len(X)
        print(f"    File {Path(fp).name} | {N_shard} samples")

        A = torch.full((N_shard,), shard_idx, dtype=torch.int32)

        X_all_list.append(X)
        V_all_list.append(V)
        t_all_list.append(T)
        agent_all_list.append(A)

    X_all = torch.cat(X_all_list, dim=0)
    V_all = torch.cat(V_all_list, dim=0)
    t_all = torch.cat(t_all_list, dim=0)
    agent_all = torch.cat(agent_all_list, dim=0)

    N_total = len(X_all)
    print(f"=== [2] Total samples across all val shards: {N_total} ===")
    return X_all, V_all, t_all, agent_all


def build_agent_index(agent_all: torch.Tensor):
    """
    Build mapping: agent_group_id -> list of row indices.

    agent_all: (N,) int32
    """
    print("=== [3] Building agent-group index ===")
    agent_to_rows = {}
    N_total = len(agent_all)

    for idx in range(N_total):
        aid = int(agent_all[idx].item())
        agent_to_rows.setdefault(aid, []).append(idx)

    print(f"    Found {len(agent_to_rows)} agent-groups (shards)")
    return agent_to_rows


# ============================================================
# === Build SMALL set
# ============================================================
def build_small_set(X_all, V_all, t_all, agent_all, agent_to_rows):
    """
    Build quick_val.pt (SMALL set) from val shards.

    - Iterates agent-groups in sorted order.
    - Iterates rows per agent-group in original order (no shuffle) to match
      parquet_processor's per-user timestamp ordering semantics.
    - Applies strict validity checks.
    - Deduplicates by EN coordinates.
    """
    print("=== [4] Building SMALL quick_val set ===")

    selected_rows = []
    hash_set = set()

    agent_ids = sorted(agent_to_rows.keys())
    for aid in agent_ids:
        rows = agent_to_rows[aid]  # original order from parquet_processor

        for r in rows:
            if len(selected_rows) >= SMALL_SIZE:
                break

            X_t = X_all[r]
            V_s = V_all[r]
            t_s = t_all[r]

            if not sample_is_valid(X_t, V_s, t_s):
                continue

            h = hash_tensor_en(X_t[:, :2])  # EN only
            if h in hash_set:
                continue

            hash_set.add(h)
            selected_rows.append(r)

        if len(selected_rows) >= SMALL_SIZE:
            break

    print(f"    Collected {len(selected_rows)} SMALL samples")
    if len(selected_rows) < SMALL_SIZE:
        raise RuntimeError(
            f"Insufficient valid SMALL samples: {len(selected_rows)} < SMALL_SIZE={SMALL_SIZE}"
        )

    idx_tensor = torch.tensor(selected_rows, dtype=torch.long)
    X_small = X_all.index_select(0, idx_tensor)
    V_small = V_all.index_select(0, idx_tensor)
    t_small = t_all.index_select(0, idx_tensor)

    torch.save({"X_t": X_small, "V": V_small, "t": t_small}, SMALL_PATH)
    print(f"    Saved SMALL set: {SMALL_PATH} (N={len(X_small)})")

    return X_small, V_small, t_small, hash_set, selected_rows


# ============================================================
# === Build BIG set (non-overlapping)
# ============================================================
def build_big_set(X_all, V_all, t_all, agent_all, agent_to_rows, existing_hash):
    """
    Build quick_val_big.pt (BIG set) from val shards.

    - Uses the same val shards as SMALL.
    - Uses the SAME hashing rule.
    - Ensures no overlap with SMALL by skipping any sample whose EN hash is already in existing_hash.
    - Also deduplicates within the BIG set itself (via the same hash set).
    """
    print("=== [5] Building BIG quick_val_big set (non-overlapping) ===")

    selected_rows = []
    hash_set = set(existing_hash)  # start from SMALL hashes

    agent_ids = sorted(agent_to_rows.keys())
    for aid in agent_ids:
        rows = agent_to_rows[aid]  # original order

        for r in rows:
            if len(selected_rows) >= BIG_SIZE:
                break

            X_t = X_all[r]
            V_s = V_all[r]
            t_s = t_all[r]

            if not sample_is_valid(X_t, V_s, t_s):
                continue

            h = hash_tensor_en(X_t[:, :2])
            if h in hash_set:
                continue  # either already in SMALL or already picked for BIG

            hash_set.add(h)
            selected_rows.append(r)

        if len(selected_rows) >= BIG_SIZE:
            break

    print(f"    Collected {len(selected_rows)} BIG samples (non-overlapping)")

    if len(selected_rows) < BIG_SIZE:
        raise RuntimeError(
            f"Insufficient valid BIG samples: {len(selected_rows)} < BIG_SIZE={BIG_SIZE}"
        )

    idx_tensor = torch.tensor(selected_rows, dtype=torch.long)
    X_big = X_all.index_select(0, idx_tensor)
    V_big = V_all.index_select(0, idx_tensor)
    t_big = t_all.index_select(0, idx_tensor)

    torch.save({"X_t": X_big, "V": V_big, "t": t_big}, BIG_PATH)
    print(f"    Saved BIG set: {BIG_PATH} (N={len(X_big)})")

    return X_big, V_big, t_big, selected_rows


# ============================================================
# === Main
# ============================================================
def main():
    X_all, V_all, t_all, agent_all = load_val_shards()
    agent_to_rows = build_agent_index(agent_all)

    # SMALL set
    X_small, V_small, t_small, small_hash, small_rows = build_small_set(
        X_all, V_all, t_all, agent_all, agent_to_rows
    )

    # BIG set
    X_big, V_big, t_big, big_rows = build_big_set(
        X_all, V_all, t_all, agent_all, agent_to_rows, small_hash
    )

    print("=== [6] Summary ===")
    print(f"    SMALL: {len(X_small)} samples -> {SMALL_PATH}")
    print(f"    BIG:   {len(X_big)} samples -> {BIG_PATH}")
    print("=== DONE: quick_val + quick_val_big generated ===")


if __name__ == "__main__":
    main()
