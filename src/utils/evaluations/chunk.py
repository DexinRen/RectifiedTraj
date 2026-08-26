import csv
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
from utils.evaluations.result_io import write_rows_to_csv


def _runtime_device_label() -> str:
    raw = str(
        os.getenv(
            "RECTIFIEDTRAJ_RUNTIME_DEVICE_EFFECTIVE",
            os.getenv("RECTIFIEDTRAJ_DEVICE", "unknown"),
        )
    ).strip().lower()
    if raw.startswith("cuda"):
        return "cuda"
    if raw == "cpu":
        return "cpu"
    return raw or "unknown"


class ChunkEvaluator:
    """
    Evaluate chunk-wise ENU denoising on preprocessed test chunks.

    Outputs:
        - CSV: chunk_evaluation_summary.csv
        - CSV: chunk_bytewise_summary.csv
        - CSV: chunk_pointwise_summary.csv
        Heatmap PNGs are generated later by utils/data_visualizer/make_heatmaps.py
        from the aggregated summary CSVs.
    """

    def __init__(self, output_dir: str = "./bin/test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "chunk_evaluation_summary.csv"
        self.logger = logging.getLogger("ChunkEvaluator")

        header_cols = [
            "model_name",
            "model_tag",
            "device",
            "dataset_name",
            "K",
            "Q1",
            "Q2",
            "err_l1_mean_full",
            "err_l1_median_full",
            "err_l1_p95_full",
            "err_l1_std_full",
            "err_mean_full",
            "err_median_full",
            "err_p95_full",
            "err_std_full",
            "err_l1_mean_mid",
            "err_l1_median_mid",
            "err_l1_p95_mid",
            "err_l1_std_mid",
            "err_mean_mid",
            "err_median_mid",
            "err_p95_mid",
            "err_std_mid",
            "num_tested_chunks",
            "attempted_chunks",
            "accepted_chunks",
            "rejected_chunks",
            "chunk_rejection_rate",
            "attempted_points",
            "accepted_points",
            "rejected_points",
            "point_rejection_rate",
            "attempted_requests",
            "accepted_requests",
            "rejected_requests",
            "request_rejection_rate",
            "valhalla_error_code_counts",
            "adapter_error_code_counts",
            "test_timestamp",
            "model_full_name",
        ]

        if not self.csv_path.exists():
            self.csv_path.write_text(",".join(header_cols) + "\n")
        else:
            with open(self.csv_path, "r", newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)
            if not rows or rows[0] != header_cols:
                fixed_rows = [header_cols]
                if rows:
                    header = rows[0]
                    for row in rows[1:]:
                        row_map = {k: v for k, v in zip(header, row)}
                        fixed_rows.append([row_map.get(col, "") for col in header_cols])
                with open(self.csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(fixed_rows)

    def _append_row(self, row: Dict):
        def _fmt(value, fmt: str):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return "NA"
            return format(value, fmt)

        csv_row = (
            f"{row['model_name']},{row.get('model_tag', 'NA')},{row.get('device', _runtime_device_label())},{row.get('dataset_name', 'NA')},"
            f"{_fmt(row.get('K'), 'd')},{_fmt(row.get('Q1'), 'd')},{_fmt(row.get('Q2'), 'd')},"
            f"{_fmt(row.get('err_l1_mean_full'), '.6f')},{_fmt(row.get('err_l1_median_full'), '.6f')},{_fmt(row.get('err_l1_p95_full'), '.6f')},{_fmt(row.get('err_l1_std_full'), '.6f')},"
            f"{_fmt(row.get('err_mean_full'), '.6f')},{_fmt(row.get('err_median_full'), '.6f')},{_fmt(row.get('err_p95_full'), '.6f')},{_fmt(row.get('err_std_full'), '.6f')},"
            f"{_fmt(row.get('err_l1_mean_mid'), '.6f')},{_fmt(row.get('err_l1_median_mid'), '.6f')},{_fmt(row.get('err_l1_p95_mid'), '.6f')},{_fmt(row.get('err_l1_std_mid'), '.6f')},"
            f"{_fmt(row.get('err_mean_mid'), '.6f')},{_fmt(row.get('err_median_mid'), '.6f')},{_fmt(row.get('err_p95_mid'), '.6f')},{_fmt(row.get('err_std_mid'), '.6f')},"
            f"{_fmt(row.get('num_tested_chunks'), 'd')},"
            f"{_fmt(row.get('attempted_chunks'), 'd')},{_fmt(row.get('accepted_chunks'), 'd')},"
            f"{_fmt(row.get('rejected_chunks'), 'd')},{_fmt(row.get('chunk_rejection_rate'), '.8f')},"
            f"{_fmt(row.get('attempted_points'), 'd')},{_fmt(row.get('accepted_points'), 'd')},"
            f"{_fmt(row.get('rejected_points'), 'd')},{_fmt(row.get('point_rejection_rate'), '.8f')},"
            f"{_fmt(row.get('attempted_requests'), 'd')},{_fmt(row.get('accepted_requests'), 'd')},"
            f"{_fmt(row.get('rejected_requests'), 'd')},{_fmt(row.get('request_rejection_rate'), '.8f')},"
            f"{';'.join(f'{key}:{value}' for key, value in sorted((row.get('valhalla_error_code_counts') or {}).items())) or 'NA'},"
            f"{';'.join(f'{key}:{value}' for key, value in sorted((row.get('adapter_error_code_counts') or {}).items())) or 'NA'},"
            f"{row.get('test_timestamp')},{row.get('model_full_name', row.get('model_name', ''))}\n"
        )
        with open(self.csv_path, "a") as f:
            f.write(csv_row)

    def _save_positionwise_heatmap(
        self,
        rows: list[Dict],
        *,
        dataset_name: str,
        value_key: str,
        column_prefix: str,
        out_csv_name: str,
    ) -> None:
        if not rows:
            return

        meta_cols = [
            "model_name",
            "model_tag",
            "dataset_name",
            "model_root",
            "Q1",
            "Q2",
        ]

        def _normalize_display_name(name: str | None) -> str | None:
            value = str(name or "NA")
            return value

        csv_rows = []
        max_len = 0

        for row in rows:
            display_name = _normalize_display_name(row.get("model_name"))
            if display_name is None:
                continue
            values = row.get(value_key)
            if values is None:
                continue
            values = np.asarray(values, dtype=float).reshape(-1)
            if values.size == 0:
                continue
            valid = values[np.isfinite(values)]
            if valid.size == 0:
                continue
            max_len = max(max_len, int(values.size))
            csv_rows.append({
                "model_name": display_name,
                "model_tag": row.get("model_tag"),
                "dataset_name": row.get("dataset_name", dataset_name),
                "model_root": row.get("model_root", ""),
                "Q1": row.get("Q1", ""),
                "Q2": row.get("Q2", ""),
                "model_full_name": row.get("model_full_name", row.get("model_name", "")),
                "values": values,
            })

        if not csv_rows or max_len <= 0:
            return

        out_csv = self.output_dir / out_csv_name
        header = meta_cols + [f"{column_prefix}_{i}" for i in range(max_len)] + ["model_full_name"]
        merged: dict[tuple[str, ...], list[float]] = {}

        # Keep previous rows so multi-pass chunk runs (e.g., per-kalman-mode) do not overwrite prior models.
        if out_csv.exists():
            with out_csv.open("r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    meta = tuple(str(row.get(col, "")).strip() for col in meta_cols)
                    model_full = str(row.get("model_full_name", "")).strip()
                    if not meta[0]:
                        continue
                    try:
                        vals = []
                        i = 0
                        while True:
                            col = f"{column_prefix}_{i}"
                            if col not in row:
                                break
                            vals.append(float(row.get(col, "nan")))
                            i += 1
                    except Exception:
                        continue
                    merged[(*meta, model_full)] = vals

        for r in csv_rows:
            key = tuple(str(r.get(col, "")).strip() for col in meta_cols)
            model_full = str(r.get("model_full_name", "") or "")
            merged[key] = [float(v) for v in np.asarray(r["values"], dtype=float)]
            merged[(*key, model_full)] = merged.pop(key)

        max_len = max((len(v) for v in merged.values()), default=0)
        if max_len <= 0:
            return
        header = meta_cols + [f"{column_prefix}_{i}" for i in range(max_len)] + ["model_full_name"]

        with out_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for key, vals in sorted(merged.items(), key=lambda x: x[0]):
                meta = key[: len(meta_cols)]
                model_full = key[len(meta_cols)] if len(key) > len(meta_cols) else ""
                padded = list(vals) + [float("nan")] * (max_len - len(vals))
                writer.writerow([*meta, *padded, model_full])

    def save_bytewise_heatmap(self, rows: list[Dict], dataset_name: str = "chunk_test") -> None:
        self._save_positionwise_heatmap(
            rows,
            dataset_name=dataset_name,
            value_key="byte_mean",
            column_prefix="byte",
            out_csv_name="chunk_bytewise_summary.csv",
        )

    def save_pointwise_heatmap(self, rows: list[Dict], dataset_name: str = "chunk_test") -> None:
        self._save_positionwise_heatmap(
            rows,
            dataset_name=dataset_name,
            value_key="point_mean",
            column_prefix="point",
            out_csv_name="chunk_pointwise_summary.csv",
        )

    def save_chunk_p_val_rows(self, rows: list[Dict]) -> None:
        if not rows:
            return

        out_csv = self.output_dir / "chunk_p_val.csv"
        existing_rows: list[dict[str, str]] = []
        if out_csv.exists():
            with out_csv.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)

        merged_rows = existing_rows + [{str(k): v for k, v in row.items()} for row in rows]
        field_order = [
            "sample_index",
            "dataset_name",
            "model_name",
            "model_tag",
            "device",
            "K",
            "Q1",
            "Q2",
            "n_points_full",
            "n_points_mid",
            "mean_l2_err_full",
            "median_l2_err_full",
            "p95_l2_err_full",
            "std_l2_err_full",
            "mean_l1_err_full",
            "median_l1_err_full",
            "p95_l1_err_full",
            "std_l1_err_full",
            "mean_l2_err_mid",
            "median_l2_err_mid",
            "p95_l2_err_mid",
            "std_l2_err_mid",
            "mean_l1_err_mid",
            "median_l1_err_mid",
            "p95_l1_err_mid",
            "std_l1_err_mid",
            "test_timestamp",
            "model_full_name",
        ]
        write_rows_to_csv(merged_rows, out_csv, field_order=field_order)
