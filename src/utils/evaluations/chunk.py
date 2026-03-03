import csv
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt


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
    """

    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "chunk_evaluation_summary.csv"
        self.logger = logging.getLogger("ChunkEvaluator")

        header_cols = [
            "model_name",
            "model_tag",
            "device",
            "dataset_name",
            "denoise_method",
            "K",
            "Q1",
            "Q2",
            "t_delta",
            "N_steps",
            "err_mean_full",
            "err_median_full",
            "err_p95_full",
            "err_std_full",
            "err_mean_mid",
            "err_median_mid",
            "err_p95_mid",
            "err_std_mid",
            "avg_denoise_time_sec",
            "avg_denoise_time_sec_per_point",
            "latency_p50_ms",
            "latency_p95_ms",
            "latency_max_ms",
            "throughput_points_per_sec",
            "peak_rss_mb",
            "peak_vram_mb",
            "calibration_time_sec",
            "calibration_peak_rss_mb",
            "calibration_peak_vram_mb",
            "num_tested_chunks",
            "test_timestamp",
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
            f"{row['denoise_method']},"
            f"{_fmt(row.get('K'), 'd')},{_fmt(row.get('Q1'), 'd')},{_fmt(row.get('Q2'), 'd')},"
            f"{_fmt(row.get('t_delta'), '.4f')},{_fmt(row.get('N_steps'), 'd')},"
            f"{_fmt(row.get('err_mean_full'), '.6f')},{_fmt(row.get('err_median_full'), '.6f')},{_fmt(row.get('err_p95_full'), '.6f')},{_fmt(row.get('err_std_full'), '.6f')},"
            f"{_fmt(row.get('err_mean_mid'), '.6f')},{_fmt(row.get('err_median_mid'), '.6f')},{_fmt(row.get('err_p95_mid'), '.6f')},{_fmt(row.get('err_std_mid'), '.6f')},"
            f"{_fmt(row.get('avg_denoise_time_sec', row.get('avg_time_s')), '.6f')},"
            f"{_fmt(row.get('avg_denoise_time_sec_per_point', row.get('avg_time_per_point_s')), '.8f')},"
            f"{_fmt(row.get('latency_p50_ms'), '.4f')},"
            f"{_fmt(row.get('latency_p95_ms'), '.4f')},"
            f"{_fmt(row.get('latency_max_ms'), '.4f')},"
            f"{_fmt(row.get('throughput_points_per_sec'), '.4f')},"
            f"{_fmt(row.get('peak_rss_mb'), '.4f')},"
            f"{_fmt(row.get('peak_vram_mb'), '.4f')},"
            f"{_fmt(row.get('calibration_time_sec'), '.6f')},"
            f"{_fmt(row.get('calibration_peak_rss_mb'), '.4f')},"
            f"{_fmt(row.get('calibration_peak_vram_mb'), '.4f')},"
            f"{_fmt(row.get('num_tested_chunks'), 'd')},{row.get('test_timestamp')}\n"
        )
        with open(self.csv_path, "a") as f:
            f.write(csv_row)

    def save_bytewise_heatmap(self, rows: list[Dict], dataset_name: str = "chunk_test") -> None:
        if not rows:
            return

        def _normalize_display_name(name: str | None) -> str | None:
            value = str(name or "NA")
            return value

        matrices = []
        labels = []
        csv_rows = []

        for row in rows:
            display_name = _normalize_display_name(row.get("model_name"))
            if display_name is None:
                continue
            byte_mean = row.get("byte_mean")
            if byte_mean is None:
                continue
            byte_mean = np.asarray(byte_mean, dtype=float)
            if byte_mean.shape[0] != 32:
                continue
            mean_val = float(np.mean(byte_mean)) if np.mean(byte_mean) > 0 else 1.0
            norm = byte_mean / mean_val

            matrices.append(norm)
            labels.append(display_name)
            csv_rows.append({
                "model_name": display_name,
                "model_tag": row.get("model_tag"),
                "dataset_name": row.get("dataset_name", dataset_name),
                "bytes": norm,
            })

        if not matrices:
            return

        out_csv = self.output_dir / "chunk_bytewise_summary.csv"
        header = ["model_name", "model_tag", "dataset_name"] + [f"byte_{i}" for i in range(32)]
        merged: dict[tuple[str, str, str], list[float]] = {}

        # Keep previous rows so multi-pass chunk runs (e.g., per-kalman-mode) do not overwrite prior models.
        if out_csv.exists():
            with out_csv.open("r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = str(row.get("model_name", "")).strip()
                    tag = str(row.get("model_tag", "")).strip()
                    ds = str(row.get("dataset_name", "")).strip()
                    if not name:
                        continue
                    try:
                        vals = [float(row.get(f"byte_{i}", "nan")) for i in range(32)]
                    except Exception:
                        continue
                    merged[(name, tag, ds)] = vals

        for r in csv_rows:
            key = (
                str(r.get("model_name", "")).strip(),
                str(r.get("model_tag", "")).strip(),
                str(r.get("dataset_name", "")).strip(),
            )
            merged[key] = [float(v) for v in np.asarray(r["bytes"], dtype=float)]

        with out_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for (name, tag, ds), vals in sorted(merged.items(), key=lambda x: x[0][0]):
                writer.writerow([name, tag, ds, *vals])

        matrices = []
        labels = []
        for (name, _tag, _ds), vals in sorted(merged.items(), key=lambda x: x[0][0]):
            arr = np.asarray(vals, dtype=float)
            if arr.shape[0] != 32:
                continue
            matrices.append(arr)
            labels.append(name)
        if not matrices:
            return

        heat = np.vstack(matrices)
        # Display-only normalization: scale each model row independently so
        # one model's outlier does not wash out other rows in a shared colormap.
        row_min = np.nanmin(heat, axis=1, keepdims=True)
        row_max = np.nanmax(heat, axis=1, keepdims=True)
        row_span = row_max - row_min
        row_span[row_span <= 0] = 1.0
        heat_display = (heat - row_min) / row_span

        plt.figure(figsize=(18, max(3, 0.4 * len(labels))))
        plt.imshow(heat_display, cmap="Greys", aspect="auto", vmin=0.0, vmax=1.0)
        plt.colorbar(label="Row-wise normalized intensity (per model)")
        plt.yticks(ticks=range(len(labels)), labels=labels)
        plt.xticks(ticks=range(32), labels=[str(i) for i in range(32)])
        plt.xlabel("Byte index")
        plt.ylabel("Model")
        plt.title(f"Chunk Byte-wise Error Heatmap ({dataset_name}, per-model scale)")
        out_png = self.output_dir / "chunk_bytewise_heatmap.png"
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close()
