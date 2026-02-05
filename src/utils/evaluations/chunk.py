import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class ChunkEvaluator:
    """
    Evaluate chunk-wise ENU denoising on preprocessed test chunks.

    Outputs:
        - CSV: chunk_evaluation_summary.csv
    """

    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = Path(output_dir)
        self.csv_path = self.output_dir / "chunk_evaluation_summary.csv"
        self.logger = logging.getLogger("ChunkEvaluator")

        header_cols = [
            "model_name",
            "model_tag",
            "dataset_name",
            "denoise_method",
            "K",
            "Q1",
            "Q2",
            "t_delta",
            "N_steps",
            "err_mean_full",
            "err_median_full",
            "err_std_full",
            "err_mean_mid",
            "err_median_mid",
            "err_std_mid",
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
            f"{row['model_name']},{row.get('model_tag', 'NA')},{row.get('dataset_name', 'NA')},"
            f"{row['denoise_method']},"
            f"{_fmt(row.get('K'), 'd')},{_fmt(row.get('Q1'), 'd')},{_fmt(row.get('Q2'), 'd')},"
            f"{_fmt(row.get('t_delta'), '.4f')},{_fmt(row.get('N_steps'), 'd')},"
            f"{_fmt(row.get('err_mean_full'), '.6f')},{_fmt(row.get('err_median_full'), '.6f')},{_fmt(row.get('err_std_full'), '.6f')},"
            f"{_fmt(row.get('err_mean_mid'), '.6f')},{_fmt(row.get('err_median_mid'), '.6f')},{_fmt(row.get('err_std_mid'), '.6f')},"
            f"{_fmt(row.get('num_tested_chunks'), 'd')},{row.get('test_timestamp')}\n"
        )
        with open(self.csv_path, "a") as f:
            f.write(csv_row)

    def save_bytewise_heatmap(self, rows: list[Dict], dataset_name: str = "chunk_test") -> None:
        if not rows:
            return

        matrices = []
        labels = []
        csv_rows = []

        for row in rows:
            byte_mean = row.get("byte_mean")
            if byte_mean is None:
                continue
            byte_mean = np.asarray(byte_mean, dtype=float)
            if byte_mean.shape[0] != 32:
                continue
            mean_val = float(np.mean(byte_mean)) if np.mean(byte_mean) > 0 else 1.0
            norm = byte_mean / mean_val

            matrices.append(norm)
            labels.append(str(row.get("model_name", "NA")))
            csv_rows.append({
                "model_name": row.get("model_name"),
                "model_tag": row.get("model_tag"),
                "dataset_name": row.get("dataset_name", dataset_name),
                "bytes": norm,
            })

        if not matrices:
            return

        out_csv = self.output_dir / "chunk_bytewise_summary.csv"
        with out_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            header = ["model_name", "model_tag", "dataset_name"] + [f"byte_{i}" for i in range(32)]
            writer.writerow(header)
            for r in csv_rows:
                writer.writerow([
                    r["model_name"],
                    r["model_tag"],
                    r["dataset_name"],
                    *list(map(float, r["bytes"])),
                ])

        heat = np.vstack(matrices)
        cmap = LinearSegmentedColormap.from_list("errmap", ["white", "red", "purple"])
        plt.figure(figsize=(18, max(3, 0.4 * len(labels))))
        plt.imshow(heat, cmap=cmap, aspect="auto")
        plt.colorbar(label="Normalized L2 error")
        plt.yticks(ticks=range(len(labels)), labels=labels)
        plt.xticks(ticks=range(32), labels=[str(i) for i in range(32)])
        plt.xlabel("Byte index")
        plt.ylabel("Model")
        plt.title(f"Chunk Byte-wise Error Heatmap ({dataset_name})")
        out_png = self.output_dir / "chunk_bytewise_heatmap.png"
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close()
