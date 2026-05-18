import csv
from pathlib import Path
from typing import Iterable


def aggregate_csv_files(
    csv_paths: Iterable[str | Path],
    output_path: str | Path,
) -> Path:
    paths = [Path(p) for p in csv_paths if Path(p).exists()]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    headers: list[str] = []
    rows: list[dict[str, str]] = []

    for path in sorted(paths):
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue
            for field in reader.fieldnames:
                if field not in headers:
                    headers.append(field)
            for row in reader:
                rows.append({str(k): str(v) for k, v in row.items() if k is not None})

    if not headers:
        output.write_text("", encoding="utf-8")
        return output

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in headers})
    return output


def write_rows_to_csv(
    rows: list[dict],
    output_path: str | Path,
    *,
    field_order: list[str] | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return output

    headers = list(field_order or [])
    for row in rows:
        for key in row.keys():
            key_str = str(key)
            if key_str not in headers:
                headers.append(key_str)

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in headers})
    return output


def aggregate_csv_folder(
    folder: str | Path,
    output_path: str | Path,
    *,
    skip_names: Iterable[str] | None = None,
) -> Path:
    root = Path(folder)
    skip = {str(x) for x in (skip_names or [])}
    csv_paths = [
        path
        for path in sorted(root.glob("*.csv"))
        if path.name not in skip
    ]
    return aggregate_csv_files(csv_paths, output_path)
