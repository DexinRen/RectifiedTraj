#!/usr/bin/env python3
"""Download and tailor the pinned Japan map for BlogWatcher."""

from __future__ import annotations

import json
from pathlib import Path

from baseline.models.valhalla_meili.manage import MODULE_DIR, configure_logging
from baseline.models.valhalla_meili.map_tools import (
    SOURCE_CATALOG,
    cut_dataset_map,
    download_file,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_MAP_ROOT = REPO_ROOT / "dataset" / "raw" / "map"
PROCESSED_MAP_ROOT = REPO_ROOT / "dataset" / "processed" / "map"
MAP_DIR = PROCESSED_MAP_ROOT / "BlogWatcher"
BOUNDARY_STATE = MAP_DIR / "boundary_state.json"
BLOGWATCHER_BBOX = {
    "min_lon": 136.82862854003906,
    "min_lat": 34.80961990356445,
    "max_lon": 139.8277587890625,
    "max_lat": 36.43394470214844,
}


def main() -> dict:
    """
    Purpose:
        Download the pinned Japan source and cut the hardcoded BlogWatcher map.
    Parameters:
        None.
    Return Dict:
        "error_code": int, 0 after the tailored map is created.
        "map_path": str, tailored BlogWatcher PBF path.
        "manifest_path": str, generated cutter manifest path.
    Usage:
        UTokyo runs this once before the BlogWatcher map-matching benchmark.
    TODO:
        1) Initialize map logging.
        2) Write the hardcoded box in the cutter's existing state format.
        3) Download the pinned Japan source.
        4) Cut and return the tailored BlogWatcher map.
    """

    # 1. Initialize Map Logging
    configure_logging(MAP_DIR)

    # 2. Write Hardcoded Box In Existing State Format
    boundary = {
        "parquet_processor": {
            "dataset_noisy_boundary_corners": {
                "max_lat_min_lon": [
                    BLOGWATCHER_BBOX["max_lat"],
                    BLOGWATCHER_BBOX["min_lon"],
                ],
                "max_lat_max_lon": [
                    BLOGWATCHER_BBOX["max_lat"],
                    BLOGWATCHER_BBOX["max_lon"],
                ],
                "min_lat_min_lon": [
                    BLOGWATCHER_BBOX["min_lat"],
                    BLOGWATCHER_BBOX["min_lon"],
                ],
                "min_lat_max_lon": [
                    BLOGWATCHER_BBOX["min_lat"],
                    BLOGWATCHER_BBOX["max_lon"],
                ],
            }
        }
    }
    BOUNDARY_STATE.write_text(
        json.dumps(boundary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # 3. Download Pinned Japan Source
    source = SOURCE_CATALOG["japan"]
    download_file(source["pbf_url"], RAW_MAP_ROOT / source["pbf_name"], 300.0)
    download_file(source["poly_url"], RAW_MAP_ROOT / source["poly_name"], 60.0)

    # 4. Cut And Return Tailored Map
    result = cut_dataset_map(
        dataset_name="BlogWatcher",
        state_file=BOUNDARY_STATE,
        source_name="japan",
        raw_map_root=RAW_MAP_ROOT,
        processed_map_root=PROCESSED_MAP_ROOT,
        buffer_km=1.0,
        dockerfile=MODULE_DIR / "Dockerfile.map_tools",
    )
    print(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    outcome = main()
    raise SystemExit(0 if int(outcome["error_code"]) == 0 else 1)
