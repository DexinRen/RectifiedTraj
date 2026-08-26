from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import requests


LOGGER = logging.getLogger(__name__)

MAP_TOOLS_IMAGE = "rectifiedtraj-osmium-tool:debian12"
DEFAULT_BUFFER_KM = 1.0
SOURCE_CATALOG = {
    "japan": {
        "snapshot": "2026-08-15",
        "pbf_name": "japan-260815.osm.pbf",
        "pbf_url": "https://download.geofabrik.de/asia/japan-260815.osm.pbf",
        "md5_url": "https://download.geofabrik.de/asia/japan-260815.osm.pbf.md5",
        "poly_name": "japan.poly",
        "poly_url": "https://download.geofabrik.de/asia/japan.poly",
    },
    "georgia": {
        "snapshot": "2026-08-15",
        "pbf_name": "georgia-260815.osm.pbf",
        "pbf_url": "https://download.geofabrik.de/north-america/us/georgia-260815.osm.pbf",
        "md5_url": "https://download.geofabrik.de/north-america/us/georgia-260815.osm.pbf.md5",
        "poly_name": "georgia.poly",
        "poly_url": "https://download.geofabrik.de/north-america/us/georgia.poly",
    },
}


def hash_file(path: str | Path, algorithm: str) -> dict:
    """
    Purpose:
        Compute a streaming checksum without loading a large map into memory.
    Parameters:
        path (str | Path), existing regular file.
        algorithm (str), hashlib-supported algorithm name.
    Return Dict:
        "error_code": int, 0 after hashing.
        "algorithm": str.
        "digest": str, lowercase hexadecimal checksum.
        "size_bytes": int.
    Usage:
        Map downloads and manifests record reproducible source identities.
    TODO:
        1) Validate the file and algorithm.
        2) Stream fixed-size blocks through hashlib.
        3) Return checksum and size.
    """

    # 1. Validate File And Algorithm
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Checksum input is missing: {file_path}")
    if str(algorithm) not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported checksum algorithm: {algorithm}")
    digest = hashlib.new(str(algorithm))

    # 2. Stream Fixed-Size Blocks
    with file_path.open("rb") as file_obj:
        while True:
            block = file_obj.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)

    # 3. Return Checksum And Size
    return {
        "error_code": 0,
        "algorithm": str(algorithm),
        "digest": digest.hexdigest().lower(),
        "size_bytes": int(file_path.stat().st_size),
    }


def download_text(url: str, timeout_sec: float) -> dict:
    """
    Purpose:
        Download a small required metadata file as UTF-8 text.
    Parameters:
        url (str), explicit HTTPS source URL.
        timeout_sec (float), positive HTTP timeout.
    Return Dict:
        "error_code": int, 0 on HTTP success.
        "text": str, response body.
        "http_status": int.
    Usage:
        Map download obtains Geofabrik checksum files.
    TODO:
        1) Validate URL and timeout.
        2) Perform the bounded request.
        3) Return decoded text.
    """

    # 1. Validate URL And Timeout
    if not str(url).startswith("https://"):
        raise ValueError("Map metadata URL must use HTTPS.")
    if float(timeout_sec) <= 0.0:
        raise ValueError("Download timeout must be positive.")

    # 2. Perform Bounded Request
    response = requests.get(str(url), timeout=float(timeout_sec))
    response.raise_for_status()

    # 3. Return Decoded Text
    return {
        "error_code": 0,
        "text": str(response.text),
        "http_status": int(response.status_code),
    }


def download_file(url: str, output_path: str | Path, timeout_sec: float) -> dict:
    """
    Purpose:
        Download one map artifact with resume support and atomic publication.
    Parameters:
        url (str), explicit HTTPS source URL.
        output_path (str | Path), final destination path.
        timeout_sec (float), positive connect/read timeout.
    Return Dict:
        "error_code": int, 0 after final file publication.
        "path": str.
        "size_bytes": int.
        "resumed_from_bytes": int.
    Usage:
        download_map_source downloads PBF and polygon files.
    TODO:
        1) Validate destination and available disk space.
        2) Resume from a private partial file when present.
        3) Stream response blocks without buffering the map.
        4) Atomically replace the final destination.
    """

    # 1. Validate Destination And Available Disk Space
    if not str(url).startswith("https://"):
        raise ValueError("Map download URL must use HTTPS.")
    if float(timeout_sec) <= 0.0:
        raise ValueError("Download timeout must be positive.")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return {
            "error_code": 0,
            "path": str(destination),
            "size_bytes": int(destination.stat().st_size),
            "resumed_from_bytes": int(destination.stat().st_size),
        }
    free_bytes = int(shutil.disk_usage(destination.parent).free)
    if free_bytes < 10 * 1024**3:
        raise RuntimeError(
            f"Map download requires at least 10 GiB free; found {free_bytes / 1024**3:.2f} GiB."
        )

    # 2. Resume From Private Partial File
    partial = destination.with_suffix(destination.suffix + ".part")
    resumed_from = int(partial.stat().st_size) if partial.exists() else 0
    headers = {"Range": f"bytes={resumed_from}-"} if resumed_from > 0 else {}
    response = requests.get(
        str(url),
        headers=headers,
        stream=True,
        timeout=(30.0, float(timeout_sec)),
    )
    response.raise_for_status()
    if resumed_from > 0 and int(response.status_code) != 206:
        raise RuntimeError("Map server did not honor the requested resume range.")

    # 3. Stream Response Blocks
    mode = "ab" if resumed_from > 0 else "wb"
    with partial.open(mode) as file_obj:
        for block in response.iter_content(chunk_size=8 * 1024 * 1024):
            if block:
                file_obj.write(block)

    # 4. Atomically Publish Destination
    partial.replace(destination)
    return {
        "error_code": 0,
        "path": str(destination),
        "size_bytes": int(destination.stat().st_size),
        "resumed_from_bytes": resumed_from,
    }


def parse_expected_md5(text: str) -> dict:
    """
    Purpose:
        Parse a Geofabrik MD5 sidecar body.
    Parameters:
        text (str), checksum sidecar contents.
    Return Dict:
        "error_code": int, 0 for a valid checksum.
        "md5": str, 32-character lowercase checksum.
    Usage:
        download_map_source verifies downloaded PBF files.
    TODO:
        1) Select the first whitespace-delimited token.
        2) Validate hexadecimal MD5 syntax.
        3) Return normalized checksum.
    """

    # 1. Select First Token
    tokens = str(text).strip().split()
    if not tokens:
        raise ValueError("Geofabrik MD5 sidecar is empty.")
    checksum = tokens[0].strip().lower()

    # 2. Validate MD5 Syntax
    if len(checksum) != 32 or any(char not in "0123456789abcdef" for char in checksum):
        raise ValueError(f"Invalid Geofabrik MD5 checksum: {checksum!r}")

    # 3. Return Normalized Checksum
    return {"error_code": 0, "md5": checksum}


def download_map_source(source_name: str, raw_map_root: str | Path) -> dict:
    """
    Purpose:
        Download and verify one fixed Geofabrik source snapshot and boundary.
    Parameters:
        source_name (str), one of SOURCE_CATALOG keys.
        raw_map_root (str | Path), required raw map storage directory.
    Return Dict:
        "error_code": int, 0 after checksum verification.
        "source": str.
        "pbf_path": str.
        "poly_path": str.
        "md5": str.
        "snapshot": str.
    Usage:
        The management CLI downloads Japan and Georgia before cutting.
    TODO:
        1) Resolve the fixed source catalog entry.
        2) Download PBF, MD5, and polygon files.
        3) Verify PBF checksum.
        4) Return reproducibility paths and identifiers.
    """

    # 1. Resolve Fixed Source Entry
    source_key = str(source_name).strip().lower()
    if source_key not in SOURCE_CATALOG:
        raise ValueError(f"Unsupported map source {source_name!r}; expected japan or georgia.")
    source = SOURCE_CATALOG[source_key]
    root = Path(raw_map_root)
    root.mkdir(parents=True, exist_ok=True)
    pbf_path = root / source["pbf_name"]
    poly_path = root / source["poly_name"]

    # 2. Download PBF, MD5, And Polygon Files
    expected_packet = download_text(source["md5_url"], timeout_sec=60.0)
    expected_md5 = parse_expected_md5(expected_packet["text"])["md5"]
    download_file(source["pbf_url"], pbf_path, timeout_sec=300.0)
    download_file(source["poly_url"], poly_path, timeout_sec=60.0)

    # 3. Verify PBF Checksum
    actual_md5 = hash_file(pbf_path, "md5")["digest"]
    if actual_md5 != expected_md5:
        raise RuntimeError(
            f"Geofabrik checksum mismatch for {pbf_path}: expected {expected_md5}, got {actual_md5}."
        )

    # 4. Return Reproducibility Packet
    return {
        "error_code": 0,
        "source": source_key,
        "pbf_path": str(pbf_path),
        "poly_path": str(poly_path),
        "md5": actual_md5,
        "poly_sha256": hash_file(poly_path, "sha256")["digest"],
        "snapshot": source["snapshot"],
    }


def load_processor_bounds(state_file: str | Path) -> dict:
    """
    Purpose:
        Load the canonical dataset bounds produced by parquet_processor.
    Parameters:
        state_file (str | Path), processor-generated state JSON only.
    Return Dict:
        "error_code": int, 0 for valid bounds.
        "bbox": dict, min_lon, min_lat, max_lon, max_lat.
        "metadata_sha256": str.
    Usage:
        cut_dataset_map obtains its only dataset-coordinate input here.
    TODO:
        1) Load the processor state file.
        2) Require the canonical bounds field without fallbacks.
        3) Validate all four named corners.
        4) Return the normalized lon/lat bounding box.
    """

    # 1. Load Processor State File
    path = Path(state_file).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Processor state file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Processor state JSON must be an object.")

    # 2. Require Canonical Bounds Field
    processor = payload.get("parquet_processor")
    if not isinstance(processor, dict):
        raise ValueError("Processor state is missing parquet_processor.")
    corners = processor.get("dataset_noisy_boundary_corners")
    if not isinstance(corners, dict):
        raise ValueError(
            "Processor state is missing parquet_processor.dataset_noisy_boundary_corners."
        )

    # 3. Validate Four Named Corners
    required = {
        "max_lat_min_lon",
        "max_lat_max_lon",
        "min_lat_min_lon",
        "min_lat_max_lon",
    }
    if set(corners.keys()) != required:
        raise ValueError("Processor boundary corners must contain exactly the four canonical keys.")
    coordinates: dict[str, tuple[float, float]] = {}
    for key in sorted(required):
        value = corners[key]
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"Processor corner {key} must be [lat, lon].")
        lat = float(value[0])
        lon = float(value[1])
        if not math.isfinite(lat) or not math.isfinite(lon):
            raise ValueError(f"Processor corner {key} must be finite.")
        coordinates[key] = (lat, lon)
    min_lat = coordinates["min_lat_min_lon"][0]
    max_lat = coordinates["max_lat_min_lon"][0]
    min_lon = coordinates["min_lat_min_lon"][1]
    max_lon = coordinates["min_lat_max_lon"][1]
    if not (-90.0 <= min_lat < max_lat <= 90.0):
        raise ValueError("Processor latitude bounds are invalid.")
    if not (-180.0 <= min_lon < max_lon <= 180.0):
        raise ValueError("Processor longitude bounds are invalid.")

    # 4. Return Normalized BBox
    metadata_hash = hash_file(path, "sha256")["digest"]
    return {
        "error_code": 0,
        "bbox": {
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat,
        },
        "metadata_sha256": metadata_hash,
    }


def load_poly_envelope(poly_file: str | Path) -> dict:
    """
    Purpose:
        Read the coordinate envelope of a Geofabrik POLY boundary file.
    Parameters:
        poly_file (str | Path), downloaded Geofabrik .poly file.
    Return Dict:
        "error_code": int, 0 for a valid polygon envelope.
        "bbox": dict, min_lon, min_lat, max_lon, max_lat.
    Usage:
        Soft buffering clamps the requested bbox to the source map boundary.
    TODO:
        1) Read coordinate lines while ignoring ring markers.
        2) Validate longitude and latitude values.
        3) Return the overall source envelope.
    """

    # 1. Read Coordinate Lines
    path = Path(poly_file)
    if not path.is_file():
        raise FileNotFoundError(f"Geofabrik polygon file is missing: {path}")
    points: list[tuple[float, float]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        tokens = raw_line.strip().split()
        if len(tokens) != 2:
            continue
        try:
            lon = float(tokens[0])
            lat = float(tokens[1])
        except ValueError:
            continue
        points.append((lon, lat))
    if not points:
        raise ValueError(f"Geofabrik polygon contains no coordinate points: {path}")

    # 2. Validate Coordinates
    if any(not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0) for lon, lat in points):
        raise ValueError(f"Geofabrik polygon contains invalid coordinates: {path}")

    # 3. Return Source Envelope
    return {
        "error_code": 0,
        "bbox": {
            "min_lon": min(point[0] for point in points),
            "min_lat": min(point[1] for point in points),
            "max_lon": max(point[0] for point in points),
            "max_lat": max(point[1] for point in points),
        },
    }


def apply_soft_buffer(dataset_bbox: dict, source_bbox: dict, buffer_km: float) -> dict:
    """
    Purpose:
        Expand dataset bounds by at most the requested distance per side.
    Parameters:
        dataset_bbox (dict), canonical lon/lat dataset bounds.
        source_bbox (dict), Geofabrik source-map envelope.
        buffer_km (float), nonnegative maximum expansion distance.
    Return Dict:
        "error_code": int, 0 for valid buffered bounds.
        "bbox": dict, source-clipped buffered bounds.
        "applied_buffer_km": dict, approximate expansion on each side.
    Usage:
        cut_dataset_map applies the approved 1 km soft boundary.
    TODO:
        1) Validate containment and buffer distance.
        2) Convert 1 km into local latitude/longitude deltas.
        3) Clamp each expanded side to the source map boundary.
        4) Report the actual per-side buffer.
    """

    # 1. Validate Containment And Buffer Distance
    distance_km = float(buffer_km)
    if distance_km < 0.0:
        raise ValueError("Map buffer distance must be nonnegative.")
    if distance_km > DEFAULT_BUFFER_KM:
        raise ValueError(
            f"Map buffer distance may not exceed the approved {DEFAULT_BUFFER_KM:.1f} km."
        )
    for key in ("min_lon", "min_lat", "max_lon", "max_lat"):
        if key not in dataset_bbox or key not in source_bbox:
            raise ValueError(f"Bounding boxes must contain {key}.")
    if (
        dataset_bbox["min_lon"] < source_bbox["min_lon"]
        or dataset_bbox["max_lon"] > source_bbox["max_lon"]
        or dataset_bbox["min_lat"] < source_bbox["min_lat"]
        or dataset_bbox["max_lat"] > source_bbox["max_lat"]
    ):
        raise ValueError("Dataset bounds are outside the selected Geofabrik source boundary.")

    # 2. Convert Kilometers To Local Angular Deltas
    mid_lat = 0.5 * (float(dataset_bbox["min_lat"]) + float(dataset_bbox["max_lat"]))
    lat_delta = distance_km / 110.574 if distance_km > 0.0 else 0.0
    lon_scale = 111.320 * math.cos(math.radians(mid_lat))
    if lon_scale <= 0.0:
        raise ValueError("Dataset latitude is too close to a pole for longitude buffering.")
    lon_delta = distance_km / lon_scale if distance_km > 0.0 else 0.0

    # 3. Clamp Expanded Sides To Source Boundary
    buffered = {
        "min_lon": max(float(source_bbox["min_lon"]), float(dataset_bbox["min_lon"]) - lon_delta),
        "min_lat": max(float(source_bbox["min_lat"]), float(dataset_bbox["min_lat"]) - lat_delta),
        "max_lon": min(float(source_bbox["max_lon"]), float(dataset_bbox["max_lon"]) + lon_delta),
        "max_lat": min(float(source_bbox["max_lat"]), float(dataset_bbox["max_lat"]) + lat_delta),
    }

    # 4. Report Actual Per-Side Buffer
    applied = {
        "west": (float(dataset_bbox["min_lon"]) - buffered["min_lon"]) * lon_scale,
        "south": (float(dataset_bbox["min_lat"]) - buffered["min_lat"]) * 110.574,
        "east": (buffered["max_lon"] - float(dataset_bbox["max_lon"])) * lon_scale,
        "north": (buffered["max_lat"] - float(dataset_bbox["max_lat"])) * 110.574,
    }
    return {"error_code": 0, "bbox": buffered, "applied_buffer_km": applied}


def build_map_tools_image(dockerfile: str | Path) -> dict:
    """
    Purpose:
        Build the stable local Osmium Docker image used by the map cutter.
    Parameters:
        dockerfile (str | Path), checked-in Dockerfile.map_tools path.
    Return Dict:
        "error_code": int, Docker build return code.
        "stdout": str.
        "stderr": str.
    Usage:
        cut_dataset_map builds the small tool image before extraction.
    TODO:
        1) Validate Dockerfile path.
        2) Run a bounded Docker build without a shell.
        3) Return captured build output.
    """

    # 1. Validate Dockerfile Path
    path = Path(dockerfile).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Map tools Dockerfile is missing: {path}")

    # 2. Run Bounded Docker Build
    completed = subprocess.run(
        ["docker", "build", "--tag", MAP_TOOLS_IMAGE, "--file", str(path), str(path.parent)],
        capture_output=True,
        text=True,
        timeout=900.0,
        check=False,
    )

    # 3. Return Build Output
    return {
        "error_code": int(completed.returncode),
        "stdout": str(completed.stdout),
        "stderr": str(completed.stderr),
    }


def run_osmium(arguments: list[str], mounts: list[dict], timeout_sec: float) -> dict:
    """
    Purpose:
        Run one memory-limited Osmium command inside Docker.
    Parameters:
        arguments (list[str]), Osmium arguments after the executable name.
        mounts (list[dict]), host_path, container_path, read_only entries.
        timeout_sec (float), positive command timeout.
    Return Dict:
        "error_code": int, process return code.
        "stdout": str.
        "stderr": str.
        "command": list[str].
    Usage:
        Map extraction and reference checking use this helper.
    TODO:
        1) Validate arguments and mount paths.
        2) Construct a non-shell Docker command with resource limits.
        3) Execute and return process output.
    """

    # 1. Validate Arguments And Mount Paths
    if not isinstance(arguments, list) or not arguments:
        raise ValueError("Osmium arguments must be a non-empty list.")
    if float(timeout_sec) <= 0.0:
        raise ValueError("Osmium timeout must be positive.")
    mount_args: list[str] = []
    for mount in mounts:
        host_path = Path(str(mount["host_path"])).resolve()
        if not host_path.exists():
            raise FileNotFoundError(f"Osmium mount path is missing: {host_path}")
        suffix = ":ro" if bool(mount["read_only"]) else ""
        mount_args.extend(["--volume", f"{host_path}:{mount['container_path']}{suffix}"])

    # 2. Construct Resource-Limited Docker Command
    command = [
        "docker",
        "run",
        "--rm",
        "--cpus",
        "1",
        "--memory",
        "8g",
        *mount_args,
        MAP_TOOLS_IMAGE,
        *[str(value) for value in arguments],
    ]

    # 3. Execute And Return Output
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=float(timeout_sec),
        check=False,
    )
    return {
        "error_code": int(completed.returncode),
        "stdout": str(completed.stdout),
        "stderr": str(completed.stderr),
        "command": command,
    }


def cut_dataset_map(
    *,
    dataset_name: str,
    state_file: str | Path,
    source_name: str,
    raw_map_root: str | Path,
    processed_map_root: str | Path,
    buffer_km: float,
    dockerfile: str | Path,
) -> dict:
    """
    Purpose:
        Cut one reference-complete map using processor metadata only.
    Parameters:
        dataset_name (str), explicit dataset identifier.
        state_file (str | Path), processor-generated state JSON.
        source_name (str), japan or georgia.
        raw_map_root (str | Path), raw Geofabrik storage root.
        processed_map_root (str | Path), tailored map storage root.
        buffer_km (float), soft maximum per-side buffer, approved value 1.0.
        dockerfile (str | Path), stable Osmium Dockerfile.
    Return Dict:
        "error_code": int, 0 after extraction and reference validation.
        "map_path": str.
        "manifest_path": str.
        "manifest": dict.
    Usage:
        The management CLI calls this after source map download.
    TODO:
        1) Load only processor bounds and source polygon metadata.
        2) Apply the source-clipped soft buffer.
        3) Build the stable Osmium image.
        4) Extract complete ways and validate references.
        5) Save a reproducibility manifest.
    """

    # 1. Load Processor Bounds And Source Metadata
    dataset_token = str(dataset_name).strip()
    if not dataset_token:
        raise ValueError("dataset_name must be explicit.")
    source_key = str(source_name).strip().lower()
    if source_key not in SOURCE_CATALOG:
        raise ValueError("source_name must be japan or georgia.")
    raw_root = Path(raw_map_root).resolve()
    source = SOURCE_CATALOG[source_key]
    pbf_path = raw_root / source["pbf_name"]
    poly_path = raw_root / source["poly_name"]
    if not pbf_path.is_file() or not poly_path.is_file():
        raise FileNotFoundError(
            f"Raw map source is incomplete. Expected {pbf_path} and {poly_path}."
        )
    dataset_bounds = load_processor_bounds(state_file)
    source_bounds = load_poly_envelope(poly_path)

    # 2. Apply Source-Clipped Soft Buffer
    buffered = apply_soft_buffer(
        dataset_bounds["bbox"],
        source_bounds["bbox"],
        float(buffer_km),
    )

    # 3. Build Stable Osmium Image
    image_build = build_map_tools_image(dockerfile)
    if int(image_build["error_code"]) != 0:
        raise RuntimeError(f"Failed to build Osmium Docker image: {image_build['stderr']}")

    # 4. Extract Complete Ways And Validate References
    output_dir = (Path(processed_map_root) / dataset_token).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset_token}.osm.pbf"
    if output_path.exists():
        raise FileExistsError(
            f"Tailored map already exists; remove it explicitly before rebuilding: {output_path}"
        )
    bbox = buffered["bbox"]
    bbox_token = ",".join(
        str(bbox[key]) for key in ("min_lon", "min_lat", "max_lon", "max_lat")
    )
    extract_packet = run_osmium(
        [
            "extract",
            "--bbox",
            bbox_token,
            "--strategy",
            "complete_ways",
            "--set-bounds",
            "--output",
            f"/output/{output_path.name}",
            f"/input/{pbf_path.name}",
        ],
        mounts=[
            {"host_path": pbf_path.parent, "container_path": "/input", "read_only": True},
            {"host_path": output_dir, "container_path": "/output", "read_only": False},
        ],
        timeout_sec=7200.0,
    )
    if int(extract_packet["error_code"]) != 0:
        raise RuntimeError(f"Osmium extract failed: {extract_packet['stderr']}")
    refs_packet = run_osmium(
        ["check-refs", f"/output/{output_path.name}"],
        mounts=[
            {"host_path": output_dir, "container_path": "/output", "read_only": True},
        ],
        timeout_sec=1800.0,
    )
    if int(refs_packet["error_code"]) != 0:
        raise RuntimeError(f"Osmium reference validation failed: {refs_packet['stderr']}")

    # 5. Save Reproducibility Manifest
    manifest = {
        "dataset_name": dataset_token,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state_file": str(Path(state_file).resolve()),
        "metadata_sha256": dataset_bounds["metadata_sha256"],
        "metadata_field": "parquet_processor.dataset_noisy_boundary_corners",
        "source": source_key,
        "source_snapshot": source["snapshot"],
        "source_pbf": str(pbf_path),
        "source_pbf_md5": hash_file(pbf_path, "md5")["digest"],
        "source_poly_sha256": hash_file(poly_path, "sha256")["digest"],
        "dataset_bbox": dataset_bounds["bbox"],
        "source_bbox": source_bounds["bbox"],
        "requested_buffer_km": float(buffer_km),
        "applied_buffer_km": buffered["applied_buffer_km"],
        "cut_bbox": buffered["bbox"],
        "cutter": {
            "tool": "osmium extract",
            "strategy": "complete_ways",
            "docker_image": MAP_TOOLS_IMAGE,
            "docker_memory_limit": "8g",
            "docker_cpus": 1,
        },
        "output_pbf": str(output_path),
        "output_size_bytes": int(output_path.stat().st_size),
        "output_sha256": hash_file(output_path, "sha256")["digest"],
    }
    manifest_path = output_dir / "map_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "error_code": 0,
        "map_path": str(output_path),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
    }


def ensure_dataset_map(config: dict) -> dict:
    """
    Purpose:
        Ensure one benchmark dataset has a tailored PBF before evaluation jobs start.
    Parameters:
        config (dict), explicit Valhalla config containing map and metadata paths.
    Return Dict:
        "error_code": int, 0 when the tailored PBF exists.
        "status": str, "existing" or "created".
        "map_path": str, exact dataset PBF path.
        "manifest_path": str, expected cutter manifest path.
    Usage:
        run_benchmarks task construction calls this before launching child jobs.
    TODO:
        1) Resolve the exact dataset map artifact.
        2) Return without reading metadata when the artifact exists.
        3) Cut from the configured raw Geofabrik map when it is absent.
        4) Verify that the cutter published the expected artifact.
    """

    # 1. Resolve Exact Dataset Map Artifact
    if not isinstance(config, dict):
        raise TypeError("Valhalla map preparation config must be a dict.")
    required = {
        "map_id",
        "state_file",
        "source",
        "raw_map_root",
        "processed_map_root",
        "buffer_km",
        "map_tools_dockerfile",
    }
    missing = sorted(required - set(config.keys()))
    if missing:
        raise ValueError(f"Valhalla map preparation config is missing: {', '.join(missing)}")
    map_id = str(config["map_id"]).strip()
    if not map_id:
        raise ValueError("Valhalla map_id must be explicit.")
    map_dir = Path(str(config["processed_map_root"])) / map_id
    map_path = map_dir / f"{map_id}.osm.pbf"
    manifest_path = map_dir / "map_manifest.json"

    # 2. Accept Existing Tailored Artifact Without Reading Dataset Metadata
    if map_path.is_file():
        return {
            "error_code": 0,
            "status": "existing",
            "map_path": str(map_path.resolve()),
            "manifest_path": str(manifest_path.resolve()),
        }

    # 3. Cut Missing Artifact From Raw Geofabrik Source
    LOGGER.info(
        "Tailored map missing; cutting from processor metadata | map_id=%s source=%s",
        map_id,
        config["source"],
    )
    cut_packet = cut_dataset_map(
        dataset_name=map_id,
        state_file=config["state_file"],
        source_name=config["source"],
        raw_map_root=config["raw_map_root"],
        processed_map_root=config["processed_map_root"],
        buffer_km=float(config["buffer_km"]),
        dockerfile=config["map_tools_dockerfile"],
    )

    # 4. Verify Expected Publication
    if Path(str(cut_packet["map_path"])).resolve() != map_path.resolve():
        raise RuntimeError("Map cutter published an unexpected dataset map path.")
    if not map_path.is_file():
        raise RuntimeError(f"Map cutter did not publish the expected PBF: {map_path}")
    return {
        "error_code": 0,
        "status": "created",
        "map_path": str(map_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
    }


__all__ = [
    "DEFAULT_BUFFER_KM",
    "MAP_TOOLS_IMAGE",
    "SOURCE_CATALOG",
    "apply_soft_buffer",
    "build_map_tools_image",
    "cut_dataset_map",
    "download_map_source",
    "ensure_dataset_map",
    "hash_file",
    "load_poly_envelope",
    "load_processor_bounds",
    "parse_expected_md5",
    "run_osmium",
]
