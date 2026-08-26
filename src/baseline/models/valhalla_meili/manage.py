from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from baseline.models.valhalla_meili.map_tools import (  # noqa: E402
    DEFAULT_BUFFER_KM,
    cut_dataset_map,
    download_map_source,
)
from baseline.models.valhalla_meili.model import validate_valhalla_config  # noqa: E402
from baseline.models.valhalla_meili.service import (  # noqa: E402
    build_valhalla_tiles,
    ensure_valhalla_service,
    query_valhalla_status,
    stop_valhalla_service,
)


LOGGER = logging.getLogger("valhalla_meili.manage")


def configure_logging(log_dir: str | Path) -> dict:
    """
    Purpose:
        Configure console and append-only file logging for map management.
    Parameters:
        log_dir (str | Path), project output directory for debug_info.log.
    Return Dict:
        "error_code": int, 0 after logger configuration.
        "log_path": str.
    Usage:
        main calls this before performing map or Docker operations.
    TODO:
        1) Create the explicit log directory.
        2) Clear duplicate handlers.
        3) Attach console and append-only file handlers.
    """

    # 1. Create Explicit Log Directory
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / "debug_info.log"

    # 2. Clear Duplicate Handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)

    # 3. Attach Console And File Handlers
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    return {"error_code": 0, "log_path": str(log_path)}


def build_management_config(dataset: str, port: int, auto_start: bool) -> dict:
    """
    Purpose:
        Build the complete explicit configuration shared by management commands.
    Parameters:
        dataset (str), dataset/map identifier.
        port (int), localhost service port, range 1..65535.
        auto_start (bool), whether ensure may invoke Docker Compose up.
    Return Dict:
        "error_code": int, 0 for valid configuration.
        "config": dict, validated Valhalla configuration.
    Usage:
        build, up, status, and down commands use the same configuration contract.
    TODO:
        1) Validate dataset and port.
        2) Populate all approved stable settings explicitly.
        3) Validate and return the configuration.
    """

    # 1. Validate Dataset And Port
    dataset_token = str(dataset).strip()
    port_value = int(port)
    if not dataset_token:
        raise ValueError("dataset must be explicit.")
    if port_value < 1 or port_value > 65535:
        raise ValueError("port must be between 1 and 65535.")

    # 2. Populate Approved Stable Settings
    raw_config = {
        "base_url": f"http://127.0.0.1:{port_value}",
        "costing": "auto",
        "shape_match": "map_snap",
        "window_points": 500,
        "overlap_points": 50,
        "timeout_sec": 60.0,
        "auto_start": bool(auto_start),
        "map_id": dataset_token,
        "compose_file": str(MODULE_DIR / "docker-compose.yml"),
        "processed_map_root": str(REPO_ROOT / "dataset" / "processed" / "map"),
        "port": port_value,
        "startup_timeout_sec": 120.0,
        "build_timeout_sec": 7200.0,
    }

    # 3. Validate And Return Configuration
    return validate_valhalla_config(raw_config)


def build_parser() -> dict:
    """
    Purpose:
        Build the Valhalla Meili map/service management CLI parser.
    Parameters:
        None.
    Return Dict:
        "error_code": int, 0 after parser construction.
        "parser": argparse.ArgumentParser.
    Usage:
        main parses command-line arguments with this parser.
    TODO:
        1) Create top-level parser and required subcommands.
        2) Add explicit map download/cut arguments.
        3) Add explicit Docker lifecycle arguments.
    """

    # 1. Create Parser And Subcommands
    parser = argparse.ArgumentParser(
        description="Manage Docker-backed Valhalla Meili maps and service."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 2. Add Map Download And Cut Arguments
    download_parser = subparsers.add_parser("download", help="Download fixed Geofabrik maps.")
    download_parser.add_argument(
        "--source",
        action="append",
        choices=["japan", "georgia"],
        required=True,
        help="Repeat to download both approved sources.",
    )
    download_parser.add_argument(
        "--raw-map-root",
        default=str(REPO_ROOT / "dataset" / "raw" / "map"),
    )

    cut_parser = subparsers.add_parser("cut", help="Cut a dataset map from processor metadata.")
    cut_parser.add_argument("--dataset", required=True)
    cut_parser.add_argument("--state-file", required=True)
    cut_parser.add_argument("--source", choices=["japan", "georgia"], required=True)
    cut_parser.add_argument("--buffer-km", type=float, default=DEFAULT_BUFFER_KM)
    cut_parser.add_argument(
        "--raw-map-root",
        default=str(REPO_ROOT / "dataset" / "raw" / "map"),
    )
    cut_parser.add_argument(
        "--processed-map-root",
        default=str(REPO_ROOT / "dataset" / "processed" / "map"),
    )

    # 3. Add Docker Lifecycle Arguments
    for command in ("build", "up", "status", "down"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--dataset", required=True)
        command_parser.add_argument("--port", type=int, default=8002)
    return {"error_code": 0, "parser": parser}


def execute_command(args: argparse.Namespace) -> dict:
    """
    Purpose:
        Dispatch one validated management CLI command.
    Parameters:
        args (argparse.Namespace), parsed CLI arguments.
    Return Dict:
        "error_code": int, command outcome code.
        Additional keys depend on the selected command.
    Usage:
        main delegates all external work to this dispatcher.
    TODO:
        1) Dispatch source download commands.
        2) Dispatch metadata-only map cutting.
        3) Dispatch Docker build/start/status/stop commands.
    """

    # 1. Dispatch Source Downloads
    if args.command == "download":
        results = [
            download_map_source(source_name, args.raw_map_root)
            for source_name in args.source
        ]
        return {"error_code": 0, "downloads": results}

    # 2. Dispatch Metadata-Only Map Cutting
    if args.command == "cut":
        return cut_dataset_map(
            dataset_name=args.dataset,
            state_file=args.state_file,
            source_name=args.source,
            raw_map_root=args.raw_map_root,
            processed_map_root=args.processed_map_root,
            buffer_km=float(args.buffer_km),
            dockerfile=MODULE_DIR / "Dockerfile.map_tools",
        )

    # 3. Dispatch Docker Lifecycle Commands
    config = build_management_config(
        args.dataset,
        args.port,
        auto_start=args.command == "up",
    )["config"]
    if args.command == "build":
        return build_valhalla_tiles(config)
    if args.command == "up":
        return ensure_valhalla_service(config)
    if args.command == "status":
        return query_valhalla_status(config["base_url"], timeout_sec=5.0)
    if args.command == "down":
        return stop_valhalla_service(config)
    raise ValueError(f"Unsupported management command: {args.command}")


def main(argv: list[str] | None = None) -> dict:
    """
    Purpose:
        Parse and execute one Valhalla Meili management command.
    Parameters:
        argv (list[str] | None), optional explicit CLI argument list.
    Return Dict:
        "error_code": int, command outcome code.
        "result": dict, complete command result packet.
    Usage:
        Module CLI entry point and tests call this function.
    TODO:
        1) Parse arguments.
        2) Configure append-only project logging.
        3) Execute the selected command.
        4) Print and return the result packet.
    """

    # 1. Parse Arguments
    parser = build_parser()["parser"]
    args = parser.parse_args(argv)

    # 2. Configure Append-Only Project Logging
    dataset = str(getattr(args, "dataset", "map_sources"))
    log_dir = REPO_ROOT / "dataset" / "processed" / "map" / dataset
    configure_logging(log_dir)

    # 3. Execute Selected Command
    result = execute_command(args)

    # 4. Print And Return Result Packet
    print(json.dumps(result, indent=2, default=str))
    return {"error_code": int(result["error_code"]), "result": result}


if __name__ == "__main__":
    outcome = main()
    raise SystemExit(0 if int(outcome["error_code"]) == 0 else 1)
