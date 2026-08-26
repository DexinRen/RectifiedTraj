from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path

import requests


LOGGER = logging.getLogger(__name__)


def build_compose_runtime(config: dict, *, rebuild: bool) -> dict:
    """
    Purpose:
        Build an explicit Docker Compose command environment for one map.
    Parameters:
        config (dict), validated Valhalla baseline configuration.
        rebuild (bool), true only for the explicit tile preparation command.
    Return Dict:
        "error_code": int, 0 for valid runtime.
        "command_prefix": list[str], Docker Compose command prefix.
        "environment": dict[str,str], subprocess environment.
        "map_dir": Path, dataset-specific processed map directory.
    Usage:
        Service start, build, and stop helpers share this runtime packet.
    TODO:
        1) Resolve and validate the Compose file.
        2) Resolve the dataset map directory.
        3) Create a deterministic project name and environment.
    """

    # 1. Resolve And Validate Compose File
    compose_file = Path(str(config["compose_file"])).resolve()
    if not compose_file.is_file():
        raise FileNotFoundError(f"Valhalla Compose file is missing: {compose_file}")

    # 2. Resolve Dataset Map Directory
    map_id = str(config["map_id"]).strip()
    map_dir = (Path(str(config["processed_map_root"])) / map_id).resolve()
    if not map_dir.is_dir():
        raise FileNotFoundError(f"Processed map directory is missing: {map_dir}")

    # 3. Create Deterministic Project Environment
    project_token = "".join(char.lower() if char.isalnum() else "-" for char in map_id).strip("-")
    if not project_token:
        raise ValueError("map_id does not contain a valid Docker project token.")
    environment = dict(os.environ)
    environment.update(
        {
            "VALHALLA_DATA_DIR": str(map_dir),
            "VALHALLA_PORT": str(config["port"]),
            "VALHALLA_USE_TILES_IGNORE_PBF": "False" if rebuild else "True",
            "VALHALLA_FORCE_REBUILD": "True" if rebuild else "False",
            "VALHALLA_SERVER_THREADS": "1" if rebuild else "2",
        }
    )
    command_prefix = [
        "docker",
        "compose",
        "-p",
        f"rectifiedtraj-valhalla-{project_token}",
        "-f",
        str(compose_file),
    ]
    return {
        "error_code": 0,
        "command_prefix": command_prefix,
        "environment": environment,
        "map_dir": map_dir,
    }


def run_compose_command(runtime: dict, arguments: list[str], timeout_sec: float) -> dict:
    """
    Purpose:
        Execute one bounded Docker Compose lifecycle command.
    Parameters:
        runtime (dict), packet returned by build_compose_runtime.
        arguments (list[str]), Compose subcommand arguments.
        timeout_sec (float), positive process timeout in seconds.
    Return Dict:
        "error_code": int, subprocess return code.
        "stdout": str.
        "stderr": str.
        "command": list[str].
    Usage:
        Tile preparation and service lifecycle functions call this helper.
    TODO:
        1) Validate the timeout and arguments.
        2) Execute Docker without a shell.
        3) Return captured process output.
    """

    # 1. Validate Timeout And Arguments
    if float(timeout_sec) <= 0.0:
        raise ValueError("Docker command timeout must be positive.")
    if not isinstance(arguments, list) or not arguments:
        raise ValueError("Docker Compose arguments must be a non-empty list.")
    command = list(runtime["command_prefix"]) + [str(item) for item in arguments]

    # 2. Execute Docker Without A Shell
    try:
        completed = subprocess.run(
            command,
            cwd=str(Path.cwd()),
            env=runtime["environment"],
            capture_output=True,
            text=True,
            timeout=float(timeout_sec),
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Docker CLI is not installed or not available on PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        return {
            "error_code": -1,
            "stdout": str(exc.stdout or ""),
            "stderr": f"Docker command timed out after {timeout_sec} seconds.",
            "command": command,
        }

    # 3. Return Process Output
    return {
        "error_code": int(completed.returncode),
        "stdout": str(completed.stdout),
        "stderr": str(completed.stderr),
        "command": command,
    }


def query_valhalla_status(base_url: str, timeout_sec: float) -> dict:
    """
    Purpose:
        Query the local Docker Valhalla status endpoint once.
    Parameters:
        base_url (str), local service origin.
        timeout_sec (float), positive HTTP timeout.
    Return Dict:
        "error_code": int, 0 when HTTP 200 returns valid JSON.
        "http_status": int | None.
        "status": dict | None.
        "message": str.
    Usage:
        Service initialization and readiness polling call this helper.
    TODO:
        1) Send a bounded local HTTP request.
        2) Validate status code and JSON body.
        3) Return a structured readiness packet.
    """

    # 1. Send Bounded Local HTTP Request
    if float(timeout_sec) <= 0.0:
        raise ValueError("Valhalla status timeout must be positive.")
    url = str(base_url).rstrip("/") + "/status"
    try:
        response = requests.get(url, timeout=float(timeout_sec))
    except requests.RequestException as exc:
        return {
            "error_code": -1,
            "http_status": None,
            "status": None,
            "message": f"Valhalla status request failed: {exc.__class__.__name__}",
        }

    # 2. Validate Status Code And JSON Body
    if int(response.status_code) != 200:
        return {
            "error_code": int(response.status_code),
            "http_status": int(response.status_code),
            "status": None,
            "message": f"Valhalla status returned HTTP {response.status_code}.",
        }
    try:
        status_payload = response.json()
    except requests.JSONDecodeError:
        return {
            "error_code": -2,
            "http_status": int(response.status_code),
            "status": None,
            "message": "Valhalla status returned invalid JSON.",
        }
    if not isinstance(status_payload, dict):
        raise TypeError("Valhalla status JSON must be an object.")

    # 3. Return Readiness Packet
    return {
        "error_code": 0,
        "http_status": int(response.status_code),
        "status": status_payload,
        "message": "Valhalla service is ready.",
    }


def wait_for_valhalla(base_url: str, startup_timeout_sec: float) -> dict:
    """
    Purpose:
        Poll the local Valhalla status endpoint until ready or timed out.
    Parameters:
        base_url (str), local service origin.
        startup_timeout_sec (float), total positive readiness deadline.
    Return Dict:
        "error_code": int, 0 when ready, -1 on timeout.
        "status": dict | None.
        "message": str.
    Usage:
        Docker service start and explicit tile build use this readiness guard.
    TODO:
        1) Establish the readiness deadline.
        2) Poll with short bounded requests.
        3) Return the final status or timeout packet.
    """

    # 1. Establish Readiness Deadline
    if float(startup_timeout_sec) <= 0.0:
        raise ValueError("startup_timeout_sec must be positive.")
    deadline = time.monotonic() + float(startup_timeout_sec)
    last_status: dict | None = None

    # 2. Poll With Short Bounded Requests
    while time.monotonic() < deadline:
        last_status = query_valhalla_status(base_url, timeout_sec=2.0)
        if int(last_status["error_code"]) == 0:
            return last_status
        time.sleep(0.5)

    # 3. Return Timeout Packet
    last_message = last_status["message"] if isinstance(last_status, dict) else "no response"
    return {
        "error_code": -1,
        "status": None,
        "message": f"Valhalla did not become ready: {last_message}",
    }


def query_valhalla_container_pid(config: dict) -> dict:
    """
    Purpose:
        Resolve the host PID of the active dataset-specific Valhalla container.
    Parameters:
        config (dict), validated Valhalla configuration.
    Return Dict:
        "error_code": int, 0 for one positive host PID.
        "container_id": str, Docker container identifier.
        "container_pid": int, host PID used as an RSS process-tree root.
        "message": str.
    Usage:
        Service initialization exposes Docker memory ownership to benchmark monitors.
    TODO:
        1) Resolve the Compose service container identifier.
        2) Inspect its host PID without a shell.
        3) Validate and return the PID.
    """

    # 1. Resolve Compose Container Identifier
    runtime = build_compose_runtime(config, rebuild=False)
    compose_packet = run_compose_command(
        runtime,
        ["ps", "-q", "valhalla"],
        timeout_sec=15.0,
    )
    container_id = str(compose_packet["stdout"]).strip()
    if int(compose_packet["error_code"]) != 0 or not container_id:
        return {
            "error_code": 1,
            "container_id": container_id,
            "container_pid": None,
            "message": "Valhalla Compose container ID is unavailable.",
        }

    # 2. Inspect Host PID Without Shell
    completed = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Pid}}", container_id],
        capture_output=True,
        text=True,
        timeout=15.0,
        check=False,
    )
    if int(completed.returncode) != 0:
        return {
            "error_code": 2,
            "container_id": container_id,
            "container_pid": None,
            "message": str(completed.stderr),
        }

    # 3. Validate And Return PID
    pid_token = str(completed.stdout).strip()
    if not pid_token.isdigit() or int(pid_token) <= 0:
        return {
            "error_code": 3,
            "container_id": container_id,
            "container_pid": None,
            "message": f"Docker returned an invalid Valhalla PID: {pid_token!r}",
        }
    return {
        "error_code": 0,
        "container_id": container_id,
        "container_pid": int(pid_token),
        "message": "Valhalla container PID resolved.",
    }


def ensure_valhalla_service(config: dict) -> dict:
    """
    Purpose:
        Create one fresh Docker service for a single benchmark testing item.
    Parameters:
        config (dict), validated Valhalla baseline configuration.
    Return Dict:
        "error_code": int, 0 when service is ready.
        "message": str.
        "status": dict | None.
        "docker": dict | None, Compose command result.
    Usage:
        ValhallaMeiliBaselineModel.initialize calls this outside prediction time.
    TODO:
        1) Require explicit automatic lifecycle ownership.
        2) Build tiles when the tailored PBF is newer or tiles are missing.
        3) Remove any old Compose instance and reject a foreign port owner.
        4) Start a fresh Docker service and wait for readiness.
        5) Resolve the container host PID for RSS sampling.
    """

    # 1. Require Automatic Lifecycle Ownership
    if not bool(config["auto_start"]):
        return {
            "error_code": 3,
            "message": "Per-item Valhalla lifecycle requires auto_start=true.",
            "status": None,
            "docker": None,
            "container_pid": None,
        }

    # 2. Build Missing Or Stale Tiles
    runtime = build_compose_runtime(config, rebuild=False)
    tile_archive = runtime["map_dir"] / "valhalla_tiles.tar"
    pbf_files = sorted(runtime["map_dir"].glob("*.osm.pbf"))
    if len(pbf_files) != 1:
        raise RuntimeError(
            f"Expected exactly one tailored PBF under {runtime['map_dir']}, found {len(pbf_files)}."
        )
    tiles_stale = (
        not tile_archive.is_file()
        or float(tile_archive.stat().st_mtime) < float(pbf_files[0].stat().st_mtime)
    )
    if tiles_stale:
        build_packet = build_valhalla_tiles(config)
        if int(build_packet["error_code"]) != 0:
            stop_valhalla_service(config)
            return {
                "error_code": 1,
                "message": str(build_packet["message"]),
                "status": None,
                "docker": build_packet["docker"],
                "container_pid": None,
            }
        pid_packet = query_valhalla_container_pid(config)
        if int(pid_packet["error_code"]) != 0:
            stop_valhalla_service(config)
            return {
                "error_code": 2,
                "message": str(pid_packet["message"]),
                "status": build_packet["status"],
                "docker": build_packet["docker"],
                "container_pid": None,
            }
        return {
            "error_code": 0,
            "message": "Valhalla tiles built and fresh service ready.",
            "status": build_packet["status"],
            "docker": build_packet["docker"],
            "container_pid": int(pid_packet["container_pid"]),
        }

    # 3. Remove Old Project And Reject Foreign Port Owner
    down_packet = run_compose_command(runtime, ["down"], timeout_sec=120.0)
    if int(down_packet["error_code"]) != 0:
        return {
            "error_code": 4,
            "message": str(down_packet["stderr"]),
            "status": None,
            "docker": down_packet,
            "container_pid": None,
        }
    status_packet = query_valhalla_status(config["base_url"], timeout_sec=2.0)
    if int(status_packet["error_code"]) == 0:
        return {
            "error_code": 5,
            "message": f"Port {config['port']} is already serving a foreign Valhalla instance.",
            "status": status_packet["status"],
            "docker": down_packet,
            "container_pid": None,
        }

    # 4. Start Fresh Service And Wait For Readiness
    docker_packet = run_compose_command(
        runtime,
        ["up", "-d", "--force-recreate", "valhalla"],
        timeout_sec=120.0,
    )
    if int(docker_packet["error_code"]) != 0:
        return {
            "error_code": 6,
            "message": docker_packet["stderr"],
            "status": None,
            "docker": docker_packet,
            "container_pid": None,
        }
    ready_packet = wait_for_valhalla(
        config["base_url"],
        startup_timeout_sec=float(config["startup_timeout_sec"]),
    )
    if int(ready_packet["error_code"]) != 0:
        stop_valhalla_service(config)
        return {
            "error_code": 7,
            "message": str(ready_packet["message"]),
            "status": None,
            "docker": docker_packet,
            "container_pid": None,
        }

    # 5. Resolve Container PID
    pid_packet = query_valhalla_container_pid(config)
    if int(pid_packet["error_code"]) != 0:
        stop_valhalla_service(config)
        return {
            "error_code": 8,
            "message": str(pid_packet["message"]),
            "status": ready_packet["status"],
            "docker": docker_packet,
            "container_pid": None,
        }
    return {
        "error_code": 0,
        "message": "Fresh per-item Valhalla service is ready.",
        "status": ready_packet["status"],
        "docker": docker_packet,
        "container_pid": int(pid_packet["container_pid"]),
    }


def build_valhalla_tiles(config: dict) -> dict:
    """
    Purpose:
        Explicitly build dataset-specific Valhalla tiles through Docker.
    Parameters:
        config (dict), validated management configuration.
    Return Dict:
        "error_code": int, 0 when tiles exist and service becomes ready.
        "message": str.
        "docker": dict, Compose command result.
        "status": dict | None.
    Usage:
        The manage prepare/build command calls this before evaluation.
    TODO:
        1) Require exactly one tailored PBF.
        2) Start the pinned Docker image in rebuild mode.
        3) Wait for the built service to become ready.
        4) Verify the resulting tile archive.
        5) Recreate the container with two serving threads.
    """

    # 1. Require Exactly One Tailored PBF
    runtime = build_compose_runtime(config, rebuild=True)
    pbf_files = sorted(runtime["map_dir"].glob("*.osm.pbf"))
    if len(pbf_files) != 1:
        raise RuntimeError(
            f"Expected exactly one tailored PBF under {runtime['map_dir']}, found {len(pbf_files)}."
        )

    # 2. Start Docker In Rebuild Mode
    docker_packet = run_compose_command(
        runtime,
        ["up", "-d", "--force-recreate", "valhalla"],
        timeout_sec=120.0,
    )
    if int(docker_packet["error_code"]) != 0:
        return {
            "error_code": 1,
            "message": docker_packet["stderr"],
            "docker": docker_packet,
            "status": None,
        }

    # 3. Wait For Built Service
    ready_packet = wait_for_valhalla(
        config["base_url"],
        startup_timeout_sec=float(config["build_timeout_sec"]),
    )
    if int(ready_packet["error_code"]) != 0:
        return {
            "error_code": 2,
            "message": ready_packet["message"],
            "docker": docker_packet,
            "status": None,
        }

    # 4. Verify Tile Archive
    tile_archive = runtime["map_dir"] / "valhalla_tiles.tar"
    if not tile_archive.is_file():
        return {
            "error_code": 3,
            "message": f"Valhalla reported ready but tile archive is missing: {tile_archive}",
            "docker": docker_packet,
            "status": ready_packet["status"],
        }

    # 5. Recreate In Resource-Bounded Serve Mode
    serve_runtime = build_compose_runtime(config, rebuild=False)
    serve_packet = run_compose_command(
        serve_runtime,
        ["up", "-d", "--force-recreate", "valhalla"],
        timeout_sec=120.0,
    )
    if int(serve_packet["error_code"]) != 0:
        return {
            "error_code": 4,
            "message": serve_packet["stderr"],
            "docker": {"build": docker_packet, "serve": serve_packet},
            "status": None,
        }
    serve_ready = wait_for_valhalla(
        config["base_url"],
        startup_timeout_sec=float(config["startup_timeout_sec"]),
    )
    return {
        "error_code": int(serve_ready["error_code"]),
        "message": (
            "Valhalla tiles built and service ready in serve mode."
            if int(serve_ready["error_code"]) == 0
            else str(serve_ready["message"])
        ),
        "docker": {"build": docker_packet, "serve": serve_packet},
        "status": serve_ready.get("status"),
    }


def stop_valhalla_service(config: dict) -> dict:
    """
    Purpose:
        Stop the deterministic dataset-specific Docker Compose project.
    Parameters:
        config (dict), validated Valhalla management configuration.
    Return Dict:
        "error_code": int, Docker Compose return code.
        "message": str.
        "docker": dict, captured Compose result.
    Usage:
        The management CLI invokes this only for an explicit down command.
    TODO:
        1) Resolve the dataset Compose runtime.
        2) Run Docker Compose down.
        3) Return the captured result.
    """

    # 1. Resolve Dataset Compose Runtime
    runtime = build_compose_runtime(config, rebuild=False)

    # 2. Run Docker Compose Down
    docker_packet = run_compose_command(runtime, ["down"], timeout_sec=120.0)

    # 3. Return Captured Result
    return {
        "error_code": int(docker_packet["error_code"]),
        "message": (
            "Valhalla service stopped."
            if int(docker_packet["error_code"]) == 0
            else docker_packet["stderr"]
        ),
        "docker": docker_packet,
    }


__all__ = [
    "build_compose_runtime",
    "build_valhalla_tiles",
    "ensure_valhalla_service",
    "query_valhalla_container_pid",
    "query_valhalla_status",
    "run_compose_command",
    "stop_valhalla_service",
    "wait_for_valhalla",
]
