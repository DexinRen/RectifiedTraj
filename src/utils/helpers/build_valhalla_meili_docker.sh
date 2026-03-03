#!/usr/bin/env bash
set -euo pipefail

print_usage() {
  cat <<'EOF'
Build and start a Valhalla Meili Docker runtime with persistent artifacts.

Usage:
  build_valhalla_meili_docker.sh [options]

Options:
  --dataset <name>           Dataset token (default: NUMOSIM_Kanto)
  --map <path>               Input sliced map path (default: ./dataset/map_processed/map_<dataset>.osm.pbf)
  --runtime-root <path>      Runtime root directory (default: ./src/baseline/models/valhalla_meili/runtime)
  --image <name>             Docker image (default: ghcr.io/valhalla/valhalla-scripted:latest)
  --container-name <name>    Docker container name (default: valhalla_meili_<dataset_token>)
  --host-port <port>         Host port (default: 8002)
  --container-port <port>    Container port (default: 8002)
  --extra-args "<args>"      Extra args passed to docker run
  --timeout-sec <seconds>    Startup wait timeout (default: 900)
  --poll-sec <seconds>       Status poll interval (default: 2)
  --save-image-tar <path>    Optional: docker save image to a tar file
  --skip-pull                Skip docker pull
  --stop-after-ready         Stop container after build is complete
  -h, --help                 Show this help

Environment overrides:
  VALHALLA_DOCKER_IMAGE, VALHALLA_DOCKER_CONTAINER, VALHALLA_PORT,
  VALHALLA_CONTAINER_PORT, VALHALLA_DOCKER_EXTRA_ARGS, VALHALLA_RUNTIME_DIR,
  VALHALLA_STARTUP_TIMEOUT_SEC, VALHALLA_STARTUP_POLL_SEC

Notes:
  - Compiled runtime data is persisted under:
      <runtime-root>/<dataset_token>/
  - This path is mounted into container as /custom_files, matching baseline runtime behavior.
EOF
}

safe_token() {
  local raw="${1:-default}"
  raw="$(echo "${raw}" | tr '[:upper:]' '[:lower:]')"
  raw="$(echo "${raw}" | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//')"
  if [[ -z "${raw}" ]]; then
    raw="default"
  fi
  echo "${raw}"
}

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "[valhalla-build] missing command: ${cmd}" >&2
    exit 1
  fi
}

assert_int() {
  local name="$1"
  local value="$2"
  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "[valhalla-build] ${name} must be an integer, got: ${value}" >&2
    exit 1
  fi
}

DATASET="${DATASET:-NUMOSIM_Kanto}"
MAP_PATH="${MAP_PATH:-}"
RUNTIME_ROOT="${VALHALLA_RUNTIME_DIR:-./src/baseline/models/valhalla_meili/runtime}"
DOCKER_IMAGE="${VALHALLA_DOCKER_IMAGE:-ghcr.io/valhalla/valhalla-scripted:latest}"
CONTAINER_NAME_OVERRIDE="${VALHALLA_DOCKER_CONTAINER:-}"
HOST_PORT="${VALHALLA_PORT:-8002}"
CONTAINER_PORT="${VALHALLA_CONTAINER_PORT:-8002}"
EXTRA_ARGS="${VALHALLA_DOCKER_EXTRA_ARGS:-}"
TIMEOUT_SEC="${VALHALLA_STARTUP_TIMEOUT_SEC:-900}"
POLL_SEC="${VALHALLA_STARTUP_POLL_SEC:-2}"
SKIP_PULL=0
STOP_AFTER_READY=0
SAVE_IMAGE_TAR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --map)
      MAP_PATH="$2"
      shift 2
      ;;
    --runtime-root)
      RUNTIME_ROOT="$2"
      shift 2
      ;;
    --image)
      DOCKER_IMAGE="$2"
      shift 2
      ;;
    --container-name)
      CONTAINER_NAME_OVERRIDE="$2"
      shift 2
      ;;
    --host-port)
      HOST_PORT="$2"
      shift 2
      ;;
    --container-port)
      CONTAINER_PORT="$2"
      shift 2
      ;;
    --extra-args)
      EXTRA_ARGS="$2"
      shift 2
      ;;
    --timeout-sec)
      TIMEOUT_SEC="$2"
      shift 2
      ;;
    --poll-sec)
      POLL_SEC="$2"
      shift 2
      ;;
    --save-image-tar)
      SAVE_IMAGE_TAR="$2"
      shift 2
      ;;
    --skip-pull)
      SKIP_PULL=1
      shift
      ;;
    --stop-after-ready)
      STOP_AFTER_READY=1
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "[valhalla-build] unknown option: $1" >&2
      print_usage
      exit 2
      ;;
  esac
done

assert_int "host-port" "${HOST_PORT}"
assert_int "container-port" "${CONTAINER_PORT}"
assert_int "timeout-sec" "${TIMEOUT_SEC}"

DATASET_TOKEN="$(safe_token "${DATASET}")"
if [[ -z "${MAP_PATH}" ]]; then
  MAP_PATH="./dataset/map_processed/map_${DATASET}.osm.pbf"
fi
if [[ ! -f "${MAP_PATH}" ]]; then
  echo "[valhalla-build] map file not found: ${MAP_PATH}" >&2
  exit 1
fi

require_cmd docker
require_cmd curl

mkdir -p "${RUNTIME_ROOT}"
RUNTIME_ROOT_ABS="$(cd "${RUNTIME_ROOT}" && pwd)"
RUNTIME_DIR="${RUNTIME_ROOT_ABS}/${DATASET_TOKEN}"
mkdir -p "${RUNTIME_DIR}"
MAP_DST="${RUNTIME_DIR}/input.osm.pbf"

echo "[valhalla-build] staging map: ${MAP_PATH} -> ${MAP_DST}"
cp -f "${MAP_PATH}" "${MAP_DST}"

if [[ "${SKIP_PULL}" -eq 0 ]]; then
  echo "[valhalla-build] pulling image: ${DOCKER_IMAGE}"
  docker pull "${DOCKER_IMAGE}" >/dev/null
fi

CONTAINER_NAME="${CONTAINER_NAME_OVERRIDE:-valhalla_meili_${DATASET_TOKEN}}"

if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  echo "[valhalla-build] removing existing container: ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

RUN_CMD=(
  docker run -d
  --name "${CONTAINER_NAME}"
  -p "${HOST_PORT}:${CONTAINER_PORT}"
  -v "${RUNTIME_DIR}:/custom_files"
)
if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARR=(${EXTRA_ARGS})
  RUN_CMD+=("${EXTRA_ARGS_ARR[@]}")
fi
RUN_CMD+=("${DOCKER_IMAGE}")

echo "[valhalla-build] starting container: ${CONTAINER_NAME}"
CONTAINER_ID="$("${RUN_CMD[@]}")"
echo "[valhalla-build] container id: ${CONTAINER_ID}"

STATUS_URL="http://127.0.0.1:${HOST_PORT}/status"
echo "[valhalla-build] waiting for service: ${STATUS_URL}"
DEADLINE=$((SECONDS + TIMEOUT_SEC))
while ! curl -fsS "${STATUS_URL}" >/dev/null 2>&1; do
  if (( SECONDS >= DEADLINE )); then
    echo "[valhalla-build] startup timeout after ${TIMEOUT_SEC}s" >&2
    echo "[valhalla-build] recent container logs:" >&2
    docker logs --tail 120 "${CONTAINER_NAME}" >&2 || true
    exit 1
  fi
  sleep "${POLL_SEC}"
done
echo "[valhalla-build] service is ready"

if [[ -n "${SAVE_IMAGE_TAR}" ]]; then
  mkdir -p "$(dirname "${SAVE_IMAGE_TAR}")"
  echo "[valhalla-build] exporting image tar: ${SAVE_IMAGE_TAR}"
  docker save -o "${SAVE_IMAGE_TAR}" "${DOCKER_IMAGE}"
fi

if [[ "${STOP_AFTER_READY}" -eq 1 ]]; then
  echo "[valhalla-build] stopping container after build: ${CONTAINER_NAME}"
  docker stop "${CONTAINER_NAME}" >/dev/null
fi

echo
echo "Valhalla Meili runtime is ready."
echo "Runtime dir (persisted compiled data): ${RUNTIME_DIR}"
echo "Container: ${CONTAINER_NAME}"
echo "Endpoint: ${STATUS_URL}"
echo
echo "Suggested env for benchmark:"
echo "  export VALHALLA_URL=http://127.0.0.1:${HOST_PORT}"
echo "  export VALHALLA_RUNTIME_DIR=${RUNTIME_ROOT}"
echo "  export VALHALLA_DOCKER_CONTAINER=${CONTAINER_NAME}"
