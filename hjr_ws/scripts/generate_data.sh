#!/bin/bash
# Run the data-generation pipeline inside a Docker container with GPU access.
#
# The script bind-mounts the current working directory to /workspace inside
# the container, installs the hj_reachability submodule in editable mode,
# and then executes src/generate_data.py.  Output is written to ./data/ on
# the host machine.
#
# Prerequisites:
#   1. Docker with the NVIDIA container toolkit installed.
#   2. The Docker image has been built: ./scripts/build_docker_image.sh
#   3. The hj_reachability submodule is present: git submodule update --init
#
# Usage:
#   ./scripts/generate_data.sh

set -euo pipefail

CONTAINER_NAME="hjr-data-generation"
IMAGE_NAME="hjr-ws"
IMAGE_TAG="latest"

echo "Starting data generation in container '${CONTAINER_NAME}'..."

docker run --rm \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    -v "${PWD}:/workspace" \
    "${IMAGE_NAME}:${IMAGE_TAG}" \
    /bin/bash -c "pip install -e hj_reachability && python src/generate_data.py"

echo "Data generation finished. Output written to ./data/"
