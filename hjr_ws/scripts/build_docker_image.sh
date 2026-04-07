#!/bin/bash
# Build the HJR workspace Docker image.
#
# Run this script from the repository root before using generate_data.sh
# or before opening the project in a VS Code Dev Container for the first time.
#
# Usage:
#   ./scripts/build_docker_image.sh

set -euo pipefail

IMAGE_NAME="hjr-ws"
TAG="latest"

# The Dockerfile lives at the repository root
DOCKERFILE_PATH="."

echo "Building Docker image ${IMAGE_NAME}:${TAG}..."
docker build -t "${IMAGE_NAME}:${TAG}" "${DOCKERFILE_PATH}"
echo "Build complete: ${IMAGE_NAME}:${TAG}"
