#!/bin/bash

# Set the image name and tag
IMAGE_NAME="ros2-ws"
TAG="latest"

# Set path to the Dockerfile
DOCKERFILE_PATH="."

# Build docker image
docker build -t ${IMAGE_NAME}:${TAG} ${DOCKERFILE_PATH}