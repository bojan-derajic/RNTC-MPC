#!/bin/bash

CONTAINER_NAME="mlflow_server"

IMAGE_NAME="pytorch-ws"
IMAGE_TAG="latest"

SERVER_PORT="8080"

CMD="mlflow server --host 0.0.0.0 --port ${SERVER_PORT}"

docker run --rm --name ${CONTAINER_NAME} \
    -p ${SERVER_PORT}:${SERVER_PORT} \
    -v ${PWD}:/workspace \
    ${IMAGE_NAME}:${IMAGE_TAG} \
    ${CMD}