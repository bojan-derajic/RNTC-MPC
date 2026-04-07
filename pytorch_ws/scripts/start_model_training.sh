#!/bin/bash

CONTAINER_NAME="model_training"

IMAGE_NAME="pytorch-ws"
IMAGE_TAG="latest"

CMD="python src/train_rntc.py"

docker run --rm --name ${CONTAINER_NAME} \
    --gpus all \
    -v ${PWD}:/workspace \
    ${IMAGE_NAME}:${IMAGE_TAG} \
    ${CMD}