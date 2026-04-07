#!/bin/bash

CONTAINER_NAME="sim_experiments"

IMAGE_NAME="ros2-ws"
IMAGE_TAG="latest"

if docker ps -a --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping existing container: ${CONTAINER_NAME}"
    docker kill ${CONTAINER_NAME}
    sleep 4
fi

CMD="source /opt/ros/jazzy/setup.bash && \
     source /home/ubuntu/ros2_ws/install/setup.bash && \
     /home/ubuntu/ros2_ws/scripts/install_hsl_lib.sh && \
     ros2 launch simulation_bringup jackal_robot.launch.py"

docker run --rm --name ${CONTAINER_NAME} \
    --gpus all \
    -v ${PWD}:/home/ubuntu/ros2_ws \
    --entrypoint /bin/bash \
    ${IMAGE_NAME}:${IMAGE_TAG} \
    -c "${CMD}"