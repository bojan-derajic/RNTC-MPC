FROM osrf/ros:jazzy-desktop-full

ARG USERNAME=ubuntu

# Install apt packages
RUN apt update -q \
    && apt upgrade -q -y \
    && apt install -y --no-install-recommends \
    software-properties-common \
    python3-pip \
    xauth \
    terminator \
    sudo \
    libblas-dev \
    liblapack-dev \
    libmetis-dev \
    pkg-config \
    gfortran \
    && apt clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install ROS packages
RUN apt update -q \
    && apt install -y --no-install-recommends \
    ros-$ROS_DISTRO-robot-localization \
    ros-$ROS_DISTRO-navigation2 \
    ros-$ROS_DISTRO-nav2-bringup \
    ros-$ROS_DISTRO-plotjuggler-ros \
    ros-$ROS_DISTRO-ros2-control \
    ros-$ROS_DISTRO-ros2-controllers \ 
    ros-$ROS_DISTRO-gz-ros2-control 

RUN echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers 
USER $USERNAME

# Create workspace so that user own this directory
RUN mkdir -p /home/$USERNAME/ros2_ws
WORKDIR /home/$USERNAME/ros2_ws

# Install Python requirements
COPY ./requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Copy configuration files
RUN echo 'source /opt/ros/'$ROS_DISTRO'/setup.bash' >> /home/$USERNAME/.bashrc \
    && echo 'source /home/'$USERNAME'/ros2_ws/install/setup.bash' >> /home/$USERNAME/.bashrc

CMD ["/bin/bash"]
