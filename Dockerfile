# ============================================================================
# Dockerfile: 5-Layer Agentic UAV Architecture (ROS 2 Humble + Gazebo Harmonic)
# Optimized for Fedora VM Host deployment with X11/Wayland & DRI acceleration
# Base: Ubuntu 22.04 LTS (Jammy) / ros:humble
# ============================================================================

FROM ros:humble

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash
ENV ROS_DISTRO=humble

# 1. Install system utilities, graphics libraries, and developer tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    lsb-release \
    wget \
    git \
    sudo \
    python3-pip \
    python3-dev \
    python3-colcon-common-extensions \
    python3-rosdep \
    mesa-utils \
    libgl1-mesa-dri \
    libgl1-mesa-glx \
    libvulkan1 \
    libegl1 \
    libxext6 \
    libx11-6 \
    x11-apps \
    net-tools \
    iputils-ping \
    ros-humble-desktop \
    && rm -rf /var/lib/apt/lists/*

# 2. Add OSRF repository and install Gazebo Harmonic + ROS GZ Harmonic bridge
RUN wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/gazebo-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gazebo-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/gazebo-stable.list && \
    apt-get update && \
    (apt-get install -y --no-install-recommends gz-harmonic ros-humble-ros-gzharmonic || \
     apt-get install -y --no-install-recommends gz-harmonic ros-humble-ros-gz) && \
    apt-get install -y --no-install-recommends \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-camera-info-manager \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    && rm -rf /var/lib/apt/lists/*

# 3. Setup non-root developer user matching Fedora VM default (UID=1000, GID=1000)
ARG USERNAME=uav_operator
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME && \
    usermod -aG video,audio $USERNAME

# 4. Install Python AI/ML dependencies
WORKDIR /ros2_ws
COPY requirements.txt /ros2_ws/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir "setuptools<=58.2.0" wheel && \
    python3 -m pip install --no-cache-dir --ignore-installed \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch torchvision && \
    python3 -m pip install --no-cache-dir --ignore-installed \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r /ros2_ws/requirements.txt

# 5. Copy Agentic UAV ROS 2 workspace packages
COPY src /ros2_ws/src
RUN chown -R $USERNAME:$USERNAME /ros2_ws

# 6. Switch to non-root user and initialize rosdep
USER $USERNAME
WORKDIR /ros2_ws

RUN rosdep update && \
    /bin/bash -c "source /opt/ros/humble/setup.bash && \
    rosdep install --from-paths src --ignore-src -r -y || true"

# 7. Build the ROS 2 workspace
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release"

# 8. Setup runtime environment and entrypoint
COPY entrypoint.sh /ros2_ws/entrypoint.sh
USER root
RUN chmod +x /ros2_ws/entrypoint.sh && \
    echo "source /opt/ros/humble/setup.bash" >> /home/$USERNAME/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> /home/$USERNAME/.bashrc
USER $USERNAME

# Environment variables for GUI and DDS communications
ENV DISPLAY=:0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV QT_X11_NO_MITSHM=1

ENTRYPOINT ["/ros2_ws/entrypoint.sh"]
CMD ["ros2", "launch", "agentic_uav", "agentic_uav.launch.py"]
