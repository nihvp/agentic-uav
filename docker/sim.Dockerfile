# PX4 SITL + Gazebo Harmonic + Micro XRCE-DDS Agent (Ubuntu 24.04, amd64 or arm64).
FROM ubuntu:24.04

ARG PX4_TAG=v1.16.0
ARG XRCE_AGENT_TAG=v2.4.3
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      sudo git ca-certificates lsb-release wget curl build-essential cmake \
      python3 python3-pip python3-venv tzdata \
    && rm -rf /var/lib/apt/lists/*

# Replace the image's default uid-1000 user with one that matches the Fedora host user.
RUN userdel -r ubuntu 2>/dev/null || true; \
    useradd -m -u 1000 -s /bin/bash px4 && echo "px4 ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/px4

RUN git clone --branch ${PX4_TAG} --depth 1 --recurse-submodules --shallow-submodules \
      https://github.com/PX4/PX4-Autopilot.git /opt/PX4-Autopilot \
    && chown -R px4:px4 /opt/PX4-Autopilot

# PX4's setup script installs Gazebo Harmonic and the Python build deps on 24.04.
USER px4
RUN bash /opt/PX4-Autopilot/Tools/setup/ubuntu.sh --no-nuttx && sudo rm -rf /var/lib/apt/lists/*
RUN cd /opt/PX4-Autopilot && make px4_sitl

USER root
RUN git clone --branch ${XRCE_AGENT_TAG} --depth 1 https://github.com/eProsima/Micro-XRCE-DDS-Agent.git /tmp/xrce \
    && cmake -S /tmp/xrce -B /tmp/xrce/build && cmake --build /tmp/xrce/build -j"$(nproc)" \
    && cmake --install /tmp/xrce/build && ldconfig && rm -rf /tmp/xrce

COPY docker/sim_entrypoint.sh /usr/local/bin/sim_entrypoint.sh
RUN chmod +x /usr/local/bin/sim_entrypoint.sh
USER px4
WORKDIR /opt/PX4-Autopilot
ENTRYPOINT ["/usr/local/bin/sim_entrypoint.sh"]
