# ROS 2 Jazzy agent: px4_msgs underlay + agentic_uav workspace.
FROM ros:jazzy

ARG PX4_MSGS_BRANCH=release/1.16
ENV DEBIAN_FRONTEND=noninteractive PIP_BREAK_SYSTEM_PACKAGES=1 PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10

RUN apt-get update && apt-get install -y --no-install-recommends \
      git python3-pip python3-pytest python3-colcon-common-extensions \
      ros-jazzy-ros-gz-bridge ros-jazzy-cv-bridge \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch in its own layer: it is the largest download and rarely changes.
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision
COPY docker/agent-requirements.txt /tmp/agent-requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/agent-requirements.txt

# px4_msgs must match the PX4 version running in the sim container.
RUN git clone --depth 1 --branch ${PX4_MSGS_BRANCH} https://github.com/PX4/px4_msgs.git /opt/px4_ws/src/px4_msgs \
    && . /opt/ros/jazzy/setup.sh && cd /opt/px4_ws && colcon build --event-handlers console_direct- \
    && rm -rf build log

COPY src /ws/src
RUN . /opt/ros/jazzy/setup.sh && . /opt/px4_ws/install/setup.sh && cd /ws \
    && colcon build --symlink-install && chown -R 1000:1000 /ws

COPY docker/agent_entrypoint.sh /usr/local/bin/agent_entrypoint.sh
RUN chmod +x /usr/local/bin/agent_entrypoint.sh && mkdir -p /data && chown 1000:1000 /data
USER ubuntu
WORKDIR /ws
ENTRYPOINT ["/usr/local/bin/agent_entrypoint.sh"]
CMD ["ros2", "launch", "agentic_uav_bringup", "agent.launch.py"]
