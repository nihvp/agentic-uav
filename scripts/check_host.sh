#!/usr/bin/env bash
# Check that this machine can run the stack. Prints OK / WARN / FAIL per item.
#   scripts/check_host.sh [--gpu-test]   (--gpu-test runs nvidia-smi inside a container)
set -uo pipefail
fails=0
ok()   { printf '  OK    %s\n' "$*"; }
warn() { printf '  WARN  %s\n' "$*"; }
fail() { printf '  FAIL  %s\n' "$*"; fails=$((fails + 1)); }

echo "Host"
[[ "$(uname -m)" == x86_64 ]] && ok "arch x86_64" || warn "arch $(uname -m) (images are tested on x86_64)"
# shellcheck disable=SC1091
. /etc/os-release 2>/dev/null && ok "OS ${PRETTY_NAME:-unknown}"
[[ "$(id -u)" == 1000 ]] && ok "uid 1000 (matches container users)" \
  || warn "uid $(id -u): containers run as uid 1000, mounted ./src may get permission errors"
ok "CPUs $(nproc), RAM $(free -g | awk '/Mem:/ {print $2}') GiB"
avail=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
if (( avail >= 100 )); then ok "disk free ${avail} GiB"; else warn "disk free ${avail} GiB (images + models need ~100 GiB)"; fi

echo "GPU"
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  ok "$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
else
  fail "nvidia-smi missing or failing: install the NVIDIA driver"
fi

echo "Docker"
if ! command -v docker >/dev/null; then
  fail "docker not installed"
else
  if docker info >/dev/null 2>&1; then
    ok "docker $(docker version --format '{{.Server.Version}}') usable without sudo"
  else
    fail "docker info failed: start dockerd, or run: sudo usermod -aG docker \$USER (then log out and in)"
  fi
  if docker compose version >/dev/null 2>&1; then ok "$(docker compose version)"; else fail "docker compose plugin missing"; fi
  if docker info 2>/dev/null | grep -qi 'runtimes:.*nvidia'; then
    ok "NVIDIA runtime registered with Docker"
  else
    fail "NVIDIA Container Toolkit not configured: install nvidia-container-toolkit, then run
          sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
  fi
  if [[ "${1:-}" == --gpu-test ]]; then
    if docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L; then ok "GPU visible inside a container"
    else fail "GPU not visible inside a container"; fi
  fi
fi

echo "Display"
if [[ -n "${DISPLAY:-}" ]]; then
  ok "DISPLAY=${DISPLAY}, session ${XDG_SESSION_TYPE:-unknown}"
  [[ "${XDG_SESSION_TYPE:-x11}" == x11 ]] || warn "not an X11 session; Gazebo GUI may need XWayland"
  command -v xhost >/dev/null || warn "xhost missing (sudo apt install x11-xserver-utils); needed for the Gazebo GUI"
else
  warn "no DISPLAY (SSH?): use scripts/run.sh --headless, or run from the AnyDesk desktop terminal"
fi

echo "Ports"
for port in 8888/udp 11434/tcp; do
  p=${port%/*}; proto=${port#*/}
  if ss -lnH "--$proto" 2>/dev/null | awk '{print $5}' | grep -q ":$p\$"; then
    warn "port $port already in use (8888 = PX4 XRCE agent, 11434 = Ollama); stop the other service"
  else
    ok "port $port free"
  fi
done

echo
if (( fails == 0 )); then echo "Ready."; else echo "$fails problem(s) to fix."; exit 1; fi
