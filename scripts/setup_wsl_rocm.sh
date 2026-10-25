#!/usr/bin/env bash
# Setup ROCm 7.2 + PyTorch (ROCm wheels) on WSL2 Ubuntu 24.04 for MiniMind.
# Idempotent: re-running skips steps that are already done.
# Usage: bash scripts/setup_wsl_rocm.sh [--yes]
set -euo pipefail

YES=${1:-}
VENV="${HOME}/venv-rocm"
WHEEL_DIR="${HOME}/rocm-wheels"
BASE="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2"
TORCH="torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl"
TV="torchvision-0.24.0%2Brocm7.2.0.gitb919bd0c-cp312-cp312-linux_x86_64.whl"
TRI="triton-3.5.1%2Brocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl"

say() { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }
warn() { printf "\033[1;33m[!] %s\033[0m\n" "$*"; }

say "0/5 preflight"
[ -e /dev/dxg ] || { echo "ERROR: /dev/dxg not found. Windows driver / WSL2 GPU passthrough is missing."; exit 1; }
. /etc/os-release
[ "${VERSION_ID}" = "24.04" ] || warn "tested on Ubuntu 24.04, current: ${VERSION_ID}"
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' || { echo "ERROR: need Python 3.12"; exit 1; }
id -nG | grep -qw render || warn "user not in 'render' group; run: sudo usermod -aG render,video \$USER && wsl --shutdown"

say "1/5 base packages"
sudo apt-get update -qq
sudo apt-get install -y -qq curl wget git python3-venv python3-pip build-essential

if command -v rocminfo >/dev/null 2>&1; then
  say "2/5 ROCm already installed, skip"
else
  say "2/5 install ROCm 7.2 (wsl usecase, no dkms)"
  DEB="amdgpu-install_7.2.70200-1_all.deb"
  wget -q -O "/tmp/${DEB}" "https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/${DEB}"
  sudo apt-get install -y "/tmp/${DEB}"
  sudo amdgpu-install -y --usecase=wsl,rocm --no-dkms
fi

say "3/5 python venv"
[ -d "${VENV}" ] || python3 -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install -q --upgrade pip wheel

say "4/5 PyTorch ROCm wheels (three wheels must be installed together)"
mkdir -p "${WHEEL_DIR}"
for f in "${TORCH}" "${TV}" "${TRI}"; do
  [ -f "${WHEEL_DIR}/${f}" ] || wget -q -c -O "${WHEEL_DIR}/${f}" "${BASE}/${f}"
done
pip install -q --find-links "${WHEEL_DIR}" \
  "torch==2.9.1+rocm7.2.0.lw.git7e1940d4" \
  "torchvision==0.24.0+rocm7.2.0.gitb919bd0c" \
  "triton==3.5.1+rocm7.2.0.gita272dfa8"

say "5/5 verify"
rocminfo | grep -E "gfx|Marketing" | head -4
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
PY

cat <<'TIP'

Add this to ~/.bashrc to reduce HSA allocator fragmentation in WSL:
  export PYTORCH_ALLOC_CONF=expandable_segments:True
TIP
