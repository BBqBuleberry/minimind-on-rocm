# 环境搭建：Windows 11 + WSL2 + ROCm 7.2 + PyTorch ROCm

> 实测环境：i5-14600KF / DDR4 32G / RX 7800 XT 16GB / Win11 26H2 / WSL 3.0.3 / Ubuntu 24.04.5
> 全程不需要 Docker，不需要从源码编译。

## 0. 前置检查

```bash
winver                 # Windows 11 22H2 及以上
# AMD Software 版本 >= 25.x（24.x 会导致 HIP 组件过旧）
wsl --version          # 2.x 或以上
wsl -l -v              # Ubuntu 必须是 VERSION 2
```

任务管理器中确认：GPU 名称含 `RX 7800 XT`，专用 GPU 内存 16.0 GB。

## 1. Ubuntu 基础环境

```bash
sudo apt update
sudo apt install -y curl wget git python3-venv python3-pip build-essential
sudo usermod -aG render,video $USER
```

改完必须 `wsl --shutdown` 再进入，组权限才生效。验证：`id` 输出里有 `render`、`video`。

```bash
ls -l /dev/dxg      # 必须存在，这是 Windows 把 GPU 透传给 WSL 的设备节点
```

## 2. 安装 ROCm（WSL usecase）

```bash
wget https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/amdgpu-install_7.2.70200-1_all.deb
sudo apt install -y ./amdgpu-install_7.2.70200-1_all.deb
amdgpu-install --list-usecase          # 确认列表里有 wsl
sudo amdgpu-install -y --usecase=wsl,rocm --no-dkms
```

> `--no-dkms` 是硬性要求：WSL 没有真实内核模块，装 DKMS 必失败。

验证：

```bash
rocminfo | grep -E "gfx|Marketing"     # 期望 gfx1101 / Radeon RX 7800 XT
```

> WSL 中 `rocm-smi` **一定失败**（`amdgpu not found in modules`），因为显卡经 dxg 透传、
> 没有 amdgpu 内核模块。请用 `rocminfo` 作为验收工具。

## 3. PyTorch ROCm 轮子（必须成套安装）

```bash
python3 -m venv ~/venv-rocm && source ~/venv-rocm/bin/activate
pip install --upgrade pip wheel
mkdir -p wheels && cd wheels
BASE=https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2
wget -c "$BASE/torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl"
wget -c "$BASE/torchvision-0.24.0%2Brocm7.2.0.gitb919bd0c-cp312-cp312-linux_x86_64.whl"
wget -c "$BASE/triton-3.5.1%2Brocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl"
pip install --find-links . "torch==2.9.1+rocm7.2.0.lw.git7e1940d4" \
  "torchvision==0.24.0+rocm7.2.0.gitb919bd0c" "triton==3.5.1+rocm7.2.0.gita272dfa8"
```

> 三个包互锁：`torch` 的依赖里钉死了 `triton==...+rocm7.2.0`，PyPI 上没有这个版本，
> 必须本地一起装，否则报 `No matching distribution found for triton`。

验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 4. 训练前必设的环境变量

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True   # 缓解 WSL 下的 HSA 内存池碎片
```

建议写入 `~/.bashrc`。

## 5. 验收清单

- [ ] `/dev/dxg` 存在
- [ ] `id` 含 render/video
- [ ] `rocminfo` 出现 gfx1101
- [ ] `torch.cuda.is_available()` 为 True
- [ ] fp16 4096³ 矩阵乘 >= 20 TFLOPS
