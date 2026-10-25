# MiniMind on AMD RX 7800 XT — 实验记录

> 目标：在 AMD RDNA3 消费级显卡（RX 7800 XT, gfx1101）上，不依赖 CUDA，完成
> 「本地推理 → WSL + ROCm + PyTorch 环境 → 小模型 full SFT → LoRA 单变量实验」的端到端复现。
> 全部实验在 Windows 11 + WSL2 中完成，未使用 Docker，未从源码编译任何框架。

---

## TL;DR

| 阶段 | 内容 | 关键结果 |
|---|---|---|
| 1 | LM Studio + ROCm 运行时，7B GGUF 推理 | Qwen2.5-7B Q4_K_M，**95.9 tok/s**，显存 5.6 GB |
| 2 | WSL2 + ROCm 7.2 + PyTorch ROCm 轮子 | `rocminfo` 认出 gfx1101；fp16 矩阵乘 **45.7 TFLOPS** |
| 3 | MiniMind-3（64M）推理 + full SFT 400 步 | loss 6.69 → ~4.0；模型身份由「通义千问」变为「MiniMind」 |
| 4 | LoRA rank 单变量实验（8/16/32）+ 学习率对照 | loss 1.052/0.947/0.843，但**定性检查发现 r32 输出崩坏**；lr 降至 5e-5 可恢复稳定却学不到身份 → 选 r=8 |

---

## 0. 复现范围声明（Scope）

**本项目不是 MiniMind 的完整复现**，而是在 AMD/ROCm 环境下对**关键链路的最小可运行验证 + 自有实验**。
说清楚边界，比含糊地说"我复现了"更有价值。

| 模块 | 原作者做法 | 本项目做法 | 状态 |
|---|---|---|---|
| 环境 | 8×RTX 3090 + CUDA 12.2 + Ubuntu 20.04 | 单卡 RX 7800 XT + WSL2 + ROCm 7.2 | ✅ 等价替代 |
| 预训练（from scratch） | `pretrain_t2t_mini` 1.2 GB，约 1.2 h/epoch | **未执行** | ❌ 未覆盖 |
| 完整 SFT | `sft_t2t_mini` 1.6 GB，约 1.1 h/epoch | 前 2000 条子集，400 步 | 🟡 最小验证 |
| LoRA 微调 | 官方 `lora_identity` / `lora_medical` | 自建中文身份数据集（392 条）+ 三档 rank | 🟢 自有扩展 |
| 推理 | transformers / llama.cpp / ollama / vllm | transformers（本地权重） | ✅ 已完成 |
| RLHF / DPO / GRPO / RLAIF | 完整实现 | **未执行** | ❌ 未覆盖 |
| 蒸馏（黑盒/白盒） | 完整实现 | **未执行** | ❌ 未覆盖 |
| Tool Call / Agentic RL | 完整实现 | **未执行** | ❌ 未覆盖 |
| Tokenizer 训练 | 有 | 直接使用官方词表 | ❌ 未覆盖 |
| Benchmark（C-Eval 等） | lm-evaluation-harness | **未执行** | ❌ 未覆盖 |
| 模型转换 / 部署 | GGUF、ollama、OpenAI API | **未执行** | ❌ 未覆盖 |
| MoE 版本（198M-A64M） | 有 | 未涉及 | ❌ 未覆盖 |

**结论**：属于 *pipeline validation / feasibility study*，不是 *reproduction*。
严格意义的复现要求「同数据、同超参、同预算下复现原作者报告的指标」，本项目未满足（数据量差 3 个数量级、未跑 benchmark）。

**可以主张的部分**：ROCm/WSL 环境适配、崩溃归因、LoRA 单变量实验与数据集构建——这些是原作者未覆盖的内容。

> 方法学层面的 10 条教训（每一条都有本仓库数据支撑）见 [PITFALLS.md](./PITFALLS.md)。
---

## 1. 环境

| 项 | 版本 |
|---|---|
| CPU / 内存 | i5-14600KF / DDR4 16G×2 3600 |
| GPU | AMD Radeon RX 7800 XT 16GB（gfx1101, RDNA3, 60 CU） |
| OS | Windows 11 26H2（26300.9711） |
| 驱动 | AMD Adrenalin 26.10.1（日期 2026-10-18） |
| WSL | 3.0.3.0，Ubuntu 24.04.5 LTS，内核 6.18.54.1-microsoft-standard-WSL2 |
| ROCm | 7.2（`amdgpu-install_7.2.70200-1_all.deb`，`--usecase=wsl,rocm --no-dkms`） |
| PyTorch | torch 2.9.1+rocm7.2.0.lw / torchvision 0.24.0 / triton 3.5.1（AMD 官方 whl） |
| 框架 | transformers 4.57.6 / datasets 3.6.0 / modelscope 1.37.0 |

环境分层：图形工具（LM Studio）用于快速验证；`~/ai/rocm-test/.venv` 用于计算环境验证；
`~/ai/minimind-venv/.venv` 用于训练，三者互不污染。

---

## 2. 阶段一：Windows 图形界面推理（LM Studio）

- LM Studio 0.4.26，运行时 `llama.cpp-win-x86_64-amd-rocm-avx2` 2.62.1（Extension Pack）
- 模型 `lmstudio-community/Qwen2.5-7B-Instruct-GGUF` Q4_K_M（4.68 GB）
- 加载参数：Context 8192、GPU Offload 28/28、Flash Attention 开
- 实测：**95.92 tok/s**（200 字中文回答）、TTFT 0.08 s、专用显存峰值 5.6 GB

> 判据：ROCm 后端生成速度应落在 50–75 tok/s 以上；若落在 25–40 说明走了 Vulkan，
> 10 以下说明回退 CPU。本机 95.9 tok/s 属上游水平。

## 3. 阶段二：WSL2 + ROCm + PyTorch

- WSL 内 GPU 透传依赖 `/dev/dxg`（存在即说明 Windows 驱动侧已就绪）
- 用户需加入 `render`、`video` 组；安装后必须 `wsl --shutdown` 再进入
- 验收：`rocminfo` 输出 `gfx1101` / `Radeon RX 7800 XT`；
  `torch.cuda.is_available()` → True；fp16 4096³ 矩阵乘 20 次 → **45.7 TFLOPS**（理论峰值 74 TFLOPS 的 61%）

> 注意：WSL 中 `rocm-smi` 必然失败（`amdgpu not found in modules`），
> 因为 WSL 没有 amdgpu 内核模块，显卡经 dxg 透传；验收工具应为 `rocminfo`。

## 4. 阶段三：MiniMind 推理与 full SFT

- 模型：`gongjy/minimind-3`（64M，Dense，8 层 / d_model 768，词表 6400）
- 推理：`eval_llm.py --load_from ./minimind-3`，GPU fp16，解码速度约 200–230 tok/s
- SFT：起点 `pretrain_768.pth`，数据 `sft_t2t_mini.jsonl` 前 2000 条，
  `batch 4 / seq 384 / lr 1e-5 / 1 epoch`，约 400 步
- 结果：loss 6.6887 → ~4.0；输出身份由「我是通义千问」变为「我是 MiniMind，由 jingyaogong 创建」
- 产出：`out/full_sft_sub_768.pth`（132 MB, fp16）、`checkpoints/*_resume.pth`（619 MB）

> 观察：minimind-3 默认会先输出一段规划文本再 `</think>` 回答，
> `max_new_tokens` 小于 256 时会截断在「内心独白」阶段，表现为答非所问。
> 推荐推理参数：`--max_new_tokens 320 --temperature 0.6`。

## 5. 阶段四：LoRA 单变量实验

统一设置：起点 `full_sft_sub`，数据 `lora_identity.jsonl`（**仅 91 条**），
`bs=4 / seq=340 / 2 epochs`，共 **46 步**（23 步/epoch）。
LoRA 仅作用于 q_proj / o_proj 两类方阵（每层 2 处，共 16 处）。

### 5.1 rank 扫描（learning_rate = 1e-4）

| 实验 | rank | lr | LoRA 参数 | 占比 | 末段 loss | s/step |
|---|---|---|---|---|---|---|
| E1 | 8 | 1e-4 | 0.197 M | 0.31% | 1.0520 | 1.44 |
| E0 | 16 | 1e-4 | 0.393 M | 0.61% | 0.9473 | 1.89 |
| E2 | 32 | 1e-4 | 0.786 M | 1.22% | 0.8429 | 1.50 |

### 5.2 定性验证（固定 prompt `hi`，temperature 0.6，max_new_tokens 96）

| rank | 输出摘要 | 判定 |
|---|---|---|
| 8 | 中文自我介绍「我是 MiniMind，一个高效的小参数 AI 模型…」，语义完整、无重复 | 正常 |
| 16 | 中文自我介绍，但同一句重复两遍 | 轻度退化 |
| 32 | 英文碎片（"hot-books"）反复出现，语义崩坏 | 严重退化 |

### 5.3 学习率对照（rank 固定 32，lr 1e-4 → 5e-5）

| 实验 | rank | lr | 末段 loss | 对 `hi` 的输出 | 判定 |
|---|---|---|---|---|---|
| E2 | 32 | 1e-4 | 0.8429 | 英文碎片重复循环 | 崩坏 |
| E3 | 32 | 5e-5 | 1.8510 | 「今天想聊聊最近的天气怎么样，有什么想聊的吗？」中文通顺、无重复 | 稳定，但未学到身份 |

### 5.4 结论

1. LoRA 参数量与 rank 严格线性：理论值 0.197 / 0.393 / 0.786 M，实测完全一致，
   验证了参数化补丁与「LoRA 只挂方阵」的实现假设。
2. 在 lr=1e-4 下，rank 越大训练 loss 越低，但**生成质量反而越差**：
   r32 的 loss 最低（0.8429），输出却完全崩坏。训练 loss 与生成质量出现方向背离。
3. 根因是「容量 × 步长」的组合过大：`lora_identity` 只含英文同质样本（同一句自我介绍的多种问法），
   rank 越大、增量矩阵能表达的方向越多，对基座的扰动越大，46 步内即过拟合到英文模板，
   并把基底的中文能力挤压掉；小模型在偏离分布后极易陷入 token 重复。
4. 学习率对照（E3）把原因拆开：把 lr 从 1e-4 降到 5e-5，r32 恢复中文通顺、不再重复，
   但 loss 停在 1.85、没能学到身份。说明**问题不是 rank 32 本身不可用，
   而是「大 rank + 大步长」在少量同质数据上过拟合**；降学习率能换来稳定，代价是学不到目标。
5. 因此在同一数据与步数下，**r=8 + lr=1e-4 是唯一同时满足「输出稳定」与「学到身份」的配置**；
   r=16 已出现重复退化，r=32 在中高学习率下不可用。
6. 方法学结论：**loss 不能作为 LoRA 超参的唯一选型依据**，必须配合固定 prompt 的定性检查；
   单看 loss 会选出崩坏的模型。

### 5.5 局限

- 数据仅 91 条，总步数仅 46；学习率与 cosine 调度强耦合，lr 差异同时改变了总更新量。
- 仅记录训练 loss，未划分验证集，结论只对拟合能力成立。
- 每组仅跑一次，时间数据未取多次中位数，波动约 ±20%。
- 定性检查只用了 1 个 prompt、每档 1 次采样，未做多 prompt 与多次重复。

### 5.6 换数据验证：中文身份 + 通用对话混合（392 条）

目的：验证「r32 崩坏源于英文同质数据」这一假设。**只改数据**，其余参数与前一轮完全一致。

数据构造（`tools/make_identity_zh.py` 生成）：
- 192 条中文身份样本：16 种问法 × 4 种前缀 × 3 种答案模板，名字统一为「小澜」；
- 200 条通用对话：从 `sft_sub_20k.jsonl` 抽取，并**过滤掉**含 `jingyaogong` / `MiniMind` / `Qwen` / `GPT` 的行，避免与目标身份直接冲突；
- 合计 392 条，`bs=4 / seq=340 / 2 epochs` → **196 步**。

| rank | LoRA | 末段 loss | `hi` | `你是谁？` | `推荐一些中国美食` |
|---|---|---|---|---|---|
| 8 | 0.197 M | 1.4788 | 「我是小澜，一直在学习新东西…」（略散） | **仍答「由 jingyaogong 创建」** | 给出宫保鸡丁等，基本可用 |
| 16 | 0.393 M | 1.4104 | 「我是小澜，一直在学习，但…」（语病） | 「我是小澜，也是一个小澜」（重复） | **答成身份**，未回答问题 |
| 32 | 0.786 M | 1.3228 | 「我是小澜，一直在学习中文…」 | 「我是小澜，一直在学习和运行」 | 给出通用美食建议 |

**结论（续）**

7. 英文碎片崩坏消失：此前 r32 输出 `hot-books` 循环的现象不再出现，
   说明当时的崩坏是「**英文同质数据 + 大 rank**」的组合所致，而非 rank=32 本身不可用。
8. 但出现新的失败模式——**身份答案过度泛化**：r16 连「推荐中国美食」都答「我是小澜」；
   小模型在少量数据上会把最高频答案当成万能模板。
9. r8 仍残留旧身份（「由 jingyaogong 创建」），说明 196 步不足以覆盖底座的强先验；
   身份微调需要更多步数、更大数据量或更强约束，而不是更小的 rank。
10. 三种失败模式（英文崩坏 / 重复 / 身份泛化）都没体现在训练 loss 上（1.32–1.48 区间），
    再次印证第 5.4 节第 6 条：**loss 不能作为微调效果的选型依据**。

**本节局限**

- 192 条身份样本由模板组合而成（16×4×3），句式多样性有限，容易诱发「模板化回答」；
- 未做多种子重复，单次运行的结果受采样随机性影响；
- 混合比例（192:200）未做扫描，「身份泛化」是否随比例变化仍未知。

### 5.7 步数对照：196 步 → 294 步（2 → 3 epochs）

目的：验证「旧身份残留」是步数不足还是数据不足。**只改 epochs**，数据、rank、lr 全部不变。

| rank | 196 步（2 ep）`你是谁？` | 294 步（3 ep）`你是谁？` | 变化 |
|---|---|---|---|
| 8 | 「我是小澜，也是一个小澜」+ **仍出现 jingyaogong** | 「我是小澜，一个小澜的中文小澜，只在中文语言学习…」 | 旧身份**消失**，但重复加剧 |
| 16 | 「我是小澜，也是一个小澜」 | 「我是小澜，不是 AI，也不是 人类，也不是 智能，也不是 人类」 | 重复加剧 |
| 32 | 「我是小澜，一直在学习和运行」 | 「我是小澜，一个 6400 万参数的中文小模型，在一张 AMD 显卡上运行，并在一张 MiniMind 显卡上运行」 | **最贴近目标模板**，但续写出无意义从句 |

**结论（续）**

11. 旧身份覆盖成功：294 步后 r8 不再输出「由 jingyaogong 创建」，
    说明此前残留的直接原因是**步数不足**，而非数据里缺少中文身份样本。
12. 但重复退化随步数加重：r8 出现「一个小澜的中文小澜」、r16 出现「不是 AI，也不是人类，也不是智能，也不是人类」。
    模板化数据训练越久，模型越倾向于把高频片段反复拼接，而不是生成新内容。
13. r32 在 294 步时最接近目标模板（复现了「6400 万参数 / AMD 显卡」等要素），
    但会续写出「在一张 MiniMind 显卡上运行」「CM 显卡上蹭蹭」这类无意义从句——
    说明它学到了句子骨架，但没有学到语义边界。
14. **末步 loss 不可用于横向比较**：r8 从 1.4788 升到 1.8231，r32 从 1.3228 升到 1.5480，
    但行为明显更接近目标。末步 loss 来自单个 batch，噪声极大；应以多 prompt 定性检查
    配合「多 batch 平均 loss」判断。

**本节局限**

- 单种子、单次运行；未做 early stopping 扫描（例如在 200–250 步之间取最优）。
- 训练数据为模板组合生成，句式多样性不足，是重复退化的直接诱因。
- 步数只取 196 / 294 两点，未定位「旧身份刚消失」的临界步数。

## 6. 已知问题与绕过

| 现象 | 根因 | 处理 |
|---|---|---|
| ROCm 引擎可用但仅 8 tok/s | 驱动 24.5.1 过旧，HIP 组件不匹配 | 升级至 Adrenalin 26.10.1 |
| 运行时包下载仅 40 KB/s | 全局代理访问海外 CDN | 关闭代理直连，恢复 3 MB/s |
| `rocm-smi`: amdgpu not found in modules | WSL 无内核模块，走 /dev/dxg | 改用 `rocminfo` 验收 |
| `pip install torch` 报找不到 triton | torch 钉死 `triton==3.5.1+rocm7.2.0`，PyPI 无此版本 | 本地四个 whl 一起安装（`--find-links`） |
| `ImportError: cannot import name BaseTzInfo` | pandas 与 pytz 版本不匹配 | `pip install --force-reinstall pandas==2.2.3 pytz==2024.1` |
| `ValueError: model_kwargs not used: token_type_ids` | tokenizer 返回了模型不接收的字段 | `inputs.pop("token_type_ids", None)` |
| `HSA exception: MemoryRegion::BlockAllocator::alloc failed`，严重时整个 WSL 会话被终止 | WSL 下每步微量资源泄漏，累积约 400–500 步耗尽主机可见内存池 | `PYTORCH_ALLOC_CONF=expandable_segments:True`；单进程 ≤300 步；`--from_resume 1` 分块训练 |
| 回答像「内心独白」、答非所问 | 模型先输出思考文本，token 预算不足被截断 | `--max_new_tokens ≥256`，`temperature 0.6` |

## 7. 复现步骤（最短路径）

```bash
# 1. 环境
wsl --update
sudo apt install -y curl wget git python3-venv python3-pip build-essential
sudo usermod -aG render,video $USER   # 之后 wsl --shutdown

# 2. ROCm（WSL usecase，禁止 dkms）
wget https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/amdgpu-install_7.2.70200-1_all.deb
sudo apt install -y ./amdgpu-install_7.2.70200-1_all.deb
sudo amdgpu-install -y --usecase=wsl,rocm --no-dkms
rocminfo | grep -E "gfx|Marketing"

# 3. PyTorch ROCm 轮子（必须同版本成套安装）
pip install --find-links ./wheels "torch==2.9.1+rocm7.2.0.lw.git7e1940d4" \
  "torchvision==0.24.0+rocm7.2.0.gitb919bd0c" "triton==3.5.1+rocm7.2.0.gita272dfa8"
python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 4. 模型与数据
git clone --depth 1 https://github.com/jingyaogong/minimind && cd minimind
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3
modelscope download --model gongjy/minimind-3-pytorch pretrain_768.pth --local_dir ./pth_cache
cp ./pth_cache/pretrain_768.pth ./out/pretrain_768.pth

# 5. 推理
python eval_llm.py --load_from ./minimind-3

# 6. SFT（单进程 ≤300 步，避免 WSL 资源泄漏）
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd trainer && python train_full_sft.py --epochs 1 --batch_size 4 --max_seq_len 384 \
  --learning_rate 1e-5 --data_path ../dataset/sft_t2t_mini.jsonl \
  --from_weight pretrain --save_weight full_sft_sub --num_workers 0

# 7. LoRA 单变量实验
python train_lora.py --lora_rank 16 --lora_name lora_r16 --from_weight full_sft_sub \
  --data_path ../dataset/lora_identity.jsonl --epochs 2 --batch_size 4 \
  --max_seq_len 340 --learning_rate 1e-4 --num_workers 0
```

> `train_lora.py` 与 `eval_llm.py` 需各加一个 `--lora_rank` 参数并传入
> `apply_lora(model, rank=args.lora_rank)`：上游实现中 rank 固定为 16。

## 8. 可以讲的三件事

1. **异构环境排障**：从「窗口莫名消失」定位到 `BlockAllocator::alloc failed`，
   用分步脚本（逐段打印 + 落盘日志）把故障缩到 backward 阶段，再归结为 WSL 主机内存池泄漏，
   最终用「限制单进程步数 + 分块续训」绕过，而非重装环境。
2. **依赖互锁**：AMD 的 torch 轮子把 triton 钉死为 `+rocm7.2.0` 特制版本，
   PyPI 上不存在，必须成套本地安装；这类问题不能靠「换个源」解决，只能读依赖声明。
3. **实验设计**：三组实验只改 rank，参数量先预测后验证（0.197/0.393/0.786 M 全中），
   并在结论里明确写出「数据同质导致趋势不可外推」「只有训练 loss 没有验证集」两条局限。

---

## 9. 建议的前置知识（面向已修完基础课的大学生）

假设已修：数据结构、计算机组成原理、操作系统、计算机网络、高等数学、线性代数、概率论与数理统计。
下面说明这些课在本项目里**具体用在哪**，以及还需要补什么。

### 9.1 已修课程的落点

| 课程 | 在本项目中的具体用处 |
|---|---|
| 操作系统 | WSL 的虚拟化模型、进程与会话（为什么关窗口进程就死）、用户组权限（`render`/`video`）、`/dev/dxg` 设备文件、动态库加载与 `LD_LIBRARY_PATH`、`2>&1 \| tee` 的管道与重定向 |
| 计算机组成原理 | GPU 的 CU/SIMD 结构（本卡 60 CU）、显存—主机内存层次、fp16/bf16 的位宽与精度（为什么 fp16 求和会溢出为 inf）、PCIe 与显存带宽 |
| 线性代数 | 矩阵乘（45.7 TFLOPS 基准）、注意力的 `QK^T`、LoRA 的低秩分解 `B·A`（rank 的几何含义就是增量的秩） |
| 概率论与数理统计 | softmax、temperature/top-p 采样（同一 prompt 两次回答不同的原因）、交叉熵 loss 与困惑度、训练/验证集与泛化的概念 |
| 数据结构 | JSONL 数据管道、tokenizer 的 BPE 合并、KV cache 的追加式存储、DataLoader 的批处理与打乱 |
| 计算机网络 | 本地 API 端口（1234 / 8998）、HTTP 与代理（下载 40 KB/s 的原因）、CDN 就近节点 |
| 高等数学 | 梯度与链式法则（`loss.backward()` 的数学本质）、学习率与更新步长、余弦退火调度 |

> 结论：基础课**够用**，本项目没有用到超出本科范围的理论。真正的缺口在工程侧与深度学习侧。

### 9.2 需要额外补的四块（按优先级排序）

**① Linux 与开发环境（1～2 周，收益最高）**

- 资源：MIT《The Missing Semester of Your CS Education》（missing-semester.mit.edu），看前 6 讲
- 学到什么程度：能解释 `export` 与 `~/.bashrc` 的关系、能自己建 venv、能用 `ldd` 查依赖、能读懂重定向与管道
- 对应本项目：阶段二、阶段三的全部环境搭建

**② Python 工程与 PyTorch（2～3 周）**

- 资源：PyTorch 官方教程 "Learn the Basics"；《动手学深度学习》(d2l.ai) 第 2～5 章
- 学到什么程度：能独立写出"数据加载 → 前向 → loss → 反向 → 更新"的训练循环，并能解释每一行
- 对应本项目：`train_full_sft.py`、`train_lora.py`

**③ Transformer 与 LLM 训练（3～4 周，核心）**

- 资源：Jay Alammar《The Illustrated Transformer》；Karpathy《Neural Networks: Zero to Hero》（从零手写 GPT）；李宏毅机器学习课程中"自注意力/Transformer"部分
- 学到什么程度：能画图讲清 token → embedding → attention → logits 的流动；能解释 RoPE、RMSNorm、GQA、KV cache 各解决什么问题
- 对应本项目：精读 `model/model_minimind.py`（200 行，最好的教材）

**④ GPU 异构计算与实验方法（长期，边做边补）**

- 资源：ROCm 官方文档（安装页 + 兼容性矩阵）；可选《Programming Massively Parallel Processors》了解 GPU 编程模型
- 学到什么程度：知道"驱动—运行时—框架"三层各自负责什么；遇到 `HSA` / `hipBLAS` / `MIOpen` 报错时知道该查哪一层
- 实验方法没有教材，规则只有三条：**先有基线、一次只改一个变量、每个结论都写清局限**

### 9.3 8 周自学安排（每天 1.5～2 小时）

| 周 | 任务 | 验收标准 |
|---|---|---|
| 1 | Missing Semester 前 6 讲 + 在 WSL 里练 | 不看教程完成建 venv、装包、重定向日志、查进程 |
| 2 | d2l 第 2～4 章 + PyTorch 官方 4 节 | 手写线性回归训练循环并解释每步 |
| 3 | Karpathy 前 3 讲 | 手写字符级语言模型并训练出可读输出 |
| 4 | 精读 `model_minimind.py` 并逐行注释 | 白板讲清一次前向的全部张量形状 |
| 5 | 精读 `train_full_sft.py` + `dataset/lm_dataset.py` | 说清 label 为什么 shift 一位、padding 如何处理 |
| 6 | **清空重来**：新建环境，只凭文档复现阶段二 | 独立跑通 `rocminfo` 与 `torch.cuda.is_available()` |
| 7 | 复现阶段三：推理 + 200 步 SFT | 独立定位至少 1 个报错（不求助） |
| 8 | 重做本文件的 rank 实验，写自己的结论 | 产出"结论 + 证据 + 局限"三段式报告 |

> 第 6 周是分水岭：能否脱离教程复现，决定你是"跑过"还是"会做"。

### 9.4 自检清单（能答上来即入门）

1. 为什么 WSL 里 `rocm-smi` 一定失败，而 `rocminfo` 可以？
2. `torch.cuda.is_available()` 返回 True 时，底层经过了哪几层？
3. 为什么 `pip install torch` 装不上，必须四个 whl 一起本地装？
4. LoRA 的 rank 在数学上改变了什么？为什么它只挂在方阵上？
5. 同一个问题问两次答案不同，是模型坏了还是采样导致的？
6. 训练 loss 从 0.84 降到 0.84 但输出崩坏，可能的原因有哪些？
7. 为什么 fp16 的矩阵求和会变成 `-inf`，而 fp32 不会？
8. 你的实验里，哪个结论只对训练集成立？依据是什么？

### 9.5 学习方式：三次复现法

1. **第一遍照着做**（本次已完成）：建立地图，知道每一步存在、大概耗时、会在哪摔跤。
2. **第二遍关掉教程做**：只保留本文件第 7 节的命令清单，卡住再查。这是把"看过"变成"会做"的关键一步。
3. **第三遍改一个变量做**：换数据、换 rank、换学习率，并按第 5 节的格式写结论。

三遍走完，这个项目才真正属于你；面试被追问到第三层问题时，也只有走完第三遍的人答得上来。

---

## 10. 待办（下次继续时从这里接）

- [ ] **讨论：在 MiniMind 之上做自己的增量并开源**（用户明确要求：不止于复现）
      - 候选方向：ROCm/WSL 一键适配脚本 + 基准数据 + 实验脚手架；中文身份数据集；中文教程
      - 已有素材：8 个坑的完整定位过程、rank/lr 实验数据、record_exp.py / compare.py
      - 可先做：向上游提 `--lora_rank` 参数化 PR（train_lora.py + eval_llm.py 各一处）
- [ ] 换数据验证「r32 崩坏源于数据同质」的假设（只换数据，其余参数不动）
- [ ] 把 E3 补进 experiments.csv：
      `python record_exp.py --tag e3_r32_lr5e5 --rank 32 --bs 4 --seq 340 --log logs/e3.log`
- [ ] 从本文件压一段 150 字左右的简历项目描述
- [ ] 按第 9 节走一遍「清空重来」（第 6 周任务），验证脱离教程能否独立复现

> 原则提醒：复现不丢人，但要把「我做的」和「上游的」写清楚；
> 想成为"自己的项目"，需要一个别人能直接用的产物（脚本 / 数据 / 基准 / 教程）。
