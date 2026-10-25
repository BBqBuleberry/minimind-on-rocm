# Benchmarks

原始数据，均为本机实测（Radeon RX 7800 XT 16 GB，WSL2 + ROCm 7.2）。

- `inference_7800xt.json` — 推理与矩阵乘基准
- `lora_runs_7800xt.json` — LoRA 训练日志解析结果（loss、步数、参数量）

## 测量方法

- 推理 tok/s：`eval_llm.py` 内置统计（生成 token 数 / 生成耗时），每档取预热后的第二次问答；
- TFLOPS：fp16 `4096x4096` 矩阵乘 20 次取平均，`2*N^3/t`；
- 训练耗时：日志首末时间戳差 / 步数；
- 显存：任务管理器「专用 GPU 内存」峰值（WSL 中 `rocm-smi` 不可用）。

## 注意

- 单次测量，未取多次中位数，波动约 ±20%；
- 训练 loss 为逐 batch 值，噪声大，横向比较请用 `last3_mean` 而非 `final_loss`。
