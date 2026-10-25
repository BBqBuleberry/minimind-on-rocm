# 已知问题与处理（ROCm / WSL 特有）

> 按出现频率排序。每条都给出「现象 → 根因 → 处理」，方便直接搜索定位。

| # | 现象 | 根因 | 处理 |
|---|---|---|---|
| 1 | LM Studio 显示 ROCm 引擎可用，但只有 8 tok/s | 驱动过旧（如 24.5.1），HIP 运行库不匹配，静默退回 CPU | 升级 Adrenalin 到 25.x/26.x 后重测 |
| 2 | ROCm 运行包/模型下载只有 40 KB/s | 全局代理访问海外 CDN 反而更慢 | 关闭全局代理直连，实测回到 3 MB/s |
| 3 | `rocm-smi` 报 `amdgpu not found in modules` | WSL 无 amdgpu 内核模块，显卡经 `/dev/dxg` 透传 | 属正常，改用 `rocminfo` 验收 |
| 4 | `pip install torch` 报找不到 triton | AMD 轮子把 `triton` 钉死为 `+rocm7.2.0` 特制版，PyPI 无此版本 | 本地三个 whl 一起装（`--find-links`） |
| 5 | `ImportError: cannot import name BaseTzInfo` | pandas 与 pytz 版本不匹配 | `pip install --force-reinstall pandas==2.2.3 pytz==2024.1` |
| 6 | `ValueError: model_kwargs not used: token_type_ids` | tokenizer 返回了模型不接收的字段 | `inputs.pop("token_type_ids", None)` |
| 7 | `HSA exception: MemoryRegion::BlockAllocator::alloc failed`，严重时整个 WSL 会话被终止（窗口消失） | WSL 下每步有微量资源泄漏，约 400–500 步后耗尽主机可见内存池 | 设 `PYTORCH_ALLOC_CONF=expandable_segments:True`；单进程 ≤300 步；`--from_resume 1` 分块续训；提高 WSL 内存（`.wslconfig`） |
| 8 | 回答像「内心独白」、答非所问 | 模型先输出 `<think>` 规划文本，`max_new_tokens` 太小被截断 | `--max_new_tokens >= 256`，`temperature 0.6` |

## 排查方法（可复用）

1. **分步打印 + 日志落盘**：把「加载 → 前向 → 反向 → 更新」拆开逐步打印，日志用 `tee` 保存。
   会话被硬杀时，日志最后一行就是死因。
2. **外层 shell 标记**：`{ echo START; python -u train.py; echo "EXIT=$?"; } >> log 2>&1`，
   有 `EXIT` 说明进程正常退出（Python 异常），没有则说明被外部杀掉（驱动/内存）。
3. **区分进程级异常与设备级故障**：Python 报错会打印 traceback；设备故障只会让会话消失。

## WSL 内存配置

`C:\Users\<你>\.wslconfig`：

```ini
[wsl2]
memory=24GB
swap=8GB
```

改完 `wsl --shutdown` 生效。物理内存 32 GB 时，默认只分 50% 给 WSL，容易在训练时吃紧。
