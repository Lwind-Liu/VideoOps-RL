# 离线服务器交付说明

最终交付物是 `dist/VideoOps-RL-offline-server.zip`。它包含三部开放影片及关键帧、QVHighlights 10,310 条标注与 12,562 个 CLIP 特征文件、查询向量、v2 SFT/GRPO 数据、Qwen3-VL-2B-Instruct 和 OpenAI CLIP 权重、代码、配置和本地报告。11.84 GB 的原始特征 tar 不重复装包，只装入实际运行需要的约 0.9 GiB CLIP 子集。

服务器仍需预装 Linux、NVIDIA 驱动/CUDA、Python 3.11 和 PyTorch CUDA。`server/requirements-llm-grpo.txt` 是 Python 训练栈清单；如果服务器完全不能联网，应在上传前根据服务器的 CUDA/Python 版本另行制作 wheelhouse，因为 PyTorch wheel 与服务器环境强绑定，本项目不能在未知环境下提前选一个必然兼容的 wheel。

## 上传前（推荐路径）

```powershell
powershell -ExecutionPolicy Bypass -File scripts\prepare_offline_assets.ps1
```

脚本把 GitHub Release 的三个可续传分片下载到 `offline_assets/` 并逐个校验 SHA-256。将仓库连同该目录上传服务器；Bootstrap 会优先使用上传资产，缺失时才联网下载。原有 `scripts/build_server_package.py` 仍可用于从本机完整资产重建单个 ZIP，但周末复现不需要重复打包。

## 服务器上

```bash
cd VideoOps-RL
RUN_MODE=smoke bash bootstrap_server.sh
```

`run_all.sh` 会自动检测 GPU 数量。8 卡机器走兼容路径；24 卡机器自动调用 `run_all_24gpu.sh`：SFT 使用全部 24 卡，GRPO 使用 GPU 0--19 训练、GPU 20--23 运行 TP=4 的 vLLM，评测阶段 val/test 各用 12 卡并行分片。任何一步失败都会停止，不会静默进入下一阶段。

Smoke 通过且 `training_signal_smoke_*.json` 的门禁均为 true 后，再运行：

```bash
RUN_MODE=full bash bootstrap_server.sh
bash server/collect_run_bundle.sh
```

## 预期产物

- `artifacts/sft_qwen3vl2b/`：SFT LoRA checkpoint；
- `artifacts/sft_qwen3vl2b_merged/`：合并 LoRA 后、供 vLLM 和 GRPO 使用的完整 SFT 模型；
- `artifacts/grpo_qwen3vl2b/`：GRPO checkpoint；
- `outputs/vllm.log`：rollout 服务日志；
- `outputs/metrics/`：SFT/GRPO 每步 Trainer 指标 JSONL；
- `outputs/traces/`：带 request、成本、延迟、错误和状态哈希的逐工具轨迹；
- `outputs/reports/training_signal_*.json`：reward 方差、提交率、调用数和 zero-advantage 近似审计；
- `outputs/handoff/`：日志、指标、报告、压缩 trace 和 checkpoint 哈希回传包。

Go 条件：目标模式所需的 8 或 24 卡可见、Qwen/CLIP 权重完整、QV 特征文件不少于 10,000、v2 train 数据可读、SFT loss 有下降且样例能生成合法工具 JSON。No-go 条件：工具 JSON 大量解析失败、组内 reward 恒定、OOM、vLLM health check 失败、rollout 服务与 trainer 使用了重叠 GPU，或验证数据被训练脚本读取。
