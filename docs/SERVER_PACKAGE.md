# 离线服务器交付说明

最终交付物是 `dist/VideoOps-RL-offline-server.zip`。它包含三部开放影片及关键帧、QVHighlights 10,310 条标注与 12,562 个 CLIP 特征文件、查询向量、v2 SFT/GRPO 数据、Qwen3-VL-2B-Instruct 和 OpenAI CLIP 权重、代码、配置和本地报告。11.84 GB 的原始特征 tar 不重复装包，只装入实际运行需要的约 0.9 GiB CLIP 子集。

服务器仍需预装 Linux、NVIDIA 驱动/CUDA、Python 3.11 和 PyTorch CUDA。`server/requirements-llm-grpo.txt` 是 Python 训练栈清单；如果服务器完全不能联网，应在上传前根据服务器的 CUDA/Python 版本另行制作 wheelhouse，因为 PyTorch wheel 与服务器环境强绑定，本项目不能在未知环境下提前选一个必然兼容的 wheel。

## 上传前

```powershell
python -m pytest -q
python scripts\build_server_package.py
Get-FileHash dist\VideoOps-RL-offline-server.zip -Algorithm SHA256
```

打包器会拒绝不完整的模型，检查模型文件准确大小和 safetensors 头，计算每个文件 SHA-256，并用 `ZipFile.testzip()` 检查归档损坏。硬上限是 50 GiB。

## 服务器上

```bash
unzip VideoOps-RL-offline-server.zip
cd VideoOps-RL
pip install -r server/requirements-llm-grpo.txt
python server/preflight.py
bash server/run_all.sh
```

`run_all.sh` 会自动检测 GPU 数量。8 卡机器走兼容路径；24 卡机器自动调用 `run_all_24gpu.sh`：SFT 使用全部 24 卡，GRPO 使用 GPU 0--19 训练、GPU 20--23 运行 TP=4 的 vLLM，评测阶段 val/test 各用 12 卡并行分片。任何一步失败都会停止，不会静默进入下一阶段。

24 卡首次 smoke：

```bash
SFT_EPOCHS=0.1 GRPO_STEPS=20 EVAL_TASKS=48 bash server/run_all.sh
```

通过后运行默认完整配置：

```bash
bash server/run_all.sh
```

## 预期产物

- `artifacts/sft_qwen3vl2b/`：SFT LoRA checkpoint；
- `artifacts/sft_qwen3vl2b_merged/`：合并 LoRA 后、供 vLLM 和 GRPO 使用的完整 SFT 模型；
- `artifacts/grpo_qwen3vl2b/`：GRPO checkpoint；
- `outputs/vllm.log`：rollout 服务日志；
- Trainer 日志中的 reward、KL、completion length、tool-use 轨迹。

Go 条件：目标模式所需的 8 或 24 卡可见、Qwen/CLIP 权重完整、QV 特征文件不少于 10,000、v2 train 数据可读、SFT loss 有下降且样例能生成合法工具 JSON。No-go 条件：工具 JSON 大量解析失败、组内 reward 恒定、OOM、vLLM health check 失败、rollout 服务与 trainer 使用了重叠 GPU，或验证数据被训练脚本读取。
