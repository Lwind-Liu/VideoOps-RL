# VideoOps-RL

VideoOps-RL 是一个面向长视频媒资理解的多模态、多 Agent、Agentic RL 工程。模型根据自然语言需求，主动调用字幕检索、视觉检索、关键帧核验、上下文扩展和证据审计工具，最终提交可追溯的高光时间段。

项目与 TCL/雷鸟材料中的长视频切分、关键帧、结构化剧情理解、内容生成和精细运营方向对齐，但完全使用公开数据独立实现，不声称使用任何内部数据、服务或代码。

> 当前状态：数据、算法、训练脚本、8/24 卡调度和离线资产已完成并通过本地审计；H200 上的 SFT/GRPO 尚未实际运行。请先执行 Smoke，成功后再执行 Full。训练完成前不能声称 RL 已带来指标提升。

## 给服务器执行者：从这里开始

### 1. 机器要求

- Linux；
- Python 3.11 或 3.12；
- 基础镜像已经安装可用的 CUDA PyTorch；
- 至少 8 张可见 CUDA GPU，推荐 8×H200 或 24×H200；
- 至少 20 GiB 可用磁盘，建议预留 50 GiB；
- 能访问 GitHub Release 和 Python 包索引，或提供内部镜像地址；
- 系统命令包含 `bash`、`curl`、`sha256sum` 和 `tar`。

启动器会在下载 5.5 GiB 资产前检查 Python、CUDA、GPU 数量和磁盘空间，不满足要求会直接停止。

### 2. 任务平台填写

- 代码仓库：`https://github.com/Lwind-Liu/VideoOps-RL`
- 类型：分支
- Git 分支：`main`
- 加载路径：`/root/code`

如果平台只允许阿里内部 GitLab，请将本仓库镜像到内部 GitLab，保持 `main` 分支和目录结构不变。

### 3. 第一次只跑 Smoke

任务平台已经自动将代码放到 `/root/code` 时：

```bash
cd /root/code
RUN_MODE=smoke bash bootstrap_server.sh
```

普通服务器尚未拉取代码时：

```bash
git clone --branch main https://github.com/Lwind-Liu/VideoOps-RL.git
cd VideoOps-RL
RUN_MODE=smoke bash bootstrap_server.sh
```

这条命令会自动完成：主机门禁、断点下载、双层 SHA-256 校验、解压、代码更新、依赖安装、完整 preflight、0.1 epoch SFT、模型合并、vLLM 启动、20-step Agentic GRPO 和48 条任务评测。不要跳过 Smoke 直接跑 Full。

Smoke 的 Go 条件：

- `server/preflight.py` 输出中的 `passed` 为 `true`；
- SFT 能产生 loss，且没有 OOM/NaN；
- `artifacts/sft_qwen3vl2b/` 和 `artifacts/sft_qwen3vl2b_merged/` 已生成；
- vLLM health check 通过；
- GRPO completion 中出现合法工具调用，组内 reward 不是恒定值；
- `artifacts/grpo_qwen3vl2b/` 已生成；
- val/test 评测报告已生成；
- `outputs/run_all.log` 最后没有 traceback、OOM 或子进程失败。

出现以下任意情况时不要继续 Full：工具 JSON 大量解析失败、reward 恒定、OOM、NaN、vLLM 无法启动、训练与 rollout GPU 重叠、评测报告缺失或日志中出现未处理异常。

### 4. Smoke 通过后跑 Full

```bash
cd /root/code
RUN_MODE=full bash bootstrap_server.sh
```

启动器支持重复执行：已经通过校验的下载分片不会重新下载，代码会使用 Git 仓库中的最新版本覆盖离线包内的旧代码快照。

如果服务器不能访问 GitHub Release，请先将三个 Release 分片转存到同一个内部 HTTP 目录，然后执行：

```bash
VIDEOOPS_BASE_URL="https://内部地址/VideoOps-RL/offline-v2.0.0" \
RUN_MODE=smoke bash bootstrap_server.sh
```

如果基础镜像已经安装 README 指定的全部依赖，可以设置 `INSTALL_DEPS=0`；否则不要设置。

## 一键命令内部执行顺序

```text
Git clone main
  → Python/CUDA/GPU/磁盘门禁
  → 下载 3 个 Release 分片（支持断点续传）
  → 逐分片 SHA-256
  → 拼接完整 ZIP 并再次 SHA-256
  → 解压模型、数据、特征和代码
  → 用 Git 最新代码覆盖运行快照
  → 安装锁定依赖和本项目
  → 完整 preflight
  → Qwen3-VL-2B LoRA SFT
  → 合并 LoRA 与基础模型
  → 启动并检查 vLLM rollout 服务
  → 多轮工具环境 Agentic GRPO
  → 关闭 vLLM
  → held-out val/test 评测
  → 保存日志、checkpoint 和报告
```

24 卡路径：24 卡 SFT；GPU 0 合并；GPU 20–23 运行 TP=4 vLLM；GPU 0–19 运行 GRPO；最后 val/test 各使用 12 卡并行评测。

8 卡路径：8 卡 SFT；GPU 0 合并；GPU 6–7 运行 TP=2 vLLM；GPU 0–5 运行 GRPO；最后在 GPU 0 顺序评测 val/test。

## 跑完后必须回传什么

无论成功还是失败，都先执行：

```bash
cd /root/code
bash server/collect_run_bundle.sh
```

脚本会生成：

```text
outputs/handoff/VideoOps-RL-run-report-<UTC时间>.tar.gz
outputs/handoff/VideoOps-RL-run-report-<UTC时间>.tar.gz.sha256
```

压缩包只包含机器环境、Git 版本、依赖版本、日志、评测报告、checkpoint 文件清单和 checkpoint SHA-256，不包含数 GB 模型权重。计算 checkpoint SHA-256 可能需要几分钟，请等待脚本结束。

请回传以下内容：

1. `VideoOps-RL-run-report-*.tar.gz`；
2. 对应的 `.sha256` 文件；
3. 完整 `outputs/run_all.log`；
4. 完整 `.videoops-bootstrap/bootstrap.log`；
5. 完整 `outputs/vllm.log`；
6. `outputs/reports/checkpoint_all_val_eval.json`；
7. `outputs/reports/checkpoint_all_test_eval.json`；
8. SFT 和 GRPO checkpoint 的服务器路径或对象存储链接；
9. checkpoint 文件的 SHA-256；
10. 实际执行命令、开始/结束时间和使用的 GPU 数量；
11. 如果失败，附上失败阶段、第一条异常和日志最后 200 行，不要只发截图。

可直接复制下面的回传模板：

```text
VideoOps-RL 运行结果
- Git commit:
- 运行模式: smoke / full
- 机器: GPU 型号 × 数量
- 实际命令:
- 开始时间:
- 结束时间:
- 总状态: success / failed
- 失败阶段（成功则填 none）:
- SFT checkpoint:
- GRPO checkpoint:
- checkpoint SHA-256:
- val: mIoU / success_rate / mean_reward / parse_error_rate
- test: mIoU / success_rate / mean_reward / parse_error_rate
- 回传压缩包:
- 其他异常:
```

如果任务失败，也必须运行收集脚本并回传压缩包。不要自行删除失败现场、覆盖日志或直接重跑 Full。

## 主要输出目录

- `artifacts/sft_qwen3vl2b/`：SFT LoRA adapter；
- `artifacts/sft_qwen3vl2b_merged/`：合并后的完整 SFT 模型，供 vLLM 和 GRPO 使用；
- `artifacts/grpo_qwen3vl2b/`：GRPO checkpoint；
- `outputs/run_all.log`：一键链路总日志；
- `.videoops-bootstrap/bootstrap.log`：从主机门禁、下载到依赖安装的完整启动日志；
- `outputs/vllm.log`：rollout 服务日志；
- `outputs/reports/checkpoint_all_val_eval.json`：验证集评测；
- `outputs/reports/checkpoint_all_test_eval.json`：测试集评测；
- `outputs/eval_*_shard_*.log`：24 卡分片评测日志；
- `outputs/handoff/`：可回传的小型运行包。

## 数据与算法

数据包括 QVHighlights 10,310 条人工查询任务和 12,562 个公开 2 秒 CLIP 特征文件，以及 3 部 CC BY 开放影片、306 个关键帧、178 条字幕和66 条原始多模态任务。train/val/test 的任务和视频交集均为 0。

训练数据包括 3,575 条成功且审计通过的 SFT train 轨迹，以及 9,018 条无标签泄漏的 GRPO train 记录，其中 7,218 条来自 QVHighlights，1,800 条来自开放影片多模态任务，比例约为 80%/20%。GRPO prompt 不包含目标时间段、teacher reward 或 saliency 标签。

算法由 BM25 字幕检索、OpenAI CLIP 视觉检索、多模态 Noisy-OR 时序证据图、自适应变量长度 proposal、硬接地约束和语义审计组成。工具环境包含 `search_transcript`、`search_visual`、`inspect_keyframe`、`expand_context`、`request_audit` 和 `submit`。TimelineScout、VisionAnalyst、EvidenceAuditor 和 Coordinator 是具有明确权限边界的功能 Agent，不应描述成四个独立大模型服务。

Reward 同时考虑 temporal IoU、shot set、语义审计、模态覆盖、过程质量、saliency，以及工具调用、重复调用和非法调用成本。训练顺序固定为成功轨迹 LoRA SFT，再以同一任务的多条 rollout 做 GRPO 组内相对优化。

训练前完整 QVHighlights 评测中，自适应 proposal 在 val/test 达到 0.482/0.487 mIoU 和49.9%/50.1% R@1@0.5；固定 10 秒窗口约为 10% R@1@0.5。这里是项目定义的 R@1-style temporal IoU，不是 QVHighlights 官方 mAP。

## 已验证与待验证边界

已经验证：43 个本地测试通过；离线 ZIP 内 13,022 个文件逐文件 SHA-256 一致；模型、CLIP、12,562 个特征文件、训练数据和脚本完整；SFT train/val/test 共5,133 条轨迹可转换为原生 `tool_calls + tools schema`；Transformers 5.15 的 Qwen3-VL Processor 能正确渲染工具轨迹；TRL 1.9.2、Transformers 5.15.0、vLLM 0.25.1 和 DeepSpeed 0.19.5 的接口与约束已对齐。

尚未验证：目标 H200 镜像的 CUDA kernel 兼容性、实际峰值显存与吞吐、SFT 后工具合法率、GRPO 组内 reward 方差、训练后相对 SFT 的增益。以上必须以服务器日志和 checkpoint 评测为准。

## 本地开发与文档

```powershell
cd C:\Users\f1567\Desktop\面试\VideoOps-RL
$env:PYTHONPATH=(Resolve-Path src).Path
python -m pytest -q
```

- [GitHub/服务器快速启动](docs/GITHUB_SERVER_QUICKSTART.md)
- [服务器离线包说明](docs/SERVER_PACKAGE.md)
- [Algorithm v2 详解](docs/ALGORITHM_V2_GUIDE.md)
- [完整全链路教程](docs/FULL_PIPELINE_TUTORIAL.md)

GitHub Release：[`offline-v2.0.0`](https://github.com/Lwind-Liu/VideoOps-RL/releases/tag/offline-v2.0.0)。
