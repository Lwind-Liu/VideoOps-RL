# VideoOps-RL

面向长视频媒资的多模态、多智能体 Agentic RL 工程：用户输入自然语言需求，系统在字幕与关键帧中主动检索、核验和审计，最终提交可追溯的高光时间段。

项目与 TCL/雷鸟材料中的“长视频切分、关键帧、结构化剧情理解、内容生成与精细运营”方向对齐，但使用公开 CC BY 影片独立实现，不声称使用内部数据、服务或代码。

## GitHub 服务器一键启动

```bash
git clone --branch main https://github.com/Lwind-Liu/VideoOps-RL.git
cd VideoOps-RL
RUN_MODE=smoke bash bootstrap_server.sh
```

Smoke 通过后执行 `RUN_MODE=full bash bootstrap_server.sh`。脚本自动下载 Release 离线资产、执行双层 SHA-256 校验、解压，并根据 GPU 数量选择 8 卡或 24 卡路径。任务平台填写方式见 [GitHub 服务器快速启动](docs/GITHUB_SERVER_QUICKSTART.md)。

## 当前完成度（Algorithm v2）

- 数据：QVHighlights 10,310 个人工查询任务及公开 2 秒 CLIP 特征；另有 3 部公开视频、306 个关键帧、178 条字幕、66 个原始多模态任务；
- 算法：BM25 + OpenAI CLIP、多模态 Noisy-OR 时序证据图、自适应变量长度 proposal、语义审计、过程奖励与硬接地约束；
- 环境：文本检索、视觉检索、关键帧/特征核验、上下文扩展、证据审计、提交 6 类工具；
- Agent：固定切片、单 Agent、TimelineScout + VisionAnalyst + EvidenceAuditor + Coordinator；
- Reward：时间 IoU、证据支持、模态覆盖、审计通过、工具成本、非法调用惩罚；
- 训练数据：3,575 条成功审计 SFT train 轨迹、9,018 条无标签泄漏 GRPO train 采样记录（80% QV / 20% 原始多模态）；
- 服务器：Qwen3-VL-2B-Instruct LoRA SFT、TRL GRPO、DeepSpeed ZeRO-2，自动支持 8×H200 兼容路径和 24×H200 快速路径；
- 离线交付：视频、字幕、关键帧、训练数据、模型权重、代码、配置、报告统一打包，硬上限 50 GiB。

训练前完整 QVHighlights 评测中，自适应时序 proposal 在 val/test 达到 0.482/0.487 mIoU 和 49.9%/50.1% R@1@0.5；固定 10 秒窗口只有约 10% R@1@0.5。该指标是项目的 R@1-style temporal IoU，不是官方 mAP。H200 SFT/GRPO 尚未运行，不能声称训练后提升。

## 本地复现

```powershell
cd C:\Users\f1567\Desktop\面试\VideoOps-RL
python scripts\build_qv_query_index.py
python scripts\evaluate_algorithm_v2.py
python scripts\evaluate_pretraining_stack.py
python scripts\build_training_data.py
python -m pytest -q
python scripts\build_server_package.py
```

## 服务器训练

```bash
unzip VideoOps-RL-offline-server.zip
cd VideoOps-RL
pip install -r server/requirements-llm-grpo.txt
bash server/run_all.sh
```

服务器基础环境需要 Linux、CUDA 驱动、Python 3.11 和至少 8 张 H200；检测到 24 张时自动启用 24 卡训练与并行评测。项目数据和模型不需要在线下载。

先读：[Algorithm v2 详解](docs/ALGORITHM_V2_GUIDE.md)；再读[完整全链路教程](docs/FULL_PIPELINE_TUTORIAL.md)；机器执行清单见[服务器交付说明](docs/SERVER_PACKAGE.md)。
