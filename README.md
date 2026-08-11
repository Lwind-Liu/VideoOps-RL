# VideoOps-RL

面向长视频媒资理解的多模态、多 Agent、Agentic RL 工程。项目已经包含代码、数据、模型权重和训练配置；执行者的任务只是启动整套训练，并在结束后回传结果。

> 执行者不要手动逐个运行 SFT、vLLM、GRPO 或评测脚本。只运行下面的一条启动命令，脚本会自动先做 Smoke，Smoke 成功后继续 Full；任何阶段失败都会停止。

## 执行者只需要做两步

### 第一步：启动

下面两种情况二选一，不要依次执行。

如果任务平台已经自动把仓库加载到 `/root/code`：

```bash
cd /root/code && bash bootstrap_server.sh
```

如果是普通服务器，尚未拉取仓库：

```bash
git clone --branch main https://github.com/Lwind-Liu/VideoOps-RL.git
cd VideoOps-RL && bash bootstrap_server.sh
```

启动后无需再输入其他训练命令。脚本会自动下载并校验离线资产、安装依赖、选择 8 卡或24 卡路径、完成 Smoke 和 Full，并保存 checkpoint、日志和评测报告。

机器需要满足：Linux、Python 3.11/3.12、可用的 CUDA PyTorch、至少 8 张可见 GPU、至少20 GiB 可用磁盘，并且能够访问 GitHub Release 和 Python 包索引。条件不满足时脚本会在大文件下载前直接报错。

### 第二步：结束后打包回传

无论训练成功还是失败，都回到 Git 仓库目录执行：

```bash
cd /root/code && bash server/collect_run_bundle.sh
```

普通服务器如果仓库不在 `/root/code`，就在实际的 `VideoOps-RL` 仓库目录运行同一条脚本。

脚本会生成：

```text
outputs/handoff/VideoOps-RL-run-report-<UTC时间>.tar.gz
outputs/handoff/VideoOps-RL-run-report-<UTC时间>.tar.gz.sha256
```

请回传：

1. 上述 `.tar.gz`；
2. 上述 `.tar.gz.sha256`；
3. SFT 和 GRPO checkpoint 的服务器路径或对象存储链接。

压缩包已经包含 Git commit、机器与 GPU 信息、依赖版本、完整启动日志、训练日志、vLLM 日志、val/test 评测报告、checkpoint 文件清单及 checkpoint SHA-256，不需要再手动挑选日志。

同时附上这一段文字：

```text
VideoOps-RL 运行结果
- 状态: success / failed
- 运行机器: GPU 型号 × 数量
- 开始时间:
- 结束时间:
- 结果压缩包:
- SFT checkpoint:
- GRPO checkpoint:
- 如果失败，失败阶段和第一条异常:
```

如果失败，不要删除现场、不要覆盖日志，也不要自行改代码后重跑；先生成并回传结果包。

## 任务平台填写

- 代码仓库：`https://github.com/Lwind-Liu/VideoOps-RL`
- 类型：分支
- Git 分支：`main`
- 加载路径：`/root/code`
- 启动命令：`cd /root/code && bash bootstrap_server.sh`

如果平台只能读取阿里内部 GitLab，请把本仓库镜像到内部 GitLab，保持 `main` 分支和目录结构不变。服务器如果不能访问 GitHub Release，还需要把 Release 分片转存到内部 HTTP 地址，并设置 `VIDEOOPS_BASE_URL`。

## 项目说明（执行者无需操作）

项目使用 QVHighlights 10,310 条人工查询、12,562 个公开 CLIP 特征文件，以及3 部 CC BY 开放影片、306 个关键帧、178 条字幕和66条原始多模态任务。训练数据包括3,575 条成功审计 SFT train 轨迹和9,018 条无标签泄漏 GRPO train 记录。

算法包含 BM25 字幕检索、OpenAI CLIP 视觉检索、多模态 Noisy-OR 时序证据图、自适应时间 proposal、硬接地约束、证据审计和多目标过程奖励。TimelineScout、VisionAnalyst、EvidenceAuditor 和 Coordinator 是有明确权限边界的功能 Agent，不是四个独立大模型服务。

训练前完整 QVHighlights 评测中，自适应 proposal 在 val/test 达到0.482/0.487 mIoU 和49.9%/50.1% R@1@0.5。H200 上的 SFT/GRPO 尚未实际完成，因此训练后提升必须以本次服务器日志和 checkpoint 评测为准。

详细技术文档：

- [服务器执行与资产说明](docs/SERVER_PACKAGE.md)
- [Algorithm v2 详解](docs/ALGORITHM_V2_GUIDE.md)
- [完整全链路教程](docs/FULL_PIPELINE_TUTORIAL.md)
- [离线训练包 Release](https://github.com/Lwind-Liu/VideoOps-RL/releases/tag/offline-v2.0.0)
