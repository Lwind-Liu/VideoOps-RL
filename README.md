# VideoOps-RL

面向长视频媒资理解的多模态、多 Agent、Agentic RL 工程。`main` 分支保存代码、协议和启动器，GitHub Release 保存数据、模型权重等大文件；两者由启动脚本自动组装为完整训练环境。

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

启动后无需再输入其他训练命令。脚本会自动查找或下载并校验离线资产、安装依赖、选择 8 卡或24 卡路径、完成 Smoke 和 Full，并保存 checkpoint、日志和评测报告。

机器需要满足：Linux、Python 3.11/3.12、可用的 CUDA PyTorch、至少 8 张可见 GPU、至少 60 GiB 可用磁盘，并且能够访问 GitHub Release 和 Python 包索引。启动器会先审计仓库契约和主机条件，未通过时不会下载大文件。

Primus 等平台如果能拉代码但不能下载 GitHub Release 大文件，请把 Release 的三个分片作为数据集或 OSS 目录挂载到容器。脚本会自动扫描 `/root/code/offline_assets`、`/root/input`、`/input`、`/mnt/data`、`/mnt/oss`、`/root/oss` 和 `/dataset` 的本层及一层子目录；找到三片并通过 SHA-256 校验后，会直接使用本地资产，不再访问 GitHub Release。若平台挂载路径特殊，启动命令写成：

```bash
cd /root/code && VIDEOOPS_ASSET_DIR=/path/to/offline_assets bash bootstrap_server.sh
```

若三片转存为内部 HTTP/OSS 地址，保证 URL 目录下直接可访问 `VideoOps-RL-offline-server.zip.part-00/01/02`，然后写成：

```bash
cd /root/code && VIDEOOPS_BASE_URL=https://internal.example/offline-v2.0.0 bash bootstrap_server.sh
```

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

- [业务等价工具、Agentic RL 必要性与训练信号](docs/AGENTIC_RL_REPRODUCTION.md)
- [服务器执行与资产说明](docs/SERVER_PACKAGE.md)
- [Algorithm v2 详解](docs/ALGORITHM_V2_GUIDE.md)
- [完整全链路教程](docs/FULL_PIPELINE_TUTORIAL.md)
- [离线训练包 Release](https://github.com/Lwind-Liu/VideoOps-RL/releases/tag/offline-v2.0.0)
- [训练数据质量审计](data/training/training_data_audit_v2.json)

服务器没有权重、希望先从本地上传时，先在 Windows 运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/prepare_offline_assets.ps1
```

上传仓库和生成的 `offline_assets/` 后，Bootstrap 会优先使用本地分片。训练会额外保存 SFT/GRPO 指标、逐工具调用 trace 和训练信号门禁报告。
