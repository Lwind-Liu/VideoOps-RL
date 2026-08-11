# GitHub 拉取与一键训练

仓库分支 `main` 只保存代码、协议和启动器。完整 5.525 GiB 离线包以三个小于 2 GiB 的分片放在 GitHub Release `offline-v2.0.0`，启动器会自动下载、断点续传、逐片校验、合并、校验完整 ZIP、解压、安装依赖并选择 8/24 卡训练路径。

## 任务平台填写

- 代码仓库：`https://github.com/Lwind-Liu/VideoOps-RL`
- 类型：分支
- Git 分支：`main`
- 下载路径：`/root/code`

进入容器后：

```bash
cd /root/code
RUN_MODE=smoke bash bootstrap_server.sh
```

Smoke 通过后：

```bash
RUN_MODE=full bash bootstrap_server.sh
```

启动脚本首先检查 Python 3.11/3.12、CUDA PyTorch、至少 8 张可见 GPU、20 GiB 可用磁盘，以及 `curl`/`sha256sum`。门禁不通过时不会开始下载大文件。依赖版本已锁定，其中 TRL 1.9.2 对应 vLLM 0.25.1；DeepSpeed 在确认镜像已有 CUDA PyTorch 后以非隔离方式安装。

执行顺序固定为：主机门禁 → 三分片断点下载 → 分片与整包 SHA-256 → 解压 → 用 Git 最新代码覆盖运行快照 → 安装依赖 → 完整 preflight → LoRA SFT → 合并模型 → 启动 vLLM → Agentic GRPO → 关闭 rollout 服务 → val/test 评测。

如平台提供启动命令栏，可直接填写后一条命令。下载支持续传，重复执行不会重新下载已经通过 SHA-256 校验的分片。

只下载和准备资产、不开始训练：

```bash
PREPARE_ONLY=1 bash bootstrap_server.sh
```

若镜像已经安装全部 Python 依赖，可跳过 pip：

```bash
INSTALL_DEPS=0 RUN_MODE=full bash bootstrap_server.sh
```
