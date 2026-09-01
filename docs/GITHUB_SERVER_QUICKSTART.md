# GitHub 拉取与一键训练

仓库分支 `main` 只保存代码、协议和启动器。完整 5.525 GiB 离线包以三个小于 2 GiB 的分片放在 GitHub Release `offline-v2.0.0`，也可以提前挂载为 Primus 数据集或内部 OSS 目录。启动器会自动查找或下载、断点续传、逐片校验、合并、校验完整 ZIP、解压、安装依赖并选择 8/24 卡训练路径。

## 任务平台填写

- 代码仓库：`https://github.com/Lwind-Liu/VideoOps-RL`
- 类型：分支
- Git 分支：`main`
- 下载路径：`/root/code`

进入容器后：

```bash
cd /root/code
bash bootstrap_server.sh
```

该命令会自动先运行 Smoke，成功后继续 Full，不需要执行者手动拆分阶段。

启动脚本首先审计仓库的一键执行契约，再检查 Python 3.11/3.12、CUDA PyTorch、至少 8 张可见 GPU、60 GiB 可用磁盘，以及 `curl`、`sha256sum`、`awk`、`tee`、`tar`。门禁不通过时不会开始下载大文件。依赖版本已锁定，其中 TRL 1.9.2 对应 vLLM 0.25.1；DeepSpeed 在确认镜像已有 CUDA PyTorch 后以非隔离方式安装。

执行顺序固定为：主机门禁 → 三分片断点下载 → 分片与整包 SHA-256 → 解压 → 用 Git 最新代码覆盖运行快照 → 安装依赖 → 完整 preflight → LoRA SFT → 合并模型 → 启动 vLLM → Agentic GRPO → 关闭 rollout 服务 → val/test 评测。

只检查当前 Git checkout 是否完整，不下载资产、不需要 GPU：

```bash
python server/audit_one_click_contract.py
```

如平台提供启动命令栏，可直接填写后一条命令。下载支持续传，重复执行不会重新下载已经通过 SHA-256 校验的分片。

## Primus 数据集或内部 OSS

如果容器能拉取 Git 仓库但下载 GitHub Release 大文件超时，把下面三个分片上传为 Primus 数据集或内部 OSS 目录：

```text
VideoOps-RL-offline-server.zip.part-00
VideoOps-RL-offline-server.zip.part-01
VideoOps-RL-offline-server.zip.part-02
```

挂载目录如果是 `/root/code/offline_assets`、`/root/input`、`/input`、`/mnt/data`、`/mnt/oss`、`/root/oss` 或 `/dataset` 的本层或一层子目录，启动命令不用改：

```bash
cd /root/code
bash bootstrap_server.sh
```

如果挂载路径特殊，显式指定目录：

```bash
cd /root/code
VIDEOOPS_ASSET_DIR=/path/to/offline_assets bash bootstrap_server.sh
```

如果三片放在内部 HTTP/OSS 地址，显式指定 URL 前缀：

```bash
cd /root/code
VIDEOOPS_BASE_URL=https://internal.example/offline-v2.0.0 bash bootstrap_server.sh
```

只下载和准备资产、不开始训练：

```bash
PREPARE_ONLY=1 bash bootstrap_server.sh
```

若镜像已经安装全部 Python 依赖，可跳过 pip：

```bash
INSTALL_DEPS=0 bash bootstrap_server.sh
```
