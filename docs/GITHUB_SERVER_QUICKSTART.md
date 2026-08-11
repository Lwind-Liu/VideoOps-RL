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

如平台提供启动命令栏，可直接填写后一条命令。下载支持续传，重复执行不会重新下载已经通过 SHA-256 校验的分片。

只下载和准备资产、不开始训练：

```bash
PREPARE_ONLY=1 bash bootstrap_server.sh
```

若镜像已经安装全部 Python 依赖，可跳过 pip：

```bash
INSTALL_DEPS=0 RUN_MODE=full bash bootstrap_server.sh
```
