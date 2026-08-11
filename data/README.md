# 数据目录

真实数据不直接提交到仓库。建议按以下结构保存本地缓存：

```text
data/
  raw/              # 原始视频或官方下载缓存
  annotations/      # 数据集标注
  processed/        # shot/ASR/frame/embedding 中间结果
  trajectories/     # Agent 轨迹和 reward
  manifests/        # 数据版本、split、hash、下载记录
```

第一阶段优先接入 QVHighlights；随后加入 ActivityNet Captions 和 NExT-QA。所有实验必须记录视频 split，不能把同一视频的不同问题随机分到训练和测试中。

当前首个真实媒体样本为 `Tears of Steel`。来源、许可证、hash、stream 和字幕质量记录在 `manifests/tears_of_steel_source.json`；原片目录被 `.gitignore` 排除，不能提交 372 MB 二进制文件。
