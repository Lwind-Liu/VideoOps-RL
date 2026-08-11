# S1.5 统一媒体目录与完整性审计

## `media.json` 是什么

它不是把所有镜头和字幕再复制一遍，而是后续 Agent 打开一个视频样本时使用的统一入口。它记录：

- 样本 ID、父素材 ID 和统一时间轴；
- 视频、音频、字幕的路径、大小与 SHA-256；
- `shots.jsonl`、`utterances.jsonl`、`evidence_units.jsonl` 和关键帧目录；
- 镜头、台词、证据单元和关键帧数量；
- 镜头算法、阈值、关键帧策略和对齐规则。

## 为什么还需要审计

“文件存在”不代表数据可用。例如：关键帧可能少一张，字幕引用可能指向不存在的 ID，两个镜头之间可能有 1 ms 空洞，或者重跑脚本后媒体文件已变化但旧 manifest 没更新。

本次审计覆盖四层：

1. 媒体层：时长、stream 类型、16 kHz 单声道 ASR 音频；
2. 文件层：核心文件哈希与冻结 manifest 一致；
3. 时间层：98 个镜头连续覆盖 0--480,000 ms，字幕全部在范围内；
4. 引用层：证据单元与镜头逐一对应，台词 ID 和重叠毫秒精确一致。

## 复现命令

```powershell
python scripts/build_media_catalog.py `
  --sample-dir data/processed/tears_of_steel/s1 `
  --shot-dir data/processed/tears_of_steel/s1/shots_ffmpeg_t035 `
  --manifest data/manifests/tears_of_steel_s1.json
```

只有所有检查通过，S1 数据才允许进入 S2 Agent baseline。
