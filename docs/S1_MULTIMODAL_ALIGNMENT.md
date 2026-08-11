# S1.4 字幕与镜头的多模态对齐

## 问题、输入和输出

字幕按“句子时间段”组织，画面按“镜头时间段”组织，两者的边界通常不同。本阶段用统一的毫秒时间轴建立二者关系。

输入：

- `shots.jsonl`：镜头起止时间和关键帧；
- `subtitles_en.srt`：字幕起止时间和文本。

输出：

- `utterances.jsonl`：标准化台词记录；
- `evidence_units.jsonl`：逐镜头的关键帧、台词引用和检索文本；
- `alignment_summary.json`：覆盖率与跨边界统计。

## 对齐规则

所有区间使用半开形式 `[start_ms, end_ms)`。镜头和字幕的交集长度大于 0 时才建立关系：

```text
overlap_start = max(shot_start, utterance_start)
overlap_end   = min(shot_end, utterance_end)
有效关系条件：overlap_start < overlap_end
```

因此，一句字幕如果跨过剪辑点，会同时引用到前后镜头；如果字幕恰好在新镜头起点开始，则只属于新镜头。

## 真实边界案例

`utt_0007` 的时间是 18,000—21,000 ms，镜头边界位于 20,250 ms：

- `shot_0002` 获得 2,250 ms 重叠；
- `shot_0003` 获得 750 ms 重叠。

这不是重复标注错误。台词从桥上对话画面延续到想象中的机器人画面，字幕语义和视觉镜头确实跨越了剪辑点。

## 本次结果

- 98 个镜头；
- 67 句字幕，全部至少关联一个镜头；
- 88 条镜头—台词关系；
- 55 个有台词镜头，43 个无台词镜头；
- 20 句字幕跨越镜头边界；
- 单个镜头最多关联 6 句字幕。

## 当前限制

SRT 只有句级时间，没有逐词时间。跨镜头台词会把完整句子放入两个镜头的 `transcript`，但 `utterance_refs` 保存了各自真实的重叠范围。以后若使用带词级时间戳的 ASR，可以进一步判断一句话的哪些词落在哪个镜头中。

## 复现命令

```powershell
python scripts/build_evidence_units.py `
  --shots data/processed/tears_of_steel/s1/shots_ffmpeg_t035/shots.jsonl `
  --subtitles data/processed/tears_of_steel/s1/subtitles_en.srt `
  --output-dir data/processed/tears_of_steel/s1/shots_ffmpeg_t035 `
  --language en
```

## 对 Agent 的意义

一个证据单元现在同时具有：

- 时间范围：用于定位和 IoU reward；
- 关键帧：用于视觉理解；
- 字幕：用于低成本文本检索；
- 原始引用 ID 和重叠范围：用于 EvidenceAuditor 检查答案依据。

Agent 可以先搜索便宜的字幕，再只查看命中镜头的关键帧，而不必把整段视频一次性交给 VLM。
