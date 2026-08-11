# S1.2：派生样本与解复用

## 1. 本次要做什么

完整原片约 12 分钟。实验阶段从原片第 20 秒开始取 8 分钟，即源时间轴：

```text
[20 s, 500 s)
```

派生样本重新从 `0 s` 开始计时：

```text
源视频 20 s  -> 样本 0 s
源视频 500 s -> 样本 480 s
```

这样字幕、帧、音频和后续事件都可以使用更简单的 0—480 秒实验时间轴。

## 2. 为什么派生样本需要重编码一次

H.264 不是每一帧都能独立解码。直接使用 `-c copy` 截取时，起点通常需要退到前一个关键帧，未必精确落在第 20 秒。

本阶段选择重编码一次：

- 获得精确起点；
- 将输出时间戳归零；
- 统一输出为 H.264 + AAC 的 MP4；
- 后续实验全部复用该样本，不反复重编码。

原始 MOV 始终保留，因此派生过程可重新运行。

## 3. 解复用输出

```text
sample_20s_500s.mp4
├── video_only.mp4
├── audio_asr_16k_mono.wav
└── subtitles_en.srt
```

- `video_only.mp4`：只保留 H.264 视频流，不再次编码；
- `audio_asr_16k_mono.wav`：把音频解码成 16 kHz、单声道、16-bit PCM；
- `subtitles_en.srt`：由显式 SRT 解析器截取同一时间区间，并把字幕时间减去 20 秒。不能只依赖 FFmpeg 的字幕 seek；本次验收发现其会平移时间但保留 480 秒以后的 cue。

这里要区分：

```text
demux：从容器中取出某条已有 stream
decode：把压缩编码恢复成可直接处理的数据
encode：把数据重新压缩成某种编码
```

视频只做 demux；ASR 音频需要 decode 和重采样；实验样本创建时进行一次视频和音频 encode。

## 4. 可复现命令

```powershell
powershell -ExecutionPolicy Bypass -File scripts\prepare_s1_sample.ps1
```
