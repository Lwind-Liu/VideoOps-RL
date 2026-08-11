# S1：真实视频、容器与统一时间轴

## 1. 为什么选择 Tears of Steel

- Blender Foundation 的开放电影；
- 官方声明采用 CC BY 3.0；
- 约 12 分钟，包含英语对白、人物、动作和视觉特效；
- 官方提供 720p 原片及中英文字幕；
- 同一素材后续可以用于剧情理解、高光、分龄和画质增强路由。

我们保留完整原片作为 immutable source，实验样本必须由原片派生，不能覆盖源文件。

## 2. 视频文件不是一串图片

`MOV` 是容器。当前文件包含两条 stream：

| Stream | 类型 | 编码 | 关键参数 |
|---:|---|---|---|
| 0 | video | H.264 Main | 1280×534、24 fps、yuv420p |
| 1 | audio | MP3 | 44.1 kHz、双声道 |

容器负责把不同 stream 组织在同一时间线上。视频编码决定画面如何压缩，音频编码决定声音如何压缩。

文件名包含 `720p`，但实际编码画面是 1280×534。这是因为影片使用宽银幕画幅，没有把上下黑边编码为有效画面。工程上必须相信 `ffprobe`，不能只相信文件名。

## 3. 帧率和 time base

视频：

```text
frame_rate = 24/1
time_base  = 1/24 second
```

在这份素材中，一帧约为：

```text
1 / 24 second = 41.667 ms
```

音频：

```text
sample_rate = 44100 Hz
time_base   = 1/44100 second
```

这表示音视频使用不同计时粒度。项目内部不能把“第几帧”和“第几个音频采样点”当成同一种坐标，必须统一换算成毫秒时间戳。

## 4. 为什么音视频时长略有不同

```text
video duration = 734.166667 s
audio duration = 734.093061 s
difference     ≈ 73.6 ms
```

这类微小差异并不自动代表音画不同步，可能来自编码帧边界和 time base。真正验收需要比较时间戳、最终播放时长和可接受阈值。

## 5. 字幕质量检查

- 英文字幕：76 条 cue，时间格式检查通过，作为 canonical subtitle；
- 中文字幕：77 条 cue，首条结束时间写成 `00:00:06`，缺少毫秒部分，作为辅助翻译而不是时间对齐真值。

这说明“官方数据”也需要质量检查。多语言字幕条数不同，也不能默认逐行一一对应。

## 6. 本阶段命令

查看容器：

```powershell
ffprobe -v error `
  -show_entries format=filename,format_name,duration,size,bit_rate `
  -of default=noprint_wrappers=1 `
  data\raw\tears_of_steel\tears_of_steel_720p.mov
```

查看 stream：

```powershell
ffprobe -v error -show_streams -of json `
  data\raw\tears_of_steel\tears_of_steel_720p.mov
```

## 7. 你需要掌握的结论

1. 容器、视频编码和音频编码不是同一个概念；
2. 帧率决定每秒画面数量，time base 决定时间戳的最小计量单位；
3. 文件名和扩展名不能代替媒体探测；
4. 所有模态最终必须统一到毫秒时间轴；
5. 原始数据必须用 URL、许可证、文件大小和 hash 冻结；
6. 字幕也需要格式和内容质量检查。

