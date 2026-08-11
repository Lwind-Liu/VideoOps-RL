param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [double]$StartSeconds = 20.0,
    [double]$DurationSeconds = 480.0
)

$ErrorActionPreference = "Stop"

$rawDir = Join-Path $ProjectRoot "data\raw\tears_of_steel"
$outputDir = Join-Path $ProjectRoot "data\processed\tears_of_steel\s1"
$sourceVideo = Join-Path $rawDir "tears_of_steel_720p.mov"
$sourceSubtitle = Join-Path $rawDir "TOS-en.srt"
$sampleVideo = Join-Path $outputDir "sample_20s_500s.mp4"
$videoOnly = Join-Path $outputDir "video_only.mp4"
$audioAsr = Join-Path $outputDir "audio_asr_16k_mono.wav"
$subtitleOut = Join-Path $outputDir "subtitles_en.srt"
$subtitleClipper = Join-Path $ProjectRoot "scripts\clip_srt.py"

foreach ($command in @("ffmpeg", "ffprobe")) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Required command is unavailable: $command"
    }
}

foreach ($source in @($sourceVideo, $sourceSubtitle)) {
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Required source file is missing: $source"
    }
}

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

# Re-encode once so that the experiment sample starts exactly at t=0.
& ffmpeg -hide_banner -loglevel error -y `
    -i $sourceVideo `
    -ss $StartSeconds -t $DurationSeconds `
    -map 0:v:0 -map 0:a:0 `
    -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p `
    -c:a aac -b:a 192k `
    -map_metadata -1 -movflags +faststart `
    $sampleVideo
if ($LASTEXITCODE -ne 0) { throw "Failed to create experiment sample." }

# Demux the encoded video stream without decoding or re-encoding it again.
& ffmpeg -hide_banner -loglevel error -y `
    -i $sampleVideo -map 0:v:0 -c:v copy -an $videoOnly
if ($LASTEXITCODE -ne 0) { throw "Failed to create video-only stream." }

# Decode audio to the common ASR input format: PCM signed 16-bit, mono, 16 kHz.
& ffmpeg -hide_banner -loglevel error -y `
    -i $sampleVideo -map 0:a:0 -vn -ac 1 -ar 16000 -c:a pcm_s16le $audioAsr
if ($LASTEXITCODE -ne 0) { throw "Failed to create ASR audio." }

# Subtitle seek semantics differ from audio/video streams. Parse the SRT explicitly,
# retain only overlapping cues, clip boundaries, and reset timestamps to zero.
& python $subtitleClipper `
    --input $sourceSubtitle `
    --output $subtitleOut `
    --start-seconds $StartSeconds `
    --duration-seconds $DurationSeconds
if ($LASTEXITCODE -ne 0) { throw "Failed to create aligned subtitle." }

Write-Output "Created S1 outputs in $outputDir"
Get-ChildItem -LiteralPath $outputDir -File | Select-Object Name, Length
