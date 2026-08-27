param(
    [string]$Repository = "Lwind-Liu/VideoOps-RL",
    [string]$ReleaseTag = "offline-v2.0.0",
    [string]$OutputDirectory = "offline_assets"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$manifestPath = Join-Path $root "release_manifest.sha256"
$outputPath = Join-Path $root $OutputDirectory
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null

$expected = @{}
foreach ($line in Get-Content -LiteralPath $manifestPath) {
    if ($line -match '^([0-9a-f]{64})\s+(.+)$') {
        $expected[$Matches[2]] = $Matches[1]
    }
}

$baseUrl = "https://github.com/$Repository/releases/download/$ReleaseTag"
if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
    throw "curl.exe is required for resumable release downloads."
}
foreach ($name in $expected.Keys | Sort-Object) {
    $target = Join-Path $outputPath $name
    $valid = Test-Path -LiteralPath $target
    if ($valid) {
        $valid = (Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash.ToLowerInvariant() -eq $expected[$name]
    }
    if (-not $valid) {
        Write-Host "Downloading $name"
        & curl.exe --fail --location --retry 8 --retry-all-errors --continue-at - --output $target "$baseUrl/$name"
        if ($LASTEXITCODE -ne 0) {
            throw "Download failed: $name"
        }
    }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash.ToLowerInvariant()
    if ($actual -ne $expected[$name]) {
        throw "Checksum mismatch: $target"
    }
}

Copy-Item -LiteralPath $manifestPath -Destination (Join-Path $outputPath "release_manifest.sha256") -Force
Write-Host "Offline assets are ready: $outputPath"
Write-Host "Upload the repository with this directory, then run RUN_MODE=smoke bash bootstrap_server.sh."
