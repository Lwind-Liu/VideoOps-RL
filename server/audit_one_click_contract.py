"""Validate the public checkout before downloading multi-gigabyte assets."""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE_TAG = "offline-v2.0.0"
PART_NAMES = [f"VideoOps-RL-offline-server.zip.part-{index:02d}" for index in range(3)]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def read_manifest() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in (ROOT / "release_manifest.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        entries[name] = digest
    return entries


def require_text(path: str, snippets: list[str], errors: list[str]) -> None:
    target = ROOT / path
    if not target.is_file():
        errors.append(f"missing file: {path}")
        return
    text = target.read_text(encoding="utf-8")
    for snippet in snippets:
        if snippet not in text:
            errors.append(f"{path} is missing contract token: {snippet}")


def check_release(manifest: dict[str, str], errors: list[str]) -> None:
    url = f"https://api.github.com/repos/Lwind-Liu/VideoOps-RL/releases/tags/{RELEASE_TAG}"
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        release = json.load(response)
    assets = {asset["name"]: asset for asset in release.get("assets", [])}
    for name, digest in manifest.items():
        asset = assets.get(name)
        if not asset:
            errors.append(f"release asset missing: {name}")
            continue
        if asset.get("state") != "uploaded" or int(asset.get("size", 0)) <= 0:
            errors.append(f"release asset is not ready: {name}")
        if asset.get("digest") != f"sha256:{digest}":
            errors.append(f"release digest mismatch: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-release", action="store_true", help="also query the public GitHub Release API")
    args = parser.parse_args()
    errors: list[str] = []

    try:
        manifest = read_manifest()
    except Exception as error:
        raise SystemExit(f"Invalid release_manifest.sha256: {error}") from error
    if list(manifest) != PART_NAMES:
        errors.append(f"manifest must contain exactly: {', '.join(PART_NAMES)}")
    for name, digest in manifest.items():
        if not SHA256_RE.fullmatch(digest):
            errors.append(f"invalid SHA-256 for {name}")

    require_text(
        "bootstrap_server.sh",
        [
            'RUN_MODE=${RUN_MODE:-auto}',
            "python server/audit_one_click_contract.py",
            "sha256sum --check release_manifest.sha256",
            "run_pipeline smoke",
            "run_pipeline full",
            'bash server/run_all.sh',
        ],
        errors,
    )
    require_text("server/run_all.sh", ["run_all_24gpu.sh", "run_all_8gpu.sh"], errors)
    require_text("server/run_all_24gpu.sh", ["server/preflight.py", "--require-traces"], errors)
    require_text("server/run_all_8gpu.sh", ["server/preflight.py", "--require-traces"], errors)
    require_text("server/collect_run_bundle.sh", ["checkpoint_sha256.txt", "outputs/traces"], errors)
    require_text("README.md", ["bash bootstrap_server.sh", "server/collect_run_bundle.sh"], errors)
    for path in (
        "server/train_sft.py",
        "server/train_llm_grpo.py",
        "server/evaluate_checkpoint.py",
        "server/analyze_training_run.py",
        "src/videoops_rl/tool_gateway.py",
    ):
        if not (ROOT / path).is_file():
            errors.append(f"missing runtime file: {path}")

    if args.check_release:
        try:
            check_release(manifest, errors)
        except Exception as error:
            errors.append(f"GitHub Release check failed: {error}")

    report = {
        "passed": not errors,
        "release_tag": RELEASE_TAG,
        "parts": PART_NAMES,
        "release_checked": args.check_release,
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
