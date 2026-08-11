"""Re-run P1 data and 50 GiB offline-package gates without rebuilding data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from videoops_rl.dataset_protocol import audit_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/registry/p1_dataset_manifest.json")
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((REPO_ROOT / args.manifest).read_text(encoding="utf-8"))
    report = audit_manifest(manifest, REPO_ROOT, verify_hashes=not args.skip_hashes)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
