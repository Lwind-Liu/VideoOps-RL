"""Build the bounded three-video dataset used by all pre-server stages."""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from videoops_rl.dataset_protocol import (  # noqa: E402
    HARD_LIMIT_BYTES, TARGET_LIMIT_BYTES, audit_manifest, package_inventory, read_jsonl, sha256_file,
)

TAG_RE = re.compile(r"<[^>]+>")
WORD_RE = re.compile(r"[A-Za-z0-9']+")


def clean_text(text: str) -> str:
    return " ".join(TAG_RE.sub(" ", html.unescape(text)).replace("♪", " ").split())


def meaningful(text: str) -> bool:
    value = clean_text(text)
    return len(WORD_RE.findall(value)) >= 4 and not (value.startswith("(") and value.endswith(")"))


def artifact(path: Path) -> dict:
    return {"path": path.relative_to(REPO_ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def overlapping_shots(units: list[dict], start_ms: int, end_ms: int) -> list[str]:
    return [unit["shot_id"] for unit in units if unit["end_ms"] > start_ms and unit["start_ms"] < end_ms]


def dialogue_task(prefix: str, index: int, video_id: str, split: str, utterance: dict, units: list[dict]) -> dict:
    text = clean_text(utterance["text"])
    start = max(0, utterance["start_ms"] - 1500)
    end = utterance["end_ms"] + 1500
    return {
        "schema_version": "videoops.task.v1", "task_id": f"{prefix}_{index:03d}",
        "video_id": video_id, "split": split,
        "query": f"Locate the moment where the dialogue says: {text[:120]}",
        "target_segments": [{"start_ms": start, "end_ms": end}],
        "evidence": {"search_terms": [text], "shot_ids": overlapping_shots(units, start, end), "required_modalities": ["text", "image"]},
        "task_type": "dialogue_grounded_highlight", "difficulty": "easy_exact_dialogue",
    }


def video_spec(video_id: str, split: str, duration_ms: int, media: Path, evidence_root: Path, subtitle: Path, source_url: str) -> dict:
    keyframes = sorted((evidence_root / "keyframes").glob("*.jpg"))
    return {
        "video_id": video_id, "split": split, "duration_ms": duration_ms, "license": "CC BY 3.0",
        "source_url": source_url, "evidence_root": evidence_root.relative_to(REPO_ROOT).as_posix(),
        "artifacts": [
            artifact(media), artifact(subtitle), artifact(evidence_root / "shots.jsonl"),
            artifact(evidence_root / "utterances.jsonl"), artifact(evidence_root / "evidence_units.jsonl"),
        ],
        "keyframe_inventory": {"count": len(keyframes), "bytes": sum(path.stat().st_size for path in keyframes)},
    }


def main() -> None:
    tos_root = REPO_ROOT / "data/processed/tears_of_steel/s1/shots_ffmpeg_t035"
    sintel_root = REPO_ROOT / "data/processed/sintel/shots_t035"
    bbb_root = REPO_ROOT / "data/processed/big_buck_bunny/shots_t035"
    tos_units, sintel_units, bbb_units = (read_jsonl(root / "evidence_units.jsonl") for root in (tos_root, sintel_root, bbb_root))
    tos_utterances, sintel_utterances = (read_jsonl(root / "utterances.jsonl") for root in (tos_root, sintel_root))

    tasks: list[dict] = []
    for original in read_jsonl(REPO_ROOT / "data/annotations/highlight_tasks.jsonl"):
        tasks.append({
            "schema_version": "videoops.task.v1", "task_id": original["task_id"],
            "video_id": "tears_of_steel_train", "split": "train", "query": original["query"],
            "target_segments": [{"start_ms": original["target_start_ms"], "end_ms": original["target_end_ms"]}],
            "evidence": {
                "search_terms": original["search_terms"],
                "shot_ids": overlapping_shots(tos_units, original["target_start_ms"], original["target_end_ms"]),
                "required_modalities": ["text", "image"],
            },
            "task_type": "business_query_highlight", "difficulty": "medium_paraphrase",
        })
    selected_tos = [item for item in tos_utterances if meaningful(item["text"])][:30]
    tasks.extend(dialogue_task("tos_auto", index, "tears_of_steel_train", "train", item, tos_units) for index, item in enumerate(selected_tos, 1))

    candidates = [item for item in sintel_utterances if meaningful(item["text"]) and item["start_ms"] < 720_000]
    selected_sintel = candidates[:: max(1, len(candidates) // 15)][:15]
    tasks.extend(dialogue_task("sintel_val", index, "sintel_val", "val", item, sintel_units) for index, item in enumerate(selected_sintel, 1))

    bbb_visual = [
        ("bunny wakes in a sunny meadow", ["shot_0004"], ["bunny", "waking", "meadow"]),
        ("bunny watches a purple butterfly", ["shot_0015"], ["bunny", "purple butterfly", "flowers"]),
        ("three small bullies hide behind a tree", ["shot_0018", "shot_0020"], ["three bullies", "tree", "hiding"]),
        ("flying squirrel holds the captured butterfly", ["shot_0034"], ["flying squirrel", "butterfly", "captured"]),
        ("bunny discovers the damaged flowers", ["shot_0044", "shot_0045"], ["bunny", "damaged flowers", "sad"]),
        ("bunny plans revenge in the forest", ["shot_0062"], ["bunny", "forest", "planning revenge"]),
        ("bullies carry wood through the forest", ["shot_0070"], ["bullies", "wood", "forest"]),
        ("bunny prepares a rope trap near a tree", ["shot_0083"], ["bunny", "rope trap", "tree"]),
        ("flying squirrel balances on a tree branch", ["shot_0095"], ["flying squirrel", "tree branch", "balancing"]),
        ("flying squirrel is launched through the air", ["shot_0108", "shot_0109"], ["flying squirrel", "launched", "air"]),
        ("flying squirrel falls toward wooden spikes", ["shot_0114"], ["flying squirrel", "wooden spikes", "falling"]),
    ]
    bbb_by_id = {unit["shot_id"]: unit for unit in bbb_units}
    visual_tags: dict[str, list[str]] = {}
    for index, (description, shot_ids, tags) in enumerate(bbb_visual, 1):
        start = min(bbb_by_id[shot_id]["start_ms"] for shot_id in shot_ids)
        end = max(bbb_by_id[shot_id]["end_ms"] for shot_id in shot_ids)
        tasks.append({
            "schema_version": "videoops.task.v1", "task_id": f"bbb_test_{index:03d}",
            "video_id": "big_buck_bunny_test", "split": "test", "query": f"Find the moment where the {description}.",
            "target_segments": [{"start_ms": start, "end_ms": end}],
            "evidence": {"search_terms": tags, "shot_ids": shot_ids, "required_modalities": ["image"]},
            "task_type": "visual_only_highlight", "difficulty": "visual_no_dialogue",
        })
        for shot_id in shot_ids:
            visual_tags[f"big_buck_bunny_test:{shot_id}"] = tags

    tos_tags = json.loads((REPO_ROOT / "data/annotations/visual_evidence.json").read_text(encoding="utf-8"))
    visual_tags.update({f"tears_of_steel_train:{shot_id}": tags for shot_id, tags in tos_tags.items()})

    registry = REPO_ROOT / "data/registry"
    registry.mkdir(parents=True, exist_ok=True)
    tasks_path = registry / "formal_tasks_v1.jsonl"
    tasks_path.write_text("".join(json.dumps(task, ensure_ascii=False, separators=(",", ":")) + "\n" for task in tasks), encoding="utf-8")
    tags_path = registry / "visual_tags_v1.json"
    tags_path.write_text(json.dumps(visual_tags, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    videos = [
        video_spec("tears_of_steel_train", "train", 480_000, REPO_ROOT / "data/processed/tears_of_steel/s1/sample_20s_500s.mp4", tos_root, REPO_ROOT / "data/processed/tears_of_steel/s1/subtitles_en.srt", "https://mango.blender.org/"),
        video_spec("sintel_val", "val", 888_093, REPO_ROOT / "data/raw/sintel/sintel_360p.mp4", sintel_root, REPO_ROOT / "data/raw/sintel/sintel_en.srt", "https://video.blender.org/videos/watch/0eb052d0-fd51-43e6-aa33-ecdbf77a5d40"),
        video_spec("big_buck_bunny_test", "test", 596_566, REPO_ROOT / "data/raw/big_buck_bunny/big_buck_bunny_240p.mp4", bbb_root, REPO_ROOT / "data/raw/big_buck_bunny/empty.srt", "https://video.blender.org/videos/watch/bf1f3fb5-b119-4f9f-9930-8e20e892b898"),
    ]
    manifest = {
        "schema_version": "videoops.dataset_manifest.v1", "dataset_id": "videoops_formal_v1",
        "scope": {"role": "pre_server_training", "business_task": "query_driven_highlight_localization", "claim_boundary": "small engineering corpus for end-to-end validation, not benchmark SOTA"},
        "capacity": {"hard_limit_bytes": HARD_LIMIT_BYTES, "target_limit_bytes": TARGET_LIMIT_BYTES, "policy": "fully offline server handoff"},
        "videos": videos,
        "tasks": {"path": tasks_path.relative_to(REPO_ROOT).as_posix(), "count": len(tasks), "schema": "schemas/videoops_task_v1.schema.json", "visual_tags": tags_path.relative_to(REPO_ROOT).as_posix()},
        "split_policy": {"unit": "video_id", "production_splits": ["train", "val", "test"], "rule": "one source film belongs to one split", "smoke_excluded_from_metrics": True},
        "current_inventory": package_inventory(REPO_ROOT),
    }
    manifest_path = registry / "formal_dataset_manifest_v1.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = audit_manifest(manifest, REPO_ROOT, verify_hashes=True)
    report_path = REPO_ROOT / "outputs/reports/formal_data_audit_v1.json"
    report_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    split_counts = {split: sum(task["split"] == split for task in tasks) for split in ("train", "val", "test")}
    print(json.dumps({"passed": audit["passed"], "videos": len(videos), "tasks": len(tasks), "splits": split_counts}, indent=2))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
