"""Cross-artifact integrity checks for a processed video sample."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AuditCheck:
    name: str
    passed: bool
    details: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def check_records(
    shots: list[dict[str, object]],
    utterances: list[dict[str, object]],
    evidence_units: list[dict[str, object]],
    artifact_dir: Path,
    duration_ms: int,
) -> list[AuditCheck]:
    """Validate timeline, identifiers, references, and keyframe paths."""
    checks: list[AuditCheck] = []

    shot_ids = [str(shot["shot_id"]) for shot in shots]
    utterance_ids = [str(utterance["utterance_id"]) for utterance in utterances]
    evidence_shot_ids = [str(unit["shot_id"]) for unit in evidence_units]

    contiguous = bool(shots)
    if shots:
        contiguous = (
            round(float(shots[0]["start_s"]) * 1000) == 0
            and round(float(shots[-1]["end_s"]) * 1000) == duration_ms
            and all(
                round(float(left["end_s"]) * 1000)
                == round(float(right["start_s"]) * 1000)
                for left, right in zip(shots, shots[1:])
            )
        )
    checks.append(
        AuditCheck(
            "shot_timeline_contiguous",
            contiguous,
            f"{len(shots)} shots cover 0..{duration_ms} ms without gaps or overlaps",
        )
    )

    unique_ids = len(shot_ids) == len(set(shot_ids)) and len(utterance_ids) == len(
        set(utterance_ids)
    )
    checks.append(
        AuditCheck(
            "record_ids_unique",
            unique_ids,
            f"shot_ids={len(shot_ids)}, utterance_ids={len(utterance_ids)}",
        )
    )

    keyframe_paths = [artifact_dir / str(shot["keyframe_path"]) for shot in shots]
    existing_keyframes = sum(path.is_file() for path in keyframe_paths)
    discovered_keyframes = list((artifact_dir / "keyframes").glob("shot_*.jpg"))
    checks.append(
        AuditCheck(
            "one_keyframe_per_shot",
            existing_keyframes == len(shots) == len(discovered_keyframes),
            f"referenced={existing_keyframes}, discovered={len(discovered_keyframes)}, shots={len(shots)}",
        )
    )

    utterances_in_range = all(
        0 <= int(item["start_ms"]) < int(item["end_ms"]) <= duration_ms
        for item in utterances
    )
    checks.append(
        AuditCheck(
            "utterances_within_timeline",
            utterances_in_range,
            f"{len(utterances)} utterances checked against 0..{duration_ms} ms",
        )
    )

    evidence_matches_shots = (
        len(evidence_units) == len(shots)
        and evidence_shot_ids == shot_ids
        and all(
            int(unit["start_ms"]) == round(float(shot["start_s"]) * 1000)
            and int(unit["end_ms"]) == round(float(shot["end_s"]) * 1000)
            and str(unit["keyframe_path"]) == str(shot["keyframe_path"])
            for shot, unit in zip(shots, evidence_units)
        )
    )
    checks.append(
        AuditCheck(
            "evidence_units_match_shots",
            evidence_matches_shots,
            f"evidence_units={len(evidence_units)}, shots={len(shots)}",
        )
    )

    utterance_by_id = {str(item["utterance_id"]): item for item in utterances}
    linked_ids: set[str] = set()
    valid_references = True
    reference_count = 0
    for unit in evidence_units:
        for reference in unit["utterance_refs"]:
            reference_count += 1
            utterance_id = str(reference["utterance_id"])
            utterance = utterance_by_id.get(utterance_id)
            if utterance is None:
                valid_references = False
                continue
            linked_ids.add(utterance_id)
            expected_start = max(int(unit["start_ms"]), int(utterance["start_ms"]))
            expected_end = min(int(unit["end_ms"]), int(utterance["end_ms"]))
            valid_references = valid_references and (
                expected_start < expected_end
                and int(reference["overlap_start_ms"]) == expected_start
                and int(reference["overlap_end_ms"]) == expected_end
                and int(reference["overlap_ms"]) == expected_end - expected_start
            )
    checks.append(
        AuditCheck(
            "utterance_references_exact",
            valid_references,
            f"validated {reference_count} shot-utterance overlap records",
        )
    )

    missing_links = sorted(set(utterance_ids) - linked_ids)
    checks.append(
        AuditCheck(
            "all_utterances_linked",
            not missing_links,
            "all utterances linked" if not missing_links else f"unlinked={missing_links}",
        )
    )
    return checks
