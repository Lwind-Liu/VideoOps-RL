import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from videoops_rl.media_audit import check_records


class MediaAuditTests(unittest.TestCase):
    def valid_records(self, directory: Path):
        keyframe_dir = directory / "keyframes"
        keyframe_dir.mkdir()
        (keyframe_dir / "shot_0001.jpg").write_bytes(b"jpeg")
        (keyframe_dir / "shot_0002.jpg").write_bytes(b"jpeg")
        shots = [
            {"shot_id": "shot_0001", "start_s": 0.0, "end_s": 5.0, "keyframe_path": "keyframes/shot_0001.jpg"},
            {"shot_id": "shot_0002", "start_s": 5.0, "end_s": 10.0, "keyframe_path": "keyframes/shot_0002.jpg"},
        ]
        utterances = [
            {"utterance_id": "utt_0001", "start_ms": 4_000, "end_ms": 6_000}
        ]
        evidence = [
            {
                "shot_id": "shot_0001", "start_ms": 0, "end_ms": 5_000,
                "keyframe_path": "keyframes/shot_0001.jpg",
                "utterance_refs": [{"utterance_id": "utt_0001", "overlap_start_ms": 4_000, "overlap_end_ms": 5_000, "overlap_ms": 1_000}],
            },
            {
                "shot_id": "shot_0002", "start_ms": 5_000, "end_ms": 10_000,
                "keyframe_path": "keyframes/shot_0002.jpg",
                "utterance_refs": [{"utterance_id": "utt_0001", "overlap_start_ms": 5_000, "overlap_end_ms": 6_000, "overlap_ms": 1_000}],
            },
        ]
        return shots, utterances, evidence

    def test_valid_cross_artifact_records_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = self.valid_records(root)
            checks = check_records(*records, root, duration_ms=10_000)
            self.assertTrue(all(check.passed for check in checks))

    def test_timeline_gap_is_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shots, utterances, evidence = self.valid_records(root)
            shots[1]["start_s"] = 5.5
            checks = check_records(shots, utterances, evidence, root, 10_000)
            result = {check.name: check.passed for check in checks}
            self.assertFalse(result["shot_timeline_contiguous"])
            self.assertFalse(result["evidence_units_match_shots"])

    def test_wrong_overlap_is_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shots, utterances, evidence = self.valid_records(root)
            evidence[0]["utterance_refs"][0]["overlap_ms"] = 999
            checks = check_records(shots, utterances, evidence, root, 10_000)
            result = {check.name: check.passed for check in checks}
            self.assertFalse(result["utterance_references_exact"])


if __name__ == "__main__":
    unittest.main()
