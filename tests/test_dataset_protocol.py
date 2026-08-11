from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from videoops_rl.dataset_protocol import audit_task_records, package_inventory, safe_repo_path


class DatasetProtocolTests(unittest.TestCase):
    def test_rejects_path_outside_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                safe_repo_path(Path(directory), "../outside.bin")

    def test_rejects_video_level_split_leakage(self):
        tasks = [self._task("a", "video_1", "train"), self._task("b", "video_1", "test")]
        checks = audit_task_records(tasks, {"video_1": 10_000})
        split_check = next(check for check in checks if check["name"] == "video_level_split_isolation")
        self.assertFalse(split_check["passed"])

    def test_allows_many_smoke_tasks_on_one_video(self):
        tasks = [self._task("a", "video_1", "smoke"), self._task("b", "video_1", "smoke")]
        self.assertTrue(all(check["passed"] for check in audit_task_records(tasks, {"video_1": 10_000})))

    def test_rejects_target_outside_video(self):
        task = self._task("a", "video_1", "train")
        task["target_segments"] = [{"start_ms": 9_000, "end_ms": 11_000}]
        self.assertFalse(audit_task_records([task], {"video_1": 10_000})[0]["passed"])

    def test_inventory_ignores_tmp_and_pycache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir(); (root / "data/file.bin").write_bytes(b"1234")
            (root / "tmp").mkdir(); (root / "tmp/large.bin").write_bytes(b"x" * 100)
            (root / "__pycache__").mkdir(); (root / "__pycache__/cache.bin").write_bytes(b"y" * 100)
            self.assertEqual(package_inventory(root)["bytes"], 4)

    def test_json_schema_rejects_missing_target_segments(self):
        schema_path = Path(__file__).resolve().parents[1] / "schemas/videoops_task_v1.schema.json"
        import json

        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        task = self._task("a", "video_1", "train")
        task["schema_version"] = "videoops.task.v1"
        task["evidence"].update({"search_terms": [], "required_modalities": ["text"]})
        task.pop("target_segments")
        self.assertTrue(list(Draft202012Validator(schema).iter_errors(task)))

    @staticmethod
    def _task(task_id: str, video_id: str, split: str):
        return {
            "task_id": task_id, "video_id": video_id, "split": split, "query": "find the event",
            "target_segments": [{"start_ms": 1_000, "end_ms": 2_000}], "evidence": {"shot_ids": ["shot_1"]},
        }


if __name__ == "__main__":
    unittest.main()
