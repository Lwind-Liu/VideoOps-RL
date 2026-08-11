import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from videoops_rl import Segment, SyntheticVideoEnv, VideoTask


def build_env() -> SyntheticVideoEnv:
    segments = [
        Segment("seg_01", 0, 5000, "控制室设备正常运行", ("control_room", "normal")),
        Segment("seg_02", 5000, 10000, "主角发现设备冒烟", ("person", "smoke", "alarm")),
        Segment("seg_03", 10000, 15000, "主角立即关闭电源", ("person", "shutdown", "safety")),
    ]
    task = VideoTask(
        "task_02",
        "主角如何发现并处理设备异常？",
        frozenset({"seg_02", "seg_03"}),
        "主角先发现设备冒烟，随后关闭电源。",
    )
    return SyntheticVideoEnv(segments, [task])


class MultiSegmentTaskTests(unittest.TestCase):
    def test_complete_answer_with_complete_evidence_succeeds(self):
        env = build_env()
        env.reset("task_02")
        search = env.step({"type": "search_transcript", "query": "主角"})
        self.assertEqual(search.info["search_metrics"]["recall"], 1.0)
        env.step({"type": "inspect_segment", "segment_id": "seg_02"})
        env.step({"type": "inspect_segment", "segment_id": "seg_03"})
        answer = env.step(
            {"type": "answer", "segment_ids": ["seg_02", "seg_03"]}
        )
        self.assertTrue(answer.info["success"])

    def test_partial_answer_is_incorrect_even_with_evidence(self):
        env = build_env()
        env.reset("task_02")
        env.step({"type": "inspect_segment", "segment_id": "seg_02"})
        answer = env.step({"type": "answer", "segment_ids": ["seg_02"]})
        self.assertFalse(answer.info["correct"])
        self.assertTrue(answer.info["evidence"])

    def test_complete_answer_without_all_evidence_is_rejected(self):
        env = build_env()
        env.reset("task_02")
        env.step({"type": "inspect_segment", "segment_id": "seg_02"})
        answer = env.step(
            {"type": "answer", "segment_ids": ["seg_02", "seg_03"]}
        )
        self.assertTrue(answer.info["correct"])
        self.assertFalse(answer.info["evidence"])

