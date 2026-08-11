import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from videoops_rl import Segment, SyntheticVideoEnv, VideoTask


def build_env() -> SyntheticVideoEnv:
    segments = [
        Segment("seg_01", 0, 5000, "主角走进房间", ("person", "room")),
        Segment("seg_02", 5000, 10000, "主角发现异常并说出真相", ("person", "surprise")),
        Segment("seg_03", 10000, 15000, "两人开始争吵", ("person", "conflict")),
    ]
    tasks = [VideoTask("task_01", "发现异常", frozenset({"seg_02"}), "seg_02")]
    return SyntheticVideoEnv(segments, tasks)


class SmokeTests(unittest.TestCase):
    def test_evidence_grounded_episode_succeeds(self):
        env = build_env()
        observation = env.reset("task_01")
        self.assertEqual(observation["query"], "发现异常")
        search = env.step({"type": "search_transcript", "query": "发现异常"})
        self.assertIn("seg_02", search.info["hits"])
        inspect = env.step({"type": "inspect_segment", "segment_id": "seg_02"})
        self.assertGreater(inspect.reward, 0)
        answer = env.step({"type": "answer", "segment_ids": ["seg_02"]})
        self.assertTrue(answer.terminated)
        self.assertEqual(answer.reward, 1.0)
        self.assertEqual(answer.info["reward_components"], {"grounded_success": 1.0})

    def test_answer_without_evidence_is_rejected(self):
        env = build_env()
        env.reset("task_01")
        result = env.step({"type": "answer", "segment_ids": ["seg_02"]})
        self.assertTrue(result.terminated)
        self.assertLess(result.reward, 0)
        self.assertFalse(result.info["evidence"])

    def test_duplicate_inspection_is_penalized(self):
        env = build_env()
        env.reset("task_01")
        env.step({"type": "inspect_segment", "segment_id": "seg_02"})
        duplicate = env.step({"type": "inspect_segment", "segment_id": "seg_02"})
        self.assertAlmostEqual(duplicate.reward, -0.15)
        self.assertEqual(duplicate.info["error"], "duplicate_inspection")

    def test_precise_search_beats_broad_search(self):
        precise_env = build_env()
        precise_env.reset("task_01")
        precise = precise_env.step({"type": "search_transcript", "query": "发现异常"})

        broad_env = build_env()
        broad_env.reset("task_01")
        broad = broad_env.step({"type": "search_transcript", "query": "主角"})

        self.assertEqual(precise.info["hits"], ["seg_02"])
        self.assertEqual(broad.info["hits"], ["seg_01", "seg_02"])
        self.assertEqual(precise.info["search_metrics"]["precision"], 1.0)
        self.assertEqual(broad.info["search_metrics"]["precision"], 0.5)
        self.assertGreater(precise.reward, broad.reward)

    def test_search_without_hits_is_penalized(self):
        env = build_env()
        env.reset("task_01")
        result = env.step({"type": "search_transcript", "query": "天气"})
        self.assertAlmostEqual(result.reward, -0.10)
        self.assertEqual(result.info["search_metrics"]["f1"], 0.0)
