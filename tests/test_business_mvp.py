import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from videoops_rl.business_env import RealHighlightEnv, load_jsonl, load_tasks, temporal_iou
from videoops_rl.coordinator import run_sequence


ROOT = Path(__file__).parents[1]


class BusinessMVPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = ROOT / "data/processed/tears_of_steel/s1/shots_ffmpeg_t035"
        cls.units = load_jsonl(base / "evidence_units.jsonl")
        cls.utterances = load_jsonl(base / "utterances.jsonl")
        cls.visual = json.loads((ROOT / "data/annotations/visual_evidence.json").read_text(encoding="utf-8"))
        cls.tasks = load_tasks(ROOT / "data/annotations/highlight_tasks.jsonl")

    def test_temporal_iou(self):
        self.assertAlmostEqual(temporal_iou(0, 10, 5, 15), 1 / 3)

    def test_rule_agent_generates_grounded_highlight(self):
        env = RealHighlightEnv(self.units, self.utterances, self.visual, self.tasks[0])
        episode = run_sequence(env, ["search", "inspect", "audit", "submit"])
        self.assertTrue(episode.result["success"])
        self.assertTrue(episode.result["audit_passed"])
        self.assertGreaterEqual(episode.result["temporal_iou"], 0.5)

    def test_audit_is_not_valid_before_inspection(self):
        env = RealHighlightEnv(self.units, self.utterances, self.visual, self.tasks[0])
        env.step("search")
        self.assertFalse(env.valid_action_mask()[3])


if __name__ == "__main__":
    unittest.main()
