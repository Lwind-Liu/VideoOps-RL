import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from videoops_rl.alignment import (
    ShotInterval,
    align_shots_and_utterances,
    cues_to_utterances,
    overlap_interval,
)
from videoops_rl.srt import SubtitleCue


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.shots = [
            ShotInterval("shot_0001", 0, 5_000, 2_500, "keyframes/shot_0001.jpg"),
            ShotInterval("shot_0002", 5_000, 10_000, 7_500, "keyframes/shot_0002.jpg"),
            ShotInterval("shot_0003", 10_000, 15_000, 12_500, "keyframes/shot_0003.jpg"),
        ]

    def test_half_open_intervals_do_not_overlap_at_touching_boundary(self):
        self.assertIsNone(overlap_interval(0, 5_000, 5_000, 7_000))

    def test_crossing_utterance_links_to_both_shots(self):
        utterances = cues_to_utterances(
            [SubtitleCue(4_000, 6_000, "crosses the cut")], "en"
        )
        units = align_shots_and_utterances(self.shots, utterances)

        self.assertEqual(units[0]["utterance_refs"][0]["overlap_ms"], 1_000)
        self.assertEqual(units[1]["utterance_refs"][0]["overlap_ms"], 1_000)
        self.assertFalse(units[2]["has_dialogue"])

    def test_exact_boundary_utterance_only_links_to_second_shot(self):
        utterances = cues_to_utterances([SubtitleCue(5_000, 7_000, "second")], "en")
        units = align_shots_and_utterances(self.shots, utterances)

        self.assertFalse(units[0]["has_dialogue"])
        self.assertTrue(units[1]["has_dialogue"])
        self.assertEqual(units[1]["transcript"], "second")


if __name__ == "__main__":
    unittest.main()
