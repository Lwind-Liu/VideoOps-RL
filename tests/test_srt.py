import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from videoops_rl.srt import SubtitleCue, clip_cues, parse_srt, render_srt


class SubtitleTests(unittest.TestCase):
    def test_clip_shifts_and_removes_out_of_range_cues(self):
        cues = [
            SubtitleCue(19_000, 21_000, "crosses start"),
            SubtitleCue(23_000, 24_500, "first dialogue"),
            SubtitleCue(499_000, 501_000, "crosses end"),
            SubtitleCue(510_000, 511_000, "outside"),
        ]
        result = clip_cues(cues, start_ms=20_000, end_ms=500_000)
        self.assertEqual(
            result,
            [
                SubtitleCue(0, 1_000, "crosses start"),
                SubtitleCue(3_000, 4_500, "first dialogue"),
                SubtitleCue(479_000, 480_000, "crosses end"),
            ],
        )

    def test_rendered_srt_can_be_parsed_again(self):
        cues = [SubtitleCue(3_000, 4_500, "You're a jerk, Thom.")]
        self.assertEqual(parse_srt(render_srt(cues)), cues)

