import unittest

from videoops_rl.shots import build_shots


class ShotTests(unittest.TestCase):
    def test_builds_contiguous_shots_and_midpoints(self):
        shots = build_shots([(5.0, 0.6), (12.0, 0.4)], duration_s=20.0)

        self.assertEqual([shot.shot_id for shot in shots], ["shot_0001", "shot_0002", "shot_0003"])
        self.assertEqual([(shot.start_s, shot.end_s) for shot in shots], [(0.0, 5.0), (5.0, 12.0), (12.0, 20.0)])
        self.assertEqual([shot.keyframe_s for shot in shots], [2.5, 8.5, 16.0])
        self.assertIsNone(shots[0].start_scene_score)
        self.assertEqual(shots[1].start_scene_score, 0.6)

    def test_ignores_out_of_range_and_deduplicates_boundaries(self):
        shots = build_shots([(-1.0, 0.5), (5.0, 0.4), (5.0, 0.7), (20.0, 0.9)], 20.0)

        self.assertEqual(len(shots), 2)
        self.assertEqual(shots[1].start_scene_score, 0.7)

    def test_rejects_non_positive_duration(self):
        with self.assertRaises(ValueError):
            build_shots([], 0.0)


if __name__ == "__main__":
    unittest.main()
