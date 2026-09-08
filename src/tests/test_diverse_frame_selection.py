import unittest

from tools.select_diverse_frames import FrameFeature, euclidean, select_diverse


def feature(frame, vector, quality=0.8):
    return FrameFeature(frame, list(vector), quality, 0.5, 0.5, 0.5, 0.1)


class DiverseFrameSelectionTests(unittest.TestCase):
    def test_selects_visually_different_frames(self):
        items = [
            feature(0, [0.0, 0.0], 1.0),
            feature(100, [0.05, 0.05], 0.9),
            feature(200, [1.0, 1.0], 0.8),
        ]
        selected = select_diverse(items, 2, min_frame_gap=50, temporal_weight=0)
        self.assertEqual([0, 200], [item.frame_index for item in selected])

    def test_minimum_gap_is_respected_when_possible(self):
        items = [feature(frame, [frame / 100.0], 1.0) for frame in (0, 10, 100)]
        selected = select_diverse(items, 2, min_frame_gap=50)
        self.assertGreaterEqual(abs(selected[1].frame_index - selected[0].frame_index), 50)

    def test_selection_is_deterministic(self):
        items = [feature(frame, [frame / 100.0, 1 - frame / 100.0])
                 for frame in (0, 25, 50, 75, 100)]
        first = select_diverse(items, 3, 20)
        second = select_diverse(list(reversed(items)), 3, 20)
        self.assertEqual([item.frame_index for item in first],
                         [item.frame_index for item in second])

    def test_empty_input_returns_empty(self):
        self.assertEqual([], select_diverse([], 3))

    def test_feature_dimension_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            euclidean([0.0], [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
