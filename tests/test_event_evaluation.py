import unittest

from annotation.schema import TemporalEvent
from tools.evaluate_entrance_events import evaluate_events


class EntranceEventEvaluationTests(unittest.TestCase):
    def test_event_metrics_with_one_false_positive(self):
        truth = [
            TemporalEvent("gt-1", "entering", 8, 12, [1], {"crossing_frame": 10}),
            TemporalEvent("gt-2", "leaving", 28, 32, [2], {"crossing_frame": 30}),
        ]
        predictions = [
            TemporalEvent("pred-1", "entering", 9, 13, [9], {"crossing_frame": 11}, "prediction"),
            TemporalEvent("pred-2", "leaving", 29, 33, [8], {"crossing_frame": 31}, "prediction"),
            TemporalEvent("pred-3", "entering", 50, 55, [7], {"crossing_frame": 52}, "prediction"),
        ]
        report = evaluate_events(truth, predictions, tolerance_frames=2)
        self.assertEqual(2, report["overall"]["true_positive"])
        self.assertEqual(1, report["overall"]["false_positive"])
        self.assertEqual(0, report["overall"]["false_negative"])
        self.assertAlmostEqual(2 / 3, report["overall"]["precision"], places=5)

    def test_track_id_can_be_required(self):
        truth = [TemporalEvent("gt", "entering", 1, 3, [1])]
        predictions = [TemporalEvent("pred", "entering", 1, 3, [2], source="prediction")]
        report = evaluate_events(truth, predictions, require_track_id=True)
        self.assertEqual(0, report["overall"]["true_positive"])

    def test_uncertain_events_are_excluded(self):
        truth = [TemporalEvent("u", "uncertain", 1, 3, [1])]
        report = evaluate_events(truth, [])
        self.assertEqual(0, report["overall"]["false_negative"])


if __name__ == "__main__":
    unittest.main()
