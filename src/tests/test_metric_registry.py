import unittest

from behavior.metric_registry import assess_metric_readiness, readiness_summary


class MetricRegistryTests(unittest.TestCase):
    def as_map(self, signals):
        return {item["key"]: item for item in assess_metric_readiness(signals)}

    def test_basic_video_metrics_can_become_ready(self):
        metrics = self.as_map({"confirmed_entrance_events", "calibrated_tracks", "video_fps"})
        self.assertTrue(metrics["entrance_traffic"]["ready"])
        self.assertTrue(metrics["trajectory_speed"]["ready"])
        self.assertFalse(metrics["body_orientation"]["ready"])

    def test_pose_does_not_imply_disease_or_survival(self):
        metrics = self.as_map({"validated_pose_model", "calibrated_tracks"})
        self.assertTrue(metrics["body_orientation"]["ready"])
        self.assertFalse(metrics["survival_rate"]["ready"])
        self.assertFalse(metrics["disease_risk"]["ready"])

    def test_summary_lists_missing_external_metrics(self):
        summary = readiness_summary({"confirmed_entrance_events"})
        self.assertIn("entrance_traffic", summary["ready_metrics"])
        self.assertLess(summary["ready_count"], summary["total_count"])


if __name__ == "__main__":
    unittest.main()
