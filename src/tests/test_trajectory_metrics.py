from dataclasses import dataclass
import unittest

from behavior.trajectory_metrics import TrajectoryMetricsAnalyzer


@dataclass
class TrackStub:
    track_id: int
    center: tuple
    confidence: float = 0.9
    state: str = "confirmed"


class TrajectoryMetricsTests(unittest.TestCase):
    def test_empty_report_is_unknown(self):
        analyzer = TrajectoryMetricsAnalyzer()
        analyzer.set_video_fps(25)
        report = analyzer.build_report()
        self.assertEqual("unknown", report["status"])
        self.assertEqual(0, report["unique_track_ids"])

    def test_reports_normalized_speed_and_track_quality(self):
        analyzer = TrajectoryMetricsAnalyzer({"min_track_frames": 3})
        analyzer.set_video_fps(10)
        for frame, x in enumerate((10, 20, 30)):
            analyzer.update([TrackStub(7, (x, 50))], frame, (100, 100))
        report = analyzer.build_report()
        self.assertEqual("descriptive_only", report["status"])
        self.assertEqual(1, report["unique_track_ids"])
        self.assertAlmostEqual(1.0, report["motion"]["mean_track_speed_normalized_per_second"])
        self.assertFalse(report["motion"]["physical_scale_calibrated"])
        self.assertIn("tracked_trajectories", report["available_signals"])

    def test_fragmented_tracks_raise_quality_warning(self):
        analyzer = TrajectoryMetricsAnalyzer({"min_track_frames": 3})
        analyzer.set_video_fps(10)
        analyzer.update([TrackStub(1, (10, 10)), TrackStub(2, (20, 20))], 0, (100, 100))
        analyzer.update([TrackStub(3, (30, 30))], 1, (100, 100))
        report = analyzer.build_report()
        self.assertEqual("warning", report["status"])
        self.assertEqual(1.0, report["quality"]["short_track_fraction"])

    def test_physical_scale_is_opt_in(self):
        analyzer = TrajectoryMetricsAnalyzer({"min_track_frames": 2, "pixels_per_mm": 2})
        analyzer.set_video_fps(10)
        analyzer.update([TrackStub(4, (0, 0))], 0, (100, 100))
        analyzer.update([TrackStub(4, (20, 0))], 1, (100, 100))
        track = analyzer.build_report()["tracks"]["4"]
        self.assertEqual(10.0, track["path_length_mm"])
        self.assertEqual(100.0, track["mean_speed_mm_per_second"])

    def test_invalid_scale_is_rejected(self):
        with self.assertRaises(ValueError):
            TrajectoryMetricsAnalyzer({"pixels_per_mm": 0})


if __name__ == "__main__":
    unittest.main()
