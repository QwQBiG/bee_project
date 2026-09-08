from dataclasses import dataclass
import unittest

from behavior.entrance_adapter import TrackEntranceAnalyzer


@dataclass
class TrackStub:
    track_id: int
    center: tuple
    confidence: float = 0.9
    state: str = "confirmed"


def config(**overrides):
    result = {
        "enabled": True,
        "geometry": {
            "origin": [0.0, 0.0], "inward": [1.0, 0.0],
            "outside_depth": 0.3, "inside_depth": 0.6, "half_width": 0.2,
        },
        "min_confirm_frames": 1,
        "cooldown_frames": 2,
        "min_displacement": 0.1,
        "missing_tolerance": 1,
    }
    result.update(overrides)
    return result


class EntranceAdapterTests(unittest.TestCase):
    def test_disabled_analyzer_does_not_change_pipeline(self):
        analyzer = TrackEntranceAnalyzer({"enabled": False})
        self.assertEqual([], analyzer.update([TrackStub(1, (10, 10))], 0, (100, 100)))
        self.assertFalse(analyzer.build_report()["enabled"])

    def test_normalized_track_creates_entering_event(self):
        analyzer = TrackEntranceAnalyzer(config())
        events = []
        for frame, x in enumerate((10, 45, 75)):
            events.extend(analyzer.update([TrackStub(1, (x, 5))], frame, (100, 100)))
        self.assertEqual(1, len(events))
        self.assertEqual("entering", events[0].event_type)
        self.assertEqual(1, analyzer.build_report()["counts"]["entering"])

    def test_lost_track_in_buffer_becomes_uncertain(self):
        analyzer = TrackEntranceAnalyzer(config())
        analyzer.update([TrackStub(2, (10, 5))], 0, (100, 100))
        analyzer.update([TrackStub(2, (45, 5))], 1, (100, 100))
        analyzer.update([], 2, (100, 100))
        events = analyzer.update([], 3, (100, 100))
        self.assertEqual("uncertain", events[0].event_type)

    def test_enabled_analyzer_requires_geometry(self):
        with self.assertRaises(ValueError):
            TrackEntranceAnalyzer({"enabled": True})


if __name__ == "__main__":
    unittest.main()
