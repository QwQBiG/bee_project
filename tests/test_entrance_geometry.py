import unittest

from behavior.entrance_events import EntranceEventDetector, EntranceGeometry


class EntranceGeometryTests(unittest.TestCase):
    def test_lateral_corridor_rejects_far_track(self):
        geometry = EntranceGeometry((0, 0), (1, 0), 0, 10, half_width=5)
        detector = EntranceEventDetector(
            geometry, min_confirm_frames=1, cooldown_frames=30, min_displacement=1)
        for frame_id, x in enumerate((-5, 3, 12)):
            detector.update(1, (x, 20), frame_id)
        self.assertEqual(0, detector.counts()["entering"])

    def test_first_event_is_not_blocked_by_cooldown(self):
        geometry = EntranceGeometry((0, 0), (1, 0), 0, 10, half_width=5)
        detector = EntranceEventDetector(
            geometry, min_confirm_frames=1, cooldown_frames=30, min_displacement=1)
        event = None
        for frame_id, x in enumerate((-5, 3, 12)):
            event = detector.update(2, (x, 0), frame_id) or event
        self.assertIsNotNone(event)
        self.assertEqual("entering", event.event_type)

    def test_invalid_corridor_width_is_rejected(self):
        with self.assertRaises(ValueError):
            EntranceGeometry((0, 0), (1, 0), 0, 10, half_width=0)


if __name__ == "__main__":
    unittest.main()
