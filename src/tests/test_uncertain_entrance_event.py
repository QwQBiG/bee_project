import unittest

from behavior.entrance_events import EntranceEventDetector, EntranceGeometry


class UncertainEntranceEventTests(unittest.TestCase):
    def test_track_lost_in_buffer_is_uncertain(self):
        detector = EntranceEventDetector(
            EntranceGeometry((0, 0), (1, 0), 0, 10), min_displacement=2)
        detector.update(9, (-5, 0), 0)
        detector.update(9, (4, 0), 1)
        event = detector.finish_track(9)
        self.assertIsNotNone(event)
        self.assertEqual("uncertain", event.event_type)
        self.assertEqual("uncertain", event.status)
        self.assertEqual(1, detector.counts()["uncertain"])

    def test_stable_track_end_does_not_create_event(self):
        detector = EntranceEventDetector(EntranceGeometry((0, 0), (1, 0), 0, 10))
        detector.update(2, (-4, 0), 0)
        self.assertIsNone(detector.finish_track(2))


if __name__ == "__main__":
    unittest.main()
