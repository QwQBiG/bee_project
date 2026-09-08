from pathlib import Path
import tempfile
import unittest

from annotation.schema import BeeInstance, FrameAnnotation, Keypoint, TemporalEvent, VideoAnnotation
from behavior.entrance_events import EntranceEventDetector, EntranceGeometry
from tools.prepare_unlabeled_dataset import assign_split, frame_plan, infer_scene


class AnnotationSchemaTests(unittest.TestCase):
    def test_manual_annotation_round_trip(self):
        annotation = VideoAnnotation(
            video_id="inside-001", source_path="raw/inside.mp4", scene="inside_ir",
            width=640, height=480, fps=25.0, frame_count=100,
            frames=[FrameAnnotation(5, 200.0, [BeeInstance(
                instance_id="5-0", bbox=[10, 20, 40, 30], track_id=7,
                keypoints=[Keypoint("head", 18, 30), Keypoint("thorax", 30, 32),
                           Keypoint("abdomen_tip", 45, 35)])])],
            events=[TemporalEvent("event-1", "entering", 4, 8, [7])],
        )
        self.assertEqual([], annotation.validate(require_manual=True))
        with tempfile.TemporaryDirectory() as directory:
            path = annotation.save(Path(directory) / "annotation.json")
            text = path.read_text(encoding="utf-8")
            self.assertIn('"head"', text)
            self.assertIn('"entering"', text)

    def test_prediction_is_not_gold_truth(self):
        annotation = VideoAnnotation(
            video_id="outside-001", source_path="raw/outside.mp4",
            scene="outside_entrance", width=100, height=100, fps=30.0, frame_count=10,
            frames=[FrameAnnotation(0, 0.0, [BeeInstance(
                instance_id="0-0", bbox=[5, 5, 20, 20], source="prediction", confidence=0.8)])],
        )
        errors = annotation.validate(require_manual=True)
        self.assertTrue(any("非人工实例" in item for item in errors))

    def test_rejects_out_of_bounds_keypoint(self):
        instance = BeeInstance("bad", [0, 0, 10, 10],
                               keypoints=[Keypoint("head", 200, 5)])
        self.assertTrue(any("超出画面" in item for item in instance.validate(100, 100)))


class EntranceEventTests(unittest.TestCase):
    def setUp(self):
        geometry = EntranceGeometry(origin=(0, 0), inward=(1, 0),
                                    outside_depth=0, inside_depth=10)
        self.detector = EntranceEventDetector(
            geometry, min_confirm_frames=2, max_transition_frames=20,
            cooldown_frames=2, min_displacement=5)

    def feed(self, track_id, xs):
        events = []
        for frame_id, x in enumerate(xs):
            event = self.detector.update(track_id, (x, 0), frame_id)
            if event:
                events.append(event)
        return events

    def test_entering(self):
        events = self.feed(1, [-5, -3, 2, 6, 11, 13])
        self.assertEqual(1, len(events))
        self.assertEqual("entering", events[0].event_type)
        self.assertEqual({"entering": 1, "leaving": 0, "uncertain": 0, "net_flow": 1},
                         self.detector.counts())

    def test_leaving(self):
        events = self.feed(2, [15, 13, 8, 3, -2, -4])
        self.assertEqual("leaving", events[0].event_type)

    def test_return_to_same_side_is_not_counted(self):
        events = self.feed(3, [-5, 2, 7, 3, -2, -4])
        self.assertEqual([], events)

    def test_frames_must_increase(self):
        self.detector.update(4, (-3, 0), 5)
        with self.assertRaises(ValueError):
            self.detector.update(4, (-2, 0), 5)


class DatasetPreparationTests(unittest.TestCase):
    def test_frame_plan_is_bounded_and_unique(self):
        self.assertEqual([0, 2, 4, 7, 9], frame_plan(10, 5))
        self.assertEqual(list(range(3)), frame_plan(3, 10))

    def test_scene_inference(self):
        self.assertEqual("inside_ir", infer_scene(Path("巢内/红外/test.mp4")))
        self.assertEqual("inside_ir", infer_scene(Path("clips/inside_B-5-1.mp4")))
        self.assertEqual("outside_entrance", infer_scene(Path("outside/entrance/test.mp4")))

    def test_split_is_stable(self):
        self.assertEqual(assign_split("video-001"), assign_split("video-001"))


if __name__ == "__main__":
    unittest.main()
