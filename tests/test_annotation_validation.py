from pathlib import Path
import tempfile
import unittest

from annotation.schema import BeeInstance, BehaviorSegment, FrameAnnotation, Keypoint, VideoAnnotation
from tools.validate_annotations import validate_path


class AnnotationValidationTests(unittest.TestCase):
    def test_behavior_round_trip_and_report(self):
        annotation = VideoAnnotation(
            video_id="dance-001", source_path="raw/dance.mp4", scene="inside_ir",
            width=320, height=240, fps=30.0, frame_count=90,
            behaviors=[BehaviorSegment(
                segment_id="behavior-1", label="waggle_dance_candidate",
                start_frame=10, end_frame=40, track_ids=[3], reviewer="reviewer-a")],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = annotation.save(Path(directory) / "dance.json")
            loaded = VideoAnnotation.load(path)
            self.assertEqual("waggle_dance_candidate", loaded.behaviors[0].label)
            report = validate_path(Path(directory), require_manual=True)
            self.assertTrue(report["valid"])
            self.assertEqual(1, report["behaviors"])

    def test_invalid_behavior_range(self):
        annotation = VideoAnnotation(
            video_id="bad", source_path="bad.mp4", scene="inside_ir",
            width=10, height=10, fps=10.0, frame_count=10,
            behaviors=[BehaviorSegment("bad-1", "falling", 5, 12)],
        )
        self.assertTrue(any("超出视频帧范围" in item for item in annotation.validate()))

    def test_pose_gold_requires_complete_orientation_points(self):
        annotation = VideoAnnotation(
            video_id="pose", source_path="pose.mp4", scene="inside_ir",
            width=100, height=100, fps=30.0, frame_count=2,
            frames=[FrameAnnotation(0, 0.0, [BeeInstance(
                "bee-1", [10, 10, 20, 20],
                keypoints=[Keypoint("head", 20, 15), Keypoint("thorax", 18, 18)],
            )])],
        )
        with tempfile.TemporaryDirectory() as directory:
            annotation.save(Path(directory) / "pose.json")
            report = validate_path(Path(directory), pose_gold=True)
            self.assertFalse(report["valid"])
            self.assertIn("abdomen_tip", report["results"][0]["errors"][0])

    def test_pose_gold_can_require_track_ids(self):
        annotation = VideoAnnotation(
            video_id="pose", source_path="pose.mp4", scene="inside_ir",
            width=100, height=100, fps=30.0, frame_count=2,
            frames=[FrameAnnotation(0, 0.0, [BeeInstance(
                "bee-1", [10, 10, 20, 20],
                keypoints=[Keypoint("head", 20, 15), Keypoint("thorax", 18, 18),
                           Keypoint("abdomen_tip", 12, 20)],
            )])],
        )
        errors = annotation.validate_pose_gold(require_track_ids=True)
        self.assertTrue(any("track_id" in item for item in errors))

if __name__ == "__main__":
    unittest.main()
