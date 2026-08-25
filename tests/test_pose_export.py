import json
from pathlib import Path
import tempfile
import unittest

from annotation.schema import BeeInstance, FrameAnnotation, Keypoint, VideoAnnotation
from tools.export_yolo_pose import DEFAULT_CLASSES, export_annotations, pose_row


class PoseExportTests(unittest.TestCase):
    def test_pose_row_contains_three_keypoints(self):
        instance = BeeInstance(
            "1-0", [10, 20, 40, 20], category="bee",
            keypoints=[Keypoint("head", 45, 30), Keypoint("thorax", 30, 30),
                       Keypoint("abdomen_tip", 15, 30, 1)],
        )
        row = pose_row(instance, 100, 100,
                       {name: index for index, name in enumerate(DEFAULT_CLASSES)})
        fields = row.split()
        self.assertEqual(14, len(fields))
        self.assertEqual("0", fields[0])
        self.assertEqual("1", fields[-1])
    def test_training_row_omits_track_id_by_default(self):
        instance = BeeInstance(
            "1-0", [10, 20, 40, 20], category="bee", track_id=7,
        )
        class_map = {name: index for index, name in enumerate(DEFAULT_CLASSES)}
        self.assertEqual(14, len(pose_row(instance, 100, 100, class_map).split()))
        self.assertEqual(
            "7", pose_row(instance, 100, 100, class_map, include_track_id=True).split()[-1])


    def test_pose_row_can_collapse_unverified_classes(self):
        instance = BeeInstance("1-0", [10, 10, 20, 20], category="worker_bee")
        row = pose_row(instance, 100, 100, {"bee": 0}, category_override="bee")
        self.assertEqual("0", row.split()[0])
        self.assertEqual(14, len(row.split()))

    def test_non_manual_instance_is_excluded_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = root / "frames" / "video-1"
            frames.mkdir(parents=True)
            (frames / "frame_00000000.jpg").write_bytes(b"test-image")
            annotation = VideoAnnotation(
                video_id="video-1", source_path="raw.mp4", scene="inside_ir",
                width=100, height=100, fps=30.0, frame_count=5,
                frames=[FrameAnnotation(0, 0.0, [
                    BeeInstance("manual", [1, 1, 10, 10]),
                    BeeInstance("pred", [20, 20, 10, 10], source="prediction", confidence=0.9),
                ])], metadata={"split": "val"},
            )
            annotation_path = annotation.save(root / "annotations" / "video-1.json")
            summary = export_annotations([annotation_path], root / "frames", root / "export")
            self.assertEqual(1, summary["instances"])
            self.assertEqual(1, summary["skipped_instances"])
            label = root / "export" / "labels" / "val" / "video-1_00000000.txt"
            self.assertEqual(1, len(label.read_text(encoding="utf-8").splitlines()))
            self.assertTrue((root / "export" / "data.yaml").exists())
            self.assertEqual(
                "images/val/video-1_00000000.jpg",
                (root / "export" / "val.txt").read_text(encoding="utf-8").strip())
            mapping = json.loads(
                (root / "export" / "annotation_map.json").read_text(encoding="utf-8"))
            self.assertEqual("video-1", mapping["images"]["video-1_00000000"]["video_id"])
            dataset_meta = json.loads(
                (root / "export" / "dataset_meta.json").read_text(encoding="utf-8"))
            self.assertFalse(dataset_meta["training_ready"])
            self.assertFalse(dataset_meta["pose_labels_ready"])
            self.assertEqual(
                {"manual": 1, "prediction": 0, "interpolated": 0},
                dataset_meta["source_counts"])


if __name__ == "__main__":
    unittest.main()
