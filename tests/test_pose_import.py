import json
from pathlib import Path
import tempfile
import unittest

from annotation.schema import BeeInstance, FrameAnnotation, Keypoint, VideoAnnotation
from tools.export_yolo_pose import export_annotations
from tools.import_yolo_pose import import_dataset, parse_pose_row


class PoseImportTests(unittest.TestCase):
    def test_export_import_round_trip_preserves_pose_and_track(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = root / "frames" / "video-1"
            frames.mkdir(parents=True)
            (frames / "frame_00000002.jpg").write_bytes(b"image")
            annotation = VideoAnnotation(
                video_id="video-1", source_path="raw.mp4", scene="inside_ir",
                width=200, height=100, fps=25.0, frame_count=10,
                frames=[FrameAnnotation(2, 80.0, [BeeInstance(
                    "bee-1", [20, 10, 80, 40], category="worker_bee", track_id=7,
                    keypoints=[
                        Keypoint("head", 90, 30, 2),
                        Keypoint("thorax", 60, 30, 2),
                        Keypoint("abdomen_tip", 30, 30, 1),
                    ],
                )])],
                metadata={"split": "train", "sha256": "abc"},
            )
            source = annotation.save(root / "annotations" / "video-1.json")
            export_root = root / "export"
            export_annotations([source], root / "frames", export_root, include_track_ids=True)
            dataset_meta = json.loads(
                (export_root / "dataset_meta.json").read_text(encoding="utf-8"))
            self.assertTrue(dataset_meta["pose_labels_ready"])
            self.assertFalse(dataset_meta["training_ready"])
            import_dataset(
                export_root, export_root / "annotation_map.json", root / "unreviewed")
            unreviewed = VideoAnnotation.load(root / "unreviewed" / "video-1.json")
            self.assertEqual("interpolated", unreviewed.frames[0].instances[0].source)
            self.assertTrue(unreviewed.validate(require_manual=True))
            summary = import_dataset(
                export_root, export_root / "annotation_map.json", root / "imported",
                reviewed=True)
            self.assertEqual({"videos": 1, "frames": 1, "instances": 1}, summary)
            imported = VideoAnnotation.load(root / "imported" / "video-1.json")
            instance = imported.frames[0].instances[0]
            self.assertEqual("manual", instance.source)
            self.assertEqual(7, instance.track_id)
            self.assertEqual(["head", "thorax", "abdomen_tip"],
                             [point.name for point in instance.keypoints])
            self.assertAlmostEqual(80.0, instance.bbox[2], places=4)
            self.assertEqual("abc", imported.metadata["sha256"])

    def test_rejects_wrong_field_count(self):
        with self.assertRaisesRegex(ValueError, "字段数"):
            parse_pose_row("0 0.5 0.5 0.2", 100, 100, ["bee"], 0, 0)


if __name__ == "__main__":
    unittest.main()
