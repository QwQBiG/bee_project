import argparse
import json
from pathlib import Path
import tempfile
import unittest

from tools.train_yolo import training_options, validate_dataset


class TrainingPreflightTests(unittest.TestCase):
    def make_dataset(self, root: Path, metadata):
        data = root / "data.yaml"
        data.write_text(
            "path: ./\ntrain: train.txt\nval: val.txt\nkpt_shape: [3, 3]\nnames:\n  0: bee\n",
            encoding="utf-8")
        (root / "dataset_meta.json").write_text(
            json.dumps(metadata), encoding="utf-8")
        return data

    def test_ready_pose_dataset_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self.make_dataset(Path(directory), {
                "task": "pose",
                "pose_labels_ready": True,
                "training_ready": True,
                "split_counts": {"train": 10, "val": 5, "test": 0},
            })
            report = validate_dataset(data, "pose")
            self.assertTrue(report["training_ready"])

    def test_unreviewed_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self.make_dataset(Path(directory), {
                "task": "pose",
                "pose_labels_ready": False,
                "training_ready": False,
                "split_counts": {"train": 20, "val": 0, "test": 0},
            })
            with self.assertRaisesRegex(ValueError, "人工金标准"):
                validate_dataset(data, "pose")

    def test_training_options_reject_zero_batch(self):
        args = argparse.Namespace(
            data="data.yaml", epochs=10, batch=0, imgsz=640, workers=0,
            lr0=0.001,
            seed=42, patience=5, project="runs", name="test", amp=True, cache=None)
        with self.assertRaises(ValueError):
            training_options(args, "cpu")


if __name__ == "__main__":
    unittest.main()
