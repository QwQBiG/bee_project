"""Offline structural tests for ``models.trainer`` scene-aware extensions."""

from __future__ import annotations

import inspect
import unittest


class TrainerStructuralTest(unittest.TestCase):
    def setUp(self):
        from models.trainer import YOLOTrainer
        self.cls = YOLOTrainer
        self.instance = YOLOTrainer({
            "epochs": 1, "batch_size": 1, "device": "cpu",
            "learning_rate": 0.001, "weight_decay": 0.0,
        })

    def test_train_pose_signature(self):
        sig = inspect.signature(self.instance.train_pose)
        for required in ("data_yaml", "base_model", "checkpoint_dir",
                         "run_name", "imgsz"):
            self.assertIn(required, sig.parameters, required)
        self.assertEqual(sig.parameters["base_model"].default, "yolov8n-pose.pt")

    def test_train_detection_task_routes_scene(self):
        sig = inspect.signature(self.instance.train_detection_task)
        self.assertIn("scene", sig.parameters)
        self.assertIn("imgsz=1280",
                      self.instance.train_detection_task.__doc__ or "")

    def test_train_detection_task_rejects_unknown_scene(self):
        with self.assertRaises(ValueError):
            self.instance.train_detection_task(
                data_yaml="nope.yaml", scene="sky")

    def test_validate_accepts_pose_task(self):
        sig = inspect.signature(self.instance.validate)
        self.assertEqual(sig.parameters["task"].default, "detect")
        # validate() gracefully returns {} when ultralytics is missing.
        try:
            from ultralytics import YOLO  # noqa: F401
        except ImportError:
            self.assertEqual(
                self.instance.validate("w.pt", "d.yaml", task="pose"), {})

    def test_export_model_now_accepts_imgsz_and_opset(self):
        sig = inspect.signature(self.instance.export_model)
        for name in ("imgsz", "opset", "simplify", "dynamic"):
            self.assertIn(name, sig.parameters, name)
        self.assertEqual(sig.parameters["imgsz"].default, 640)
        self.assertEqual(sig.parameters["opset"].default, 12)


if __name__ == "__main__":
    unittest.main()
