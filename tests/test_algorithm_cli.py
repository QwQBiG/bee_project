"""Offline structural + pure-computation tests for inference.algorithm_cli."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np


class MockSess:
    """Fake onnxruntime InferenceSession (no onnxruntime install needed)."""

    def __init__(self, outputs: np.ndarray):
        self._outputs = outputs

    def get_inputs(self):
        input_info = SimpleNamespace()
        input_info.name = "images"
        input_info.shape = [1, 3, 640, 640]
        return [input_info]

    def get_outputs(self):
        return ["output0"]

    def run(self, _out_names, _feed):  # noqa: N803
        return [self._outputs]


def _install_mock_session(monkeypatch, output_array: np.ndarray,
                          onnxruntime_missing: bool = False) -> None:
    import sys
    if onnxruntime_missing:
        monkeypatch.setitem(sys.modules, "onnxruntime", None)
        return

    import sys
    fake_ort = SimpleNamespace()
    fake_ort.SessionOptions = type("SessionOptions", (), {
        "log_severity_level": 0,
        "log_verbosity_level": 3,
    })
    fake_ort.InferenceSession = lambda *a, **kw: MockSess(output_array)
    fake_ort.get_available_providers = lambda: ["CPUExecutionProvider"]
    sys.modules["onnxruntime"] = fake_ort


class AlgorithmCliStructureTest(unittest.TestCase):
    def test_import_and_entry_points_exist(self):
        from inference import algorithm_cli as cli
        self.assertTrue(callable(cli.main))
        self.assertTrue(callable(cli.build_parser))
        self.assertTrue(callable(cli.load_runtime_config))
        self.assertTrue(callable(cli._emit))

    def test_parser_matches_competition_contract(self):
        from inference.algorithm_cli import build_parser
        parser = build_parser()
        actions = {a.dest for a in parser._actions}
        for required in ("image_path", "task", "frame_id",
                         "sequence_id", "image_id", "config"):
            self.assertIn(required, actions, required)
        # image_path is mandatory.
        with self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_scene_from_path_hints(self):
        from inference.algorithm_cli import infer_scene_from_path
        self.assertEqual(infer_scene_from_path("/data/A-5-1/0001.jpg"),
                         "outside")
        self.assertEqual(infer_scene_from_path("/data/B-5-3_IR/frame001.png"),
                         "inside")
        self.assertIsNone(infer_scene_from_path("/tmp/unlabelled/7.jpg"))

    def test_image_saturation_classifier(self):
        from inference.algorithm_cli import infer_scene_from_image
        # Infrared-looking image: all channels identical.
        ir = np.stack([np.full((32, 32), 128, dtype=np.uint8)] * 3, axis=-1)
        self.assertTrue("inside_ir" in infer_scene_from_image(ir))
        # Colourful visible-light image: R>G>B large differences.
        vis = np.zeros((32, 32, 3), dtype=np.uint8)
        vis[..., 0] = 40; vis[..., 1] = 160; vis[..., 2] = 220
        self.assertTrue("outside" in infer_scene_from_image(vis))

    def test_multiclass_nms_shape_and_ordering(self):
        from inference.algorithm_cli import multiclass_nms
        # Two perfectly-separated boxes, both above conf.
        boxes = np.array([[10, 10, 20, 20], [50, 50, 20, 20]], dtype=np.float32)
        scores = np.array([0.9, 0.7], dtype=np.float32)
        keep = multiclass_nms(boxes, scores, iou_threshold=0.5,
                              conf_threshold=0.5)
        self.assertEqual(keep.shape[0], 2)
        # Descending confidence order.
        self.assertGreaterEqual(keep[0, 4], keep[1, 4])

    def test_emit_exact_single_line_and_returns_code(self):
        from io import StringIO
        from inference.algorithm_cli import _emit
        buffer = StringIO()
        import sys
        old_stdout = sys.stdout
        try:
            sys.stdout = buffer
            code = _emit({"code": 1, "image_id": "x", "detections": [],
                          "processing_time_ms": 5}, code=1)
        finally:
            sys.stdout = old_stdout
        self.assertEqual(code, 1)
        output = buffer.getvalue()
        self.assertTrue(output.endswith("\n"))
        # Exactly one newline at the very end (single line of JSON).
        stripped = output.rstrip("\n")
        self.assertNotIn("\n", stripped)
        payload = json.loads(stripped)
        self.assertEqual(payload["processing_time_ms"], 5)
        self.assertEqual(payload["image_id"], "x")

    def test_missing_image_returns_code_zero_with_error(self):
        import io, sys
        from inference.algorithm_cli import main
        buf_out = io.StringIO()
        buf_err = io.StringIO()
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = buf_out, buf_err
            code = main(["--image_path",
                         "C:\\definitely does not exist\\" + "X" * 32 + ".jpg"])
        finally:
            sys.stdout, sys.stderr = old_out, old_err
        self.assertEqual(code, 0)
        payload = json.loads(buf_out.getvalue().splitlines()[-1])
        self.assertEqual(payload["code"], 0)
        self.assertIn("error", payload)
        self.assertIn("message", payload)


if __name__ == "__main__":
    unittest.main()
