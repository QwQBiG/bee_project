"""Tests for ``tools.prepare_mendeley_pose`` (label-only, offline)."""

from __future__ import annotations

import inspect
import unittest
from pathlib import Path

from tools.prepare_mendeley_pose import (
    MENDELEY_POSE_FIELDS,
    PROJECT_POSE_FIELDS,
    adapt_row_to_3kp,
)


class MendeleyPoseAdaptationTest(unittest.TestCase):
    def test_module_is_importable(self):
        import tools.prepare_mendeley_pose as m  # noqa: WPS111
        self.assertTrue(callable(m.main))

    def test_2kp_to_3kp_typical_row(self):
        """A standard Mendeley 11-field bee row grows a middle thorax kp."""
        row = (
            "0 0.500 0.500 0.100 0.040 "
            "0.450 0.480 2 "
            "0.550 0.520 2"
        )
        self.assertEqual(len(row.split()), MENDELEY_POSE_FIELDS)
        adapted = adapt_row_to_3kp(row)
        fields = adapted.split()
        self.assertEqual(len(fields), PROJECT_POSE_FIELDS)
        # class, cx, cy, w, h unchanged
        self.assertEqual(fields[0], "0")
        expected_floats = {1: 0.5, 2: 0.5, 3: 0.1, 4: 0.04}
        for idx, expected in expected_floats.items():
            self.assertAlmostEqual(float(fields[idx]), expected, places=5)
        # head keypoint unchanged: (0.45, 0.48, vis=2.0)
        self.assertAlmostEqual(float(fields[5]), 0.45, places=5)
        self.assertAlmostEqual(float(fields[6]), 0.48, places=5)
        self.assertAlmostEqual(float(fields[7]), 2.0, places=5)
        # thorax keypoint injected at mid-point: ((0.45+0.55)/2, (0.48+0.52)/2)
        self.assertAlmostEqual(float(fields[8]), 0.5, places=5)
        self.assertAlmostEqual(float(fields[9]), 0.5, places=5)
        self.assertAlmostEqual(float(fields[10]), 0.1, places=5)  # pseudo vis
        # abdomen/stinger keypoint unchanged
        self.assertAlmostEqual(float(fields[11]), 0.55, places=5)
        self.assertAlmostEqual(float(fields[12]), 0.52, places=5)
        self.assertAlmostEqual(float(fields[13]), 2.0, places=5)

    def test_thorax_visibility_can_be_configured(self):
        row = (
            "0 0.5 0.5 0.1 0.04 "
            "0.45 0.48 2 "
            "0.55 0.52 2"
        )
        adapted = adapt_row_to_3kp(row, pseudolabel_vis=0.05)
        self.assertAlmostEqual(float(adapted.split()[10]), 0.05, places=5)

    def test_empty_line_passes_through(self):
        self.assertEqual(adapt_row_to_3kp("   "), "")
        self.assertEqual(adapt_row_to_3kp(""), "")

    def test_bad_field_count_raises(self):
        # only 5 fields (detection row, no keypoints)
        with self.assertRaises(ValueError):
            adapt_row_to_3kp("0 0.5 0.5 0.1 0.04")

    def test_ordered_keypoint_names_match_project_schema(self):
        """The 3 keypoint positions must match annotation/schema.py KEYPOINT_NAMES."""
        import annotation.schema as schema
        names = schema.KEYPOINT_NAMES
        self.assertEqual(names, ("head", "thorax", "abdomen_tip"))
        # We use index=0 -> head, index=1 -> thorax, index=2 -> abdomen_tip
        # in the 3-kp adapted label; this assertion guards against drift.
        self.assertEqual(len(names), 3)

    def test_exported_constants_are_stable(self):
        self.assertEqual(MENDELEY_POSE_FIELDS, 11)
        self.assertEqual(PROJECT_POSE_FIELDS, 14)


if __name__ == "__main__":
    unittest.main()
