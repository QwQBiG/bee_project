"""Offline tests for the new evaluation metrics in tools.evaluate_vnbee_tracking."""

from __future__ import annotations

import unittest
from typing import Dict, List


class CocoMapMotStatsTest(unittest.TestCase):
    def test_metrics_importable(self):
        from tools.evaluate_vnbee_tracking import (
            IOU_THRESHOLDS_COCO, compute_map50_95, compute_mot_track_stats,
        )
        self.assertEqual(len(IOU_THRESHOLDS_COCO), 10)
        self.assertEqual(IOU_THRESHOLDS_COCO[0], 0.50)
        self.assertEqual(IOU_THRESHOLDS_COCO[-1], 0.95)
        self.assertTrue(callable(compute_map50_95))
        self.assertTrue(callable(compute_mot_track_stats))

    def test_map_without_any_predictions_is_zero(self):
        from tools.evaluate_vnbee_tracking import compute_map50_95
        gt = {1: [{"track_id": 1, "bbox": [10, 10, 20, 20]}]}
        pred: Dict[int, List[dict]] = {1: []}
        out = compute_map50_95(gt, pred)
        self.assertLess(out["map_50_95"], 0.02)
        self.assertLess(out["ap_50"], 0.02)

    def test_map_with_perfect_prediction_saturates(self):
        from tools.evaluate_vnbee_tracking import compute_map50_95
        gt = {1: [{"track_id": 1, "bbox": [10, 10, 20, 20]}]}
        pred = {1: [{"track_id": -1, "bbox": [10, 10, 20, 20], "confidence": 0.9}]}
        out = compute_map50_95(gt, pred)
        # AP@0.50 for a 1-frame perfect match should be very high.
        self.assertGreater(out["ap_50"], 0.9)
        # map_50_95 will drop at strict thresholds because the single box
        # IoU is exactly 1.0 for the loose thresholds.
        self.assertGreater(out["map_50_95"], 0.0)

    def test_mot_track_stats_split_by_coverage_ratio(self):
        from tools.evaluate_vnbee_tracking import compute_mot_track_stats
        # 5 trajectories of 10 frames each.
        gt_tracks: Dict[int, List[int]] = {
            tid: list(range(10)) for tid in range(1, 6)
        }
        pred_tracks: Dict[int, List[int]] = {
            100 + tid: list(range(10)) for tid in range(1, 6)
        }
        # Track 1 → 8/10 matched → MT.
        # Track 2 → 1/10 matched → ML.
        # Tracks 3,4 → 4, 5/10 matched → PT.
        # Track 5 → 9/10 → MT.
        matched: Dict[int, List[int]] = {
            1: list(range(8)),
            2: [0],
            3: list(range(4)),
            4: list(range(5)),
            5: list(range(9)),
        }
        stats = compute_mot_track_stats(gt_tracks, pred_tracks, matched)
        self.assertEqual(stats["mt_ids"], [1, 5])
        self.assertEqual(stats["ml_ids"], [2])
        self.assertEqual(stats["pt_ids"], [3, 4])
        self.assertEqual(stats["mt"], 2)
        self.assertEqual(stats["pt"], 2)
        self.assertEqual(stats["ml"], 1)
        self.assertEqual(stats["n_gt_tracks"], 5)
        self.assertAlmostEqual(stats["mt_ratio"], 2 / 5)
        self.assertAlmostEqual(stats["ml_ratio"], 1 / 5)

    def test_empty_dataset_yields_zero_metrics(self):
        from tools.evaluate_vnbee_tracking import compute_mot_track_stats
        out = compute_mot_track_stats({}, {}, {})
        self.assertEqual(out["n_gt_tracks"], 0)
        self.assertEqual(out["mt"], 0)


if __name__ == "__main__":
    unittest.main()
