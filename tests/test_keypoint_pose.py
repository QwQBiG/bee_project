from dataclasses import dataclass, field
from types import SimpleNamespace
import unittest

from inference.keypoint_pose import (
    KeypointPoseEstimator,
    PoseDetection,
    bbox_iou,
    match_pose_to_tracks,
)


@dataclass
class TrackStub:
    track_id: int
    bbox: list
    pose: object = field(default_factory=lambda: SimpleNamespace(
        head_bbox=[1, 1, 2, 2], abdomen_bbox=[5, 1, 2, 2],
        orientation=45.0, head_tail_known=False))


def detection(confidence=0.9, point_confidence=0.8):
    return PoseDetection(
        bbox=(10, 10, 20, 10),
        keypoints={
            "head": (26, 15, point_confidence),
            "thorax": (20, 15, point_confidence),
            "abdomen_tip": (14, 15, point_confidence),
        },
        confidence=confidence,
    )


class KeypointPoseTests(unittest.TestCase):
    def test_iou_uses_xywh_boxes(self):
        self.assertEqual(1.0, bbox_iou([10, 10, 20, 10], [10, 10, 20, 10]))
        self.assertEqual(0.0, bbox_iou([0, 0, 5, 5], [10, 10, 5, 5]))

    def test_disabled_mode_clears_legacy_fake_head_tail(self):
        track = TrackStub(1, [10, 10, 20, 10])
        result = match_pose_to_tracks([track], [])
        self.assertEqual(1, result["unmatched_tracks"])
        self.assertIsNone(track.pose.head_bbox)
        self.assertIsNone(track.pose.abdomen_bbox)
        self.assertFalse(track.pose.head_tail_known)
        self.assertEqual("body_axis_unoriented", track.pose.orientation_kind)

    def test_valid_keypoints_create_directed_orientation(self):
        track = TrackStub(2, [10, 10, 20, 10])
        result = match_pose_to_tracks([track], [detection()])
        self.assertEqual(1, result["matched_tracks"])
        self.assertTrue(track.pose.head_tail_known)
        self.assertEqual("head_direction", track.pose.orientation_kind)
        self.assertEqual("validated_pose_model", track.pose.source)
        self.assertAlmostEqual(0.0, track.pose.orientation)
        self.assertIsNotNone(track.pose.head_bbox)

    def test_low_confidence_keypoint_is_rejected(self):
        track = TrackStub(3, [10, 10, 20, 10])
        result = match_pose_to_tracks(
            [track], [detection(point_confidence=0.2)], min_keypoint_confidence=0.35)
        self.assertEqual(0, result["matched_tracks"])
        self.assertEqual(1, result["rejected_pose_detections"])
        self.assertFalse(track.pose.head_tail_known)

    def test_pose_detection_is_not_reused_for_multiple_tracks(self):
        tracks = [TrackStub(1, [10, 10, 20, 10]), TrackStub(2, [11, 10, 20, 10])]
        result = match_pose_to_tracks(tracks, [detection()])
        self.assertEqual(1, result["matched_tracks"])
        self.assertEqual(1, result["unmatched_tracks"])

    def test_estimator_defaults_to_safe_disabled_mode(self):
        estimator = KeypointPoseEstimator()
        track = TrackStub(4, [10, 10, 20, 10])
        estimator.update(None, [track])
        report = estimator.build_report()
        self.assertFalse(report["enabled"])
        self.assertFalse(report["head_tail_supported"])
        self.assertIsNone(track.pose.head_bbox)


if __name__ == "__main__":
    unittest.main()
