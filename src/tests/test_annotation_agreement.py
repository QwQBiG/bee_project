import unittest

from annotation.schema import BeeInstance, FrameAnnotation, Keypoint, VideoAnnotation
from tools.compare_pose_annotations import compare_annotations, greedy_match


def instance(instance_id, head=(20, 50), thorax=(50, 50), abdomen=(80, 50),
             bbox=(10, 30, 80, 40), source="manual"):
    return BeeInstance(
        instance_id=instance_id,
        bbox=list(bbox),
        source=source,
        confidence=0.9 if source == "prediction" else None,
        keypoints=[
            Keypoint("head", *head),
            Keypoint("thorax", *thorax),
            Keypoint("abdomen_tip", *abdomen),
        ],
    )


def annotation(item):
    return VideoAnnotation(
        video_id="video", source_path="video.mp4", scene="inside_ir",
        width=100, height=100, fps=30, frame_count=10,
        frames=[FrameAnnotation(frame_index=0, timestamp_ms=0, instances=[item])],
    )


class AnnotationAgreementTests(unittest.TestCase):
    def test_perfect_annotations_need_no_adjudication(self):
        report = compare_annotations(annotation(instance("a")), annotation(instance("b")))
        self.assertEqual(0, report["summary"]["adjudication_frame_count"])
        self.assertEqual(1.0, report["summary"]["mean_bbox_iou"])
        self.assertEqual(0.0, report["summary"]["mean_keypoint_normalized_error"])

    def test_head_tail_reversal_is_flagged(self):
        first = annotation(instance("a"))
        second = annotation(instance("b", head=(80, 50), abdomen=(20, 50)))
        report = compare_annotations(first, second)
        self.assertEqual(1, report["summary"]["head_tail_reversal_count"])
        self.assertEqual([0], report["summary"]["adjudication_frames"])

    def test_unmatched_instance_is_flagged(self):
        first = annotation(instance("a"))
        second = annotation(instance("b", bbox=(0, 0, 10, 10),
                                         head=(2, 2), thorax=(5, 5), abdomen=(8, 8)))
        report = compare_annotations(first, second, min_match_iou=0.3)
        self.assertEqual(1, report["summary"]["unmatched_first"])
        self.assertEqual(1, report["summary"]["unmatched_second"])

    def test_prediction_input_is_rejected(self):
        with self.assertRaises(ValueError):
            compare_annotations(annotation(instance("a", source="prediction")),
                                annotation(instance("b")))

    def test_greedy_match_is_one_to_one(self):
        left = [instance("a"), instance("b")]
        right = [instance("c")]
        matches, unmatched_left, unmatched_right = greedy_match(left, right, 0.3)
        self.assertEqual(1, len(matches))
        self.assertEqual(1, len(unmatched_left))
        self.assertEqual([], unmatched_right)


if __name__ == "__main__":
    unittest.main()
