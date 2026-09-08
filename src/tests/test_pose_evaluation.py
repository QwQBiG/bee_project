import unittest

from annotation.schema import BeeInstance, FrameAnnotation, Keypoint, VideoAnnotation
from tools.evaluate_pose import evaluate_pose


def instance(instance_id, head, thorax, abdomen, source="manual"):
    return BeeInstance(
        instance_id, [0, 0, 100, 50], source=source,
        confidence=0.9 if source == "prediction" else None,
        keypoints=[Keypoint("head", *head), Keypoint("thorax", *thorax),
                   Keypoint("abdomen_tip", *abdomen)],
    )


class PoseEvaluationTests(unittest.TestCase):
    def annotation(self, bee_instance):
        return VideoAnnotation(
            video_id="pose-1", source_path="pose.mp4", scene="inside_ir",
            width=200, height=100, fps=30.0, frame_count=2,
            frames=[FrameAnnotation(0, 0.0, [bee_instance])],
        )

    def test_perfect_pose(self):
        truth = self.annotation(instance("gt", (80, 25), (50, 25), (20, 25)))
        prediction = self.annotation(instance("pred", (80, 25), (50, 25), (20, 25), "prediction"))
        report = evaluate_pose(truth, prediction)
        self.assertEqual(1.0, report["pck"])
        self.assertEqual(0.0, report["mean_orientation_error_degrees"])
        self.assertEqual(0.0, report["head_tail_reversal_rate"])

    def test_head_tail_reversal(self):
        truth = self.annotation(instance("gt", (80, 25), (50, 25), (20, 25)))
        prediction = self.annotation(instance("pred", (20, 25), (50, 25), (80, 25), "prediction"))
        report = evaluate_pose(truth, prediction, pck_threshold=1.0)
        self.assertEqual(180.0, report["mean_orientation_error_degrees"])
        self.assertEqual(1.0, report["head_tail_reversal_rate"])

    def test_invalid_ground_truth_is_rejected(self):
        truth = self.annotation(instance("gt", (80, 25), (50, 25), (20, 25), "prediction"))
        prediction = self.annotation(instance("pred", (80, 25), (50, 25), (20, 25), "prediction"))
        with self.assertRaises(ValueError):
            evaluate_pose(truth, prediction)


if __name__ == "__main__":
    unittest.main()
