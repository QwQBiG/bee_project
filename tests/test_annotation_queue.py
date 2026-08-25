import tempfile
from pathlib import Path
import unittest

from tools.build_annotation_queue import (
    FrameCandidate,
    frame_features,
    parse_yolo_boxes,
    score_candidates,
    select_diverse_queue,
)


def candidate(frame, count, overlap=0.0, small=0.0, edge=0.0, area=0.01):
    return FrameCandidate(
        image_id=f"frame_{frame}", video_id="video", frame_index=frame,
        image_path=f"images/{frame}.jpg", label_path=f"labels/{frame}.txt",
        detection_count=count, small_object_fraction=small,
        edge_object_fraction=edge, overlapping_object_fraction=overlap,
        mean_box_area=area,
    )


class AnnotationQueueTests(unittest.TestCase):
    def test_frame_features_detect_difficulty_signals(self):
        boxes = [(0.02, 0.02, 0.02, 0.02), (0.025, 0.025, 0.02, 0.02)]
        features = frame_features(boxes)
        self.assertEqual(2, features["detection_count"])
        self.assertEqual(1.0, features["small_object_fraction"])
        self.assertEqual(1.0, features["edge_object_fraction"])
        self.assertEqual(1.0, features["overlapping_object_fraction"])

    def test_invalid_yolo_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.txt"
            path.write_text("0 0.5 0.5 -0.1 0.2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_yolo_boxes(path)

    def test_difficult_frame_receives_higher_priority(self):
        easy = candidate(0, 2, area=0.02)
        difficult = candidate(50, 10, overlap=0.8, small=0.8, edge=0.5, area=0.0005)
        score_candidates([easy, difficult])
        self.assertGreater(difficult.priority_score, easy.priority_score)
        self.assertIn("目标重叠或遮挡", difficult.review_reasons)

    def test_selection_respects_gap_before_fallback(self):
        items = [candidate(frame, 5) for frame in (0, 10, 50, 100)]
        for index, item in enumerate(items):
            item.priority_score = 1.0 - index * 0.1
        selected = select_diverse_queue(items, count=3, min_frame_gap=40)
        self.assertEqual([0, 50, 100], [item.frame_index for item in selected])

    def test_selection_is_deterministic(self):
        items = score_candidates([candidate(frame, frame // 10 + 1) for frame in (0, 50, 100)])
        first = select_diverse_queue(items, 2, 20)
        second = select_diverse_queue(items, 2, 20)
        self.assertEqual([item.frame_index for item in first],
                         [item.frame_index for item in second])


    def test_multivideo_queue_balances_quota(self):
        items = []
        for video_id in ("first", "second"):
            for frame in (0, 50, 100):
                item = candidate(frame, frame // 50 + 1)
                item.video_id = video_id
                item.image_id = f"{video_id}_{frame}"
                items.append(item)
        selected = select_diverse_queue(score_candidates(items), 4, 20)
        self.assertEqual({"first": 2, "second": 2}, dict(__import__('collections').Counter(item.video_id for item in selected)))
if __name__ == "__main__":
    unittest.main()
