import unittest

from tools.plan_grouped_split import (
    assign_grouped_splits,
    audit_split,
    scene_split_counts,
)


def video(scene, index, digest=None):
    return {
        "video_id": f"{scene}-{index}",
        "source_path": f"D:/data/{scene}-{index}.mp4",
        "sha256": digest or f"hash-{scene}-{index}",
        "scene": scene,
    }


class GroupedSplitTests(unittest.TestCase):
    def test_four_videos_per_scene_use_two_one_one(self):
        self.assertEqual({"train": 2, "val": 1, "test": 1}, scene_split_counts(4))

    def test_assignment_is_scene_balanced_and_stable(self):
        videos = [video(scene, index) for scene in ("inside_ir", "outside_entrance")
                  for index in range(4)]
        first = assign_grouped_splits(videos, "seed")
        second = assign_grouped_splits(list(reversed(videos)), "seed")
        self.assertEqual([(item["video_id"], item["split"]) for item in first],
                         [(item["video_id"], item["split"]) for item in second])
        audit = audit_split(first)
        self.assertTrue(audit["valid"])
        self.assertEqual({"train": 2, "val": 1, "test": 1},
                         audit["scene_split_counts"]["inside_ir"])

    def test_duplicate_hash_across_splits_is_detected(self):
        items = [
            {**video("inside_ir", 1, "same"), "split": "train"},
            {**video("inside_ir", 2, "same"), "split": "test"},
            {**video("inside_ir", 3), "split": "val"},
        ]
        audit = audit_split(items)
        self.assertFalse(audit["valid"])
        self.assertTrue(any("sha256" in error for error in audit["errors"]))

    def test_same_video_never_appears_twice(self):
        videos = [video("outside_entrance", index) for index in range(4)]
        assigned = assign_grouped_splits(videos)
        self.assertEqual(len(assigned), len({item["video_id"] for item in assigned}))


if __name__ == "__main__":
    unittest.main()
