from pathlib import Path
import tempfile
import unittest
import zipfile

from tools.build_easy_pose_seed_task import (
    make_pose_row,
    map_apic_keypoints,
    write_cvat_archive,
)


class EasyPoseSeedTaskTests(unittest.TestCase):
    def test_maps_apic_axis_to_three_competition_points(self):
        raw = {
            "0": [10, 10], "1": [14, 10], "2": [10, 14], "3": [14, 14],
            "4": [20, 20], "5": [30, 30], "6": [8, 8],
        }

        self.assertEqual(
            [(12.0, 12.0), (20.0, 20.0), (30.0, 30.0)],
            map_apic_keypoints(raw))
        fields = make_pose_row(raw, 40, 40).split()
        self.assertEqual(14, len(fields))
        self.assertEqual(["2", "2", "2"], fields[7:14:3])

    def test_rejects_incomplete_apic_axis(self):
        with self.assertRaisesRegex(ValueError, "缺少"):
            map_apic_keypoints({"0": [1, 1]})

    def test_writes_one_clean_cvat_subset_with_empty_ir_label(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "task.zip"
            report = write_cvat_archive(output, [
                ("01_visible.jpg", b"visible", "0 0.5 0.5 0.2 0.2 0.4 0.4 2 0.5 0.5 2 0.6 0.6 2"),
                ("02_ir.jpg", b"ir", ""),
            ])

            self.assertEqual({"images": 2, "annotation_rows": 1}, report)
            with zipfile.ZipFile(output) as archive:
                self.assertEqual({
                    "data.yaml", "train.txt",
                    "images/train/01_visible.jpg", "images/train/02_ir.jpg",
                    "labels/train/01_visible.txt", "labels/train/02_ir.txt",
                }, set(archive.namelist()))
                self.assertEqual(
                    ["./images/train/01_visible.jpg", "./images/train/02_ir.jpg"],
                    archive.read("train.txt").decode("utf-8").splitlines())
                self.assertEqual(b"", archive.read("labels/train/02_ir.txt"))
                for name in ("data.yaml", "train.txt", "labels/train/01_visible.txt"):
                    self.assertNotIn(b"\r", archive.read(name))


if __name__ == "__main__":
    unittest.main()
