from pathlib import Path
import tempfile
import unittest
import zipfile

from tools.build_cvat_pose_upload import build_single_task_archive


class CvatPoseUploadTests(unittest.TestCase):
    def make_source(self, path: Path, duplicate: bool = False) -> None:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("data.yaml", "path: ./\r\ntrain: train.txt\r\nval: val.txt\r\n")
            archive.writestr("annotation_map.json", "{}\r\n")
            archive.writestr("dataset_meta.json", "{}\r\n")
            archive.writestr("images/train/bee-a.jpg", b"image-a")
            archive.writestr("labels/train/bee-a.txt", (
                "0 0.5 0.5 0.2 0.2 0.4 0.5 2 0.5 0.5 2 0.6 0.5 2\r\n"))
            second_name = "bee-a" if duplicate else "bee-b"
            archive.writestr(f"images/val/{second_name}.jpg", b"image-b")
            archive.writestr(f"labels/val/{second_name}.txt", "")

    def test_merges_train_and_val_into_one_clean_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.zip", root / "upload.zip"
            self.make_source(source)

            report = build_single_task_archive(source, output)

            self.assertEqual(2, report["images"])
            self.assertEqual(["train"], report["cvat_subsets"])
            self.assertEqual(1, report["annotation_rows"])
            with zipfile.ZipFile(output) as archive:
                names = set(archive.namelist())
                self.assertEqual({
                    "data.yaml", "train.txt",
                    "images/train/bee-a.jpg", "images/train/bee-b.jpg",
                    "labels/train/bee-a.txt", "labels/train/bee-b.txt",
                }, names)
                self.assertNotIn(b"\r", archive.read("data.yaml"))
                self.assertNotIn(b"\r", archive.read("labels/train/bee-a.txt"))
                self.assertEqual(
                    ["./images/train/bee-a.jpg", "./images/train/bee-b.jpg"],
                    archive.read("train.txt").decode("utf-8").splitlines())

    def test_rejects_duplicate_names_across_subsets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.zip"
            self.make_source(source, duplicate=True)

            with self.assertRaisesRegex(ValueError, "文件名重复"):
                build_single_task_archive(source, root / "upload.zip")

    def test_rejects_detection_box_disguised_as_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.zip", root / "upload.zip"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("images/train/bee.jpg", b"image")
                archive.writestr(
                    "labels/train/bee.txt",
                    "0 0.5 0.5 0.2 0.2 0 0 0 0 0 0 0 0 0\n")

            with self.assertRaisesRegex(ValueError, "不是真实姿态预标注"):
                build_single_task_archive(source, output)


if __name__ == "__main__":
    unittest.main()
