import unittest
from unittest.mock import patch

from utils.common import normalize_device


class DeviceSelectionTests(unittest.TestCase):
    def test_numeric_cuda_device_is_normalized(self):
        self.assertEqual(normalize_device("0"), "cuda:0")
        self.assertEqual(normalize_device(2), "cuda:2")

    def test_named_devices_are_supported(self):
        self.assertEqual(normalize_device("cuda"), "cuda:0")
        self.assertEqual(normalize_device("CUDA:01"), "cuda:1")
        self.assertEqual(normalize_device("mps"), "mps")
        self.assertEqual(normalize_device("cpu"), "cpu")

    @patch("utils.common.get_device", return_value="cpu")
    def test_auto_uses_runtime_detection(self, get_device):
        self.assertEqual(normalize_device("auto"), "cpu")
        self.assertEqual(normalize_device(None), "cpu")
        self.assertEqual(get_device.call_count, 2)

    def test_invalid_device_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "不支持的设备写法"):
            normalize_device("gpu")
        with self.assertRaisesRegex(ValueError, "无效 CUDA 设备"):
            normalize_device("cuda:first")


if __name__ == "__main__":
    unittest.main()
