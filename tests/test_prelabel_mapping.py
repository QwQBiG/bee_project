import unittest

from tools.prelabel_yolo import canonical_class, result_instances


class TensorStub:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value


class BoxStub:
    xyxy = TensorStub([[10, 20, 30, 50]])
    cls = TensorStub([2])
    conf = TensorStub([0.75])


class ResultStub:
    boxes = BoxStub()
    keypoints = None


class PrelabelMappingTests(unittest.TestCase):
    def test_known_model_classes_are_canonicalized(self):
        self.assertEqual("worker_bee", canonical_class("Worker Bee"))
        self.assertEqual("pollen_bee", canonical_class("pollenbee"))
        self.assertEqual("varroa_mite", canonical_class("Varroa Mite"))

    def test_unknown_class_falls_back_to_bee(self):
        self.assertEqual("bee", canonical_class("honey_bee_unknown"))

    def test_detection_becomes_prediction_instance(self):
        instances = result_instances(ResultStub(), 8, {2: "Queen Bee"})
        self.assertEqual(1, len(instances))
        self.assertEqual("queen_bee", instances[0].category)
        self.assertEqual([10.0, 20.0, 20.0, 30.0], instances[0].bbox)
        self.assertEqual("prediction", instances[0].source)
        self.assertEqual(0.75, instances[0].confidence)


if __name__ == "__main__":
    unittest.main()
