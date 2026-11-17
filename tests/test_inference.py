import unittest
from types import SimpleNamespace

from src.inference import normalize_result, load_model


class TestInference(unittest.TestCase):
    def test_shared_class_filter(self):
        boxes = SimpleNamespace(xyxy=SimpleNamespace(tolist=lambda: [[0, 0, 3, 4], [0, 0, 5, 5]]),
                                conf=SimpleNamespace(tolist=lambda: [0.9, 0.7]),
                                cls=SimpleNamespace(tolist=lambda: [0.0, 1.0]))
        result = SimpleNamespace(boxes=boxes, names={0: "car", 1: "chair"})
        out = normalize_result(result, "abc.jpg")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["category"], "car")


if __name__ == "__main__":
    unittest.main()
