import unittest

from src.evaluation import evaluate, iou_xyxy


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.images = [{"name": "image.jpg", "targets": [
            {"category": "car", "bbox": [0, 0, 10, 10]}]}]

    def test_exact_match_has_unit_map(self):
        predictions = {"image.jpg": [{"category": "car", "bbox": [0, 0, 10, 10], "score": 0.9}]}
        result = evaluate(self.images, predictions)
        self.assertAlmostEqual(result["map50_95"], 1.0)
        self.assertAlmostEqual(result["map50"], 1.0)
        self.assertEqual(result["evaluated_classes"], ["car"])

    def test_bad_prediction_scores_zero(self):
        predictions = {"image.jpg": [{"category": "car", "bbox": [20, 20, 30, 30], "score": 0.99}]}
        self.assertAlmostEqual(evaluate(self.images, predictions)["map50_95"], 0.0)

    def test_iou(self):
        self.assertEqual(iou_xyxy([0, 0, 1, 1], [2, 2, 3, 3]), 0.0)
        self.assertAlmostEqual(iou_xyxy([0, 0, 2, 2], [0, 0, 2, 2]), 1.0)


if __name__ == "__main__":
    unittest.main()
