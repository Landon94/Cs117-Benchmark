import json
import tempfile
import unittest
from pathlib import Path

from src.dataset import load_eval_samples


class TestSubsetLoader(unittest.TestCase):
    def test_eval_samples_use_shared_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images" / "test").mkdir(parents=True)
            (root / "labels" / "test").mkdir(parents=True)
            (root / "images" / "test" / "a.jpg").write_bytes(b"x")
            (root / "labels" / "test" / "a.json").write_text(json.dumps({
                "name": "a.jpg",
                "labels": [
                    {"category": "bike", "box2d": {"x1": 0, "y1": 0, "x2": 5, "y2": 5}},
                    {"category": "traffic sign", "box2d": {"x1": 0, "y1": 0, "x2": 5, "y2": 5}},
                ]}))
            splits = root / "splits"
            splits.mkdir()
            (splits / "test.txt").write_text("a.jpg\n")
            out = load_eval_samples(root, "test", splits_dir=splits)
            self.assertEqual(out[0]["name"], "a.jpg")
            self.assertEqual([t["category"] for t in out[0]["targets"]], ["bicycle"])


if __name__ == "__main__":
    unittest.main()
