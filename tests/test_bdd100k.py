import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from src.bdd100k import load_annotations, extract_validation_subset


class TestBDD100K(unittest.TestCase):
    def test_aggregate_json_and_zip_sampling(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = [{"name": f"im{i}.jpg", "labels": [
                {"category": "car", "box2d": {"x1": 1, "y1": 2, "x2": 4, "y2": 6}},
                {"category": "traffic sign", "box2d": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}}
            ]} for i in range(4)]
            labels = root / "labels.zip"
            images = root / "images.zip"
            with zipfile.ZipFile(labels, "w") as z:
                z.writestr("bdd100k_labels_images_val.json", json.dumps(records))
                z.writestr("__MACOSX/._extra.json", "broken")
            with zipfile.ZipFile(images, "w") as z:
                for i in range(4):
                    z.writestr(f"bdd/images/val/im{i}.jpg", b"image")
            parsed = load_annotations(labels)
            self.assertEqual(len(parsed), 4)
            self.assertEqual(parsed["im0.jpg"][0]["category"], "car")
            self.assertEqual(len(parsed["im0.jpg"]), 1)
            manifest = root / "split.txt"
            a = extract_validation_subset(images, parsed, root / "out", count=2,
                                          seed=42, manifest_path=manifest)
            b = extract_validation_subset(images, parsed, root / "out", count=2,
                                          seed=7, manifest_path=manifest)
            self.assertEqual([x["name"] for x in a], [x["name"] for x in b])
            self.assertEqual(len(manifest.read_text().splitlines()), 2)

    def test_per_image_json(self):
        with tempfile.TemporaryDirectory() as temp:
            labels = Path(temp) / "per_image.zip"
            with zipfile.ZipFile(labels, "w") as z:
                z.writestr("val/example.jpg.json", json.dumps({"labels": [
                    {"category": "person", "box2d": {"x1": 0, "y1": 0,
                     "x2": 4, "y2": 4}}]}))
            annotations = load_annotations(labels)
            self.assertIn("example.jpg", annotations)
            self.assertEqual(annotations["example.jpg"][0]["category"], "person")


if __name__ == "__main__":
    unittest.main()
