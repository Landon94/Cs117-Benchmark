import json
import tempfile
import unittest
from pathlib import Path

from src.run_checkin import build_parser, write_results


class TestCheckinRunner(unittest.TestCase):
    def test_measured_result_saved_and_placeholder_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "results" / "baseline_val.json"
            with self.assertRaises(ValueError):
                write_results(target, {"measurements": {"map50_95": None}})
            self.assertFalse(target.exists())
            write_results(target, {"measurements": {"map50_95": 0.1234}})
            self.assertEqual(json.loads(target.read_text())["measurements"]["map50_95"], 0.1234)

    def test_parser_defaults(self):
        args = build_parser().parse_args(["--images", "images.zip", "--labels", "labels.zip"])
        self.assertEqual((args.split, args.count, args.seed, args.model),
                         ("val", 1000, 42, "yolo11m.pt"))


if __name__ == "__main__":
    unittest.main()
