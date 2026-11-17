"""Reproducible measured baseline runner for CS117's BDD100K check-in.

Example:
  python -m src.run_checkin --images /path/images.zip --labels /path/labels.zip
  python -m src.run_checkin --subset data/bdd100k_subset --split test
The output does NOT contain a score until the actual model runs successfully.
"""

import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.bdd100k import CLASSES, extract_validation_subset, load_annotations
from src.dataset import load_eval_samples
from src.evaluation import evaluate
from src.inference import load_model, predict_samples


def write_results(path, record):
    """Write a JSON result atomically; never create a fabricated placeholder score."""
    if not isinstance(record.get("measurements", {}).get("map50_95"), (int, float)):
        raise ValueError("Refusing to write baseline without measured mAP")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def build_parser():
    parser = argparse.ArgumentParser(description="CS117 BDD100K measured baseline")
    parser.add_argument("--subset", help="Prepared subset dir (src.prepare_subset); "
                        "uses splits/<split>.txt and ignores --images/--labels/--count")
    parser.add_argument("--images", help="BDD100K images ZIP")
    parser.add_argument("--labels", help="BDD100K labels JSON or ZIP")
    parser.add_argument("--model", default="yolo11m.pt", help="yolo11m.pt or rtdetr-l.pt")
    parser.add_argument("--split", default=None, choices=("train", "val", "test"),
                        help="default: test with --subset, else val")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-dir", default=".cache/bdd100k_checkin")
    parser.add_argument("--manifest", default="splits/checkin_val.txt")
    parser.add_argument("--output", default="results/baseline_val.json")
    parser.add_argument("--predictions-output", default="results/baseline_predictions_val.json")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.count <= 0 or args.imgsz <= 0:
        parser.error("count and imgsz must be positive")
    if args.subset:
        args.split = args.split or "test"
        samples = load_eval_samples(args.subset, args.split)
        image_path, label_path = Path(args.subset), Path(args.subset)
        args.manifest = f"splits/{args.split}.txt"
    else:
        args.split = args.split or "val"
        if not args.images or not args.labels:
            parser.error("Provide --subset, or both --images and --labels")
        image_path = Path(args.images)
        label_path = Path(args.labels)
        if not image_path.is_file() or not label_path.exists():
            parser.error("Provide accessible --images ZIP and --labels JSON/ZIP/directory")
        annotations = load_annotations(label_path, split=args.split)
        samples = extract_validation_subset(
            image_path, annotations, args.cache_dir, count=args.count,
            seed=args.seed, split=args.split, manifest_path=args.manifest)
    model = load_model(args.model)
    predictions, timing = predict_samples(model, samples, imgsz=args.imgsz, device=args.device)
    metrics = evaluate(samples, predictions)

    command = "python -m src.run_checkin " + " ".join(
        shlex.quote(a) for a in (argv if argv is not None else sys.argv[1:]))
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "pretrained": True,
        "dataset": "BDD100K", "split": args.split,
        "image_count": len(samples), "seed": args.seed, "imgsz": args.imgsz,
        "classes_evaluated": list(CLASSES),
        "excluded_bdd_categories": ["rider", "traffic sign"],
        "source_images": str(image_path), "source_labels": str(label_path),
        "selection_manifest": args.manifest,
        "method": "101-point shared-class AP, IoU 0.50:0.95",
        "metric_note": "Not full official COCOeval (no crowd/area-range breakdown)",
        "measurements": metrics, "timing": timing, "command": command,
    }
    # Do not save a result until predictions have been evaluated successfully.
    write_results(args.output, record)
    out_pred = Path(args.predictions_output)
    out_pred.parent.mkdir(parents=True, exist_ok=True)
    out_pred.write_text(json.dumps(predictions, indent=2) + "\n")
    print(f"Measured {args.model} {metrics['metric']} = {metrics['map50_95']:.4f} "
          f"on {len(samples)} BDD100K {args.split} images")
    print(f"Saved: {args.output} and {args.predictions_output}")


if __name__ == "__main__":
    main()
