"""Pretrained YOLO11 / RT-DETR predictions on a fixed BDD100K image list.

The models use COCO-pretrained weights. Results use normalized shared labels
and record per-image inference time from Ultralytics (excluding data loading).
"""

import statistics
from pathlib import Path

from src.bdd100k import CLASSES

COCO_ALIASES = {"person": "person", "bicycle": "bicycle", "car": "car",
                "motorcycle": "motorcycle", "bus": "bus", "train": "train",
                "truck": "truck", "traffic light": "traffic light"}


def normalize_result(result, name):
    """Convert one Ultralytics Result object into JSON-ready predictions."""
    detections = []
    if result.boxes is None:
        return detections
    class_names = result.names
    for xyxy, confidence, class_id in zip(result.boxes.xyxy.tolist(),
                                         result.boxes.conf.tolist(),
                                         result.boxes.cls.tolist()):
        label = class_names[int(class_id)]
        category = COCO_ALIASES.get(label)
        if category in CLASSES:
            detections.append({"category": category, "bbox": [float(x) for x in xyxy],
                               "score": float(confidence)})
    return detections


def load_model(model_name):
    """Load supported COCO-pretrained checkpoints, no training or fine-tuning."""
    from ultralytics import RTDETR, YOLO

    lower = model_name.lower()
    if lower.startswith("rtdetr"):
        return RTDETR(model_name)
    if lower.startswith("yolo11"):
        return YOLO(model_name)
    raise ValueError("Use a YOLO11 or RT-DETR pretrained checkpoint")


def predict_samples(model, samples, *, imgsz=640, device=None, conf=0.001):
    """Evaluate the exact supplied sample list, no unmeasured example scores."""
    if not samples:
        raise ValueError("Cannot benchmark an empty image list")
    predictions = {}
    times_ms = []
    for sample in samples:
        name = sample["name"]
        image_path = sample["image_path"]
        if not Path(image_path).is_file():
            raise FileNotFoundError(image_path)
        kwargs = {"source": image_path, "imgsz": imgsz, "conf": conf,
                  "verbose": False, "max_det": 100}
        if device is not None:
            kwargs["device"] = device
        outputs = model.predict(**kwargs)
        if len(outputs) != 1:
            raise RuntimeError(f"Expected one result for {name}, got {len(outputs)}")
        result = outputs[0]
        predictions[name] = normalize_result(result, name)
        speed = getattr(result, "speed", {}) or {}
        if "inference" in speed:
            times_ms.append(float(speed["inference"]))
    stats = {"reported_inference_ms": len(times_ms),
             "mean_inference_ms": statistics.mean(times_ms) if times_ms else None,
             "median_inference_ms": statistics.median(times_ms) if times_ms else None}
    return predictions, stats
