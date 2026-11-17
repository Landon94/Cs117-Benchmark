"""COCO-style 101-point AP for the shared BDD100K / COCO categories.

This project metric is mAP@0.5:0.95 over classes with ground-truth objects.
We do not implement COCO's crowd, area-range, or multi-maxDet breakdown.
Report the metric with these limitations, rather than as official COCOeval.
"""

from src.bdd100k import CLASSES

IOU_THRESHOLDS = tuple(round(0.50 + i * 0.05, 2) for i in range(10))


def iou_xyxy(a, b):
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0.0, right-left) * max(0.0, bottom-top)
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _ap_for_class(samples, predictions, category, iou_threshold):
    targets = {}
    detections = []
    total_gt = 0
    for sample in samples:
        name = sample["name"]
        boxes = [obj["bbox"] for obj in sample["targets"]
                 if obj["category"] == category]
        targets[name] = boxes
        total_gt += len(boxes)
        current = [d for d in predictions.get(name, []) if d["category"] == category]
        # Stable sort keeps exact ties deterministic.
        for detection in sorted(current, key=lambda d: -d["score"])[:100]:
            detections.append((detection["score"], name, detection["bbox"]))
    if total_gt == 0:
        return None
    detections.sort(key=lambda d: (-d[0], d[1]))
    matched = {name: set() for name in targets}
    true_pos = []
    false_pos = []
    for _, name, box in detections:
        candidates = [(iou_xyxy(box, gt_box), idx)
                      for idx, gt_box in enumerate(targets[name])
                      if idx not in matched[name]]
        best_iou, best_idx = max(candidates, default=(0.0, -1))
        if best_iou >= iou_threshold:
            matched[name].add(best_idx)
            true_pos.append(1)
            false_pos.append(0)
        else:
            true_pos.append(0)
            false_pos.append(1)
    tp = fp = 0
    recalls = []
    precisions = []
    for is_tp, is_fp in zip(true_pos, false_pos):
        tp += is_tp
        fp += is_fp
        recalls.append(tp / total_gt)
        precisions.append(tp / (tp + fp))
    # COCO convention: interpolate precision at 101 evenly spaced recalls.
    return sum(max((p for p, r in zip(precisions, recalls) if r >= k / 100),
                   default=0.0) for k in range(101)) / 101


def evaluate(samples, predictions, classes=CLASSES):
    """Return an auditable scalar mAP and per-class AP by IoU threshold."""
    names = [sample["name"] for sample in samples]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate image identifiers in evaluation sample")
    if set(predictions) - set(names):
        raise ValueError("Predictions contain images outside the evaluation split")
    per_class = {}
    all_ap = []
    ap50 = []
    for category in classes:
        by_threshold = {}
        for threshold in IOU_THRESHOLDS:
            score = _ap_for_class(samples, predictions, category, threshold)
            if score is not None:
                by_threshold[f"{threshold:.2f}"] = score
                all_ap.append(score)
                if threshold == 0.50:
                    ap50.append(score)
        if by_threshold:
            per_class[category] = by_threshold
    if not all_ap:
        raise ValueError("No supported ground-truth categories found in this split")
    return {"metric": "mAP@0.5:0.95", "map50_95": sum(all_ap) / len(all_ap),
            "map50": sum(ap50) / len(ap50), "evaluated_classes": list(per_class),
            "per_class_ap": per_class, "n_images": len(samples),
            "notes": "101-point AP on shared classes; excludes crowd/area COCOeval options"}
