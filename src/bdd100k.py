"""Reproducible BDD100K validation sampling and annotation conversion.

Supports the standard BDD100K list-of-records label JSON, optionally inside a
ZIP, as well as archives of one annotation JSON file per image.
"""

import json
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

# Only classes shared between BDD100K detection and COCO pretrained models.
CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "traffic light")
BDD_TO_SHARED = {"person": "person", "bike": "bicycle", "car": "car",
                 "motor": "motorcycle", "bus": "bus", "train": "train",
                 "truck": "truck", "traffic light": "traffic light"}


def _valid_member(name):
    parts = PurePosixPath(name).parts
    return bool(parts) and not any(
        p == "__MACOSX" or p.startswith(".") or p == ".." for p in parts
    )


def _belongs_to_split(path, split):
    """Identify explicit split markers in directory components or file names."""
    path = PurePosixPath(path)
    parts = [p.lower() for p in path.parts]
    if split.lower() in parts:
        return True
    basename = path.stem.lower()
    return bool(re.search(r"(?:^|[_\-.])" + re.escape(split.lower()) + r"(?:$|[_\-.])", basename))


def _annotated_objects(labels):
    parsed = []
    for label in labels or []:
        shared = BDD_TO_SHARED.get(label.get("category"))
        rect = label.get("box2d")
        if not shared or not isinstance(rect, dict):
            continue
        try:
            box = [float(rect[k]) for k in ("x1", "y1", "x2", "y2")]
        except (ValueError, TypeError, KeyError):
            continue
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        parsed.append({"category": shared, "bbox": box})
    return parsed


def _parse_document(doc, filename):
    """Return name -> target boxes for an aggregate or single-image JSON."""
    if isinstance(doc, dict):
        if "frames" in doc and isinstance(doc["frames"], list):
            records = doc["frames"]
        elif "name" in doc:
            records = [doc]
        elif "labels" in doc:
            base = PurePosixPath(filename).name
            if base.lower().endswith(".json"):
                base = base[:-5]
            if not base.lower().endswith((".jpg", ".jpeg")):
                base += ".jpg"
            records = [{"name": base, "labels": doc["labels"]}]
        else:
            raise ValueError(f"Unsupported label document: {filename}")
    elif isinstance(doc, list):
        records = doc
    else:
        raise ValueError(f"Unsupported label document: {filename}")
    mapping = {}
    for record in records:
        if not isinstance(record, dict) or not record.get("name"):
            continue
        name = PurePosixPath(record["name"]).name
        if name in mapping:
            raise ValueError(f"Duplicate BDD100K annotation for {name}")
        mapping[name] = _annotated_objects(record.get("labels"))
    return mapping


def load_annotations(source, split="val"):
    """Read BDD100K annotations from a JSON, directory, or ZIP.

    Explicit train/val/test markers are honored; non-split-named source JSONs
    are accepted as supplied by caller. Raise on duplicate image names.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Annotations not found: {source}")
    if source.is_dir():
        files = [(p.relative_to(source).as_posix(), p.read_bytes())
                 for p in sorted(source.rglob("*.json"))
                 if _valid_member(p.relative_to(source).as_posix())]
    elif source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            files = [(name, archive.read(name)) for name in archive.namelist()
                     if name.lower().endswith(".json") and _valid_member(name)]
    elif source.suffix.lower() == ".json":
        files = [(source.name, source.read_bytes())]
    else:
        raise ValueError("Annotations must be .json, .zip, or a JSON directory")
    if not files:
        raise ValueError("No annotation JSON files found")
    selected = [(name, contents) for name, contents in files if _belongs_to_split(name, split)]
    if not selected:
        if any(_belongs_to_split(name, s) for name, _ in files for s in ("train", "val", "test")):
            raise ValueError(f"Found annotation splits but no '{split}' labels")
        selected = files
    mapping = {}
    for name, contents in selected:
        doc = json.loads(contents)
        records = _parse_document(doc, name)
        overlap = set(mapping).intersection(records)
        if overlap:
            raise ValueError(f"Duplicate annotation images: {sorted(overlap)[:3]}")
        mapping.update(records)
    if not mapping:
        raise ValueError("No named image annotations found")
    return mapping


def extract_validation_subset(image_zip, annotations, output_dir, *, count=1000,
                              seed=42, split="val", manifest_path=None):
    """Select annotated images reproducibly and extract only selected JPGs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(image_zip) as archive:
        members = [m for m in archive.namelist()
                   if m.lower().endswith((".jpg", ".jpeg")) and _valid_member(m)]
        eligible_split = [m for m in members if split.lower() in [p.lower() for p in PurePosixPath(m).parts]]
        if eligible_split:
            members = eligible_split
        names = {}
        for member in members:
            name = PurePosixPath(member).name
            if name in annotations:
                if name in names:
                    raise ValueError(f"Duplicate annotated image in ZIP: {name}")
                names[name] = member
        if len(names) < count:
            raise ValueError(f"Need {count} annotated {split} images, found {len(names)}")
        manifest = Path(manifest_path) if manifest_path else None
        if manifest and manifest.exists():
            selected = [n.strip() for n in manifest.read_text().splitlines() if n.strip()]
            if len(selected) != count or len(set(selected)) != count or not set(selected).issubset(names):
                raise ValueError("Existing selection manifest does not match available annotated images")
        else:
            selected = sorted(np.random.default_rng(seed).choice(sorted(names), size=count, replace=False).tolist())
        result = []
        for name in sorted(selected):
            target = output_dir / name
            if not target.exists():
                with archive.open(names[name]) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            result.append({"name": name, "image_path": str(target), "targets": annotations[name]})
    if manifest and not manifest.exists():
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("\n".join(sorted(selected)) + "\n")
    return result
