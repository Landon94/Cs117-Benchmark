# import yaml
# import json
# import numpy as np
# from pathlib import Path


# def load_config(config_path="configs/experiment.yaml"):
#     with open(config_path, "r") as f:
#         return yaml.safe_load(f)


# def create_test_split(image_dir, label_dir, config, rng):
#     """
#     Returns sorted list of randomly selected image paths with annotations
#     """
#     image_dir = Path(image_dir)
#     label_dir = Path(label_dir)
    
#     test_size = config["dataset"]["test_size"]

#     images = sorted(Path(image_dir).glob("*.jpg"))
#     test_images = rng.choice(images, size=test_size, replace=False)

#     test_data = get_labels_for_images(label_dir, test_images)

#     return sorted(test_data, key=lambda x: x["image_path"])


# def get_labels_for_images(label_dir, images):
#     """
#     Given the selected images, match and return the corresponding labels for that image
#     """
#     test_data = []

#     for path in images:
#         image_name = Path(path).stem

#         json_path = label_dir / f"{image_name}.json"
        
#         if not json_path.exists():
#             raise FileNotFoundError(f"Missing annotation: {json_path}")
        
#         with open(json_path, "r") as f:
#             annotation = json.load(f)
    
#         test_data.append(
#             {"image_path": path, "image_annotations": annotation}
#         )
    
#     return test_data


# if __name__ == "__main__":
#     # TEST
#     rng = np.random.default_rng(1234)
#     config = load_config()
#     res = create_test_split("/Users/landonkarkut/Downloads/100k_images/test", "/Users/landonkarkut/Downloads/100k_labels/test", config, rng)
#     print(res[0])


"""BDD100K dataset helpers for local directories and Colab ZIP archives."""

import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import yaml


def load_config(config_path="configs/experiment.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_labels_for_images(label_dir, images):
    """Pair extracted image paths with their per-image JSON annotations."""
    label_dir = Path(label_dir)
    samples = []

    for image_path in images:
        image_path = Path(image_path)
        label_path = label_dir / f"{image_path.stem}.json"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing annotation: {label_path}")

        with label_path.open("r") as f:
            annotation = json.load(f)
        samples.append({
            "image_path": image_path,
            "image_annotations": annotation,
        })

    return sorted(samples, key=lambda sample: sample["image_path"].name)


def create_test_split(image_dir, label_dir, config, rng):
    """Original helper: sample from images and annotations already extracted."""
    test_size = config["dataset"]["test_size"]
    images = sorted(Path(image_dir).glob("*.jpg"))
    if len(images) < test_size:
        raise ValueError(f"Only {len(images)} JPGs available; need {test_size}")
    chosen = rng.choice(images, size=test_size, replace=False)
    return get_labels_for_images(label_dir, chosen)


def _entries_for_split(entries, split):
    """Pick entries in the requested split; allow archives holding only one split."""
    entries = list(entries)
    split_entries = [
        name for name in entries
        if split.lower() in (part.lower() for part in PurePosixPath(str(name)).parts)
    ]
    if split_entries:
        return split_entries

    known_splits = {"train", "val", "test"}
    if any(
        known_splits.intersection(part.lower() for part in PurePosixPath(str(n)).parts)
        for n in entries
    ):
        raise ValueError(f"Found split folders, but none for '{split}'")
    return entries


def _label_stem(path):
    """Accept both <id>.json and <id>.jpg.json label filenames."""
    stem = PurePosixPath(str(path)).name
    if stem.lower().endswith(".json"):
        stem = stem[:-5]
    if stem.lower().endswith(".jpg"):
        stem = stem[:-4]
    return stem


def _unique_index(entries, key_fn, source_name):
    index = {}
    for entry in entries:
        key = key_fn(entry)
        if key in index:
            raise ValueError(f"Duplicate filename '{key}' in {source_name}")
        index[key] = entry
    return index


def create_test_split_from_zip(
    image_zip,
    label_source,
    output_dir,
    config,
    rng,
    split="val",
    manifest_path=None,
):
    """Select a reproducible test subset without extracting the full BDD100K ZIP.

    Args:
        image_zip: ZIP in Drive containing JPGs, e.g. .../bdd100k_data/images.zip.
        label_source: ZIP of per-image JSONs OR a directory containing per-image JSONs.
        output_dir: Local Colab workspace, e.g. /content/bdd100k_data.
        config: Dict with config['dataset']['test_size'].
        rng: numpy Generator, created with the configured seed.
        split: BDD100K split to sample from (default 'val').
        manifest_path: Optional file of selected JPG basenames. If it exists,
            reuse it; otherwise generate it and save it for reproducibility.

    Returns:
        Sorted list of {'image_path': Path, 'image_annotations': dict}.
    """
    image_zip = Path(image_zip)
    label_source = Path(label_source)
    output_dir = Path(output_dir)
    test_size = int(config["dataset"]["test_size"])

    if not image_zip.is_file():
        raise FileNotFoundError(f"Image ZIP not found: {image_zip}")
    if not label_source.exists():
        raise FileNotFoundError(f"Labels not found: {label_source}")

    local_images = output_dir / "images" / split
    local_labels = output_dir / "labels" / split
    local_images.mkdir(parents=True, exist_ok=True)
    local_labels.mkdir(parents=True, exist_ok=True)

    # Only inspect ZIP member names: no pixel data is decoded or extracted yet.
    with zipfile.ZipFile(image_zip) as images_archive:
        jpg_members = _entries_for_split(
            (m for m in images_archive.namelist()
             if m.lower().endswith((".jpg", ".jpeg")) and not m.endswith("/")),
            split,
        )
        image_index = _unique_index(
            jpg_members,
            lambda m: PurePosixPath(m).name,
            image_zip.name,
        )

        label_archive = None
        try:
            if label_source.is_file() and label_source.suffix.lower() == ".zip":
                label_archive = zipfile.ZipFile(label_source)
                json_members = _entries_for_split(
                    (m for m in label_archive.namelist()
                     if m.lower().endswith(".json") and not m.endswith("/")),
                    split,
                )
                label_index = _unique_index(
                    json_members, _label_stem, label_source.name
                )
            elif label_source.is_dir():
                json_paths = sorted(label_source.rglob("*.json"))
                json_paths = _entries_for_split(
                    (p.relative_to(label_source).as_posix() for p in json_paths),
                    split,
                )
                label_index = _unique_index(
                    json_paths, _label_stem, str(label_source)
                )
                label_index = {
                    k: label_source / v for k, v in label_index.items()
                }
            else:
                raise ValueError("label_source must be a ZIP or a directory of JSON files")

            # Avoid sampling any image that has no corresponding annotation.
            eligible_names = sorted(
                filename for filename in image_index
                if Path(filename).stem in label_index
            )
            if test_size > len(eligible_names):
                raise ValueError(
                    f"Requested {test_size} images, but only {len(eligible_names)} "
                    f"annotated {split} images found in the archives"
                )

            if manifest_path is not None:
                manifest_path = Path(manifest_path)

            if manifest_path is not None and manifest_path.is_file():
                selected = [
                    name.strip() for name in manifest_path.read_text().splitlines()
                    if name.strip()
                ]
                if len(selected) != test_size or len(set(selected)) != test_size:
                    raise ValueError("Existing split manifest has the wrong count or duplicates")
                missing = set(selected) - set(eligible_names)
                if missing:
                    raise ValueError(f"Manifest references missing images: {sorted(missing)[:5]}")
            else:
                selected = sorted(
                    rng.choice(eligible_names, size=test_size, replace=False).tolist()
                )
                if manifest_path is not None:
                    manifest_path.parent.mkdir(parents=True, exist_ok=True)
                    manifest_path.write_text("\n".join(selected) + "\n")

            selected_paths = []
            for filename in sorted(selected):
                src_member = image_index[filename]
                target_img = local_images / filename
                if not target_img.is_file():
                    # Write to a fixed local path, avoiding any unsafe archive paths.
                    with images_archive.open(src_member) as src, target_img.open("wb") as dst:
                        shutil.copyfileobj(src, dst)

                label_member = label_index[Path(filename).stem]
                if label_archive is not None:
                    label_bytes = label_archive.read(label_member)
                else:
                    label_bytes = Path(label_member).read_bytes()

                # Check the JSON before writing the selected label file.
                json.loads(label_bytes)
                (local_labels / f"{Path(filename).stem}.json").write_bytes(label_bytes)
                selected_paths.append(target_img)

        finally:
            if label_archive is not None:
                label_archive.close()

    return get_labels_for_images(local_labels, selected_paths)
