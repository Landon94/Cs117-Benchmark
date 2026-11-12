
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
    """Sample from images and annotations already extracted."""

    test_size = config["dataset"]["test_size"]

    images = sorted(
        p for p in Path(image_dir).glob("*.jpg")
        if _is_real_archive_member(p.name)
    )

    if len(images) < test_size:
        raise ValueError(
            f"Only {len(images)} JPGs available; need {test_size}"
        )

    chosen = rng.choice(images, size=test_size, replace=False)

    return get_labels_for_images(label_dir, chosen)


def _is_real_archive_member(name):
    """
    Ignore macOS metadata files such as:
      - __MACOSX/
      - ._filename.jpg
      - ._filename.json
      - .DS_Store

    Check every directory component, not just the filename.
    """
    parts = PurePosixPath(str(name)).parts

    return not any(
        part == "__MACOSX" or part.startswith(".")
        for part in parts
    )


def _entries_for_split(entries, split):
    """
    Select files belonging to the requested dataset split.
    Also supports archives containing only one split.
    """
    entries = list(entries)

    split_entries = [
        name for name in entries
        if split.lower() in (
            part.lower()
            for part in PurePosixPath(str(name)).parts
        )
    ]

    if split_entries:
        return split_entries

    known_splits = {"train", "val", "test"}

    if any(
        known_splits.intersection(
            part.lower()
            for part in PurePosixPath(str(name)).parts
        )
        for name in entries
    ):
        raise ValueError(
            f"Found split folders, but none for '{split}'"
        )

    return entries


def _label_stem(path):
    """Accept both <id>.json and <id>.jpg.json labels."""

    stem = PurePosixPath(str(path)).name

    if stem.lower().endswith(".json"):
        stem = stem[:-5]

    if stem.lower().endswith(".jpg"):
        stem = stem[:-4]

    return stem


def _unique_index(entries, key_fn, source_name):
    """Build a lookup dictionary and reject duplicate filenames."""

    index = {}

    for entry in entries:
        key = key_fn(entry)

        if key in index:
            raise ValueError(
                f"Duplicate filename '{key}' in {source_name}"
            )

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
    """
    Randomly select annotated images directly from BDD100K ZIP archives.

    Only selected images and their JSON labels are extracted.

    Args:
        image_zip:
            Path to images.zip in Google Drive.

        label_source:
            Path to labels.zip or an extracted labels directory.

        output_dir:
            Local Colab extraction directory.

        config:
            Experiment configuration dictionary.

        rng:
            NumPy random generator.

        split:
            Dataset split, default "val".

        manifest_path:
            Optional text file storing selected filenames.
            Reuse the file if it already exists.

    Returns:
        Sorted list of dictionaries containing:
            - image_path
            - image_annotations
    """

    image_zip = Path(image_zip)
    label_source = Path(label_source)
    output_dir = Path(output_dir)

    test_size = int(config["dataset"]["test_size"])

    if not image_zip.is_file():
        raise FileNotFoundError(
            f"Image ZIP not found: {image_zip}"
        )

    if not label_source.exists():
        raise FileNotFoundError(
            f"Labels not found: {label_source}"
        )

    local_images = output_dir / "images" / split
    local_labels = output_dir / "labels" / split

    local_images.mkdir(parents=True, exist_ok=True)
    local_labels.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(image_zip) as images_archive:

        # Find real JPG images and ignore macOS metadata.
        jpg_members = _entries_for_split(
            (
                member
                for member in images_archive.namelist()
                if member.lower().endswith((".jpg", ".jpeg"))
                and not member.endswith("/")
                and _is_real_archive_member(member)
            ),
            split,
        )

        image_index = _unique_index(
            jpg_members,
            lambda m: PurePosixPath(m).name,
            image_zip.name,
        )

        label_archive = None

        try:
            # Case 1: Labels stored in ZIP.
            if (
                label_source.is_file()
                and label_source.suffix.lower() == ".zip"
            ):
                label_archive = zipfile.ZipFile(label_source)

                json_members = _entries_for_split(
                    (
                        member
                        for member in label_archive.namelist()
                        if member.lower().endswith(".json")
                        and not member.endswith("/")
                        and _is_real_archive_member(member)
                    ),
                    split,
                )

                label_index = _unique_index(
                    json_members,
                    _label_stem,
                    label_source.name,
                )

            # Case 2: Labels already extracted.
            elif label_source.is_dir():

                json_paths = sorted(
                    path
                    for path in label_source.rglob("*.json")
                    if _is_real_archive_member(
                        path.relative_to(label_source).as_posix()
                    )
                )

                json_paths = _entries_for_split(
                    (
                        path.relative_to(label_source).as_posix()
                        for path in json_paths
                    ),
                    split,
                )

                label_index = _unique_index(
                    json_paths,
                    _label_stem,
                    str(label_source),
                )

                label_index = {
                    key: label_source / value
                    for key, value in label_index.items()
                }

            else:
                raise ValueError(
                    "label_source must be a ZIP or JSON directory"
                )

            # Keep only images with matching annotations.
            eligible_names = sorted(
                filename
                for filename in image_index
                if Path(filename).stem in label_index
            )

            if test_size > len(eligible_names):
                raise ValueError(
                    f"Requested {test_size} images, but only "
                    f"{len(eligible_names)} annotated {split} "
                    f"images were found"
                )

            # Reuse an existing test split if available.
            if manifest_path is not None:
                manifest_path = Path(manifest_path)

            if (
                manifest_path is not None
                and manifest_path.is_file()
            ):
                selected = [
                    name.strip()
                    for name in manifest_path.read_text().splitlines()
                    if name.strip()
                ]

                if (
                    len(selected) != test_size
                    or len(set(selected)) != test_size
                ):
                    raise ValueError(
                        "Existing split manifest has wrong count "
                        "or duplicate filenames"
                    )

                missing = set(selected) - set(eligible_names)

                if missing:
                    raise ValueError(
                        f"Manifest references missing images: "
                        f"{sorted(missing)[:5]}"
                    )

            else:
                # Reproducible selection without replacement.
                selected = sorted(
                    rng.choice(
                        eligible_names,
                        size=test_size,
                        replace=False,
                    ).tolist()
                )

            selected_paths = []

            # Extract only selected images and labels.
            for filename in sorted(selected):

                image_member = image_index[filename]
                target_img = local_images / filename

                if not target_img.is_file():
                    with (
                        images_archive.open(image_member) as src,
                        target_img.open("wb") as dst,
                    ):
                        shutil.copyfileobj(src, dst)

                label_member = label_index[Path(filename).stem]

                if label_archive is not None:
                    label_bytes = label_archive.read(label_member)
                else:
                    label_bytes = Path(label_member).read_bytes()

                # Validate JSON before saving it.
                try:
                    json.loads(label_bytes)

                except (
                    json.JSONDecodeError,
                    UnicodeDecodeError,
                ) as exc:
                    raise ValueError(
                        f"Invalid JSON annotation for image {filename!r}; "
                        f"label member {str(label_member)!r} "
                        f"in {str(label_source)!r}; "
                        f"size={len(label_bytes)} bytes; "
                        f"first 100 bytes={label_bytes[:100]!r}"
                    ) from exc

                label_path = (
                    local_labels / f"{Path(filename).stem}.json"
                )

                label_path.write_bytes(label_bytes)
                selected_paths.append(target_img)

        finally:
            if label_archive is not None:
                label_archive.close()

    # Save the manifest after successful extraction/validation.
    if (
        manifest_path is not None
        and not manifest_path.exists()
    ):
        manifest_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        manifest_path.write_text(
            "\n".join(sorted(selected)) + "\n"
        )

    return get_labels_for_images(
        local_labels,
        selected_paths,
    )
