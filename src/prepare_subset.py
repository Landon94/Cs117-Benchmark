"""Build the BDD100K subset from the Kaggle zip (solesensei/solesensei_bdd100k).

Run once, locally:
    python -m src.prepare_subset ~/Downloads/solesensei_bdd100k.zip

Splits (sizes and seed from configs/experiment.yaml):
    train, val -> disjoint samples of the official train split (development)
    test       -> sample of the official val split (held-out)

Writes splits/{train,val,test}.txt (tracked in git) and
data/bdd100k_subset/{images,labels}/<split>/ plus data/bdd100k_subset.zip.
Labels become one <stem>.json per image, as src/dataset.py expects.
"""

import json
import sys
import zipfile
from pathlib import Path, PurePosixPath

import ijson
import numpy as np

from src.dataset import load_config

LABELS_MEMBER = {
    "train": "bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json",
    "val": "bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json",
}
IMAGE_PREFIX = {
    "train": "bdd100k/bdd100k/images/100k/train/",
    "val": "bdd100k/bdd100k/images/100k/val/",
}


def index_images(zf, official_split):
    prefix = IMAGE_PREFIX[official_split]
    index = {}
    for name in zf.namelist():
        if not name.startswith(prefix) or not name.lower().endswith(".jpg"):
            continue
        base = PurePosixPath(name).name
        if base.startswith("."):
            continue
        if base in index:
            raise ValueError(f"Duplicate image name {base} in {official_split}")
        index[base] = name
    return index


def label_names(zf, official_split):
    with zf.open(LABELS_MEMBER[official_split]) as f:
        return [item for item in ijson.items(f, "item.name")]


def labels_for(zf, official_split, wanted):
    out = {}
    with zf.open(LABELS_MEMBER[official_split]) as f:
        for item in ijson.items(f, "item", use_float=True):
            if item["name"] in wanted:
                out[item["name"]] = item
                if len(out) == len(wanted):
                    break
    return out


def eligible(zf, official_split):
    images = index_images(zf, official_split)
    labeled = set(label_names(zf, official_split))
    names = sorted(n for n in images if n in labeled)
    print(f"{official_split}: {len(images)} images, {len(labeled)} labels, "
          f"{len(names)} usable")
    return images, names


def main(zip_path):
    config = load_config()["dataset"]
    rng = np.random.default_rng(config["seed"])
    n_train, n_val, n_test = (int(config[k]) for k in
                              ("train_size", "val_size", "test_size"))
    repo_splits = Path("splits")
    out_root = Path("data/bdd100k_subset")
    repo_splits.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        tr_images, tr_names = eligible(zf, "train")
        va_images, va_names = eligible(zf, "val")

        pick = rng.choice(tr_names, size=n_train + n_val, replace=False).tolist()
        chosen = {
            "train": (sorted(pick[:n_train]), "train", tr_images),
            "val": (sorted(pick[n_train:]), "train", tr_images),
            "test": (sorted(rng.choice(va_names, size=n_test,
                                       replace=False).tolist()),
                     "val", va_images),
        }

        for split, (names, official, images) in chosen.items():
            (repo_splits / f"{split}.txt").write_text("\n".join(names) + "\n")
            img_dir = out_root / "images" / split
            lbl_dir = out_root / "labels" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            for name in names:
                with zf.open(images[name]) as src:
                    (img_dir / name).write_bytes(src.read())

        # One streaming pass per official split for labels.
        for official in ("train", "val"):
            wanted = {}
            for split, (names, off, _) in chosen.items():
                if off == official:
                    wanted.update({n: split for n in names})
            found = labels_for(zf, official, set(wanted))
            if len(found) != len(wanted):
                raise RuntimeError(f"Missing labels in {official}")
            for name, item in found.items():
                path = out_root / "labels" / wanted[name] / f"{Path(name).stem}.json"
                path.write_text(json.dumps(item))

    zip_out = Path("data/bdd100k_subset.zip")
    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_STORED) as z:
        for p in sorted(out_root.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(out_root).as_posix())
    print("Wrote", zip_out, f"({zip_out.stat().st_size / 2**20:.0f} MB)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
