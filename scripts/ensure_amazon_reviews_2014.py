"""Idempotently download and prepare 2014 categories: fixed catalog and splits.

Run from the repository root, for example
``python -m scripts.ensure_amazon_reviews_2014 --category Sports_and_Outdoors`` or
``--category all`` for the three categories of the TIGER paper. Existing complete
inputs and outputs are verified and reused; partial outputs require manual inspection
instead of an implicit overwrite.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
YEAR = ROOT / "data/amazon_reviews/2014"
CATEGORIES = ("Beauty", "Sports_and_Outdoors", "Toys_and_Games")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def run_module(module, category):
    subprocess.run([sys.executable, "-m", module, "--category", category],
                   cwd=ROOT, check=True)


def prepare(category):
    raw = YEAR / "raw" / category
    download_manifest = raw / "manifest.json"
    source_files = {"reviews": raw / "reviews_{}_5.json.gz".format(category),
                    "metadata": raw / "meta_{}.json.gz".format(category)}
    if not any(path.exists() for path in (*source_files.values(), download_manifest)):
        run_module("scripts.download_amazon_reviews_2014", category)
    if not download_manifest.is_file() or not all(path.is_file() for path in source_files.values()):
        raise RuntimeError("partial {0} download; inspect raw/{0} before retrying".format(category))
    source = json.loads(download_manifest.read_text())
    for name, path in source_files.items():
        if digest(path) != source["files"][name]["sha256"]:
            raise RuntimeError("download checksum mismatch: {}".format(path))

    catalog = YEAR / "catalog" / category / "items.jsonl"
    split_manifest = YEAR / "train/raw" / category / "split_manifest.json"
    splits = {split: YEAR / split / "raw" / category / "examples.jsonl"
              for split in ("train", "validation", "test")}
    outputs = (catalog, split_manifest, *splits.values())
    if not any(path.exists() for path in outputs):
        run_module("scripts.prepare_amazon_reviews_2014", category)
    if not all(path.is_file() for path in outputs):
        raise RuntimeError("partial {} split; inspect outputs before retrying".format(category))
    manifest = json.loads(split_manifest.read_text())
    if manifest.get("category") != category:
        raise RuntimeError("split manifest category mismatch")
    for name, path in source_files.items():
        if manifest["source"][name]["sha256"] != digest(path):
            raise RuntimeError("split manifest source differs from current download")
    if manifest.get("catalog", {}).get("sha256") != digest(catalog):
        raise RuntimeError("split manifest catalog checksum mismatch")
    for split, path in splits.items():
        if not path.stat().st_size:
            raise RuntimeError("empty {} split".format(split))
    print("{} ready: {} items, {} train, {} validation, {} test".format(
        category, manifest["counts"]["catalog_items"], manifest["counts"]["train_examples"],
        manifest["counts"]["validation_examples"], manifest["counts"]["test_examples"]),
        flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", choices=CATEGORIES + ("all",), default="Beauty")
    args = parser.parse_args()
    for category in CATEGORIES if args.category == "all" else (args.category,):
        prepare(category)


if __name__ == "__main__":
    main()
