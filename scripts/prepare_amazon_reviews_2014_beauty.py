"""Idempotently download and prepare the 2014 Beauty fixed catalog and splits.

Run from the repository root with ``python -m scripts.prepare_amazon_reviews_2014_beauty``.
Existing complete inputs and outputs are verified and reused; partial outputs require
manual inspection instead of an implicit overwrite.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
YEAR = ROOT / "data/amazon_reviews/2014"
CATEGORY = "Beauty"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def run_module(module):
    subprocess.run([sys.executable, "-m", module, "--category", CATEGORY],
                   cwd=ROOT, check=True)


def prepare():
    raw = YEAR / "raw" / CATEGORY
    download_manifest = raw / "manifest.json"
    source_files = {"reviews": raw / "reviews_Beauty_5.json.gz",
                    "metadata": raw / "meta_Beauty.json.gz"}
    if not any(path.exists() for path in (*source_files.values(), download_manifest)):
        run_module("scripts.download_amazon_reviews_2014")
    if not download_manifest.is_file() or not all(path.is_file() for path in source_files.values()):
        raise RuntimeError("partial Beauty download; inspect raw/Beauty before retrying")
    source = json.loads(download_manifest.read_text())
    for name, path in source_files.items():
        if digest(path) != source["files"][name]["sha256"]:
            raise RuntimeError("download checksum mismatch: {}".format(path))

    catalog = YEAR / "catalog/Beauty/items.jsonl"
    split_manifest = YEAR / "train/raw/Beauty/split_manifest.json"
    splits = {split: YEAR / split / "raw/Beauty/examples.jsonl"
              for split in ("train", "validation", "test")}
    outputs = (catalog, split_manifest, *splits.values())
    if not any(path.exists() for path in outputs):
        run_module("scripts.prepare_amazon_reviews_2014")
    if not all(path.is_file() for path in outputs):
        raise RuntimeError("partial Beauty split; inspect outputs before retrying")
    manifest = json.loads(split_manifest.read_text())
    if manifest.get("category") != CATEGORY:
        raise RuntimeError("split manifest category mismatch")
    for name, path in source_files.items():
        if manifest["source"][name]["sha256"] != digest(path):
            raise RuntimeError("split manifest source differs from current download")
    if manifest.get("catalog", {}).get("sha256") != digest(catalog):
        raise RuntimeError("split manifest catalog checksum mismatch")
    for split, path in splits.items():
        if not path.stat().st_size:
            raise RuntimeError("empty {} split".format(split))
    print("Beauty ready: {} items, {} train, {} validation, {} test".format(
        manifest["counts"]["catalog_items"], manifest["counts"]["train_examples"],
        manifest["counts"]["validation_examples"], manifest["counts"]["test_examples"]),
        flush=True)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()


if __name__ == "__main__":
    main()
