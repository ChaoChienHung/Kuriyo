"""Download one category of the McAuley Lab Amazon Reviews 2023 dataset.

Run from the repository root, for example:
    python3 scripts/download_amazon_reviews.py --category All_Beauty
"""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.request


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = "McAuley-Lab/Amazon-Reviews-2023"
REVISION = "2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e"
BASE_URL = "https://huggingface.co/datasets/{}/resolve/{}".format(DATASET, REVISION)


def download_and_validate(url, destination, required_fields):
    temporary = destination.with_name(destination.name + ".part")
    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    byte_count = 0
    row_count = 0
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Kuriyo-data-preparation/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                byte_count += len(chunk)

        with temporary.open("r", encoding="utf-8") as source:
            for row_count, line in enumerate(source, start=1):
                record = json.loads(line)
                if not required_fields.issubset(record):
                    missing = required_fields - record.keys()
                    raise ValueError("{} row {} lacks {}".format(destination.name, row_count, sorted(missing)))
        if row_count == 0:
            raise ValueError("{} is empty".format(destination.name))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"url": url, "bytes": byte_count, "rows": row_count, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="All_Beauty", help="Amazon Reviews 2023 category name")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "data/raw/amazon_reviews_2023")
    parser.add_argument("--force", action="store_true", help="replace an existing category download")
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", args.category):
        parser.error("category must contain only letters, numbers, and underscores")

    directory = args.output_root / args.category
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        "reviews": ("raw/review_categories/{}.jsonl".format(args.category),
                    {"user_id", "parent_asin", "rating", "timestamp"}),
        "metadata": ("raw/meta_categories/meta_{}.jsonl".format(args.category),
                     {"parent_asin", "title"}),
    }
    manifest_path = directory / "manifest.json"
    if any((directory / Path(path).name).exists() for path, _ in files.values()) and not args.force:
        parser.error("download already exists; use --force to replace it")

    manifest = {
        "dataset": DATASET,
        "revision": REVISION,
        "category": args.category,
        "downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "files": {},
    }
    for name, (path, required_fields) in files.items():
        destination = directory / Path(path).name
        url = BASE_URL + "/" + path
        print("Downloading {}: {}".format(name, url), flush=True)
        manifest["files"][name] = download_and_validate(url, destination, required_fields)
        print("Validated {} rows in {}".format(manifest["files"][name]["rows"], destination), flush=True)

    temporary_manifest = manifest_path.with_suffix(".json.part")
    with temporary_manifest.open("w", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write("\n")
    os.replace(temporary_manifest, manifest_path)
    print("Wrote {}".format(manifest_path), flush=True)


if __name__ == "__main__":
    main()
