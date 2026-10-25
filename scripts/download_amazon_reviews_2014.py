"""Download a 2014 Amazon Reviews 5-core category and its product metadata.

Example:
    python3 scripts/download_amazon_reviews_2014.py --category Beauty
"""

import argparse
import ast
import datetime as dt
import gzip
import hashlib
import json
import os
from pathlib import Path
import urllib.request


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PAGE = "https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html"
BASE_URL = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
CATEGORIES = {
    "Beauty": "Beauty",
    "Sports_and_Outdoors": "Sports_and_Outdoors",
    "Toys_and_Games": "Toys_and_Games",
}


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

        # These older files contain Python dictionary literals rather than strict JSON.
        # literal_eval parses data without executing expressions from the source file.
        with gzip.open(temporary, "rt", encoding="utf-8") as source:
            for row_count, line in enumerate(source, start=1):
                record = ast.literal_eval(line)
                if not isinstance(record, dict) or not required_fields.issubset(record):
                    raise ValueError("{} has invalid row {}".format(destination.name, row_count))
        if row_count == 0:
            raise ValueError("{} is empty".format(destination.name))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"url": url, "bytes": byte_count, "rows": row_count, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", choices=sorted(CATEGORIES), default="Beauty")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "data/amazon_reviews/2014/raw")
    parser.add_argument("--force", action="store_true", help="replace an existing category download")
    args = parser.parse_args()

    category = CATEGORIES[args.category]
    directory = args.output_root / category
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        "reviews": ("reviews_{}_5.json.gz".format(category),
                    {"reviewerID", "asin", "overall", "unixReviewTime"}),
        "metadata": ("meta_{}.json.gz".format(category), {"asin"}),
    }
    manifest_path = directory / "manifest.json"
    if any((directory / filename).exists() for filename, _ in files.values()) and not args.force:
        parser.error("download already exists; use --force to replace it")

    manifest = {
        "dataset": "Amazon Reviews 2014",
        "source_page": SOURCE_PAGE,
        "category": category,
        "review_subset": "5-core",
        "downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "files": {},
    }
    for name, (filename, required_fields) in files.items():
        url = BASE_URL + "/" + filename
        destination = directory / filename
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
