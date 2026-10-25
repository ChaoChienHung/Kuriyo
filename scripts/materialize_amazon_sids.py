"""Convert prepared Amazon next-item examples from ASINs to Semantic IDs.

Example (from the repository root):
    python3 -m scripts.materialize_amazon_sids --year 2014 --category Beauty \
        --method sentence-t5_rqvae \
        --sid-file data/amazon_reviews/2014/train/sids/sentence-t5_rqvae/Beauty/semantic_ids.jsonl
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from utils.amazon_reviews_2014 import write_jsonl_line


DATA_ROOT = ROOT / "data/amazon_reviews"
SPLITS = ("train", "validation", "test")
METHODS = ("sentence-t5_rqvae", "sentence-t5_kmeans", "sentence-t5_lsh")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sids(path):
    sids = {}
    seen = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            record = json.loads(line)
            asin, sid = record["asin"], record["sid"]
            if (not isinstance(asin, str) or not isinstance(sid, list)
                    or not sid or any(not isinstance(token, int) or token < 0 for token in sid)):
                raise ValueError("invalid SID row {}".format(line_number))
            if asin in sids or tuple(sid) in seen:
                raise ValueError("duplicate ASIN or SID at row {}".format(line_number))
            sids[asin] = sid
            seen.add(tuple(sid))
    if not sids:
        raise ValueError("empty SID catalog")
    return sids


def convert_example(row, sids):
    try:
        history_sids = [sids[asin] for asin in row["history"]]
        target_sid = sids[row["target_asin"]]
    except KeyError as error:
        raise ValueError("example references an item without SID: {}".format(error)) from error
    if not history_sids:
        raise ValueError("example has empty history")
    return {"user_id": row["user_id"], "history_sids": history_sids,
            "target_sid": target_sid, "target_asin": row["target_asin"],
            "target_timestamp": row["target_timestamp"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", choices=("2014",), default="2014")
    parser.add_argument("--category", default="Beauty")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--sid-file", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--force", action="store_true", help="replace existing SID split outputs")
    args = parser.parse_args()
    if not args.sid_file.is_file():
        parser.error("SID file missing: {}".format(args.sid_file))
    year_dir = args.data_root / args.year
    source_paths = {split: year_dir / split / "raw" / args.category / "examples.jsonl"
                    for split in SPLITS}
    output_dirs = {split: year_dir / split / "sids" / args.method / args.category
                   for split in SPLITS}
    output_paths = {split: directory / "examples.jsonl"
                    for split, directory in output_dirs.items()}
    catalog_path = output_dirs["train"] / "semantic_ids.jsonl"
    manifest_path = output_dirs["train"] / "split_manifest.json"
    for split, path in source_paths.items():
        if not path.is_file():
            parser.error("{} split missing: {}".format(split, path))
    if not args.force and any(path.exists() for path in
                              (*output_paths.values(), manifest_path)):
        parser.error("SID split output already exists; use --force to replace it")

    sid_digest = sha256(args.sid_file)
    if catalog_path.exists() and sha256(catalog_path) != sid_digest:
        parser.error("existing SID catalog differs from --sid-file: {}".format(catalog_path))
    producer_manifest_path = args.sid_file.parent / "manifest.json"
    if producer_manifest_path.is_file():
        with producer_manifest_path.open(encoding="utf-8") as source:
            producer_manifest = json.load(source)
        if producer_manifest.get("method") and producer_manifest["method"] != args.method:
            parser.error("SID producer method differs from --method")

    sids = load_sids(args.sid_file)
    item_path = year_dir / "train/raw" / args.category / "items.jsonl"
    with item_path.open("r", encoding="utf-8") as source:
        item_asins = {json.loads(line)["asin"] for line in source}
    if item_asins != sids.keys():
        raise ValueError("SID catalog does not cover exactly this dataset's items")

    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    temporary_paths = [path.with_name(path.name + ".part") for path in
                       (*output_paths.values(), catalog_path, manifest_path)]
    try:
        counts = {}
        for split in SPLITS:
            count = 0
            temporary = output_paths[split].with_name("examples.jsonl.part")
            with source_paths[split].open("r", encoding="utf-8") as source, \
                    temporary.open("w", encoding="utf-8") as output:
                for line in source:
                    write_jsonl_line(output, convert_example(json.loads(line), sids))
                    count += 1
            counts[split] = count
        if not catalog_path.exists():
            shutil.copyfile(args.sid_file, catalog_path.with_name("semantic_ids.jsonl.part"))
        manifest = {"dataset": "Amazon Reviews {}".format(args.year),
                    "category": args.category, "method": args.method,
                    "sid_source": str(args.sid_file.resolve()),
                    "sid_sha256": sid_digest,
                    "items_sha256": sha256(item_path),
                    "source_examples_sha256": {split: sha256(path) for split, path in source_paths.items()},
                    "counts": counts,
                    "schema": ["user_id", "history_sids", "target_sid",
                               "target_asin", "target_timestamp"]}
        with manifest_path.with_name("split_manifest.json.part").open(
                "w", encoding="utf-8") as output:
            json.dump(manifest, output, ensure_ascii=False, indent=2)
            output.write("\n")
        for split in SPLITS:
            os.replace(output_paths[split].with_name("examples.jsonl.part"), output_paths[split])
        if not catalog_path.exists():
            os.replace(catalog_path.with_name("semantic_ids.jsonl.part"), catalog_path)
        os.replace(manifest_path.with_name("split_manifest.json.part"), manifest_path)
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
    print(json.dumps({"method": args.method, "category": args.category,
                      "counts": counts, "items": len(sids)}), flush=True)


if __name__ == "__main__":
    main()
