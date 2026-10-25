"""Create TIGER-style train/eval/test examples from Amazon Reviews 2014 5-core.

Run from the repository root:
    python3 -m scripts.prepare_amazon_reviews_2014 --category Beauty
"""

import argparse
from contextlib import ExitStack
import datetime as dt
import json
import os
from pathlib import Path

from utils.amazon_reviews_2014 import (
    build_examples, file_sha256, load_catalog, load_sequences, write_jsonl_line,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data/amazon_reviews"
CATEGORIES = ("Beauty", "Sports_and_Outdoors", "Toys_and_Games")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", choices=CATEGORIES, default="Beauty")
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--force", action="store_true", help="replace existing split outputs")
    args = parser.parse_args()
    if args.max_history < 1:
        parser.error("--max-history must be positive")

    source_dir = DATA_ROOT / "raw/2014" / args.category
    review_path = source_dir / "reviews_{}_5.json.gz".format(args.category)
    metadata_path = source_dir / "meta_{}.json.gz".format(args.category)
    if not review_path.is_file() or not metadata_path.is_file():
        parser.error("download both 2014 files first: {}".format(source_dir))

    output_dirs = {split: DATA_ROOT / split / "2014" / args.category
                   for split in ("train", "eval", "test")}
    output_paths = {split: directory / "examples.jsonl" for split, directory in output_dirs.items()}
    items_path = output_dirs["train"] / "items.jsonl"
    manifest_path = output_dirs["train"] / "split_manifest.json"
    if not args.force and any(path.exists() for path in (*output_paths.values(), items_path, manifest_path)):
        parser.error("split output already exists; use --force to replace it")
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    print("Reading reviews from {}".format(review_path), flush=True)
    users, item_ids, review_count = load_sequences(review_path)
    eligible_users = {user_id: events for user_id, events in users.items() if len(events) >= 5}
    if not eligible_users:
        parser.error("no users have at least five interactions")
    eligible_items = {event[2] for events in eligible_users.values() for event in events}
    catalog = load_catalog(metadata_path, eligible_items)

    temporary_paths = [path.with_name(path.name + ".part")
                       for path in (*output_paths.values(), items_path, manifest_path)]
    try:
        counts = {split: 0 for split in output_paths}
        with ExitStack() as stack:
            outputs = {split: stack.enter_context(path.with_name(path.name + ".part").open("w", encoding="utf-8"))
                       for split, path in output_paths.items()}
            items_output = stack.enter_context(items_path.with_name(items_path.name + ".part").open("w", encoding="utf-8"))
            for user_id in sorted(eligible_users):
                train, evaluation, test = build_examples(eligible_users[user_id], args.max_history)
                for record in train:
                    write_jsonl_line(outputs["train"], {"user_id": user_id, **record})
                    counts["train"] += 1
                for split, record in (("eval", evaluation), ("test", test)):
                    write_jsonl_line(outputs[split], {"user_id": user_id, **record})
                    counts[split] += 1
            for item_id in sorted(catalog):
                write_jsonl_line(items_output, catalog[item_id])

        manifest = {
            "dataset": "Amazon Reviews 2014 5-core",
            "category": args.category,
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source": {
                "reviews": {"path": str(review_path.relative_to(REPO_ROOT)), "sha256": file_sha256(review_path)},
                "metadata": {"path": str(metadata_path.relative_to(REPO_ROOT)), "sha256": file_sha256(metadata_path)},
            },
            "rules": {
                "minimum_user_interactions": 5,
                "ordering": "unixReviewTime ascending; source row order breaks ties",
                "train": "all except each user's last two events; one next-item example per prefix of length >=1",
                "eval": "penultimate event, with preceding events as history",
                "test": "last event, with preceding events including the eval event as history",
                "max_history": args.max_history,
                "duplicate_user_item_events": "preserved",
            },
            "counts": {
                "source_reviews": review_count,
                "source_users": len(users),
                "eligible_users": len(eligible_users),
                "catalog_items": len(catalog),
                "train_examples": counts["train"],
                "eval_examples": counts["eval"],
                "test_examples": counts["test"],
            },
        }
        for split, path in output_paths.items():
            os.replace(path.with_name(path.name + ".part"), path)
        os.replace(items_path.with_name(items_path.name + ".part"), items_path)
        with manifest_path.with_name(manifest_path.name + ".part").open("w", encoding="utf-8") as output:
            json.dump(manifest, output, ensure_ascii=False, indent=2)
            output.write("\n")
        os.replace(manifest_path.with_name(manifest_path.name + ".part"), manifest_path)
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)

    print(json.dumps(manifest["counts"], indent=2), flush=True)
    print("Wrote train/eval/test under {}".format(DATA_ROOT), flush=True)


if __name__ == "__main__":
    main()
