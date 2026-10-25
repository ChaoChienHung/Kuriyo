"""Prepare Amazon Reviews 2014 5-core sequences for next-item recommendation."""

import ast
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path


def read_python_literal_gzip(path):
    """Read legacy McAuley data safely; its rows are Python dict literals."""
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            record = ast.literal_eval(line)
            if not isinstance(record, dict):
                raise ValueError("{} row {} is not a dict".format(path, line_number))
            yield record


def load_sequences(review_path):
    users = defaultdict(list)
    item_ids = set()
    review_count = 0
    for row_index, row in enumerate(read_python_literal_gzip(review_path)):
        try:
            user_id = row["reviewerID"]
            item_id = row["asin"]
            timestamp = int(row["unixReviewTime"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid review row {}".format(row_index + 1)) from error
        if not user_id or not item_id:
            raise ValueError("empty user or item in review row {}".format(row_index + 1))
        users[user_id].append((timestamp, row_index, item_id))
        item_ids.add(item_id)
        review_count += 1
    for events in users.values():
        # Source row order resolves equal timestamps reproducibly.
        events.sort(key=lambda event: (event[0], event[1]))
    return users, item_ids, review_count


def build_examples(events, max_history):
    """Return train prefixes, one validation example, and one test example."""
    if len(events) < 5:
        raise ValueError("a user needs at least five interactions")
    item_ids = [event[2] for event in events]

    def example(target_index):
        return {
            "history": item_ids[max(0, target_index - max_history):target_index],
            "target_asin": item_ids[target_index],
            "target_timestamp": events[target_index][0],
        }

    train = [example(i) for i in range(1, len(events) - 2)]
    return train, example(len(events) - 2), example(len(events) - 1)


def item_text(metadata):
    """Construct the item text used later by a content encoder."""
    fields = []
    for label, key in (("Title", "title"), ("Brand", "brand"), ("Price", "price")):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            fields.append("{}: {}".format(label, value))
    categories = metadata.get("categories") or []
    if isinstance(categories, list):
        paths = [" > ".join(str(part) for part in path) for path in categories if isinstance(path, list)]
        if paths:
            fields.append("Categories: {}".format("; ".join(paths)))
    return " | ".join(fields)


def load_catalog(metadata_path, item_ids):
    catalog = {}
    for metadata in read_python_literal_gzip(metadata_path):
        item_id = metadata.get("asin")
        if item_id in item_ids and item_id not in catalog:
            catalog[item_id] = {
                "asin": item_id,
                "title": metadata.get("title"),
                "brand": metadata.get("brand"),
                "price": metadata.get("price"),
                "categories": metadata.get("categories"),
                "content_text": item_text(metadata),
            }
    missing = item_ids - catalog.keys()
    if missing:
        raise ValueError("{} reviewed items have no metadata".format(len(missing)))
    return catalog


def write_jsonl_line(output, record):
    output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
