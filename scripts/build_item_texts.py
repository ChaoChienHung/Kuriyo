"""Build an opt-in item-text file for a content encoder without changing the catalog.

Writes data/amazon_reviews/2014/catalog/<category>/texts/<recipe>.jsonl ({"asin",
"content_text"}, catalog order) and <recipe>.manifest.json (catalog and metadata
sha256, recipe description, coverage). An existing complete output with the same
inputs is reused; a different or partial one is refused.

    python3 -m scripts.build_item_texts --category Beauty --recipe description
"""

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

from utils.amazon_reviews_2014 import (
    TEXT_RECIPES, file_sha256, item_text_with_description, read_python_literal_gzip, write_jsonl_line,
)

CATEGORIES = ("Beauty", "Sports_and_Outdoors", "Toys_and_Games")
BUILDERS = {"description": item_text_with_description}


def output_paths(year_root, category, recipe):
    folder = year_root / "catalog" / category / "texts"
    return folder / "{}.jsonl".format(recipe), folder / "{}.manifest.json".format(recipe)


def build(year_root, category, recipe):
    catalog = year_root / "catalog" / category / "items.jsonl"
    metadata = year_root / "raw" / category / "meta_{}.json.gz".format(category)
    if not catalog.is_file() or not metadata.is_file():
        raise ValueError("{} catalog or raw metadata missing; prepare the category first".format(category))
    texts, manifest_path = output_paths(year_root, category, recipe)
    expected = {"recipe": recipe, "fields": TEXT_RECIPES[recipe],
                "catalog_sha256": file_sha256(catalog), "metadata_sha256": file_sha256(metadata)}
    if texts.exists() or manifest_path.exists():
        if not (texts.is_file() and manifest_path.is_file()):
            raise ValueError("partial item-text output: {}; inspect and remove it".format(texts.parent))
        recorded = json.loads(manifest_path.read_text())
        if any(recorded.get(key) != value for key, value in expected.items()) \
                or recorded.get("sha256") != file_sha256(texts):
            raise ValueError("{} was built from different inputs; refusing to overwrite".format(texts))
        return recorded, False
    asins = [json.loads(line)["asin"] for line in catalog.open(encoding="utf-8")]
    wanted = set(asins)
    built = {}
    for record in read_python_literal_gzip(metadata):
        asin = record.get("asin")
        if asin in wanted and asin not in built:
            built[asin] = BUILDERS[recipe](record)
    missing = wanted - set(built)
    if missing:
        raise ValueError("{} catalog items have no metadata".format(len(missing)))
    texts.parent.mkdir(parents=True, exist_ok=True)
    temporary = texts.with_suffix(".jsonl.part")
    with temporary.open("w", encoding="utf-8") as output:
        for asin in asins:
            write_jsonl_line(output, {"asin": asin, "content_text": built[asin]})
    os.replace(temporary, texts)
    with_description = sum("Description: " in built[asin] for asin in asins)
    manifest = {**expected, "item_count": len(asins), "items_with_description": with_description,
                "sha256": file_sha256(texts)}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest, True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", choices=CATEGORIES, default="Beauty")
    parser.add_argument("--recipe", choices=sorted(TEXT_RECIPES), required=True)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/amazon_reviews")
    args = parser.parse_args(argv)
    try:
        manifest, created = build(args.data_root / "2014", args.category, args.recipe)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({"created": created, **manifest}))


if __name__ == "__main__":
    main()
