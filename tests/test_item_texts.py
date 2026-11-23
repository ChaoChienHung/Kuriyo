"""Opt-in item-text files for content encoders (scripts/build_item_texts.py)."""

import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.build_item_texts import build
from utils.amazon_reviews_2014 import item_text, item_text_with_description


METADATA = [
    {"asin": "B1", "title": "Red Lipstick", "brand": "Acme", "price": 9.5,
     "categories": [["Beauty", "Makeup"]], "description": "  Long\n lasting   colour. "},
    {"asin": "B2", "title": "Mint Shampoo", "categories": [["Beauty", "Hair"]]},
    {"asin": "B3", "title": "Not in catalog"},
]


def write_inputs(year, category="Beauty"):
    raw = year / "raw" / category
    raw.mkdir(parents=True)
    with gzip.open(raw / "meta_{}.json.gz".format(category), "wt", encoding="utf-8") as output:
        for record in METADATA:
            output.write(repr(record) + "\n")
    catalog = year / "catalog" / category
    catalog.mkdir(parents=True)
    (catalog / "items.jsonl").write_text("".join(json.dumps({"asin": asin}) + "\n" for asin in ("B2", "B1")))


class ItemTextTests(unittest.TestCase):
    def test_description_is_appended_after_catalog_fields(self):
        self.assertEqual(item_text_with_description(METADATA[0]),
                         item_text(METADATA[0]) + " | Description: Long lasting colour.")
        self.assertEqual(item_text_with_description(METADATA[1]), item_text(METADATA[1]))

    def test_build_writes_catalog_order_and_reuses_or_refuses(self):
        with TemporaryDirectory() as temporary:
            year = Path(temporary) / "2014"
            write_inputs(year)
            catalog = (year / "catalog/Beauty/items.jsonl").read_bytes()
            manifest, created = build(year, "Beauty", "description")
            self.assertTrue(created)
            rows = [json.loads(line) for line in
                    (year / "catalog/Beauty/texts/description.jsonl").read_text().splitlines()]
            self.assertEqual([row["asin"] for row in rows], ["B2", "B1"])
            self.assertIn("Description: Long lasting colour.", rows[1]["content_text"])
            self.assertEqual((manifest["item_count"], manifest["items_with_description"]), (2, 1))
            self.assertEqual((year / "catalog/Beauty/items.jsonl").read_bytes(), catalog)
            self.assertFalse(build(year, "Beauty", "description")[1])
            (year / "catalog/Beauty/items.jsonl").write_text(json.dumps({"asin": "B1"}) + "\n")
            with self.assertRaises(ValueError):
                build(year, "Beauty", "description")

    def test_missing_metadata_is_refused(self):
        with TemporaryDirectory() as temporary:
            year = Path(temporary) / "2014"
            write_inputs(year)
            (year / "catalog/Beauty/items.jsonl").write_text(json.dumps({"asin": "B9"}) + "\n")
            with self.assertRaises(ValueError):
                build(year, "Beauty", "description")


if __name__ == "__main__":
    unittest.main()
