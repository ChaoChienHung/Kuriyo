import unittest
import gzip
from pathlib import Path
from tempfile import TemporaryDirectory

from utils.amazon_reviews_2014 import build_examples, item_text, load_sequences


class AmazonReviews2014PreparationTests(unittest.TestCase):
    def test_leave_last_two_out_and_history_cap(self):
        events = [(i, i, item) for i, item in enumerate("ABCDE", start=1)]
        train, evaluation, test = build_examples(events, max_history=2)

        self.assertEqual([(row["history"], row["target_asin"]) for row in train],
                         [(["A"], "B"), (["A", "B"], "C")])
        self.assertEqual((evaluation["history"], evaluation["target_asin"]),
                         (["B", "C"], "D"))
        self.assertEqual((test["history"], test["target_asin"]),
                         (["C", "D"], "E"))
        self.assertEqual(test["target_timestamp"], 5)

    def test_short_user_cannot_be_split(self):
        with self.assertRaises(ValueError):
            build_examples([(1, 0, "A")] * 4, max_history=20)

    def test_equal_timestamps_follow_source_row_order(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.json.gz"
            rows = [
                {"reviewerID": "u", "asin": "later", "unixReviewTime": 2},
                {"reviewerID": "u", "asin": "first", "unixReviewTime": 1},
                {"reviewerID": "u", "asin": "second", "unixReviewTime": 1},
            ]
            with gzip.open(path, "wt", encoding="utf-8") as output:
                for row in rows:
                    output.write(repr(row) + "\n")
            users, _, _ = load_sequences(path)
            self.assertEqual([event[2] for event in users["u"]],
                             ["first", "second", "later"])

    def test_item_text_uses_available_content(self):
        text = item_text({"title": "Soap", "brand": "Acme", "price": 4.5,
                          "categories": [["Beauty", "Bath"]]})
        self.assertIn("Title: Soap", text)
        self.assertIn("Categories: Beauty > Bath", text)


if __name__ == "__main__":
    unittest.main()
