"""Verify SID materialization preserves split boundaries and item mapping."""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SIDMaterializationTests(unittest.TestCase):
    def test_materializes_each_split_and_refuses_overwrite(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            data_root = base / "amazon_reviews"
            sid_dir = base / "sid_run"
            sid_dir.mkdir()
            sid_file = sid_dir / "semantic_ids.jsonl"
            sids = [{"asin": "A", "sid": [1, 2, 3, 0]},
                    {"asin": "B", "sid": [1, 2, 3, 1]}]
            sid_file.write_text("".join(json.dumps(row) + "\n" for row in sids))
            (sid_dir / "manifest.json").write_text(json.dumps({"method": "sentence-t5_rqvae"}))
            item_dir = data_root / "2014/train/raw/Beauty"
            item_dir.mkdir(parents=True)
            (item_dir / "items.jsonl").write_text(
                "".join(json.dumps({"asin": asin}) + "\n" for asin in ("A", "B")))
            example = {"user_id": "U", "history": ["A"], "target_asin": "B",
                       "target_timestamp": 123}
            for split in ("train", "validation", "test"):
                directory = data_root / "2014" / split / "raw/Beauty"
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "examples.jsonl").write_text(json.dumps(example) + "\n")
            command = [sys.executable, "-m", "scripts.materialize_amazon_sids",
                       "--year", "2014", "--category", "Beauty",
                       "--method", "sentence-t5_rqvae", "--sid-file", str(sid_file),
                       "--data-root", str(data_root)]
            first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            for split in ("train", "validation", "test"):
                path = (data_root / "2014" / split / "sids/sentence-t5_rqvae/Beauty/examples.jsonl")
                row = json.loads(path.read_text())
                self.assertEqual(row["history_sids"], [[1, 2, 3, 0]])
                self.assertEqual(row["target_sid"], [1, 2, 3, 1])
            second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("--force", second.stderr)


if __name__ == "__main__":
    unittest.main()
