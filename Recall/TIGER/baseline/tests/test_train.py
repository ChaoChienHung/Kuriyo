"""Small integration contract for the paper-configured TIGER entry point."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[4]


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
                     "T5 modeling dependencies are not installed")
class PaperBaselineTests(unittest.TestCase):
    def test_one_step_resume_and_held_out_test(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            data_root = base / "amazon_reviews"
            year = data_root / "2014"
            catalog = year / "catalog/Beauty/items.jsonl"
            catalog.parent.mkdir(parents=True)
            catalog.write_text("".join(
                json.dumps({"asin": "I{:02}".format(index)}) + "\n"
                for index in range(12)))
            sid_run = year / "artifacts/sids/sentence-t5_rqvae/Beauty/vtest"
            sid_run.mkdir(parents=True)
            (sid_run / "semantic_ids.jsonl").write_text("".join(
                json.dumps({"asin": "I{:02}".format(index),
                            "sid": [index // 4, index % 4, 0, 0]}) + "\n"
                for index in range(12)))
            (sid_run / "manifest.json").write_text(json.dumps({
                "method": "sentence-t5_rqvae", "item_count": 12,
                "codebook_size": 4, "limit_items": None,
                "source_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest()}))
            row = {"user_id": "U", "history": ["I00"],
                   "target_asin": "I01", "target_timestamp": 1}
            for split in ("train", "validation", "test"):
                path = year / split / "raw/Beauty/examples.jsonl"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(row) + "\n")

            def run(*arguments):
                result = subprocess.run([sys.executable, "-m", *arguments], cwd=ROOT,
                                        capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stderr[-2000:])

            run("scripts.materialize_amazon_sids", "--data-root", str(data_root),
                "--method", "sentence-t5_rqvae",
                "--sid-file", str(sid_run / "semantic_ids.jsonl"))
            base_command = ("Recall.TIGER.baseline.train",
                            "--data-root", str(data_root), "--sid-run", str(sid_run),
                            "--runs-dir", str(base / "runs"), "--run-name", "smoke",
                            "--batch-size", "1", "--eval-every", "1",
                            "--save-every", "1", "--eval-batch-size", "1",
                            "--max-validation-examples", "1", "--device", "cpu")
            run(*base_command, "--steps", "1")
            run(*base_command, "--steps", "2", "--resume")
            run(*base_command, "--eval-test-only")
            run_dir = base / "runs/smoke"
            self.assertEqual(json.loads((run_dir / "test-metrics.json").read_text())
                             ["test_examples"], 1)
            self.assertTrue((run_dir / "best.pt").is_file())
            history = [json.loads(line) for line in (run_dir / "train.jsonl").read_text().splitlines()]
            self.assertEqual([row["step"] for row in history], [1, 2])
            self.assertNotEqual(history[0]["session_id"], history[1]["session_id"])
            events = [json.loads(line)["event"] for line in (run_dir / "events.log").read_text().splitlines()]
            self.assertIn("resume", events)
            self.assertIn("test_completed", events)
