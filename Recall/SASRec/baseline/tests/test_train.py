"""Integration contracts for the SASRec baseline entry point."""

import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[4]
TORCH = importlib.util.find_spec("torch") is not None
if TORCH:
    import torch

    from Recall.SASRec.baseline.train import evaluate


def write_split(data_root, items=12):
    year = data_root / "2014"
    catalog = year / "catalog/Beauty/items.jsonl"
    catalog.parent.mkdir(parents=True)
    catalog.write_text("".join(json.dumps({"asin": "I{:02}".format(i)}) + "\n" for i in range(items)))
    rows = {"train": [{"user_id": "U{}".format(u), "history": ["I{:02}".format((u + j) % items) for j in range(k)],
                       "target_asin": "I{:02}".format((u + k) % items), "target_timestamp": 1}
                      for u in range(4) for k in (1, 2, 3)],
            "validation": [{"user_id": "U{}".format(u), "history": ["I{:02}".format((u + j) % items) for j in range(4)],
                            "target_asin": "I{:02}".format((u + 4) % items), "target_timestamp": 2}
                           for u in range(4)],
            "test": [{"user_id": "U{}".format(u), "history": ["I{:02}".format((u + j) % items) for j in range(5)],
                      "target_asin": "I{:02}".format((u + 5) % items), "target_timestamp": 3}
                     for u in range(4)]}
    for split, examples in rows.items():
        path = year / split / "raw/Beauty/examples.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in examples))
    (year / "train/raw/Beauty/split_manifest.json").write_text(json.dumps({"catalog": {}}))


class FixedRanker(torch.nn.Module if TORCH else object):
    """Stand-in model that always ranks items 3, 1, 2, 4 (indices are 1-based)."""
    num_items = 4

    def forward(self, sequence):
        return torch.zeros(sequence.shape[0], sequence.shape[1], 1)

    def scores(self, hidden):
        return torch.tensor([[3.0, 2.0, 4.0, 1.0]]).expand(hidden.shape[0], -1)


@unittest.skipUnless(TORCH, "torch is not installed")
class SASRecTrainTests(unittest.TestCase):
    def test_evaluate_wires_full_ranking_into_shared_metrics(self):
        asins = ["A", "B", "C", "D"]
        rows = [{"history": [1], "target": 3}, {"history": [2], "target": 2},
                {"history": [1], "target": 4}]
        metrics = evaluate(FixedRanker(), rows, asins, 20, "cpu", 2, ks=(1, 3))
        self.assertAlmostEqual(metrics["recall@1"], 1 / 3)
        self.assertAlmostEqual(metrics["recall@3"], 2 / 3)
        self.assertAlmostEqual(metrics["ndcg@3"], (1 + 1 / math.log2(4)) / 3)
        self.assertEqual(metrics["examples"], 3)

    def test_train_resume_and_held_out_test(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            write_split(base / "amazon_reviews")

            def run(*arguments, name="smoke"):
                result = subprocess.run([sys.executable, "-m", "Recall.SASRec.baseline.train",
                                         "--data-root", str(base / "amazon_reviews"),
                                         "--runs-dir", str(base / "runs"), "--run-name", name,
                                         "--batch-size", "2", "--eval-batch-size", "3",
                                         "--eval-every", "1", "--hidden", "16", "--heads", "2",
                                         "--max-history", "4", "--device", "cpu", *arguments],
                                        cwd=ROOT, capture_output=True, text=True, timeout=300)
                return result

            first = run("--epochs", "1")
            self.assertEqual(first.returncode, 0, first.stderr[-2000:])
            self.assertNotEqual(run("--epochs", "1").returncode, 0, "must refuse to overwrite a run")
            mismatch = run("--epochs", "2", "--resume", "--hidden", "32")
            self.assertNotEqual(mismatch.returncode, 0)
            resumed = run("--epochs", "2", "--resume")
            self.assertEqual(resumed.returncode, 0, resumed.stderr[-2000:])
            tested = run("--eval-test-only")
            self.assertEqual(tested.returncode, 0, tested.stderr[-2000:])
            self.assertNotEqual(run("--eval-test-only").returncode, 0, "test must run once")

            run_dir = base / "runs/smoke"
            history = [json.loads(line) for line in (run_dir / "train/train.jsonl").read_text().splitlines()]
            self.assertEqual([row["step"] for row in history], [1, 2, 3, 4])
            self.assertEqual([row["epoch"] for row in history], [1, 1, 2, 2])
            self.assertNotEqual(history[0]["session_id"], history[-1]["session_id"])
            straight = run("--epochs", "2", name="straight")
            self.assertEqual(straight.returncode, 0, straight.stderr[-2000:])
            uninterrupted = [json.loads(line)["loss"] for line in
                             (base / "runs/straight/train/train.jsonl").read_text().splitlines()]
            for resumed_loss, straight_loss in zip([row["loss"] for row in history], uninterrupted):
                self.assertAlmostEqual(resumed_loss, straight_loss, places=5,
                                       msg="resume must continue the same sample/RNG stream")
            self.assertTrue(all(math.isfinite(row["loss"]) for row in history))
            events = [json.loads(line)["event"] for line in (run_dir / "events.log").read_text().splitlines()]
            for name in ("training_started", "resume", "validation", "checkpoint_saved", "test_completed"):
                self.assertIn(name, events)
            self.assertTrue((run_dir / "validation/amazon2014-beauty/epoch-000002.json").is_file())
            self.assertTrue((run_dir / "checkpoints/best.pt").is_file())
            test_metrics = json.loads((run_dir / "test/amazon2014-beauty/metrics.json").read_text())
            self.assertEqual((test_metrics["split"], test_metrics["examples"]), ("test", 4))
            settings = json.loads((run_dir / "settings.json").read_text())
            self.assertEqual((settings["training"]["loss"], settings["model"]["config"]["max_history"]),
                             ("ce", 4))
            self.assertEqual(settings["run"]["seed"], 42)
            self.assertEqual(settings["data"]["max_history"], 4)
            self.assertIn("splits", settings["data"])


if __name__ == "__main__":
    unittest.main()
