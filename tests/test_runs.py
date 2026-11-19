"""Contracts for the shared run directory layout in core/runs.py."""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
class RunLayoutTests(unittest.TestCase):
    def settings(self, scale=True):
        return {"run": {"git_revision": "new", "seed": 1}, "data": {"sha": "a"},
                "training": {"scale": scale}}

    KEYS = {"data_sha": "data.sha", "seed": "run.seed", "scale": ("training.scale", False)}

    def test_fresh_run_refuses_existing_directory_and_resume_checks_settings(self):
        from core.runs import RunError, RunLayout, open_run, save_checkpoint
        with TemporaryDirectory() as directory:
            layout = RunLayout(Path(directory) / "run", "amazon2014-beauty")
            open_run(layout, self.settings(), self.KEYS, resume=False)
            self.assertTrue(layout.settings.is_file())
            with self.assertRaises(RunError):
                open_run(layout, self.settings(), self.KEYS, resume=False)
            with self.assertRaisesRegex(RunError, "latest.pt"):
                open_run(layout, self.settings(), self.KEYS, resume=True)
            save_checkpoint(layout.latest, {"step": 1})
            with self.assertRaisesRegex(RunError, "training.scale"):
                open_run(layout, self.settings(scale=False), self.KEYS, resume=True)
            open_run(layout, self.settings(), self.KEYS, resume=True)

    def test_legacy_flat_run_is_checked_then_moved(self):
        from core.runs import RunError, RunLayout, open_run
        with TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            root.mkdir()
            # Older runs did not record "scale"; its legacy default is False.
            (root / "config.json").write_text(json.dumps({"data_sha": "a", "seed": 1, "git_revision": "old"}))
            for name in ("latest.pt", "best.pt", "train.jsonl", "best-metrics.json",
                         "validation-step-20000.json"):
                (root / name).write_text("x")
            layout = RunLayout(root, "amazon2014-beauty")
            with self.assertRaisesRegex(RunError, "scale"):
                open_run(layout, self.settings(scale=True), self.KEYS, resume=True)
            self.assertTrue((root / "latest.pt").is_file(), "a refused migration must not move files")
            open_run(layout, self.settings(scale=False), self.KEYS, resume=True)
            for path in (layout.latest, layout.best, layout.train_log, layout.best_metrics,
                         layout.validation / "step-020000.json", root / "config.json"):
                self.assertTrue(path.is_file(), path)
            settings = json.loads(layout.settings.read_text())
            self.assertEqual(settings["run"]["git_revision"], "old")
            events = [json.loads(line)["event"] for line in layout.events.read_text().splitlines()]
            self.assertIn("legacy_layout_migrated", events)

    def test_selection_keeps_best_and_stops_after_patience(self):
        import torch
        from core.run_logging import RunLogger
        from core.runs import RunLayout, Selection
        with TemporaryDirectory() as directory:
            layout = RunLayout(directory, "set")
            selection = Selection(patience=2)
            model = torch.nn.Linear(1, 1)
            with RunLogger(directory) as logger:
                for value, ndcg in enumerate((0.1, 0.3, 0.2, 0.25), start=1):
                    selection.record(layout, logger, model, {"ndcg@10": ndcg}, "epoch", value)
            self.assertEqual(selection.best, 0.3)
            self.assertTrue(selection.stopped)
            self.assertEqual(json.loads(layout.best_metrics.read_text())["ndcg@10"], 0.3)
            self.assertEqual(len(list(layout.validation.glob("epoch-*.json"))), 4)


if __name__ == "__main__":
    unittest.main()
