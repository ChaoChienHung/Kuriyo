"""Exercise batch-stage orchestration without Slurm, downloads, or a GPU."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run.sbatch"


class RunSbatchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in ("AGENTS.md", "README.md"):
            (self.root / name).touch()
        self.trace = self.root / "commands.log"
        self.env = dict(os.environ, SLURM_SUBMIT_DIR=str(self.root), STAGE="sid",
                        SID_VERSION="test-v1", RUN_NAME="test-v1",
                        TMPDIR=str(self.root / "tmp"), HF_HOME=str(self.root / "hf"),
                        RUN_TRACE=str(self.trace), FAIL_PREPARE="0")
        self.env["PATH"] = str(self.root / "bin") + os.pathsep + os.environ["PATH"]
        self.executable("bin/srun", 'exec "$@"\n')
        self.executable("bin/git", 'echo test-revision\n')
        self.executable(".venv/bin/python", '''
echo "$2" >> "$RUN_TRACE"
case "$2" in
  pip)
    test "$3" = install
    test "$4" = --extra-index-url
    test "$5" = https://download.pytorch.org/whl/cu126
    test "$6" = torch==2.7.1+cu126
    test "$7" = -r
    test "$8" = Representations/SID/requirements.txt
    test "$9" = -r
    test "${10}" = Recall/TIGER/baseline/requirements.txt
    ;;
  scripts.prepare_amazon_reviews_2014_beauty)
    if [ "$FAIL_PREPARE" = 1 ]; then exit 17; fi
    mkdir -p data/amazon_reviews/2014/catalog/Beauty
    echo '{}' > data/amazon_reviews/2014/catalog/Beauty/items.jsonl
    ;;
  Representations.SID.scripts.generate_sids)
    test -s data/amazon_reviews/2014/catalog/Beauty/items.jsonl
    echo "$*" > "$RUN_TRACE.sid"
    ;;
  Recall.TIGER.baseline.train)
    echo "$*" > "$RUN_TRACE.train"
    ;;
  *) exit 99 ;;
esac
''')

    def executable(self, name, body):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nset -eu\n" + body)
        path.chmod(0o755)

    def run_job(self):
        return subprocess.run(["bash", str(SCRIPT)], cwd=self.root, env=self.env,
                              text=True, capture_output=True)

    def test_sid_prepares_inputs_before_generation(self):
        result = self.run_job()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.trace.read_text().splitlines(), [
            "pip", "scripts.prepare_amazon_reviews_2014_beauty",
            "Representations.SID.scripts.generate_sids",
        ])

    def write_sid(self, manifest):
        sid = self.root / (
            "data/amazon_reviews/2014/artifacts/sids/"
            "sentence-t5_rqvae/Beauty/test-v1")
        sid.mkdir(parents=True)
        (sid / "semantic_ids.jsonl").write_text("{}\n")
        (sid / "manifest.json").write_text(json.dumps(manifest) + "\n")

    def test_sid_generates_with_requested_recipe(self):
        self.env["SID_RECIPE"] = "paper-tf-adagrad"
        result = self.run_job()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--recipe paper-tf-adagrad", Path(str(self.trace) + ".sid").read_text())

    def test_sid_rejects_unknown_recipe_before_work(self):
        self.env["SID_RECIPE"] = "adam"
        result = self.run_job()
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.trace.exists())

    def test_sid_refuses_to_reuse_catalog_from_another_recipe(self):
        self.write_sid({"paper_baseline": True})  # manifest from before recipes existed
        result = self.run_job()
        self.assertEqual(result.returncode, 2)
        self.assertIn("built with recipe paper-pytorch, not SID_RECIPE=stable", result.stderr)
        self.env["SID_RECIPE"] = "paper-pytorch"
        self.assertEqual(self.run_job().returncode, 0)

    def test_sid_reuses_completed_catalog(self):
        self.write_sid({"recipe": "stable"})
        result = self.run_job()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Reusing completed SID catalog", result.stdout)
        self.assertEqual(self.trace.read_text().splitlines(), [
            "pip", "scripts.prepare_amazon_reviews_2014_beauty",
        ])

    def test_failed_preparation_stops_before_generation(self):
        self.env["FAIL_PREPARE"] = "1"
        result = self.run_job()
        self.assertEqual(result.returncode, 17)
        self.assertEqual(self.trace.read_text().splitlines(), [
            "pip", "scripts.prepare_amazon_reviews_2014_beauty",
        ])

    def prepare_train_stage(self):
        self.env["STAGE"] = "train"
        split = self.root / "data/amazon_reviews/2014/train/sids/sentence-t5_rqvae/Beauty/test-v1"
        split.mkdir(parents=True)
        (split / "split_manifest.json").write_text("{}\n")

    def test_train_resumes_baseline_run_from_variant_directory(self):
        self.prepare_train_stage()
        run = self.root / "Recall/TIGER/baseline/runs/test-v1"
        run.mkdir(parents=True)
        (run / "latest.pt").write_text("")
        result = self.run_job()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--resume", Path(str(self.trace) + ".train").read_text())

    def test_train_refuses_run_left_at_legacy_family_path(self):
        self.prepare_train_stage()
        legacy = self.root / "Recall/TIGER/runs/test-v1"
        legacy.mkdir(parents=True)
        (legacy / "latest.pt").write_text("")
        result = self.run_job()
        self.assertEqual(result.returncode, 2)
        self.assertIn("legacy path", result.stderr)
        self.assertEqual(self.trace.read_text().splitlines(), ["pip"])


if __name__ == "__main__":
    unittest.main()
