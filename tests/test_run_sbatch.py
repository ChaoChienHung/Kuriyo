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
  scripts.ensure_amazon_reviews_2014)
    if [ "$FAIL_PREPARE" = 1 ]; then exit 17; fi
    test "$3" = --category
    mkdir -p "data/amazon_reviews/2014/catalog/$4"
    echo '{}' > "data/amazon_reviews/2014/catalog/$4/items.jsonl"
    for split in train validation; do
      mkdir -p "data/amazon_reviews/2014/$split/raw/$4"
      echo '{}' > "data/amazon_reviews/2014/$split/raw/$4/examples.jsonl"
    done
    ;;
  Representations.SID.scripts.generate_sids)
    test "$3" = --items
    test -s "$4"
    echo "$*" > "$RUN_TRACE.sid"
    ;;
  Recall.TIGER.baseline.train)
    echo "$*" > "$RUN_TRACE.train"
    ;;
  Recall.SASRec.baseline.train|Recall.FMRec.baseline.train)
    echo "$*" > "$RUN_TRACE.$2"
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
            "pip", "scripts.ensure_amazon_reviews_2014",
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
            "pip", "scripts.ensure_amazon_reviews_2014",
        ])

    def test_failed_preparation_stops_before_generation(self):
        self.env["FAIL_PREPARE"] = "1"
        result = self.run_job()
        self.assertEqual(result.returncode, 17)
        self.assertEqual(self.trace.read_text().splitlines(), [
            "pip", "scripts.ensure_amazon_reviews_2014",
        ])

    def prepare_train_stage(self):
        self.env["STAGE"] = "train"
        split = self.root / "data/amazon_reviews/2014/train/sids/sentence-t5_rqvae/Beauty/test-v1"
        split.mkdir(parents=True)
        (split / "split_manifest.json").write_text("{}\n")

    def test_train_resumes_baseline_run_from_variant_directory(self):
        # A flat run from before 2026-11-19; the trainer moves it into checkpoints/.
        self.prepare_train_stage()
        run = self.root / "Recall/TIGER/baseline/runs/test-v1"
        run.mkdir(parents=True)
        (run / "latest.pt").write_text("")
        result = self.run_job()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--resume", Path(str(self.trace) + ".train").read_text())

    def test_train_uses_t5x_adafactor_scaling_unless_disabled(self):
        self.prepare_train_stage()
        result = self.run_job()
        self.assertEqual(result.returncode, 0, result.stderr)
        args = Path(str(self.trace) + ".train").read_text().split()
        self.assertIn("--adafactor-scale-parameter", args)
        self.assertIn("adafactor_scale_parameter=1", result.stdout)
        self.env["ADAFACTOR_SCALE_PARAMETER"] = "0"
        result = self.run_job()
        self.assertEqual(result.returncode, 0, result.stderr)
        args = Path(str(self.trace) + ".train").read_text().split()
        self.assertIn("--no-adafactor-scale-parameter", args)
        self.assertNotIn("--adafactor-scale-parameter", args)

    def test_invalid_adafactor_scaling_stops_before_work(self):
        self.env["ADAFACTOR_SCALE_PARAMETER"] = "yes"
        result = self.run_job()
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.trace.exists())

    def test_train_refuses_run_left_at_legacy_family_path(self):
        self.prepare_train_stage()
        legacy = self.root / "Recall/TIGER/runs/test-v1"
        legacy.mkdir(parents=True)
        (legacy / "latest.pt").write_text("")
        result = self.run_job()
        self.assertEqual(result.returncode, 2)
        self.assertIn("legacy path", result.stderr)
        self.assertEqual(self.trace.read_text().splitlines(), ["pip"])

    def prepare_raw_split(self):
        base = self.root / "data/amazon_reviews/2014"
        for name in ("catalog/Beauty/items.jsonl", "train/raw/Beauty/examples.jsonl",
                     "validation/raw/Beauty/examples.jsonl"):
            (base / name).parent.mkdir(parents=True, exist_ok=True)
            (base / name).write_text("{}\n")

    def model_args(self, module):
        return Path(f"{self.trace}.{module}").read_text().split()

    def test_sasrec_train_runs_its_variant_without_sid_or_preparation(self):
        self.env.update(MODEL="sasrec", STAGE="train")
        self.prepare_raw_split()
        result = self.run_job()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.trace.read_text().splitlines(),
                         ["pip", "Recall.SASRec.baseline.train"])
        self.assertEqual(self.model_args("Recall.SASRec.baseline.train")[2:],
                         ["--category", "Beauty", "--seed", "42", "--run-name", "test-v1",
                          "--device", "cuda"])

    def test_category_seed_and_train_args_reach_train_and_test(self):
        self.env.update(MODEL="sasrec", STAGE="all", CATEGORY="Toys_and_Games", SEED="7",
                        TRAIN_ARGS="--loss bce")
        del self.env["RUN_NAME"]
        result = self.run_job()
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.model_args("Recall.SASRec.baseline.train")[2:]
        self.assertEqual(args, ["--category", "Toys_and_Games", "--seed", "7", "--run-name",
                                "sasrec-paper-toys_and_games-v1", "--device", "cuda", "--loss", "bce"])
        self.env["STAGE"] = "test"
        checkpoints = self.root / "Recall/SASRec/baseline/runs/sasrec-paper-toys_and_games-v1/checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        (checkpoints / "best.pt").write_text("")
        result = self.run_job()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.model_args("Recall.SASRec.baseline.train")[2:], args + ["--eval-test-only"])

    def test_tiger_category_uses_its_own_sid_and_run(self):
        self.env.update(CATEGORY="Sports_and_Outdoors")
        del self.env["RUN_NAME"]
        result = self.run_job()
        self.assertEqual(result.returncode, 0, result.stderr)
        sid_args = Path(str(self.trace) + ".sid").read_text().split()
        self.assertIn("data/amazon_reviews/2014/catalog/Sports_and_Outdoors/items.jsonl", sid_args)
        self.assertIn("data/amazon_reviews/2014/artifacts/sids/sentence-t5_rqvae/"
                      "Sports_and_Outdoors/test-v1", sid_args)
        self.assertIn("run=tiger-paper-sports_and_outdoors-v1", result.stdout)

    def test_fmrec_all_prepares_then_resumes_existing_run(self):
        self.env.update(MODEL="fmrec", STAGE="all")
        self.prepare_raw_split()
        checkpoints = self.root / "Recall/FMRec/baseline/runs/test-v1/checkpoints"
        checkpoints.mkdir(parents=True)
        (checkpoints / "latest.pt").write_text("")
        result = self.run_job()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.trace.read_text().splitlines(), [
            "pip", "scripts.ensure_amazon_reviews_2014", "Recall.FMRec.baseline.train"])
        self.assertIn("--resume", self.model_args("Recall.FMRec.baseline.train"))

    def test_model_train_refuses_run_without_checkpoint(self):
        self.env.update(MODEL="fmrec", STAGE="train")
        self.prepare_raw_split()
        (self.root / "Recall/FMRec/baseline/runs/test-v1").mkdir(parents=True)
        result = self.run_job()
        self.assertEqual(result.returncode, 2)
        self.assertIn("without checkpoint", result.stderr)
        self.assertEqual(self.trace.read_text().splitlines(), ["pip"])

    def test_model_train_stops_when_split_is_missing(self):
        self.env.update(MODEL="sasrec", STAGE="train")
        result = self.run_job()
        self.assertEqual(result.returncode, 2)
        self.assertIn("STAGE=prepare", result.stderr)
        self.assertEqual(self.trace.read_text().splitlines(), ["pip"])

    def test_model_test_stage_evaluates_once(self):
        self.env.update(MODEL="sasrec", STAGE="test")
        result = self.run_job()
        self.assertEqual(result.returncode, 2, "no run directory here")
        self.assertIn("No best checkpoint for STAGE=test", result.stderr)
        run = self.root / "Recall/SASRec/baseline/runs/test-v1"
        run.mkdir(parents=True)
        (run / "best.pt").write_text("")  # flat run from before 2026-11-19
        result = self.run_job()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--eval-test-only", self.model_args("Recall.SASRec.baseline.train"))

    def test_groups_stage_reevaluates_without_training(self):
        self.env.update(MODEL="fmrec", STAGE="groups")
        checkpoints = self.root / "Recall/FMRec/baseline/runs/test-v1/checkpoints"
        checkpoints.mkdir(parents=True)
        (checkpoints / "best.pt").write_text("")
        result = self.run_job()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.trace.read_text().splitlines(), ["pip", "Recall.FMRec.baseline.train"])
        args = self.model_args("Recall.FMRec.baseline.train")
        self.assertIn("--eval-groups", args)
        self.assertNotIn("--eval-test-only", args)
        self.env.update(MODEL="tiger", RUN_NAME="tiger-none")
        result = self.run_job()
        self.assertEqual(result.returncode, 2)
        self.assertIn("No best checkpoint for STAGE=groups", result.stderr)

    def test_invalid_model_settings_stop_before_work(self):
        for settings in ({"MODEL": "bert4rec"}, {"MODEL": "sasrec", "STAGE": "sid"},
                         {"CATEGORY": "Books"}, {"SEED": "-1"}, {"TRAIN_ARGS": "--loss bce; rm"},
                         {"MODEL": "fmrec", "STAGE": "train", "ADAFACTOR_SCALE_PARAMETER": "1"}):
            with self.subTest(settings=settings):
                env = dict(self.env, **settings)
                result = subprocess.run(["bash", str(SCRIPT)], cwd=self.root, env=env,
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(self.trace.exists())


if __name__ == "__main__":
    unittest.main()
