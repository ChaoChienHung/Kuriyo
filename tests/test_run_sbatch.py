"""Exercise batch-stage orchestration without Slurm, downloads, or a GPU."""

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
    test "$4" = -r
    test "$5" = requirements-soc.txt
    ;;
  scripts.prepare_amazon_reviews_2014_beauty)
    if [ "$FAIL_PREPARE" = 1 ]; then exit 17; fi
    mkdir -p data/amazon_reviews/2014/catalog/Beauty
    echo '{}' > data/amazon_reviews/2014/catalog/Beauty/items.jsonl
    ;;
  Representations.SID.scripts.generate_sids)
    test -s data/amazon_reviews/2014/catalog/Beauty/items.jsonl
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

    def test_sid_reuses_completed_catalog(self):
        sid = self.root / (
            "data/amazon_reviews/2014/artifacts/sids/"
            "sentence-t5_rqvae/Beauty/test-v1")
        sid.mkdir(parents=True)
        for name in ("semantic_ids.jsonl", "manifest.json"):
            (sid / name).write_text("{}\n")
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


if __name__ == "__main__":
    unittest.main()
