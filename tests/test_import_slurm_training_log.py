import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from scripts.import_slurm_training_log import import_log


class SlurmImportTests(unittest.TestCase):
    def test_sparse_stages_provenance_and_no_invented_completion(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'slurm.out'
            source.write_text('job=12 commit=abc stage=all run=example\n'
                              'epoch 100/20000 loss=0.5\n'
                              'step 100/200000 loss=2.0 lr=0.01\n'
                              '{"validation_examples": 3, "ndcg@10": 0.1, "step": 100}\n')
            output = root / 'archive'
            counts = import_log(source, output)
            self.assertEqual(counts['sid']['train'], 1)
            self.assertEqual(counts['tiger']['train'], 1)
            row = json.loads((output / 'tiger/train.jsonl').read_text())
            self.assertEqual(row['source_line'], 3)
            self.assertIsNone(row['timestamp'])
            self.assertEqual(row['step'], 100)
            events = [json.loads(line)['event'] for line in (output / 'tiger/events.log').read_text().splitlines()]
            self.assertEqual(events, ['source_job', 'validation'])
            original = (output / 'tiger/train.jsonl').read_bytes()
            with self.assertRaises(FileExistsError):
                import_log(source, output)
            self.assertEqual((output / 'tiger/train.jsonl').read_bytes(), original)

    def test_unrecognized_log_does_not_create_archive(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'empty.out'
            source.write_text('pip installed\n')
            with self.assertRaises(ValueError):
                import_log(source, root / 'archive')
            self.assertFalse((root / 'archive').exists())
