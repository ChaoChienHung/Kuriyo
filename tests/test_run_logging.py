"""Scheduler-independent logging and resume contracts."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from core.run_logging import RunLogger


class RunLoggingTests(unittest.TestCase):
    def test_resume_appends_and_identifies_sessions(self):
        with TemporaryDirectory() as directory:
            for _ in range(2):
                with RunLogger(directory) as logger:
                    logger.train(step=1, loss=0.5, learning_rate=0.01)
            rows = [json.loads(line) for line in (Path(directory) / 'train.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertNotEqual(rows[0]['session_id'], rows[1]['session_id'])
            self.assertTrue(all(row['timestamp'] and row['elapsed_seconds'] >= 0 for row in rows))

    def test_error_is_recorded_and_not_swallowed(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'bad batch'):
                with RunLogger(directory):
                    raise ValueError('bad batch')
            rows = [json.loads(line) for line in (Path(directory) / 'events.log').read_text().splitlines()]
            self.assertEqual(rows[-1]['event'], 'session_failed')
            self.assertIn('bad batch', rows[-1]['traceback'])
