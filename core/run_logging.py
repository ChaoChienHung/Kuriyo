"""Append-only, single-writer run logs independent of the job scheduler."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback
import uuid


class RunLogger:
    """One session per invocation; repeated steps after resume remain distinguishable."""

    def __init__(self, run_dir, train_file='train.jsonl'):
        """train_file is relative to run_dir; core.runs.RunLayout uses train/train.jsonl."""
        self.run_dir = Path(run_dir)
        self.train_file = train_file
        self.session_id = uuid.uuid4().hex
        self.started = time.monotonic()

    def _append(self, filename, values):
        record = dict(values, timestamp=datetime.now(timezone.utc).isoformat(),
                      session_id=self.session_id,
                      elapsed_seconds=time.monotonic() - self.started)
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        path = self.run_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as output:
            output.write(line + '\n')

    def train(self, **values):
        self._append(self.train_file, values)

    def event(self, name, **values):
        self._append('events.log', dict(values, event=name))

    def __enter__(self):
        self.event('session_started')
        return self

    def __exit__(self, error_type, error, tb):
        if error is None:
            self.event('session_completed')
        else:
            self.event('session_failed', error_type=error_type.__name__,
                       message=str(error), traceback=''.join(traceback.format_exception(error_type, error, tb)))
        return False
