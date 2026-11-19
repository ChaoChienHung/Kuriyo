"""Run directories shared by every trained model: layout, settings, checkpoints, selection.

Layout of ``runs/<RUN_NAME>/``::

    settings.json                    what defines the run: data and preprocessing, evaluation
                                     sets, model, training, seed, code revision
    events.log                       lifecycle events (core.run_logging)
    train/train.jsonl                one record per optimizer step
    checkpoints/latest.pt            resumable state (model, optimizer, sampler, RNG)
    checkpoints/best.pt              weights selected on validation
    validation/<set>/<unit>-<n>.json one file per evaluation, e.g. epoch-000005.json
    validation/<set>/best.json       metrics of checkpoints/best.pt
    test/<set>/metrics.json          written once, only by --eval-test-only

Runs written before 2026-11-19 are flat (``config.json``, ``latest.pt``,
``validation-epoch-5.json`` ...). ``open_run`` and ``open_for_test`` move them into this
layout after checking that their recorded settings match the invocation.
"""

import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

import torch

from core.experiment import git_revision, write_json
from core.run_logging import RunLogger

LEGACY_CONFIG = "config.json"
_MISSING = object()


class RunError(ValueError):
    """A run directory cannot be used as requested; the message says why."""


def eval_set_name(year, category):
    """Name of an evaluation set, e.g. amazon2014-beauty; one folder per set."""
    return "amazon{}-{}".format(year, category.lower())


class RunLayout:
    def __init__(self, root, eval_set):
        self.root = Path(root)
        self.eval_set = eval_set
        self.settings = self.root / "settings.json"
        self.events = self.root / "events.log"
        self.train_log = self.root / "train" / "train.jsonl"
        self.checkpoints = self.root / "checkpoints"
        self.latest = self.checkpoints / "latest.pt"
        self.best = self.checkpoints / "best.pt"
        self.validation = self.root / "validation" / eval_set
        self.best_metrics = self.validation / "best.json"
        self.test = self.root / "test" / eval_set
        self.test_metrics = self.test / "metrics.json"

    def validation_file(self, unit, value):
        return self.validation / "{}-{:06d}.json".format(unit, value)

    @property
    def is_legacy(self):
        return (self.root / LEGACY_CONFIG).is_file() and not self.settings.is_file()

    def logger(self):
        return RunLogger(self.root, train_file=str(self.train_log.relative_to(self.root)))


def display_path(path, root):
    """Repository-relative path when possible, so settings do not leak home directories."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)


def run_section(name, family, variant, implementation, root, device, seed):
    return {"name": name, "family": family, "variant": variant,
            "implementation": implementation, "git_revision": git_revision(root),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "device": device, "seed": seed}


def lookup(settings, dotted):
    value = settings
    for part in dotted.split("."):
        value = value[part]
    return value


def _normalized(value):
    return json.loads(json.dumps(value))


def _key_specs(keys):
    """keys maps a legacy config.json key to a dotted settings path, or to (path, default)
    when runs older than that key lacked it."""
    for legacy_key, spec in keys.items():
        dotted, default = spec if isinstance(spec, tuple) else (spec, _MISSING)
        yield legacy_key, dotted, default


def check_settings(previous, current, keys, what):
    for _, dotted, default in _key_specs(keys):
        try:
            old = lookup(previous, dotted)
        except KeyError:
            if default is _MISSING:
                raise RunError("{} mismatch: {} missing from the run's settings".format(what, dotted))
            old = default
        new = lookup(current, dotted)
        if _normalized(old) != _normalized(new):
            raise RunError("{} mismatch: {} (run has {!r}, this invocation {!r})".format(
                what, dotted, old, new))


def migrate_legacy(layout, settings, keys):
    """Move a flat pre-2026-11-19 run into the layout; keep config.json beside settings.json."""
    config = json.loads((layout.root / LEGACY_CONFIG).read_text())
    for legacy_key, dotted, default in _key_specs(keys):
        old = config.get(legacy_key, default)
        if old is _MISSING:
            raise RunError("legacy run lacks {} in {}".format(legacy_key, LEGACY_CONFIG))
        new = lookup(settings, dotted)
        if _normalized(old) != _normalized(new):
            raise RunError("legacy run differs from this invocation: {} (run has {!r}, this "
                           "invocation {!r})".format(legacy_key, old, new))
    moves = [(layout.root / "latest.pt", layout.latest), (layout.root / "best.pt", layout.best),
             (layout.root / "train.jsonl", layout.train_log),
             (layout.root / "best-metrics.json", layout.best_metrics),
             (layout.root / "test-metrics.json", layout.test_metrics)]
    for path in layout.root.glob("validation-*-*.json"):
        match = re.fullmatch(r"validation-(epoch|step)-(\d+)\.json", path.name)
        if match:
            moves.append((path, layout.validation_file(match.group(1), int(match.group(2)))))
    moves = [(source, target) for source, target in moves if source.exists()]
    for _, target in moves:
        if target.exists():
            raise RunError("cannot migrate legacy run; {} already exists".format(target))
    for source, target in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
    migrated = copy.deepcopy(settings)
    migrated["run"].update(git_revision=config.get("git_revision"), created_utc=None,
                           slurm_job_id=None,
                           migrated_utc=datetime.now(timezone.utc).isoformat(),
                           migrated_from="config.json of the flat layout used before 2026-11-19; "
                                         "settings not listed in its keys are this invocation's")
    write_json(layout.settings, migrated)
    with layout.logger() as logger:
        logger.event("legacy_layout_migrated", moved=[str(target.relative_to(layout.root))
                                                      for _, target in moves])


def open_run(layout, settings, keys, resume):
    """Create a fresh run or validate a resume; returns the run's recorded settings.

    keys lists what must match on resume (see _key_specs)."""
    if layout.is_legacy:
        if not resume:
            raise RunError("run directory already exists: {}".format(layout.root))
        migrate_legacy(layout, settings, keys)
    if resume:
        if not layout.latest.is_file():
            raise RunError("--resume requires {}".format(layout.latest))
        previous = json.loads(layout.settings.read_text())
        check_settings(previous, settings, keys, "resume settings")
        return previous
    if layout.root.exists():
        raise RunError("run directory already exists: {}".format(layout.root))
    layout.root.mkdir(parents=True)
    write_json(layout.settings, settings)
    return settings


def evaluate_test_once(layout, settings, keys, legacy_keys, model, evaluate, test_sha256):
    """Load checkpoints/best.pt, evaluate held-out test once, and write test/<set>/metrics.json.

    keys must match between the selected run and this invocation; legacy_keys are the
    resume keys used to migrate a flat run first."""
    if layout.is_legacy:
        migrate_legacy(layout, settings, legacy_keys)
    if not layout.best.is_file() or layout.test_metrics.exists():
        raise RunError("best checkpoint missing or test already evaluated")
    previous = json.loads(layout.settings.read_text())
    check_settings(previous, settings, keys, "test settings")
    model.load_state_dict(torch.load(layout.best, map_location="cpu", weights_only=True))
    with layout.logger() as logger:
        metrics = evaluate()
        metrics.update(split="test", set=layout.eval_set, test_sha256=test_sha256,
                       selected_validation=json.loads(layout.best_metrics.read_text()))
        layout.test.mkdir(parents=True, exist_ok=True)
        write_json(layout.test_metrics, metrics)
        logger.event("test_completed", metrics=metrics)
        print(json.dumps(metrics), flush=True)
    return metrics


def save_checkpoint(path, state):
    """Atomically save state plus the global torch (and CUDA) RNG state."""
    state = dict(state, torch_rng_state=torch.get_rng_state())
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, device):
    """Load a checkpoint saved by save_checkpoint and restore the global RNG state."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    torch.set_rng_state(state["torch_rng_state"])
    if device == "cuda" and "cuda_rng_state" in state:
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    return state


class Selection:
    """Validation bookkeeping: keeps the best NDCG@10 checkpoint and stops after
    `patience` evaluations without gain (0 never stops)."""

    def __init__(self, patience=0, best=-1.0, bad_evals=0, stopped=False):
        self.patience, self.best, self.bad_evals, self.stopped = patience, best, bad_evals, stopped

    @classmethod
    def from_checkpoint(cls, state, patience=0):
        return cls(patience, state["best_ndcg10"], state.get("bad_evals", 0),
                   state.get("stopped_early", False))

    def state(self):
        return {"best_ndcg10": self.best, "bad_evals": self.bad_evals,
                "stopped_early": self.stopped}

    def record(self, layout, logger, model, metrics, unit, value):
        """Write one validation result; save best.pt when NDCG@10 improves."""
        layout.validation.mkdir(parents=True, exist_ok=True)
        write_json(layout.validation_file(unit, value), metrics)
        logger.event("validation", metrics=metrics)
        print(json.dumps(metrics), flush=True)
        if metrics["ndcg@10"] > self.best:
            self.best, self.bad_evals = metrics["ndcg@10"], 0
            layout.checkpoints.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), layout.best)
            write_json(layout.best_metrics, metrics)
        else:
            self.bad_evals += 1
        self.stopped = bool(self.patience) and self.bad_evals >= self.patience


def add_run_arguments(parser, runs_dir):
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--runs-dir", type=Path, default=runs_dir)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-test-only", action="store_true",
                        help="evaluate checkpoints/best.pt on held-out test once")


def check_run_arguments(args):
    """Return an error message for invalid shared arguments, or None."""
    if Path(args.run_name).name != args.run_name or args.run_name in ("", ".", ".."):
        return "--run-name must be one directory name"
    for limit in (getattr(args, "max_train_examples", None), getattr(args, "max_validation_examples", None)):
        if limit is not None and limit < 1:
            return "example limits must be positive"
    if args.device == "cuda" and not torch.cuda.is_available():
        return "CUDA is unavailable"
    if args.device == "mps" and not torch.backends.mps.is_available():
        return "MPS is unavailable"
    return None


def configure_threads(device):
    if device == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
