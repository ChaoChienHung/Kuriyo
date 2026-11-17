"""Compatibility entry point for TIGER sequence training.

The implementation lives in ``Recall.TIGER.baseline.train_sequence``.
"""

from pathlib import Path
import sys


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Recall.TIGER.baseline.train_sequence import main


if __name__ == "__main__":
    main()
