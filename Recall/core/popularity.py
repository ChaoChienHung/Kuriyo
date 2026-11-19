"""Group evaluation requests by how often their target was a training target.

Overall metrics are dominated by popular items; grouping shows how a model does on
items it rarely or never learned to predict (cold start and the long tail). The group
of a request depends only on the train split, so every model gets the same groups.
"""

from collections import Counter

GROUPS = ("0", "1-4", "5+")
GROUPING = "times the target is a target in the train split used for training: 0, 1-4, 5+"


def target_counts(targets):
    """Counter of train targets (ASINs or catalog indices; use the same kind for requests)."""
    return Counter(targets)


def popularity_group(count):
    if count < 0:
        raise ValueError("count must be non-negative")
    return "0" if count == 0 else "1-4" if count < 5 else "5+"


def popularity_groups(targets, counts):
    """One group label per request target, in request order."""
    return [popularity_group(counts.get(target, 0)) for target in targets]
