"""Metrics for one relevant next item per recommendation request."""

import math


def ranking_metrics(recommendations_and_targets, ks=(5, 10)):
    """Return Recall@K and NDCG@K for a sequence of (candidates, target)."""
    if not ks or any(k < 1 for k in ks):
        raise ValueError("ks must contain positive integers")
    hits = {k: 0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    count = 0
    for recommendations, target in recommendations_and_targets:
        count += 1
        for k in ks:
            try:
                rank = recommendations[:k].index(target) + 1
            except ValueError:
                continue
            hits[k] += 1
            ndcg[k] += 1 / math.log2(rank + 1)
    if not count:
        raise ValueError("cannot evaluate an empty set of requests")
    return {**{"recall@{}".format(k): hits[k] / count for k in ks},
            **{"ndcg@{}".format(k): ndcg[k] / count for k in ks}}
