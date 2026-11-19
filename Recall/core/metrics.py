"""Metrics for one relevant next item per recommendation request."""

import math


def _summarize(hits, ndcg, count, ks):
    return {**{"recall@{}".format(k): hits[k] / count for k in ks},
            **{"ndcg@{}".format(k): ndcg[k] / count for k in ks}}


def ranking_metrics(recommendations_and_targets, ks=(5, 10), groups=None):
    """Return Recall@K and NDCG@K for a sequence of (candidates, target).

    groups, when given, holds one label per request; the result then also has
    "groups": {label: metrics and "examples"} (labels sorted). Overall numbers are
    accumulated exactly as without groups."""
    if not ks or any(k < 1 for k in ks):
        raise ValueError("ks must contain positive integers")
    hits = {k: 0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    count = 0
    by_group = {}
    labels = iter(groups) if groups is not None else None
    for recommendations, target in recommendations_and_targets:
        count += 1
        group = None
        if labels is not None:
            label = next(labels, None)
            if label is None:
                raise ValueError("fewer groups than requests")
            group = by_group.setdefault(label, ({k: 0 for k in ks}, {k: 0.0 for k in ks}, [0]))
            group[2][0] += 1
        for k in ks:
            try:
                rank = recommendations[:k].index(target) + 1
            except ValueError:
                continue
            hits[k] += 1
            ndcg[k] += 1 / math.log2(rank + 1)
            if group is not None:
                group[0][k] += 1
                group[1][k] += 1 / math.log2(rank + 1)
    if not count:
        raise ValueError("cannot evaluate an empty set of requests")
    result = _summarize(hits, ndcg, count, ks)
    if labels is not None:
        if next(labels, None) is not None:
            raise ValueError("more groups than requests")
        result["groups"] = {label: {**_summarize(*by_group[label][:2], by_group[label][2][0], ks),
                                    "examples": by_group[label][2][0]}
                            for label in sorted(by_group)}
    return result
