"""Full-catalog ranking evaluation shared by item-ID Recall models."""

import torch

from Recall.core.metrics import ranking_metrics

FULL_CATALOG = "full catalog, inner product with item embeddings; history items not filtered"


@torch.no_grad()
def rank_full_catalog(rows, asins, score_batch, batch_size, num_items, ks=(5, 10), index_offset=0):
    """Rank every catalog item for each row; score_batch(rows) returns (rows, columns) scores.

    Column c is item asins[c - index_offset]: 0 when column 0 is the first item, 1 when
    column 0 is padding. The caller puts the model in eval mode."""
    requests = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        top = score_batch(batch).topk(min(max(ks), num_items), dim=-1).indices.cpu().tolist()
        requests.extend(([asins[index - index_offset] for index in indices], asins[row["target"] - 1])
                        for indices, row in zip(top, batch))
    return {**ranking_metrics(requests, ks=ks), "examples": len(rows), "ranking": FULL_CATALOG}
