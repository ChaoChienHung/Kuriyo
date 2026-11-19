"""Batch the prepared Amazon 2014 ASIN split for FMRec.

Catalog and example loading live in Recall.core.data.
"""

import torch


def collate(rows, max_history):
    """Left-pad the most recent `max_history` items to the batch maximum.

    The last position is always the most recent real item; the model reads its
    prediction there, as upstream FMRec does.
    """
    histories = [row["history"][-max_history:] for row in rows]
    width = max(map(len, histories))
    sequence = torch.zeros(len(rows), width, dtype=torch.long)
    for index, history in enumerate(histories):
        sequence[index, width - len(history):] = torch.as_tensor(history)
    target = torch.tensor([row["target"] for row in rows], dtype=torch.long)
    return sequence, target
