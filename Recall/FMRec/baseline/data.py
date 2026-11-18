"""Read the prepared Amazon 2014 ASIN splits into item-index sequences."""

import json

import torch


def load_catalog(path):
    """Return ASINs in catalog order; item index i + 1 belongs to asins[i]."""
    with path.open(encoding="utf-8") as source:
        asins = [json.loads(line)["asin"] for line in source if line.strip()]
    if len(set(asins)) != len(asins):
        raise ValueError("duplicate ASIN in catalog {}".format(path))
    return asins


def read_examples(path, item_index, limit=None):
    """Rows with integer history/target; unknown ASINs are an error, not a skip."""
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if limit is not None and len(rows) >= limit:
                break
            row = json.loads(line)
            try:
                history = [item_index[asin] for asin in row["history"]]
                target = item_index[row["target_asin"]]
            except KeyError as error:
                raise ValueError("{} has an ASIN outside the catalog: {}".format(path, error))
            if not history:
                raise ValueError("{} has an empty history".format(path))
            rows.append({"user_id": row["user_id"], "history": history, "target": target})
    if not rows:
        raise ValueError("no examples in {}".format(path))
    return rows


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
