"""Read the prepared Amazon 2014 ASIN splits and build SASRec training windows."""

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


def user_sequences(rows, max_history):
    """Rebuild each user's chronological train sequence from its prefix examples.

    The prepared train file holds one example per prefix, in order, contiguous per
    user. The sequence is the first history item followed by every target; each
    example's (capped) history must equal the sequence before its target, so a
    reordered or edited file fails loudly instead of training on wrong order.
    """
    sequences = {}
    order = []
    for row in rows:
        user = row["user_id"]
        if user not in sequences:
            if len(row["history"]) != 1:
                raise ValueError("user {} does not start with a length-1 prefix".format(user))
            sequences[user] = list(row["history"])
            order.append(user)
        elif order[-1] != user:
            raise ValueError("train examples of user {} are not contiguous".format(user))
        sequence = sequences[user]
        if row["history"] != sequence[-max_history:]:
            raise ValueError("train example of user {} is not the next prefix".format(user))
        sequence.append(row["target"])
    return [sequences[user] for user in order]


def windows(sequence, max_history):
    """Split one sequence into windows of at most max_history + 1 items, from the end.

    Consecutive windows share one item, so every item after the first is a
    training target exactly once.
    """
    result = []
    end = len(sequence)
    while end > 1:
        start = max(0, end - max_history - 1)
        result.append(sequence[start:end])
        end = start + 1
    return result[::-1]


def collate_windows(batch, max_history):
    """Left-pad inputs w[:-1] and targets w[1:] to max_history; target 0 is ignored."""
    inputs = torch.zeros(len(batch), max_history, dtype=torch.long)
    targets = torch.zeros(len(batch), max_history, dtype=torch.long)
    for index, window in enumerate(batch):
        length = len(window) - 1
        inputs[index, max_history - length:] = torch.as_tensor(window[:-1])
        targets[index, max_history - length:] = torch.as_tensor(window[1:])
    return inputs, targets


def collate_histories(rows, max_history):
    """Left-pad the most recent max_history items; the last position is the newest item."""
    inputs = torch.zeros(len(rows), max_history, dtype=torch.long)
    for index, row in enumerate(rows):
        history = row["history"][-max_history:]
        inputs[index, max_history - len(history):] = torch.as_tensor(history)
    target = torch.tensor([row["target"] for row in rows], dtype=torch.long)
    return inputs, target
