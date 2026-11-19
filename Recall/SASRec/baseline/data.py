"""Build SASRec training windows from the prepared Amazon 2014 ASIN split.

Catalog and example loading live in Recall.core.data.
"""

import torch


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


def pad_sequences(sequences):
    """(B, S) item tensor, right-padded with 0; the items each row must not sample."""
    seen = torch.zeros(len(sequences), max(map(len, sequences)), dtype=torch.long)
    for index, sequence in enumerate(sequences):
        seen[index, :len(sequence)] = torch.as_tensor(sequence)
    return seen


def sample_negatives(targets, seen, num_items, generator):
    """One negative per position, uniform over items 1..num_items not in that row's seen.

    As in the original SASRec code (random_neq over the user's train items): rejection
    sampling, so each negative is uniform over the user's unseen items. targets is
    (B, L); seen is (B, S) from pad_sequences and must leave at least one item unseen.
    """
    if (seen.ne(0).sum(1) >= num_items).any():
        raise ValueError("a user has seen every item; no negative can be sampled")
    negatives = torch.randint(1, num_items + 1, targets.shape, generator=generator)
    clash = negatives[..., None].eq(seen[:, None, :]).any(-1)
    while clash.any():
        negatives[clash] = torch.randint(1, num_items + 1, (int(clash.sum()),), generator=generator)
        clash = negatives[..., None].eq(seen[:, None, :]).any(-1)
    return negatives


def collate_histories(rows, max_history):
    """Left-pad the most recent max_history items; the last position is the newest item."""
    inputs = torch.zeros(len(rows), max_history, dtype=torch.long)
    for index, row in enumerate(rows):
        history = row["history"][-max_history:]
        inputs[index, max_history - len(history):] = torch.as_tensor(history)
    target = torch.tensor([row["target"] for row in rows], dtype=torch.long)
    return inputs, target
