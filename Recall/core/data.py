"""Batch helpers shared by token-based Recall models."""

import torch


def pad_token_sequences(sequences, pad_id=0):
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("sequences must be nonempty")
    padded = torch.full((len(sequences), max(map(len, sequences))), pad_id,
                        dtype=torch.long)
    for index, sequence in enumerate(sequences):
        padded[index, :len(sequence)] = torch.as_tensor(sequence, dtype=torch.long)
    return padded
