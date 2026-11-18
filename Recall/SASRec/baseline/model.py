"""SASRec (Kang & McAuley, ICDM 2018): causal self-attention over item IDs."""

import math

import torch
from torch import nn
import torch.nn.functional as F

LOSSES = ("ce", "bce")


def attention_allowed(sequence):
    """(B, L, L) bool: query i may attend key j iff j <= i and j is a real item.

    A padding query attends only itself, so no row is fully masked (no NaN), and
    real positions never see padding; outputs at real positions are therefore
    independent of how much left padding precedes them.
    """
    length = sequence.shape[1]
    causal = torch.ones(length, length, dtype=torch.bool, device=sequence.device).tril()
    real_key = sequence.ne(0)[:, None, :]
    diagonal = torch.eye(length, dtype=torch.bool, device=sequence.device)
    return (causal & real_key) | diagonal


class SASRec(nn.Module):
    def __init__(self, num_items, max_history=20, hidden=64, heads=2, layers=2, dropout=0.5):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.num_items = num_items
        self.max_history = max_history
        self.heads = heads
        self.item_embedding = nn.Embedding(num_items + 1, hidden, padding_idx=0)
        self.position_embedding = nn.Embedding(max_history, hidden)
        self.embedding_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=hidden, dropout=dropout,
                                           activation="relu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(hidden)
        nn.init.normal_(self.item_embedding.weight, std=hidden ** -0.5)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def forward(self, sequence):
        """Return (B, L, H) hidden states; sequence is left-padded to width max_history."""
        if sequence.shape[1] != self.max_history:
            raise ValueError("sequence width must equal max_history (right-aligned positions)")
        positions = torch.arange(self.max_history, device=sequence.device)
        hidden = self.item_embedding(sequence) * math.sqrt(self.item_embedding.embedding_dim)
        hidden = self.embedding_dropout(hidden + self.position_embedding(positions))
        hidden = hidden * sequence.ne(0).unsqueeze(-1)
        blocked = ~attention_allowed(sequence).repeat_interleave(self.heads, dim=0)
        hidden = self.encoder(hidden, mask=blocked)
        return self.final_norm(hidden)

    def scores(self, hidden):
        """Inner product with every real item; column j is item j + 1."""
        return hidden @ self.item_embedding.weight[1:].T

    def loss(self, inputs, targets, kind="ce", generator=None):
        hidden = self.forward(inputs)
        real = targets.ne(0)
        hidden, targets = hidden[real], targets[real]
        if kind == "ce":
            return F.cross_entropy(self.scores(hidden), targets - 1)
        if kind != "bce":
            raise ValueError("loss must be one of {}".format(LOSSES))
        negatives = torch.randint(1, self.num_items + 1, targets.shape, generator=generator).to(targets.device)
        clash = negatives.eq(targets)
        negatives[clash] = negatives[clash] % self.num_items + 1
        positive = (hidden * self.item_embedding(targets)).sum(-1)
        negative = (hidden * self.item_embedding(negatives)).sum(-1)
        return (F.binary_cross_entropy_with_logits(positive, torch.ones_like(positive))
                + F.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative)))
