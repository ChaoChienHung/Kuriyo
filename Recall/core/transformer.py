"""Reusable encoder-decoder Transformer for token-based Recall models."""

import math

import torch
from torch import nn


class EncoderDecoderTransformer(nn.Module):
    def __init__(self, vocab_size, max_source_length, max_target_length,
                 d_model=128, heads=4, layers=2, feedforward_dim=256, dropout=0.1):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        if min(vocab_size, max_source_length, max_target_length) < 1:
            raise ValueError("vocabulary and sequence lengths must be positive")
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.source_position = nn.Embedding(max_source_length, d_model)
        self.target_position = nn.Embedding(max_target_length, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, heads, feedforward_dim, dropout,
                                       batch_first=True, norm_first=True), layers)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, heads, feedforward_dim, dropout,
                                       batch_first=True, norm_first=True), layers)
        self.output = nn.Linear(d_model, vocab_size)
        self.scale = math.sqrt(d_model)

    def encode(self, source):
        positions = torch.arange(source.shape[1], device=source.device)
        source_mask = source.eq(0)
        hidden = self.token_embedding(source) * self.scale + self.source_position(positions)
        return self.encoder(hidden, src_key_padding_mask=source_mask), source_mask

    def decode(self, target_input, memory, source_mask):
        positions = torch.arange(target_input.shape[1], device=target_input.device)
        hidden = self.token_embedding(target_input) * self.scale + self.target_position(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            target_input.shape[1], device=target_input.device)
        decoded = self.decoder(hidden, memory, tgt_mask=causal_mask,
                               memory_key_padding_mask=source_mask)
        return self.output(decoded)

    def forward(self, source, target_input):
        memory, source_mask = self.encode(source)
        return self.decode(target_input, memory, source_mask)
