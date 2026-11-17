"""Initial TIGER sequence-model configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TIGERConfig:
    max_history: int = 20
    d_model: int = 128
    heads: int = 4
    layers: int = 2
    feedforward_dim: int = 256
