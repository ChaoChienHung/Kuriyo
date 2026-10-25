"""Initial TIGER configuration, based on the paper's Beauty setup."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TIGERConfig:
    max_history: int = 20
    content_embedding_dim: int = 768
    semantic_id_levels: int = 3
    codebook_size: int = 256
    codeword_dim: int = 32
    append_collision_token: bool = True
