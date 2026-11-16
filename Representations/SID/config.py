"""Initial configuration for the Sentence-T5 + RQ-VAE SID producer."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SIDConfig:
    text_model: str = "sentence-transformers/sentence-t5-base"
    content_embedding_dim: int = 768
    semantic_id_levels: int = 3
    codebook_size: int = 256
    codeword_dim: int = 32
    append_collision_token: bool = True
