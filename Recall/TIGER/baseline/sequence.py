"""Small encoder-decoder Transformer for next-item Semantic ID prediction."""

from collections import defaultdict

import torch

from Recall.core.transformer import EncoderDecoderTransformer


class SIDTokenizer:
    PAD = 0
    BOS = 1
    EOS = 2

    def __init__(self, semantic_ids, codebook_size):
        if not semantic_ids:
            raise ValueError("empty SID catalog")
        self.semantic_ids = semantic_ids
        self.codebook_size = codebook_size
        self.collision_offset = 3 + 3 * codebook_size
        max_collision = max(sid[3] for sid in semantic_ids.values())
        self.vocab_size = self.collision_offset + max_collision + 1
        self.item_tokens = {asin: tuple(self.encode_sid(sid))
                            for asin, sid in semantic_ids.items()}
        self.sid_item = {tuple(sid): asin for asin, sid in semantic_ids.items()}
        self.token_item = {tokens: asin for asin, tokens in self.item_tokens.items()}
        if len(self.token_item) != len(self.item_tokens) or len(self.sid_item) != len(self.item_tokens):
            raise ValueError("SIDs are not unique")
        children = defaultdict(set)
        for tokens in self.item_tokens.values():
            for depth in range(4):
                children[tokens[:depth]].add(tokens[depth])
        self.children = {key: sorted(value) for key, value in children.items()}

    def encode_sid(self, sid):
        if len(sid) != 4:
            raise ValueError("expected three semantic codes and one collision token")
        if any(not 0 <= code < self.codebook_size for code in sid[:3]):
            raise ValueError("semantic code outside codebook")
        if sid[3] < 0:
            raise ValueError("collision token must be nonnegative")
        return [3 + level * self.codebook_size + code for level, code in enumerate(sid[:3])] + [
            self.collision_offset + sid[3]]

    def history_tokens(self, history, max_history):
        if not history:
            raise ValueError("empty history")
        result = []
        for asin in history[-max_history:]:
            result.extend(self.item_tokens[asin])
        return result

    def target_tokens(self, asin):
        return [*self.item_tokens[asin], self.EOS]


class TIGERTransformer(EncoderDecoderTransformer):
    def __init__(self, vocab_size, max_history=20, d_model=128, heads=4,
                 layers=2, feedforward_dim=256, dropout=0.1):
        super().__init__(vocab_size, max_source_length=max_history * 4,
                         max_target_length=5, d_model=d_model, heads=heads,
                         layers=layers, feedforward_dim=feedforward_dim,
                         dropout=dropout)


@torch.no_grad()
def constrained_beam_search(model, tokenizer, history, max_history, beam_size, device):
    """Decode only SIDs present in the item catalog; return ranked ASINs."""
    source = torch.tensor([tokenizer.history_tokens(history, max_history)],
                          dtype=torch.long, device=device)
    memory, source_mask = model.encode(source)
    beams = [((), 0.0)]
    for _ in range(4):
        candidates = []
        for prefix, score in beams:
            decoder_input = torch.tensor([[tokenizer.BOS, *prefix]],
                                         dtype=torch.long, device=device)
            logits = model.decode(decoder_input, memory, source_mask)[0, -1]
            log_probs = logits.log_softmax(-1)
            for token in tokenizer.children[prefix]:
                candidates.append(((*prefix, token), score + log_probs[token].item()))
        beams = sorted(candidates, key=lambda candidate: candidate[1], reverse=True)[:beam_size]
    completed = []
    for prefix, score in beams:
        decoder_input = torch.tensor([[tokenizer.BOS, *prefix]],
                                     dtype=torch.long, device=device)
        logits = model.decode(decoder_input, memory, source_mask)[0, -1]
        score += logits.log_softmax(-1)[tokenizer.EOS].item()
        completed.append((prefix, score))
    completed.sort(key=lambda candidate: candidate[1], reverse=True)
    return [tokenizer.token_item[prefix] for prefix, _ in completed]
