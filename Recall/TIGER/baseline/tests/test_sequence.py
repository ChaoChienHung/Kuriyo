"""Focused tests for the optional PyTorch TIGER pipeline."""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("numpy"),
                     "TIGER modeling dependencies are not installed")
class TIGERSequenceTests(unittest.TestCase):
    def test_sequence_decoding_returns_catalog_items(self):
        import torch
        from Recall.TIGER.baseline.sequence import SIDTokenizer, TIGERTransformer, constrained_beam_search

        torch.set_num_threads(1)
        tokenizer = SIDTokenizer({"A": [0, 0, 0, 0], "B": [0, 0, 0, 1]}, 2)
        model = TIGERTransformer(tokenizer.vocab_size, max_history=2,
                                 d_model=16, heads=4, layers=1,
                                 feedforward_dim=32, dropout=0)
        model.eval()
        output = constrained_beam_search(model, tokenizer, ["A"], 2, 2, "cpu")
        self.assertEqual(set(output), {"A", "B"})

    def test_sequence_loader_reads_materialized_sids(self):
        from Recall.TIGER.baseline.sequence import SIDTokenizer
        from Recall.TIGER.baseline.train_sequence import collate, read_examples

        tokenizer = SIDTokenizer({"A": [0, 0, 0, 0], "B": [0, 0, 0, 1]}, 2)
        record = {"user_id": "U", "history_sids": [[0, 0, 0, 0]],
                  "target_sid": [0, 0, 0, 1], "target_asin": "B",
                  "target_timestamp": 123}
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "examples.jsonl"
            path.write_text(json.dumps(record) + "\n")
            rows = read_examples(path, None, tokenizer)
            self.assertEqual(rows[0]["history"], ["A"])
            source, decoder_input, labels = collate(rows, tokenizer, 20)
            self.assertEqual(tuple(source.shape), (1, 4))
            self.assertEqual(tuple(labels.shape), (1, 5))
            self.assertEqual(decoder_input[0, 0].item(), tokenizer.BOS)

