"""Focused tests for the optional PyTorch TIGER pipeline."""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("numpy"),
                     "TIGER modeling dependencies are not installed")
class TIGERPipelineTests(unittest.TestCase):
    def test_rqvae_forward_and_codebook_gradients(self):
        import torch
        from Recall.TIGER.rqvae import RQVAE

        model = RQVAE(4, hidden_dims=(8,), latent_dim=2, levels=3, codebook_size=4)
        loss, codes, _ = model(torch.randn(6, 4))
        self.assertEqual(tuple(codes.shape), (6, 3))
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(all(codebook.grad is not None for codebook in model.codebooks))

    def test_collision_suffix_makes_ids_unique(self):
        from Recall.TIGER.scripts.generate_sids import unique_semantic_ids

        sids, groups, largest = unique_semantic_ids(
            ["B", "A", "C"], [[1, 2, 3], [1, 2, 3], [4, 5, 6]])
        self.assertEqual(sids["A"], [1, 2, 3, 0])
        self.assertEqual(sids["B"], [1, 2, 3, 1])
        self.assertEqual(sids["C"], [4, 5, 6, 0])
        self.assertEqual((groups, largest), (1, 2))

    def test_sequence_decoding_returns_catalog_items(self):
        import torch
        from Recall.TIGER.sequence import SIDTokenizer, TIGERTransformer, constrained_beam_search

        torch.set_num_threads(1)
        tokenizer = SIDTokenizer({"A": [0, 0, 0, 0], "B": [0, 0, 0, 1]}, 2)
        model = TIGERTransformer(tokenizer.vocab_size, max_history=2,
                                 d_model=16, heads=4, layers=1,
                                 feedforward_dim=32, dropout=0)
        model.eval()
        output = constrained_beam_search(model, tokenizer, ["A"], 2, 2, "cpu")
        self.assertEqual(set(output), {"A", "B"})

    def test_sequence_loader_reads_materialized_sids(self):
        from Recall.TIGER.sequence import SIDTokenizer
        from scripts.train_tiger_sequence import collate, read_examples

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


if __name__ == "__main__":
    unittest.main()
