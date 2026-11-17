"""Contracts for reusable Recall components."""

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("numpy"),
                     "Recall modeling dependencies are not installed")
class RecallCoreTests(unittest.TestCase):
    def test_encoder_decoder_shapes_and_gradients(self):
        import torch
        from Recall.core.transformer import EncoderDecoderTransformer

        torch.set_num_threads(1)
        model = EncoderDecoderTransformer(20, max_source_length=6,
                                          max_target_length=3, d_model=16,
                                          heads=4, layers=1, feedforward_dim=32,
                                          dropout=0)
        source = torch.tensor([[1, 2, 0], [3, 4, 5]])
        target = torch.tensor([[1, 6], [1, 7]])
        logits = model(source, target)
        self.assertEqual(tuple(logits.shape), (2, 2, 20))
        logits.square().mean().backward()
        self.assertIsNotNone(model.token_embedding.weight.grad)

    def test_padding_and_provenance(self):
        from Recall.core.data import pad_token_sequences
        from Recall.core.experiment import file_sha256, write_json

        padded = pad_token_sequences([[1, 2], [3]])
        self.assertEqual(padded.tolist(), [[1, 2], [3, 0]])
        with self.assertRaises(ValueError):
            pad_token_sequences([[]])
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.json"
            write_json(path, {"seed": 42})
            self.assertIn('"seed": 42', path.read_text())
            self.assertEqual(len(file_sha256(path)), 64)

    def test_decoder_cannot_see_future_tokens(self):
        import torch
        from Recall.core.transformer import EncoderDecoderTransformer
        torch.manual_seed(7)
        model = EncoderDecoderTransformer(20, 6, 3, d_model=16, heads=4,
                                          layers=1, feedforward_dim=32, dropout=0).eval()
        source = torch.tensor([[3, 4]])
        a = model(source, torch.tensor([[1, 6, 7]]))
        b = model(source, torch.tensor([[1, 6, 9]]))
        torch.testing.assert_close(a[:, :2], b[:, :2])

    def test_source_padding_does_not_change_predictions(self):
        import torch
        from Recall.core.transformer import EncoderDecoderTransformer
        torch.manual_seed(7)
        model = EncoderDecoderTransformer(20, 6, 3, d_model=16, heads=4,
                                          layers=1, feedforward_dim=32, dropout=0).eval()
        target = torch.tensor([[1, 6]])
        a = model(torch.tensor([[3, 4]]), target)
        b = model(torch.tensor([[3, 4, 0, 0]]), target)
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
