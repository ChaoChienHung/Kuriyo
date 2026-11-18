"""Behavioural contracts of the FMRec baseline model and data helpers."""

import importlib.util
import unittest

TORCH = importlib.util.find_spec("torch") is not None
if TORCH:
    import torch

    from Recall.FMRec.baseline.data import collate
    from Recall.FMRec.baseline.model import FMRec, attention_allowed


def small_model(**overrides):
    settings = dict(num_items=12, hidden=16, heads=2, decoder1_layers=1, decoder2_layers=1,
                    dropout=0.0, embedding_dropout=0.0, reconstruction_dims=(8,),
                    sample_steps=4)
    settings.update(overrides)
    torch.manual_seed(0)
    return FMRec(**settings).eval()


@unittest.skipUnless(TORCH, "torch is not installed")
class FMRecModelTests(unittest.TestCase):
    def test_causal_mode_hides_later_positions(self):
        model = small_model(attention="causal", lambda_uncertainty=0.0)
        sequence = torch.tensor([[3, 4, 5, 6]])
        changed = torch.tensor([[3, 4, 5, 9]])
        x_t, t = torch.zeros(1, 16), torch.full((1,), 0.5)
        hidden = []
        for seq in (sequence, changed):
            history = model.encode_history(seq)
            allowed = attention_allowed(seq, "causal")
            state = history
            for block in model.decoder1:
                state = block(state, allowed)
            hidden.append(state)
        torch.testing.assert_close(hidden[0][:, :3], hidden[1][:, :3])
        self.assertFalse(torch.allclose(hidden[0][:, 3], hidden[1][:, 3]))

    def test_bidirectional_mode_never_attends_padding_keys(self):
        allowed = attention_allowed(torch.tensor([[0, 0, 4, 5]]), "bidirectional")[0, 0]
        self.assertFalse(allowed[2:, :2].any())
        self.assertTrue(allowed[2:, 2:].all())

    def test_left_padding_does_not_change_prediction(self):
        # The per-position uncertainty draw depends on the padded width, so fix it
        # at its mean; the start noise z_0 stays random but seeded.
        for attention in ("bidirectional", "causal"):
            model = small_model(attention=attention)
            original = model._noise
            model._noise = lambda shape, generator, device: (
                torch.zeros(shape, device=device) if len(shape) == 3 else original(shape, generator, device))
            short = torch.tensor([[2, 7, 4]])
            padded = torch.tensor([[0, 0, 0, 2, 7, 4]])
            outputs = [model.sample(seq, generator=torch.Generator().manual_seed(1)) for seq in (short, padded)]
            torch.testing.assert_close(outputs[0], outputs[1])

    def test_collate_left_pads_and_caps_history(self):
        sequence, target = collate([{"history": [1, 2, 3, 4], "target": 5},
                                    {"history": [6], "target": 7}], max_history=3)
        self.assertEqual(sequence.tolist(), [[2, 3, 4], [0, 0, 6]])
        self.assertEqual(target.tolist(), [5, 7])

    def test_losses_finite_and_gradients_reach_every_part(self):
        model = small_model().train()
        sequence = torch.tensor([[0, 1, 2], [3, 4, 5]])
        losses = model.loss(sequence, torch.tensor([6, 7]), alpha=0.2, beta=0.4)
        for name, value in losses.items():
            self.assertTrue(torch.isfinite(value), name)
        losses["loss"].backward()
        for name in ("item_embedding.weight", "time_mlp.0.weight", "decoder1.0.attention.qkv.weight",
                     "decoder2.0.feedforward.0.weight", "reconstruction.0.weight"):
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(gradient.abs().sum().item(), 0, name)

    def test_padding_item_never_ranked(self):
        model = small_model()
        scores = model.scores(model.sample(torch.tensor([[1, 2]]), generator=torch.Generator().manual_seed(0)))
        self.assertEqual(scores.shape, (1, 13))
        self.assertEqual(scores[0, 0].item(), float("-inf"))
        self.assertNotIn(0, scores.topk(12).indices.tolist()[0])

    def test_sampler_shape_and_seeded_determinism(self):
        model = small_model()
        sequence = torch.tensor([[0, 1, 2], [3, 4, 5]])
        first = model.sample(sequence, generator=torch.Generator().manual_seed(3))
        second = model.sample(sequence, generator=torch.Generator().manual_seed(3))
        self.assertEqual(first.shape, (2, 16))
        torch.testing.assert_close(first, second)

    def test_mode_time_schedule_stays_in_unit_interval(self):
        for scale in (1.0, 0.4, -0.5):
            t = small_model(mode_scale=scale).sample_time(4096, "cpu")
            self.assertGreaterEqual(t.min().item(), 0.001 - 1e-6)
            self.assertLessEqual(t.max().item(), 1.0 + 1e-6)

    def test_balanced_mse_targets_history_set(self):
        model = small_model()
        target = model.history_targets(torch.tensor([[0, 2, 2, 5]]))
        self.assertEqual(target[0].nonzero().flatten().tolist(), [2, 5])


if __name__ == "__main__":
    unittest.main()
