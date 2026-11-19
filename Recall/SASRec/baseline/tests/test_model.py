"""Behavioural contracts of the SASRec baseline model and data helpers."""

import importlib.util
import unittest

TORCH = importlib.util.find_spec("torch") is not None
if TORCH:
    import torch

    from Recall.SASRec.baseline.data import (collate_histories, collate_windows, pad_sequences,
                                             sample_negatives, user_sequences, windows)
    from Recall.SASRec.baseline.model import SASRec, attention_allowed


def small_model(**overrides):
    settings = dict(num_items=12, max_history=6, hidden=16, heads=2, layers=2, dropout=0.0)
    settings.update(overrides)
    torch.manual_seed(0)
    return SASRec(**settings).eval()


def prefixes(user, items, max_history):
    return [{"user_id": user, "history": items[:k][-max_history:], "target": items[k]}
            for k in range(1, len(items))]


@unittest.skipUnless(TORCH, "torch is not installed")
class SASRecModelTests(unittest.TestCase):
    def test_causal_mask_hides_later_items(self):
        model = small_model()
        first = model(torch.tensor([[0, 0, 3, 4, 5, 6]]))
        changed = model(torch.tensor([[0, 0, 3, 4, 5, 9]]))
        torch.testing.assert_close(first[:, 2:5], changed[:, 2:5])
        self.assertFalse(torch.allclose(first[:, 5], changed[:, 5]))

    def test_real_positions_never_attend_padding(self):
        allowed = attention_allowed(torch.tensor([[0, 0, 4, 5]]))[0]
        self.assertFalse(allowed[2:, :2].any())
        self.assertTrue(allowed[3, 2:].all())
        self.assertTrue(allowed.any(dim=-1).all(), "every query needs one key, or attention is NaN")

    def test_padding_positions_do_not_leak_into_real_outputs(self):
        model = small_model()
        sequence = collate_histories([{"history": [2, 7, 4], "target": 1}], 6)[0]
        before = model(sequence)
        with torch.no_grad():
            model.position_embedding.weight[:3] += 5.0
        after = model(sequence)
        torch.testing.assert_close(before[:, 3:], after[:, 3:])
        self.assertTrue(torch.isfinite(after).all())

    def test_losses_are_finite_and_reach_every_parameter(self):
        for kind in ("ce", "bce"):
            model = small_model().train()
            inputs, targets = collate_windows([[1, 2, 3, 4], [5, 6]], 6)
            negatives = sample_negatives(targets, pad_sequences([[1, 2, 3, 4], [5, 6]]), 12,
                                         torch.Generator().manual_seed(0))
            loss = model.loss(inputs, targets, kind, negatives)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, "{} gets no gradient under {}".format(name, kind))

    def test_padding_item_is_never_ranked(self):
        model = small_model()
        scores = model.scores(model(torch.tensor([[0, 0, 0, 1, 2, 3]]))[:, -1])
        self.assertEqual(scores.shape, (1, 12))


@unittest.skipUnless(TORCH, "torch is not installed")
class SASRecDataTests(unittest.TestCase):
    def test_user_sequences_rebuild_the_prefix_file(self):
        rows = prefixes("A", [1, 2, 3, 4, 5, 6, 7], 3) + prefixes("B", [8, 9, 10], 3)
        self.assertEqual(user_sequences(rows, 3), [[1, 2, 3, 4, 5, 6, 7], [8, 9, 10]])

    def test_user_sequences_reject_reordered_rows(self):
        rows = prefixes("A", [1, 2, 3, 4], 3)
        with self.assertRaises(ValueError):
            user_sequences([rows[0], rows[2], rows[1]], 3)
        split = prefixes("A", [1, 2, 3], 3)
        with self.assertRaises(ValueError):
            user_sequences([split[0], *prefixes("B", [4, 5], 3), split[1]], 3)

    def test_windows_train_each_target_once_with_capped_context(self):
        sequence = list(range(1, 11))
        parts = windows(sequence, 3)
        self.assertTrue(all(len(part) <= 4 for part in parts))
        self.assertEqual([item for part in parts for item in part[1:]], sequence[1:])
        self.assertEqual(windows([5], 3), [])

    def test_collate_left_aligns_newest_item_and_ignores_padding_targets(self):
        inputs, targets = collate_windows([[1, 2, 3]], 4)
        self.assertEqual(inputs.tolist(), [[0, 0, 1, 2]])
        self.assertEqual(targets.tolist(), [[0, 0, 2, 3]])
        history, target = collate_histories([{"history": [1, 2, 3, 4, 5], "target": 6}], 4)
        self.assertEqual((history.tolist(), target.tolist()), ([[2, 3, 4, 5]], [6]))

    def test_negatives_avoid_every_item_the_user_has_seen(self):
        seen = pad_sequences([[1, 2, 3, 4, 5, 6, 7, 8, 9], [10]])
        targets = torch.ones(2, 50, dtype=torch.long)
        negatives = sample_negatives(targets, seen, 10, torch.Generator().manual_seed(0))
        self.assertTrue(negatives[0].eq(10).all(), "only item 10 is unseen by user 0")
        self.assertFalse(negatives[1].eq(10).any())
        self.assertTrue(((negatives >= 1) & (negatives <= 10)).all())
        self.assertEqual(len(negatives[1].unique()), 9, "every unseen item can be drawn")
        with self.assertRaises(ValueError):
            sample_negatives(targets[:1], pad_sequences([list(range(1, 11))]), 10, torch.Generator())

    def test_paper_defaults(self):
        model = SASRec(num_items=12)
        self.assertEqual((model.heads, model.item_embedding.embedding_dim), (1, 50))


if __name__ == "__main__":
    unittest.main()
