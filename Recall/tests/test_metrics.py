"""Shared next-item Recall metrics."""

import unittest

from Recall.core.metrics import ranking_metrics


class RecallMetricsTests(unittest.TestCase):
    def test_single_relevant_item(self):
        metrics = ranking_metrics([(["A", "B", "C"], "B"), (["C", "B"], "A")], ks=(1, 3))
        self.assertEqual(metrics["recall@1"], 0)
        self.assertEqual(metrics["recall@3"], 0.5)
        self.assertAlmostEqual(metrics["ndcg@3"], 0.5 / 1.584962500721156)

    def test_empty_requests_rejected(self):
        with self.assertRaises(ValueError):
            ranking_metrics([])

    def test_invalid_beam_keeps_original_rank(self):
        metrics = ranking_metrics([([None, "A"], "A")], ks=(1, 2))
        self.assertEqual(metrics["recall@1"], 0)
        self.assertEqual(metrics["recall@2"], 1)
        self.assertAlmostEqual(metrics["ndcg@2"], 1 / 1.584962500721156)
