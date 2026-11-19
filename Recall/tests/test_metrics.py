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


class PopularityGroupTests(unittest.TestCase):
    def test_groups_split_requests_without_changing_overall_metrics(self):
        from Recall.core.popularity import popularity_groups, target_counts

        requests = [(["A", "B"], "A"), (["B", "C"], "C"), (["D", "A"], "E"), (["B", "A"], "B")]
        counts = target_counts(["A"] * 7 + ["C"] * 2)
        groups = popularity_groups([target for _, target in requests], counts)
        self.assertEqual(groups, ["5+", "1-4", "0", "0"])
        grouped = ranking_metrics(requests, ks=(1, 2), groups=groups)
        plain = ranking_metrics(requests, ks=(1, 2))
        self.assertEqual({key: grouped[key] for key in plain}, plain)
        self.assertEqual(list(grouped["groups"]), ["0", "1-4", "5+"])
        self.assertEqual(grouped["groups"]["0"], {"recall@1": 0.5, "recall@2": 0.5, "ndcg@1": 0.5,
                                                  "ndcg@2": 0.5, "examples": 2})
        self.assertEqual(grouped["groups"]["5+"]["recall@1"], 1.0)
        self.assertEqual(grouped["groups"]["1-4"]["ndcg@2"], 1 / 1.584962500721156)

    def test_groups_must_match_requests(self):
        with self.assertRaises(ValueError):
            ranking_metrics([(["A"], "A")], groups=["0", "5+"])
