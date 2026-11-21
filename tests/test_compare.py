"""Run comparison statistics (core/stats.py, core/compare.py)."""

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from core.compare import compare, main, metric_values
from core.stats import holm, paired_bootstrap, paired_t, summary, t_quantile, t_two_sided_p, welch_t


class StatsTests(unittest.TestCase):
    def test_t_distribution_matches_tables(self):
        self.assertAlmostEqual(t_quantile(0.975, 2), 4.3027, places=4)
        self.assertAlmostEqual(t_quantile(0.975, 4), 2.7764, places=4)
        self.assertAlmostEqual(t_two_sided_p(2.2281, 10), 0.05, places=4)
        self.assertAlmostEqual(t_two_sided_p(0.0, 5), 1.0)

    def test_summary_uses_sample_sd_and_t_interval(self):
        result = summary([1.0, 2.0, 3.0])
        self.assertEqual(result["sd"], 1.0)
        self.assertAlmostEqual(result["ci"][1] - 2.0, 4.3027 / math.sqrt(3), places=4)
        self.assertIsNone(summary([0.5])["sd"])

    def test_welch_and_paired_t(self):
        self.assertIsNone(welch_t([1.0], [1.0, 2.0]))
        # Welch's t for equal sizes and variances reduces to Student's t with df = 2n - 2.
        result = welch_t([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
        self.assertAlmostEqual(result["t"], -1 / math.sqrt(2 / 3))
        self.assertAlmostEqual(result["df"], 4.0)
        paired = paired_t([1.0, 2.0, 3.0])
        self.assertAlmostEqual(paired["t"], 2.0 / (1.0 / math.sqrt(3)))
        self.assertEqual(paired["df"], 2)

    def test_bootstrap_detects_shift_and_not_noise(self):
        rng = np.random.default_rng(1)
        shifted = paired_bootstrap(rng.normal(0.5, 1.0, 400), resamples=2000)
        self.assertLess(shifted["p"], 0.001)
        self.assertLess(shifted["ci"][0], shifted["mean"])
        self.assertGreater(shifted["ci"][0], 0)
        null = paired_bootstrap(np.tile([-1.0, 1.0], 200), resamples=2000)
        self.assertGreater(null["p"], 0.5)
        self.assertEqual(paired_bootstrap([1.0, 2.0, 0.0], resamples=50, seed=3),
                         paired_bootstrap([1.0, 2.0, 0.0], resamples=50, seed=3))

    def test_holm(self):
        self.assertEqual(holm([0.01, 0.04, 0.03, None]), [0.03, 0.06, 0.06, None])


def write_run(root, name, seed, ranks, users=None):
    run = root / name
    validation = run / "validation/amazon2014-beauty"
    validation.mkdir(parents=True)
    (run / "settings.json").write_text(json.dumps({"run": {"seed": seed}}))
    users = users or ["U{}".format(i) for i in range(len(ranks))]
    (validation / "best-ranks.json").write_text(json.dumps({"max_rank": 10, "users": users, "ranks": ranks}))
    (validation / "best.json").write_text(json.dumps({"recall@10": sum(0 < r <= 10 for r in ranks) / len(ranks)}))
    return run


class CompareTests(unittest.TestCase):
    def test_metric_values_from_ranks(self):
        np.testing.assert_allclose(metric_values([1, 3, 0, 11], "ndcg@10"), [1.0, 0.5, 0.0, 0.0])
        np.testing.assert_allclose(metric_values([1, 6, 0], "recall@5"), [1.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            metric_values([1], "mrr@10")

    def test_three_seeds_significant_and_single_seed_no_claim(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(0)
            weak = [write_run(root, "weak-{}".format(s), s, [int(r) for r in rng.choice([0, 9], 3000)])
                    for s in (42, 43, 44)]
            strong = [write_run(root, "strong-{}".format(s), s, [int(r) for r in rng.choice([1, 2, 0], 3000)])
                      for s in (42, 43, 44)]
            report = compare([("weak", weak), ("strong", strong)], metric="recall@10", resamples=500)
            self.assertEqual(report["users"], 3000)
            self.assertEqual(report["settings"][0]["seeds"], [42, 43, 44])
            comparison = report["comparisons"][0]
            self.assertGreater(comparison["difference"], 0)
            self.assertEqual(comparison["verdict"], "significantly better")
            self.assertEqual(report["warnings"], [])
            single = compare([("weak", weak[:1]), ("strong", strong[:1])], metric="recall@10", resamples=100)
            self.assertEqual(single["comparisons"][0]["verdict"], "single seed: no claim")

    def test_large_gap_without_significance_is_no_difference(self):
        from core.compare import verdict
        base, other = {"sd": 0.001}, {"sd": 0.001}
        comparison = {"difference": 0.002, "welch": {"p_holm": 0.11},
                      "paired_t": {"p_holm": 0.2}, "paired_bootstrap": {"p_holm": 0.2}}
        self.assertEqual(verdict(comparison, base, other), "no difference shown")
        comparison["welch"]["p_holm"] = 0.01
        self.assertTrue(verdict(comparison, base, other).startswith("tendency better"))

    def test_refuses_different_users(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            a = write_run(root, "a", 42, [1, 0, 2])
            b = write_run(root, "b", 42, [1, 0, 2], users=["X", "Y", "Z"])
            with self.assertRaises(ValueError):
                compare([("a", [a]), ("b", [b])])
            with self.assertRaises(SystemExit):
                main(["--setting", "a", str(a), "--setting", "b", str(root / "missing")])


if __name__ == "__main__":
    unittest.main()
