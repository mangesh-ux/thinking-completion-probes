"""Synthetic checks; these tests are not new scientific evidence."""

import unittest

import numpy as np

from thinking_completion import metrics


class MetricsTests(unittest.TestCase):
    def test_weights_are_mean_one_and_equal_total_per_problem(self):
        groups = np.array(["a", "a", "b", "c", "c", "c"])
        w = metrics.weights(groups)
        self.assertAlmostEqual(float(w.mean()), 1.0)
        for group in set(groups):
            self.assertAlmostEqual(float(w[groups == group].sum()), 2.0)

    def test_logloss_is_mean_of_problem_means_not_rows(self):
        y = np.array([0, 1, 0])
        p = np.array([0.1, 0.9, 0.8])
        groups = np.array(["a", "a", "b"])
        expected = (-np.log(0.9) - np.log(0.2)) / 2
        self.assertAlmostEqual(metrics.score(y, p, groups), expected)
        self.assertNotAlmostEqual(metrics.score(y, p, groups), float(metrics.binary_losses(y, p).mean()))

    def test_invalid_predictions_fail_closed(self):
        for p in (np.array([np.nan, 0.5]), np.array([np.inf, 0.5]), np.array([-0.1, 0.5]), np.array([1.1, 0.5]), np.array([[0.5, 0.5]])):
            with self.subTest(p=p), self.assertRaises(ValueError):
                metrics.binary_losses(np.array([0, 1]), p)

    def test_boundary_predictions_are_clipped_only_for_logloss(self):
        loss = metrics.binary_losses(np.array([0, 1]), np.array([0.0, 1.0]))
        self.assertTrue(np.isfinite(loss).all())
        self.assertLess(float(loss.sum()), 1e-10)

    def test_bootstrap_retains_repeated_problem_draws(self):
        values = np.array([-5.0, 2.0, 9.0, 20.0, 31.0])
        indices = np.random.default_rng(20261008).integers(0, 5, size=(2000, 5))
        expected = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
        np.testing.assert_array_equal(metrics.bootstrap_interval(values), expected)
        deduplicated = np.array([values[np.unique(row)].mean() for row in indices])
        self.assertFalse(np.array_equal(expected, np.quantile(deduplicated, [0.025, 0.975])))

    def test_class_invalid_paired_draws_suppress_interval(self):
        positive = np.array([True, False])
        negative = np.array([False, True])
        result = metrics.paired_bootstrap(np.array([0.1, 0.2]), positive, negative)
        self.assertLess(result["valid_resamples"], 1900)
        self.assertGreater(result["invalid_fraction"], 0.4)
        self.assertIsNone(result["problem_cluster_bootstrap_95_percentile_ci"])

    def test_matched_concordance_half_ties_and_problem_weighting(self):
        data = {"labels": np.array([1, 0, 1, 0, 0, 1]), "groups": np.array(["a", "a", "b", "b", "b", "c"])}
        groups, values = metrics.matched(data, np.array([0.5, 0.5, 0.8, 0.2, 0.9, 0.7]))
        self.assertEqual(groups, ["a", "b"])
        np.testing.assert_array_equal(values, [0.5, 0.5])

    def test_evaluation_preserves_historical_shape_and_pairing(self):
        data = {"labels": np.tile(np.array([0, 1], dtype=np.int8), 8), "groups": np.repeat(np.array([f"g{g:02d}" for g in range(8)]), 2)}
        predictions = {name: np.full(16, 0.5) for name in metrics.MODELS}
        predictions["Clex"] = np.tile([0.25, 0.75], 8)
        result = metrics.evaluate_predictions(data, predictions, 0.5)
        self.assertEqual(set(result["metrics"]), set(metrics.MODELS) | {"TRAIN_prevalence"})
        self.assertEqual(set(result["paired_comparisons"]), {"Clex_vs_Blex", "C_vs_B", "Blex_vs_B"})
        self.assertEqual(result["rows"], 16)
        self.assertEqual(result["problems"], 8)
        self.assertAlmostEqual(result["paired_comparisons"]["Clex_vs_Blex"]["mean_problem_log_loss_improvement_nats"], np.log(0.75 / 0.5))
        self.assertEqual(result["matched_control"]["paired_concordance_difference"], 0.5)
        self.assertTrue(result["matched_control"]["adequate_support"])
        self.assertEqual(result["matched_control"]["difference_bootstrap_95_percentile_ci"], [0.5, 0.5])

    def test_insufficient_matched_groups_cannot_claim_within_problem_support(self):
        data = {"labels": np.array([0, 1, 0]), "groups": np.array(["a", "a", "b"])}
        result = metrics.evaluate_predictions(data, {name: np.array([0.2, 0.8, 0.3]) for name in metrics.MODELS}, 0.5)
        self.assertFalse(result["matched_control"]["adequate_support"])
        self.assertIsNone(result["matched_control"]["difference_bootstrap_95_percentile_ci"])

    def test_all_models_required_and_extra_predictions_rejected(self):
        data = {"labels": np.array([0, 1]), "groups": np.array(["a", "a"])}
        predictions = {name: np.array([0.2, 0.8]) for name in metrics.MODELS}
        for changed in ({"B": predictions["B"]}, {**predictions, "future": predictions["B"]}):
            with self.assertRaises(ValueError):
                metrics.evaluate_predictions(data, changed, 0.5)


if __name__ == "__main__":
    unittest.main()
