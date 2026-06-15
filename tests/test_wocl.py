from __future__ import annotations

import unittest

import numpy as np

from ocl_python.algorithm import OCL, _guard_attribute_weights, _normalized_weight_entropy
from ocl_python.cli import _parse_methods, _parse_weight_min


class WOCLTests(unittest.TestCase):
    def test_parse_methods_accepts_wocl(self) -> None:
        self.assertEqual(_parse_methods("wocl"), ["wocl"])
        self.assertIn("wocl", _parse_methods("all"))

    def test_parse_weight_min_auto(self) -> None:
        self.assertIsNone(_parse_weight_min("auto"))
        self.assertEqual(_parse_weight_min("0.02"), 0.02)

    def test_alpha_zero_keeps_uniform_weights(self) -> None:
        features = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [1, 1, 1],
                [1, 0, 1],
                [2, 1, 1],
                [2, 0, 1],
            ],
            dtype=np.int64,
        )

        result = OCL(
            variant="wocl",
            seed=7,
            max_outer_loops=5,
            weight_alpha=0.0,
        ).fit_predict(features, n_clusters=2)

        self.assertIsNotNone(result.attribute_weights)
        expected = np.full(features.shape[1], 1.0 / features.shape[1])
        np.testing.assert_allclose(result.attribute_weights, expected, atol=1e-12)

    def test_alpha_zero_matches_full_ocl(self) -> None:
        features = np.array(
            [
                [0, 0, 1],
                [0, 1, 1],
                [1, 1, 0],
                [1, 0, 0],
                [2, 1, 0],
                [2, 0, 0],
                [3, 1, 1],
                [3, 0, 1],
            ],
            dtype=np.int64,
        )

        full = OCL(seed=13, max_outer_loops=5).fit_predict(features, n_clusters=2)
        wocl = OCL(
            variant="wocl",
            seed=13,
            max_outer_loops=5,
            weight_alpha=0.0,
        ).fit_predict(features, n_clusters=2)

        np.testing.assert_array_equal(wocl.assignments, full.assignments)
        np.testing.assert_allclose(wocl.objective_history, full.objective_history)
        self.assertEqual(wocl.learned_orders, full.learned_orders)

    def test_weight_mix_zero_matches_full_ocl_assignments(self) -> None:
        features = np.array(
            [
                [0, 0, 1],
                [0, 1, 1],
                [1, 1, 0],
                [1, 0, 0],
                [2, 1, 0],
                [2, 0, 0],
                [3, 1, 1],
                [3, 0, 1],
            ],
            dtype=np.int64,
        )

        full = OCL(seed=17, max_outer_loops=5).fit_predict(features, n_clusters=2)
        wocl = OCL(
            variant="wocl",
            seed=17,
            max_outer_loops=5,
            weight_mix=0.0,
        ).fit_predict(features, n_clusters=2)

        np.testing.assert_array_equal(wocl.assignments, full.assignments)
        np.testing.assert_allclose(wocl.objective_history, full.objective_history)

    def test_weight_delay_postpones_updates(self) -> None:
        features = np.array(
            [
                [0, 0, 0, 1],
                [0, 1, 0, 1],
                [1, 0, 1, 1],
                [1, 1, 1, 1],
                [2, 0, 1, 0],
                [2, 1, 1, 0],
                [3, 0, 0, 0],
                [3, 1, 0, 0],
            ],
            dtype=np.int64,
        )

        result = OCL(
            variant="wocl",
            seed=19,
            max_outer_loops=1,
            weight_delay=2,
        ).fit_predict(features, n_clusters=2)

        self.assertEqual(len(result.weight_history), 1)

    def test_entropy_guard_rejects_concentrated_weights(self) -> None:
        previous = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
        candidate = np.array([0.97, 0.01, 0.01, 0.01], dtype=np.float64)

        guarded = _guard_attribute_weights(
            previous,
            candidate,
            weight_guard="entropy",
            entropy_min=0.7,
        )

        np.testing.assert_allclose(guarded, previous)
        self.assertLess(_normalized_weight_entropy(candidate), 0.7)

    def test_no_guard_accepts_concentrated_weights(self) -> None:
        previous = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
        candidate = np.array([0.97, 0.01, 0.01, 0.01], dtype=np.float64)

        guarded = _guard_attribute_weights(
            previous,
            candidate,
            weight_guard="none",
            entropy_min=0.7,
        )

        np.testing.assert_allclose(guarded, candidate)

    def test_objective_guard_accepts_non_worse_weights(self) -> None:
        X = np.array([[0, 0], [1, 1]], dtype=np.int64)
        mode_probs = [
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
            np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64),
        ]
        distance_matrices = [
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
        ]
        assignments = np.array([0, 1], dtype=np.int64)
        previous = np.array([0.5, 0.5], dtype=np.float64)
        candidate = np.array([0.9, 0.1], dtype=np.float64)

        guarded = _guard_attribute_weights(
            previous,
            candidate,
            weight_guard="objective",
            entropy_min=0.0,
            X=X,
            mode_probs=mode_probs,
            distance_matrices=distance_matrices,
            assignments=assignments,
        )

        np.testing.assert_allclose(guarded, candidate)

    def test_objective_guard_rejects_worse_weights(self) -> None:
        X = np.array([[0, 0], [1, 1]], dtype=np.int64)
        mode_probs = [
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
            np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64),
        ]
        distance_matrices = [
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
        ]
        assignments = np.array([0, 1], dtype=np.int64)
        previous = np.array([0.5, 0.5], dtype=np.float64)
        candidate = np.array([0.1, 0.9], dtype=np.float64)

        guarded = _guard_attribute_weights(
            previous,
            candidate,
            weight_guard="objective",
            entropy_min=0.0,
            X=X,
            mode_probs=mode_probs,
            distance_matrices=distance_matrices,
            assignments=assignments,
        )

        np.testing.assert_allclose(guarded, previous)

    def test_objective_entropy_guard_rejects_low_entropy_candidate(self) -> None:
        X = np.array([[0, 0], [1, 1]], dtype=np.int64)
        mode_probs = [
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
            np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64),
        ]
        distance_matrices = [
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
        ]
        assignments = np.array([0, 1], dtype=np.int64)
        previous = np.array([0.5, 0.5], dtype=np.float64)
        candidate = np.array([0.999, 0.001], dtype=np.float64)

        guarded = _guard_attribute_weights(
            previous,
            candidate,
            weight_guard="objective_entropy",
            entropy_min=0.5,
            X=X,
            mode_probs=mode_probs,
            distance_matrices=distance_matrices,
            assignments=assignments,
        )

        np.testing.assert_allclose(guarded, previous)

    def test_objective_entropy_guard_accepts_non_worse_high_entropy_candidate(self) -> None:
        X = np.array([[0, 0], [1, 1]], dtype=np.int64)
        mode_probs = [
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
            np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64),
        ]
        distance_matrices = [
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
        ]
        assignments = np.array([0, 1], dtype=np.int64)
        previous = np.array([0.5, 0.5], dtype=np.float64)
        candidate = np.array([0.6, 0.4], dtype=np.float64)

        guarded = _guard_attribute_weights(
            previous,
            candidate,
            weight_guard="objective_entropy",
            entropy_min=0.9,
            X=X,
            mode_probs=mode_probs,
            distance_matrices=distance_matrices,
            assignments=assignments,
        )

        np.testing.assert_allclose(guarded, candidate)

    def test_weight_updates_stay_normalized_and_finite(self) -> None:
        features = np.array(
            [
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 1, 0, 1],
                [2, 1, 0, 0],
                [2, 1, 0, 0],
                [3, 1, 0, 0],
                [3, 0, 0, 0],
            ],
            dtype=np.int64,
        )
        weight_min = 0.01 / features.shape[1]

        result = OCL(
            variant="wocl",
            seed=3,
            max_outer_loops=10,
            weight_min=weight_min,
        ).fit_predict(features, n_clusters=2)

        self.assertGreaterEqual(len(result.weight_history), 1)
        for weights in result.weight_history:
            arr = np.asarray(weights, dtype=np.float64)
            self.assertTrue(np.all(np.isfinite(arr)))
            self.assertAlmostEqual(float(arr.sum()), 1.0, places=12)
            self.assertTrue(np.all(arr >= weight_min - 1e-12))

    def test_constant_attributes_do_not_create_nan_weights(self) -> None:
        features = np.array(
            [
                [0, 1, 0],
                [0, 1, 0],
                [1, 1, 0],
                [1, 1, 0],
                [2, 1, 0],
                [2, 1, 0],
            ],
            dtype=np.int64,
        )

        result = OCL(
            variant="wocl",
            seed=11,
            max_outer_loops=5,
            weight_gamma=2.0,
        ).fit_predict(features, n_clusters=2)

        self.assertIsNotNone(result.attribute_weights)
        weights = np.asarray(result.attribute_weights, dtype=np.float64)
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
