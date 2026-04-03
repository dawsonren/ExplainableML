"""
Tests for ALE shared utilities, GValues, tree partitioning, and end-to-end ALE.
"""

import numpy as np
import unittest
from ale.shared import (
    calculate_K,
    calculate_edges,
    calculate_bin_index,
    calculate_bins,
    calculate_deltas,
    linear_interpolation,
)
from ale.ale_vim import GValues, calculate_g_values, observation_to_path
from ale.tree_partitioning import generate_connected_kdforest_and_paths
from ale import ALE


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

class TestShared(unittest.TestCase):
    def test_calculate_edges_continuous(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        edges = calculate_edges(x, K=4, categorical=False)
        # should produce 5 edge values (4 bins), all unique
        self.assertEqual(len(edges), 5)
        self.assertAlmostEqual(edges[0], 1.0)
        self.assertAlmostEqual(edges[-1], 5.0)

    def test_calculate_edges_continuous_clamps_K(self):
        # only 3 unique values -> can't have more than 2 bins
        x = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        edges = calculate_edges(x, K=10, categorical=False)
        self.assertLessEqual(len(edges) - 1, 2)

    def test_calculate_edges_categorical(self):
        x = np.array([3, 1, 2, 1, 3])
        edges = calculate_edges(x, K=3, categorical=True)
        np.testing.assert_array_equal(edges, [1, 2, 3])

    def test_calculate_bin_index_continuous(self):
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        K = 3
        # value at left boundary -> bin 0 (0-based)
        self.assertEqual(calculate_bin_index(0.0, edges, K, categorical=False), 0)
        # value at right boundary -> bin K-1 (0-based, clipped)
        self.assertEqual(calculate_bin_index(3.0, edges, K, categorical=False), K - 1)
        # value in the middle
        self.assertEqual(calculate_bin_index(1.5, edges, K, categorical=False), 1)

    def test_calculate_bin_index_categorical(self):
        edges = np.array([1, 2, 3])
        K = 3
        self.assertEqual(calculate_bin_index(1, edges, K, categorical=True), 0)
        self.assertEqual(calculate_bin_index(2, edges, K, categorical=True), 1)
        self.assertEqual(calculate_bin_index(3, edges, K, categorical=True), 2)

    def test_calculate_bins_counts_sum_to_n(self):
        x = np.array([0.5, 1.5, 2.5, 0.8, 1.2, 2.8, 0.3])
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        k_x, N_k = calculate_bins(x, edges, categorical=False)
        self.assertEqual(N_k.sum(), len(x))
        self.assertEqual(len(k_x), len(x))

    def test_linear_interpolation_basic(self):
        # midpoint interpolation
        result = linear_interpolation(
            x=np.array([1.5]),
            x0=np.array([1.0]),
            x1=np.array([2.0]),
            y0=np.array([0.0]),
            y1=np.array([10.0]),
        )
        np.testing.assert_allclose(result, [5.0])

    def test_linear_interpolation_zero_width(self):
        # when x0 == x1, should return y0 (no division error)
        result = linear_interpolation(
            x=np.array([1.0]),
            x0=np.array([1.0]),
            x1=np.array([1.0]),
            y0=np.array([7.0]),
            y1=np.array([10.0]),
        )
        np.testing.assert_allclose(result, [7.0])

    def test_calculate_K_continuous(self):
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        self.assertEqual(calculate_K(edges, categorical=False), 3)

    def test_calculate_K_categorical(self):
        edges = np.array([1, 2, 3])
        self.assertEqual(calculate_K(edges, categorical=True), 3)


# ---------------------------------------------------------------------------
# Deltas and GValues (existing tests, cleaned up)
# ---------------------------------------------------------------------------

class TestDeltas(unittest.TestCase):
    def test_calculate_deltas_categorical(self):
        X = np.array([
            [1, 1],
            [1, 2],
            [1, 3],
            [2, 1],
            [2, 2],
            [2, 3],
            [3, 1],
            [3, 2],
            [3, 3],
        ])
        f = lambda X: X[:, 0] + X[:, 1] ** 2
        K = 3
        # Last category in each feature has no right edge, so delta=0
        true_deltas = np.array([
            [1, 1, 1, 1, 1, 1, 0, 0, 0],
            [3, 5, 0, 3, 5, 0, 3, 5, 0],
        ])
        for idx in range(2):
            edges = calculate_edges(X[:, idx], K, categorical=True)
            k_x = calculate_bin_index(X[:, idx], edges, K, categorical=True)
            deltas = calculate_deltas(f, X, idx, edges, k_x)
            np.testing.assert_array_equal(deltas, true_deltas[idx])

    def test_calculate_deltas_continuous(self):
        X = np.array([
            [1, 1],
            [1.4, 2],
            [1.8, 3],
            [2, 1],
            [2.4, 2],
            [2.8, 3],
            [3, 1],
            [3.4, 2],
            [3.8, 3],
        ])
        f = lambda X: X[:, 0] + X[:, 1] ** 2
        K = 3
        true_deltas = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [3, 5, 7, 3, 5, 7, 3, 5, 7],
        ])
        true_g_values = np.array([
            [1, 2, 3],
            [3, 8, 15]
        ])

        for idx in range(2):
            edges = np.array([1, 2, 3, 4])
            k_x = calculate_bin_index(X[:, idx], edges, K, categorical=False)
            deltas = calculate_deltas(f, X, idx, edges, k_x)
            np.testing.assert_array_equal(deltas, true_deltas[idx])
            g_values, _, _ = calculate_g_values(
                "connected", X, idx + 1, edges, deltas, [False, False], {}, 3, 3, k_x
            )
            for l in range(3):
                np.testing.assert_array_equal(g_values[:, l], true_g_values[idx])


class TestGValues(unittest.TestCase):
    """Test GValues centering and shapes."""

    def setUp(self):
        self.X = np.array([
            [1, 1],
            [1.4, 2],
            [1.8, 3],
            [2, 1],
            [2.4, 2],
            [2.8, 3],
            [3, 1],
            [3.4, 2],
            [3.8, 3],
        ])
        self.edges = np.array([1, 2, 3, 4])
        self.K = 3
        self.L = 3
        # all paths see the same deltas for f(X) = X1 + X2^2
        # g_values are cumulative sums of average deltas per bin
        self.g_values_x0 = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=float)
        self.g_values_x1 = np.array([[3, 3, 3], [8, 8, 8], [15, 15, 15]], dtype=float)
        self.l_xs = np.array([
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0, 0, 0, 1, 1, 1, 2, 2, 2]
        ])

    def test_centered_shape_no_interpolation(self):
        gv = GValues(
            self.g_values_x0, self.X[:, 0], self.edges,
            calculate_bin_index(self.X[:, 0], self.edges, self.K, False),
            self.l_xs[0], "x", False, False
        )
        self.assertEqual(gv.centered_g_values.shape, (self.K, self.L))

    def test_centered_shape_interpolation(self):
        gv = GValues(
            self.g_values_x0, self.X[:, 0], self.edges,
            calculate_bin_index(self.X[:, 0], self.edges, self.K, False),
            self.l_xs[0], "x", True, False
        )
        # interpolated version pads a zero row -> (K+1, L)
        self.assertEqual(gv.centered_g_values.shape, (self.K + 1, self.L))

    def test_no_interpolation_x_centering(self):
        expected = np.array([
            [-1, 0, 1],
            [-5, 0, 7]
        ])
        for idx in range(2):
            k_x = calculate_bin_index(self.X[:, idx], self.edges, self.K, False)
            g_vals = self.g_values_x0 if idx == 0 else self.g_values_x1
            gv = GValues(g_vals, self.X[:, idx], self.edges, k_x, self.l_xs[idx], "x", False, False)
            for l in range(self.L):
                np.testing.assert_allclose(gv.centered_g_values[:, l], expected[idx])

    def test_no_interpolation_y_centering(self):
        expected = np.array([
            [-1, 0, 1],
            [-17 / 3, -2 / 3, 19 / 3]
        ])
        for idx in range(2):
            k_x = calculate_bin_index(self.X[:, idx], self.edges, self.K, False)
            g_vals = self.g_values_x0 if idx == 0 else self.g_values_x1
            gv = GValues(g_vals, self.X[:, idx], self.edges, k_x, self.l_xs[idx], "y", False, False)
            for l in range(self.L):
                np.testing.assert_allclose(gv.centered_g_values[:, l], expected[idx])

    def test_interpolation_x_centering(self):
        expected = np.array([
            [-1.4, -0.4, 0.6, 1.6],
            [-3, 0, 5, 12]
        ])
        for idx in range(2):
            k_x = calculate_bin_index(self.X[:, idx], self.edges, self.K, False)
            g_vals = self.g_values_x0 if idx == 0 else self.g_values_x1
            gv = GValues(g_vals, self.X[:, idx], self.edges, k_x, self.l_xs[idx], "x", True, False)
            for l in range(self.L):
                np.testing.assert_allclose(gv.centered_g_values[:, l], expected[idx])

    def test_interpolation_y_centering(self):
        expected = np.array([
            [
                [-1, 0, 1, 2],
                [-1.4, -0.4, 0.6, 1.6],
                [-1.8, -0.8, 0.2, 1.2]
            ],
            [
                [-11 / 3, -2 / 3, 13 / 3, 34 / 3],
                [-11 / 3, -2 / 3, 13 / 3, 34 / 3],
                [-11 / 3, -2 / 3, 13 / 3, 34 / 3]
            ]
        ])
        for idx in range(2):
            k_x = calculate_bin_index(self.X[:, idx], self.edges, self.K, False)
            g_vals = self.g_values_x0 if idx == 0 else self.g_values_x1
            gv = GValues(g_vals, self.X[:, idx], self.edges, k_x, self.l_xs[idx], "y", True, False)
            for l in range(self.L):
                np.testing.assert_allclose(gv.centered_g_values[:, l], expected[idx, l, :])

    def test_lookup_locals_no_interpolation(self):
        k_x = calculate_bin_index(self.X[:, 0], self.edges, self.K, False)
        gv = GValues(
            self.g_values_x0, self.X[:, 0], self.edges,
            k_x, self.l_xs[0], "x", False, False
        )
        # lookup for bin 1 (0-based), path 0
        result = gv.lookup_locals(np.array([1]), np.array([0]), np.array([2.5]))
        np.testing.assert_allclose(result, gv.centered_g_values[1, 0])

    def test_lookup_locals_interpolation(self):
        k_x = calculate_bin_index(self.X[:, 0], self.edges, self.K, False)
        gv = GValues(
            self.g_values_x0, self.X[:, 0], self.edges,
            k_x, self.l_xs[0], "x", True, False
        )
        # midpoint of bin 1 (0-based, edges 2..3) → average of centered_g_values[1] and [2]
        result = gv.lookup_locals(np.array([1]), np.array([0]), np.array([2.5]))
        expected = 0.5 * (gv.centered_g_values[1, 0] + gv.centered_g_values[2, 0])
        np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# Tree partitioning
# ---------------------------------------------------------------------------

class TestTreePartitioning(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.n = 100
        self.X = rng.standard_normal((self.n, 3))
        self.f = lambda X: X[:, 0] + X[:, 1]
        self.K = 10
        self.L = 10
        self.categorical = [False, False, False]

    def test_forest_produces_L_paths(self):
        edges = calculate_edges(self.X[:, 0], self.K, categorical=False)
        k_x, _ = calculate_bins(self.X[:, 0], edges, categorical=False)
        deltas = calculate_deltas(self.f, self.X, 0, edges, k_x)
        _, paths = generate_connected_kdforest_and_paths(
            self.X, 1, edges, deltas, self.categorical, {}, self.L
        )
        self.assertEqual(len(paths), self.L)

    def test_all_observations_assigned_to_paths(self):
        edges = calculate_edges(self.X[:, 0], self.K, categorical=False)
        k_x, _ = calculate_bins(self.X[:, 0], edges, categorical=False)
        deltas = calculate_deltas(self.f, self.X, 0, edges, k_x)
        _, paths = generate_connected_kdforest_and_paths(
            self.X, 1, edges, deltas, self.categorical, {}, self.L
        )
        # collect all indices across all paths and intervals
        all_indices = set()
        for path in paths:
            for interval in path:
                all_indices.update(interval.tolist())
        self.assertEqual(all_indices, set(range(self.n)))

    def test_route_returns_valid_indices(self):
        edges = calculate_edges(self.X[:, 0], self.K, categorical=False)
        k_x, _ = calculate_bins(self.X[:, 0], edges, categorical=False)
        deltas = calculate_deltas(self.f, self.X, 0, edges, k_x)
        forest, _ = generate_connected_kdforest_and_paths(
            self.X, 1, edges, deltas, self.categorical, {}, self.L
        )
        # route a training point — returned indices should be a subset of training indices
        info = forest.route(self.X[0, :])
        self.assertTrue(len(info["indices"]) > 0)
        self.assertTrue(all(0 <= idx < self.n for idx in info["indices"]))

    def test_observation_to_path_mapping(self):
        edges = calculate_edges(self.X[:, 0], self.K, categorical=False)
        k_x, _ = calculate_bins(self.X[:, 0], edges, categorical=False)
        deltas = calculate_deltas(self.f, self.X, 0, edges, k_x)
        _, paths = generate_connected_kdforest_and_paths(
            self.X, 1, edges, deltas, self.categorical, {}, self.L
        )
        l_x = observation_to_path(paths, self.n)
        self.assertEqual(len(l_x), self.n)
        # every observation should be assigned to a valid path
        self.assertTrue(all(0 <= l < self.L for l in l_x))


# ---------------------------------------------------------------------------
# End-to-end ALE
# ---------------------------------------------------------------------------

class TestALEEndToEnd(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.n = 200
        self.X = rng.standard_normal((self.n, 2))
        self.f_linear = lambda X: X[:, 0] * 3.0
        self.K = 10

    def test_main_vim_detects_important_feature(self):
        ale = ALE(self.f_linear, self.X, K=self.K, verbose=False)
        vim_x1 = ale.ale_main_vim(1)
        vim_x2 = ale.ale_main_vim(2)
        # X1 is the only relevant feature
        self.assertGreater(vim_x1, 0.1)
        self.assertAlmostEqual(vim_x2, 0.0, places=5)

    def test_explain_returns_dataframe(self):
        ale = ALE(self.f_linear, self.X, K=self.K, verbose=False)
        df = ale.explain(include=("main",))
        self.assertEqual(df.shape, (1, 2))  # 1 VIM type, 2 features

    def test_explain_local_shape(self):
        ale = ALE(self.f_linear, self.X, K=self.K, verbose=False)
        ale.explain(include=("total_connected",))
        X_explain = self.X[:5, :]
        result = ale.explain_local(X_explain)
        self.assertEqual(result.shape, (5, 2))

    def test_explain_global_shape(self):
        ale = ALE(self.f_linear, self.X, K=self.K, verbose=False)
        ale.explain(include=("total_connected",))
        result = ale.explain_global()
        self.assertEqual(result.shape, (2,))

    def test_total_connected_vim_nonzero_for_relevant_feature(self):
        ale = ALE(self.f_linear, self.X, K=self.K, verbose=False)
        vim = ale.ale_total_vim(1, method="connected")
        self.assertGreater(vim, 0.1)

    def test_additive_signal_both_features_detected(self):
        f = lambda X: X[:, 0] + X[:, 1] ** 2
        ale = ALE(f, self.X, K=self.K, verbose=False)
        vim_x1 = ale.ale_main_vim(1)
        vim_x2 = ale.ale_main_vim(2)
        self.assertGreater(vim_x1, 0.01)
        self.assertGreater(vim_x2, 0.01)


if __name__ == "__main__":
    unittest.main()
