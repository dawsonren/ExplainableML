"""
Tests for the local_method variants in ALE.explain_local:
  - "path_rep" (default): fresh forward-difference using path-bin representatives
  - "path_integral": background-averaged path integral
  - "multi_path_interpolate": structure-aware multi-path with linear depth interp
"""

import numpy as np
import unittest
from ale import ALE


class TestExplainLocalDispatcher(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.n = 300
        self.X = rng.standard_normal((self.n, 3))
        self.f_additive = lambda X: 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2]
        self.K = 8

    def _fit(self, f):
        ale = ALE(f, self.X, K=self.K, verbose=False)
        ale.explain(include=("total_connected",))
        return ale

    def test_default_is_path_rep(self):
        ale = self._fit(self.f_additive)
        X_explain = self.X[:5]
        default = ale.explain_local(X_explain)
        explicit = ale.explain_local(X_explain, local_method="path_rep")
        np.testing.assert_allclose(default, explicit)

    def test_unknown_method_raises(self):
        ale = self._fit(self.f_additive)
        with self.assertRaises(ValueError):
            ale.explain_local(self.X[:2], local_method="bogus")
        with self.assertRaises(ValueError):
            ale.explain_local(self.X[:2], local_method="interpolate")
        with self.assertRaises(ValueError):
            ale.explain_local(self.X[:2], local_method="self")
        with self.assertRaises(ValueError):
            ale.explain_local(self.X[:2], local_method="multi_path")

    def test_shape_consistent_across_methods(self):
        ale = self._fit(self.f_additive)
        X_explain = self.X[:10]
        for method in ("path_rep", "path_integral", "multi_path_interpolate"):
            result = ale.explain_local(X_explain, local_method=method)
            self.assertEqual(result.shape, (10, 3), f"shape wrong for {method}")


class TestExplainLocalPathIntegral(unittest.TestCase):
    """
    Tests for local_method='path_integral'.

    For a linear model f(x) = β·x the path-integral contribution from a single
    background point x_bg telescopes exactly to β_j * (x*_j - x_bg_j) regardless
    of bin boundaries or which intermediate training point the forest routes to.
    Averaging over all background points gives β_j * (x*_j - mean_j), where
    mean_j = X[:,j].mean().  This should hold to floating-point precision.
    """

    def setUp(self):
        rng = np.random.default_rng(7)
        self.n = 400
        self.d = 3
        self.X = rng.standard_normal((self.n, self.d))
        self.beta = np.array([2.0, -1.5, 0.5])
        self.f_linear = lambda X: np.dot(X, self.beta)

    def _fit(self):
        ale = ALE(self.f_linear, self.X, K=10, verbose=False)
        ale.explain(include=("total_connected",))
        return ale

    def test_path_integral_recovers_linear_coefficients(self):
        """path_integral gives β_j * (x*_j − mean_j) exactly for a linear model."""
        ale = self._fit()
        X_explain = self.X[:20]
        result = ale.explain_local(X_explain, local_method="path_integral")
        mean_X = ale.X_values.mean(axis=0)
        expected = (X_explain - mean_X) * self.beta
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_path_integral_boundary_interp_recovers_linear_coefficients(self):
        """boundary_interp=True also gives β_j * (x*_j − mean_j) for a linear model."""
        ale = self._fit()
        X_explain = self.X[:20]
        result = ale.explain_local(X_explain, local_method="path_integral",
                                   boundary_interp=True)
        mean_X = ale.X_values.mean(axis=0)
        expected = (X_explain - mean_X) * self.beta
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_path_integral_coefficient_direction(self):
        """The sign and relative magnitude of recovered coefficients match β."""
        ale = self._fit()
        rng = np.random.default_rng(99)
        X_explain = rng.standard_normal((50, self.d))
        result = ale.explain_local(X_explain, local_method="path_integral")
        for j in range(self.d):
            r = np.corrcoef(result[:, j], X_explain[:, j])[0, 1]
            expected_sign = np.sign(self.beta[j])
            self.assertGreater(r * expected_sign, 0.99,
                               msg=f"feature {j}: correlation sign or magnitude wrong")


class TestExplainLocalMultiPathInterpolate(unittest.TestCase):
    """
    Tests for local_method='multi_path_interpolate'.
    """

    def setUp(self):
        rng = np.random.default_rng(11)
        self.n = 400
        self.d = 3
        self.X = rng.standard_normal((self.n, self.d))
        self.beta = np.array([2.0, -1.5, 0.5])
        self.f_linear = lambda X: np.dot(X, self.beta)

    def _fit(self, f=None, K=10):
        ale = ALE(f or self.f_linear, self.X, K=K, verbose=False)
        ale.explain(include=("total_connected",))
        return ale

    def test_shape(self):
        ale = self._fit()
        X_explain = self.X[:8]
        result = ale.explain_local(X_explain, local_method="multi_path_interpolate")
        self.assertEqual(result.shape, (8, self.d))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_recovers_linear_coefficients(self):
        """For a linear model the per-bin mean-delta is constant across all
        nodes at any given bin, so the interpolation mix collapses and the
        method must still recover beta * (x*_j - mean_j) exactly."""
        ale = self._fit()
        X_explain = self.X[:20]
        result = ale.explain_local(X_explain, local_method="multi_path_interpolate")
        mean_X = ale.X_values.mean(axis=0)
        expected = (X_explain - mean_X) * self.beta
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_background_subsample(self):
        ale = self._fit()
        X_explain = self.X[:5]
        full = ale.explain_local(X_explain, local_method="multi_path_interpolate")
        sub = ale.explain_local(
            X_explain, local_method="multi_path_interpolate",
            background_size=50, background_seed=123,
        )
        self.assertEqual(sub.shape, full.shape)
        self.assertTrue(np.all(np.isfinite(sub)))

    def test_subtree_mean_delta_consistent(self):
        """Internal node mean_delta[k] = count-weighted average of children's."""
        ale = self._fit()
        for j in range(self.d):
            forest = ale.connected_forest[j]
            K = forest.K
            stack = [forest.root]
            while stack:
                node = stack.pop()
                if node.is_leaf:
                    continue
                stack.append(node.left)
                stack.append(node.right)
                for k in range(K):
                    nL = node.left.n_per_bin[k]
                    nR = node.right.n_per_bin[k]
                    total = nL + nR
                    if total == 0:
                        self.assertTrue(np.isnan(node.mean_delta[k]))
                        continue
                    sL = node.left.mean_delta[k] * nL if nL > 0 else 0.0
                    sR = node.right.mean_delta[k] * nR if nR > 0 else 0.0
                    expected = (sL + sR) / total
                    np.testing.assert_allclose(
                        node.mean_delta[k], expected, atol=1e-12
                    )

    def test_subtree_mean_delta_matches_leaf_occurrence_mean(self):
        """Global invariant: at every node and bin k, mean_delta[k] equals the
        leaf-occurrence-weighted mean of deltas over the subtree."""
        ale = self._fit()

        def collect_leaf_occurrences(node, k):
            if node.is_leaf:
                if not node.R[k]:
                    return np.empty(0, dtype=int)
                return np.unique(np.asarray(node.R[k], dtype=int))
            left = collect_leaf_occurrences(node.left, k)
            right = collect_leaf_occurrences(node.right, k)
            if left.size == 0:
                return right
            if right.size == 0:
                return left
            return np.concatenate([left, right])

        for j in range(self.d):
            forest = ale.connected_forest[j]
            K = forest.K
            deltas_j = ale.deltas[j]
            stack = [forest.root]
            while stack:
                node = stack.pop()
                if not node.is_leaf:
                    stack.append(node.left)
                    stack.append(node.right)
                for k in range(K):
                    occ = collect_leaf_occurrences(node, k)
                    self.assertEqual(int(node.n_per_bin[k]), int(occ.size))
                    if occ.size == 0:
                        self.assertTrue(np.isnan(node.mean_delta[k]))
                    else:
                        np.testing.assert_allclose(
                            node.mean_delta[k], float(deltas_j[occ].mean()), atol=1e-12
                        )

    def test_assign_middle_node_weights_edge_cases(self):
        from ale.tree_partitioning import ConnectedKDForest

        class _N:
            pass
        nodes = [_N() for _ in range(5)]

        self.assertEqual(ConnectedKDForest.assign_middle_node_weights(nodes[:3], 0), [])

        out = ConnectedKDForest.assign_middle_node_weights([nodes[0]], 3)
        self.assertEqual(len(out), 3)
        for pairs in out:
            self.assertIs(pairs[0][0], nodes[0])
            self.assertAlmostEqual(pairs[0][1], 1.0)
            self.assertAlmostEqual(pairs[1][1], 0.0)

        out = ConnectedKDForest.assign_middle_node_weights([nodes[0], nodes[1]], 2)
        self.assertEqual(len(out), 2)
        for pairs in out:
            self.assertIs(pairs[0][0], nodes[0])
            self.assertAlmostEqual(pairs[0][1], 1.0)
            self.assertAlmostEqual(pairs[1][1], 0.0)

        out = ConnectedKDForest.assign_middle_node_weights(
            [nodes[0], nodes[1], nodes[2]], 3
        )
        self.assertEqual(len(out), 3)
        for pairs in out:
            self.assertIs(pairs[0][0], nodes[1])
            self.assertAlmostEqual(pairs[0][1], 1.0)
            self.assertAlmostEqual(pairs[1][1], 0.0)

        path = [nodes[0], nodes[1], nodes[2], nodes[3], nodes[4]]
        L, LCA, R = nodes[1], nodes[2], nodes[3]
        out = ConnectedKDForest.assign_middle_node_weights(path, 2)
        self.assertEqual(len(out), 2)
        (n_lo1, w_lo1), (n_hi1, w_hi1) = out[0]
        self.assertIs(n_lo1, L); self.assertIs(n_hi1, LCA)
        self.assertAlmostEqual(w_lo1, 1.0 / 3.0)
        self.assertAlmostEqual(w_hi1, 2.0 / 3.0)
        self.assertAlmostEqual(w_lo1 + w_hi1, 1.0)
        (n_lo2, w_lo2), (n_hi2, w_hi2) = out[1]
        self.assertIs(n_lo2, LCA); self.assertIs(n_hi2, R)
        self.assertAlmostEqual(w_lo2, 2.0 / 3.0)
        self.assertAlmostEqual(w_hi2, 1.0 / 3.0)
        self.assertAlmostEqual(w_lo2 + w_hi2, 1.0)


class TestExplainLocalWeights(unittest.TestCase):
    """
    Tests for ALE.explain_local_weights — the linear-weights view of the
    multi_path_interpolate local estimator.
    """

    def setUp(self):
        rng = np.random.default_rng(11)
        self.n = 300
        self.d = 3
        self.X = rng.standard_normal((self.n, self.d))
        self.beta = np.array([2.0, -1.5, 0.5])
        self.f_linear = lambda X: np.dot(X, self.beta)
        self.f_nonlinear = lambda X: X[:, 0] ** 2 - 0.5 * X[:, 1] * X[:, 2]

    def _fit(self, f=None, K=8):
        ale = ALE(f or self.f_linear, self.X, K=K, verbose=False)
        ale.explain(include=("total_connected",))
        return ale

    def test_shape(self):
        ale = self._fit()
        Xe = self.X[:5]
        W = ale.explain_local_weights(Xe, local_method="multi_path_interpolate")
        self.assertEqual(W.shape, (5, self.d, self.n))
        self.assertTrue(np.all(np.isfinite(W)))

    def test_equivalence_multi_path_interpolate(self):
        ale = self._fit()
        Xe = self.X[:8]
        E = ale.explain_local(Xe, local_method="multi_path_interpolate")
        W = ale.explain_local_weights(Xe, local_method="multi_path_interpolate")
        recon = np.zeros_like(E)
        for j in range(self.d):
            recon[:, j] = W[:, j, :] @ ale.deltas[j]
        np.testing.assert_allclose(recon, E, atol=1e-10)

    def test_equivalence_on_nonlinear_model(self):
        """Equivalence should hold for any model — not just linear."""
        ale = self._fit(f=self.f_nonlinear)
        Xe = self.X[:6]
        E = ale.explain_local(Xe, local_method="multi_path_interpolate")
        W = ale.explain_local_weights(Xe, local_method="multi_path_interpolate")
        recon = np.zeros_like(E)
        for j in range(self.d):
            recon[:, j] = W[:, j, :] @ ale.deltas[j]
        np.testing.assert_allclose(recon, E, atol=1e-10)

    def test_linear_recovery_via_weights(self):
        """W @ deltas on a linear model still recovers beta * (x - mean)."""
        ale = self._fit()
        Xe = self.X[:10]
        W = ale.explain_local_weights(Xe, local_method="multi_path_interpolate")
        recon = np.zeros((Xe.shape[0], self.d))
        for j in range(self.d):
            recon[:, j] = W[:, j, :] @ ale.deltas[j]
        mean_X = ale.X_values.mean(axis=0)
        expected = (Xe - mean_X) * self.beta
        np.testing.assert_allclose(recon, expected, atol=1e-6)

    def test_background_subsample_equivalence(self):
        """With a bg subsample, W still satisfies W @ deltas == explain_local."""
        ale = self._fit()
        Xe = self.X[:4]
        bg_size = 40
        seed = 31
        W = ale.explain_local_weights(
            Xe, local_method="multi_path_interpolate",
            background_size=bg_size, background_seed=seed,
        )
        E = ale.explain_local(
            Xe, local_method="multi_path_interpolate",
            background_size=bg_size, background_seed=seed,
        )
        recon = np.zeros_like(E)
        for j in range(self.d):
            recon[:, j] = W[:, j, :] @ ale.deltas[j]
        np.testing.assert_allclose(recon, E, atol=1e-10)

    def test_invalid_method(self):
        ale = self._fit()
        with self.assertRaises(ValueError):
            ale.explain_local_weights(self.X[:2], local_method="path_rep")
        with self.assertRaises(ValueError):
            ale.explain_local_weights(self.X[:2], local_method="path_integral")


class TestExplainLocalCategorical(unittest.TestCase):
    def test_categorical_feature_uses_lookup_fallback(self):
        """For categorical features, path_rep falls back to direct g-value
        lookup (forward differences are ill-defined on categoricals)."""
        rng = np.random.default_rng(1)
        n = 150
        X = np.column_stack([
            rng.standard_normal(n),
            rng.choice([0, 1, 2], size=n),
        ])
        f = lambda X: X[:, 0] + (X[:, 1] == 1).astype(float)
        ale = ALE(
            f, X, K=6, categorical=[False, True], verbose=False
        )
        ale.explain(include=("total_connected",))
        X_explain = X[:10]
        result = ale.explain_local(X_explain, local_method="path_rep")
        self.assertEqual(result.shape, (10, 2))
        self.assertTrue(np.all(np.isfinite(result[:, 1])))


if __name__ == "__main__":
    unittest.main()
