"""
Tests for the local_method variants in ALE.explain_local:
  - "interpolate" (default): linear interpolation of centered g-value curve
  - "path_rep": fresh forward-difference using path-bin representatives
  - "self": fresh forward-difference using the explain point's own x_\\j
"""

import numpy as np
import unittest
from ale import ALE


class TestExplainLocalMethods(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.n = 300
        self.X = rng.standard_normal((self.n, 3))
        # additive linear model: all three methods should agree
        self.f_additive = lambda X: 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2]
        # interactive model: methods should diverge
        self.f_interactive = lambda X: X[:, 0] * X[:, 1] + 0.3 * X[:, 2]
        self.K = 8

    def _fit(self, f):
        ale = ALE(f, self.X, K=self.K, verbose=False)
        ale.explain(include=("total_connected",))
        return ale

    def test_default_is_interpolate(self):
        ale = self._fit(self.f_additive)
        X_explain = self.X[:5]
        default = ale.explain_local(X_explain)
        explicit = ale.explain_local(X_explain, local_method="interpolate")
        np.testing.assert_allclose(default, explicit)

    def test_unknown_method_raises(self):
        ale = self._fit(self.f_additive)
        with self.assertRaises(ValueError):
            ale.explain_local(self.X[:2], local_method="bogus")

    def test_shape_consistent_across_methods(self):
        ale = self._fit(self.f_additive)
        X_explain = self.X[:10]
        for method in ("interpolate", "path_rep", "self"):
            result = ale.explain_local(X_explain, local_method=method)
            self.assertEqual(result.shape, (10, 3), f"shape wrong for {method}")

    def test_additive_model_methods_agree(self):
        # For an additive linear model, f(x_j, x_\j) - f(x_j_left, x_\j)
        # depends only on x_j, not on x_\j. So "path_rep" and "self"
        # should both closely match "interpolate" (which on an additive
        # model captures the exact linear slope via bin-averaged deltas).
        ale = self._fit(self.f_additive)
        X_explain = self.X[:20]
        interp = ale.explain_local(X_explain, local_method="interpolate")
        path_rep = ale.explain_local(X_explain, local_method="path_rep")
        self_m = ale.explain_local(X_explain, local_method="self")
        # "self" vs "path_rep" should match exactly on additive models
        # because the term is independent of x_\j.
        np.testing.assert_allclose(self_m, path_rep, atol=1e-10)
        # Both should be close to interpolate within bin-discretization error.
        np.testing.assert_allclose(interp, self_m, atol=0.5)

    def test_interactive_model_methods_differ(self):
        # On an interactive model, "self" and "path_rep" should disagree
        # with "interpolate" on at least some points.
        ale = self._fit(self.f_interactive)
        X_explain = self.X[:30]
        interp = ale.explain_local(X_explain, local_method="interpolate")
        self_m = ale.explain_local(X_explain, local_method="self")
        # Look at the interacting feature (index 0): differences should be
        # non-trivial for at least some explain points.
        diffs = np.abs(interp[:, 0] - self_m[:, 0])
        self.assertGreater(diffs.max(), 1e-3)

    def test_self_method_formula(self):
        # "self" local effect for feature j equals
        #   g*(x_j_left) + [f(x_j, x_\j) - f(x_j_left, x_\j)]
        # Verify this matches a manual computation on a single point.
        ale = self._fit(self.f_interactive)
        j = 0  # feature index (0-based)
        # use a training point so route -> its own index / path
        x = self.X[7:8]
        result = ale.explain_local(x, local_method="self")[0, j]

        # recompute manually
        edges = ale.edges[j]
        K = len(edges) - 1
        x_j = x[0, j]
        k_star = int(np.clip(np.searchsorted(edges, x_j, side="right") - 1, 0, K - 1))
        x_j_left = edges[k_star]

        x_left = x.copy()
        x_left[0, j] = x_j_left
        term = float((ale.f(x) - ale.f(x_left))[0])

        # base is g*(x_j_left); averaged over the routed path(s)
        forest = ale.connected_forest[j]
        info = forest.route_and_pick_representative(x[0], ale.X_values, levels_up=0)
        l_xs = [ale.observation_to_path[j][xi] for xi in info["indices"]]
        centered = ale.centered_g_values[j].centered_g_values
        bases = [centered[k_star, l] for l in l_xs]
        expected = np.mean([b + term for b in bases])

        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestExplainLocalCategorical(unittest.TestCase):
    def test_categorical_feature_uses_interpolate_fallback(self):
        # For categorical features, local_method is ignored; all three
        # methods should return identical results.
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
        interp = ale.explain_local(X_explain, local_method="interpolate")
        path_rep = ale.explain_local(X_explain, local_method="path_rep")
        self_m = ale.explain_local(X_explain, local_method="self")
        # The categorical column (index 1) must be identical across methods.
        np.testing.assert_allclose(interp[:, 1], path_rep[:, 1])
        np.testing.assert_allclose(interp[:, 1], self_m[:, 1])


if __name__ == "__main__":
    unittest.main()
