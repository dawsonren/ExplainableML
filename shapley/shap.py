"""
Calculate Shapley values for explanation.
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from shap.explainers import KernelExplainer, PermutationExplainer, TreeExplainer

from utils import Explanation


import numpy as np

def permutation_shapley_values(f, X, x_explain, num_samples=100, rng=None):
    """
    Vectorized permutation-based Shapley values for a single instance.
    Standard (naive) Monte Carlo estimation — fully vectorized over samples.

    Parameters
    ----------
    f : callable
        Vectorized model prediction function. Accepts array of shape (m, p)
        and returns a 1D array of length m (or shape (m, 1)).
    X : np.ndarray, shape (n, p)
        Background dataset used to sample reference rows.
    x_explain : np.ndarray, shape (p,)
        The instance to explain.
    num_samples : int
        Number of random permutations (Monte Carlo samples).
    rng : np.random.Generator or int or None
        Random generator or seed.

    Returns
    -------
    np.ndarray, shape (p,)
        Estimated Shapley values per feature.
    """
    X = np.asarray(X)
    x_explain = np.asarray(x_explain, dtype=X.dtype)
    n, p = X.shape

    if isinstance(rng, (int, np.integer)) or rng is None:
        rng = np.random.default_rng(rng)

    # Draw all permutations and all background samples
    perms = np.array([rng.permutation(p) for _ in range(num_samples)])
    rows = X[rng.integers(0, n, size=num_samples)]

    # Build all "with_k" and "without_k" matrices (each num_samples * p rows)
    with_k = np.zeros((num_samples * p, p))
    without_k = np.zeros((num_samples * p, p))

    for i in range(num_samples):
        perm = perms[i]
        row = rows[i]
        row_perm = row[perm]
        x_perm = x_explain[perm]

        # Fill in the p rows for this sample
        wk = np.tile(x_perm, (p, 1))
        for pos in range(p - 1):
            wk[pos, pos + 1:] = row_perm[pos + 1:]
        wout = wk.copy()
        for pos in range(p):
            wout[pos, pos] = row_perm[pos]

        inv = np.argsort(perm)
        start = i * p
        end = start + p
        with_k[start:end] = wk[:, inv]
        without_k[start:end] = wout[:, inv]

    # single batch prediction
    y_with = np.asarray(f(with_k)).reshape(-1)
    y_without = np.asarray(f(without_k)).reshape(-1)
    deltas = y_with - y_without

    # Aggregate contributions per feature across all permutations
    phis = np.zeros(p, dtype=float)
    for i in range(num_samples):
        perm = perms[i]
        start = i * p
        end = start + p
        for pos, feat in enumerate(perm):
            phis[feat] += deltas[start + pos]

    phis /= num_samples
    return phis


class SHAP(Explanation):
    def __init__(self, f, X, feature_names=None, categorical=None, verbose=True):
        super().__init__(f, X, feature_names=feature_names, categorical=categorical)
        self.verbose = verbose

    def explain(self, method="kernel_shap", num_samples=100):
        # take mean of |phi_j| over all samples
        shap_values = np.zeros((self.n, self.d))
        if method == "permutation_shap":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                explainer = PermutationExplainer(self.f, self.X_values, feature_names=self.feature_names)
                shap_values = explainer.shap_values(self.X_values, npermutations=num_samples)
        elif method == "kernel_shap":
             with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                explainer = KernelExplainer(self.f, self.X_values, feature_names=self.feature_names)
                shap_values = explainer.shap_values(self.X_values, nsamples=num_samples, silent=True)
        elif method == "tree_shap":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                explainer = TreeExplainer(self.f, feature_names=self.feature_names)
                shap_values = explainer.shap_values(self.X_values)
        elif method == "permutation":
            for i in range(self.n):
                shap_values[i, :] = permutation_shapley_values(
                    self.f,
                    self.X_values,
                    self.X_values[i, :],
                    num_samples=num_samples
                )

        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        explanation = {}
        for j in range(self.d):
            explanation[self.feature_names[j]] = mean_abs_shap[j]
        return explanation

    def explain_local(
        self, x_explain, method="permutation", num_samples=100
    ):
        # check dimension of x_explain
        if x_explain.ndim != 1 or x_explain.shape[0] != self.d:
            raise ValueError(f"x_explain must be a 1-D array of length {self.d}.")
        # check method
        if method not in ["permutation", "permutation_shap", "kernel_shap"]:
            raise ValueError("method must be either 'permutation' or 'conditional'.")
        # convert x_explain to numpy array if it's a pandas Series
        if isinstance(x_explain, pd.Series):
            x_explain = x_explain.values

        if method == "permutation_shap":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                explainer = PermutationExplainer(self.f, self.X_values, feature_names=self.feature_names)
                shap_values = explainer.shap_values(x_explain, npermutations=num_samples)
        elif method == "kernel_shap":
             with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                explainer = KernelExplainer(self.f, self.X_values, feature_names=self.feature_names)
                shap_values = explainer.shap_values(x_explain, nsamples=num_samples, silent=True)
        elif method == "tree_shap":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                explainer = TreeExplainer(self.f, feature_names=self.feature_names)
                shap_values = explainer.shap_values(x_explain)
        elif method == "permutation":
            shap_values = permutation_shapley_values(
                self.f,
                self.X_values,
                x_explain,
                num_samples=num_samples
            )

        explanation = {}
        for j in range(self.d):
            explanation[self.feature_names[j]] = shap_values[j]

        if self.verbose:
            # plot the feature importance, use barh
            fig, ax = plt.subplots(figsize=(6, 1.5 * self.d))
            y_pos = np.arange(len(explanation))
            ax.barh(y_pos, explanation.values(), align="center")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(explanation.keys())
            ax.invert_yaxis()  # labels read top-to-bottom
            ax.set_xlabel("Feature Weight")
            ax.set_title("SHAP Feature Importance")

            # nicely formatted table of feature | value
            print("Feature | Value")
            print("-------------------")
            for feature_idx in range(1, self.d + 1):
                actual_value = explanation[self.feature_names[feature_idx - 1]]
                print(f"{self.feature_names[feature_idx - 1]} | {actual_value}")
            print("-------------------")

        return explanation
