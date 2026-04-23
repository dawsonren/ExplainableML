"""
Calculate Shapley values for explanation.
"""

import warnings
from math import factorial

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import check_random_state

from shap import kmeans, sample
from shap.maskers import Independent
from shap.explainers import (
    KernelExplainer,
    PermutationExplainer,
    TreeExplainer,
    ExactExplainer,
    LinearExplainer,
)

from utils import Explanation


class SHAP(Explanation):
    def __init__(self, f, X, feature_names=None, categorical=None, verbose=True):
        super().__init__(f, X, feature_names=feature_names, categorical=categorical)
        self.verbose = verbose

    def explain_local(
        self,
        X_explain,
        method="permutation",
        kwargs=None,
        sample_method=None,
        sample_size=1000,
        random_state=None
    ):
        if kwargs is None:
            kwargs = {}

        # convert X_explain to numpy array if it's a pandas Series
        if isinstance(X_explain, pd.Series):
            X_explain = X_explain.values

        # promote X_explain to a 2D array if necessary
        if X_explain.ndim == 1:
            X_explain = X_explain[np.newaxis, :]

        rng = check_random_state(random_state)

        # sample from X_values if sample is not None
        if sample_method == "kmeans":
            X_values = kmeans(self.X_values, sample_size).data
        elif sample_method == "sample":
            X_values = sample(self.X_values, sample_size, random_state=rng)
        else:
            X_values = self.X_values

        masker = Independent(X_values, max_samples=self.n)

        if method == "permutation_shap":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                explainer = PermutationExplainer(
                    self.f, masker, feature_names=self.feature_names
                )
                # expects npermutations
                shap_values = explainer.shap_values(X_explain, **kwargs)
        elif method == "kernel_shap":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                explainer = KernelExplainer(
                    self.f, X_values, feature_names=self.feature_names
                )
                shap_values = explainer.shap_values(
                    X_explain, silent=True, **kwargs
                )
        elif method == "tree_shap":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                explainer = TreeExplainer(self.f, feature_names=self.feature_names)
                shap_values = explainer.shap_values(X_explain)
        elif method == "exact_shap":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                explainer = ExactExplainer(
                    self.f, masker, feature_names=self.feature_names
                )
                shap_values = explainer(X_explain).values
        elif method == "linear_shap":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                explainer = LinearExplainer(
                    self.f, masker, feature_names=self.feature_names, **kwargs
                )
                shap_values = explainer.shap_values(X_explain)
        else:
            raise ValueError(f"SHAP: Unknown method {method}")

        return shap_values
    
    def shim(
        self,
        x_explain,
    ):
        """Exact Shapley via coalition enumeration. O(2^d * n_background) per point.
        Equivalent to the closed-form d=2 formula. Practical up to d ~ 10."""
        n_explain, d = x_explain.shape
        explanation = np.zeros((n_explain, d))
        denom = factorial(d)

        for i in range(n_explain):
            x = x_explain[i]
            v = np.empty(1 << d)
            for mask in range(1 << d):
                Z = self.X_values.copy()
                for j in range(d):
                    if mask & (1 << j):
                        Z[:, j] = x[j]
                v[mask] = self.f(Z).mean()
            for feat in range(d):
                phi = 0.0
                for mask in range(1 << d):
                    if not (mask & (1 << feat)):
                        s = bin(mask).count('1')
                        w = factorial(s) * factorial(d - s - 1) / denom
                        phi += w * (v[mask | (1 << feat)] - v[mask])
                explanation[i, feat] = phi

        return explanation

    def explain_global(
        self,
        method="permutation",
        kwargs=None,
        sample_method=None,
        sample_size=1000,
        random_state=None,
    ):
        """
        The global effect is the average of the absolute values of the local effects.
        """
        local_effects = self.explain_local(
            self.X_values,
            method=method,
            kwargs=kwargs,
            sample_method=sample_method,
            sample_size=sample_size,
            random_state=random_state,
        )

        return np.mean(np.abs(local_effects), axis=0)