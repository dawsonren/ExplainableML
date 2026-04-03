"""
Calculate Shapley values for explanation.
"""

import warnings
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
        # convert X_explain to numpy array if it's a pandas Series
        if isinstance(X_explain, pd.Series):
            X_explain = X_explain.values

        # promote X_explain to a 2D array if necessary
        if X_explain.ndim == 1:
            X_explain = X_explain[np.newaxis, :]

        # sample from X_values if sample is not None
        if sample_method == "kmeans":
            X_values = kmeans(self.X_values, sample_size).data
        elif sample_method == "sample":
            rng = check_random_state(random_state)
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
                shap_values = explainer.shap_values(X_explain, **kwargs)[
                    0
                ]
        elif method == "kernel_shap":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                explainer = KernelExplainer(
                    self.f, X_values, feature_names=self.feature_names
                )
                shap_values = explainer.shap_values(
                    X_explain, silent=True, **kwargs
                )[0]
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
        # this only works for 2 variables but is exact!
        explanation = np.zeros((x_explain.shape[0], 2))
        for i in range(x_explain.shape[0]):
            f_ss = self.f(x_explain[i, :].reshape(1, -1))
            bs = self.X_values.copy()
            bs[:, 1] = x_explain[i, 1]
            sb = self.X_values.copy()
            sb[:, 0] = x_explain[i, 0]
            bb = self.X_values.copy()
            f_bs = self.f(bs)
            f_sb = self.f(sb)
            f_bb = self.f(bb)

            explanation[i, 0] = 0.5 * (f_ss - f_bs).mean() + 0.5 * (f_sb - f_bb).mean()
            explanation[i, 1] = 0.5 * (f_ss - f_sb).mean() + 0.5 * (f_bs - f_bb).mean()

        return explanation

    def explain_global(
        self,
        method="permutation",
        kwargs=None,
        sample_method=None,
        sample_size=1000,
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
        )

        return np.mean(np.abs(local_effects), axis=0)