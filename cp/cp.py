import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils import Explanation


def _ceteris_paribus(f, X, feature_idx, explain_idx, bins=10, categorical=False):
    """
    Calculate Ceteris Paribus (CP) values for a given feature.

    Parameters:
    - f: function to evaluate the model
    - X: numpy array of shape (n, p)
    - feature_idx: 1-based index of the feature
    - explain_idx: 0-based index of the observation to explain
    - bins: number of bins for CP calculation

    Returns:
    - edges: bin edges
    - predictions: CP values at bin edges
    """
    idx = feature_idx - 1  # convert to 0-based index
    x = X[:, idx]

    # equal-width bin edges
    if categorical:
        edges = np.unique(x)
        bins = len(edges)
    else:
        edges = np.linspace(x.min(), x.max(), bins)
        edges[0], edges[-1] = x.min(), x.max()

    # calculate predictions for each edge
    predictions = np.zeros(bins)
    X_temp = np.tile(X[explain_idx, :], (bins, 1))
    X_temp[:, idx] = edges
    predictions = f(X_temp)

    return edges, predictions


class CeterisParibus(Explanation):
    def __init__(self, f, X, feature_names=None, bins=10, categorical=None):
        """
        Ceteris Paribus (CP) explainer.

        Parameters:
        - f: function to evaluate the model
        - X: numpy array of shape (n, p)
        - feature_idx: 1-based index of the feature
        - explain_idx: 0-based index of the observation to explain
        - bins: number of bins for CP calculation
        """
        super().__init__(f, X, feature_names=feature_names, categorical=categorical)
        self.bins = bins

    def explain_local(self, explain_idx):
        """
        Produce CP plots for all features for a given observation.
        """
        n_features = self.X.shape[1]
        fig, axes = plt.subplots(n_features, 1, figsize=(6, 4 * n_features))
        explain_X = self.X_values[explain_idx, :]
        explain_X = explain_X.reshape(-1, 1)
        actual_prediction = self.f(explain_X.T)[0]
        # nicely formatted table of feature | value
        print("Feature | Value")
        print("-------------------")
        for feature_idx in range(1, n_features + 1):
            actual_value = explain_X[feature_idx - 1]
            print(f"{self.feature_names[feature_idx - 1]} | {actual_value}")
        print("-------------------")
        print(f"Prediction: {actual_prediction}")

        for feature_idx in range(1, n_features + 1):
            edges, predictions = _ceteris_paribus(
                self.f,
                self.X_values,
                feature_idx,
                explain_idx,
                self.bins,
                categorical=self.categorical[feature_idx - 1],
            )
            # black line plot without markers
            axes[feature_idx - 1].plot(edges, predictions, marker=None, color="black")
            axes[feature_idx - 1].set_title(f"Feature {feature_idx}")
            axes[feature_idx - 1].set_xlabel(self.feature_names[feature_idx - 1])
            axes[feature_idx - 1].set_ylabel("Model Prediction")
            # mark the actual feature value using a dot
            actual_value = explain_X[feature_idx - 1]
            axes[feature_idx - 1].plot(
                actual_value, actual_prediction, marker="o", color="black"
            )  # black dot

        fig.tight_layout()
