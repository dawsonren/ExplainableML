"""
Calculate Shapley values for explanation.
"""

import numpy as np
import matplotlib.pyplot as plt

from utils import Explanation


def permutation_shapley_values(f, X, explain_idx, feature_idx, num_samples=100):
    """
    Calculate Shapley values for a specific feature.

    Parameters:
        f: function to evaluate the model
        X: numpy array of shape (n, p)
        explain_idx: 0-index of the instance to explain
        feature_idx: 1-index of the feature to explain
        num_samples: number of samples to use for estimation

    Returns:
        shap_values: estimated Shapley values for the specified feature
    """
    # get x
    x = X[explain_idx, :]
    idx = feature_idx - 1
    p = X.shape[1]

    # Initialize Shapley value
    shap_value = 0

    # Average over samples
    for _ in range(num_samples):
        # draw a random instance from X
        z = X[np.random.choice(X.shape[0]), :]

        # choose a random permutation of feature indices
        perm = np.random.permutation(p)

        # position where feature j appears in this permutation
        j_pos = int(np.where(perm == idx)[0][0])

        # order the features according to the permutation
        x_order = x[perm]
        z_order = z[perm]

        # construct two new instances
        prefix = x_order[:j_pos]
        suffix = z_order[j_pos + 1:]
        x_with_j = np.array([*prefix, x_order[j_pos], *suffix])
        x_without_j = np.array([*prefix, z_order[j_pos], *suffix])

        # unpermute back to original order
        inv_perm = np.empty(p, dtype=int)
        inv_perm[perm] = np.arange(p)
        x_with_j = x_with_j[inv_perm].reshape(1, -1)
        x_without_j = x_without_j[inv_perm].reshape(1, -1)

        # get marginal contribution
        shap_value += f(x_with_j) - f(x_without_j)

    return shap_value / num_samples


class SHAP(Explanation):
    def __init__(self, f, X, feature_names=None, categorical=None, verbose=True):
        super().__init__(f, X, feature_names=feature_names, categorical=categorical)
        self.verbose = verbose

    def explain_local(self, explain_idx, num_samples=100):
        # check explain_idx is int
        if not isinstance(explain_idx, int):
            raise ValueError("explain_idx must be an integer.")
        
        explanation = {}
        for i in range(self.d):
            explanation[self.feature_names[i]] = permutation_shapley_values(
                self.f, self.X_values, explain_idx, i + 1, num_samples=num_samples
            )

        if self.verbose:
            # plot the feature importance, use barh
            fig, ax = plt.subplots(figsize=(6, 1.5 * self.d))
            y_pos = np.arange(len(explanation))
            ax.barh(y_pos, explanation.values(), align="center")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(explanation.keys())
            ax.invert_yaxis()  # labels read top-to-bottom
            ax.set_xlabel("Feature Weight")
            ax.set_title("LIME Feature Importance")

            # nicely formatted table of feature | value
            print("Feature | Value")
            print("-------------------")
            for feature_idx in range(1, self.d + 1):
                actual_value = explanation[self.feature_names[feature_idx - 1]]
                print(f"{self.feature_names[feature_idx - 1]} | {actual_value}")
            print("-------------------")

        return explanation
