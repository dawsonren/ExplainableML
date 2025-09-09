import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge, lars_path

from utils import Explanation

KERNEL_WIDTH_MULTIPLIER = 0.75


def exponential_kernel(d, kernel_width):
    return np.sqrt(np.exp(-(d**2) / kernel_width**2))


def k_lasso(X, y, weights, relevant_features):
    # NOTE: weighting the data and labels is equivalent to solving
    # the weighted LASSO problem
    weighted_data = (X - np.average(X, axis=0, weights=weights)) * np.sqrt(
        weights[:, np.newaxis]
    )
    weighted_labels = (y - np.average(y, weights=weights)) * np.sqrt(weights)
    nonzero = range(weighted_data.shape[1])

    # construct lasso path
    _, _, coefs = lars_path(
        weighted_data, weighted_labels, method="lasso", verbose=False
    )

    # find the last non-zero coefficients
    for i in range(coefs.shape[1] - 1, 0, -1):
        nonzero = coefs[:, i].nonzero()[0]
        if len(nonzero) <= relevant_features:
            break
    used_features = nonzero

    return used_features


class LIME(Explanation):
    def __init__(self, f, X, feature_names=None, categorical=None, verbose=True, kernel_width=None):
        """
        Linear Interpretable Model Explanations explainer.

        Parameters:
        - f: function to evaluate the model
        - X: numpy array of shape (n, p)
        - feature_names: list of feature names
        - categorical: list of booleans indicating if feature is categorical
        """
        super().__init__(f, X, feature_names=feature_names, categorical=categorical)
        self.verbose = verbose
        self.kernel_width = kernel_width if kernel_width is not None else KERNEL_WIDTH_MULTIPLIER * self.d

    def calculate_coefficients(self, n_X, n_y, distances, relevant_features):
        """Takes perturbed data, labels and distances, returns explanation.

        Parameters:
            n_X : perturbed data
            n_y : model predictions on perturbed data
            distances: distances to original data point
            num_features: maximum number of features in explanation

        Returns:
            local_exp: list of tuples (feature id, feature weight), sorted by absolute value of weight
            score: R^2 of the local linear model
            local_pred: prediction of the local linear model on the original data point
        """
        # construct weights using kernel function
        weights = exponential_kernel(
            distances, kernel_width=self.kernel_width
        )

        # select num_features features
        used_features = k_lasso(n_X, n_y, weights, relevant_features)

        # fit the local model
        easy_model = Ridge(alpha=1, fit_intercept=True)
        easy_model.fit(n_X[:, used_features], n_y, sample_weight=weights)

        # score the local model
        prediction_score = easy_model.score(
            n_X[:, used_features], n_y, sample_weight=weights
        )

        # get the prediction on the original data point
        local_pred = easy_model.predict(n_X[0, used_features].reshape(1, -1))

        return (
            dict(
                zip(
                    [self.feature_names[i] for i in used_features],
                    easy_model.coef_.flatten(),
                )
            ),
            prediction_score,
            local_pred,
        )

    def generate_perturbed_data(self, explain_X, num_samples):
        """
        Generate perturbed data around explain_X.

        Parameters:
            explain_X: the data point to explain, shape (d, 1)
            num_samples: number of perturbed samples to generate

        Returns:
            n_X: perturbed data, shape (num_samples, d)
            n_y: model predictions on perturbed data, shape (num_samples,)
            distances: distances from perturbed data to explain_X, shape (num_samples,)
        """
        # generate perturbations
        n_X = np.zeros((num_samples, self.d))
        for i in range(self.d):
            if self.categorical[i]:
                # for categorical features, weighted sample from the existing categories
                categories = self.X[self.feature_names[i]].unique()
                weights = self.X[self.feature_names[i]].value_counts(normalize=True)
                n_X[:, i] = np.random.choice(categories, size=num_samples, p=weights)
            else:
                # for continuous features, sample from a normal distribution
                # NOTE: this does not take into account correlation between variables
                # however for sufficiently small perturbations, this may be reasonable
                std_dev = np.std(self.X_values[:, i])
                n_X[:, i] = np.random.normal(
                    loc=explain_X[i], scale=std_dev, size=num_samples
                )

        # get model predictions on perturbed data
        n_y = self.f(n_X)

        # compute euclidean distances for continuous features
        # and hamming distance for categorical features
        distances = np.zeros(num_samples)
        for i in range(self.d):
            if self.categorical[i]:
                distances += (n_X[:, i] != explain_X[i]).astype(int)
            else:
                distances += (n_X[:, i] - explain_X[i]) ** 2
        distances = np.sqrt(distances)

        return n_X, n_y, distances

    def explain_local(self, explain_idx, relevant_features=10, num_samples=5000):
        """
        Produce LIME plots for all features for a given observation.
        """
        # check explain_idx is int
        if not isinstance(explain_idx, int):
            raise ValueError("explain_idx must be an integer.")

        n_features = self.X.shape[1]
        explain_X = self.X_values[explain_idx, :]
        explain_X = explain_X.reshape(-1, 1)
        actual_prediction = self.f(explain_X.T)[0]

        # generate perturbed data
        n_X, n_y, distances = self.generate_perturbed_data(explain_X, num_samples)

        # calculate coefficients
        (
            local_exp,
            score,
            local_pred,
        ) = self.calculate_coefficients(
            n_X, n_y, distances, relevant_features=relevant_features
        )

        if self.verbose:
            # plot the feature importance, use barh
            fig, ax = plt.subplots(figsize=(6, 1.5 * n_features))
            feature_names = local_exp.keys()
            feature_weights = local_exp.values()
            y_pos = np.arange(len(feature_names))
            ax.barh(y_pos, feature_weights, align="center")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_names)
            ax.invert_yaxis()  # labels read top-to-bottom
            ax.set_xlabel("Feature Weight")
            ax.set_title("LIME Feature Importance")

            # nicely formatted table of feature | value
            print("R^2 of local model:", score)
            print("Feature | Value")
            print("-------------------")
            for feature_idx in range(1, n_features + 1):
                actual_value = explain_X[feature_idx - 1]
                print(f"{self.feature_names[feature_idx - 1]} | {actual_value}")
            print("-------------------")
            print(f"Prediction: {actual_prediction}")
            print(f"Local Prediction: {local_pred}")

        return local_exp
