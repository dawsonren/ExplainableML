"""
Python class that handles ALE explanations.

NOTE: This class is designed to work with both numpy arrays and pandas DataFrames.
It automatically detects the type of input data and adjusts accordingly. Feature
indices are 1-based to match R conventions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ale.ale_plot import _ale_1d, _ale_2d
from ale.shared import relabel_categorical_features
from ale.ale_vim import (
    _ale_main_vim,
    _ale_interaction_vim,
    _diagnostic_statistic,
    _ale_total_vim
)


class ALE:
    def __init__(self, f, X, feature_names=None, bins=10, categorical=None):
        """
        Initialize the ALE object.

        Parameters:
        - f: The model function that takes a 2D numpy array and returns predictions.
        - X: The input data as a 2D numpy array or pandas DataFrame.
        - feature_names: List of feature names. If None and X is a DataFrame, use its columns.
        - bins: Number of bins for ALE calculation.
        - categorical: List of booleans indicating if each feature is categorical.
                       If None, all features are treated as continuous.
        """
        if not callable(f):
            raise ValueError("f must be a callable function.")
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise ValueError("X must be a 2D numpy array or pandas DataFrame.")
        if categorical is not None and not isinstance(categorical, list):
            raise ValueError("categorical must be a list of booleans or None.")

        self.is_dataframe = isinstance(X, pd.DataFrame)
        self.bins = bins
        self.n, self.d = X.shape

        # takes index of predictor to a dictionary of label to numerical label
        self.label_to_num = {}
        # takes index of predictor to a dictionary of numerical to original label
        self.num_to_label = {}

        if self.is_dataframe:
            # store the DataFrame and its values
            self.X = X.copy()
            self.X_values = X.values

            # if feature_names is None, use DataFrame columns
            if feature_names is None:
                self.feature_names = X.columns.tolist()
            else:
                raise ValueError("If X is a DataFrame, feature_names must be None.")

            # get categorical features from DataFrame
            self.categorical = [X[col].dtype == "category" for col in X.columns]

            # preprocess categorical features
            self._preprocess_categorical_features()
        else:
            self.X_values = X.copy()
            if feature_names is None:
                self.feature_names = [f"X{i+1}" for i in range(self.d)]
            else:
                self.feature_names = feature_names
            self.X = pd.DataFrame(X, columns=self.feature_names)

            if categorical is None:
                self.categorical = [False] * self.d
            else:
                if len(categorical) != X.shape[1]:
                    raise ValueError(
                        "Length of categorical must match number of features in X."
                    )
                self.categorical = categorical

                self._preprocess_categorical_features()

        self.f = self._wrap_convert_function(f)

    def _wrap_convert_function(self, f):
        """
        Wrap the model function to convert categorical features back to original labels.

        Parameters:
        - f: The original model function.

        Returns:
        - A wrapped function that converts categorical features before calling f.
        """

        def wrapped_function(X_input):
            X_converted = X_input.copy()
            for j in range(self.d):
                if self.categorical[j]:
                    # convert numerical labels back to original labels
                    num_to_label_map = self.num_to_label[j]
                    X_converted[:, j] = np.array(
                        [num_to_label_map[int(val)] for val in X_input[:, j]]
                    )

            if self.is_dataframe:
                return f(pd.DataFrame(X_converted, columns=self.feature_names))
            else:
                return f(X_converted)

        return wrapped_function

    def _preprocess_categorical_features(self):
        """
        Preprocess categorical features by relabeling them.
        This method modifies the X attribute in place.
        """
        for j in range(self.d):
            if self.categorical[j]:
                self.X_values[:, j], self.label_to_num[j], self.num_to_label[j] = (
                    relabel_categorical_features(self.X_values, j, self.categorical)
                )

    def _get_feature_index(self, feature):
        """
        Get the 0-based index of a feature given its 1-based index or name.
        """
        if isinstance(feature, int):
            idx = feature - 1
        else:
            idx = self.feature_names.index(feature)

        if idx < 0 or idx >= self.d:
            raise ValueError(
                "Feature index out of bounds, must be between 1 and d inclusive."
            )
        return idx

    def ale_1d(self, feature):
        """
        Plot the 1D ALE for a given feature index (1-based).

        Parameters:
        - feature: 1-based index or feature name.
        """
        idx = self._get_feature_index(feature)

        edges, curve = _ale_1d(
            self.f,
            self.X_values,
            idx + 1,
            bins=self.bins,
            categorical=self.categorical[idx],
        )
        # convert edges to original labels if categorical
        if self.categorical[idx]:
            edges = np.array(
                [
                    self.num_to_label[idx][int(e)] if e in self.num_to_label[idx] else e
                    for e in edges
                ]
            )
        plt.figure()
        if self.categorical[idx]:
            # plot categorical feature as a bar chart
            plt.bar(edges, curve, width=0.5, align='center', alpha=0.7)
            # draw horizontal line at y=0
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        else:
            plt.plot(edges, curve)
            # draw horizontal line at y=0
            plt.hlines(
                y=0,
                xmin=edges.min(),
                xmax=edges.max(),
                color="black",
                linestyle="--",
                alpha=0.5,
            )
            # draw vertical lines from 0 to the value of curve at each edge
            plt.vlines(
                x=edges, ymin=0, ymax=curve, color="red", linestyle="--", alpha=0.5
            )
        plt.xlabel(f"{self.feature_names[idx]}")
        plt.ylabel("ALE (centered)")
        plt.title(f"ALE for {self.feature_names[idx]}")
        plt.show()

    def ale_2d(self, feature_1, feature_2):
        """
        Plot the 2D ALE for a pair of feature indices (1-based).

        Parameters:
        - feature_1: 1-based index or feature name of the first feature.
        - feature_2: 1-based index or feature name of the second feature.
        """
        idx_1 = self._get_feature_index(feature_1)
        idx_2 = self._get_feature_index(feature_2)
        if idx_1 == idx_2:
            raise ValueError("Feature indices must be different for 2D ALE.")

        edges1_interaction, edges2_interaction, curve_interaction = _ale_2d(
            self.f,
            self.X_values,
            idx_1 + 1,
            idx_2 + 1,
            bins=self.bins,
            categorical_1=self.categorical[idx_1],
            categorical_2=self.categorical[idx_2],
        )
        # convert edges to original labels if categorical
        if self.categorical[idx_1]:
            edges1_interaction = np.array(
                [
                    self.num_to_label[idx_1][int(e)]
                    if e in self.num_to_label[idx_1]
                    else e
                    for e in edges1_interaction
                ]
            )
        if self.categorical[idx_2]:
            edges2_interaction = np.array(
                [
                    self.num_to_label[idx_2][int(e)]
                    if e in self.num_to_label[idx_2]
                    else e
                    for e in edges2_interaction
                ]
            )

        plt.figure()
        if self.categorical[idx_1] and self.categorical[idx_2]:
            K = len(edges1_interaction)
            # grouped bar plot
            x = np.arange(K)  # the label locations
            width = 0.25  # the width of the bars
            multiplier = 0
            max_value = np.abs(curve_interaction).max()

            fig, ax = plt.subplots(layout='constrained')

            for i in range(K):
                offset = width * multiplier
                rects = ax.bar(x + offset, curve_interaction[:, i], width, label=edges2_interaction[i])
                ax.bar_label(rects)
                multiplier += 1

            # draw horizontal line at y=0
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel('Interaction ALE (centered)')
            ax.set_xticks(x + width, edges1_interaction)
            ax.set_ylim(-1.1 * max_value, 1.1 * max_value)
            ax.legend()
            ax.set_title(f"ALE Interaction between {self.feature_names[idx_1]} and {self.feature_names[idx_2]}")

        elif self.categorical[idx_1] and not self.categorical[idx_2]:
            # for each level in the first categorical feature
            # plot a separate line for the second feature
            for i, edge1 in enumerate(edges1_interaction):
                plt.plot(edges2_interaction, curve_interaction[i], label=edge1)
            plt.legend(title=self.feature_names[idx_1])
            plt.xlabel(f"{self.feature_names[idx_2]}")
            plt.ylabel("Interaction ALE (centered)")
            plt.title(
                f"ALE Interaction between {self.feature_names[idx_1]} and {self.feature_names[idx_2]}"
            )
        elif not self.categorical[idx_1] and self.categorical[idx_2]:
            # for each level in the second categorical feature
            # plot a separate line for the first feature
            for i, edge2 in enumerate(edges2_interaction):
                plt.plot(edges1_interaction, curve_interaction[:, i], label=edge2)
            plt.legend(title=self.feature_names[idx_2])
            plt.xlabel(f"{self.feature_names[idx_1]}")
            plt.ylabel("Interaction ALE (centered)")
            plt.title(
                f"ALE Interaction between {self.feature_names[idx_1]} and {self.feature_names[idx_2]}"
            )
        else:
            # if both features are continuous, plot a 2D heatmap
            edges1_mesh, edges2_mesh = np.meshgrid(edges1_interaction, edges2_interaction)
            
            plt.pcolormesh(
                edges1_mesh, edges2_mesh, curve_interaction.T, shading="auto", cmap="viridis"
            )
            plt.colorbar(label="Interaction ALE (centered)")
            plt.xlabel(f"{self.feature_names[idx_1]}")
            plt.ylabel(f"{self.feature_names[idx_2]}")
            plt.title(
                f"ALE Interaction between {self.feature_names[idx_1]} and {self.feature_names[idx_2]}"
            )
        plt.show()

    def ale_main_vim(self, feature):
        """
        Calculate the main effect variable importance measure (VIM) for a given feature index (1-based).

        Parameters:
        - feature: 1-based index or feature name.

        Returns:
        - The main effect VIM value.
        """
        idx = self._get_feature_index(feature)

        return _ale_main_vim(
            self.f,
            self.X_values,
            idx + 1,
            bins=self.bins,
            categorical=self.categorical[idx],
        )

    def ale_interaction_vim(self, feature):
        """
        Calculate the interaction effect variable importance measure (VIM) for a given feature index (1-based).

        Parameters:
        - feature: 1-based index or feature name.

        Returns:
        - The interaction effect VIM value.
        """
        idx = self._get_feature_index(feature)

        return _ale_interaction_vim(
            self.f, self.X_values, idx + 1, bins=self.bins, categorical=self.categorical
        )

    def diagnostic_statistic(self):
        """
        Return the R^2 statistic for the second-order ALE model.

        Returns:
        - The R^2 statistic value.
        """
        return _diagnostic_statistic(
            self.f, self.X_values, bins=self.bins, categorical=self.categorical
        )

    def ale_total_vim(self, feature, method="connected"):
        """
        Calculate the total ALE VIM for a given feature index (1-based).

        Parameters:
        - feature: 1-based index or feature name.
        - method: "connected" or "quantile" for path generation.

        Returns:
        - The total ALE VIM value.
        """
        idx = self._get_feature_index(feature)
        if method not in ["connected", "quantile"]:
            raise ValueError("Method must be either 'connected' or 'quantile'.")

        return _ale_total_vim(
                self.f,
                self.X_values,
                idx + 1,
                method=method,
                bins=self.bins,
                categorical=self.categorical,
            )
        
    def explain(self, include=('main', 'total_quantile', 'total_connected')):
        """
        Generate VIM explanations for all features in the dataset.

        Parameters:
        - include: A tuple specifying which explanations to include.
                   Options are 'main', 'total_quantile', 'total_connected', and 'interaction'.
                   Default is ('main', 'total').

        Returns:
        - A pandas DataFrame containing the explanations for each feature.
        """
        if not (set(include) <= set(("main", "total_quantile", "total_connected", "interaction"))):
            raise ValueError("Included explanations must belong to the set (\"main\", \"total_quantile\", \"total_connected\", \"interaction\").")

        explanations = {}
        for i in range(self.d):
            print(f"Calculating explanations for feature {i + 1} ({self.feature_names[i]})")
            explanation_i = {}

            if 'main' in include:
                explanation_i['main'] = np.sqrt(self.ale_main_vim(i + 1))
            if 'total_quantile' in include:
                explanation_i['total_quantile'] = np.sqrt(self.ale_total_vim(i + 1, method='quantile'))
            if 'total_connected' in include:
                explanation_i['total_connected'] = np.sqrt(self.ale_total_vim(i + 1, method='connected'))
            if 'interaction' in include:
                explanation_i['interaction'] = np.sqrt(self.ale_interaction_vim(i + 1))

            explanations[self.feature_names[i]] = explanation_i
        
        df = pd.DataFrame(explanations)
        df.set_index(pd.Index(include), inplace=True)
        return df
