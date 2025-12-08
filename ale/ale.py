"""
Python class that handles ALE explanations.
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
    _ale_total_vim,
    _ale_local_vim,
)
from utils import Explanation, bin_selection


class ALE(Explanation):
    def __init__(
        self,
        f,
        X,
        feature_names=None,
        K=None,
        L=None,
        levels_up=None,
        categorical=None,
        verbose=True,
        interpolate=True,
        centering="x"
    ):
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
        super().__init__(f, X, feature_names, categorical)

        self.K = bin_selection(self.n) if K is None else K
        self.L = self.n // self.K if L is None else L
        self.levels_up = 0 if levels_up is None else levels_up

        if self.K >= self.n / 2:
            print(
                f"Warning: Number of bins ({self.K}) is too large for the dataset size ({self.n}). Consider reducing the number of bins."
            )

        if centering not in ["x", "y"]:
            raise ValueError("centering must be either 'x' or 'y'.")

        self.verbose = verbose
        self.interpolate = interpolate
        self.centering = centering
        # wrap f to handle categorical feature conversion
        self.original_f = f
        self.f = self._wrap_convert_function(self.f)
        # takes index of predictor to a dictionary of label to numerical label
        self.label_to_num = {}
        # takes index of predictor to a dictionary of numerical to original label
        self.num_to_label = {}
        # preprocess categorical features
        self._preprocess_categorical_features()

        # keep track of connected paths from total connected VIM
        self.connected_paths = {}
        # keep track of connected forest from total connected VIM
        self.connected_forest = {}
        # keep track of g-values from total connected VIM
        self.g_values = {}
        # keep track of centered g-values from total connected VIM
        self.centered_g_values = {}
        # keep track of edges, populated by ale_total_vim
        self.edges = {}
        # keep track of path for each observation for each feature
        self.observation_to_path = {}

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
            bins=self.K,
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
            plt.bar(edges, curve, width=0.5, align="center", alpha=0.7)
            # draw horizontal line at y=0
            plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)
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
            bins=self.K,
            categorical_1=self.categorical[idx_1],
            categorical_2=self.categorical[idx_2],
        )
        # convert edges to original labels if categorical
        if self.categorical[idx_1]:
            edges1_interaction = np.array(
                [
                    (
                        self.num_to_label[idx_1][int(e)]
                        if e in self.num_to_label[idx_1]
                        else e
                    )
                    for e in edges1_interaction
                ]
            )
        if self.categorical[idx_2]:
            edges2_interaction = np.array(
                [
                    (
                        self.num_to_label[idx_2][int(e)]
                        if e in self.num_to_label[idx_2]
                        else e
                    )
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

            fig, ax = plt.subplots(layout="constrained")

            for i in range(K):
                offset = width * multiplier
                rects = ax.bar(
                    x + offset,
                    curve_interaction[:, i],
                    width,
                    label=edges2_interaction[i],
                )
                ax.bar_label(rects)
                multiplier += 1

            # draw horizontal line at y=0
            plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel("Interaction ALE (centered)")
            ax.set_xticks(x + width, edges1_interaction)
            ax.set_ylim(-1.1 * max_value, 1.1 * max_value)
            ax.legend()
            ax.set_title(
                f"ALE Interaction between {self.feature_names[idx_1]} and {self.feature_names[idx_2]}"
            )

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
            edges1_mesh, edges2_mesh = np.meshgrid(
                edges1_interaction, edges2_interaction
            )

            plt.pcolormesh(
                edges1_mesh,
                edges2_mesh,
                curve_interaction.T,
                shading="auto",
                cmap="viridis",
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
            bins=self.K,
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
            self.f, self.X_values, idx + 1, self.K, categorical=self.categorical
        )

    def diagnostic_statistic(self):
        """
        Return the R^2 statistic for the second-order ALE model.

        Returns:
        - The R^2 statistic value.
        """
        # TODO: this is wrong!
        # return _diagnostic_statistic(
        #     self.f, self.X_values, self.K, categorical=self.categorical
        # )

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

        total_vim, forest, paths, g_values, centered_g_values, edges, observation_to_path = _ale_total_vim(
            self.f,
            self.X_values,
            idx + 1,
            self.K,
            self.L,
            self.categorical,
            self.label_to_num,
            method=method,
            interpolate=self.interpolate and not self.categorical[idx],
            centering=self.centering
        )
        # store the generated paths for potential reuse
        if method == "connected":
            self.connected_paths[idx] = paths
            self.connected_forest[idx] = forest
            self.g_values[idx] = g_values
            self.centered_g_values[idx] = centered_g_values
            self.edges[idx] = edges
            self.observation_to_path[idx] = observation_to_path
        return total_vim

    def explain(self, include=("main", "total_quantile", "total_connected")):
        """
        Generate VIM explanations for all features in the dataset.

        Parameters:
        - include: A tuple specifying which explanations to include.
                   Options are 'main', 'total_quantile', 'total_connected', and 'interaction'.
                   Default is ('main', 'total_quantile', 'total_connected').

        Returns:
        - A pandas DataFrame containing the explanations for each feature.
        """
        if not (
            set(include)
            <= set(("main", "total_quantile", "total_connected", "interaction"))
        ):
            raise ValueError(
                'Included explanations must belong to the set ("main", "total_quantile", "total_connected", "interaction").'
            )

        explanations = {}
        for i in range(self.d):
            if self.verbose:
                print(
                    f"Calculating explanations for feature {i + 1} ({self.feature_names[i]})"
                )
            explanation_i = {}

            if "main" in include:
                explanation_i["main"] = np.sqrt(self.ale_main_vim(i + 1))
            if "total_quantile" in include:
                explanation_i["total_quantile"] = np.sqrt(
                    self.ale_total_vim(i + 1, method="quantile")
                )
            if "total_connected" in include:
                explanation_i["total_connected"] = np.sqrt(
                    self.ale_total_vim(i + 1, method="connected")
                )
            if "interaction" in include:
                explanation_i["interaction"] = np.sqrt(self.ale_interaction_vim(i + 1))

            explanations[self.feature_names[i]] = explanation_i

        df = pd.DataFrame(explanations)
        df.set_index(pd.Index(include), inplace=True)
        return df

    def explain_local(self, x_explain, method="tree"):
        """
        Produce ALE local variable importances for all features for a given observation.
        """
        # check dimension of x_explain
        if x_explain.ndim != 1 or x_explain.shape[0] != self.d:
            raise ValueError(f"x_explain must be a 1-D array of length {self.d}.")
        # check method
        if method not in ["tree", "nn"]:
            raise ValueError("method must be either 'tree' or 'nn'.")
        # convert x_explain to numpy array if it's a pandas Series
        if isinstance(x_explain, pd.Series):
            x_explain = x_explain.values

        explanations = {}
        for i in range(self.d):
            if self.verbose:
                print(
                    f"Generating local explanation for feature {i + 1} ({self.feature_names[i]})"
                )
            if i not in self.connected_paths or i not in self.connected_forest:
                raise ValueError(
                    f"Connected paths/forest for feature index {i + 1} not found. Please run ale_total_vim with method='connected' first."
                )
            local_vim = _ale_local_vim(
                self.X_values,
                i + 1,
                x_explain,
                self.centered_g_values[i],
                self.K,
                self.categorical,
                self.observation_to_path[i],
                forest=self.connected_forest[i],
                method=method,
                interpolate=self.interpolate and not self.categorical[i],
            )
            explanations[self.feature_names[i]] = local_vim

        return explanations

    def plot_connected_paths(self, feature_1, feature_2):
        """
        Plot the connected paths for a pair of feature indices (1-based).

        Parameters:
        - feature_1: 1-based index or feature name of the examined feature.
        - feature_2: 1-based index or feature name of the secondary feature (plotted on y axis)
        """
        idx_1 = self._get_feature_index(feature_1)
        idx_2 = self._get_feature_index(feature_2)
        if idx_1 == idx_2:
            raise ValueError("Feature indices must be different for connected paths plot.")

        if idx_1 not in self.connected_paths or idx_2 not in self.connected_paths:
            raise ValueError(
                f"Connected paths for feature indices {idx_1 + 1} and/or {idx_2 + 1} not found. Please run ale_total_vim with method='connected' first."
            )

        paths = self.connected_paths[idx_1]
        if not paths:
            raise ValueError(
                f"No connected path found between features {idx_1 + 1} and {idx_2 + 1}."
            )

        X = self.X_values
        # show the connected paths
        plt.scatter(X[:, idx_1], X[:, idx_2], alpha=0.3)
        for path in paths:
            # technically the averaging happens within intervals, but this is fine
            flat_path = [item for interval in path for item in interval]
            plt.plot(X[flat_path, idx_1], X[flat_path, idx_2])

        plt.xlabel(f"{self.feature_names[idx_1]}")
        plt.ylabel(f"{self.feature_names[idx_2]}")
        plt.title(
            f"Connected Paths for {self.feature_names[idx_1]}"
        )

    def plot_ale_ice(self, feature, centered=True):
        idx = self._get_feature_index(feature)
        categorical = self.categorical[idx]

        # check if connected paths and g_values exist
        if idx not in self.connected_paths or idx not in self.connected_forest:
            raise ValueError(
                f"Connected paths/forest for feature index {idx + 1} not found. Please run ale_total_vim with method='connected' first."
            )

        # get centered g_values
        y_axis = self.centered_g_values[idx] if centered else self.g_values[idx]
        edges = self.edges[idx]
        # plot versus feature edges
        for l in range(len(y_axis)):
            # if categorical, plot as bar chart
            if categorical:
                # map back to original category labels
                original_edges = [self.num_to_label[idx][int(e)] for e in edges]
                plt.bar(original_edges, y_axis[l, :], width=0.5, align="center", alpha=0.7)
                # draw horizontal line at y=0
                plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)
            else:
                # plot 
                plt.plot(edges, y_axis[l, :], color="blue", alpha=0.8)
        plt.xlabel(f"{self.feature_names[idx]}")
        plt.ylabel("Centered g-values")
        plt.title(
            f"Centered g-values for {self.feature_names[idx]}"
        )
        # plot horizontal lines at edges
        if not categorical:
            for e in edges:
                plt.axvline(x=e, color="black", linestyle="--", alpha=0.4)
    

# Bootstrap ALE is just like ALE but subsamples the data with replacement
# to create multiple explanations. We hope that this reduces variance in the
# explanations.
class BootstrapALE(Explanation):
    def __init__(
        self,
        f,
        X,
        replications=5,
        feature_names=None,
        K=None,
        L=None,
        levels_up=None,
        categorical=None,
        verbose=True,
        interpolate=True,
        centering="x",
        seed=None
    ):
        """
        Initialize the BootstrapALE object.

        Parameters:
        - f: The model function that takes a 2D numpy array and returns predictions.
        - X: The input data as a 2D numpy array or pandas DataFrame.
        - replications: Number of bootstrap replications.
        - feature_names: List of feature names. If None and X is a DataFrame, use its columns.
        - bins: Number of bins for ALE calculation.
        - categorical: List of booleans indicating if each feature is categorical.
                       If None, all features are treated as continuous.
        """
        super().__init__(f, X, feature_names, categorical)

        self.replications = replications
        self.K = bin_selection(self.n) if K is None else K
        self.L = self.n // self.K if L is None else L
        self.levels_up = 0 if levels_up is None else levels_up

        if self.K >= self.n / 2:
            print(
                f"Warning: Number of bins ({self.K}) is too large for the dataset size ({self.n}). Consider reducing the number of bins."
            )

        if centering not in ["x", "y"]:
            raise ValueError("centering must be either 'x' or 'y'.")

        self.verbose = verbose
        self.interpolate = interpolate
        self.centering = centering
        self.seed = 42 if seed is None else seed

        # create ALE objects for each replication
        if replications > 1:
            self.ale_replications = []
            rng = np.random.default_rng(self.seed)
            for _ in range(self.replications):
                # bootstrap sample with replacement
                indices = rng.choice(self.n, size=self.n, replace=True)
                X_bootstrap = self.X_values[indices, :]
                ale_r = ALE(
                    f,
                    X_bootstrap,
                    feature_names=self.feature_names,
                    K=self.K,
                    L=self.L,
                    levels_up=self.levels_up,
                    categorical=self.categorical,
                    verbose=self.verbose,
                    interpolate=self.interpolate,
                    centering=self.centering
                )
                self.ale_replications.append(ale_r)
        else:
            self.ale_replications = [
                ALE(
                    f,
                    self.X_values,
                    feature_names=self.feature_names,
                    K=self.K,
                    L=self.L,
                    levels_up=self.levels_up,
                    categorical=self.categorical,
                    verbose=self.verbose,
                    interpolate=self.interpolate,
                    centering=self.centering
                )
            ]

    def explain(self, include=("main", "total_quantile", "total_connected")):
        """
        Generate VIM explanations for all features in the dataset using bootstrap replications.

        Parameters:
        - include: A tuple specifying which explanations to include.
                   Options are 'main', 'total_quantile', 'total_connected', and 'interaction'.
                   Default is ('main', 'total_quantile', 'total_connected').

        Returns:
        - A pandas DataFrame containing the averaged explanations for each feature.
        """
        explanations = {}
        for ale_r in self.ale_replications:
            exp_r = ale_r.explain(include=include)
            for feature in exp_r.columns:
                if feature not in explanations:
                    explanations[feature] = {key: [] for key in include}
                for key in include:
                    explanations[feature][key].append(exp_r.loc[key, feature])

        for feature in explanations:
            for key in explanations[feature]:
                explanations[feature][key] = np.mean(explanations[feature][key])

        df = pd.DataFrame(explanations)
        df.set_index(pd.Index(include), inplace=True)
        return df
    
    def explain_local(self, x_explain, method="tree"):
        """
        Produce ALE local variable importances for all features for a given observation
        using bootstrap replications.
        """
        explanations = {}
        for ale_r in self.ale_replications:
            exp_r = ale_r.explain_local(x_explain, method=method)
            for feature in exp_r:
                if feature not in explanations:
                    explanations[feature] = []
                explanations[feature].append(exp_r[feature])

        for feature in explanations:
            explanations[feature] = np.mean(explanations[feature], axis=0)

        return explanations
    
    def plot_ale_ice(self, feature, centered=True):
        """
        Plot the ALE ICE curves for a given feature using bootstrap replications.
        """
        # only plot the first replication for simplicity
        self.ale_replications[0].plot_ale_ice(feature, centered=centered)

    def plot_connected_paths(self, feature_1, feature_2):
        """
        Plot the connected paths for a pair of feature indices (1-based)
        using bootstrap replications.
        """
        # only plot the first replication for simplicity
        self.ale_replications[0].plot_connected_paths(feature_1, feature_2)