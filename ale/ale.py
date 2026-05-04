"""
Python class that handles ALE explanations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ale.ale_plot import _ale_1d, _ale_2d
from ale.shared import calculate_edges, calculate_K, relabel_categorical_features
from ale.ale_vim import (
    _ale_main_vim,
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
        categorical=None,
        verbose=True,
        interpolate=True,
        centering="x",
        knn_smooth=None,
        edges=None,
        random_seed=42,
    ):
        """
        Initialize the ALE object.

        Parameters:
        - f: The model function that takes a 2D numpy array and returns predictions.
        - X: The input data as a 2D numpy array or pandas DataFrame.
        - feature_names: List of feature names. If None and X is a DataFrame, use its columns.
        - K: number of bins for the ALE calculation.
        - L: number of paths to create.
        - categorical: list of bool of whether or not each feature is categorical.
        - verbose: whether to print verbose output.
        - interpolate: whether to interpolate the ALE values.
        - centering: how to center the ALE values.
        - knn_smooth: if set, smooth deltas within each bin using KNN averaging.
        - edges: optional dict mapping 1-indexed feature index to edges array.
                 Features not in the dict will have edges calculated as usual.
        - random_seed: seed used by method='random' path generation. Per-feature
                 seed is derived as random_seed + feature_idx so different
                 features get different random partitions but are reproducible.
        """
        super().__init__(f, X, feature_names, categorical, log_queries=False)

        self.K = bin_selection(self.n) if K is None else K
        self.L = self.n // self.K if L is None else L

        if self.K >= self.n / 2:
            print(
                f"Warning: Number of bins ({self.K}) is too large for the dataset size ({self.n}). Consider reducing the number of bins."
            )

        if centering not in ["x", "y"]:
            raise ValueError("centering must be either 'x' or 'y'.")

        self.verbose = verbose
        self.interpolate = interpolate
        self.centering = centering
        self.knn_smooth = knn_smooth
        self.random_seed = random_seed
        # wrap f to handle categorical feature conversion
        self.original_f = f
        self.f = self._wrap_convert_function(self.f)
        # takes index of predictor to a dictionary of label to numerical label
        self.label_to_num = {}
        # takes index of predictor to a dictionary of numerical to original label
        self.num_to_label = {}
        # preprocess categorical features
        self._preprocess_categorical_features()

        # compute edges once per feature from the training data
        self.edges = {}
        edges = edges or {}
        for j in range(self.d):
            if (j + 1) in edges:
                self.edges[j] = edges[j + 1]
            else:
                self.edges[j] = calculate_edges(
                    self.X_values[:, j], self.K, self.categorical[j]
                )
        # actual K per feature (may differ from requested K after dedup)
        self.K_per_feature = {
            j: calculate_K(self.edges[j], self.categorical[j])
            for j in range(self.d)
        }

        # keep track of connected paths from total connected VIM
        self.connected_paths = {}
        # keep track of connected forest from total connected VIM
        self.connected_forest = {}
        # keep track of g-values from total connected VIM
        self.g_values = {}
        # keep track of centered g-values from total connected VIM
        self.centered_g_values = {}
        # keep track of path for each observation for each feature
        self.observation_to_path = {}
        # keep track of raw deltas and bin assignments for each feature
        self.deltas = {}
        self.k_x = {}
        # lazy cache of f(self.X_values) — populated on first path_integral call
        # and shared across features. Invariant: depends only on (f, X), so safe
        # to keep for the lifetime of the ALE object.
        self._f_X_cache = None

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
            edges=self.edges[idx],
            categorical=self.categorical[idx],
            interpolate=self.interpolate and not self.categorical[idx],
            centering=self.centering,
        )

    def ale_total_vim(self, feature, method="connected"):
        """
        Calculate the total ALE VIM for a given feature index (1-based).

        Parameters:
        - feature: 1-based index or feature name.
        - method: "connected", "quantile", or "random" for path generation.

        Returns:
        - The total ALE VIM value.
        """
        idx = self._get_feature_index(feature)
        if method not in ["connected", "quantile", "random"]:
            raise ValueError("Method must be 'connected', 'quantile', or 'random'.")

        seed = None
        if method == "random":
            # per-feature offset so different features get different partitions
            seed = self.random_seed + idx

        (
            total_vim,
            forest,
            paths,
            g_values,
            centered_g_values,
            observation_to_path,
            deltas,
            k_x,
        ) = _ale_total_vim(
            self.f,
            self.X_values,
            idx + 1,
            self.K,
            self.L,
            self.categorical,
            self.label_to_num,
            method=method,
            interpolate=self.interpolate and not self.categorical[idx],
            centering=self.centering,
            edges=self.edges[idx],
            knn_smooth=self.knn_smooth,
            seed=seed,
        )
        # store the generated paths for potential reuse. Both connected and
        # random produce a `paths` structure; random has no forest (local
        # explanations are not supported for random).
        if method in ("connected", "random"):
            self.connected_paths[idx] = paths
            if forest is not None:
                self.connected_forest[idx] = forest
            self.g_values[idx] = g_values
            self.centered_g_values[idx] = centered_g_values
            self.observation_to_path[idx] = observation_to_path
            self.deltas[idx] = deltas
            self.k_x[idx] = k_x
        return total_vim

    def explain(self, include=("main", "total_quantile", "total_connected")):
        """
        Generate VIM explanations for all features in the dataset.

        Parameters:
        - include: A tuple specifying which explanations to include.
                   Options are 'main', 'total_quantile', 'total_connected', 'total_random'.
                   Default is ('main', 'total_quantile', 'total_connected').

        Returns:
        - A pandas DataFrame containing the explanations for each feature.
        """
        allowed = ("main", "total_quantile", "total_connected", "total_random")
        if not set(include) <= set(allowed):
            raise ValueError(
                f"Included explanations must belong to the set {allowed}."
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
            if "total_random" in include:
                explanation_i["total_random"] = np.sqrt(
                    self.ale_total_vim(i + 1, method="random")
                )

            explanations[self.feature_names[i]] = explanation_i

        df = pd.DataFrame(explanations)
        df.set_index(pd.Index(include), inplace=True)
        return df

    def _prepare_explain(self, X_explain):
        """Convert and validate X_explain for local explanation methods."""
        if isinstance(X_explain, pd.Series):
            X_explain = X_explain.values
        if X_explain.ndim == 1:
            X_explain = X_explain[np.newaxis, :]
        for j in range(self.d):
            if j not in self.connected_paths or j not in self.connected_forest:
                raise ValueError(
                    f"Connected paths/forest for feature index {j + 1} not found. "
                    "Please run ale_total_vim with method='connected' first."
                )
        return X_explain

    def _get_f_X(self):
        """Lazily compute and cache f(self.X_values). Reused across features
        and across calls within the lifetime of this ALE object."""
        if self._f_X_cache is None:
            self._f_X_cache = np.asarray(self.f(self.X_values)).ravel()
        return self._f_X_cache

    def explain_local(
        self,
        X_explain,
        levels_up=0,
        local_method="interpolate",
        background_size=None,
        background_seed=None,
        boundary_interp=False,
    ):
        """
        Produce ALE local explanations by routing each point through the
        connected KD-forest and looking up its path's centered g-value.

        Parameters:
        - X_explain: Points to explain (n_explain, d) or (d,).
        - levels_up: Number of levels to go up in the tree for smoothing.
                     0 = exact leaf, higher = average over more paths.
        - local_method: How to compute the within-bin term added to g*(x_j_left):
            - "interpolate" (default): alpha * Delta_{k,l} via linear
              interpolation of the centered g-value curve (no extra f calls).
            - "path_rep": mean over path members in (path l, bin k) of
              [f(x_j, x*_\\j) - f(x_j_left, x*_\\j)]. Costs extra f evals.
              Raises ValueError if the (path, bin) cell is empty.
            - "self": [f(x_j, x_\\j) - f(x_j_left, x_\\j)] using the explain
              point's own off-feature values. Costs extra f evals.
          For categorical features, local_method is ignored and piecewise
          lookup is used.
            - "path_integral": for each background x in self.X_values, walk
              through ALE bins from x_j to x*_j, accumulating partial-bin
              f-differences at the endpoints and pre-computed deltas at
              interpolated middle bins (off-j coords linearly interpolated,
              routed to the nearest training index via the forest). Returns
              the mean over the background. Does not use g_values.
        - background_size: only used by local_method='path_integral'. If set
          and < n, average over a random subsample of size background_size of
          self.X_values rather than the full training set. Reduces cost
          linearly in this size. Forest, deltas, and f(X) cache are still
          built/computed on the full training set.
        - background_seed: RNG seed for the background subsample. Defaults to
          self.random_seed.
        - boundary_interp: only used by local_method='path_integral'. If True,
          replace the partial-bin f-evaluations at x's and x*'s bins with a
          linear-interpolation approximation that scales the routed
          observation's full-bin delta. Eliminates all f-calls during
          explain_local at the cost of some accuracy. Recommended when f is
          expensive (NNs, RFs).
        """
        if local_method not in ("interpolate", "path_rep", "self", "path_integral"):
            raise ValueError(
                f"Unknown local_method {local_method!r}; must be one of "
                "'interpolate', 'path_rep', 'self', 'path_integral'."
            )
        X_explain = self._prepare_explain(X_explain)

        # path_integral-only: pre-compute f(X) cache and background subsample
        f_X = None
        background_indices = None
        if local_method == "path_integral":
            # f(X) cache is only useful when boundary_interp=False (it's only
            # consumed by the f-eval boundary mode); skip the eval otherwise.
            if not boundary_interp:
                f_X = self._get_f_X()
            if background_size is not None and background_size < self.n:
                seed = self.random_seed if background_seed is None else background_seed
                rng = np.random.default_rng(seed)
                background_indices = rng.choice(
                    self.n, size=int(background_size), replace=False
                )

        explanations = np.zeros((X_explain.shape[0], self.d))
        for j in range(self.d):
            if self.verbose:
                print(
                    f"Generating local explanation for feature {j + 1} ({self.feature_names[j]})"
                )
            explanations[:, j] = _ale_local_vim(
                self.X_values,
                j + 1,
                X_explain,
                self.centered_g_values[j],
                self.categorical,
                self.observation_to_path[j],
                forest=self.connected_forest[j],
                levels_up=levels_up,
                edges=self.edges[j],
                f=self.f,
                k_x=self.k_x[j],
                local_method=local_method,
                deltas=self.deltas[j],
                f_X=f_X,
                background_indices=background_indices,
                boundary_interp=boundary_interp,
            )

        return explanations

    def explain_global(self, levels_up=0, local_method="interpolate"):
        """
        The global effect is the sample variance of the local effects across all X values.
        """
        local_explanations = self.explain_local(
            self.X_values, levels_up=levels_up, local_method=local_method
        )
        return np.var(local_explanations, axis=0)

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
            raise ValueError(
                "Feature indices must be different for connected paths plot."
            )

        if idx_1 not in self.connected_paths or idx_2 not in self.connected_paths:
            raise ValueError(
                f"Connected paths for feature indices {idx_1 + 1} and/or {idx_2 + 1} not found. Please run ale_total_vim with method='connected' first."
            )

        paths = self.connected_paths[idx_1]
        if not paths:
            raise ValueError(
                f"No connected path found between features {idx_1 + 1} and {idx_2 + 1}."
            )

        categorical_1 = self.categorical[idx_1]
        edges_1 = self.edges[idx_1]

        if not categorical_1:
            for e in edges_1:
                plt.axvline(x=e, color="black", linestyle="--", alpha=0.4)

        X = self.X_values
        # show the connected paths
        plt.scatter(X[:, idx_1], X[:, idx_2], alpha=0.3)
        for path in paths:
            # technically the averaging happens within intervals, but this is fine
            flat_path = [item for interval in path for item in interval]
            plt.plot(X[flat_path, idx_1], X[flat_path, idx_2])

        plt.xlabel(f"{self.feature_names[idx_1]}")
        plt.ylabel(f"{self.feature_names[idx_2]}")
        plt.title(f"Connected Paths for {self.feature_names[idx_1]}")

    def plot_ale_ice(self, feature):
        idx = self._get_feature_index(feature)
        categorical = self.categorical[idx]

        # check if connected paths and g_values exist
        if idx not in self.connected_paths or idx not in self.connected_forest:
            raise ValueError(
                f"Connected paths/forest for feature index {idx + 1} not found. Please run ale_total_vim with method='connected' first."
            )

        # get centered g_values
        y_axis = self.centered_g_values[idx].centered_g_values
        edges = self.edges[idx]
        # plot versus feature edges
        for l in range(y_axis.shape[1]):
            # if categorical, plot as bar chart
            if categorical:
                # map back to original category labels
                original_edges = [self.num_to_label[idx][int(e)] for e in edges]
                plt.bar(
                    original_edges, y_axis[:, l], width=0.5, align="center", alpha=0.7
                )
                # draw horizontal line at y=0
                plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)
            else:
                # plot
                plt.plot(edges, y_axis[:, l], color="blue", alpha=0.8)
        plt.xlabel(f"{self.feature_names[idx]}")
        plt.ylabel("Centered g-values")
        plt.title(f"Centered g-values for {self.feature_names[idx]}")
        # plot horizontal lines at edges
        if not categorical:
            for e in edges:
                plt.axvline(x=e, color="black", linestyle="--", alpha=0.4)

    def plot_paths_summary(self, feature_1, feature_2, figsize=(21, 5), cmap="tab10"):
        """
        Combined three-subplot visualization linking data partitions to g-value curves.

        Left subplot — scatter of (feature_1, feature_2) with each path's observations
        colored uniquely. Bin centroid trajectories are drawn as thin lines.

        Middle subplot — heatmap of f evaluated across the grid of feature_1 and feature_2,
        with the same path centroids overlaid.

        Right subplot — centered g-value curve for each path vs feature_1 edges,
        using the same per-path colors as the left subplot.

        Parameters
        ----------
        feature_1 : int or str
            Feature whose ALE paths / g-value curves are shown (x-axis in all subplots).
        feature_2 : int or str
            Feature shown on the y-axis of the left scatter subplot.
        figsize : tuple, default (21, 5)
        cmap : str, default "tab10"
            Matplotlib colormap name. Should have at least L distinct colors.

        Returns
        -------
        fig, (ax_data, ax_heatmap, ax_effect)
        """
        idx_1 = self._get_feature_index(feature_1)
        idx_2 = self._get_feature_index(feature_2)

        if idx_1 == idx_2:
            raise ValueError("feature_1 and feature_2 must be different.")
        if idx_1 not in self.connected_paths:
            raise ValueError(
                f"Connected paths for feature {idx_1 + 1} not found. "
                "Run ale_total_vim(method='connected') first."
            )

        paths = self.connected_paths[idx_1]
        edges = self.edges[idx_1]
        g_vals = self.centered_g_values[idx_1].centered_g_values  # (K+1, L) or (K, L)
        L = g_vals.shape[1]

        cmap_obj = plt.get_cmap(cmap)
        colors = [cmap_obj(l / max(L - 1, 1)) for l in range(L)]

        fig, (ax_data, ax_heatmap, ax_effect) = plt.subplots(1, 3, figsize=figsize)

        # --- Left: scatter + path intervals ---
        ax_data.scatter(
            self.X_values[:, idx_1], self.X_values[:, idx_2],
            color="lightgray", alpha=0.3, s=5, zorder=1,
        )
        for l, path in enumerate(paths):
            c = colors[l]
            centroids_x, centroids_y = [], []
            for k, interval in enumerate(path):
                if len(interval) > 0:
                    pts = self.X_values[interval]
                    # double check that all points are within the bin edges
                    assert np.all(pts[:, idx_1] >= edges[k]) and np.all(pts[:, idx_1] <= edges[k + 1]), "Points are not within bin edges"
                    ax_data.scatter(
                        pts[:, idx_1], pts[:, idx_2],
                        color=c, alpha=0.6, s=10, zorder=2,
                    )
                    centroids_x.append(pts[:, idx_1].mean())
                    centroids_y.append(pts[:, idx_2].mean())
            if centroids_x:
                ax_data.plot(centroids_x, centroids_y, color=c, alpha=0.5, linewidth=1)
                ax_heatmap.plot(centroids_x, centroids_y, color=c, alpha=0.8, linewidth=2, marker='o', markersize=3)
        for e in edges:
            ax_data.axvline(e, color="black", linestyle="--", alpha=0.3)
        ax_data.set_xlabel(self.feature_names[idx_1])
        ax_data.set_ylabel(self.feature_names[idx_2])
        ax_data.set_title(
            f"Paths: {self.feature_names[idx_1]} vs {self.feature_names[idx_2]}"
        )

        # --- Middle: Heatmap of f ---
        grid_size = 50
        x1_min, x1_max = self.X_values[:, idx_1].min(), self.X_values[:, idx_1].max()
        x2_min, x2_max = self.X_values[:, idx_2].min(), self.X_values[:, idx_2].max()
        x1_grid = np.linspace(x1_min, x1_max, grid_size)
        x2_grid = np.linspace(x2_min, x2_max, grid_size)
        xx1, xx2 = np.meshgrid(x1_grid, x2_grid)

        X_grid = np.zeros((grid_size * grid_size, self.d))
        for j in range(self.d):
            if self.categorical[j]:
                vals, counts = np.unique(self.X_values[:, j], return_counts=True)
                X_grid[:, j] = vals[np.argmax(counts)]
            else:
                X_grid[:, j] = self.X_values[:, j].mean()
        
        X_grid[:, idx_1] = xx1.ravel()
        X_grid[:, idx_2] = xx2.ravel()
        
        z = self.f(X_grid).reshape(grid_size, grid_size)
        
        mesh = ax_heatmap.pcolormesh(xx1, xx2, z, shading="auto", cmap="viridis", alpha=0.5)
        fig.colorbar(mesh, ax=ax_heatmap, label="f(x)")
        
        for e in edges:
            ax_heatmap.axvline(e, color="black", linestyle="--", alpha=0.3)
        
        ax_heatmap.set_xlabel(self.feature_names[idx_1])
        ax_heatmap.set_ylabel(self.feature_names[idx_2])
        ax_heatmap.set_title(f"Heatmap of {self.f.__name__ if hasattr(self.f, '__name__') else 'model'}")

        # --- Right: g-value curves ---
        for l in range(L):
            ax_effect.plot(edges, g_vals[:, l], color=colors[l], label=f"Path {l}")
        for e in edges:
            ax_effect.axvline(e, color="black", linestyle="--", alpha=0.3)
        ax_effect.axhline(0, color="black", alpha=0.2)
        ax_effect.set_xlabel(self.feature_names[idx_1])
        ax_effect.set_ylabel("Centered g-value")
        ax_effect.set_title(f"G-value curves for {self.feature_names[idx_1]}")
        ax_effect.legend(loc="best", fontsize="small")

        fig.tight_layout()
        return fig, (ax_data, ax_heatmap, ax_effect)


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
        categorical=None,
        verbose=True,
        interpolate=True,
        centering="x",
        knn_smooth=None,
        seed=None,
        random_seed=42,
    ):
        """
        Initialize the BootstrapALE object.

        Parameters:
        - f: The model function that takes a 2D numpy array and returns predictions.
        - X: The input data as a 2D numpy array or pandas DataFrame.
        - replications: Number of bootstrap replications.
        - feature_names: List of feature names. If None and X is a DataFrame, use its columns.
        - K: Number of bins for ALE calculation.
        - L: Number of paths to create.
        - categorical: List of booleans indicating if each feature is categorical.
                       If None, all features are treated as continuous.
        """
        super().__init__(f, X, feature_names, categorical, log_queries=False)

        self.replications = replications
        self.K = bin_selection(self.n) if K is None else K
        self.L = self.n // self.K if L is None else L

        if self.K >= self.n / 2:
            print(
                f"Warning: Number of bins ({self.K}) is too large for the dataset size ({self.n}). Consider reducing the number of bins."
            )

        if centering not in ["x", "y"]:
            raise ValueError("centering must be either 'x' or 'y'.")

        self.verbose = verbose
        self.interpolate = interpolate
        self.centering = centering
        self.knn_smooth = knn_smooth
        self.seed = 42 if seed is None else seed
        self.random_seed = random_seed

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
                    categorical=self.categorical,
                    verbose=self.verbose,
                    interpolate=self.interpolate,
                    centering=self.centering,
                    knn_smooth=self.knn_smooth,
                    random_seed=self.random_seed,
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
                    categorical=self.categorical,
                    verbose=self.verbose,
                    interpolate=self.interpolate,
                    centering=self.centering,
                    knn_smooth=self.knn_smooth,
                    random_seed=self.random_seed,
                )
            ]

    def explain(self, include=("main", "total_quantile", "total_connected")):
        """
        Generate VIM explanations for all features in the dataset using bootstrap replications.

        Parameters:
        - include: A tuple specifying which explanations to include.
                   Options are 'main', 'total_quantile', 'total_connected', 'total_random'.
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

    def explain_local(
        self,
        X_explain,
        levels_up=0,
        local_method="interpolate",
        background_size=None,
        background_seed=None,
        boundary_interp=False,
    ):
        """
        Produce ALE local explanations using bootstrap replications.
        Averages local explanations across replications.
        """
        all_explanations = []
        for ale_r in self.ale_replications:
            exp_r = ale_r.explain_local(
                X_explain,
                levels_up=levels_up,
                local_method=local_method,
                background_size=background_size,
                background_seed=background_seed,
                boundary_interp=boundary_interp,
            )
            all_explanations.append(exp_r)

        return np.mean(all_explanations, axis=0)

