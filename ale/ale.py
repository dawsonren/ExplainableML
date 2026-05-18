"""
Python class that handles ALE explanations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ale.ale_plot import _ale_1d, _ale_2d
from ale.shared import calculate_edges, calculate_K, relabel_categorical_features
from ale.vim import (
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
        - edges: optional dict mapping 0-indexed feature index to edges array.
                 Features not in the dict will have edges calculated as usual.
        - random_seed: seed used by method='random' path generation. Per-feature
                 seed is derived as random_seed + j so different features get
                 different random partitions but are reproducible.
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
            if j in edges:
                self.edges[j] = edges[j]
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
        Get the 0-based index of a feature given its 0-based index or name.
        """
        if isinstance(feature, int):
            idx = feature
        else:
            idx = self.feature_names.index(feature)

        if idx < 0 or idx >= self.d:
            raise ValueError(
                f"Feature index {idx} out of bounds; must be in [0, {self.d})."
            )
        return idx

    def ale_1d(self, feature):
        """
        Plot the 1D ALE for a given feature index (0-based).

        Parameters:
        - feature: 0-based index or feature name.
        """
        idx = self._get_feature_index(feature)

        edges, curve = _ale_1d(
            self.f,
            self.X_values,
            idx,
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
        Plot the 2D ALE for a pair of feature indices (0-based).

        Parameters:
        - feature_1: 0-based index or feature name of the first feature.
        - feature_2: 0-based index or feature name of the second feature.
        """
        idx_1 = self._get_feature_index(feature_1)
        idx_2 = self._get_feature_index(feature_2)
        if idx_1 == idx_2:
            raise ValueError("Feature indices must be different for 2D ALE.")

        edges1_interaction, edges2_interaction, curve_interaction = _ale_2d(
            self.f,
            self.X_values,
            idx_1,
            idx_2,
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
        Calculate the main effect variable importance measure (VIM) for a given feature index (0-based).

        Parameters:
        - feature: 0-based index or feature name.

        Returns:
        - The main effect VIM value.
        """
        idx = self._get_feature_index(feature)

        return _ale_main_vim(
            self.f,
            self.X_values,
            idx,
            edges=self.edges[idx],
            categorical=self.categorical[idx],
            interpolate=self.interpolate and not self.categorical[idx],
            centering=self.centering,
        )

    def ale_total_vim(self, feature, method="connected"):
        """
        Calculate the total ALE VIM for a given feature index (0-based).

        Parameters:
        - feature: 0-based index or feature name.
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
            idx,
            self.K,
            self.L,
            self.categorical,
            self.label_to_num,
            method=method,
            interpolate=self.interpolate and not self.categorical[idx],
            centering=self.centering,
            edges=self.edges[idx],
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
                    f"Calculating explanations for feature {i} ({self.feature_names[i]})"
                )
            explanation_i = {}

            if "main" in include:
                explanation_i["main"] = np.sqrt(self.ale_main_vim(i))
            if "total_quantile" in include:
                explanation_i["total_quantile"] = np.sqrt(
                    self.ale_total_vim(i, method="quantile")
                )
            if "total_connected" in include:
                explanation_i["total_connected"] = np.sqrt(
                    self.ale_total_vim(i, method="connected")
                )
            if "total_random" in include:
                explanation_i["total_random"] = np.sqrt(
                    self.ale_total_vim(i, method="random")
                )

            explanations[self.feature_names[i]] = explanation_i

        df = pd.DataFrame(explanations)
        df.set_index(pd.Index(include), inplace=True)
        return df

    def check_main_total_invariant(
        self, explanation=None, tol: float = 1e-6, verbose: bool = True
    ) -> bool:
        """
        Verify the theoretical invariant `main <= total_*` per feature.

        The main-effect ALE is a single-feature projection while the total-effect
        ALE collects all interactions involving that feature, so by construction
        main <= total in any decomposition. If this fails, the partition / path
        approximation is biasing the total downward (or the main upward).

        Parameters
        ----------
        explanation : optional pre-computed DataFrame from `.explain()`. If
                      omitted, runs explain() with defaults.
        tol         : numerical slack — violations within tol are ignored.
        verbose     : print per-feature violations.

        Returns True iff no violation exceeds tol.
        """
        if explanation is None:
            explanation = self.explain()
        if "main" not in explanation.index:
            raise ValueError("explanation must include the 'main' row")
        total_rows = [r for r in explanation.index if r.startswith("total_")]
        if not total_rows:
            raise ValueError("explanation must include at least one 'total_*' row")

        main_row = explanation.loc["main"]
        ok = True
        for col in explanation.columns:
            main_val = float(main_row[col])
            for total_name in total_rows:
                total_val = float(explanation.loc[total_name, col])
                diff = main_val - total_val
                if diff > tol:
                    ok = False
                    if verbose:
                        print(
                            f"  [VIOLATION] feature={col!r}: main={main_val:.6f} > "
                            f"{total_name}={total_val:.6f}  (excess={diff:.6f})"
                        )
        if ok and verbose:
            print("  [OK] main <= total_* holds for every feature.")
        return ok

    def print_forest(self, feature, **kwargs) -> str:
        """
        Pretty-print the ConnectedKDForest for a feature. Requires that
        `ale_total_vim(feature, method='connected')` has been called (e.g. via
        `.explain()` with `total_connected` included).

        Extra kwargs are forwarded to `ConnectedKDForest.pretty_print`
        (e.g. max_depth, show_thresholds, show_stats, max_bins_shown).
        """
        idx = self._get_feature_index(feature)
        if idx not in self.connected_forest:
            raise ValueError(
                f"No connected forest stored for feature {feature!r}. "
                "Call ale_total_vim(..., method='connected') or .explain() first."
            )
        kwargs.setdefault("feature_names", list(self.feature_names))
        return self.connected_forest[idx].pretty_print(**kwargs)

    def _prepare_explain(self, X_explain):
        """Convert and validate X_explain for local explanation methods."""
        if isinstance(X_explain, pd.Series):
            X_explain = X_explain.values
        if X_explain.ndim == 1:
            X_explain = X_explain[np.newaxis, :]
        for j in range(self.d):
            if j not in self.connected_paths or j not in self.connected_forest:
                raise ValueError(
                    f"Connected paths/forest for feature index {j} not found. "
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
        local_method="path_rep",
        background_size=None,
        background_seed=None,
        boundary_interp=False,
    ):
        """
        Produce ALE local explanations by routing each point through the
        connected KD-forest and looking up its path's centered g-value.

        Parameters:
        - X_explain: Points to explain (n_explain, d) or (d,).
        - local_method: How to compute the within-bin term added to g*(x_j_left):
            - "path_rep" (default): mean over path members in (path l, bin k)
              of [f(x_j, x*_\\j) - f(x_j_left, x*_\\j)]. Costs extra f evals.
              Raises ValueError if the (path, bin) cell is empty. For
              categorical features the centered g-value curve is looked up
              directly (fresh forward-differences are ill-defined there).
            - "path_integral": for each background x in self.X_values, walk
              through ALE bins from x_j to x*_j, accumulating partial-bin
              f-differences at the endpoints and pre-computed deltas at
              interpolated middle bins (off-j coords linearly interpolated,
              routed to the nearest training index via the forest). Returns
              the mean over the background. Does not use g_values.
            - "multi_path_interpolate": structure-aware path integral. Routes
              x and x* into the connected KD-forest at their respective bins
              to get leaves A and B, then traverses the tree path A → LCA → B
              via depth-interpolation. Each middle bin mixes two adjacent
              interior-path nodes via standard linear interpolation
              (positions p_i = i*(D-1)/(M+1)). Boundary terms reuse
              path_integral's alpha-scaling, with the delta source swapped to
              the relevant leaf's per-bin mean delta. Never calls f during
              explain_local. Does not use g_values.
        - background_size: only used by local_method in {'path_integral',
          'multi_path_interpolate'}. If set and < n, average over a random
          subsample of size background_size of self.X_values rather than the
          full training set. Reduces cost linearly in this size. Forest,
          deltas, and f(X) cache are still built/computed on the full
          training set.
        - background_seed: RNG seed for the background subsample. Defaults to
          self.random_seed.
        - boundary_interp: only used by local_method='path_integral'. If True,
          replace the partial-bin f-evaluations at x's and x*'s bins with a
          linear-interpolation approximation that scales the routed
          observation's full-bin delta. Eliminates all f-calls during
          explain_local at the cost of some accuracy. Recommended when f is
          expensive (NNs, RFs).
        """
        if local_method not in ("path_rep", "path_integral", "multi_path_interpolate"):
            raise ValueError(
                f"Unknown local_method {local_method!r}; must be one of "
                "'path_rep', 'path_integral', 'multi_path_interpolate'."
            )
        X_explain = self._prepare_explain(X_explain)

        # path_integral / multi_path_interpolate: pre-compute f(X) cache
        # (path_integral only, and only when boundary_interp=False) and the
        # background subsample (shared by both).
        f_X = None
        background_indices = None
        if local_method in ("path_integral", "multi_path_interpolate"):
            if local_method == "path_integral" and not boundary_interp:
                # f(X) cache only useful in the f-eval boundary mode.
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
                    f"Generating local explanation for feature {j} ({self.feature_names[j]})"
                )
            explanations[:, j] = _ale_local_vim(
                self.X_values,
                j,
                X_explain,
                self.centered_g_values[j],
                self.categorical,
                self.observation_to_path[j],
                forest=self.connected_forest[j],
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

    def explain_local_weights(
        self, X_explain, local_method="multi_path_interpolate",
        background_size=None, background_seed=None,
    ):
        """
        Return per-observation linear weights for the local explanation.

        For local_method='multi_path_interpolate' the estimator is linear in
        the training deltas. This method returns

            W of shape (n_explain, d, n_train)

        such that for every feature j:
            explain_local(X_explain, local_method='multi_path_interpolate')[i, j]
            == W[i, j] @ self.deltas[j]

        Each `coef * node.mean_delta[k]` term in the estimator distributes
        `coef / node.n_per_bin[k]` across the unique observations in that
        node's subtree at bin k.

        Notes:
        - Computed on demand; can be expensive for large n_train (memory is
          O(n_explain · d · n_train)).
        - If a `background_size` is provided, observations outside the bg
          subsample get weight 0 (they were never routed).
        - Other local_method values are not linear in deltas in the same way
          and raise ValueError.
        """
        from ale.vim import _ale_local_vim_multi_path

        if local_method != "multi_path_interpolate":
            raise ValueError(
                "explain_local_weights only supports 'multi_path_interpolate'; "
                f"got {local_method!r}."
            )
        X_explain = self._prepare_explain(X_explain)

        background_indices = None
        if background_size is not None and background_size < self.n:
            seed = self.random_seed if background_seed is None else background_seed
            rng = np.random.default_rng(seed)
            background_indices = rng.choice(
                self.n, size=int(background_size), replace=False
            )

        n_explain = X_explain.shape[0]
        W = np.zeros((n_explain, self.d, self.n), dtype=float)
        for j in range(self.d):
            if self.verbose:
                print(
                    f"Generating local-weight explanation for feature {j} ({self.feature_names[j]})"
                )
            W[:, j, :] = _ale_local_vim_multi_path(
                self.X_values,
                j,
                X_explain,
                self.deltas[j],
                self.edges[j],
                self.categorical,
                self.connected_forest[j],
                self.k_x[j],
                background_indices=background_indices,
                return_weights=True,
            )
        return W

    def explain_global(self, local_method="path_rep"):
        """
        The global effect is the sample variance of the local effects across all X values.
        """
        local_explanations = self.explain_local(
            self.X_values, local_method=local_method
        )
        return np.var(local_explanations, axis=0)


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
        local_method="path_rep",
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
                local_method=local_method,
                background_size=background_size,
                background_seed=background_seed,
                boundary_interp=boundary_interp,
            )
            all_explanations.append(exp_r)

        return np.mean(all_explanations, axis=0)

