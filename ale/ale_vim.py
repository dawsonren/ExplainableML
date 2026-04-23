from dataclasses import dataclass
from typing import Optional

import numpy as np

from ale.shared import (
    calculate_bins,
    calculate_deltas,
    calculate_edges,
    calculate_K,
    linear_interpolation,
    calculate_bin_index,
)
from ale.tree_partitioning import (
    generate_connected_kdforest_and_paths,
    ConnectedKDForest,
)
from utils import generalized_distance


def _knn_smooth_deltas(X, deltas, k_x, K, knn, categorical, std_devs):
    """
    Smooth deltas via KNN averaging within each bin.

    For each bin, compute pairwise distances among all observations in that bin
    (using all features), then replace each observation's delta with the mean
    of its knn nearest neighbors' deltas.

    Parameters:
        X: training data, shape (n, d)
        deltas: raw deltas, shape (n,)
        k_x: 1-indexed bin assignments, shape (n,)
        K: number of bins
        knn: number of nearest neighbors
        categorical: list of booleans for each feature
        std_devs: per-feature standard deviations, shape (d,)

    Returns:
        Smoothed deltas, shape (n,)
    """
    smoothed = deltas.copy()

    for k in range(1, K + 1):
        mask = k_x == k
        if mask.sum() == 0:
            continue

        indices = np.where(mask)[0]
        X_bin = X[indices]
        deltas_bin = deltas[indices]
        n_bin = len(indices)

        # if fewer points than knn, just average all
        k_eff = min(knn, n_bin)

        # compute pairwise distances within the bin
        for i, obs_idx in enumerate(indices):
            dists = generalized_distance(
                X_bin[i], X_bin, categorical, std_devs
            )
            # argsort and take k_eff nearest (includes self at dist=0)
            neighbor_idx = np.argsort(dists)[:k_eff]
            smoothed[obs_idx] = np.mean(deltas_bin[neighbor_idx])

    return smoothed


@dataclass
class GValues:
    """
    The g_values are the accumulated values for each of L paths along K bins.
    g_values has shape (K, L)
    Nkl has shape (K, L) and contains the number of observations in each bin for each path
    centering can be "x" or "y"
    interpolate is a boolean indicating whether to interpolate g_values

    centered_g_values, K and L should not be modified and will be computed
    when get_centered is called
    """
    g_values: np.ndarray
    xj: np.ndarray
    bins: np.ndarray
    k_x: np.ndarray # 1-indexed bin indices for each observation
    l_x: np.ndarray # 0-indexed path indices for each observation
    centering: str
    interpolate: bool
    categorical: bool

    n: Optional[int] = None
    K: Optional[int] = None
    L: Optional[int] = None
    centered_g_values: Optional[np.ndarray] = None

    def __post_init__(self):
        self.xj_mean = self.xj.mean()
        self.K, self.L = self.g_values.shape
        self.n = len(self.xj)

        # compute centered_g_values
        self._compute_centered()

    def _compute_centered(self):
        # centering will be a row vector of length L, broadcast over the
        # (K, L) g_values matrix
        if self.centering == "x":
            # k_bar is 0-based bin index
            k_bar = calculate_bin_index(self.xj_mean, self.bins, self.K, self.categorical)

            if self.interpolate:
                # center so that xj_mean is exactly zero for each of the paths
                centering = linear_interpolation(
                    x=self.xj_mean,
                    x0=self.bins[k_bar],
                    x1=self.bins[k_bar + 1],
                    y0=self.g_values[k_bar - 1, :] if k_bar != 0 else np.zeros(self.L),
                    y1=self.g_values[k_bar, :],
                )
            else:
                # center so that the bin that contains xj_mean is exactly zero
                centering = self.g_values[k_bar, :]
        elif self.centering == "y":
            # center so that the weighted mean g-value across observations is zero on each path
            if self.interpolate:
                padded_g_values = np.pad(self.g_values, ((1, 0), (0, 0)), mode='constant', constant_values=0)
                # n length vector: interpolate g-value at each observation's x_j position
                g_values_per_observation = linear_interpolation(
                    x=self.xj,
                    x0=self.bins[self.k_x],
                    x1=self.bins[self.k_x + 1],
                    y0=padded_g_values[self.k_x, self.l_x],
                    y1=padded_g_values[self.k_x + 1, self.l_x],
                )
                # for each path, compute the average g value
                centering = np.zeros(self.L)
                for l in range(self.L):
                    centering[l] = np.mean(g_values_per_observation[self.l_x == l])
            else:
                g_values_per_observation = self.g_values[self.k_x, self.l_x]
                centering = np.zeros(self.L)
                for l in range(self.L):
                    centering[l] = np.mean(g_values_per_observation[self.l_x == l])

        if self.interpolate:
            self.centered_g_values = np.pad(self.g_values, ((1, 0), (0, 0)), mode='constant', constant_values=0) - centering
        else:
            self.centered_g_values = self.g_values - centering

    def lookup_locals(self, k_idxs, l_idxs, x_explain_j):
        # k_idxs and l_idxs are 0-indexed
        if not self.interpolate:
            return self.centered_g_values[k_idxs, l_idxs]
        else:

            return linear_interpolation(
                x=x_explain_j,
                x0=self.bins[k_idxs],
                x1=self.bins[k_idxs + 1],
                y0=self.centered_g_values[k_idxs, l_idxs],
                y1=self.centered_g_values[k_idxs + 1, l_idxs],
            )

    def plot_centered(self, ax):
        K, L = self.K, self.L
        if self.interpolate:
            for l in range(L):
                ax.plot(self.bins, self.centered_g_values[:, l], color=f"C{l}")
            for bin_edge in self.bins:
                ax.axvline(bin_edge, color='gray', linestyle='--')
        else:
            for l in range(L):
                for k in range(K):
                    ax.hlines(self.centered_g_values[k, l], self.bins[k], self.bins[k + 1], colors=f"C{l}")
            for bin_edge in self.bins:
                ax.axvline(bin_edge, color='gray', linestyle='--')

        ax.set_title(f"Centered g values (centering={self.centering}, interpolate={self.interpolate})")
        return ax


def _ale_main_vim(
    f, X, feature_idx, edges, categorical, interpolate=True, centering="x"
):
    idx = feature_idx - 1  # convert to 0-based index
    x = X[:, idx]
    n = len(x)
    K = calculate_K(edges, categorical)
    k_x, _ = calculate_bins(x, edges, categorical)
    deltas = calculate_deltas(f, X, idx, edges, k_x)
    # get average delta within each bin
    curve = np.zeros(K)
    for k in range(K):
        mask = k_x == k
        if mask.any():
            curve[k] = np.mean(deltas[mask])

    # accumulate
    curve = curve.cumsum()

    # center
    if centering == "x":
        k_bar = calculate_bin_index(x.mean(), edges, K, categorical)
        centered_curve = curve - curve[k_bar]
    elif centering == "y":
        centered_curve = curve - np.mean(curve)

    if interpolate:
        # NOTE: for observations in the last bin, use the value at the left edge
        last_bin = K - 1
        right_k = np.where(k_x == last_bin, k_x, k_x + 1)
        interpolated_values = linear_interpolation(
            x,
            edges[k_x],
            edges[right_k],
            centered_curve[k_x],
            centered_curve[right_k],
        )
        return (1 / n) * np.sum(interpolated_values ** 2)
    else:
        return np.var(centered_curve[k_x])


def __generate_quantile_delta_values(L, K, deltas, k_x):
    paths = np.zeros((K, L))
    for k in range(1, K + 1):
        delta_k = deltas[k_x == k]
        if len(delta_k) == 0:
            continue
        for l in range(1, L + 1):
            u = (l - (1 / 2)) / L
            # compute the u-quantile of the deltas for bin k
            paths[k - 1, l - 1] = np.quantile(delta_k, u)

    return paths


def observation_to_path(paths, n):
    l_x = np.zeros(n, dtype=int)
    for l, path in enumerate(paths):
        for k, interval in enumerate(path):
            for obs in interval:
                l_x[obs] = int(l)
    return l_x


def calculate_g_values(method, X, feature_idx, edges, deltas, categorical, label_to_num, L, K, k_x):
    # collect deltas along paths ordered by total effect size
    if method == "connected":
        forest, paths = generate_connected_kdforest_and_paths(
            X, feature_idx, edges, deltas, categorical, label_to_num, L
        )
        deltas_by_path = np.zeros((K, L))
        # average across paths with multiple elements
        for l, path in enumerate(paths):
            for k, interval in enumerate(path):
                deltas_by_path[k, l] = np.mean(deltas[interval])
    elif method == "quantile":
        paths = None
        forest = None
        deltas_by_path = __generate_quantile_delta_values(L, K, deltas, k_x)
    else:
        raise ValueError(f"Unknown method: {method}")

    return deltas_by_path.cumsum(axis=0), paths, forest

def _ale_total_vim(
    f,
    X,
    feature_idx,
    K,
    L,
    categorical,
    label_to_num,
    method="connected",
    interpolate=True,
    centering="x",
    edges=None,
    knn_smooth=None,
):
    idx = feature_idx - 1  # convert to 0-based index
    x_j = X[:, idx]
    n = X.shape[0]

    if edges is None:
        edges = calculate_edges(x_j, K, categorical[idx])

    K = calculate_K(edges, categorical[idx])

    k_x, _ = calculate_bins(x_j, edges, categorical[idx])
    deltas = calculate_deltas(f, X, idx, edges, k_x)

    if knn_smooth is not None and knn_smooth > 0:
        std_devs = X.std(axis=0)
        deltas = _knn_smooth_deltas(X, deltas, k_x, K, knn_smooth, categorical, std_devs)

    g_values, paths, forest = calculate_g_values(method, X, feature_idx, edges, deltas, categorical, label_to_num, L, K, k_x)

    if method == "connected":
        l_x = observation_to_path(paths, n)

        centered_g_values = GValues(
            g_values, x_j, edges, k_x, l_x, centering, interpolate, categorical[idx]
        )
    elif method == "quantile":
        # Assign each observation to the nearest quantile path based on its delta
        l_x = np.zeros(n, dtype=int)
        for k in range(K):
            mask = k_x == k
            delta_k = deltas[mask]
            if len(delta_k) == 0:
                continue
            indices = np.where(mask)[0]
            quantile_vals = np.array([
                np.quantile(delta_k, (l + 0.5) / L) for l in range(L)
            ])
            # TODO: this is not quite right, __generate_quantile_delta_values should provide
            # you with the appropriate observation_to_path
            for i, obs_idx in enumerate(indices):
                l_x[obs_idx] = int(np.argmin(np.abs(quantile_vals - deltas[obs_idx])))

        centered_g_values = GValues(
            g_values, x_j, edges, k_x, l_x, centering, interpolate, categorical[idx]
        )

    # find variance over paths/observations
    ale_vim = np.var(centered_g_values.lookup_locals(k_x, l_x, x_j))

    return (
        ale_vim,
        forest,
        paths,
        g_values,
        centered_g_values,
        l_x,
        deltas,
        k_x,
    )


def _local_term_self(f, x_explain, idx, x_j_left):
    """f(x_j, x_\\j) - f(x_j_left, x_\\j). Scalar."""
    X_batch = np.tile(x_explain, (2, 1))
    X_batch[0, idx] = x_j_left
    X_batch[1, idx] = x_explain[idx]
    f_vals = f(X_batch)
    return f_vals[1] - f_vals[0]


def _local_term_path_rep(f, X, idx, x_explain, x_j_left, rep_idxs):
    """mean_{x* in rep_idxs} [ f(x_j, x*_\\j) - f(x_j_left, x*_\\j) ]. Scalar."""
    X_left_rep = X[rep_idxs].copy()
    X_point_rep = X[rep_idxs].copy()
    X_left_rep[:, idx] = x_j_left
    X_point_rep[:, idx] = x_explain[idx]
    return float(np.mean(f(X_point_rep) - f(X_left_rep)))


def _ale_local_vim(
    X,
    feature_idx,
    X_explain,
    g_values: GValues,
    categorical,
    observation_to_path,
    forest: ConnectedKDForest,
    levels_up: int = 0,
    edges=None,
    f=None,
    k_x=None,
    local_method="interpolate",
):
    """
    Compute local ALE explanations for one feature.

    local_method:
      - "interpolate" (default): g*(x_j_left) + alpha * Delta_{k,l} via linear
        interpolation of the centered g-value curve.
      - "path_rep": g*(x_j_left) + mean_{x* in path l ∩ bin k}
        [f(x_j, x*_\\j) - f(x_j_left, x*_\\j)]. Requires f and k_x.
      - "self": g*(x_j_left) + [f(x_j, x_\\j) - f(x_j_left, x_\\j)]. Requires f.

    For categorical features or when the fitted g-values are piecewise-constant
    (interpolate=False), local_method is ignored and piecewise lookup is used.
    """
    idx = feature_idx - 1  # convert to 0-based index

    if edges is None:
        raise ValueError("edges must be provided to _ale_local_vim")
    if local_method not in ("interpolate", "path_rep", "self"):
        raise ValueError(
            f"Unknown local_method {local_method!r}; must be one of "
            "'interpolate', 'path_rep', 'self'."
        )

    # short-circuit: categorical or piecewise g-values fall back to piecewise lookup
    effective_method = local_method
    if categorical[idx] or not g_values.interpolate:
        effective_method = "interpolate"

    if effective_method in ("path_rep", "self") and f is None:
        raise ValueError(f"f must be provided for local_method={local_method!r}")
    if effective_method == "path_rep" and k_x is None:
        raise ValueError("k_x must be provided for local_method='path_rep'")

    K = calculate_K(edges, categorical[idx])

    local_effects = np.zeros(X_explain.shape[0])

    for i in range(X_explain.shape[0]):
        x_explain = X_explain[i, :]
        x_idxs = forest.route_and_pick_representative(
            x_explain, X, levels_up=levels_up
        )["indices"]
        k_star = calculate_bin_index(x_explain[idx], edges, K, categorical[idx])
        l_xs = [observation_to_path[x_idx] for x_idx in x_idxs]

        if effective_method == "interpolate":
            # Use k_star (x_explain's bin) for g-value lookup, not the
            # training point's bin. The training point determines the path
            # (l_x), but the g-value should be evaluated at x_explain's
            # position. The forest may return a training point from an
            # adjacent bin due to boundary effects in the KD-tree split.
            effects = [
                g_values.lookup_locals(k_star, l_x, x_explain[idx])
                for l_x in l_xs
            ]
        else:
            x_j_left = edges[k_star]
            # "self" term is path-independent; compute once
            if effective_method == "self":
                term_self = _local_term_self(f, x_explain, idx, x_j_left)

            effects = []
            for l_x in l_xs:
                # base g*(x_j_left): padded centered_g_values[k_star, l_x]
                # (padded with leading zero row, so row k_star = value at edges[k_star])
                base = g_values.centered_g_values[k_star, l_x]

                if effective_method == "self":
                    term = term_self
                else:  # path_rep
                    forest_reps_in_bin = [
                        xi for xi in x_idxs
                        if observation_to_path[xi] == l_x and k_x[xi] == k_star
                    ]
                    if forest_reps_in_bin:
                        rep_idxs = np.asarray(forest_reps_in_bin, dtype=int)
                    else:
                        rep_mask = (observation_to_path == l_x) & (k_x == k_star)
                        rep_idxs = np.where(rep_mask)[0]
                        if len(rep_idxs) == 0:
                            raise ValueError(
                                f"Empty (path l={l_x}, bin k={k_star}) cell for "
                                f"feature {feature_idx}; cannot compute "
                                f"local_method='path_rep'."
                            )
                    term = _local_term_path_rep(
                        f, X, idx, x_explain, x_j_left, rep_idxs
                    )

                effects.append(base + term)

        local_effects[i] = np.mean(effects)

    return local_effects




