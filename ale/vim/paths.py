"""
Path generation and main/total VIM computation.

A "path" is a sequence of (bin -> observation indices) cells used to accumulate
g-values for the L conditional curves. Three generation methods:
  - "connected": KD-forest-based, structure-aware (default)
  - "quantile": per-bin quantile partitioning of deltas
  - "random": random per-bin partitioning (debug/baseline)
"""

import numpy as np

from ale.shared import (
    calculate_bins,
    calculate_deltas,
    calculate_edges,
    calculate_K,
    linear_interpolation,
    calculate_bin_index,
)
from ale.tree_partitioning import generate_connected_kdforest_and_paths
from ale.vim.g_values import GValues


def _ale_main_vim(
    f, X, idx, edges, categorical, interpolate=True, centering="x"
):
    x = X[:, idx]
    K = calculate_K(edges, categorical)
    k_x, _ = calculate_bins(x, edges, categorical)
    deltas = calculate_deltas(f, X, idx, edges, k_x, categorical=categorical)
    curve = np.zeros(K)
    for k in range(K):
        mask = k_x == k
        if mask.any():
            curve[k] = np.mean(deltas[mask])

    curve = curve.cumsum()
    # ALE value at edges[k] is the cumulative through bin k-1 (zero at edges[0]),
    # so prepend 0 to align positions with `edges`. Matches GValues convention.
    padded = np.concatenate(([0.0], curve))  # length K+1; padded[k] is value at edges[k]

    if centering == "x":
        x_mean = x.mean()
        k_bar = calculate_bin_index(x_mean, edges, K, categorical)
        if categorical:
            center = padded[k_bar + 1]
        else:
            center = linear_interpolation(
                x_mean, edges[k_bar], edges[k_bar + 1],
                padded[k_bar], padded[k_bar + 1],
            )
        padded = padded - center
    elif centering == "y":
        padded = padded - padded.mean()

    if interpolate:
        interpolated_values = linear_interpolation(
            x,
            edges[k_x],
            edges[k_x + 1],
            padded[k_x],
            padded[k_x + 1],
        )
        return np.mean(interpolated_values ** 2)
    else:
        return np.var(padded[k_x + 1])


def _generate_quantile_delta_values(L, K, deltas, k_x):
    """Per-bin (l + 0.5)/L quantiles of deltas. k_x is 0-indexed in [0, K-1]."""
    paths = np.zeros((K, L))
    for k in range(K):
        delta_k = deltas[k_x == k]
        if len(delta_k) == 0:
            continue
        for l in range(L):
            paths[k, l] = np.quantile(delta_k, (l + 0.5) / L)
    return paths


def observation_to_path(paths, n):
    l_x = np.zeros(n, dtype=int)
    for l, path in enumerate(paths):
        for k, interval in enumerate(path):
            for obs in interval:
                l_x[obs] = int(l)
    return l_x


def _generate_random_paths(L, K, deltas, k_x, seed):
    """
    Random path generation: within each bin, randomly partition observations
    into L disjoint groups. Path g-value at bin k = mean of deltas in group.
    """
    rng = np.random.default_rng(seed)
    deltas_by_path = np.zeros((K, L))
    paths = [[np.array([], dtype=int) for _ in range(K)] for _ in range(L)]

    for k in range(K):
        idxs = np.where(k_x == k)[0]
        if idxs.size == 0:
            continue
        rng.shuffle(idxs)
        groups = np.array_split(idxs, L)
        for l, group in enumerate(groups):
            if group.size == 0:
                continue
            paths[l][k] = np.asarray(group, dtype=int)
            deltas_by_path[k, l] = float(np.mean(deltas[group]))

    return deltas_by_path, paths


def calculate_g_values(method, X, idx, edges, deltas, categorical, label_to_num, L, K, k_x, seed=None):
    if method == "connected":
        forest, paths = generate_connected_kdforest_and_paths(
            X, idx, edges, deltas, categorical, label_to_num, L
        )
        deltas_by_path = np.zeros((K, L))
        for l, path in enumerate(paths):
            for k, interval in enumerate(path):
                deltas_by_path[k, l] = np.mean(deltas[interval])
    elif method == "quantile":
        paths = None
        forest = None
        deltas_by_path = _generate_quantile_delta_values(L, K, deltas, k_x)
    elif method == "random":
        forest = None
        deltas_by_path, paths = _generate_random_paths(L, K, deltas, k_x, seed)
    else:
        raise ValueError(f"Unknown method: {method}")

    return deltas_by_path.cumsum(axis=0), paths, forest


def _ale_total_vim(
    f,
    X,
    idx,
    K,
    L,
    categorical,
    label_to_num,
    method="connected",
    interpolate=True,
    centering="x",
    edges=None,
    seed=None,
):
    x_j = X[:, idx]
    n = X.shape[0]

    if edges is None:
        edges = calculate_edges(x_j, K, categorical[idx])

    K = calculate_K(edges, categorical[idx])

    k_x, _ = calculate_bins(x_j, edges, categorical[idx])
    deltas = calculate_deltas(f, X, idx, edges, k_x, categorical=categorical[idx])

    g_values, paths, forest = calculate_g_values(
        method, X, idx, edges, deltas, categorical, label_to_num, L, K, k_x, seed=seed
    )

    if method in ("connected", "random"):
        l_x = observation_to_path(paths, n)

        centered_g_values = GValues(
            g_values, x_j, edges, k_x, l_x, centering, interpolate, categorical[idx]
        )
    elif method == "quantile":
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
            for i, obs_idx in enumerate(indices):
                l_x[obs_idx] = int(np.argmin(np.abs(quantile_vals - deltas[obs_idx])))

        centered_g_values = GValues(
            g_values, x_j, edges, k_x, l_x, centering, interpolate, categorical[idx]
        )

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
