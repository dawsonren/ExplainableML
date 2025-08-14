import numpy as np

from algorithms.ale import calculate_bins, calculate_deltas, ale_1d, calculate_edges
from algorithms.tree_partitioning import generate_connected_paths


def ale_global_main(f, X, feature_idx, bins=10, categorical=False):
    idx = feature_idx - 1  # convert to 0-based index
    x = X[:, idx]
    n = len(x)
    edges, curve = ale_1d(f, X, feature_idx, bins=bins, categorical=categorical)

    k_x, _ = calculate_bins(x, edges)

    return (1 / n) * sum([curve[k_x[i] - 1] ** 2 for i in range(n)])


def ale_quantile_total(f, X, feature_idx, bins=10, categorical=False):
    idx = feature_idx - 1  # convert to 0-based index
    x = X[:, idx]
    n = len(x)

    # equal-mass bin edges
    edges = calculate_edges(x, bins, categorical)

    # calculate bin for each observation and observations per bin
    k_x, N_k = calculate_bins(x, edges)
    k_bar = np.clip(int(np.searchsorted(edges, x.mean(), side="right")), 1, bins)
    L = int(np.min(N_k))

    # calculate deltas for each observation
    deltas = calculate_deltas(f, X, idx, edges, k_x)

    # collect deltas along paths ordered by total effect size
    g_values = np.zeros((L, bins))
    for k in range(1, bins + 1):
        for l in range(1, L + 1):
            u = (l - (1 / 2)) / L
            # compute the u-quantile of the deltas for bin k
            g_values[l - 1, k - 1] = np.quantile(deltas[k - 1], u)

    # accumulate
    accumulated_g_values = g_values.cumsum(axis=1)

    centered_g_values = np.zeros_like(accumulated_g_values)
    for l in range(L):
        centered_g_values[l, :] = (
            accumulated_g_values[l, :] - accumulated_g_values[l, k_bar - 1]
        )

    # find variance over paths/observations
    average_g_value = 0
    for k in range(1, bins + 1):
        average_g_value += (N_k[k - 1] / L) * np.sum(centered_g_values[:, k - 1])
    average_g_value /= n

    ale_vim = 0
    for k in range(1, bins + 1):
        ale_vim += (N_k[k - 1] / L) * np.sum(
            (centered_g_values[:, k - 1] - average_g_value) ** 2
        )

    return ale_vim / n


def ale_connected_total(f, X, feature_idx, bins=10, categorical=False):
    n = X.shape[0]
    idx = feature_idx - 1  # convert to 0-based index
    x = X[:, idx]
    K = bins
    L = n // K

    edges = calculate_edges(x, bins, categorical)

    # TODO: handle ragged arrays for categorical features

    # reshape X into (K, L, d)
    reshaped_X = np.zeros((K, L, X.shape[1]))
    for i in range(K):
        mask = (x >= edges[i]) & (x < edges[i + 1]) if i < K - 1 else (x >= edges[i])
        reshaped_X[i] = X[mask][:L]

    # calculate bin for each observation
    k_x, N_k = calculate_bins(reshaped_X[:, :, idx].flatten(), edges)
    k_bar = np.clip(int(np.searchsorted(edges, x.mean(), side="right")), 1, bins)

    paths = generate_connected_paths(f, reshaped_X, feature_idx, edges)
    g_values = np.zeros((L, bins))
    for l, path in enumerate(paths):
        for k in range(K):
            X_left = path[k, :].copy()
            X_right = path[k, :].copy()
            X_left[idx] = edges[k]
            X_right[idx] = edges[k + 1]
            # TODO: handle vectorized function call
            delta = f(X_right.reshape(1, -1)) - f(X_left.reshape(1, -1))
            g_values[l, k] = delta[0]

    # accumulate
    accumulated_g_values = g_values.cumsum(axis=1)

    centered_g_values = np.zeros_like(accumulated_g_values)
    for l in range(L):
        centered_g_values[l, :] = (
            accumulated_g_values[l, :] - accumulated_g_values[l, k_bar - 1]
        )

    # find variance over paths/observations
    average_g_value = 0
    for k in range(1, bins + 1):
        average_g_value += (N_k[k - 1] / L) * np.sum(centered_g_values[:, k - 1])
    average_g_value /= n

    ale_vim = 0
    for k in range(1, bins + 1):
        ale_vim += (N_k[k - 1] / L) * np.sum(
            (centered_g_values[:, k - 1] - average_g_value) ** 2
        )

    return ale_vim / n
