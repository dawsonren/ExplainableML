import numpy as np

from ale.ale_plot import _ale_1d, _ale_2d
from ale.shared import (
    calculate_bins,
    calculate_deltas,
    calculate_edges,
    calculate_bins_2d,
    calculate_K
)
from ale.tree_partitioning import generate_connected_delta_values


def _ale_main_vim(f, X, feature_idx, bins, categorical):
    idx = feature_idx - 1  # convert to 0-based index
    x = X[:, idx]
    n = len(x)
    edges, curve = _ale_1d(f, X, feature_idx, bins=bins, categorical=categorical)

    k_x, _ = calculate_bins(x, edges, categorical)

    return (1 / n) * sum([curve[k_x[i] - 1] ** 2 for i in range(n)])


def _ale_interaction_vim(f, X, feature_idx, bins, categorical):
    # TODO: debug
    idx = feature_idx - 1  # convert to 0-based index
    x = X[:, idx]
    n, d = X.shape
    if categorical is None:
        categorical = [False] * X.shape[1]

    if len(categorical) != d:
        raise ValueError("Length of categorical must match number of features in X.")

    edges_idx = {}
    edges_j = {}
    curve_ij = {}

    for j in set(range(d)) - {idx}:
        edges_idx[j], edges_j[j], curve_ij[j] = _ale_2d(
            f,
            X,
            idx + 1,
            j + 1,
            bins=bins,
            categorical_1=categorical[idx],
            categorical_2=categorical[j],
        )

    k_j_x = {}
    m_j_x = {}
    for j in set(range(d)) - {idx}:
        k_j_x[j], m_j_x[j], _, _, _ = calculate_bins_2d(
            x, X[:, j], edges_idx[j], edges_j[j], categorical_1=categorical[idx], categorical_2=categorical[j]
        )

    edges, curve = _ale_1d(f, X, feature_idx, bins=bins, categorical=categorical[idx])

    k_x, _ = calculate_bins(x, edges, categorical=categorical[idx])

    vim = 0
    for i in range(n):
        importance = curve[k_x[i] - 1]
        importance += sum(
            [
                curve_ij[j][k_j_x[j][i] - 1][m_j_x[j][i] - 1]
                for j in set(range(d)) - {idx}
            ]
        )
        vim += importance**2

    return (1 / n) * vim


def _diagnostic_statistic(f, X, bins, categorical):
    """
    Return the R^2 statistic for the second-order ALE model.
    """
    # TODO: finish and debug
    n, d = X.shape
    if categorical is None:
        categorical = [False] * X.shape[1]

    if len(categorical) != d:
        raise ValueError("Length of categorical must match number of features in X.")

    y = f(X)
    e_fx = np.mean(y)
    var_x = np.var(y)

    edges_j = {}
    first_order_effect = {}
    edges_ij = {}
    second_order_effect = {}

    for j in range(d):
        edges_j[j], first_order_effect[j] = _ale_1d(
            f, X, j + 1, bins=bins, categorical=categorical[j]
        )

    for j in range(d):
        for l in range(j + 1, d):
            edge_j, edge_l, second_order_effect_jl = _ale_2d(
                f,
                X,
                j + 1,
                l + 1,
                bins=bins,
                categorical_1=categorical[j],
                categorical_2=categorical[l],
            )

    second_order_approx = np.zeros(n)
    for i in range(n):
        pass


def generate_quantile_delta_values(L, K, deltas, k_x):
    deltas_by_path = np.zeros((L, K))
    for k in range(1, K + 1):
        delta_k = deltas[k_x == k]
        if len(delta_k) == 0:
            continue
        for l in range(1, L + 1):
            u = (l - (1 / 2)) / L
            # compute the u-quantile of the deltas for bin k
            deltas_by_path[l - 1, k - 1] = np.quantile(delta_k, u)

    return deltas_by_path


def _ale_total_vim(f, X, feature_idx, method="connected", bins=10, categorical=None):
    idx = feature_idx - 1  # convert to 0-based index
    x = X[:, idx]
    n = len(x)

    edges = calculate_edges(x, bins, categorical[idx])
    K = calculate_K(edges, categorical[idx])
    L = n // K

    k_x, N_k = calculate_bins(x, edges, categorical[idx])
    # bin over average x_j value
    k_bar = np.clip(int(np.searchsorted(edges, x.mean(), side="right")), 1, K)
    
    deltas = calculate_deltas(f, X, idx, edges, k_x)

    # collect deltas along paths ordered by total effect size
    deltas_by_path = generate_connected_delta_values(X, feature_idx, edges, deltas, categorical) if method == "connected" else generate_quantile_delta_values(L, K, deltas, k_x)

    # accumulate and center
    g_values = deltas_by_path.cumsum(axis=1)
    centered_g_values = g_values - g_values[:, k_bar - 1][:, None]

    # find variance over paths/observations
    average_g_value = (1 / n) * (centered_g_values * (N_k / L)).sum()
    ale_vim = (1 / n) * ((centered_g_values - average_g_value) ** 2 * (N_k / L)).sum()

    return ale_vim
