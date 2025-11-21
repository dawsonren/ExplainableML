import numpy as np

from ale.ale_plot import _ale_1d, _ale_2d
from ale.shared import (
    calculate_bins,
    calculate_deltas,
    calculate_edges,
    calculate_bins_2d,
    calculate_K,
    linear_interpolation,
    interpolate_g_values,
    find_path_containing_observation,
    find_nearest_neighbor,
    calculate_bin_index,
)
from ale.tree_partitioning import (
    generate_connected_kdforest_and_paths,
    ConnectedKDForest,
)


def __center_g_values(g_values, x, edges, k_bar, K, interpolate, centering):
    if centering == "y":
        centered_g = np.mean(g_values, axis=1)[:, None]
        centered_g_values = g_values - centered_g
    elif centering == "x":
        if interpolate and k_bar < K:
            # assume that g behaves approximately linearly between bin points
            centered_g = linear_interpolation(
                x.mean(),
                edges[k_bar - 1],
                edges[k_bar],
                g_values[:, k_bar - 1],
                g_values[:, k_bar],
            )[:, None]
            centered_g_values = g_values - centered_g
        else:
            # if at last bin, just use last bin value, even if interpolating
            centered_g_values = g_values - g_values[:, k_bar - 1][:, None]

    return centered_g_values


def _ale_main_vim(f, X, feature_idx, bins, categorical, interpolate=True, centering="x"):
    idx = feature_idx - 1  # convert to 0-based index
    x = X[:, idx]
    n = len(x)
    edges, curve = _ale_1d(f, X, feature_idx, bins=bins, categorical=categorical)

    k_x, _ = calculate_bins(x, edges, categorical)
    if interpolate:
        interpolated_values = linear_interpolation(
            x,
            edges[k_x - 1],
            edges[np.where(k_x == calculate_K(edges, categorical), k_x - 1, k_x)],
            curve[k_x - 1],
            # NOTE: for observations in the last bin, use the value at the left edge
            curve[np.where(k_x == calculate_K(edges, categorical), k_x - 1, k_x)],
        )
        return (1 / n) * sum([interpolated_values[i] ** 2 for i in range(n)])
    else:
        return (1 / n) * sum([curve[k_x[i] - 1] ** 2 for i in range(n)])


def _ale_interaction_vim(f, X, feature_idx, K, categorical):
    idx = feature_idx - 1  # convert to 0-based index
    x = X[:, idx]
    n, d = X.shape

    curve_ij = {}
    k_j_x = {}
    m_j_x = {}

    for j in set(range(d)) - {idx}:
        edges_idx_j, edges_j, curve_ij[j] = _ale_2d(
            f,
            X,
            idx + 1,
            j + 1,
            bins=K,
            categorical_1=categorical[idx],
            categorical_2=categorical[j],
        )
        k_j_x[j], m_j_x[j], _, _, _ = calculate_bins_2d(
            x,
            X[:, j],
            edges_idx_j,
            edges_j,
            categorical_1=categorical[idx],
            categorical_2=categorical[j],
        )

    edges, curve = _ale_1d(f, X, feature_idx, bins=K, categorical=categorical[idx])

    k_x, _ = calculate_bins(x, edges, categorical=categorical[idx])

    main_and_interaction = np.zeros(n)
    for i in range(n):
        importance = curve[k_x[i] - 1]
        importance += sum(
            [
                curve_ij[j][k_j_x[j][i] - 1][m_j_x[j][i] - 1]
                for j in set(range(d)) - {idx}
            ]
        )
        main_and_interaction[i] = importance

    return np.var(main_and_interaction)


def _diagnostic_statistic(f, X, K, categorical):
    """
    Return the R^2 statistic for the second-order ALE model.
    """
    n, d = X.shape

    y = f(X)
    e_fx = np.mean(y)
    var_x = np.var(y)

    first_order_effect = {}
    k_j_x = {}
    m_j_x = {}
    l_j_x = {}
    second_order_effect = {}

    for j in range(d):
        edges_j, first_order_effect[j] = _ale_1d(
            f, X, j + 1, bins=K, categorical=categorical[j]
        )
        k_j_x[j], _ = calculate_bins(X[:, j], edges_j, categorical[j])

    for j in range(d):
        for l in range(j + 1, d):
            edge_j, edge_l, second_order_effect[(j, l)] = _ale_2d(
                f,
                X,
                j + 1,
                l + 1,
                bins=K,
                categorical_1=categorical[j],
                categorical_2=categorical[l],
            )
            m_j_x[j], l_j_x[l], _, _, _ = calculate_bins_2d(
                X[:, j],
                X[:, l],
                edge_j,
                edge_l,
                categorical_1=categorical[j],
                categorical_2=categorical[l],
            )

    second_order_approx = np.zeros(n)
    for i in range(n):
        second_order_approx[i] = e_fx
        for j in range(d):
            second_order_approx[i] += first_order_effect[j][k_j_x[j][i] - 1]
            for l in range(j + 1, d):
                second_order_approx[i] += second_order_effect[(j, l)][m_j_x[j][i] - 1][
                    l_j_x[l][i] - 1
                ]

    r_squared = 1 - (np.var(second_order_approx - y)) / var_x
    return r_squared


def __generate_quantile_delta_values(L, K, deltas, k_x):
    paths = np.zeros((L, K))
    for k in range(1, K + 1):
        delta_k = deltas[k_x == k]
        if len(delta_k) == 0:
            continue
        for l in range(1, L + 1):
            u = (l - (1 / 2)) / L
            # compute the u-quantile of the deltas for bin k
            paths[l - 1, k - 1] = np.quantile(delta_k, u)

    return paths


def _ale_total_vim(
    f, X, feature_idx, K, L, categorical, label_to_num, method="connected", interpolate=True, centering="x"
):
    idx = feature_idx - 1  # convert to 0-based index
    x = X[:, idx]
    n = len(x)

    edges = calculate_edges(x, K, categorical[idx])
    K = calculate_K(edges, categorical[idx])

    k_x, N_k = calculate_bins(x, edges, categorical[idx])
    k_bar = np.clip(int(np.searchsorted(edges, x.mean(), side="right")), 1, K)

    deltas = calculate_deltas(f, X, idx, edges, k_x)

    # collect deltas along paths ordered by total effect size
    if method == "connected":
        forest, paths = generate_connected_kdforest_and_paths(
            X, feature_idx, edges, deltas, categorical, label_to_num, L
        )
        deltas_by_path = np.zeros((L, K))
        # average across paths with multiple elements
        for l, path in enumerate(paths):
            for k, interval in enumerate(path):
                deltas_by_path[l, k] = np.mean(deltas[interval])
    elif method == "quantile":
        paths = None
        forest = None
        deltas_by_path = __generate_quantile_delta_values(L, K, deltas, k_x)
    elif method == "nearest_neighbor":
        pass

    # pad zero at beginning of path
    deltas_by_path = np.pad(
        deltas_by_path, ((0, 0), (1, 0)), mode="constant", constant_values=0
    )

    if categorical[idx]:
        # remove zero at end of path
        # NOTE: this is because the last category has a zero delta value
        deltas_by_path = deltas_by_path[:, :-1]

    # accumulate and center
    g_values = deltas_by_path.cumsum(axis=1)
    centered_g_values = __center_g_values(
        g_values, x, edges, k_bar, K, interpolate, centering
    )

    # find variance over paths/observations
    if method == "connected" and interpolate:
        interpolated_centered_g_values = interpolate_g_values(
            edges, paths, k_x, x, centered_g_values, categorical[idx]
        )
        average_g_value = np.mean(interpolated_centered_g_values)
        ale_vim = np.mean(np.power(interpolated_centered_g_values - average_g_value, 2))
    else:
        # treat the g value of each x_j observation as the same if they
        # belong to the same bin
        if not categorical[idx]:
            # pad with zero to match dimensions
            N_k = np.pad(N_k, (1, 0), mode="constant", constant_values=0)
            
        average_g_value = (1 / n) * (centered_g_values * (N_k / L)).sum()
        ale_vim = (1 / n) * (
            (centered_g_values - average_g_value) ** 2 * (N_k / L)
        ).sum()

    # create observation_to_path mapping if using connected method
    observation_to_path = None
    if method == "connected":
        observation_to_path = {}
        for l, path in enumerate(paths):
            for k, interval in enumerate(path):
                for obs in interval:
                    observation_to_path[obs] = l

    return ale_vim, forest, paths, g_values, centered_g_values, edges, observation_to_path


def _ale_local_vim(
    X,
    feature_idx,
    x_explain,
    centered_g_values,
    K,
    categorical,
    observation_to_path,
    forest: ConnectedKDForest,
    levels_up: int = 0,
    method="tree",
    interpolate=True
):
    idx = feature_idx - 1  # convert to 0-based index
    x = X[:, idx]

    edges = calculate_edges(x, K, categorical[idx])
    k_x, _ = calculate_bins(x, edges, categorical[idx])
    K = calculate_K(edges, categorical[idx])
    k_explain = calculate_bin_index(x_explain[idx], edges, K, categorical[idx])

    # find relevant x_idxs (could be multiple)
    if method == "nn":
        _, x_idxs = find_nearest_neighbor(
            x_explain, feature_idx, X[k_x == k_explain, :], categorical, K
        )
    elif method == "tree":
        x_idxs = forest.route_and_pick_representative(x_explain, X, levels_up=levels_up)[
            "indices"
        ]

    # find paths containing observation to explain
    path_idxs = [
        observation_to_path[x_idx] for x_idx in x_idxs if x_idx in observation_to_path
    ]

    effects = []

    for x_idx, path_idx in zip(x_idxs, path_idxs):
        # interpolate along the chosen path, using the value of x_explain
        if interpolate:
            effects.append(
                linear_interpolation(
                    x_explain[idx],
                    edges[k_x[x_idx] - 1],
                    edges[k_x[x_idx]],
                    centered_g_values[path_idx, k_x[x_idx] - 1],
                    centered_g_values[path_idx, k_x[x_idx]],
                )
                if k_x[x_idx] < K
                else centered_g_values[path_idx, k_x[x_idx] - 1]
            )
        else:
            effects.append(centered_g_values[path_idx, k_x[x_idx] - 1])

    return np.mean(effects)
