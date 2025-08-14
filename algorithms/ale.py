import numpy as np

from algorithms.shared import (
    calculate_bins,
    calculate_edges,
    calculate_deltas,
    calculate_bins_2d,
    calculate_deltas_2d,
    calculate_2d_finite_difference,
)


def ale_1d(f, X, feature_idx, bins, categorical=False):
    """
    Compute centered 1-D ALE values for a numpy array X.

    Inputs:
    - f: the model function, takes a 1D array of shape (p,)
    - X: numpy array of shape (n, p)
    - feature_idx: 1-based index of the feature to analyze
    - bins: number of bins for ALE calculation

    Returns:
    - edges: x_j values
    - curve: ALE curve values at edges
    """
    idx = feature_idx - 1  # convert to 0-based index
    x = X[:, idx]
    n = len(x)
    K = np.unique(x).size if categorical else bins

    edges = calculate_edges(x, bins, categorical)
    k_x, N_k = calculate_bins(x, edges)

    # calculate per-observation ALE values
    deltas = calculate_deltas(f, X, idx, edges, k_x)

    average_deltas = np.zeros(K)
    # average deltas for each bin
    for k in range(1, K + 1):
        average_deltas[k - 1] = (1 / N_k[k - 1]) * np.sum(deltas[k_x == k])

    # accumulate
    accumulated_uncentered = np.pad(average_deltas.cumsum(), (1, 0), mode="constant")

    # center
    curve = accumulated_uncentered - (1 / n) * np.sum(
        (1 / 2) * (accumulated_uncentered[:-1] + accumulated_uncentered[1:]) * N_k
    )

    return edges, curve


def ale_2d(
    f, X, feature_idx_1, feature_idx_2, bins, categorical_1=False, categorical_2=False
):
    """
    Compute centered 2-D ALE values that visualizes the effects
    of an interaction term.

    Inputs:
    - f: the model function, takes a 1D array of shape (p,)
    - X: numpy array of shape (n, p)
    - feature_idx_1: 1-based index of the first feature
    - feature_idx_2: 1-based index of the second feature
    - bins: number of bins for ALE calculation

    Returns:
    - points_1: x_j values for the first feature
    - points_2: x_l values for the second feature
    - curve: ALE curve values at bin midpoints
    """
    idx_1 = feature_idx_1 - 1  # convert to 0-based index
    idx_2 = feature_idx_2 - 1  # convert to 0-based index
    x1 = X[:, idx_1]
    x2 = X[:, idx_2]
    K = np.unique(x1).size - 1 if categorical_1 else bins
    M = np.unique(x2).size - 1 if categorical_2 else bins
    n = len(x1)

    edges_1 = calculate_edges(x1, bins, categorical_1)
    edges_2 = calculate_edges(x2, bins, categorical_2)

    # calculate bin for each observation
    k_x, m_x, N_k, N_m, N_km = calculate_bins_2d(x1, x2, edges_1, edges_2)

    # calculate per-observation ALE values
    deltas = calculate_deltas_2d(f, X, idx_1, idx_2, edges_1, edges_2, k_x, m_x)

    # average deltas for each bin
    average_deltas = np.zeros((K, M))
    for k in range(1, K + 1):
        for m in range(1, M + 1):
            mask = (k_x == k) & (m_x == m)
            if N_km[k - 1, m - 1] > 0:
                average_deltas[k - 1, m - 1] = (1 / N_km[k - 1, m - 1]) * np.sum(
                    deltas[mask]
                )
    # accumulate
    raw_accumulated = np.pad(
        average_deltas.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)), mode="constant"
    )

    # cancel main effects
    main_effect_1 = np.zeros(K)
    for k in range(1, K + 1):
        mask = k_x == k
        if N_k[k - 1] > 0:
            main_effect_1[k - 1] = (1 / N_k[k - 1]) * np.dot(
                average_deltas[k - 1, :] - average_deltas[k - 1, :], N_km[k - 1, :]
            )
    main_accumulated_1 = np.pad(main_effect_1.cumsum(), (1, 0), mode="constant")

    main_effect_2 = np.zeros(M)
    for m in range(1, M + 1):
        mask = m_x == m
        if N_m[m - 1] > 0:
            main_effect_2[m - 1] = (1 / N_m[m - 1]) * np.dot(
                average_deltas[:, m - 1] - average_deltas[:, m - 1], N_km[:, m - 1]
            )
    main_accumulated_2 = np.pad(main_effect_2.cumsum(), (1, 0), mode="constant")

    # NOTE: uses broadcasting
    accumulated_uncentered = (
        raw_accumulated - main_accumulated_1[:, None] - main_accumulated_2[None, :]
    )

    # center
    curve = (
        accumulated_uncentered
        - (1 / n)
        * (
            N_km
            * calculate_2d_finite_difference(
                accumulated_uncentered[1:, 1:],
                accumulated_uncentered[:-1, 1:],
                accumulated_uncentered[1:, :-1],
                accumulated_uncentered[:-1, :-1],
            )
        ).sum()
    )

    return edges_1, edges_2, curve
