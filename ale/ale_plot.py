import numpy as np

from ale.shared import (
    calculate_bins,
    calculate_edges,
    calculate_deltas,
    calculate_bins_2d,
    calculate_deltas_2d,
    calculate_K,
)


def _ale_1d(f, X, feature_idx, bins, categorical=False):
    """
    Assumes X is a numpy array.
    """
    idx = feature_idx - 1  # convert to 0-based index
    x = X[:, idx]
    n = len(x)

    edges = calculate_edges(x, bins, categorical)
    K = calculate_K(edges, categorical)
    k_x, N_k = calculate_bins(x, edges, categorical)

    # calculate per-observation ALE values
    deltas = calculate_deltas(f, X, idx, edges, k_x)

    average_deltas = np.zeros(K)
    # average deltas for each bin
    for k in range(1, K + 1):
        if N_k[k - 1] > 0:
            average_deltas[k - 1] = (1 / N_k[k - 1]) * np.sum(deltas[k_x == k])

    # accumulate
    accumulated_uncentered = np.pad(average_deltas.cumsum(), (1, 0), mode="constant")

    # center
    if categorical:
        # NOTE: ignore final category since it has no effect
        corrected = accumulated_uncentered[:-1]
        curve = corrected - (1 / n) * np.sum(corrected * N_k)
    else:
        # NOTE: this is a trapezoid rule for integration
        trapezoidal = (1 / 2) * (
            accumulated_uncentered[:-1] + accumulated_uncentered[1:]
        )
        curve = accumulated_uncentered - (1 / n) * np.sum(trapezoidal * N_k)

    return edges, curve


def _ale_2d(
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
    n = len(x1)

    edges_1 = calculate_edges(x1, bins, categorical_1)
    edges_2 = calculate_edges(x2, bins, categorical_2)
    K = calculate_K(edges_1, categorical_1)  # number of bins for first feature
    M = calculate_K(edges_2, categorical_2)  # number of bins for second feature

    # calculate bin for each observation
    k_x, m_x, N_k, N_m, N_km = calculate_bins_2d(
        x1, x2, edges_1, edges_2, categorical_1, categorical_2
    )

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
            else:
                # replace with nearest non-empty bin average
                # TODO: this is a hack, improve later, R uses ann package
                nearest_k = np.argmin(np.abs(np.arange(1, K + 1) - k))
                nearest_m = np.argmin(np.abs(np.arange(1, M + 1) - m))
                average_deltas[k - 1, m - 1] = (
                    average_deltas[nearest_k, m - 1] + average_deltas[k - 1, nearest_m]
                ) / 2
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
    # NOTE: for categorical, the final category has no effect
    if categorical_1 and categorical_2:
        corrected = accumulated_uncentered[:-1, :-1]
        curve = corrected - (1 / n) * (N_km * corrected).sum()
    elif categorical_1 and not categorical_2:
        # NOTE: this is a trapezoid rule for integration
        trapezoidal = (1 / 2) * (
            accumulated_uncentered[:-1, :-1] + accumulated_uncentered[:-1, 1:]
        )
        curve = accumulated_uncentered[:-1, :] - (1 / n) * (trapezoidal * N_km).sum()
    elif not categorical_1 and categorical_2:
        trapezoidal = (1 / 2) * (
            accumulated_uncentered[:-1, :-1] + accumulated_uncentered[1:, :-1]
        )
        curve = accumulated_uncentered[:, :-1] - (1 / n) * (trapezoidal * N_km).sum()
    else:
        trapezoidal = (1 / 4) * (
            accumulated_uncentered[:-1, :-1]
            + accumulated_uncentered[1:, :-1]
            + accumulated_uncentered[:-1, 1:]
            + accumulated_uncentered[1:, 1:]
        )
        curve = accumulated_uncentered - (1 / n) * (trapezoidal * N_km).sum()

    return edges_1, edges_2, curve
