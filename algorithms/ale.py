import numpy as np
from scipy.stats import ks_2samp
from sklearn.manifold import MDS

def bin_selection(n):
    # choose closest divisor to sqrt(n)
    # list of divisors of n
    divisors = [i for i in range(1, n + 1) if n % i == 0]
    closest_divisor = min(divisors, key=lambda x: abs(x - np.sqrt(n)))
    return closest_divisor 

def calculate_edges(x, bins, categorical=False):
    if categorical:
        # set to sorted unique values
        edges = np.sort(np.unique(x))
        edges = np.append(edges, edges[-1] + np.finfo(np.float16).eps)  # ensure max
    else:
        # equal-mass bin edges
        edges = np.quantile(x, np.linspace(0, 1, bins + 1))
        edges[0], edges[-1] = x.min(), x.max() + np.finfo(np.float16).eps # ensure min/max
    return edges

def calculate_bins(x, edges):
    K = len(edges) - 1  # number of bins
    n = len(x)
    # calculate bin for each observation
    k_x = np.zeros(n, dtype=int)
    for i, xi in enumerate(x):
        k_x[i] = int(np.searchsorted(edges, xi, side='right'))

    # calculate observations per bin
    N_k = np.zeros(K)
    for k in range(1, K + 1):
        mask = k_x == k
        N_k[k - 1] = mask.sum()

    return k_x, N_k

def calculate_bins_2d(x1, x2, edges_1, edges_2):
    K = len(edges_1) - 1  # number of bins for first feature
    M = len(edges_2) - 1  # number of bins for second feature
    n = len(x1)
    # calculate bin for each observation
    k_x = np.zeros(n, dtype=int)
    for i, xi in enumerate(x1):
        k_x[i] = int(np.searchsorted(edges_1, xi, side='right'))

    m_x = np.zeros(n, dtype=int)
    for i, xi in enumerate(x2):
        m_x[i] = int(np.searchsorted(edges_2, xi, side='right'))

    # calculate observations per bin
    N_km = np.zeros((K, M))
    for k in range(1, K + 1):
        for m in range(1, M + 1):
            mask = (k_x == k) & (m_x == m)
            N_km[k - 1, m - 1] = mask.sum()
    N_k = N_km.sum(axis=0)
    N_m = N_km.sum(axis=1)

    return k_x, m_x, N_k, N_m, N_km

def calculate_deltas(f, X, idx, edges, k_x):
    # use vectorization while evaluating f
    X_left = X.copy()
    X_right = X.copy()
    n = X.shape[0]
    for i in range(n):
        X_left[i, idx] = edges[k_x[i] - 1]
        X_right[i, idx] = edges[k_x[i]]
    deltas = f(X_right) - f(X_left)
    return deltas

def calculate_deltas_2d(f, X, idx_1, idx_2, edges_1, edges_2, k_x, m_x):
    X_left_up = X.copy()
    X_right_up = X.copy()
    X_left_down = X.copy()
    X_right_down = X.copy()
    n = X.shape[0]
    for i in range(n):
        X_left_up[i, idx_1] = edges_1[k_x[i] - 1]
        X_left_up[i, idx_2] = edges_2[m_x[i]]
        X_right_up[i, idx_1] = edges_1[k_x[i]]
        X_right_up[i, idx_2] = edges_2[m_x[i]]
        X_left_down[i, idx_1] = edges_1[k_x[i] - 1]
        X_left_down[i, idx_2] = edges_2[m_x[i] - 1]
        X_right_down[i, idx_1] = edges_1[k_x[i]]
        X_right_down[i, idx_2] = edges_2[m_x[i] - 1]
    deltas = (f(X_right_up) - f(X_left_up)) - (f(X_right_down) - f(X_left_down))
    return deltas


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
    accumulated_uncentered = average_deltas.cumsum()

    # average effect
    average_effect = (1 / n) * np.sum(accumulated_uncentered * N_k)

    # center
    curve = accumulated_uncentered - average_effect

    # truncate edges for continuous features
    edges = edges[:-1] if categorical else edges[1:]

    # prepend average effect for categorical features
    # to handle the dummy last bin
    curve = np.insert(curve[:-1], 0, -average_effect) if categorical else curve

    return edges, curve

def ale_2d(f, X, feature_idx_1, feature_idx_2, bins, categorical_1=False, categorical_2=False):
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
                average_deltas[k - 1, m - 1] = (1 / N_km[k - 1, m - 1]) * np.sum(deltas[mask])
    # accumulate
    raw_accumulated = average_deltas.cumsum(axis=0).cumsum(axis=1)
    
    # cancel main effects
    main_effect_1 = np.zeros(K)
    for k in range(1, K + 1):
        mask = (k_x == k)
        if N_k[k - 1] > 0:
            main_effect_1[k - 1] = (1 / N_k[k - 1]) * np.dot(average_deltas[k - 1, :] - average_deltas[k - 1, :], N_km[k - 1, :])
    main_accumulated_1 = main_effect_1.cumsum()

    main_effect_2 = np.zeros(M)
    for m in range(1, M + 1):
        mask = (m_x == m)
        if N_m[m - 1] > 0:
            main_effect_2[m - 1] = (1 / N_m[m - 1]) * np.dot(average_deltas[:, m - 1] - average_deltas[:, m - 1], N_km[:, m - 1])
    main_accumulated_2 = main_effect_2.cumsum()

    # NOTE: uses broadcasting
    accumulated_uncentered = raw_accumulated - main_accumulated_1[:, None] - main_accumulated_2[None, :]

    # center
    curve = accumulated_uncentered - (1 / n) * (N_km * accumulated_uncentered).sum()

    # truncate edges for continuous features
    points_1 = edges_1[:-1] if categorical_1 else edges_1[1:]
    points_2 = edges_2[:-1] if categorical_2 else edges_2[1:]

    return points_1, points_2, curve