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
        edges = np.unique(edges)  # remove duplicates
    return edges


def calculate_bins(x, edges):
    K = len(edges) - 1  # number of bins
    n = len(x)
    # calculate bin for each observation
    k_x = np.zeros(n, dtype=int)
    for i, xi in enumerate(x):
        k_x[i] = np.clip(1, int(np.searchsorted(edges, xi, side="right")), K)

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
        k_x[i] = np.clip(1, int(np.searchsorted(edges_1, xi, side="right")), K)

    m_x = np.zeros(n, dtype=int)
    for i, xi in enumerate(x2):
        m_x[i] = np.clip(1, int(np.searchsorted(edges_2, xi, side="right")), M)

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


def calculate_2d_finite_difference(right_up, left_up, right_down, left_down):
    return (right_up - left_up) - (right_down - left_down)


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
    deltas = calculate_2d_finite_difference(
        f(X_right_up), f(X_left_up), f(X_right_down), f(X_left_down)
    )
    return deltas


def reorder_categories():
    pass
