import numpy as np
from sklearn.manifold import MDS
import pandas as pd


def calculate_K(edges, categorical=False):
    if categorical:
        K = len(edges)  # number of unique categories
    else:
        K = len(edges) - 1  # number of bins
    return K


def calculate_edges(x, bins, categorical=False):
    if categorical:
        # set to sorted unique values
        edges = np.sort(np.unique(x))
    else:
        bins = min(
            bins, np.unique(x).size - 1
        )  # ensure bins does not exceed unique values
        # equal-mass bin edges
        edges = np.quantile(x, np.linspace(0, 1, bins + 1))
        edges = np.unique(edges)  # remove duplicates

    return edges


def calculate_bins(x, edges, categorical=False):
    K = calculate_K(edges, categorical)
    n = len(x)
    # calculate bin for each observation
    k_x = np.zeros(n, dtype=int)
    if categorical:
        k_x = np.searchsorted(edges, x) + 1
    else:
        k_x = np.clip(1, np.searchsorted(edges, x, side="right").astype(int), K)

    # calculate observations per bin
    N_k = np.zeros(K)
    for k in range(1, K + 1):
        mask = k_x == k
        N_k[k - 1] = mask.sum()

    return k_x, N_k


def calculate_bins_2d(
    x1, x2, edges_1, edges_2, categorical_1=False, categorical_2=False
):
    K = calculate_K(edges_1, categorical_1)
    M = calculate_K(edges_2, categorical_2)
    n = len(x1)
    # calculate bin for each observation
    k_x = np.zeros(n, dtype=int)
    if categorical_1:
        k_x = np.searchsorted(edges_1, x1) + 1
    else:
        k_x = np.clip(1, np.searchsorted(edges_1, x1, side="right").astype(int), K)

    m_x = np.zeros(n, dtype=int)
    if categorical_2:
        m_x = np.searchsorted(edges_2, x2) + 1
    else:
        m_x = np.clip(1, np.searchsorted(edges_2, x2, side="right").astype(int), M)

    # calculate observations per bin
    N_km = np.zeros((K, M))
    for k in range(1, K + 1):
        for m in range(1, M + 1):
            mask = (k_x == k) & (m_x == m)
            N_km[k - 1, m - 1] = mask.sum()
    N_k = N_km.sum(axis=1)
    N_m = N_km.sum(axis=0)

    return k_x, m_x, N_k, N_m, N_km


def calculate_deltas(f, X, idx, edges, k_x):
    # use vectorization while evaluating f
    X_left = X.copy()
    X_right = X.copy()
    n = X.shape[0]
    for i in range(n):
        # NOTE: skips observations that are outside the edges
        # for categorical, since the final category does
        # not have a right edge.
        if k_x[i] >= len(edges):
            continue
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
        # NOTE: skips observations that are outside the edges
        # for categorical, since the final category does
        # not have a right edge.
        if k_x[i] >= len(edges_1) or m_x[i] >= len(edges_2):
            continue
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


def tv_distance(x, y):
    return 0.5 * np.sum(np.abs(x - y))


def relabel_categorical_features(X, idx, categorical):
    """
    Relabel categorical features using 1-D MDS.

    Parameters:
    - X: The input data as numpy array.
    - idx: the index of the feature to relabel.
    - categorical: list of booleans indicating if the feature is categorical.

    Returns:
    - Relabeled feature values.
    """
    if not categorical[idx]:
        raise ValueError("Value of categorical at idx must be True.")
    if len(categorical) != X.shape[1]:
        raise ValueError("Length of categorical must match number of features in X.")

    mds = MDS(n_components=1, dissimilarity="precomputed", n_init=4, random_state=43)
    # maintains the original order of categories
    original_levels = np.unique(X[:, idx])
    # number of unique items in the feature
    K = len(original_levels)
    # the distance between each category is the
    # sum of distribution distances for all other features
    distance_matrix = np.zeros((K, K))
    n = X.shape[0]

    for j in set(range(X.shape[1])) - {idx}:
        if categorical[j]:
            # TV distance for categorical features
            contingency_table = pd.crosstab(X[:, idx], X[:, j]) / n
            contingency_table = contingency_table.values / n
            for i in range(K):
                for k in range(i + 1, K):
                    dist = tv_distance(contingency_table[i, :], contingency_table[k, :])
                    distance_matrix[i, k] += dist
                    distance_matrix[k, i] += dist
        else:
            # KS distance for continuous features
            x = X[:, j]
            g = X[:, idx]

            # quantiles over the continuous feature
            q_x_all = np.quantile(x, np.linspace(0, 1, 100))
            levels = np.unique(g)
            ecdf_vals = {}
            for lvl in levels:
                # empirical CDF for each category
                xi = x[g == lvl]

                # if no observations for this level, fill with NaN
                if xi.size == 0:
                    ecdf_vals[lvl] = np.full(q_x_all.shape, np.nan, dtype=float)
                    continue

                # sort the values for the empirical CDF
                xi_sorted = np.sort(xi)
                ecdf_vals[lvl] = (
                    np.searchsorted(xi_sorted, q_x_all, side="right") / xi_sorted.size
                )

            for i in range(K):
                for k in range(i + 1, K):
                    vi = ecdf_vals[levels[i]]
                    vk = ecdf_vals[levels[k]]
                    dist = 0
                    if not np.isnan(vi).all() or not np.isnan(vk).all():
                        dist = np.nanmax(np.abs(vi - vk))
                    distance_matrix[i, k] += dist
                    distance_matrix[k, i] += dist

    transformed = mds.fit_transform(distance_matrix)
    original_index = np.argsort(transformed[:, 0])
    new_index = np.argsort(original_index)
    # map from original label to numerical label
    label_to_num = {old: int(new) for old, new in zip(original_levels, new_index)}
    # map from numerical label to original label
    num_to_label = {new: old for old, new in label_to_num.items()}
    # relabel using the mapping
    relabeled = np.vectorize(label_to_num.get)(X[:, idx]).astype(int)

    return relabeled, label_to_num, num_to_label
