import numpy as np

def ale_global_main(edges, curve, X, feature_idx):
    idx = feature_idx - 1  # convert to 0-based index
    x = X[:, idx]
    n = len(x)
    bins = len(edges) - 1

    k_x = np.zeros(n, dtype=int)
    for i, xi in enumerate(x):
        k_x[i] = np.clip(int(np.searchsorted(edges, xi, side='right')), 1, bins)

    return (1 / n) * sum([curve[k_x[i]] ** 2 for i in range(n)])


def generate_connected_paths_2d(X, feature_idx, edges, L):
    # returns the values of x_other that define the paths
    # and the indices of the X observations that define the paths
    K = len(edges) - 1
    idx = feature_idx - 1  # convert to 0-based index
    x_j = X[:, idx]
    # NOTE: assume only 2 features
    x_other = X[:, 1 - idx]

    paths = np.zeros((L, K))
    indices = np.zeros((L, K), dtype=int)
    for i, edge_left, edge_right in zip(range(K), edges[:-1], edges[1:]):
        mask = (x_j >= edge_left) & (x_j < edge_right)
        original_idx = np.where(mask)[0]
        # choose L observations from the mask by the value of x_other
        if mask.sum() > 0:
            idxs = np.argsort(x_other[mask])[:L]
            paths[:, i] = x_other[original_idx[idxs]]
            indices[:, i] = original_idx[idxs]
        else:
            raise ValueError(f"No observations found in bin [{edge_left}, {edge_right})")

    return paths, indices


def ale_quantile_total(f, X, feature_idx, bins=10):
    idx = feature_idx - 1  # convert to 0-based index
    x = X[:, idx]
    n = len(x)

    # equal-mass bin edges
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = x.min(), x.max() + np.finfo(np.float16).eps # ensure min/max

    # calculate bin for each observation
    k_x = np.zeros(n, dtype=int)
    for i, xi in enumerate(x):
        k_x[i] = np.clip(int(np.searchsorted(edges, xi, side='right')), 1, bins)
    k_bar = np.clip(int(np.searchsorted(edges, x.mean(), side='right')), 1, bins)

    # calculate observations per bin
    N_k = np.zeros(bins)
    for k in range(1, bins + 1):
        mask = k_x == k
        N_k[k - 1] = mask.sum()
    L = int(np.min(N_k))

    deltas = []
    for k in range(1, bins + 1):
        delta_k = []
        mask = k_x == k
        for xi in X[mask, :]:
            X_left = xi.copy()
            X_right = xi.copy()
            X_left[idx] = edges[k - 1]
            X_right[idx] = edges[k]
            delta_k.append(f(X_right) - f(X_left))
        deltas.append(delta_k)

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
        centered_g_values[l, :] = accumulated_g_values[l, :] - accumulated_g_values[l, k_bar - 1]

    # find variance over paths/observations
    average_g_value = 0
    for k in range(1, bins + 1):
        average_g_value += (N_k[k - 1] / L) * np.sum(centered_g_values[:, k - 1])
    average_g_value /= n

    ale_vim = 0
    for k in range(1, bins + 1):
        ale_vim += (N_k[k - 1] / L) * np.sum((centered_g_values[:, k - 1] - average_g_value) ** 2)

    return ale_vim / n


def ale_connected_total(f, X, feature_idx, bins=10):
    idx = feature_idx - 1  # convert to 0-based index
    x = X[:, idx]
    p = X.shape[1]
    n = len(x)

    if p != 2:
        raise NotImplementedError("Connected paths are only implemented for p=2 right now.")
    
    # equal-mass bin edges
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = x.min(), x.max() + np.finfo(np.float16).eps# ensure min/max

    # calculate bin for each observation
    k_x = np.zeros(n, dtype=int)
    for i, xi in enumerate(x):
        k_x[i] = np.clip(int(np.searchsorted(edges, xi, side='right')), 1, bins)
    k_bar = np.clip(int(np.searchsorted(edges, x.mean(), side='right')), 1, bins)

    # calculate observations per bin
    N_k = np.zeros(bins)
    for k in range(1, bins + 1):
        mask = k_x == k
        N_k[k - 1] = mask.sum()
    L = int(np.min(N_k))

    # create paths
    paths, indices = generate_connected_paths_2d(X, feature_idx, edges, L)
    g_values = np.zeros((L, bins))
    for l, path in enumerate(paths):
        for m, x_other in enumerate(path):
            i = indices[l, m]
            X_left = np.zeros_like(X[i, :])
            X_right = np.zeros_like(X[i, :])
            X_left[idx] = edges[k_bar - 1]
            X_right[idx] = edges[k_bar]
            # NOTE: Assumes 2 features, so we can just swap the other feature
            X_left[1 - idx] = x_other
            X_right[1 - idx] = x_other
            delta = f(X_right) - f(X_left)
            g_values[l, m] = delta

    # accumulate
    accumulated_g_values = g_values.cumsum(axis=1)

    centered_g_values = np.zeros_like(accumulated_g_values)
    for l in range(L):
        centered_g_values[l, :] = accumulated_g_values[l, :] - accumulated_g_values[l, k_bar - 1]

    # find variance over paths/observations
    average_g_value = 0
    for k in range(1, bins + 1):
        average_g_value += (N_k[k - 1] / L) * np.sum(centered_g_values[:, k - 1])
    average_g_value /= n

    ale_vim = 0
    for k in range(1, bins + 1):
        ale_vim += (N_k[k - 1] / L) * np.sum((centered_g_values[:, k - 1] - average_g_value) ** 2)

    return ale_vim / n


def ale_connected_modified_total(f, X, feature_idx, bins=10):
    idx = feature_idx - 1  # convert to 0-based index
    x = X[:, idx]
    p = X.shape[1]
    n = len(x)

    if p != 2:
        raise NotImplementedError("Connected paths are only implemented for p=2 right now.")
    
    # equal-mass bin edges
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = x.min(), x.max() + np.finfo(np.float16).eps# ensure min/max

    # calculate bin for each observation
    k_x = np.zeros(n, dtype=int)
    for i, xi in enumerate(x):
        k_x[i] =  np.clip(int(np.searchsorted(edges, xi, side='right')), 1, bins)
    k_bar = np.clip(int(np.searchsorted(edges, x.mean(), side='right')), 1, bins)

    # calculate observations per bin
    N_k = np.zeros(bins)
    for k in range(1, bins + 1):
        mask = k_x == k
        N_k[k - 1] = mask.sum()
    L = int(np.min(N_k))

    # create paths
    # NOTE: This means that we each x_i will be used either once or not at all
    paths, indices = generate_connected_paths_2d(X, feature_idx, edges, L)
    g_values = np.zeros((L, bins))
    for l, path in enumerate(paths):
        for m, x_other in enumerate(path):
            i = indices[l, m]
            X_left = np.zeros_like(X[i, :])
            X_right = np.zeros_like(X[i, :])
            X_left[idx] = edges[k_bar - 1]
            X_right[idx] = edges[k_bar]
            # NOTE: Assumes 2 features, so we can just swap the other feature
            X_left[1 - idx] = x_other
            X_right[1 - idx] = x_other
            delta = f(X_right) - f(X_left)
            g_values[l, m] = delta

    # accumulate along a path
    accumulated_g_values = g_values.cumsum(axis=1)

    # center accumulated values with k_bar along each path
    centered_g_values = np.zeros_like(accumulated_g_values)
    for l in range(L):
        centered_g_values[l, :] = accumulated_g_values[l, :] - accumulated_g_values[l, k_bar - 1]

    # find path index for each observation
    path_indices = np.zeros(n, dtype=int)
    for i in range(n):
        # indices is a 2D array, search for the row that contains i
        path_indices[i] = np.where(indices == i)[0][0]

    # find g value for each observation
    g_values_per_observation = np.zeros(n)
    for i in range(n):
        path_idx = path_indices[i]
        bin_idx = k_x[i] - 1
        g_values_per_observation[i] = centered_g_values[path_idx, bin_idx]

    # find variance over observations
    ale_vim = 0
    average_g_value = np.mean(g_values_per_observation)
    for i in range(n):
        ale_vim += (g_values_per_observation[i] - average_g_value) ** 2

    # NOTE: This turns out to be an equivalent implementation to above
    # because each observation is only used once in the paths
    """
    # accumulate along a path
    accumulated_g_values = g_values.cumsum(axis=1)

    # center accumulated values with k_bar along each path
    centered_g_values = np.zeros_like(accumulated_g_values)
    for l in range(L):
        centered_g_values[l, :] = accumulated_g_values[l, :] - accumulated_g_values[l, k_bar - 1]

    # find variance over paths/bins
    ale_vim = 0

    average_g_value = 0
    for k in range(1, bins + 1):
        average_g_value += (N_k[k - 1] / L) * np.sum(centered_g_values[:, k - 1])
    average_g_value /= n

    for k in range(1, bins + 1):
        ale_vim += (N_k[k - 1] / L) * np.sum((centered_g_values[:, k - 1] - average_g_value) ** 2)

    return ale_vim / n
    """

    return ale_vim / n