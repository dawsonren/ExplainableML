import numpy as np

def ale_1d(f, X, feature_idx, bins=10):
    """
    Compute centered 1-D ALE values for a numpy array X.

    Inputs:
    - f: the model function, takes a 1D array of shape (p,)
    - X: numpy array of shape (n, p)
    - feature_idx: 1-based index of the feature to analyze
    - bins: number of bins for ALE calculation

    Returns:
    - midpoints: bin midpoints
    - curve: ALE curve values at bin midpoints
    """
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

    # calculate observations per bin
    N_k = np.zeros(bins)
    for k in range(1, bins + 1):
        mask = k_x == k
        N_k[k - 1] = mask.sum()

    # calculate per-observation ALE values
    deltas = np.zeros(n)
    for i in range(n):
        X_left = X[i, :].copy()
        X_left[idx] = edges[k_x[i] - 1]
        X_right = X[i, :].copy()
        X_right[idx] = edges[k_x[i]]
        deltas[i] = f(X_right) - f(X_left)

    # accumulate
    average_deltas = np.zeros(bins)
    for k in range(1, bins + 1):
        average_deltas[k - 1] = (1 / N_k[k - 1]) * np.sum(deltas[k_x == k])
    accumulated_uncentered = average_deltas.cumsum()
    
    # append zero to beginning
    accumulated_uncentered = np.pad(accumulated_uncentered, pad_width=(1, 0), mode='constant', constant_values=0)
    curve = accumulated_uncentered - (1 / n) * np.sum(accumulated_uncentered[1:] * N_k)

    return edges, curve

def ale_2d(f, X, feature_idx_1, feature_idx_2, bins=10):
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
    - midpoints: bin midpoints
    - curve: ALE curve values at bin midpoints
    """
    idx_1 = feature_idx_1 - 1  # convert to 0-based index
    idx_2 = feature_idx_2 - 1  # convert to 0-based index
    x1 = X[:, idx_1]
    x2 = X[:, idx_2]
    n = len(x1)

    # equal-mass bin edges for both features
    edges_1 = np.quantile(x1, np.linspace(0, 1, bins + 1))
    edges_2 = np.quantile(x2, np.linspace(0, 1, bins + 1))
    edges_1[0], edges_1[-1] = x1.min(), x1.max() # ensure min/max
    edges_2[0], edges_2[-1] = x2.min(), x2.max() # ensure min/max

    # calculate bin for each observation
    k_x = np.zeros(n, dtype=int)
    for i, xi in enumerate(x1):
        k_x[i] = np.clip(int(np.searchsorted(edges_1, xi, side='right')), 1, bins)

    m_x = np.zeros(n, dtype=int)
    for i, xi in enumerate(x2):
        m_x[i] = np.clip(int(np.searchsorted(edges_2, xi, side='right')), 1, bins)

    # calculate observations per bin
    N_km = np.zeros((bins, bins))
    for k in range(1, bins + 1):
        for m in range(1, bins + 1):
            mask = (k_x == k) & (m_x == m)
            N_km[k - 1, m - 1] = mask.sum()
    N_k = N_km.sum(axis=0)
    N_m = N_km.sum(axis=1)

    # calculate per-observation ALE values
    deltas = np.zeros(n)
    for i in range(n):
        X_left_up = X[i, :].copy()
        X_right_up = X[i, :].copy()
        X_left_down = X[i, :].copy()
        X_right_down = X[i, :].copy()
        X_left_up[idx_1] = edges_1[k_x[i] - 1]
        X_left_up[idx_2] = edges_2[m_x[i]]
        X_right_up[idx_1] = edges_1[k_x[i]]
        X_right_up[idx_2] = edges_2[m_x[i]]
        X_left_down[idx_1] = edges_1[k_x[i] - 1]
        X_left_down[idx_2] = edges_2[m_x[i] - 1]
        X_right_down[idx_1] = edges_1[k_x[i]]
        X_right_down[idx_2] = edges_2[m_x[i] - 1]
        deltas[i] = (f(X_right_up) - f(X_left_up)) - (f(X_right_down) - f(X_left_down))

    # accumulate
    average_deltas = np.zeros((bins, bins))
    for k in range(1, bins + 1):
        for m in range(1, bins + 1):
            mask = (k_x == k) & (m_x == m)
            if N_km[k - 1, m - 1] > 0:
                average_deltas[k - 1, m - 1] = (1 / N_km[k - 1, m - 1]) * np.sum(deltas[mask])
    raw_accumulated = average_deltas.cumsum(axis=0).cumsum(axis=1)
    
    # cancel main effects
    main_effect_1 = np.zeros(bins)
    for k in range(1, bins + 1):
        mask = (k_x == k)
        if N_k[k - 1] > 0:
            main_effect_1[k - 1] = (1 / N_k[k - 1]) * np.dot(average_deltas[k - 1, :] - average_deltas[k - 1, :], N_km[k - 1, :])
    main_accumulated_1 = main_effect_1.cumsum()

    main_effect_2 = np.zeros(bins)
    for m in range(1, bins + 1):
        mask = (m_x == m)
        if N_m[m - 1] > 0:
            main_effect_2[m - 1] = (1 / N_m[m - 1]) * np.dot(average_deltas[:, m - 1] - average_deltas[:, m - 1], N_km[:, m - 1])
    main_accumulated_2 = main_effect_2.cumsum()

    # NOTE: uses broadcasting
    accumulated_uncentered = raw_accumulated - main_accumulated_1[:, None] - main_accumulated_2[None, :]

    # append zeros
    accumulated_uncentered = np.pad(accumulated_uncentered, pad_width=((1, 0), (1, 0)), mode='constant', constant_values=0)
    curve = accumulated_uncentered - (1 / n) * (N_km * accumulated_uncentered[1:, 1:]).sum()

    return edges_1, edges_2, curve