import numpy as np
from typing import Callable, List

from algorithms.ale import calculate_bins, calculate_deltas, ale_1d, calculate_edges

def ale_global_main(f, X, feature_idx, bins=10):
    idx = feature_idx - 1  # convert to 0-based index
    x = X[:, idx]
    n = len(x)
    edges, curve = ale_1d(f, X, feature_idx, bins=bins)

    k_x, _ = calculate_bins(x, edges)

    return (1 / n) * sum([curve[k_x[i] - 1] ** 2 for i in range(n)])


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


def _precompute_diffs(X: np.ndarray,
                      j: int,
                      edges: np.ndarray,
                      f: Callable[[np.ndarray], float]
                     ) -> np.ndarray:
    """
    Pre-compute the local-effect difference

        Δ_i,k = f(x_j = z_{k+1}) - f(x_j = z_k)

    for every observation x = X[k,l,:].

    Parameters
    ----------
    X          : (K, L, d) array
    j          : index of effect variable X_j
    edges  : length K+1 array [z_0, …, z_K] with z_0 < z_1 < … < z_K
    f          : callable mapping 1-D array → scalar
    Returns
    -------
    diffs : (K, L) array with Δ_i,k in the same position as X[k,l,:]
    """
    K, L, d = X.shape
    diffs = np.empty((K, L), dtype=float)

    # copy once outside loops for speed
    base = np.empty(d, dtype=X.dtype)

    for k in range(K):
        lower, upper = edges[k], edges[k + 1]
        for l in range(L):
            base[:] = X[k, l]                       # copy the whole vector
            base[j] = lower
            f_low = f(base)
            base[j] = upper
            f_high = f(base)
            diffs[k, l] = f_high - f_low
    return diffs


def _split_leaf(R: List[List[int]],
                X: np.ndarray,
                diffs: np.ndarray,
                j: int
               ):
    """
    Implements Algorithm 1 for a single *leaf set* (R_1, …, R_K).

    Parameters
    ----------
    R     : length-K list; R[k] is a *list* of indices (into axis-1 of X)
    X     : (K, L, d) data tensor
    diffs : (K, L) pre-computed Δ values
    j     : index of X_j (not a candidate split variable)

    Returns
    -------
    m_star          : int, index of the chosen splitting variable
    medians         : length-K numpy array of medians per interval
    R_left, R_right : the two child leaf sets (same format as R)
    """
    K, _, d = X.shape
    candidate_features = [m for m in range(d) if m != j]

    best_obj = -np.inf
    m_star = None
    best_medians = None
    # --- evaluate every candidate feature -----------------------------
    for m in candidate_features:
        obj = 0.0
        left_means = np.empty(K)
        right_means = np.empty(K)
        medians = np.empty(K)

        viable = False   # at least one interval must produce a *real* split

        for k in range(K):
            idx_k = R[k]
            vals = X[k, idx_k, m]
            med = np.median(vals)
            medians[k] = med

            left_mask = vals < med
            if left_mask.all() or (~left_mask).all():     # unsplittable here
                left_means[k] = right_means[k] = 0.0
                continue

            viable = True
            dl = diffs[k, np.array(idx_k)[left_mask]].mean()
            dr = diffs[k, np.array(idx_k)[~left_mask]].mean()
            left_means[k] = dl
            right_means[k] = dr
            obj += abs(dl - dr)

        if viable and obj > best_obj:
            best_obj = obj
            m_star = m
            best_medians = medians

    # ---------- fall-back -------------------------------------------------
    if m_star is None:                 # *should* be rare: all variables flat
        m_star = candidate_features[0]
        best_medians = np.array(
            [np.median(X[k, R[k], m_star]) for k in range(K)]
        )

    # ---------- create children ------------------------------------------
    R_left, R_right = [], []
    for k in range(K):
        idx_k = np.array(R[k])
        vals = X[k, idx_k, m_star]
        med = best_medians[k]

        left_mask = vals < med
        left_idx = idx_k[left_mask].tolist()
        right_idx = idx_k[~left_mask].tolist()

        # deterministic tie-breaking ― move one obs to the empty side
        if not left_idx:
            left_idx.append(right_idx.pop(0))
        elif not right_idx:
            right_idx.append(left_idx.pop(0))

        R_left.append(left_idx)
        R_right.append(right_idx)

    return m_star, best_medians, R_left, R_right


def generate_connected_paths(
        f: Callable[[np.ndarray], float],
        X: np.ndarray,
        feature_idx: int,
        edges: np.ndarray
    ) -> List[np.ndarray]:
    """
    Full recursive algorithm producing *L* paths (matrices of shape (K, d)).

    Parameters
    ----------
    f           : callable that evaluates the prediction model on 1-D inputs
    X           : (K, L, d) tensor - exactly L obs in every interval
    feature_idx : index of the effect variable X_j (0-based)
    edges       : 1-D array of length K+1 giving the z_{k} boundaries

    Returns
    -------
    paths : list of length-L; each entry is a (K, d) NumPy array
    """
    j = feature_idx - 1  # convert to 0-based index
    K, L, d = X.shape
    diffs = _precompute_diffs(X, j, edges, f)   # Δ_i,k  (K × L)

    # initial leaf set: every interval holds all its indices
    root_R = [list(range(L)) for _ in range(K)]
    paths: List[np.ndarray] = []

    # --------------- depth-first recursion -------------------------------
    def _recurse(R):
        if all(len(r) == 1 for r in R):             # base case ⇒ a path
            path = np.empty((K, d), dtype=X.dtype)
            for k in range(K):
                path[k] = X[k, R[k][0]]
            paths.append(path)
            return

        _, _, R_left, R_right = _split_leaf(R, X, diffs, j)

        # Only recurse on children that still contain ≥1 obs per interval
        if all(len(r) for r in R_left):
            _recurse(R_left)
        if all(len(r) for r in R_right):
            _recurse(R_right)

    _recurse(root_R)
    assert len(paths) == L, "Algorithm should yield exactly L paths"
    return paths


def ale_quantile_total(f, X, feature_idx, bins=10):
    idx = feature_idx - 1  # convert to 0-based index
    x = X[:, idx]
    n = len(x)

    # equal-mass bin edges
    edges = calculate_edges(x, bins)

    # calculate bin for each observation and observations per bin
    k_x, N_k = calculate_bins(x, edges)
    k_bar = np.clip(int(np.searchsorted(edges, x.mean(), side='right')), 1, bins)
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
    n = X.shape[0]
    idx = feature_idx - 1  # convert to 0-based index
    x = X[:, idx]
    K = bins
    L = n // K

    edges = calculate_edges(x, bins)

    # reshape X into (K, L, d)
    reshaped_X = np.zeros((K, L, X.shape[1]))
    print(x)
    for i in range(K):
        print((x >= edges[i]) & (x < edges[i + 1]))
        reshaped_X[i] = X[(x >= edges[i]) & (x < edges[i + 1])][:L]

    # calculate bin for each observation
    k_x, N_k = calculate_bins(reshaped_X[:, :, idx].flatten(), edges)
    k_bar = np.clip(int(np.searchsorted(edges, x.mean(), side='right')), 1, bins)

    paths = generate_connected_paths(f, reshaped_X, feature_idx, edges)
    g_values = np.zeros((L, bins))
    for l, path in enumerate(paths):
        for k in range(K):
            X_left = path[k, :].copy()
            X_right = path[k, :].copy()
            X_left[idx] = edges[k]
            X_right[idx] = edges[k + 1]
            delta = f(X_right) - f(X_left)
            g_values[l, k] = delta

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