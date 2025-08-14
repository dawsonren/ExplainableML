from typing import Callable, List

import numpy as np


def _precompute_diffs(
    X: np.ndarray, j: int, edges: np.ndarray, f: Callable[[np.ndarray], float]
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
            base[:] = X[k, l]
            base[j] = lower
            # promote to 2D for consistency
            f_low = f(base.reshape(1, -1))
            base[j] = upper
            f_high = f(base.reshape(1, -1))
            diffs[k, l] = f_high[0] - f_low[0]
    return diffs


def _split_leaf(R: List[List[int]], X: np.ndarray, diffs: np.ndarray, j: int):
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

        viable = False  # at least one interval must produce a *real* split

        for k in range(K):
            idx_k = R[k]
            vals = X[k, idx_k, m]
            med = np.median(vals)
            medians[k] = med

            left_mask = vals < med
            if left_mask.all() or (~left_mask).all():  # unsplittable here
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
    if m_star is None:  # *should* be rare: all variables flat
        m_star = candidate_features[0]
        best_medians = np.array([np.median(X[k, R[k], m_star]) for k in range(K)])

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
    f: Callable[[np.ndarray], float], X: np.ndarray, feature_idx: int, edges: np.ndarray
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
    diffs = _precompute_diffs(X, j, edges, f)  # Δ_i,k  (K × L)

    # initial leaf set: every interval holds all its indices
    root_R = [list(range(L)) for _ in range(K)]
    paths: List[np.ndarray] = []

    # --------------- depth-first recursion -------------------------------
    def _recurse(R):
        if all(len(r) == 1 for r in R):  # base case ⇒ a path
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
