"""
Shared low-level helpers used by local-method implementations.
"""

import numpy as np


def _local_term_path_rep(f, X, idx, x_explain, x_j_left, rep_idxs):
    """mean_{x* in rep_idxs} [ f(x_j, x*_\\j) - f(x_j_left, x*_\\j) ]. Scalar."""
    X_left_rep = X[rep_idxs].copy()
    X_point_rep = X[rep_idxs].copy()
    X_left_rep[:, idx] = x_j_left
    X_point_rep[:, idx] = x_explain[idx]
    return float(np.mean(f(X_point_rep) - f(X_left_rep)))


def route_first_index_at_bin(forest, x_numeric, k):
    """
    Route an already-numeric point through the forest with a known bin `k`,
    and return the first training index in its leaf for that bin.

    Used by `path_integral`: when we construct an interpolated point z whose
    j-coord we set to a specific bin's anchor, we already know its bin and
    don't need calculate_bin_index. Also bypasses `_convert_x_new` because z
    is in numeric (post-relabel) space.
    """
    j = forest.feature_idx
    node = forest.root
    while not node.is_leaf:
        m = node.split_feature
        thr = node.thresholds[k]
        val = x_numeric[m]
        node = node.left if val < thr else node.right

    indices = node.leaf_indices_for_k(k)
    if indices.size == 0:
        raise ValueError(
            f"path_integral: empty leaf for feature {j}, bin k={k}"
        )
    return int(indices[0])
