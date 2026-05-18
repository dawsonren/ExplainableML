"""
Dispatcher for local ALE explanations: routes to one of the three
local_method implementations.
"""

import numpy as np

from ale.shared import calculate_K, calculate_bin_index
from ale.tree_partitioning import ConnectedKDForest
from ale.vim.g_values import GValues
from ale.vim.local_terms import _local_term_path_rep
from ale.vim.local_path_integral import _ale_local_vim_path_integral
from ale.vim.local_multi_path import _ale_local_vim_multi_path


def _ale_local_vim(
    X,
    idx,
    X_explain,
    g_values: GValues,
    categorical,
    observation_to_path,
    forest: ConnectedKDForest,
    edges=None,
    f=None,
    k_x=None,
    local_method="path_rep",
    deltas=None,
    f_X=None,
    background_indices=None,
    boundary_interp=False,
):
    """
    Compute local ALE explanations for one feature (0-indexed `idx`).

    local_method:
      - "path_rep" (default): g*(x_j_left) + mean_{x* in path l ∩ bin k}
        [f(x_j, x*_\\j) - f(x_j_left, x*_\\j)]. Requires f and k_x.
      - "path_integral": background-averaged path integral; see
        `_ale_local_vim_path_integral`.
      - "multi_path_interpolate": structure-aware multi-path with
        linear depth interpolation; see `_ale_local_vim_multi_path`.

    For categorical features or piecewise (interpolate=False) g-values,
    `path_rep` falls back to centered-g-value lookup (forward differences
    are ill-defined for categoricals).
    """
    if edges is None:
        raise ValueError("edges must be provided to _ale_local_vim")
    if local_method not in ("path_rep", "path_integral", "multi_path_interpolate"):
        raise ValueError(
            f"Unknown local_method {local_method!r}; must be one of "
            "'path_rep', 'path_integral', 'multi_path_interpolate'."
        )

    if local_method == "multi_path_interpolate":
        if deltas is None:
            raise ValueError("deltas must be provided for local_method='multi_path_interpolate'")
        if k_x is None:
            raise ValueError("k_x must be provided for local_method='multi_path_interpolate'")
        return _ale_local_vim_multi_path(
            X, idx, X_explain, deltas, edges, categorical, forest,
            k_x, background_indices=background_indices,
        )

    if local_method == "path_integral":
        if f is None:
            raise ValueError("f must be provided for local_method='path_integral'")
        if deltas is None:
            raise ValueError("deltas must be provided for local_method='path_integral'")
        if k_x is None:
            raise ValueError("k_x must be provided for local_method='path_integral'")
        return _ale_local_vim_path_integral(
            X, idx, X_explain, f, deltas, edges, categorical, forest,
            k_x, f_X=f_X, background_indices=background_indices,
            boundary_interp=boundary_interp,
        )

    # path_rep: categorical or piecewise g-values fall back to direct lookup
    fallback_to_lookup = categorical[idx] or not g_values.interpolate
    if not fallback_to_lookup:
        if f is None:
            raise ValueError("f must be provided for local_method='path_rep'")
        if k_x is None:
            raise ValueError("k_x must be provided for local_method='path_rep'")

    K = calculate_K(edges, categorical[idx])
    local_effects = np.zeros(X_explain.shape[0])

    for i in range(X_explain.shape[0]):
        x_explain = X_explain[i, :]
        x_idxs = forest.route_and_pick_representative(
            x_explain, X
        )["indices"]
        k_star = calculate_bin_index(x_explain[idx], edges, K, categorical[idx])
        l_xs = [observation_to_path[x_idx] for x_idx in x_idxs]

        if fallback_to_lookup:
            effects = [
                g_values.lookup_locals(k_star, l_x, x_explain[idx])
                for l_x in l_xs
            ]
        else:
            x_j_left = edges[k_star]
            effects = []
            for l_x in l_xs:
                base = g_values.centered_g_values[k_star, l_x]
                forest_reps_in_bin = [
                    xi for xi in x_idxs
                    if observation_to_path[xi] == l_x and k_x[xi] == k_star
                ]
                if forest_reps_in_bin:
                    rep_idxs = np.asarray(forest_reps_in_bin, dtype=int)
                else:
                    rep_mask = (observation_to_path == l_x) & (k_x == k_star)
                    rep_idxs = np.where(rep_mask)[0]
                    if len(rep_idxs) == 0:
                        raise ValueError(
                            f"Empty (path l={l_x}, bin k={k_star}) cell for "
                            f"feature {idx}; cannot compute "
                            f"local_method='path_rep'."
                        )
                term = _local_term_path_rep(f, X, idx, x_explain, x_j_left, rep_idxs)
                effects.append(base + term)

        local_effects[i] = np.mean(effects)

    return local_effects
