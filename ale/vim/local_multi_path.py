"""
Structure-aware multi-path local explanation: `_ale_local_vim_multi_path`.

Walks the connected KD-forest tree path between routed leaves to assemble a
boundary_interp-style local explanation that never calls f. Each middle bin
mixes two adjacent interior-path nodes via standard linear interpolation
(positions p_i = i*(D-1)/(M+1)), corresponding to local_method=
"multi_path_interpolate".
"""

import numpy as np

from ale.shared import calculate_K, calculate_bin_index
from ale.tree_partitioning import iter_subtree_indices_at_bin


def _ale_local_vim_multi_path(
    X, idx, X_explain, deltas, edges, categorical, forest, k_X,
    background_indices=None, return_weights=False,
):
    """
    Structure-aware path-integral local explanation for one feature
    ("multi_path_interpolate"). `idx` is 0-based.

    For each (x* in X_explain, x in X), routes both points into the connected
    KD-forest at their respective bins to get leaves A and B. Boundary
    (partial-bin) contributions reuse path_integral's alpha-scaled formulas
    (`boundary_interp=True` style), but the delta source is swapped from the
    routed-first-index delta to the leaf's per-bin mean delta. Middle bins
    walk the tree path A → LCA → B; each middle bin mixes two adjacent
    interior-path nodes with weights from standard linear interpolation
    (positions p_i = i*(D-1)/(M+1)).

    Always `boundary_interp`-style: this function never calls f.

    Output modes:
      return_weights=False (default): returns local_effects of shape (n_explain,).
      return_weights=True: returns W of shape (n_explain, n_train) such that
        local_effects[i] == W[i] @ deltas. Each `coef * node.mean_delta[k]`
        term distributes `coef / node.n_per_bin[k]` across the unique
        observations in that node's subtree at bin k.

    Amortization: the middle-bin recipe and the leaf-mean deltas depend only
    on (leaf_A, leaf_B, k_bg, k_star), not on the individual x_bg point. So we
    cluster the background by (k_bg, id(leaf_A)) and reduce the inner loop to
    iterate over groups (≤ K·L) rather than individual bg points (n_bg).
    """
    n_explain = X_explain.shape[0]
    cat_j = categorical[idx]
    K = calculate_K(edges, cat_j)
    d = X.shape[1]
    n_train = X.shape[0]

    X_explain_num = np.empty_like(X, shape=(n_explain, d))
    for i in range(n_explain):
        X_explain_num[i] = forest._convert_x_new(X_explain[i])

    k_X = k_X.astype(int)
    k_explain = calculate_bin_index(
        X_explain_num[:, idx], edges, K, cat_j
    ).astype(int)

    if background_indices is not None:
        bg_idx = np.asarray(background_indices, dtype=int)
    else:
        bg_idx = np.arange(n_train, dtype=int)
    X_bg = X[bg_idx]
    k_X_bg = k_X[bg_idx]
    n_bg = X_bg.shape[0]

    groups: list = []
    group_index: dict = {}
    for q in range(n_bg):
        k_bg_q = int(k_X_bg[q])
        info = forest.route_with_path(X_bg[q], k=k_bg_q)
        leaf_A_q = info["leaf"]
        key = (k_bg_q, id(leaf_A_q))
        gi = group_index.get(key)
        if gi is None:
            gi = len(groups)
            group_index[key] = gi
            kb = k_bg_q
            bw_bg = float(edges[kb + 1] - edges[kb]) if kb + 1 < len(edges) else 0.0
            groups.append({
                "k_bg": k_bg_q,
                "leaf_A": leaf_A_q,
                "path_A": info["path"],
                "n_g": 0,
                "sum_xj": 0.0,
                "bw_bg": bw_bg,
            })
        g = groups[gi]
        g["n_g"] += 1
        g["sum_xj"] += float(X_bg[q, idx])

    if return_weights:
        W = np.zeros((n_explain, n_train), dtype=float)
        subtree_cache: dict = {}

        def _obs_at(node, k):
            key = (id(node), k)
            r = subtree_cache.get(key)
            if r is None:
                r = iter_subtree_indices_at_bin(node, k)
                subtree_cache[key] = r
            return r
    else:
        local_effects = np.zeros(n_explain)

    def _apply_term(i, coef, node, k_bin, sign=1.0):
        """Accumulate `sign * coef * node.mean_delta[k_bin]` into either the
        scalar effects or per-observation weights."""
        if coef == 0.0:
            return
        if return_weights:
            n_k = int(node.n_per_bin[k_bin])
            if n_k == 0:
                return
            obs = _obs_at(node, k_bin)
            if obs.size == 0:
                return
            np.add.at(W[i], obs, sign * coef / n_k)
        else:
            v = float(node.mean_delta[k_bin])
            if np.isnan(v):
                return
            nonlocal_total[0] += sign * coef * v

    def _middle_recipe(path, M):
        if M <= 0:
            return []
        return forest.assign_middle_node_weights(path, M)

    for i in range(n_explain):
        x_star = X_explain_num[i]
        k_star = int(k_explain[i])
        x_star_j = float(x_star[idx])

        star_info = forest.route_with_path(x_star, k=k_star)
        path_star = star_info["path"]
        leaf_B = star_info["leaf"]

        bw_xstar = (
            float(edges[k_star + 1] - edges[k_star])
            if k_star + 1 < len(edges) else 0.0
        )

        tree_path_cache: dict = {}
        nonlocal_total = [0.0]

        for g in groups:
            k_bg = g["k_bg"]
            n_g = g["n_g"]
            sum_xj = g["sum_xj"]
            bw_bg = g["bw_bg"]
            leaf_A = g["leaf_A"]

            if k_bg == k_star:
                if bw_xstar > 0:
                    coef = (n_g * x_star_j - sum_xj) / bw_xstar
                    _apply_term(i, coef, leaf_A, k_bg)
                continue

            leaf_A_id = id(leaf_A)
            tree_path = tree_path_cache.get(leaf_A_id)
            if tree_path is None:
                tree_path = forest.tree_path_between(g["path_A"], path_star)
                tree_path_cache[leaf_A_id] = tree_path

            if k_bg < k_star:
                if bw_bg > 0:
                    coef_L = (edges[k_bg + 1] * n_g - sum_xj) / bw_bg
                    _apply_term(i, coef_L, leaf_A, k_bg)
                alpha_R = (x_star_j - edges[k_star]) / bw_xstar if bw_xstar > 0 else 0.0
                _apply_term(i, n_g * alpha_R, leaf_B, k_star)

                M = k_star - k_bg - 1
                if M > 0:
                    recipe = _middle_recipe(tree_path, M)
                    for off, pairs in enumerate(recipe):
                        k_mid = k_bg + 1 + off
                        for (node, w) in pairs:
                            if w == 0.0:
                                continue
                            _apply_term(i, n_g * w, node, k_mid)
            else:  # k_bg > k_star
                k_bg_plus = min(k_bg + 1, len(edges) - 1)
                bw_bg_gt = (
                    float(edges[k_bg_plus] - edges[k_bg])
                    if k_bg_plus > k_bg else 0.0
                )
                if bw_bg_gt > 0:
                    coef_R = (sum_xj - edges[k_bg] * n_g) / bw_bg_gt
                    _apply_term(i, coef_R, leaf_A, k_bg, sign=-1.0)
                alpha_L = (edges[k_star + 1] - x_star_j) / bw_xstar if bw_xstar > 0 else 0.0
                _apply_term(i, n_g * alpha_L, leaf_B, k_star, sign=-1.0)

                M = k_bg - k_star - 1
                if M > 0:
                    rev_path = list(reversed(tree_path))
                    recipe = _middle_recipe(rev_path, M)
                    for off, pairs in enumerate(recipe):
                        k_mid = k_star + 1 + off
                        for (node, w) in pairs:
                            if w == 0.0:
                                continue
                            _apply_term(i, n_g * w, node, k_mid, sign=-1.0)

        if not return_weights:
            local_effects[i] = nonlocal_total[0] / n_bg

    if return_weights:
        W /= n_bg
        return W
    return local_effects
