from dataclasses import dataclass
from typing import Optional

import numpy as np

from ale.shared import (
    calculate_bins,
    calculate_deltas,
    calculate_edges,
    calculate_K,
    linear_interpolation,
    calculate_bin_index,
)
from ale.tree_partitioning import (
    generate_connected_kdforest_and_paths,
    ConnectedKDForest,
)
from utils import generalized_distance


def _knn_smooth_deltas(X, deltas, k_x, K, knn, categorical, std_devs):
    """
    Smooth deltas via KNN averaging within each bin.

    For each bin, compute pairwise distances among all observations in that bin
    (using all features), then replace each observation's delta with the mean
    of its knn nearest neighbors' deltas.

    Parameters:
        X: training data, shape (n, d)
        deltas: raw deltas, shape (n,)
        k_x: 1-indexed bin assignments, shape (n,)
        K: number of bins
        knn: number of nearest neighbors
        categorical: list of booleans for each feature
        std_devs: per-feature standard deviations, shape (d,)

    Returns:
        Smoothed deltas, shape (n,)
    """
    smoothed = deltas.copy()

    for k in range(1, K + 1):
        mask = k_x == k
        if mask.sum() == 0:
            continue

        indices = np.where(mask)[0]
        X_bin = X[indices]
        deltas_bin = deltas[indices]
        n_bin = len(indices)

        # if fewer points than knn, just average all
        k_eff = min(knn, n_bin)

        # compute pairwise distances within the bin
        for i, obs_idx in enumerate(indices):
            dists = generalized_distance(
                X_bin[i], X_bin, categorical, std_devs
            )
            # argsort and take k_eff nearest (includes self at dist=0)
            neighbor_idx = np.argsort(dists)[:k_eff]
            smoothed[obs_idx] = np.mean(deltas_bin[neighbor_idx])

    return smoothed


@dataclass
class GValues:
    """
    The g_values are the accumulated values for each of L paths along K bins.
    g_values has shape (K, L)
    Nkl has shape (K, L) and contains the number of observations in each bin for each path
    centering can be "x" or "y"
    interpolate is a boolean indicating whether to interpolate g_values

    centered_g_values, K and L should not be modified and will be computed
    when get_centered is called
    """
    g_values: np.ndarray
    xj: np.ndarray
    bins: np.ndarray
    k_x: np.ndarray # 1-indexed bin indices for each observation
    l_x: np.ndarray # 0-indexed path indices for each observation
    centering: str
    interpolate: bool
    categorical: bool

    n: Optional[int] = None
    K: Optional[int] = None
    L: Optional[int] = None
    centered_g_values: Optional[np.ndarray] = None

    def __post_init__(self):
        self.xj_mean = self.xj.mean()
        self.K, self.L = self.g_values.shape
        self.n = len(self.xj)

        # compute centered_g_values
        self._compute_centered()

    def _compute_centered(self):
        # centering will be a row vector of length L, broadcast over the
        # (K, L) g_values matrix
        if self.centering == "x":
            # k_bar is 0-based bin index
            k_bar = calculate_bin_index(self.xj_mean, self.bins, self.K, self.categorical)

            if self.interpolate:
                # center so that xj_mean is exactly zero for each of the paths
                centering = linear_interpolation(
                    x=self.xj_mean,
                    x0=self.bins[k_bar],
                    x1=self.bins[k_bar + 1],
                    y0=self.g_values[k_bar - 1, :] if k_bar != 0 else np.zeros(self.L),
                    y1=self.g_values[k_bar, :],
                )
            else:
                # center so that the bin that contains xj_mean is exactly zero
                centering = self.g_values[k_bar, :]
        elif self.centering == "y":
            # center so that the weighted mean g-value across observations is zero on each path
            if self.interpolate:
                padded_g_values = np.pad(self.g_values, ((1, 0), (0, 0)), mode='constant', constant_values=0)
                # n length vector: interpolate g-value at each observation's x_j position
                g_values_per_observation = linear_interpolation(
                    x=self.xj,
                    x0=self.bins[self.k_x],
                    x1=self.bins[self.k_x + 1],
                    y0=padded_g_values[self.k_x, self.l_x],
                    y1=padded_g_values[self.k_x + 1, self.l_x],
                )
                # for each path, compute the average g value
                centering = np.zeros(self.L)
                for l in range(self.L):
                    centering[l] = np.mean(g_values_per_observation[self.l_x == l])
            else:
                g_values_per_observation = self.g_values[self.k_x, self.l_x]
                centering = np.zeros(self.L)
                for l in range(self.L):
                    centering[l] = np.mean(g_values_per_observation[self.l_x == l])

        if self.interpolate:
            self.centered_g_values = np.pad(self.g_values, ((1, 0), (0, 0)), mode='constant', constant_values=0) - centering
        else:
            self.centered_g_values = self.g_values - centering

    def lookup_locals(self, k_idxs, l_idxs, x_explain_j):
        # k_idxs and l_idxs are 0-indexed
        if not self.interpolate:
            return self.centered_g_values[k_idxs, l_idxs]
        else:

            return linear_interpolation(
                x=x_explain_j,
                x0=self.bins[k_idxs],
                x1=self.bins[k_idxs + 1],
                y0=self.centered_g_values[k_idxs, l_idxs],
                y1=self.centered_g_values[k_idxs + 1, l_idxs],
            )

    def plot_centered(self, ax):
        K, L = self.K, self.L
        if self.interpolate:
            for l in range(L):
                ax.plot(self.bins, self.centered_g_values[:, l], color=f"C{l}")
            for bin_edge in self.bins:
                ax.axvline(bin_edge, color='gray', linestyle='--')
        else:
            for l in range(L):
                for k in range(K):
                    ax.hlines(self.centered_g_values[k, l], self.bins[k], self.bins[k + 1], colors=f"C{l}")
            for bin_edge in self.bins:
                ax.axvline(bin_edge, color='gray', linestyle='--')

        ax.set_title(f"Centered g values (centering={self.centering}, interpolate={self.interpolate})")
        return ax


def _ale_main_vim(
    f, X, feature_idx, edges, categorical, interpolate=True, centering="x"
):
    idx = feature_idx - 1  # convert to 0-based index
    x = X[:, idx]
    n = len(x)
    K = calculate_K(edges, categorical)
    k_x, _ = calculate_bins(x, edges, categorical)
    deltas = calculate_deltas(f, X, idx, edges, k_x)
    # get average delta within each bin
    curve = np.zeros(K)
    for k in range(K):
        mask = k_x == k
        if mask.any():
            curve[k] = np.mean(deltas[mask])

    # accumulate
    curve = curve.cumsum()

    # center
    if centering == "x":
        k_bar = calculate_bin_index(x.mean(), edges, K, categorical)
        centered_curve = curve - curve[k_bar]
    elif centering == "y":
        centered_curve = curve - np.mean(curve)

    if interpolate:
        # NOTE: for observations in the last bin, use the value at the left edge
        last_bin = K - 1
        right_k = np.where(k_x == last_bin, k_x, k_x + 1)
        interpolated_values = linear_interpolation(
            x,
            edges[k_x],
            edges[right_k],
            centered_curve[k_x],
            centered_curve[right_k],
        )
        return (1 / n) * np.sum(interpolated_values ** 2)
    else:
        return np.var(centered_curve[k_x])


def __generate_quantile_delta_values(L, K, deltas, k_x):
    paths = np.zeros((K, L))
    for k in range(1, K + 1):
        delta_k = deltas[k_x == k]
        if len(delta_k) == 0:
            continue
        for l in range(1, L + 1):
            u = (l - (1 / 2)) / L
            # compute the u-quantile of the deltas for bin k
            paths[k - 1, l - 1] = np.quantile(delta_k, u)

    return paths


def observation_to_path(paths, n):
    l_x = np.zeros(n, dtype=int)
    for l, path in enumerate(paths):
        for k, interval in enumerate(path):
            for obs in interval:
                l_x[obs] = int(l)
    return l_x


def _generate_random_paths(L, K, deltas, k_x, seed):
    """
    Random path generation: within each bin, randomly partition observations
    into L disjoint groups. Path g-value at bin k = mean of deltas in group.

    Returns the same (deltas_by_path, paths) shape as the connected method so
    downstream code (observation_to_path, GValues) is reused unchanged.

    Returns
    -------
    deltas_by_path : (K, L) — mean delta per (bin, path), zero if empty.
    paths          : list of length L; paths[l][k] is an int array of the
                     training indices assigned to (path l, bin k).
    """
    rng = np.random.default_rng(seed)
    deltas_by_path = np.zeros((K, L))
    paths = [[np.array([], dtype=int) for _ in range(K)] for _ in range(L)]

    for k in range(K):
        idxs = np.where(k_x == k)[0]
        if idxs.size == 0:
            continue
        rng.shuffle(idxs)
        groups = np.array_split(idxs, L)
        for l, group in enumerate(groups):
            if group.size == 0:
                continue
            paths[l][k] = np.asarray(group, dtype=int)
            deltas_by_path[k, l] = float(np.mean(deltas[group]))

    return deltas_by_path, paths


def calculate_g_values(method, X, feature_idx, edges, deltas, categorical, label_to_num, L, K, k_x, seed=None):
    # collect deltas along paths ordered by total effect size
    if method == "connected":
        forest, paths = generate_connected_kdforest_and_paths(
            X, feature_idx, edges, deltas, categorical, label_to_num, L
        )
        deltas_by_path = np.zeros((K, L))
        # average across paths with multiple elements
        for l, path in enumerate(paths):
            for k, interval in enumerate(path):
                deltas_by_path[k, l] = np.mean(deltas[interval])
    elif method == "quantile":
        paths = None
        forest = None
        deltas_by_path = __generate_quantile_delta_values(L, K, deltas, k_x)
    elif method == "random":
        forest = None
        deltas_by_path, paths = _generate_random_paths(L, K, deltas, k_x, seed)
    else:
        raise ValueError(f"Unknown method: {method}")

    return deltas_by_path.cumsum(axis=0), paths, forest

def _ale_total_vim(
    f,
    X,
    feature_idx,
    K,
    L,
    categorical,
    label_to_num,
    method="connected",
    interpolate=True,
    centering="x",
    edges=None,
    knn_smooth=None,
    seed=None,
):
    idx = feature_idx - 1  # convert to 0-based index
    x_j = X[:, idx]
    n = X.shape[0]

    if edges is None:
        edges = calculate_edges(x_j, K, categorical[idx])

    K = calculate_K(edges, categorical[idx])

    k_x, _ = calculate_bins(x_j, edges, categorical[idx])
    deltas = calculate_deltas(f, X, idx, edges, k_x)

    if knn_smooth is not None and knn_smooth > 0:
        std_devs = X.std(axis=0)
        deltas = _knn_smooth_deltas(X, deltas, k_x, K, knn_smooth, categorical, std_devs)

    g_values, paths, forest = calculate_g_values(
        method, X, feature_idx, edges, deltas, categorical, label_to_num, L, K, k_x, seed=seed
    )

    if method in ("connected", "random"):
        l_x = observation_to_path(paths, n)

        centered_g_values = GValues(
            g_values, x_j, edges, k_x, l_x, centering, interpolate, categorical[idx]
        )
    elif method == "quantile":
        # Assign each observation to the nearest quantile path based on its delta
        l_x = np.zeros(n, dtype=int)
        for k in range(K):
            mask = k_x == k
            delta_k = deltas[mask]
            if len(delta_k) == 0:
                continue
            indices = np.where(mask)[0]
            quantile_vals = np.array([
                np.quantile(delta_k, (l + 0.5) / L) for l in range(L)
            ])
            # TODO: this is not quite right, __generate_quantile_delta_values should provide
            # you with the appropriate observation_to_path
            for i, obs_idx in enumerate(indices):
                l_x[obs_idx] = int(np.argmin(np.abs(quantile_vals - deltas[obs_idx])))

        centered_g_values = GValues(
            g_values, x_j, edges, k_x, l_x, centering, interpolate, categorical[idx]
        )

    # find variance over paths/observations
    ale_vim = np.var(centered_g_values.lookup_locals(k_x, l_x, x_j))

    return (
        ale_vim,
        forest,
        paths,
        g_values,
        centered_g_values,
        l_x,
        deltas,
        k_x,
    )


def _local_term_self(f, x_explain, idx, x_j_left):
    """f(x_j, x_\\j) - f(x_j_left, x_\\j). Scalar."""
    X_batch = np.tile(x_explain, (2, 1))
    X_batch[0, idx] = x_j_left
    X_batch[1, idx] = x_explain[idx]
    f_vals = f(X_batch)
    return f_vals[1] - f_vals[0]


def _local_term_path_rep(f, X, idx, x_explain, x_j_left, rep_idxs):
    """mean_{x* in rep_idxs} [ f(x_j, x*_\\j) - f(x_j_left, x*_\\j) ]. Scalar."""
    X_left_rep = X[rep_idxs].copy()
    X_point_rep = X[rep_idxs].copy()
    X_left_rep[:, idx] = x_j_left
    X_point_rep[:, idx] = x_explain[idx]
    return float(np.mean(f(X_point_rep) - f(X_left_rep)))


def _route_numeric_first_index_known_k(forest, x_numeric, k):
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
            f"path_integral: empty leaf for feature {j + 1}, bin k={k}"
        )
    return int(indices[0])


def _ale_local_vim_path_integral(
    X, feature_idx, X_explain, f, deltas, edges, categorical, forest, k_X,
    f_X=None, background_indices=None, boundary_interp=False,
):
    """
    Path-integral local explanation for one feature.

    For each (x* in X_explain, x in X), walks through the existing ALE bins
    from x_j to x*_j: a partial-bin f-difference at each endpoint plus a sum
    of pre-computed deltas at interpolated middle bins (off-j coords linearly
    interpolated, then routed to the nearest training index via the forest).
    The mean over background X is returned.

    Optimizations:
      - Reuses cached k_X (bin assignments for training data).
      - Precomputes f(X) once.
      - Per x*, precomputes f(x*) and the two q-invariant edge evaluations
        f(edges[k*], x*_{-j}) and f(edges[k*+1], x*_{-j}) used by the lt/gt
        branches.
      - Per x*, batches all q-dependent boundary f-evals into a single f call.
      - Routes interpolated middle points with a known-k helper (skips
        calculate_bin_index inside the descent).
    """
    idx = feature_idx - 1
    n_explain = X_explain.shape[0]
    cat_j = categorical[idx]
    K = calculate_K(edges, cat_j)
    d = X.shape[1]

    # Convert X_explain to numeric form (categoricals -> numeric labels) so
    # the linear interpolation z = A + alpha*(B-A) is well-defined.
    X_explain_num = np.empty_like(X, shape=(n_explain, d))
    for i in range(n_explain):
        X_explain_num[i] = forest._convert_x_new(X_explain[i])

    k_X = k_X.astype(int)
    k_explain = calculate_bin_index(
        X_explain_num[:, idx], edges, K, cat_j
    ).astype(int)

    # f(X) is only needed in f-eval boundary mode.
    if not boundary_interp:
        if f_X is None:
            f_X = np.asarray(f(X)).ravel()
        else:
            f_X = np.asarray(f_X).ravel()

    # Restrict the background to a subsample if requested. The forest and
    # `deltas` table are still keyed on the full X (routing returns full-X
    # indices), but we only iterate over and average across this subset.
    if background_indices is not None:
        bg_idx = np.asarray(background_indices, dtype=int)
    else:
        bg_idx = np.arange(X.shape[0], dtype=int)
    X_bg = X[bg_idx]
    k_X_bg = k_X[bg_idx]
    deltas_bg = deltas[bg_idx]
    f_X_bg = f_X[bg_idx] if not boundary_interp else None
    n_bg = X_bg.shape[0]

    local_effects = np.zeros(n_explain)

    for i in range(n_explain):
        x_star = X_explain_num[i]
        k_star = int(k_explain[i])
        x_star_j = x_star[idx]

        same_mask = (k_X_bg == k_star)
        lt_mask = (k_X_bg < k_star)
        gt_mask = (k_X_bg > k_star)
        any_same = bool(same_mask.any())
        any_lt = bool(lt_mask.any())
        any_gt = bool(gt_mask.any())

        # ---- f-eval boundary mode: assemble single batched f-call ----
        if not boundary_interp:
            batch_rows = [x_star.reshape(1, -1)]
            slice_xstar = (0, 1)
            cursor = 1

            slice_xstar_left_edge = None
            if any_lt:
                row = x_star.copy()
                row[idx] = edges[k_star]
                batch_rows.append(row.reshape(1, -1))
                slice_xstar_left_edge = (cursor, cursor + 1)
                cursor += 1

            slice_xstar_right_edge = None
            if any_gt:
                row = x_star.copy()
                row[idx] = edges[k_star + 1]
                batch_rows.append(row.reshape(1, -1))
                slice_xstar_right_edge = (cursor, cursor + 1)
                cursor += 1

            slice_same = None
            if any_same:
                X_same = X_bg[same_mask].copy()
                X_same[:, idx] = x_star_j
                batch_rows.append(X_same)
                slice_same = (cursor, cursor + X_same.shape[0])
                cursor += X_same.shape[0]

            slice_lt = None
            if any_lt:
                X_lt = X_bg[lt_mask].copy()
                X_lt[:, idx] = edges[k_X_bg[lt_mask] + 1]
                batch_rows.append(X_lt)
                slice_lt = (cursor, cursor + X_lt.shape[0])
                cursor += X_lt.shape[0]

            slice_gt = None
            if any_gt:
                X_gt = X_bg[gt_mask].copy()
                X_gt[:, idx] = edges[k_X_bg[gt_mask]]
                batch_rows.append(X_gt)
                slice_gt = (cursor, cursor + X_gt.shape[0])
                cursor += X_gt.shape[0]

            big_batch = np.vstack(batch_rows)
            f_big = np.asarray(f(big_batch)).ravel()
            f_xstar = float(f_big[slice_xstar[0]])

        # ---- linear-interp boundary mode: route x_star once for x*'s bin delta ----
        else:
            if any_same or any_lt or any_gt:
                i_xstar = _route_numeric_first_index_known_k(forest, x_star, k_star)
                d_xstar = float(deltas[i_xstar])
            # bw at x_star's bin (used for same/lt/gt). Cat last bin: numerator is 0
            # in every formula that uses it, so the placeholder 1.0 is harmless.
            bw_xstar = (
                float(edges[k_star + 1] - edges[k_star])
                if k_star + 1 < len(edges) else 1.0
            )

        total = 0.0

        # Same-bin contribution
        if any_same:
            if boundary_interp:
                if bw_xstar > 0:
                    same_contrib = (
                        (x_star_j - X_bg[same_mask, idx]) / bw_xstar
                        * deltas_bg[same_mask]
                    )
                    total += float(same_contrib.sum())
                # else: cat last bin; numerator forced to 0 → contribution 0
            else:
                s, e = slice_same
                total += float((f_big[s:e] - f_X_bg[same_mask]).sum())

        # lt-bin (k_bg < k_star): walking x_bg → x_star, sign=+1
        if any_lt:
            if boundary_interp:
                k_bg_lt = k_X_bg[lt_mask]
                # k_bg + 1 ≤ k_star ≤ K-1 ≤ len(edges)-1 here, safe.
                bw_bg_lt = edges[k_bg_lt + 1] - edges[k_bg_lt]
                with np.errstate(divide="ignore", invalid="ignore"):
                    alpha_L = np.where(
                        bw_bg_lt > 0,
                        (edges[k_bg_lt + 1] - X_bg[lt_mask, idx]) / bw_bg_lt,
                        1.0,
                    )
                term_L_lt = alpha_L * deltas_bg[lt_mask]
                alpha_R_const = (
                    (x_star_j - edges[k_star]) / bw_xstar if bw_xstar > 0 else 0.0
                )
                term_R_lt_const = alpha_R_const * d_xstar
            else:
                s, e = slice_lt
                term_L_lt = f_big[s:e] - f_X_bg[lt_mask]
                term_R_lt_const = f_xstar - float(f_big[slice_xstar_left_edge[0]])

            lt_indices = np.where(lt_mask)[0]
            mids_lt = np.zeros(lt_indices.size)
            for ii, q in enumerate(lt_indices):
                k_A = int(k_X_bg[q])
                if k_star - k_A < 2:
                    continue
                x_bg = X_bg[q]
                diff_bins = k_star - k_A
                m = 0.0
                for k_mid in range(k_A + 1, k_star):
                    if cat_j:
                        alpha = (k_mid - k_A) / diff_bins
                        z = x_bg + alpha * (x_star - x_bg)
                        z[idx] = edges[k_mid]
                    else:
                        mid_bin = 0.5 * (edges[k_mid] + edges[k_mid + 1])
                        alpha = (mid_bin - x_star[idx]) / (x_star[idx] - x_bg[idx])
                        z = x_bg + alpha * (x_star - x_bg)
                        z[idx] = mid_bin
                    i_mid = _route_numeric_first_index_known_k(forest, z, k_mid)
                    m += float(deltas[i_mid])
                mids_lt[ii] = m
            total += float((term_L_lt + mids_lt + term_R_lt_const).sum())

        # gt-bin (k_bg > k_star): walking A=x* → B=x_bg with sign=-1
        if any_gt:
            if boundary_interp:
                k_bg_gt = k_X_bg[gt_mask]
                # k_bg can be K-1 for cat (edges len K); clip to avoid OOB.
                k_bg_plus = np.minimum(k_bg_gt + 1, len(edges) - 1)
                bw_bg_gt = edges[k_bg_plus] - edges[k_bg_gt]
                with np.errstate(divide="ignore", invalid="ignore"):
                    alpha_R = np.where(
                        bw_bg_gt > 0,
                        (X_bg[gt_mask, idx] - edges[k_bg_gt]) / bw_bg_gt,
                        0.0,
                    )
                term_R_gt = alpha_R * deltas_bg[gt_mask]
                # gt requires k_bg > k_star, so k_star ≤ K-2, edges[k_star+1] is safe.
                alpha_L_const = (edges[k_star + 1] - x_star_j) / bw_xstar if bw_xstar > 0 else 0.0
                term_L_gt_const = alpha_L_const * d_xstar
            else:
                s, e = slice_gt
                term_R_gt = f_X_bg[gt_mask] - f_big[s:e]
                term_L_gt_const = float(f_big[slice_xstar_right_edge[0]]) - f_xstar
            gt_indices = np.where(gt_mask)[0]
            mids_gt = np.zeros(gt_indices.size)
            for ii, q in enumerate(gt_indices):
                k_B = int(k_X_bg[q])
                if k_B - k_star < 2:
                    continue
                x_bg = X_bg[q]
                diff_bins = k_B - k_star
                m = 0.0
                for k_mid in range(k_star + 1, k_B):
                    if cat_j:
                        alpha = (k_mid - k_star) / diff_bins
                        z = x_star + alpha * (x_bg - x_star)
                        z[idx] = edges[k_mid]
                    else:
                        mid_bin = 0.5 * (edges[k_mid] + edges[k_mid + 1])
                        alpha = (mid_bin - x_star[idx]) / (x_bg[idx] - x_star[idx])
                        z = x_star + alpha * (x_bg - x_star)
                        z[idx] = mid_bin
                    i_mid = _route_numeric_first_index_known_k(forest, z, k_mid)
                    m += float(deltas[i_mid])
                mids_gt[ii] = m
            total += float((-1.0) * (term_L_gt_const + mids_gt + term_R_gt).sum())

        local_effects[i] = total / n_bg

    return local_effects


def _ale_local_vim(
    X,
    feature_idx,
    X_explain,
    g_values: GValues,
    categorical,
    observation_to_path,
    forest: ConnectedKDForest,
    levels_up: int = 0,
    edges=None,
    f=None,
    k_x=None,
    local_method="interpolate",
    deltas=None,
    f_X=None,
    background_indices=None,
    boundary_interp=False,
):
    """
    Compute local ALE explanations for one feature.

    local_method:
      - "interpolate" (default): g*(x_j_left) + alpha * Delta_{k,l} via linear
        interpolation of the centered g-value curve.
      - "path_rep": g*(x_j_left) + mean_{x* in path l ∩ bin k}
        [f(x_j, x*_\\j) - f(x_j_left, x*_\\j)]. Requires f and k_x.
      - "self": g*(x_j_left) + [f(x_j, x_\\j) - f(x_j_left, x_\\j)]. Requires f.

    For categorical features or when the fitted g-values are piecewise-constant
    (interpolate=False), local_method is ignored and piecewise lookup is used.
    """
    idx = feature_idx - 1  # convert to 0-based index

    if edges is None:
        raise ValueError("edges must be provided to _ale_local_vim")
    if local_method not in ("interpolate", "path_rep", "self", "path_integral"):
        raise ValueError(
            f"Unknown local_method {local_method!r}; must be one of "
            "'interpolate', 'path_rep', 'self', 'path_integral'."
        )

    if local_method == "path_integral":
        if f is None:
            raise ValueError("f must be provided for local_method='path_integral'")
        if deltas is None:
            raise ValueError(
                "deltas must be provided for local_method='path_integral'"
            )
        if k_x is None:
            raise ValueError(
                "k_x must be provided for local_method='path_integral'"
            )
        return _ale_local_vim_path_integral(
            X, feature_idx, X_explain, f, deltas, edges, categorical, forest,
            k_x, f_X=f_X, background_indices=background_indices,
            boundary_interp=boundary_interp,
        )

    # short-circuit: categorical or piecewise g-values fall back to piecewise lookup
    effective_method = local_method
    if categorical[idx] or not g_values.interpolate:
        effective_method = "interpolate"

    if effective_method in ("path_rep", "self") and f is None:
        raise ValueError(f"f must be provided for local_method={local_method!r}")
    if effective_method == "path_rep" and k_x is None:
        raise ValueError("k_x must be provided for local_method='path_rep'")

    K = calculate_K(edges, categorical[idx])

    local_effects = np.zeros(X_explain.shape[0])

    for i in range(X_explain.shape[0]):
        x_explain = X_explain[i, :]
        x_idxs = forest.route_and_pick_representative(
            x_explain, X, levels_up=levels_up
        )["indices"]
        k_star = calculate_bin_index(x_explain[idx], edges, K, categorical[idx])
        l_xs = [observation_to_path[x_idx] for x_idx in x_idxs]

        if effective_method == "interpolate":
            # Use k_star (x_explain's bin) for g-value lookup, not the
            # training point's bin. The training point determines the path
            # (l_x), but the g-value should be evaluated at x_explain's
            # position. The forest may return a training point from an
            # adjacent bin due to boundary effects in the KD-tree split.
            effects = [
                g_values.lookup_locals(k_star, l_x, x_explain[idx])
                for l_x in l_xs
            ]
        else:
            x_j_left = edges[k_star]
            # "self" term is path-independent; compute once
            if effective_method == "self":
                term_self = _local_term_self(f, x_explain, idx, x_j_left)

            effects = []
            for l_x in l_xs:
                # base g*(x_j_left): padded centered_g_values[k_star, l_x]
                # (padded with leading zero row, so row k_star = value at edges[k_star])
                base = g_values.centered_g_values[k_star, l_x]

                if effective_method == "self":
                    term = term_self
                else:  # path_rep
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
                                f"feature {feature_idx}; cannot compute "
                                f"local_method='path_rep'."
                            )
                    term = _local_term_path_rep(
                        f, X, idx, x_explain, x_j_left, rep_idxs
                    )

                effects.append(base + term)

        local_effects[i] = np.mean(effects)

    return local_effects




