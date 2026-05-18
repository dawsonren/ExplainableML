"""
Path-integral local explanation: `_ale_local_vim_path_integral`.
"""

import numpy as np

from ale.shared import calculate_K, calculate_bin_index
from ale.vim.local_terms import route_first_index_at_bin


def _ale_local_vim_path_integral(
    X, idx, X_explain, f, deltas, edges, categorical, forest, k_X,
    f_X=None, background_indices=None, boundary_interp=False,
):
    """
    Path-integral local explanation for one feature (0-indexed `idx`).

    For each (x* in X_explain, x in X), walks through the existing ALE bins
    from x_j to x*_j: a partial-bin f-difference at each endpoint plus a sum
    of pre-computed deltas at interpolated middle bins (off-j coords linearly
    interpolated, then routed to the nearest training index via the forest).
    The mean over background X is returned.
    """
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

    if not boundary_interp:
        if f_X is None:
            f_X = np.asarray(f(X)).ravel()
        else:
            f_X = np.asarray(f_X).ravel()

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
                i_xstar = route_first_index_at_bin(forest, x_star, k_star)
                d_xstar = float(deltas[i_xstar])
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
            else:
                s, e = slice_same
                total += float((f_big[s:e] - f_X_bg[same_mask]).sum())

        # lt-bin (k_bg < k_star): walking x_bg → x_star, sign=+1
        if any_lt:
            if boundary_interp:
                k_bg_lt = k_X_bg[lt_mask]
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
                    i_mid = route_first_index_at_bin(forest, z, k_mid)
                    m += float(deltas[i_mid])
                mids_lt[ii] = m
            total += float((term_L_lt + mids_lt + term_R_lt_const).sum())

        # gt-bin (k_bg > k_star): walking A=x* → B=x_bg with sign=-1
        if any_gt:
            if boundary_interp:
                k_bg_gt = k_X_bg[gt_mask]
                k_bg_plus = np.minimum(k_bg_gt + 1, len(edges) - 1)
                bw_bg_gt = edges[k_bg_plus] - edges[k_bg_gt]
                with np.errstate(divide="ignore", invalid="ignore"):
                    alpha_R = np.where(
                        bw_bg_gt > 0,
                        (X_bg[gt_mask, idx] - edges[k_bg_gt]) / bw_bg_gt,
                        0.0,
                    )
                term_R_gt = alpha_R * deltas_bg[gt_mask]
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
                    i_mid = route_first_index_at_bin(forest, z, k_mid)
                    m += float(deltas[i_mid])
                mids_gt[ii] = m
            total += float((-1.0) * (term_L_gt_const + mids_gt + term_R_gt).sum())

        local_effects[i] = total / n_bg

    return local_effects
