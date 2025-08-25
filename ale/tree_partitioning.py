"""
Provide tree partitioning to create connected paths.
The implementation for categorical variable is elaborated below.

NOTE: Why can't we just rebalance the X values at the beginning?
1. By duplicating observations in intervals that have n_j(k) < L,
we change the median, this should be discouraged.
2. We don't know which observations to average together in intervals 
that have n_j(k) > L.
NOTE: I propose a two-part procedure. We always keep a priority queue
of collections based on the largest number of observations in an interval.
If we don't yet have L values in our heap, we duplicate any data points 
we may need. We also somehow need to keep track of how many times things have been
duplicated already, so that we keep the distribution of our data somewhat coherent.
Heuristically, set our priority equal to the largest observations in an interval
PLUS the number of times this data point has already been duplicated.
Once we obtain L values in our heap, we then perform our averaging procedure.
This encourages averages over fewer elements, which hopefully corresponds to "closer"
observations that are being averaged.

TODO: You might want to measure the maximum distance
between points in this group, but that feels...too complicated.
"""


from typing import List, Any
import warnings
from dataclasses import dataclass, field
from queue import PriorityQueue

import numpy as np

from ale.shared import calculate_K, calculate_bins


@dataclass(order=True)
class PrioritizedPath:
    priority: int
    path: Any=field(compare=False)


def _score_split(K, R, X, diffs, m, categorical):
    obj = 0.0
    medians = np.empty(K)
    viable = False  # at least one interval must produce a *real* split

    for k in range(K):
        idx_k = R[k]
        vals = X[idx_k, m]
        if categorical:
            levels = np.unique(vals)
            if len(levels) == 1:  # unsplittable here
                continue

            d_lvl = np.zeros(len(levels))
            for i, lvl in enumerate(levels):
                d_lvl[i] = diffs[np.array(idx_k)[vals == lvl]].mean()

            # sort levels by d_lvl
            sorted_levels = levels[np.argsort(d_lvl)]
            # median of levels, prefer right side
            med = sorted_levels[(len(sorted_levels) + 1) // 2]
        else:
            med = np.median(vals)

        medians[k] = med

        left_mask = vals < med
        if left_mask.all() or (~left_mask).all():  # unsplittable here
            continue

        viable = True
        dl = diffs[np.array(idx_k)[left_mask]].mean()
        dr = diffs[np.array(idx_k)[~left_mask]].mean()
        obj += abs(dl - dr)

    return obj, medians, viable


def _split_leaf(R: List[List[int]], X: np.ndarray, diffs: np.ndarray, j: int, categorical: List[bool]):
    """
    Implements Algorithm 1 for a single *leaf set* (R_1, …, R_K).

    Parameters
    ----------
    R     : length-K list; R[k] is a *list* of indices
    X     : (n, d) data tensor
    diffs : (n, d) pre-computed Δ values
    j     : index of X_j (not a candidate split variable)
    categorical : list of whether or not variable is categorical

    Returns
    -------
    m_star          : int, index of the chosen splitting variable
    medians         : length-K numpy array of medians per interval
    R_left, R_right : the two child leaf sets (same format as R)
    """
    K = len(R)
    d = X.shape[1]
    candidate_features = [m for m in range(d) if m != j]

    best_obj = -np.inf
    m_star = None
    best_medians = None
    # evaluate every candidate feature
    for m in candidate_features:
        obj, medians, viable = _score_split(K, R, X, diffs, m, categorical=categorical[m])

        if viable and obj > best_obj:
            best_obj = obj
            m_star = m
            best_medians = medians

    # should be rare: if all variables flat, split on first variable
    if m_star is None:
        warnings.warn("generate_connected_delta_values: all features flat, splitting on first variable")
        m_star = candidate_features[0]
        best_medians = np.array([np.median(X[R[k], m_star]) for k in range(K)])

    # create children
    R_left, R_right = [], []
    for k in range(K):
        idx_k = np.array(R[k])
        vals = X[idx_k, m_star]
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

    return R_left, R_right


def _get_priority(R, reuse_tracker):
    """
    Priority score (lower number is higher priority) for splitting
    paths R. We calculate the score as the negative of the maximum
    number of elements in an interval plus the sum of
    reuse_tracker over indices where there is only one observation
    remaining (requiring duplication).

    Parameters
    ----------
    R             : list of intervals, each interval is a list of indices
    reuse_tracker : list of length n with number of reuses

    Returns
    -------
    score         : priority score
    """
    max_elements_in_interval = len(max(R, key=lambda interval: len(interval)))
    if all([len(interval) > 1 for interval in R]):
        duplication_penalty = 0
    else:
        duplication_penalty = sum([reuse_tracker[interval[0]] for interval in R if len(interval) == 1])
    return -max_elements_in_interval + duplication_penalty


def _perform_duplication(R, reuse_tracker):
    """
    Duplicate elements of singleton intervals in R in order to
    enable splitting. Update reuse_tracker accordingly. These
    operations are performed in-place.

    Parameters
    ----------
    R               : list of intervals, each interval is a list of indices
    reuse_tracker   : list of length n with number of reuses
    """
    for interval in R:
        if len(interval) == 1:
            reuse_tracker[interval[0]] += 1
            interval.append(interval[0])
    
    assert all([len(interval) > 1 for interval in R]), "Duplication failed"


def generate_connected_delta_values(
    X: np.ndarray, feature_idx: int, edges: np.ndarray, deltas: np.ndarray, categorical: List[bool]
) -> List[np.ndarray]:
    """
    Full recursive algorithm producing a (K, L) matrix of delta values.

    Recall that Δ_j,L = f(x_j = z_{k+1}, L_k(w_l)) - f(x_j = z_k, L_k(w_l)).

    Parameters
    ----------
    f           : callable that evaluates the prediction model on 1-D inputs
    X           : (n, d) array of observations
    feature_idx : index of the effect variable X_j (0-based)
    edges       : 1-D array of giving the z_{k} boundaries
    deltas      : 1-D array of length n giving the delta values for each observation
    categorical : list of whether or not variable is categorical

    Returns
    -------
    deltas_by_path : (K, L) array of delta values
    """
    idx = feature_idx - 1  # convert to 0-based index
    x = X[:, idx]
    n = len(x)
    K = calculate_K(edges, categorical[idx])
    L = n // K  # number of paths

    k_x, _ = calculate_bins(x, edges, categorical=categorical[idx])

    # initial leaf set: every interval holds all its indices
    # NOTE: indices are from 1..n corresponding to row in X/diffs.
    root_R = []
    for k in range(1, K + 1):
        mask = k_x == k
        root_R.append(list(np.argwhere(mask)))

    # priority queue via heap
    queue = PriorityQueue()
    queue.put(PrioritizedPath(0, root_R))
    reuse_tracker = np.zeros(n)
    while queue.qsize() < L:
        R = queue.get().path
        # NOTE: in-place
        _perform_duplication(R, reuse_tracker)
        R_left, R_right = _split_leaf(R, X, deltas, idx, categorical)
        priority_left = _get_priority(R_left, reuse_tracker)
        priority_right = _get_priority(R_right, reuse_tracker)
        queue.put(PrioritizedPath(priority_left, R_left))
        queue.put(PrioritizedPath(priority_right, R_right))
    
    assert queue.qsize() == L, "Algorithm should yield exactly L paths"

    # pop everything into an iterable
    paths = []
    while queue.qsize() > 0:
        paths.append(queue.get().path)

    deltas_by_path = np.zeros((L, K))
    # average across paths with multiple elements
    for l, path in enumerate(paths):
        for k, interval in enumerate(path):
            deltas_by_path[l, k] = np.mean(deltas[interval])

    return deltas_by_path
