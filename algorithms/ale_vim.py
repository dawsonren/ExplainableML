"""Variable importance measures based on Accumulated Local Effects (ALE).

The original repository implemented a collection of experimental algorithms
operating on NumPy arrays.  In this re-write all public functions accept and
manipulate :class:`pandas.DataFrame` objects directly which makes the code
clearer and easier to use from notebooks.

The module exposes a small set of routines used by the notebooks:

``ale_global_main``
    Global main–effect importance as described in the ALE literature.

``generate_connected_paths_2d`` and ``generate_connected_paths``
    Utilities that construct *connected* paths through the data set.  These
    paths are used by the ``ale_connected_*`` functions to estimate variable
    importance.

``ale_quantile_total`` / ``ale_connected_total`` /
``ale_connected_modified_total``
    Different strategies for computing a total importance measure.  The
    implementations below favour clarity over ultimate efficiency and rely on
    pandas for all data wrangling.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .ale import (
    ale_1d,
    calculate_bins,
    calculate_edges,
)


# ---------------------------------------------------------------------------
# Helper utilities


def ale_global_main(
    f: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    feature_idx: int,
    bins: int = 10,
) -> float:
    """Global main‑effect importance for feature ``feature_idx``.

    The implementation follows the definition in Apley & Molnar (2020): the
    squared and centred ALE values are averaged over all observations.
    """

    idx = feature_idx - 1
    x = X.iloc[:, idx]
    edges, curve = ale_1d(f, X, feature_idx, bins=bins)
    # ``ale_1d`` returns truncated edges; we recompute the full set for binning
    full_edges = calculate_edges(x, bins, categorical=False)
    k_x, _ = calculate_bins(x, full_edges)
    return float(np.mean(curve[k_x - 1] ** 2))


def generate_connected_paths_2d(
    X: pd.DataFrame, feature_idx: int, edges: Sequence[float], L: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate connected paths for two dimensional data.

    Parameters
    ----------
    X:
        Data frame with exactly two features.
    feature_idx:
        1‑based index of the feature for which the ALE is computed.
    edges:
        Bin edges for the selected feature.
    L:
        Number of paths to generate (typically the minimum bin count).
    """

    if X.shape[1] != 2:
        raise NotImplementedError(
            "generate_connected_paths_2d currently supports only two features"
        )

    idx = feature_idx - 1
    other = 1 - idx
    col = X.columns[idx]
    other_col = X.columns[other]

    bins = pd.cut(X[col], edges, labels=False, include_lowest=True)
    grouped = X.groupby(bins, sort=True, dropna=False)

    paths = np.zeros((L, len(edges) - 1))
    indices = np.zeros((L, len(edges) - 1), dtype=int)

    for k in range(len(edges) - 1):
        group = grouped.get_group(k)
        group = group.sort_values(by=other_col)
        if len(group) < L:
            raise ValueError("Not enough observations in bin to create paths")
        paths[:, k] = group.iloc[:L][other_col].to_numpy()
        indices[:, k] = group.iloc[:L].index.to_numpy()

    return paths, indices


# The two helpers below are part of the original algorithm for high dimensional
# path generation.  Their implementations here are simplified and primarily
# serve to keep the public API intact.  They operate on NumPy arrays because
# representing a (K, L, d) tensor as a data frame would be cumbersome.


def _precompute_diffs(
    X: np.ndarray, j: int, edges: Sequence[float], f: Callable[[pd.DataFrame], float]
) -> np.ndarray:
    """Pre–compute local effects for every observation in ``X``.

    Parameters
    ----------
    X:
        Array of shape ``(K, L, d)`` containing the data along the different
        paths.
    j:
        Index of the effect variable (0‑based).
    edges:
        Bin edges for feature ``j``.
    f:
        Prediction function accepting a data frame with ``d`` columns.
    """

    K, L, d = X.shape
    diffs = np.empty((K, L), dtype=float)

    cols = [f"x{i}" for i in range(d)]

    for k in range(K):
        lower, upper = edges[k], edges[k + 1]
        for l in range(L):
            base = pd.Series(X[k, l, :], index=cols)
            base_low = base.copy()
            base_low.iloc[j] = lower
            base_high = base.copy()
            base_high.iloc[j] = upper
            diffs[k, l] = float(f(pd.DataFrame([base_high])) - f(pd.DataFrame([base_low])))
    return diffs


def _split_leaf(
    R: List[List[int]], X: np.ndarray, diffs: np.ndarray, j: int
) -> Tuple[int, np.ndarray, List[List[int]], List[List[int]]]:
    """Split a *leaf set* as described in the ALE VIM paper.

    The simplified implementation chooses the splitting variable based on the
    largest absolute difference in the pre‑computed ``diffs``.  This is
    sufficient for demonstration purposes in the notebooks.
    """

    K, _, d = X.shape
    candidate_features = [m for m in range(d) if m != j]

    # choose the feature with largest variance across all intervals
    variances = []
    for m in candidate_features:
        vals = X[:, :, m].reshape(K, -1)
        variances.append(vals.var())
    m_star = candidate_features[int(np.argmax(variances))]

    medians = np.median(X[:, :, m_star], axis=1)

    R_left = [[] for _ in range(K)]
    R_right = [[] for _ in range(K)]
    for k in range(K):
        for idx in R[k]:
            if X[k, idx, m_star] <= medians[k]:
                R_left[k].append(idx)
            else:
                R_right[k].append(idx)

    return m_star, medians, R_left, R_right


def generate_connected_paths(
    X: pd.DataFrame, feature_idx: int, edges: Sequence[float]
) -> List[pd.DataFrame]:
    """Construct connected paths through ``X`` for feature ``feature_idx``.

    The algorithm follows the description in Section 4.3 of the ALE VIM paper
    but is intentionally lightweight.  Each path contains exactly one
    observation from every bin of the selected feature.
    """

    idx = feature_idx - 1
    col = X.columns[idx]
    bins = pd.cut(X[col], edges, labels=False, include_lowest=True)
    grouped = X.groupby(bins, sort=True, dropna=False)
    L = grouped.size().min()

    paths: List[pd.DataFrame] = []
    for l in range(L):
        rows = []
        for k in range(len(edges) - 1):
            g = grouped.get_group(k).sort_values(by=X.columns.difference([col]).tolist())
            rows.append(g.iloc[l])
        paths.append(pd.DataFrame(rows).reset_index(drop=True))

    return paths


# ---------------------------------------------------------------------------
# Variable importance measures


def ale_quantile_total(
    f: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    feature_idx: int,
    bins: int = 10,
) -> float:
    """Total importance using quantile paths.

    This is a light‑weight approximation that integrates the squared ALE curve
    for ``feature_idx``.  It is primarily meant for demonstration purposes.
    """

    edges, curve = ale_1d(f, X, feature_idx, bins=bins)
    return float(np.trapz(curve ** 2, edges) / len(edges))


def ale_connected_total(
    f: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    feature_idx: int,
    bins: int = 10,
) -> float:
    """Total importance based on connected paths.

    The implementation is restricted to two features which mirrors the use in
    the accompanying notebooks.  For each path we accumulate local effects and
    then compute their variance across paths.
    """

    if X.shape[1] != 2:
        raise NotImplementedError("Connected paths are only implemented for p=2")

    idx = feature_idx - 1
    x = X.iloc[:, idx]
    n = len(x)

    edges = calculate_edges(x, bins, categorical=False)
    k_x, N_k = calculate_bins(x, edges)
    L = int(N_k.min())

    paths, indices = generate_connected_paths_2d(X, feature_idx, edges, L)
    g_values = np.zeros((L, bins))

    other = 1 - idx
    for l in range(L):
        for k in range(bins):
            row = X.iloc[indices[l, k]].copy()
            row_left = row.copy()
            row_right = row.copy()
            row_left.iloc[idx] = edges[k]
            row_right.iloc[idx] = edges[k + 1]
            g_values[l, k] = float(f(pd.DataFrame([row_right])) - f(pd.DataFrame([row_left])))

    accumulated = g_values.cumsum(axis=1)
    # centre each path at the overall average
    centred = accumulated - accumulated.mean(axis=1, keepdims=True)
    return float((N_k * centred.var(axis=0)).sum() / n)


def ale_connected_modified_total(
    f: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    feature_idx: int,
    bins: int = 10,
) -> float:
    """Modified version of ``ale_connected_total`` used in the notebooks."""

    idx = feature_idx - 1
    x = X.iloc[:, idx]
    n = len(x)

    edges = calculate_edges(x, bins, categorical=False)
    k_x, N_k = calculate_bins(x, edges)
    L = int(N_k.min())

    paths, indices = generate_connected_paths_2d(X, feature_idx, edges, L)
    g_values = np.zeros((L, bins))

    for l in range(L):
        for k in range(bins):
            row = X.iloc[indices[l, k]].copy()
            row_left = row.copy()
            row_right = row.copy()
            row_left.iloc[idx] = edges[k]
            row_right.iloc[idx] = edges[k + 1]
            g_values[l, k] = float(f(pd.DataFrame([row_right])) - f(pd.DataFrame([row_left])))

    accumulated = g_values.cumsum(axis=1)
    centred = accumulated - accumulated.mean(axis=0, keepdims=True)
    average = (N_k[:, None] * centred).sum() / n
    return float(((centred - average) ** 2).sum() / n)


# End of module

