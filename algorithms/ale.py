"""Accumulated Local Effects (ALE) utilities that operate on pandas data
frames.

The original implementation in this repository worked exclusively with
NumPy arrays.  For the purposes of the exercises in this kata we re-write
the helper functions so that a :class:`pandas.DataFrame` can be supplied
directly.  Only basic NumPy functionality is used where convenient, but the
data manipulation is handled with pandas which greatly simplifies the
binning and aggregation logic.

The public API mirrors the previous implementation:

``bin_selection``
    Choose a sensible number of bins for a given sample size.

``ale_1d`` and ``ale_2d``
    Compute 1‑D and 2‑D centred ALE curves.

The functions accept a callable ``f`` that maps a data frame to predictions
and return NumPy arrays in order to remain lightweight for plotting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper utilities


def bin_selection(n: int) -> int:
    """Select a number of bins close to ``sqrt(n)``.

    Parameters
    ----------
    n:
        Number of observations.
    """

    divisors = [i for i in range(1, n + 1) if n % i == 0]
    return min(divisors, key=lambda x: abs(x - np.sqrt(n)))


def calculate_edges(x: pd.Series, bins: int, categorical: bool = False) -> np.ndarray:
    """Return bin edges for ``x``.

    For categorical features the unique values are used as edges.  For
    continuous features equal–mass bins are employed.
    """

    if categorical:
        edges = np.sort(x.unique())
        # append a tiny amount to create the upper edge of the final bin
        edges = np.append(edges, edges[-1] + np.finfo(float).eps)
    else:
        quantiles = np.linspace(0, 1, bins + 1)
        edges = x.quantile(quantiles).to_numpy()
        edges[0] = x.min()
        edges[-1] = x.max() + np.finfo(float).eps
    return edges


def calculate_bins(x: pd.Series, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Assign each observation in ``x`` to a bin defined by ``edges``.

    Returns
    -------
    k_x:
        1‑based bin index for every observation.
    N_k:
        Number of observations per bin.
    """

    K = len(edges) - 1
    bins = pd.cut(x, edges, labels=False, include_lowest=True)
    k_x = bins.astype(int) + 1  # convert to 1‑based indexing
    counts = k_x.value_counts(sort=False).reindex(range(1, K + 1), fill_value=0)
    return k_x.to_numpy(), counts.to_numpy()


def calculate_bins_2d(
    x1: pd.Series, x2: pd.Series, edges_1: np.ndarray, edges_2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two‑dimensional binning used for interaction ALE plots."""

    K = len(edges_1) - 1
    M = len(edges_2) - 1

    bins1 = pd.cut(x1, edges_1, labels=False, include_lowest=True).astype(int) + 1
    bins2 = pd.cut(x2, edges_2, labels=False, include_lowest=True).astype(int) + 1

    counts = (
        pd.crosstab(bins1, bins2, dropna=False)
        .reindex(index=range(1, K + 1), columns=range(1, M + 1), fill_value=0)
    )

    N_k = counts.sum(axis=1).to_numpy()
    N_m = counts.sum(axis=0).to_numpy()
    return bins1.to_numpy(), bins2.to_numpy(), N_k, N_m, counts.to_numpy()


def calculate_deltas(
    f: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    idx: int,
    edges: np.ndarray,
    k_x: np.ndarray,
) -> np.ndarray:
    """Per‑observation local effects for feature ``idx``."""

    col = X.columns[idx]
    X_left = X.copy()
    X_right = X.copy()

    left = pd.Series(edges[k_x - 1], index=X.index)
    right = pd.Series(edges[k_x], index=X.index)
    X_left[col] = left.values
    X_right[col] = right.values

    return np.asarray(f(X_right) - f(X_left)).reshape(-1)


def calculate_deltas_2d(
    f: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    idx_1: int,
    idx_2: int,
    edges_1: np.ndarray,
    edges_2: np.ndarray,
    k_x: np.ndarray,
    m_x: np.ndarray,
) -> np.ndarray:
    """Local effects for the interaction between two features."""

    col1, col2 = X.columns[idx_1], X.columns[idx_2]

    X_lu = X.copy()
    X_ru = X.copy()
    X_ld = X.copy()
    X_rd = X.copy()

    left1 = pd.Series(edges_1[k_x - 1], index=X.index)
    right1 = pd.Series(edges_1[k_x], index=X.index)
    left2 = pd.Series(edges_2[m_x - 1], index=X.index)
    right2 = pd.Series(edges_2[m_x], index=X.index)

    # upper path
    X_lu[col1] = left1.values
    X_lu[col2] = right2.values
    X_ru[col1] = right1.values
    X_ru[col2] = right2.values

    # lower path
    X_ld[col1] = left1.values
    X_ld[col2] = left2.values
    X_rd[col1] = right1.values
    X_rd[col2] = left2.values

    return np.asarray((f(X_ru) - f(X_lu)) - (f(X_rd) - f(X_ld))).reshape(-1)


# ---------------------------------------------------------------------------
# Public API


def ale_1d(
    f: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    feature_idx: int,
    bins: int,
    categorical: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute centred 1‑D ALE values.

    Parameters
    ----------
    f:
        Model prediction function.  Must accept a data frame and return a
        one‑dimensional array of predictions.
    X:
        Data as a :class:`pandas.DataFrame`.
    feature_idx:
        1‑based index of the feature to analyse.
    bins:
        Number of bins for ALE calculation.
    categorical:
        Whether the feature is categorical.
    """

    idx = feature_idx - 1
    x = X.iloc[:, idx]
    n = len(x)
    K = x.nunique() if categorical else bins

    edges = calculate_edges(x, bins, categorical)
    k_x, N_k = calculate_bins(x, edges)

    deltas = calculate_deltas(f, X, idx, edges, k_x)

    df = pd.DataFrame({"k": k_x, "delta": deltas})
    average = (
        df.groupby("k")[["delta"]].mean().reindex(range(1, K + 1), fill_value=0)
    )

    accumulated = average["delta"].to_numpy().cumsum()
    average_effect = (accumulated * N_k).sum() / n
    curve = accumulated - average_effect

    edges = edges[:-1] if categorical else edges[1:]
    if categorical:
        curve = np.insert(curve[:-1], 0, -average_effect)

    return edges, curve


def ale_2d(
    f: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    feature_idx_1: int,
    feature_idx_2: int,
    bins: int,
    categorical_1: bool = False,
    categorical_2: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute centred 2‑D ALE values for an interaction term."""

    idx1, idx2 = feature_idx_1 - 1, feature_idx_2 - 1
    x1, x2 = X.iloc[:, idx1], X.iloc[:, idx2]
    K = x1.nunique() - 1 if categorical_1 else bins
    M = x2.nunique() - 1 if categorical_2 else bins
    n = len(X)

    edges1 = calculate_edges(x1, bins, categorical_1)
    edges2 = calculate_edges(x2, bins, categorical_2)

    k_x, m_x, N_k, N_m, N_km = calculate_bins_2d(x1, x2, edges1, edges2)
    deltas = calculate_deltas_2d(f, X, idx1, idx2, edges1, edges2, k_x, m_x)

    df = pd.DataFrame({"k": k_x, "m": m_x, "delta": deltas})
    table = (
        df.pivot_table(
            index="k", columns="m", values="delta", aggfunc="mean", dropna=False
        )
        .reindex(index=range(1, K + 1), columns=range(1, M + 1), fill_value=0)
        .to_numpy()
    )

    raw_acc = table.cumsum(axis=0).cumsum(axis=1)

    counts = pd.DataFrame(N_km, index=range(1, K + 1), columns=range(1, M + 1))
    overall_mean = (raw_acc * counts).to_numpy().sum() / n
    row_mean = (raw_acc * counts).sum(axis=1).to_numpy() / N_k
    col_mean = (raw_acc * counts).sum(axis=0).to_numpy() / N_m
    curve = raw_acc - row_mean[:, None] - col_mean[None, :] + overall_mean

    points_1 = edges1[:-1] if categorical_1 else edges1[1:]
    points_2 = edges2[:-1] if categorical_2 else edges2[1:]
    return points_1, points_2, curve


# End of module

