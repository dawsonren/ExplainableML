# cython: language_level=3
import numpy as np
cimport numpy as np

cpdef tuple ale_1d_fast(object f, np.ndarray X, int feature_idx, int bins=10):
    cdef int idx = feature_idx - 1
    cdef np.ndarray x = X[:, idx]
    cdef int n = x.shape[0]
    cdef np.ndarray edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    edges[0] = x.min()
    edges[-1] = x.max() + np.finfo(np.float16).eps

    cdef np.ndarray k_x = np.clip(np.searchsorted(edges, x, side='right'), 1, bins)
    cdef np.ndarray N_k = np.zeros(bins, dtype=np.int64)
    cdef int i
    for i in range(n):
        N_k[k_x[i]-1] += 1

    cdef np.ndarray deltas = np.zeros(n)
    cdef np.ndarray X_left
    cdef np.ndarray X_right
    for i in range(n):
        X_left = X[i,:].copy()
        X_right = X[i,:].copy()
        X_left[idx] = edges[k_x[i]-1]
        X_right[idx] = edges[k_x[i]]
        deltas[i] = f(X_right) - f(X_left)

    cdef np.ndarray average_deltas = np.zeros(bins)
    cdef int k
    for k in range(1, bins+1):
        if N_k[k-1] > 0:
            average_deltas[k-1] = (1.0/N_k[k-1]) * np.sum(deltas[k_x==k])

    cdef np.ndarray accumulated = average_deltas.cumsum()
    accumulated = np.pad(accumulated, (1,0), constant_values=0)
    cdef np.ndarray curve = accumulated - (1.0/n) * np.sum(accumulated[1:]*N_k)
    return edges, curve

cpdef double ale_global_main_fast(np.ndarray edges, np.ndarray curve, np.ndarray X, int feature_idx):
    cdef int idx = feature_idx - 1
    cdef np.ndarray x = X[:, idx]
    cdef int n = x.shape[0]
    cdef int bins = edges.shape[0] - 1
    cdef np.ndarray k_x = np.clip(np.searchsorted(edges, x, side='right'), 1, bins)
    cdef double result = 0.0
    cdef int i
    for i in range(n):
        result += curve[k_x[i]] * curve[k_x[i]]
    return result / n
