# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple ale_1d_fast(object f, np.ndarray X, int feature_idx, int bins=10):
    cdef int idx = feature_idx - 1
    cdef np.ndarray[np.double_t, ndim=1] x = X[:, idx]
    cdef int n = x.shape[0]
    cdef np.ndarray[np.double_t, ndim=1] edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    edges[0] = x.min()
    edges[-1] = x.max() + np.finfo(np.float16).eps

    cdef np.ndarray[np.int_t, ndim=1] k_x = np.clip(np.searchsorted(edges, x, side='right'), 1, bins)
    cdef np.ndarray[np.int64_t, ndim=1] N_k = np.zeros(bins, dtype=np.int64)
    cdef int i
    for i in range(n):
        N_k[k_x[i]-1] += 1

    cdef np.ndarray[np.double_t, ndim=1] deltas = np.zeros(n)
    cdef np.ndarray X_left
    cdef np.ndarray X_right
    for i in range(n):
        X_left = X[i,:].copy()
        X_right = X[i,:].copy()
        X_left[idx] = edges[k_x[i]-1]
        X_right[idx] = edges[k_x[i]]
        deltas[i] = f(X_right) - f(X_left)

    cdef np.ndarray[np.double_t, ndim=1] average_deltas = np.zeros(bins)
    cdef int k
    for k in range(1, bins+1):
        if N_k[k-1] > 0:
            average_deltas[k-1] = (1.0/N_k[k-1]) * np.sum(deltas[k_x==k])

    cdef np.ndarray[np.double_t, ndim=1] accumulated = average_deltas.cumsum()
    accumulated = np.pad(accumulated, (1,0), constant_values=0)
    cdef np.ndarray[np.double_t, ndim=1] curve = accumulated - (1.0/n) * np.sum(accumulated[1:]*N_k)
    return edges, curve

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double ale_global_main_fast(np.ndarray edges, np.ndarray curve, np.ndarray X, int feature_idx):
    cdef int idx = feature_idx - 1
    cdef np.ndarray[np.double_t, ndim=1] x = X[:, idx]
    cdef int n = x.shape[0]
    cdef int bins = edges.shape[0] - 1
    cdef np.ndarray[np.int_t, ndim=1] k_x = np.clip(np.searchsorted(edges, x, side='right'), 1, bins)
    cdef double result = 0.0
    cdef int i
    for i in range(n):
        result += curve[k_x[i]] * curve[k_x[i]]
    return result / n

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple ale_2d_fast(object f, np.ndarray X, int feature_idx_1, int feature_idx_2, int bins=10):
    cdef int idx1 = feature_idx_1 - 1
    cdef int idx2 = feature_idx_2 - 1
    cdef np.ndarray[np.double_t, ndim=1] x1 = X[:, idx1]
    cdef np.ndarray[np.double_t, ndim=1] x2 = X[:, idx2]
    cdef int n = x1.shape[0]

    cdef np.ndarray[np.double_t, ndim=1] edges_1 = np.quantile(x1, np.linspace(0, 1, bins + 1))
    cdef np.ndarray[np.double_t, ndim=1] edges_2 = np.quantile(x2, np.linspace(0, 1, bins + 1))
    edges_1[0] = x1.min()
    edges_1[-1] = x1.max()
    edges_2[0] = x2.min()
    edges_2[-1] = x2.max()

    cdef np.ndarray[np.int_t, ndim=1] k_x = np.clip(np.searchsorted(edges_1, x1, side='right'), 1, bins)
    cdef np.ndarray[np.int_t, ndim=1] m_x = np.clip(np.searchsorted(edges_2, x2, side='right'), 1, bins)

    cdef np.ndarray[np.int64_t, ndim=2] N_km = np.zeros((bins, bins), dtype=np.int64)
    cdef int i, k, m
    for i in range(n):
        N_km[k_x[i]-1, m_x[i]-1] += 1
    cdef np.ndarray N_k = N_km.sum(axis=0)
    cdef np.ndarray N_m = N_km.sum(axis=1)

    cdef np.ndarray[np.double_t, ndim=1] deltas = np.zeros(n)
    cdef np.ndarray X_left_up
    cdef np.ndarray X_right_up
    cdef np.ndarray X_left_down
    cdef np.ndarray X_right_down
    for i in range(n):
        X_left_up = X[i,:].copy()
        X_right_up = X[i,:].copy()
        X_left_down = X[i,:].copy()
        X_right_down = X[i,:].copy()
        X_left_up[idx1] = edges_1[k_x[i]-1]
        X_left_up[idx2] = edges_2[m_x[i]]
        X_right_up[idx1] = edges_1[k_x[i]]
        X_right_up[idx2] = edges_2[m_x[i]]
        X_left_down[idx1] = edges_1[k_x[i]-1]
        X_left_down[idx2] = edges_2[m_x[i]-1]
        X_right_down[idx1] = edges_1[k_x[i]]
        X_right_down[idx2] = edges_2[m_x[i]-1]
        deltas[i] = (f(X_right_up) - f(X_left_up)) - (f(X_right_down) - f(X_left_down))

    cdef np.ndarray[np.double_t, ndim=2] average_deltas = np.zeros((bins, bins))
    cdef np.ndarray mask
    for k in range(1, bins+1):
        for m in range(1, bins+1):
            if N_km[k-1, m-1] > 0:
                mask = (k_x == k) & (m_x == m)
                average_deltas[k-1, m-1] = (1.0/N_km[k-1, m-1]) * np.sum(deltas[mask])

    cdef np.ndarray[np.double_t, ndim=2] raw_accumulated = average_deltas.cumsum(axis=0).cumsum(axis=1)

    cdef np.ndarray[np.double_t, ndim=1] main_effect_1 = np.zeros(bins)
    for k in range(1, bins+1):
        if N_k[k-1] > 0 and k > 1:
            main_effect_1[k-1] = (1.0/N_k[k-1]) * np.dot(average_deltas[k-1, :] - average_deltas[k-2, :], N_km[k-1, :])
    cdef np.ndarray[np.double_t, ndim=1] main_accumulated_1 = main_effect_1.cumsum()

    cdef np.ndarray[np.double_t, ndim=1] main_effect_2 = np.zeros(bins)
    for m in range(1, bins+1):
        if N_m[m-1] > 0 and m > 1:
            main_effect_2[m-1] = (1.0/N_m[m-1]) * np.dot(average_deltas[:, m-1] - average_deltas[:, m-2], N_km[:, m-1])
    cdef np.ndarray[np.double_t, ndim=1] main_accumulated_2 = main_effect_2.cumsum()

    cdef np.ndarray[np.double_t, ndim=2] accumulated_uncentered = raw_accumulated - main_accumulated_1[:, None] - main_accumulated_2[None, :]
    accumulated_uncentered = np.pad(accumulated_uncentered, ((1,0),(1,0)), constant_values=0)
    cdef np.ndarray[np.double_t, ndim=2] curve = accumulated_uncentered - (1.0/n) * (N_km * accumulated_uncentered[1:,1:]).sum()
    return edges_1, edges_2, curve

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple generate_connected_paths_2d_fast(np.ndarray X, int feature_idx, np.ndarray edges, int L):
    cdef int K = edges.shape[0] - 1
    cdef int idx = feature_idx - 1
    cdef np.ndarray[np.double_t, ndim=1] x_j = X[:, idx]
    cdef np.ndarray[np.double_t, ndim=1] x_other = X[:, 1 - idx]
    cdef np.ndarray[np.double_t, ndim=2] paths = np.zeros((L, K))
    cdef np.ndarray[np.int64_t, ndim=2] indices = np.zeros((L, K), dtype=np.int64)
    cdef int i
    cdef np.ndarray mask
    cdef np.ndarray original_idx
    cdef np.ndarray idxs
    for i in range(K):
        mask = (x_j >= edges[i]) & (x_j < edges[i+1])
        original_idx = np.where(mask)[0]
        if mask.sum() > 0:
            idxs = np.argsort(x_other[mask])[:L]
            paths[:, i] = x_other[mask][idxs]
            indices[:, i] = original_idx[idxs]
        else:
            raise ValueError("No observations found in bin [{}, {})".format(edges[i], edges[i+1]))
    return paths, indices

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double ale_quantile_total_fast(object f, np.ndarray X, int feature_idx, int bins=10):
    cdef int idx = feature_idx - 1
    cdef np.ndarray[np.double_t, ndim=1] x = X[:, idx]
    cdef int n = x.shape[0]

    cdef np.ndarray[np.double_t, ndim=1] edges = np.quantile(x, np.linspace(0,1,bins+1))
    edges[0] = x.min()
    edges[-1] = x.max() + np.finfo(np.float16).eps

    cdef np.ndarray[np.int_t, ndim=1] k_x = np.clip(np.searchsorted(edges, x, side='right'), 1, bins)
    cdef int k_bar = int(np.clip(np.searchsorted(edges, x.mean(), side='right'), 1, bins))

    cdef np.ndarray[np.int64_t, ndim=1] N_k = np.zeros(bins, dtype=np.int64)
    cdef int i, k, l
    for i in range(n):
        N_k[k_x[i]-1] += 1
    cdef int L = int(np.min(N_k))

    cdef list deltas = []
    cdef list delta_k
    cdef np.ndarray X_left
    cdef np.ndarray X_right
    for k in range(1, bins+1):
        delta_k = []
        for i in range(n):
            if k_x[i] == k:
                X_left = X[i,:].copy()
                X_right = X[i,:].copy()
                X_left[idx] = edges[k-1]
                X_right[idx] = edges[k]
                delta_k.append(f(X_right) - f(X_left))
        deltas.append(delta_k)

    cdef np.ndarray[np.double_t, ndim=2] g_values = np.zeros((L, bins))
    cdef double u
    for k in range(1, bins+1):
        for l in range(1, L+1):
            u = (l - 0.5)/L
            g_values[l-1, k-1] = np.quantile(deltas[k-1], u)

    cdef np.ndarray[np.double_t, ndim=2] accumulated_g_values = g_values.cumsum(axis=1)

    cdef np.ndarray[np.double_t, ndim=2] centered_g_values = np.zeros_like(accumulated_g_values)
    for l in range(L):
        centered_g_values[l,:] = accumulated_g_values[l,:] - accumulated_g_values[l, k_bar-1]

    cdef double average_g_value = 0.0
    for k in range(1, bins+1):
        average_g_value += (N_k[k-1]/L) * np.sum(centered_g_values[:, k-1])
    average_g_value /= n

    cdef double ale_vim = 0.0
    for k in range(1, bins+1):
        ale_vim += (N_k[k-1]/L) * np.sum((centered_g_values[:, k-1] - average_g_value)**2)
    return ale_vim / n

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double ale_connected_total_fast(object f, np.ndarray X, int feature_idx, int bins=10):
    cdef int idx = feature_idx - 1
    cdef np.ndarray[np.double_t, ndim=1] x = X[:, idx]
    cdef int p = X.shape[1]
    cdef int n = x.shape[0]
    if p != 2:
        raise NotImplementedError("Connected paths are only implemented for p=2 right now.")

    cdef np.ndarray[np.double_t, ndim=1] edges = np.quantile(x, np.linspace(0,1,bins+1))
    edges[0] = x.min()
    edges[-1] = x.max() + np.finfo(np.float16).eps

    cdef np.ndarray[np.int_t, ndim=1] k_x = np.clip(np.searchsorted(edges, x, side='right'), 1, bins)
    cdef int k_bar = int(np.clip(np.searchsorted(edges, x.mean(), side='right'), 1, bins))

    cdef np.ndarray[np.int64_t, ndim=1] N_k = np.zeros(bins, dtype=np.int64)
    cdef int i, k, l
    for i in range(n):
        N_k[k_x[i]-1] += 1
    cdef int L = int(np.min(N_k))

    cdef tuple paths_indices = generate_connected_paths_2d_fast(X, feature_idx, edges, L)
    cdef np.ndarray[np.double_t, ndim=2] paths = paths_indices[0]
    cdef np.ndarray[np.int64_t, ndim=2] indices = paths_indices[1]

    cdef np.ndarray[np.double_t, ndim=2] g_values = np.zeros((L, bins))
    cdef double x_other
    cdef int m
    cdef np.ndarray X_left
    cdef np.ndarray X_right
    for l in range(L):
        for m in range(bins):
            x_other = paths[l, m]
            i = indices[l, m]
            X_left = np.zeros_like(X[i,:])
            X_right = np.zeros_like(X[i,:])
            X_left[idx] = edges[k_bar-1]
            X_right[idx] = edges[k_bar]
            X_left[1-idx] = x_other
            X_right[1-idx] = x_other
            g_values[l, m] = f(X_right) - f(X_left)

    cdef np.ndarray[np.double_t, ndim=2] accumulated_g_values = g_values.cumsum(axis=1)
    cdef np.ndarray[np.double_t, ndim=2] centered_g_values = np.zeros_like(accumulated_g_values)
    for l in range(L):
        centered_g_values[l,:] = accumulated_g_values[l,:] - accumulated_g_values[l, k_bar-1]

    cdef double average_g_value = 0.0
    for k in range(1, bins+1):
        average_g_value += (N_k[k-1]/L) * np.sum(centered_g_values[:, k-1])
    average_g_value /= n

    cdef double ale_vim = 0.0
    for k in range(1, bins+1):
        ale_vim += (N_k[k-1]/L) * np.sum((centered_g_values[:, k-1] - average_g_value)**2)
    return ale_vim / n

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double ale_connected_modified_total_fast(object f, np.ndarray X, int feature_idx, int bins=10):
    cdef int idx = feature_idx - 1
    cdef np.ndarray[np.double_t, ndim=1] x = X[:, idx]
    cdef int p = X.shape[1]
    cdef int n = x.shape[0]
    if p != 2:
        raise NotImplementedError("Connected paths are only implemented for p=2 right now.")

    cdef np.ndarray[np.double_t, ndim=1] edges = np.quantile(x, np.linspace(0,1,bins+1))
    edges[0] = x.min()
    edges[-1] = x.max() + np.finfo(np.float16).eps

    cdef np.ndarray[np.int_t, ndim=1] k_x = np.clip(np.searchsorted(edges, x, side='right'), 1, bins)
    cdef int k_bar = int(np.clip(np.searchsorted(edges, x.mean(), side='right'), 1, bins))

    cdef np.ndarray[np.int64_t, ndim=1] N_k = np.zeros(bins, dtype=np.int64)
    cdef int i
    for i in range(n):
        N_k[k_x[i]-1] += 1
    cdef int L = int(np.min(N_k))

    cdef tuple paths_indices = generate_connected_paths_2d_fast(X, feature_idx, edges, L)
    cdef np.ndarray[np.double_t, ndim=2] paths = paths_indices[0]
    cdef np.ndarray[np.int64_t, ndim=2] indices = paths_indices[1]

    cdef np.ndarray[np.double_t, ndim=2] g_values = np.zeros((L, bins))
    cdef double x_other
    cdef int m
    cdef np.ndarray X_left
    cdef np.ndarray X_right
    for l in range(L):
        for m in range(bins):
            x_other = paths[l, m]
            i = indices[l, m]
            X_left = np.zeros_like(X[i,:])
            X_right = np.zeros_like(X[i,:])
            X_left[idx] = edges[k_bar-1]
            X_right[idx] = edges[k_bar]
            X_left[1-idx] = x_other
            X_right[1-idx] = x_other
            g_values[l, m] = f(X_right) - f(X_left)

    cdef np.ndarray[np.double_t, ndim=2] accumulated_g_values = g_values.cumsum(axis=1)
    cdef np.ndarray[np.double_t, ndim=2] centered_g_values = np.zeros_like(accumulated_g_values)
    for l in range(L):
        centered_g_values[l,:] = accumulated_g_values[l,:] - accumulated_g_values[l, k_bar-1]

    cdef np.ndarray[np.int64_t, ndim=1] path_indices = np.zeros(n, dtype=np.int64)
    for i in range(n):
        path_indices[i] = np.where(indices == i)[0][0]

    cdef np.ndarray[np.double_t, ndim=1] g_values_per_observation = np.zeros(n)
    cdef int bin_idx
    cdef int path_idx
    for i in range(n):
        path_idx = path_indices[i]
        bin_idx = k_x[i]-1
        g_values_per_observation[i] = centered_g_values[path_idx, bin_idx]

    cdef double ale_vim = 0.0
    cdef double average_g_value = np.mean(g_values_per_observation)
    for i in range(n):
        ale_vim += (g_values_per_observation[i] - average_g_value)**2
    return ale_vim / n
