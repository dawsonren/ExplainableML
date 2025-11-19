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

from typing import List, Any, Optional, Dict
import warnings
from dataclasses import dataclass, field
from queue import PriorityQueue

import numpy as np

from ale.shared import calculate_K, calculate_bins

###
### Helper classes
###


@dataclass(order=True)
class PrioritizedPath:
    priority: int
    path: Any = field(compare=False)


@dataclass
class KDNode:
    # split definition
    split_feature: Optional[int] = None  # m_star at this node; None => leaf
    thresholds: Optional[np.ndarray] = None  # shape (K,) float/obj per interval
    # topology
    left: Optional["KDNode"] = None
    right: Optional["KDNode"] = None
    # bookkeeping for routing/inspection
    R: Optional[list] = None  # list of length K; each is list[int] (indices into X)

    @property
    def is_leaf(self) -> bool:
        return self.split_feature is None

    def leaf_indices_for_k(self, k: int) -> np.ndarray:
        """
        Return the training indices for interval k (0-based k) at this leaf.
        """
        if not self.is_leaf:
            raise ValueError("leaf_indices_for_k called on non-leaf node")
        return np.array(self.R[k]).astype(int).ravel()


class ConnectedKDForest:
    """
    A single binary tree whose nodes carry interval-specific thresholds.
    Routing uses the same node sequence, but compares against thresholds[k]
    for the x_new's interval k.
    """

    def __init__(self, feature_idx: int, edges: np.ndarray, categorical: list, label_to_num: dict):
        self.feature_idx = feature_idx
        self.edges = edges
        self.categorical = categorical
        self.root: Optional[KDNode] = None
        self.K: Optional[int] = None
        self.label_to_num = label_to_num

    def _convert_x_new(self, x_new: np.ndarray) -> np.ndarray:
        """
        Convert categorical features in x_new to numeric representation.
        """
        x_converted = x_new.copy()
        for j, is_cat in enumerate(self.categorical):
            if is_cat:
                val = x_new[j]
                if val in self.label_to_num[j]:
                    x_converted[j] = self.label_to_num[j][val]
                else:
                    raise ValueError(f"Value {val} not found in label_to_num mapping for feature {j}")
        return x_converted

    def _bin_for_xj(self, x_new: np.ndarray) -> int:
        """
        Compute the 0-based interval k for a new point along feature j.
        """
        xj = np.array([x_new[self.feature_idx]])
        k_x, _ = calculate_bins(
            xj, self.edges, categorical=self.categorical[self.feature_idx]
        )
        # calculate_bins returns 1..K; convert to 0..K-1
        return int(k_x[0] - 1)

    def route(self, x_new: np.ndarray) -> Dict[str, object]:
        """
        Route a new point to a leaf and return useful info.
        Returns:
            {
              'k': int (0-based interval for x_j),
              'node': KDNode (the leaf),
              'indices': np.ndarray of training indices in that leaf for interval k,
            }
        """
        if self.root is None:
            raise RuntimeError("Forest not built yet")
    
        x_new = self._convert_x_new(x_new)
        k = self._bin_for_xj(x_new)
        node = self.root
        while not node.is_leaf:
            m = node.split_feature
            thr = node.thresholds[k]
            # categorical thresholds are *levels*; numeric are numbers
            val = x_new[m]
            # For categorical, we used an order derived from diffs; to stay consistent,
            node = node.left if val < thr else node.right

        return {
            "k": k,
            "node": node,
            "indices": node.leaf_indices_for_k(k),
        }

    def route_and_pick_representative(
        self, x_new: np.ndarray, X: np.ndarray, strategy: str = "first"
    ) -> Dict[str, object]:
        """
        Route and select a single representative 'actual x' at the leaf.
        Strategies:
          - 'first': pick the first index in that leaf interval
          - 'median_j': pick the index whose X_j is median within the leaf interval
        """
        info = self.route(x_new)
        idxs = info["indices"]
        if idxs.size == 0:
            raise RuntimeError("Leaf is unexpectedly empty for interval k")

        if strategy == "first":
            chosen_idx = int(idxs[0])
        elif strategy == "median_j":
            xj_vals = X[idxs, self.feature_idx]
            med_val = np.median(xj_vals)
            chosen_idx = int(idxs[np.argmin(np.abs(xj_vals - med_val))])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return {
            **info,
            "representative_index": chosen_idx,
            "representative_x": X[chosen_idx].copy(),
        }


###
### Helper functions
###
def _get_priority(R):
    """
    Priority score (lower number is higher priority) for splitting
    paths R. We calculate the score as the negative of the maximum
    number of elements in an interval, encouraging splits that
    reduce large intervals.

    Parameters
    ----------
    R             : list of intervals, each interval is a list of indices

    Returns
    -------
    score         : priority score
    """
    return -len(max(R, key=lambda interval: len(interval)))


def _perform_duplication(R):
    """
    Duplicate elements of singleton intervals in R in order to
    enable splitting. These operations are performed in-place.

    Parameters
    ----------
    R               : list of intervals, each interval is a list of indices
    """
    for interval in R:
        if len(interval) == 1:
            interval.append(interval[0])

    # assert all([len(interval) > 1 for interval in R]), "Duplication failed"
    if any([len(interval) <= 1 for interval in R]):
        print(R) # DEBUG


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


def _split_leaf(
    R: List[List[int]],
    X: np.ndarray,
    diffs: np.ndarray,
    j: int,
    categorical: List[bool],
):
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
    R_left, R_right : the two child leaf sets (same format as R)
    m_star          : int, index of the chosen splitting variable
    medians         : length-K numpy array of medians per interval
    """
    K = len(R)
    d = X.shape[1]
    candidate_features = [m for m in range(d) if m != j]

    best_obj = -np.inf
    m_star = None
    best_medians = None
    # evaluate every candidate feature
    for m in candidate_features:
        obj, medians, viable = _score_split(
            K, R, X, diffs, m, categorical=categorical[m]
        )

        if viable and obj > best_obj:
            best_obj = obj
            m_star = m
            best_medians = medians

    # should be rare: if all variables flat, split on first variable
    if m_star is None:
        warnings.warn(
            "generate_connected_delta_values: all features flat, splitting on first variable"
        )
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

    return R_left, R_right, m_star, best_medians


###
### Main algorithm
###``


def generate_connected_kdforest_and_paths(
    X: np.ndarray,
    feature_idx: int,
    edges: np.ndarray,
    deltas: np.ndarray,
    categorical: list,
    label_to_num: dict,
    L: int
):
    """
    Parameters:
    - X: (n, d) data
    - feature_idx: 1-based index of the feature to split on (j)
    - edges: edges for feature j
    - deltas: (n,) array of finite differences for feature j
    - categorical: list of booleans indicating if each feature is categorical

    Returns:
    - forest: ConnectedKDForest
    - paths: list of length L, each element is a length-K numpy array of index arrays
    """
    # Convert to 0-based j
    j = feature_idx - 1
    xj = X[:, j]
    k_x, _ = calculate_bins(xj, edges, categorical=categorical[j])

    K = calculate_K(edges, categorical[j])
    forest = ConnectedKDForest(feature_idx=j, edges=edges, categorical=categorical, label_to_num=label_to_num)
    forest.K = K

    # Initial leaf set at root: list of K intervals, each a list of indices
    root_R = []
    for k in range(1, K + 1):
        mask = k_x == k
        # store as python lists of ints
        root_R.append(np.where(mask)[0].astype(int).tolist())

    root = KDNode(split_feature=None, thresholds=None, left=None, right=None, R=root_R)

    # create priority queue
    queue = PriorityQueue()
    # use same type as before for priority; keep node reference inside
    queue.put(PrioritizedPath(0, root))

    # Helper: split a KDNode into two child nodes using split logic
    def _split_node(node: KDNode):
        # Use _perform_duplication and _split_leaf to get children R's,
        # and also recover the split feature and thresholds (medians per k).
        R = node.R
        # perform duplication if needed
        _perform_duplication(R)

        # find best split (m_star, medians) and children intervals
        R_left, R_right, m_star, best_medians = _split_leaf(
            R, X, deltas, j, categorical
        )

        # Build children nodes
        left = KDNode(
            split_feature=None, thresholds=None, left=None, right=None, R=R_left
        )
        right = KDNode(
            split_feature=None, thresholds=None, left=None, right=None, R=R_right
        )

        # Convert current node to internal split node
        node.split_feature = m_star
        node.thresholds = best_medians
        node.left = left
        node.right = right
        node.R = None  # internal nodes need not keep R

        # Priorities as in your queue policy
        pr_left = _get_priority(R_left)
        pr_right = _get_priority(R_right)
        return left, pr_left, right, pr_right

    # Grow until we have L leaves in the queue
    # We maintain the queue with leaf nodes only
    while queue.qsize() < L:
        node: KDNode = queue.get().path  # path stores KDNode now
        left, pr_left, right, pr_right = _split_node(node)
        queue.put(PrioritizedPath(pr_left, left))
        queue.put(PrioritizedPath(pr_right, right))

    # set forest root
    forest.root = root

    # build L paths, each a list of K index arrays
    paths = []
    while queue.qsize() > 0:
        node: KDNode = queue.get().path
        # node.R is [ list[int], ..., list[int] ] for K intervals
        path = [np.array(interval, dtype=int) for interval in node.R]
        paths.append(path)

    return forest, paths
