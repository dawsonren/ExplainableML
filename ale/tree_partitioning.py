"""
Provide tree partitioning to create connected paths.
"""

from typing import List, Any, Optional, Dict
from dataclasses import dataclass, field
from queue import PriorityQueue

import numpy as np

from ale.shared import calculate_K, calculate_bins, calculate_bin_index

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

    def __init__(
        self, feature_idx: int, edges: np.ndarray, categorical: list, label_to_num: dict
    ):
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
                    raise ValueError(
                        f"Value {val} not found in label_to_num mapping for feature {j}"
                    )
        return x_converted

    def _bin_for_xj(self, x_new: np.ndarray) -> int:
        """
        Compute the 0-based interval k for a new point along feature j.
        """
        xj = x_new[self.feature_idx]
        return int(calculate_bin_index(
            xj, self.edges, self.K, categorical=self.categorical[self.feature_idx]
        ))

    def _collect_leaf_indices(self, node: Optional[KDNode], k: int) -> np.ndarray:
        """
        Collect all training indices for interval k from all leaf nodes
        in the subtree rooted at `node`.
        """
        if node is None:
            return np.array([], dtype=int)

        if node.is_leaf:
            return node.leaf_indices_for_k(k)

        left_idxs = self._collect_leaf_indices(node.left, k)
        right_idxs = self._collect_leaf_indices(node.right, k)

        if left_idxs.size == 0:
            return right_idxs
        if right_idxs.size == 0:
            return left_idxs
        return np.concatenate([left_idxs, right_idxs])

    def route(self, x_new: np.ndarray, levels_up: int = 0) -> Dict[str, object]:
        """
        Route a new point and return node info.

        Args
        ----
        x_new : np.ndarray
            New sample.
        levels_up : int, default 0
            0 -> return the leaf node (original behavior).
            1 -> return the parent of the leaf,
            2 -> grandparent, etc.
            If levels_up is larger than the depth of the leaf, the root is returned.

        Returns
        -------
        dict with keys:
          - 'k': int
                0-based interval for x_j.
          - 'node': KDNode
                The node at the requested height (may be internal or leaf).
          - 'indices': np.ndarray
                All training indices (for interval k) in all leaves below that node.
        """
        if self.root is None:
            raise RuntimeError("Forest not built yet")

        if levels_up < 0:
            raise ValueError("levels_up must be >= 0")

        x_new = self._convert_x_new(x_new)
        k = self._bin_for_xj(x_new)

        # Descend the tree while keeping track of the path
        path: list[KDNode] = []
        node = self.root
        while not node.is_leaf:
            path.append(node)
            m = node.split_feature
            thr = node.thresholds[k]
            val = x_new[m]
            node = node.left if val < thr else node.right

        # Include the final leaf in the path
        path.append(node)

        # Choose the node `levels_up` above the leaf, clamped at root
        target_depth_index = max(0, len(path) - 1 - levels_up)
        target_node = path[target_depth_index]

        # Collect all indices for all leaves below target_node for interval k
        indices = self._collect_leaf_indices(target_node, k)

        return {
            "k": k,
            "node": target_node,
            "indices": indices,
        }

    def route_and_pick_representative(
        self,
        x_new: np.ndarray,
        X: np.ndarray,
        strategy: str = "first",
        levels_up: int = 0,
    ) -> Dict[str, object]:
        """
        Route and select a single representative 'actual x' at the leaf.
        Strategies:
          - 'first': pick the first index in that leaf interval
          - 'median_j': pick the index whose X_j is median within the leaf interval
        """
        info = self.route(x_new, levels_up=levels_up)
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

    assert all([len(interval) > 1 for interval in R]), "Duplication failed"


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

    # assert that length of R is K
    assert len(R) == K, "R length does not match K"
    # also assert that length of each R[k] >= 1
    assert all([len(interval) >= 1 for interval in R]), "Some R[k] is empty"

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
        # warnings.warn(
        #     "generate_connected_delta_values: all features flat, splitting on first variable"
        # )
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
###


def generate_connected_kdforest_and_paths(
    X: np.ndarray,
    feature_idx: int,
    edges: np.ndarray,
    deltas: np.ndarray,
    categorical: list,
    label_to_num: dict,
    L: int,
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
    forest = ConnectedKDForest(
        feature_idx=j, edges=edges, categorical=categorical, label_to_num=label_to_num
    )
    forest.K = K

    # Initial leaf set at root: list of K intervals, each a list of indices
    root_R = []
    for k in range(K):
        mask = k_x == k
        # store as python lists of ints
        root_R.append(np.where(mask)[0].astype(int).tolist())

    if any([len(interval) == 0 for interval in root_R]):
        empty_intervals = [k for k in range(K) if len(root_R[k]) == 0]
        print(
            f"generate_connected_kdforest_and_paths: empty intervals found at root: {empty_intervals}"
        )

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
