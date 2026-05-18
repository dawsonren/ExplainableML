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
    parent: Optional["KDNode"] = None
    # bookkeeping for routing/inspection
    R: Optional[list] = None  # list of length K; each is list[int] (indices into X)
    # precomputed per-bin subtree stats (multi_path).
    # mean_delta[k] = mean of training-point deltas over leaves of this node's
    # subtree restricted to interval k. nan if the (node, k) cell is empty.
    # n_per_bin[k] = number of training points (unique) in this node's subtree
    # at interval k.
    mean_delta: Optional[np.ndarray] = None
    n_per_bin: Optional[np.ndarray] = None

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

    def route(self, x_new: np.ndarray) -> Dict[str, object]:
        """
        Route a new point and return the leaf node info.

        Returns
        -------
        dict with keys:
          - 'k': int — 0-based interval for x_j.
          - 'node': KDNode — the reached leaf.
          - 'indices': np.ndarray — training indices in the leaf for interval k.
        """
        if self.root is None:
            raise RuntimeError("Forest not built yet")

        x_new = self._convert_x_new(x_new)
        k = self._bin_for_xj(x_new)

        node = self.root
        while not node.is_leaf:
            m = node.split_feature
            thr = node.thresholds[k]
            val = x_new[m]
            node = node.left if val < thr else node.right

        return {
            "k": k,
            "node": node,
            "indices": node.leaf_indices_for_k(k),
        }

    def route_with_path(self, x_new: np.ndarray, k: Optional[int] = None) -> Dict[str, object]:
        """
        Route x_new through the tree at its interval k (computed from x_new if
        not provided) and return the full descent path (root → ... → leaf) plus
        the resulting leaf.

        Returns dict with keys:
          - 'k': int — 0-based interval used for routing.
          - 'leaf': KDNode — the leaf reached.
          - 'path': list[KDNode] — the descent path from root to leaf.
        """
        if self.root is None:
            raise RuntimeError("Forest not built yet")

        x_new = self._convert_x_new(x_new)
        if k is None:
            k = self._bin_for_xj(x_new)

        path: list[KDNode] = []
        node = self.root
        while not node.is_leaf:
            path.append(node)
            m = node.split_feature
            thr = node.thresholds[k]
            val = x_new[m]
            node = node.left if val < thr else node.right
        path.append(node)

        return {"k": int(k), "leaf": node, "path": path}

    def subtree_mean_delta(self, node: "KDNode", k: int) -> float:
        """
        Mean delta over all training points in `node`'s subtree at interval k.
        Returns nan if the (node, k) cell is empty. Backed by precomputed
        per-node arrays — O(1) lookup.
        """
        if node.mean_delta is None:
            raise RuntimeError(
                "subtree_mean_delta called before mean_delta was populated; "
                "ensure the forest was built via generate_connected_kdforest_and_paths"
            )
        return float(node.mean_delta[k])

    @staticmethod
    def tree_path_between(
        path_A: list, path_B: list
    ) -> list:
        """
        Given the two root→leaf descent paths from `route_with_path` for leaves
        A and B, return the ordered list of nodes traversing from A → LCA → B.

        The output starts at A and ends at B; LCA appears once in the middle
        (or at one end if one leaf is an ancestor of the other, though leaves
        only have leaves as descendants, so this only happens if A == B).
        """
        # Longest common prefix (root-down) gives the chain through LCA.
        lca_idx = 0
        for a, b in zip(path_A, path_B):
            if a is b:
                lca_idx += 1
            else:
                break
        # lca_idx is one past the last common node; the LCA is path_A[lca_idx-1].
        # Ascending from A to LCA: path_A[-1], path_A[-2], ..., path_A[lca_idx-1]
        up = list(reversed(path_A[lca_idx - 1:]))
        # Descending from LCA to B: path_B[lca_idx-1], path_B[lca_idx], ..., path_B[-1]
        down = path_B[lca_idx:]
        return up + down

    @staticmethod
    def assign_middle_nodes(path: list, M: int) -> list:
        """
        Pick M nodes (one per middle bin) by depth-interpolation along `path`
        (which goes A → LCA → B). Endpoints A=path[0] and B=path[-1] are
        excluded; interior positions are rounded to the nearest node along
        `path`.

        - D = len(path) - 1 (tree-edge distance from A to B).
        - For i = 1..M: t_i = round(i * D / (M + 1)); node_i = path[t_i].

        Behavior:
          - M == D: yields path[1..D] (strict walk between endpoints).
          - M < D : skips intermediate nodes; lands on shallower ancestors.
          - M > D : nodes repeat (stall), biased toward the LCA.
          - len(path) == 1 (A == B): all M assignments equal that single node.
        """
        if M <= 0:
            return []
        if len(path) == 1:
            return [path[0]] * M
        D = len(path) - 1
        out = []
        denom = M + 1
        for i in range(1, M + 1):
            t = int(round(i * D / denom))
            t = max(0, min(D, t))
            out.append(path[t])
        return out

    @staticmethod
    def assign_middle_node_weights(path: list, M: int) -> list:
        """
        Standard-linear-interpolation analogue of `assign_middle_nodes`.

        For M > 0, return a list of length M; entry i is a 2-tuple of
        (node, weight) pairs:
            [(node_lo, w_lo), (node_hi, w_hi)]
        such that the middle-bin contribution at bin k_mid is
            w_lo * node_lo.mean_delta[k_mid] + w_hi * node_hi.mean_delta[k_mid].

        Setup: interior = path[1:-1] (endpoints A=path[0], B=path[-1] are
        excluded), D = len(interior). For i = 1..M:
            p_i = i * (D - 1) / (M + 1)
            lo = floor(p_i),  hi = lo + 1   (both clipped to [0, D-1])
            w_hi = p_i - lo,  w_lo = 1 - w_hi

        Edge cases:
          - D == 0 (no interior; len(path) in {1, 2}):
              emit [(path[0], 1.0), (path[0], 0.0)] for every i (fallback to the
              A endpoint; contributes A's mean_delta[k_mid]).
          - D == 1: single interior node; emit [(interior[0], 1.0),
              (interior[0], 0.0)] for every i.
          - hi == D (clipped to D-1): emit [(interior[lo], 1.0),
              (interior[lo], 0.0)] (cannot interpolate past the last interior
              node).
        """
        if M <= 0:
            return []
        interior = path[1:-1]
        D = len(interior)
        if D == 0:
            anchor = path[0]
            return [[(anchor, 1.0), (anchor, 0.0)] for _ in range(M)]
        if D == 1:
            node = interior[0]
            return [[(node, 1.0), (node, 0.0)] for _ in range(M)]
        out = []
        denom = M + 1
        for i in range(1, M + 1):
            p = i * (D - 1) / denom
            lo = int(p)
            if lo >= D - 1:
                # at the upper edge; cannot interpolate past last node
                out.append([(interior[D - 1], 1.0), (interior[D - 1], 0.0)])
                continue
            hi = lo + 1
            w_hi = p - lo
            w_lo = 1.0 - w_hi
            out.append([(interior[lo], w_lo), (interior[hi], w_hi)])
        return out

    def route_and_pick_representative(
        self,
        x_new: np.ndarray,
        X: np.ndarray,
        strategy: str = "first",
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

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        depth, n_leaves, n_internal = self._tree_stats()
        return (
            f"ConnectedKDForest(feature_idx={self.feature_idx}, K={self.K}, "
            f"depth={depth}, internal_nodes={n_internal}, leaves={n_leaves})"
        )

    def _tree_stats(self) -> tuple:
        if self.root is None:
            return 0, 0, 0
        max_depth = 0
        n_leaves = 0
        n_internal = 0
        stack = [(self.root, 0)]
        while stack:
            node, d = stack.pop()
            max_depth = max(max_depth, d)
            if node.is_leaf:
                n_leaves += 1
            else:
                n_internal += 1
                stack.append((node.left, d + 1))
                stack.append((node.right, d + 1))
        return max_depth, n_leaves, n_internal

    def pretty_print(
        self,
        max_depth: Optional[int] = None,
        show_thresholds: bool = True,
        show_stats: bool = True,
        show_indices: bool = True,
        max_indices_shown: int = 12,
        feature_names: Optional[list] = None,
        float_fmt: str = "{:.3f}",
        max_bins_shown: int = 6,
    ) -> str:
        """
        Render the forest as a text tree. Returns the string (also printable).

        Parameters
        ----------
        max_depth        : if set, truncate the tree below this depth.
        show_thresholds  : show per-bin split thresholds at internal nodes.
        show_stats       : show per-bin n / mean_delta at leaves.
        feature_names    : optional names; falls back to "x{m}".
        float_fmt        : format spec for floats.
        max_bins_shown   : truncate per-bin arrays after this many entries.
        """
        if self.root is None:
            text = f"{self!r}\n(empty — forest not built)"
            print(text)
            return text

        def name(m: int) -> str:
            if feature_names is not None and 0 <= m < len(feature_names):
                return feature_names[m]
            return f"x{m}"

        def fmt_array(arr) -> str:
            if arr is None:
                return "?"
            vals = list(arr)
            trunc = len(vals) > max_bins_shown
            shown = vals[:max_bins_shown]
            parts = []
            for v in shown:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    parts.append("nan")
                else:
                    try:
                        parts.append(float_fmt.format(float(v)))
                    except (TypeError, ValueError):
                        parts.append(str(v))
            s = "[" + ", ".join(parts) + ("…" if trunc else "") + "]"
            return s

        lines = [repr(self)]

        def recurse(node: KDNode, prefix: str, is_last: bool, depth: int):
            connector = "└── " if is_last else "├── "
            if node.is_leaf:
                bits = [f"leaf (depth={depth})"]
                if show_stats:
                    if node.n_per_bin is not None:
                        bits.append(f"n_per_bin={fmt_array(node.n_per_bin)}")
                    if node.mean_delta is not None:
                        bits.append(f"mean_delta={fmt_array(node.mean_delta)}")
                    if node.n_per_bin is None and node.R is not None:
                        counts = [len(r) for r in node.R]
                        bits.append(f"n_per_bin={fmt_array(counts)}")
                lines.append(prefix + connector + "  ".join(bits))
                if show_indices and node.R is not None:
                    child_prefix = prefix + ("    " if is_last else "│   ")
                    n_bins = len(node.R)
                    bins_to_show = min(n_bins, max_bins_shown)
                    for k in range(bins_to_show):
                        idxs = list(node.R[k])
                        trunc_idx = len(idxs) > max_indices_shown
                        shown = idxs[:max_indices_shown]
                        idx_str = ", ".join(str(int(i)) for i in shown)
                        if trunc_idx:
                            idx_str += f", … (+{len(idxs) - max_indices_shown})"
                        lines.append(f"{child_prefix}  R[{k}] = [{idx_str}]")
                    if n_bins > bins_to_show:
                        lines.append(
                            f"{child_prefix}  … ({n_bins - bins_to_show} more bins)"
                        )
                return

            head = f"split on {name(node.split_feature)} (depth={depth})"
            if show_thresholds and node.thresholds is not None:
                head += f"  thr={fmt_array(node.thresholds)}"
            lines.append(prefix + connector + head)

            if max_depth is not None and depth >= max_depth:
                child_prefix = prefix + ("    " if is_last else "│   ")
                lines.append(child_prefix + "└── … (truncated)")
                return

            child_prefix = prefix + ("    " if is_last else "│   ")
            recurse(node.left, child_prefix, is_last=False, depth=depth + 1)
            recurse(node.right, child_prefix, is_last=True, depth=depth + 1)

        recurse(self.root, prefix="", is_last=True, depth=0)
        text = "\n".join(lines)
        print(text)
        return text


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
    j: int,
    edges: np.ndarray,
    deltas: np.ndarray,
    categorical: list,
    label_to_num: dict,
    L: int,
):
    """
    Parameters:
    - X: (n, d) data
    - j: 0-based index of the feature to split on
    - edges: edges for feature j
    - deltas: (n,) array of finite differences for feature j
    - categorical: list of booleans indicating if each feature is categorical

    Returns:
    - forest: ConnectedKDForest
    - paths: list of length L, each element is a length-K numpy array of index arrays
    """
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

    root = KDNode(
        split_feature=None, thresholds=None, left=None, right=None,
        parent=None, R=root_R,
    )

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
            split_feature=None, thresholds=None, left=None, right=None,
            parent=node, R=R_left,
        )
        right = KDNode(
            split_feature=None, thresholds=None, left=None, right=None,
            parent=node, R=R_right,
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

    # Precompute per-node, per-bin subtree mean deltas and counts via post-order
    # traversal. Used by the `multi_path` local explanation method. Cost is
    # O(|nodes| * K), amortized once per feature.
    _populate_subtree_stats(root, deltas, K)

    return forest, paths


def iter_subtree_indices_at_bin(node: "KDNode", k: int) -> np.ndarray:
    """
    Return the observation indices contributing to `node.mean_delta[k]`,
    matching the post-order recursion in `_populate_subtree_stats`. The
    returned array has length `node.n_per_bin[k]` and may contain repeats:
    `_split_leaf`'s tie-break can move one half of a singleton-duplicated
    observation to the empty-side child, so the same obs appears at bin k
    in both children's subtrees. mean_delta count-weights those repeats
    (giving the duplicated obs a higher weight in the parent's mean), so
    downstream weight distribution must respect them.

    Consumers should use `np.add.at(out, idx, coef / n_per_bin[k])` so that
    duplicate indices accumulate correctly.

    At the leaf level we still dedup R[k] via `np.unique`, mirroring
    `_populate_subtree_stats`'s leaf-level handling of intra-leaf duplicates
    introduced by `_perform_duplication`.
    """
    if node.is_leaf:
        if not node.R[k]:
            return np.empty(0, dtype=int)
        return np.unique(np.asarray(node.R[k], dtype=int))
    left = iter_subtree_indices_at_bin(node.left, k)
    right = iter_subtree_indices_at_bin(node.right, k)
    if left.size == 0:
        return right
    if right.size == 0:
        return left
    return np.concatenate([left, right])


def _populate_subtree_stats(node: "KDNode", deltas: np.ndarray, K: int) -> None:
    """
    Post-order fill of node.mean_delta (shape K) and node.n_per_bin (shape K)
    for every node in the subtree rooted at `node`.

    Invariant (per node, per bin k):
        node.mean_delta[k] = (mL*nL + mR*nR) / (nL + nR)            (children avg)
        node.n_per_bin[k] = sum of children's n_per_bin[k]           (sum count)

    At leaves both reduce to `mean / count of unique obs in R[k]` (since
    `_perform_duplication` may replicate singletons inside a leaf, and those
    duplicates share a delta value, so deduping via `np.unique` does not
    change the mean and makes the leaf's n_per_bin a true unique count).

    Beware that at INTERNAL nodes `n_per_bin[k]` is NOT always a unique count:
    `_split_leaf`'s tie-break may move one of a singleton-duplicated obs into
    the empty-side child, leaving the same obs in both children's subtrees at
    bin k. The recursion then counts that obs once per child, and `mean_delta`
    weights its delta by 2 rather than 1. This is the correct invariant for
    the multi_path estimator (each leaf occurrence is one "vote"), and the
    weight-distribution path uses the same multiplicities.
    """
    if node.is_leaf:
        mean_d = np.full(K, np.nan, dtype=float)
        n_per = np.zeros(K, dtype=int)
        for k in range(K):
            idx_k = node.R[k]
            if not idx_k:
                continue
            arr = np.asarray(idx_k, dtype=int)
            # _perform_duplication may have duplicated singleton entries; since
            # duplicates share the same delta the mean is unaffected, and we
            # report a count of unique observations for downstream weighting.
            unique = np.unique(arr)
            n_per[k] = int(unique.size)
            mean_d[k] = float(deltas[unique].mean())
        node.mean_delta = mean_d
        node.n_per_bin = n_per
        return

    _populate_subtree_stats(node.left, deltas, K)
    _populate_subtree_stats(node.right, deltas, K)

    nL = node.left.n_per_bin
    nR = node.right.n_per_bin
    mL = node.left.mean_delta
    mR = node.right.mean_delta

    n_per = nL + nR
    mean_d = np.full(K, np.nan, dtype=float)
    for k in range(K):
        total = n_per[k]
        if total == 0:
            continue
        sL = (mL[k] * nL[k]) if nL[k] > 0 else 0.0
        sR = (mR[k] * nR[k]) if nR[k] > 0 else 0.0
        mean_d[k] = (sL + sR) / total

    node.mean_delta = mean_d
    node.n_per_bin = n_per
