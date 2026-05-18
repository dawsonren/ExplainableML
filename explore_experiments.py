"""
explore_experiments.py

Streamlit app combining summarize_experiments.py (main page) and
visualize_experiments.py (detail page) for the new per-config cache layout.

Main page  : interactive summary table of all (experiment, ALE tag, SHAP tag) rows.
Detail page: full set of plots for a selected row.

Usage:
    streamlit run explore_experiments.py
    streamlit run explore_experiments.py -- --cache-dir /path/to/cache_root
"""

import os
import sys
import tempfile
from collections import OrderedDict
from urllib.parse import quote, unquote

import matplotlib
matplotlib.use("Agg")

import joblib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import r2_score

import models
from ale import ALE, BootstrapALE
from ale.plotting import plot_paths_summary
from shapley import SHAP
from experiments_io import rebuild_ale_from_run
from summarize_experiments import (
    _walk_cache, _load_tune, _fmt_cell,
)
from visualize_experiments import (
    load_cache, extract_view, detect_explanation_fn,
    plot_bias2, plot_variance, plot_f_variability, plot_f_variance,
    plot_single_replication,
    plot_mean_feature_explanations, plot_mean_function_additivity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_cache_dir() -> str:
    for i, arg in enumerate(sys.argv):
        if arg == "--cache-dir" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return "cached_explanations"


def _row_key(row: dict) -> str:
    return f"{row['config_name']}::{row['_results_file']}::{row['ale_tag']}::{row['shap_tag']}"


@st.cache_data
def load_all_rows(cache_dir: str) -> list:
    rows = _walk_cache(cache_dir)
    rows.sort(
        key=lambda r: r["rel_stddev_reduction"]
        if not np.isnan(r["rel_stddev_reduction"])
        else -np.inf,
        reverse=True,
    )
    for r in rows:
        r["_key"] = _row_key(r)
    return rows


def _results_path(cache_dir: str, config_name: str, results_file: str) -> str:
    return os.path.join(cache_dir, config_name, results_file)


@st.cache_resource(show_spinner=False)
def _load_results_cached(results_path: str):
    return load_cache(results_path)


@st.cache_data(show_spinner=False)
def generate_plots(cache_dir: str, config_name: str, results_file: str,
                   ale_tag: str, shap_tag: str) -> str:
    results_path = _results_path(cache_dir, config_name, results_file)
    plot_dir = os.path.join(
        tempfile.gettempdir(), "explainableml_plots",
        config_name, results_file, ale_tag, shap_tag,
    )
    os.makedirs(plot_dir, exist_ok=True)

    results = _load_results_cached(results_path)
    view = extract_view(results, ale_tag, shap_tag)
    true_fn = detect_explanation_fn(results)

    plot_bias2(view, true_explanation_fn=true_fn, save_dir=plot_dir)
    plot_variance(view, save_dir=plot_dir)
    plot_f_variability(view, save_dir=plot_dir)
    plot_f_variance(view, save_dir=plot_dir)
    plot_single_replication(view, r=0, save_dir=plot_dir)
    plot_mean_feature_explanations(view, save_dir=plot_dir)
    plot_mean_function_additivity(view, save_dir=plot_dir)
    return plot_dir


@st.cache_resource(show_spinner=False)
def _build_paths_ale(cache_dir: str, config_name: str, results_file: str, ale_tag: str):
    """Build an ALE object on replication 0 for the paths-summary plot."""
    results_path = _results_path(cache_dir, config_name, results_file)
    results = _load_results_cached(results_path)
    ale_entry = results["ale"].get(ale_tag)
    if ale_entry is None or ale_entry.get("config") is None:
        return None
    ec = ale_entry["config"]

    runs = _load_all_replications(cache_dir, config_name, results_file)
    if runs is None:
        return None

    run_path = os.path.join(cache_dir, config_name, _run_pkl_name(results))
    return rebuild_ale_from_run(run_path, ec, rep_idx=0)


def _run_pkl_name(results: dict) -> str:
    meta = results["experiment_meta"]
    return f"run_{meta['dgp_slug']}_{meta['fit_model_slug']}_n{meta['n']}_R{meta['replications']}.pkl"


# ---------------------------------------------------------------------------
# Interactive detail mode (d=2 only)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_all_replications(cache_dir: str, config_name: str, results_file: str):
    """Load the run_*.pkl companion for an experiment."""
    results_path = _results_path(cache_dir, config_name, results_file)
    results = _load_results_cached(results_path)
    meta = results["experiment_meta"]
    run_name = f"run_{meta['dgp_slug']}_{meta['fit_model_slug']}_n{meta['n']}_R{meta['replications']}.pkl"
    run_path = os.path.join(cache_dir, config_name, run_name)
    if not os.path.exists(run_path):
        return None
    return joblib.load(run_path)


@st.cache_resource(show_spinner=False)
def _build_interactive_ale(cache_dir, config_name, results_file, rep_idx,
                           K, L, centering, interpolate):
    runs = _load_all_replications(cache_dir, config_name, results_file)
    X, _y, model = runs[rep_idx]
    ale = ALE(
        model.predict, X,
        K=K, L=L, centering=centering,
        interpolate=interpolate, verbose=False,
    )
    ale.explain(include=("total_connected",))
    return ale, X, model


def _render_heatmaps(ale, signal_fn, grid_res: int = 80):
    X = ale.X_values
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), grid_res)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), grid_res)
    xx1, xx2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([xx1.ravel(), xx2.ravel()])

    z_model = ale.f(X_grid).reshape(grid_res, grid_res)
    panels = [("Trained f(x)", z_model)]
    if signal_fn is not None:
        z_true = signal_fn(X_grid).reshape(grid_res, grid_res)
        panels.append(("True signal", z_true))

    vmin = min(z.min() for _, z in panels)
    vmax = max(z.max() for _, z in panels)

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, z) in zip(axes, panels):
        mesh = ax.pcolormesh(xx1, xx2, z, shading="auto", cmap="viridis",
                             vmin=vmin, vmax=vmax)
        ax.scatter(X[:, 0], X[:, 1], s=6, facecolor="white",
                   edgecolor="black", linewidth=0.3, alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel("x1"); ax.set_ylabel("x2")
        fig.colorbar(mesh, ax=ax)
    fig.tight_layout()
    return fig


def _matched_path(ale, j: int, x_query: np.ndarray) -> int:
    x_row = x_query[0] if x_query.ndim == 2 else x_query
    indices = ale.connected_forest[j].route_and_pick_representative(
        x_row, ale.X_values
    )["indices"]
    return int(ale.observation_to_path[j][indices[0]])


def _decompose_ale_terms(ale, j: int, x_query: np.ndarray) -> dict:
    """Decompose the local ALE effect for feature j into its component terms.

    Returns a dict with:
      x_star:        the matched training observation (array of length d)
      g_left:        g*(x_j_left) — centered g-value at the left bin edge
      full_delta:    f(right_edge, x*_{-j}) - f(left_edge, x*_{-j})
      alpha_delta:   alpha * (g*(right) - g*(left)), the interpolation term
      term_path_rep: mean_{x* in path,bin} [f(x_j, x*_{-j}) - f(left, x*_{-j})]
      term_self:     f(x_j, x_{-j}) - f(left, x_{-j})
    """
    from ale.vim import local_term_path_rep as _local_term_path_rep
    from ale.shared import calculate_bin_index

    edges = ale.edges[j]
    gv = ale.centered_g_values[j]
    g = gv.centered_g_values
    K = len(edges) - 1
    interp = g.shape[0] == K + 1
    x_explain = x_query[0]
    xq = float(x_explain[j])

    # matched training point
    x_row = x_query[0] if x_query.ndim == 2 else x_query
    rep_info = ale.connected_forest[j].route_and_pick_representative(
        x_row, ale.X_values
    )
    x_star_idx = rep_info["indices"][0]
    x_star = ale.X_values[x_star_idx]
    l_match = int(ale.observation_to_path[j][x_star_idx])

    k = int(np.clip(np.searchsorted(edges, xq, side="right") - 1, 0, K - 1))

    # g*(x_j_left)
    g_left = float(g[k, l_match])

    # full delta: f(right_edge, x*_{-j}) - f(left_edge, x*_{-j})
    X_batch = np.tile(x_star, (2, 1)).astype(float)
    X_batch[0, j] = edges[k]
    X_batch[1, j] = edges[k + 1]
    f_vals = ale.f(X_batch)
    full_delta = float(f_vals[1] - f_vals[0])

    # alpha * delta (interpolation term)
    if interp:
        frac = (xq - edges[k]) / (edges[k + 1] - edges[k])
        alpha_delta = float(frac * (g[k + 1, l_match] - g[k, l_match]))
    else:
        alpha_delta = 0.0

    # self term: f(x_j, x_{-j}) - f(left, x_{-j}) using explain point's own values
    X_self = np.tile(x_explain, (2, 1)).astype(float)
    X_self[0, j] = edges[k]
    X_self[1, j] = x_explain[j]
    f_self = ale.f(X_self)
    term_self = float(f_self[1] - f_self[0])

    # path_rep term
    k_x = ale.k_x[j]
    obs_to_path = ale.observation_to_path[j]
    rep_mask = (obs_to_path == l_match) & (k_x == k)
    rep_idxs = np.where(rep_mask)[0]
    if len(rep_idxs) > 0:
        term_path_rep = float(_local_term_path_rep(
            ale.f, ale.X_values, j, x_explain, edges[k], rep_idxs
        ))
    else:
        term_path_rep = float("nan")

    return {
        "x_star": x_star,
        "g_left": g_left,
        "full_delta": full_delta,
        "alpha_delta": alpha_delta,
        "term_path_rep": term_path_rep,
        "term_self": term_self,
    }


@st.cache_resource(show_spinner=False)
def _build_shap(cache_dir, config_name, results_file, rep_idx):
    runs = _load_all_replications(cache_dir, config_name, results_file)
    X, _y, model = runs[rep_idx]
    return SHAP(model.predict, X, verbose=False)


@st.cache_data(show_spinner=False)
def _shap_at(cache_dir, config_name, results_file, rep_idx, x_tuple):
    shap = _build_shap(cache_dir, config_name, results_file, rep_idx)
    x_query = np.array([list(x_tuple)], dtype=float)
    return np.asarray(shap.explain_local(x_query, method="exact_shap"))[0]


def _render_feature_panel(ale, j: int, x_query: np.ndarray):
    from ale.shared import linear_interpolation

    edges = ale.edges[j]
    g = ale.centered_g_values[j].centered_g_values
    K = len(edges) - 1
    L = g.shape[1]
    interpolate = g.shape[0] == K + 1

    l_match = _matched_path(ale, j, x_query)

    fig, (ax_g, ax_sc) = plt.subplots(1, 2, figsize=(14, 5))

    if interpolate:
        x_axis = edges
        for l in range(L):
            alpha = 1.0 if l == l_match else 0.15
            lw = 2.5 if l == l_match else 1.0
            ax_g.plot(x_axis, g[:, l], alpha=alpha, linewidth=lw,
                      color="C0" if l == l_match else "gray")
        xq = float(x_query[0, j])
        k_star = int(np.clip(np.searchsorted(edges, xq, side="right") - 1, 0, K - 1))
        x0, x1e = edges[k_star], edges[k_star + 1]
        y0, y1 = g[k_star, l_match], g[k_star + 1, l_match]
        yq = float(linear_interpolation(xq, x0, x1e, np.array([y0]), np.array([y1])))
        ax_g.scatter([xq], [yq], color="red", s=80, zorder=5, label=f"x_{j}={xq:.2f}")
    else:
        midpoints = 0.5 * (edges[:-1] + edges[1:])
        for l in range(L):
            alpha = 1.0 if l == l_match else 0.15
            lw = 2.5 if l == l_match else 1.0
            ax_g.step(midpoints, g[:, l], where="mid", alpha=alpha, linewidth=lw,
                      color="C0" if l == l_match else "gray")
        xq = float(x_query[0, j])
        ax_g.axvline(xq, color="red", linestyle=":", alpha=0.7)

    for e in edges:
        ax_g.axvline(e, color="black", linestyle="--", alpha=0.15)
    ax_g.set_xlabel(f"x_{j}")
    ax_g.set_ylabel("centered g-value")
    ax_g.set_title(f"g-values for feature {j}  (matched path: {l_match})")
    ax_g.legend(loc="best")

    l_x = ale.observation_to_path[j]
    cmap = plt.get_cmap("tab20")
    X = ale.X_values
    dim_mask = l_x != l_match
    ax_sc.scatter(X[dim_mask, 0], X[dim_mask, 1],
                  c=[cmap(l % 20) for l in l_x[dim_mask]],
                  alpha=0.25, s=10)
    match_mask = l_x == l_match
    ax_sc.scatter(X[match_mask, 0], X[match_mask, 1],
                  c=[cmap(l_match % 20)], alpha=1.0, s=35,
                  edgecolor="black", linewidth=0.5,
                  label=f"path {l_match}")
    ax_sc.scatter([x_query[0, 0]], [x_query[0, 1]],
                  marker="*", s=260, color="red", edgecolor="black",
                  linewidth=0.8, zorder=6, label="query")

    for e in edges:
        if j == 0:
            ax_sc.axvline(e, color="black", linestyle="--", alpha=0.15)
        else:
            ax_sc.axhline(e, color="black", linestyle="--", alpha=0.15)

    ax_sc.set_xlabel("x_1"); ax_sc.set_ylabel("x_2")
    ax_sc.set_title(f"Training points — paths for feature {j}")
    ax_sc.legend(loc="best")

    fig.tight_layout()
    return fig


def _compute_pi_single_bg_info(ale, j: int, x_query: np.ndarray, bg_idx: int,
                               boundary_interp: bool = False) -> dict:
    """
    Decompose the path-integral contribution for feature j from one background point.

    Returns {'k_bg', 'k_star', 'segments', 'total'}.  Each segment has:
      type         – 'same' | 'lt_L' | 'lt_mid' | 'lt_R' | 'gt_L' | 'gt_mid' | 'gt_R'
      k            – bin index the segment corresponds to
      contribution – signed float contribution to the local effect
      bg_idx       – training index of the background point (or None for const terms)
      mid_idx      – training index routed to for intermediate bins (or None)
    """
    from ale.vim import route_first_index_at_bin as _route_numeric_first_index_known_k

    idx = j
    x_star = x_query[0].astype(float)
    x_bg   = ale.X_values[bg_idx].astype(float)

    edges   = ale.edges[j]
    K       = len(edges) - 1
    deltas  = ale.deltas[j]
    k_x     = ale.k_x[j]
    forest  = ale.connected_forest[j]
    cat_j   = ale.categorical[j]

    k_bg   = int(k_x[bg_idx])
    k_star = int(np.clip(np.searchsorted(edges, x_star[j], side="right") - 1, 0, K - 1))

    segments: list = []

    if k_bg == k_star:
        if boundary_interp:
            bw = float(edges[k_bg + 1] - edges[k_bg]) if k_bg + 1 < len(edges) else 1.0
            contrib = float((x_star[j] - x_bg[j]) / bw * deltas[bg_idx]) if bw > 0 else 0.0
        else:
            x_tmp = x_bg.copy(); x_tmp[j] = x_star[j]
            contrib = float(ale.f(x_tmp.reshape(1, -1)) - ale.f(x_bg.reshape(1, -1)))
        segments.append({'type': 'same', 'k': k_bg, 'contribution': contrib,
                         'bg_idx': bg_idx, 'mid_idx': None})

    elif k_bg < k_star:
        # L-term: endpoint at x_bg's right edge
        if boundary_interp:
            bw = float(edges[k_bg + 1] - edges[k_bg])
            contrib_L = float((edges[k_bg + 1] - x_bg[j]) / bw * deltas[bg_idx]) if bw > 0 else 0.0
        else:
            x_tmp = x_bg.copy(); x_tmp[j] = edges[k_bg + 1]
            contrib_L = float(ale.f(x_tmp.reshape(1, -1)) - ale.f(x_bg.reshape(1, -1)))
        segments.append({'type': 'lt_L', 'k': k_bg, 'contribution': contrib_L,
                         'bg_idx': bg_idx, 'mid_idx': None})

        # Middle bins
        diff_bins = k_star - k_bg
        for k_mid in range(k_bg + 1, k_star):
            if cat_j:
                alpha = (k_mid - k_bg) / diff_bins
                z = x_bg + alpha * (x_star - x_bg); z[j] = edges[k_mid]
            else:
                mid_bin = 0.5 * (edges[k_mid] + edges[k_mid + 1])
                alpha   = (mid_bin - x_star[j]) / (x_star[j] - x_bg[j])
                z = x_bg + alpha * (x_star - x_bg); z[j] = mid_bin
            i_mid = _route_numeric_first_index_known_k(forest, z, k_mid)
            segments.append({'type': 'lt_mid', 'k': k_mid,
                             'contribution': float(deltas[i_mid]),
                             'bg_idx': None, 'mid_idx': int(i_mid)})

        # R-term: constant wrt bg, depends only on x_star
        if boundary_interp:
            bw = float(edges[k_star + 1] - edges[k_star]) if k_star + 1 < len(edges) else 1.0
            i_xs = _route_numeric_first_index_known_k(forest, x_star, k_star)
            contrib_R = float((x_star[j] - edges[k_star]) / bw * deltas[i_xs]) if bw > 0 else 0.0
        else:
            x_tmp = x_star.copy(); x_tmp[j] = edges[k_star]
            contrib_R = float(ale.f(x_star.reshape(1, -1)) - ale.f(x_tmp.reshape(1, -1)))
        segments.append({'type': 'lt_R', 'k': k_star, 'contribution': contrib_R,
                         'bg_idx': None, 'mid_idx': None})

    else:
        # gt case: walking from x_star to x_bg, sign = -1
        # L-term: constant wrt bg, depends only on x_star
        if boundary_interp:
            bw = float(edges[k_star + 1] - edges[k_star]) if k_star + 1 < len(edges) else 1.0
            i_xs = _route_numeric_first_index_known_k(forest, x_star, k_star)
            raw_L = float((edges[k_star + 1] - x_star[j]) / bw * deltas[i_xs]) if bw > 0 else 0.0
        else:
            x_tmp = x_star.copy(); x_tmp[j] = edges[k_star + 1]
            raw_L = float(ale.f(x_tmp.reshape(1, -1)) - ale.f(x_star.reshape(1, -1)))
        segments.append({'type': 'gt_L', 'k': k_star, 'contribution': -raw_L,
                         'bg_idx': None, 'mid_idx': None})

        # Middle bins
        diff_bins = k_bg - k_star
        for k_mid in range(k_star + 1, k_bg):
            if cat_j:
                alpha = (k_mid - k_star) / diff_bins
                z = x_star + alpha * (x_bg - x_star); z[j] = edges[k_mid]
            else:
                mid_bin = 0.5 * (edges[k_mid] + edges[k_mid + 1])
                alpha   = (mid_bin - x_star[j]) / (x_bg[j] - x_star[j])
                z = x_star + alpha * (x_bg - x_star); z[j] = mid_bin
            i_mid = _route_numeric_first_index_known_k(forest, z, k_mid)
            segments.append({'type': 'gt_mid', 'k': k_mid,
                             'contribution': float(-deltas[i_mid]),
                             'bg_idx': None, 'mid_idx': int(i_mid)})

        # R-term: endpoint at x_bg's left edge
        if boundary_interp:
            k_plus = min(k_bg + 1, len(edges) - 1)
            bw = float(edges[k_plus] - edges[k_bg])
            contrib_R = float((x_bg[j] - edges[k_bg]) / bw * deltas[bg_idx]) if bw > 0 else 0.0
        else:
            x_tmp = x_bg.copy(); x_tmp[j] = edges[k_bg]
            contrib_R = float(ale.f(x_bg.reshape(1, -1)) - ale.f(x_tmp.reshape(1, -1)))
        segments.append({'type': 'gt_R', 'k': k_bg, 'contribution': -contrib_R,
                         'bg_idx': bg_idx, 'mid_idx': None})

    return {
        'k_bg': k_bg,
        'k_star': k_star,
        'segments': segments,
        'total': sum(s['contribution'] for s in segments),
    }


def _render_pi_feature_panel(ale, j: int, x_query: np.ndarray, bg_idx: int,
                              boundary_interp: bool):
    """Scatter + contribution bar for one feature in path-integral mode."""
    info = _compute_pi_single_bg_info(ale, j, x_query, bg_idx, boundary_interp)
    segments = info['segments']
    edges = ale.edges[j]
    X = ale.X_values
    x_bg   = X[bg_idx]
    x_star = x_query[0]

    # Collect all matched training indices to highlight
    mid_indices = [s['mid_idx'] for s in segments if s['mid_idx'] is not None]

    fig, (ax_sc, ax_bar) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Scatter: all points (dim), bg (blue square), query (red star), mid points (orange) ---
    ax_sc.scatter(X[:, 0], X[:, 1], s=8, color="gray", alpha=0.25, zorder=1)
    ax_sc.scatter([x_bg[0]], [x_bg[1]], marker="s", s=200, color="C0",
                  edgecolor="black", linewidth=0.8, zorder=4, label=f"x_bg (idx {bg_idx})")
    ax_sc.scatter([x_star[0]], [x_star[1]], marker="*", s=280, color="red",
                  edgecolor="black", linewidth=0.8, zorder=5, label="x_query")
    if mid_indices:
        mid_segs = [s for s in segments if s['mid_idx'] is not None]
        for s in mid_segs:
            pt = X[s['mid_idx']]
            ax_sc.scatter([pt[0]], [pt[1]], marker="D", s=120, color="C1",
                          edgecolor="black", linewidth=0.5, zorder=3)
            ax_sc.annotate(f"k={s['k']}", xy=(pt[0], pt[1]), xytext=(4, 4),
                           textcoords="offset points", fontsize=7)

    for e in edges:
        if j == 0:
            ax_sc.axvline(e, color="black", linestyle="--", alpha=0.15)
        else:
            ax_sc.axhline(e, color="black", linestyle="--", alpha=0.15)
    # highlight query bin edge pair
    k_star = info['k_star']
    k_bg_v = info['k_bg']
    for k_hi, col in [(k_star, "red"), (k_bg_v, "C0")]:
        lo, hi = edges[k_hi], edges[k_hi + 1] if k_hi + 1 < len(edges) else edges[-1]
        if j == 0:
            ax_sc.axvspan(lo, hi, alpha=0.08, color=col)
        else:
            ax_sc.axhspan(lo, hi, alpha=0.08, color=col)

    ax_sc.set_xlabel("x_1"); ax_sc.set_ylabel("x_2")
    ax_sc.set_title(f"Feature {j} path  (k_bg={k_bg_v} → k*={k_star})")
    ax_sc.legend(loc="best", fontsize=8)

    # --- Bar chart of per-segment contributions ---
    labels = [f"{s['type']}\nk={s['k']}" for s in segments]
    values = [s['contribution'] for s in segments]
    colors = []
    for s in segments:
        if s['type'] == 'same':       colors.append("C2")
        elif s['type'] in ('lt_L', 'gt_R'): colors.append("C0")   # bg endpoint
        elif s['type'] in ('lt_mid', 'gt_mid'): colors.append("C1")  # intermediate
        else:                         colors.append("gray")          # const R/L term
    bars = ax_bar.barh(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    ax_bar.axvline(0, color="black", linewidth=0.8)
    total = info['total']
    ax_bar.set_title(f"Feature {j} contributions  (total={total:.4f})")
    ax_bar.set_xlabel("Contribution")
    for bar, val in zip(bars, values):
        ax_bar.text(val + (0.002 if val >= 0 else -0.002), bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", ha="left" if val >= 0 else "right", fontsize=7)

    fig.tight_layout()
    return fig, info


def _render_path_integral_section(cache_dir, config_name, results_file, row_meta) -> None:
    runs = _load_all_replications(cache_dir, config_name, results_file)
    if runs is None:
        st.warning("Could not locate the run_*.pkl companion file for this experiment.")
        return
    R = len(runs)

    ukey = f"pi_{config_name}_{results_file}"

    rep_idx = st.number_input(
        "Replication index", min_value=0, max_value=max(R - 1, 0),
        value=0, step=1, key=f"rep_{ukey}",
    )

    cols = st.columns(5)
    with cols[0]:
        K = st.number_input("K", min_value=2, value=int(row_meta.get("K") or 10),
                            step=1, key=f"K_{ukey}")
    with cols[1]:
        L = st.number_input("L", min_value=1, value=int(row_meta.get("L") or 10),
                            step=1, key=f"L_{ukey}")
    with cols[2]:
        default_centering = str(row_meta.get("centering") or "y")
        centering = st.selectbox(
            "centering", options=["y", "x"],
            index=0 if default_centering == "y" else 1,
            key=f"centering_{ukey}",
        )
    with cols[3]:
        interpolate = st.checkbox("interpolate", value=True, key=f"interp_{ukey}")
    with cols[4]:
        boundary_interp = st.checkbox("boundary_interp", value=False,
                                      key=f"bi_{ukey}")

    with st.spinner("Fitting ALE…"):
        ale, X, model = _build_interactive_ale(
            cache_dir, config_name, results_file, int(rep_idx),
            int(K), int(L), centering, bool(interpolate), 0,
        )

    n = ale.X_values.shape[0]

    st.subheader("Query point")
    qcols = st.columns(2)
    with qcols[0]:
        x1q = st.number_input("x_1", value=0.0, step=0.1,
                               key=f"x1_{ukey}", format="%.4f")
    with qcols[1]:
        x2q = st.number_input("x_2", value=0.0, step=0.1,
                               key=f"x2_{ukey}", format="%.4f")
    x_query = np.array([[float(x1q), float(x2q)]])

    bg_idx = st.number_input(
        f"Background point index (0 – {n - 1})",
        min_value=0, max_value=n - 1, value=0, step=1,
        key=f"bg_{ukey}",
    )
    bg_idx = int(bg_idx)
    x_bg = ale.X_values[bg_idx]
    st.caption(
        f"x_bg = ({', '.join(f'{v:.4f}' for v in x_bg)})  "
        f"  |  x_query = ({x1q:.4f}, {x2q:.4f})"
    )

    with st.expander("Per-feature path decomposition", expanded=True):
        for j in range(2):
            st.subheader(f"Feature {j}")
            try:
                with st.spinner(f"Computing feature {j}…"):
                    fig, info = _render_pi_feature_panel(
                        ale, j, x_query, bg_idx, bool(boundary_interp)
                    )
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.warning(f"Feature {j} decomposition failed: {e}")

    st.subheader("Explanation comparison at query")
    comparison_rows = []
    for j in range(2):
        comparison_rows.append({"feature": f"x_{j}", "path_integral": float("nan"),
                                 "ALE (path_rep)": float("nan"), "SHAP": float("nan")})

    try:
        pi_vals = ale.explain_local(
            x_query,
            local_method="path_integral",
            boundary_interp=bool(boundary_interp),
        )[0]
        for j in range(2):
            comparison_rows[j]["path_integral"] = float(pi_vals[j])
    except Exception as e:
        st.caption(f"explain_local (path_integral) failed: {e}")

    try:
        pr_vals = ale.explain_local(x_query, local_method="path_rep")[0]
        for j in range(2):
            comparison_rows[j]["ALE (path_rep)"] = float(pr_vals[j])
    except Exception as e:
        st.caption(f"explain_local (path_rep) failed: {e}")

    try:
        shap_vals = _shap_at(cache_dir, config_name, results_file, int(rep_idx),
                             (float(x1q), float(x2q)))
        for j in range(2):
            comparison_rows[j]["SHAP"] = float(shap_vals[j])
    except Exception as e:
        st.caption(f"SHAP failed: {e}")

    st.dataframe(pd.DataFrame(comparison_rows).set_index("feature"))


def _compute_mp_single_bg_info(ale, j: int, x_query: np.ndarray, bg_idx: int) -> dict:
    """
    Decompose the multi_path contribution for feature j from one background point.

    Multi_path is always boundary_interp-style (no f calls): boundary terms use
    `alpha * leaf_mean_delta` instead of f-differences, and middle bins assign
    one tree node (possibly internal) per bin using depth-interpolation between
    the bg's leaf A and x*'s leaf B; the node's per-bin mean delta is the
    contribution.

    Returns {'k_bg', 'k_star', 'leaf_A', 'leaf_B', 'tree_path', 'segments', 'total'}.
    Each segment has:
      type         – 'same' | 'lt_L' | 'lt_mid' | 'lt_R' | 'gt_L' | 'gt_mid' | 'gt_R'
      k            – bin index the segment corresponds to
      contribution – signed float contribution to the local effect
      bg_idx       – training index of the bg point (only for bg-endpoint segments)
      node         – the assigned KDNode (for mid + endpoint segments)
      indices      – ndarray of UNIQUE training indices in the assigned node's
                     subtree restricted to bin k (used for scatter highlighting)
      obs_weights  – ndarray aligned with `indices`: each entry is this
                     observation's signed weight contribution to this segment
                     (sum of segment-coef/n_per_bin distributed across leaf
                     occurrences). Sum equals `contribution / mean_delta[k]`.
      node_is_leaf – bool
    """
    from ale.tree_partitioning import iter_subtree_indices_at_bin
    x_star = x_query[0].astype(float)
    x_bg = ale.X_values[bg_idx].astype(float)

    edges = ale.edges[j]
    K = len(edges) - 1
    forest = ale.connected_forest[j]
    k_x = ale.k_x[j]

    k_bg = int(k_x[bg_idx])
    k_star = int(np.clip(np.searchsorted(edges, x_star[j], side="right") - 1, 0, K - 1))

    # Route both points through the forest to get leaves + descent paths.
    bg_info = forest.route_with_path(x_bg, k=k_bg)
    star_info = forest.route_with_path(x_star, k=k_star)
    leaf_A = bg_info["leaf"]
    leaf_B = star_info["leaf"]
    path_A = bg_info["path"]
    path_B = star_info["path"]
    tree_path = forest.tree_path_between(path_A, path_B)

    bw_xstar = float(edges[k_star + 1] - edges[k_star]) if k_star + 1 < len(edges) else 0.0
    bw_bg = float(edges[k_bg + 1] - edges[k_bg]) if k_bg + 1 < len(edges) else 0.0

    def _idxs(node, k):
        return forest._collect_leaf_indices(node, k)

    def _safe_mean(node, k):
        v = float(node.mean_delta[k])
        return 0.0 if np.isnan(v) else v

    def _obs_weights(node, k, coef):
        """Per-observation signed weight contribution from a single
        `coef * node.mean_delta[k]` term. Distributes `coef / n_per_bin[k]`
        across all leaf occurrences in the subtree, then collapses to a
        unique-index view (multiplicities folded into the weight).

        Returns (unique_indices, weights_aligned) with shape (m,), (m,).
        """
        n_k = int(node.n_per_bin[k])
        if n_k == 0 or coef == 0.0:
            return np.empty(0, dtype=int), np.empty(0, dtype=float)
        occ = iter_subtree_indices_at_bin(node, k)
        if occ.size == 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=float)
        uniq, counts = np.unique(occ, return_counts=True)
        return uniq, (coef / n_k) * counts.astype(float)

    segments: list = []

    def _push_segment(seg_type, k, coef, node, sign=1.0, bg_idx_field=None):
        signed_coef = sign * coef
        mean_d = _safe_mean(node, k)
        contribution = float(signed_coef * mean_d)
        uniq, w = _obs_weights(node, k, signed_coef)
        segments.append({
            'type': seg_type,
            'k': k,
            'contribution': contribution,
            'bg_idx': bg_idx_field,
            'node': node,
            'indices': uniq,
            'obs_weights': w,
            'node_is_leaf': bool(node.is_leaf),
        })

    if k_bg == k_star:
        coef = (x_star[j] - x_bg[j]) / bw_xstar if bw_xstar > 0 else 0.0
        _push_segment('same', k_bg, coef, leaf_A, bg_idx_field=bg_idx)

    elif k_bg < k_star:
        alpha_L = (edges[k_bg + 1] - x_bg[j]) / bw_bg if bw_bg > 0 else 0.0
        alpha_R = (x_star[j] - edges[k_star]) / bw_xstar if bw_xstar > 0 else 0.0

        _push_segment('lt_L', k_bg, alpha_L, leaf_A, bg_idx_field=bg_idx)

        M = k_star - k_bg - 1
        if M > 0:
            nodes = forest.assign_middle_nodes(tree_path, M)
            for off, node in enumerate(nodes):
                k_mid = k_bg + 1 + off
                _push_segment('lt_mid', k_mid, 1.0, node)

        _push_segment('lt_R', k_star, alpha_R, leaf_B)

    else:
        # gt: walk x* → x_bg, sign = -1
        k_bg_plus = min(k_bg + 1, len(edges) - 1)
        bw_bg_gt = float(edges[k_bg_plus] - edges[k_bg]) if k_bg_plus > k_bg else 0.0
        alpha_R = (x_bg[j] - edges[k_bg]) / bw_bg_gt if bw_bg_gt > 0 else 0.0
        alpha_L = (edges[k_star + 1] - x_star[j]) / bw_xstar if bw_xstar > 0 else 0.0

        _push_segment('gt_L', k_star, alpha_L, leaf_B, sign=-1.0)

        M = k_bg - k_star - 1
        if M > 0:
            rev_path = list(reversed(tree_path))
            nodes = forest.assign_middle_nodes(rev_path, M)
            for off, node in enumerate(nodes):
                k_mid = k_star + 1 + off
                _push_segment('gt_mid', k_mid, 1.0, node, sign=-1.0)

        _push_segment('gt_R', k_bg, alpha_R, leaf_A, sign=-1.0,
                      bg_idx_field=bg_idx)

    return {
        'k_bg': k_bg,
        'k_star': k_star,
        'leaf_A': leaf_A,
        'leaf_B': leaf_B,
        'tree_path': tree_path,
        'segments': segments,
        'total': float(sum(s['contribution'] for s in segments)),
    }


def _render_mp_feature_panel(ale, j: int, x_query: np.ndarray, bg_idx: int):
    """Scatter + contribution bar for one feature in multi_path mode.

    The scatter shades subtree members for each segment's assigned tree node
    (multiple training points when the node is internal — visualizing the
    "averaging more" behavior of higher ancestors).
    """
    info = _compute_mp_single_bg_info(ale, j, x_query, bg_idx)
    segments = info['segments']
    edges = ale.edges[j]
    X = ale.X_values
    x_bg = X[bg_idx]
    x_star = x_query[0]

    fig, (ax_sc, ax_bar) = plt.subplots(1, 2, figsize=(14, 5))

    # All training points (dim).
    ax_sc.scatter(X[:, 0], X[:, 1], s=8, color="gray", alpha=0.20, zorder=1)

    # Distinct colors per middle segment to show progression through tree.
    mid_segs = [s for s in segments if s['type'] in ('lt_mid', 'gt_mid')]
    n_mid = len(mid_segs)
    cmap = plt.get_cmap("plasma")

    # Find a per-figure scale so the largest |obs_weight| renders at a
    # readable size; small weights still get a baseline so they are visible.
    all_w = np.concatenate(
        [np.abs(s.get('obs_weights', np.empty(0))) for s in segments]
    ) if segments else np.empty(0)
    w_max = float(all_w.max()) if all_w.size and all_w.max() > 0 else 1.0
    BASE = 18.0
    SCALE = 180.0

    def _sizes(weights):
        return BASE + SCALE * (np.abs(weights) / w_max)

    # Highlight bg-leaf and query-leaf subtree members at their respective bins.
    for s in segments:
        if s['type'] in ('lt_L', 'gt_R', 'same'):
            color, marker = "C0", "o"
        elif s['type'] in ('lt_R', 'gt_L'):
            color, marker = "C3", "o"
        else:
            continue
        idxs = s['indices']
        weights = s.get('obs_weights', np.zeros(idxs.size))
        if idxs.size:
            ax_sc.scatter(
                X[idxs, 0], X[idxs, 1], s=_sizes(weights), color=color, alpha=0.55,
                edgecolor="black", linewidth=0.4, zorder=2, marker=marker,
            )

    # Middle nodes: color-coded by their position in the walk; size scaled
    # by each observation's signed weight contribution (replaces old
    # subtree-size proxy so dots reflect actual importance).
    for i, s in enumerate(mid_segs):
        idxs = s['indices']
        weights = s.get('obs_weights', np.zeros(idxs.size))
        if not idxs.size:
            continue
        col = cmap(i / max(n_mid - 1, 1))
        ax_sc.scatter(
            X[idxs, 0], X[idxs, 1], s=_sizes(weights), color=col, alpha=0.65,
            edgecolor="black", linewidth=0.4, zorder=3,
            marker="D" if s['node_is_leaf'] else "P",
        )
        # Label at centroid.
        cx, cy = float(X[idxs, 0].mean()), float(X[idxs, 1].mean())
        tag = f"k={s['k']}" + ("" if s['node_is_leaf'] else f" (n={idxs.size})")
        ax_sc.annotate(tag, xy=(cx, cy), xytext=(4, 4),
                       textcoords="offset points", fontsize=7, color="black")

    ax_sc.scatter([x_bg[0]], [x_bg[1]], marker="s", s=200, color="C0",
                  edgecolor="black", linewidth=0.8, zorder=5,
                  label=f"x_bg (idx {bg_idx})")
    ax_sc.scatter([x_star[0]], [x_star[1]], marker="*", s=280, color="red",
                  edgecolor="black", linewidth=0.8, zorder=6, label="x_query")

    for e in edges:
        if j == 0:
            ax_sc.axvline(e, color="black", linestyle="--", alpha=0.15)
        else:
            ax_sc.axhline(e, color="black", linestyle="--", alpha=0.15)

    k_star_v = info['k_star']
    k_bg_v = info['k_bg']
    for k_hi, col in [(k_star_v, "red"), (k_bg_v, "C0")]:
        lo = edges[k_hi]
        hi = edges[k_hi + 1] if k_hi + 1 < len(edges) else edges[-1]
        if j == 0:
            ax_sc.axvspan(lo, hi, alpha=0.08, color=col)
        else:
            ax_sc.axhspan(lo, hi, alpha=0.08, color=col)

    leaf_A_depth = _node_depth(info['leaf_A'])
    leaf_B_depth = _node_depth(info['leaf_B'])
    D = len(info['tree_path']) - 1
    M = max(0, abs(k_star_v - k_bg_v) - 1)
    ax_sc.set_xlabel("x_1"); ax_sc.set_ylabel("x_2")
    ax_sc.set_title(
        f"Feature {j}  (k_bg={k_bg_v} → k*={k_star_v}; "
        f"depth A={leaf_A_depth}, B={leaf_B_depth}, D={D}, M={M})"
    )
    ax_sc.legend(loc="best", fontsize=8)

    # --- Bar chart ---
    labels = [f"{s['type']}\nk={s['k']}" for s in segments]
    values = [s['contribution'] for s in segments]
    colors = []
    for s in segments:
        if s['type'] == 'same':                    colors.append("C2")
        elif s['type'] in ('lt_L', 'gt_R'):        colors.append("C0")
        elif s['type'] in ('lt_R', 'gt_L'):        colors.append("C3")
        else:                                       colors.append("C1")
    bars = ax_bar.barh(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    ax_bar.axvline(0, color="black", linewidth=0.8)
    ax_bar.set_title(f"Feature {j} contributions  (total={info['total']:.4f})")
    ax_bar.set_xlabel("Contribution")
    for bar, val in zip(bars, values):
        ax_bar.text(val + (0.002 if val >= 0 else -0.002),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center",
                    ha="left" if val >= 0 else "right", fontsize=7)

    fig.tight_layout()
    return fig, info


def _render_mp_query_weight_panel(ale, x_query: np.ndarray, local_method: str):
    """Per-feature scatter showing each training observation's signed weight
    contribution to the local explanation at `x_query`. Weights come from
    `ALE.explain_local_weights`, which exposes the linear-combination view of
    the multi_path / multi_path_interpolate estimator over deltas.
    """
    d = ale.X_values.shape[1]
    W = ale.explain_local_weights(x_query, local_method=local_method)[0]  # (d, n_train)
    X = ale.X_values
    deltas = [ale.deltas[j] for j in range(d)]

    n_cols = d
    fig, axes = plt.subplots(1, n_cols, figsize=(5.0 * n_cols, 4.5), squeeze=False)
    axes = axes[0]
    for j in range(d):
        ax = axes[j]
        w = W[j]
        recon = float(w @ deltas[j])

        wmax = float(np.abs(w).max())
        norm = mcolors.TwoSlopeNorm(vmin=-wmax if wmax > 0 else -1.0,
                                    vcenter=0.0,
                                    vmax=wmax if wmax > 0 else 1.0)
        size_scale = (np.abs(w) / wmax) * 120.0 + 8.0 if wmax > 0 else np.full_like(w, 8.0)
        sc = ax.scatter(X[:, 0], X[:, 1], c=w, cmap="coolwarm", norm=norm,
                        s=size_scale, alpha=0.85, edgecolor="black",
                        linewidth=0.3, zorder=2)
        ax.scatter([x_query[0, 0]], [x_query[0, 1]], marker="*", s=260,
                   color="gold", edgecolor="black", linewidth=0.8, zorder=4,
                   label="x_query")
        ax.set_xlabel("x_1"); ax.set_ylabel("x_2")
        ax.set_title(f"Feature {j}  W·delta = {recon:.4f}")
        ax.legend(loc="best", fontsize=8)
        fig.colorbar(sc, ax=ax, shrink=0.85, label="weight on delta[obs]")

    fig.suptitle(f"explain_local_weights ({local_method}) at x_query", fontsize=11)
    fig.tight_layout()
    return fig


def _node_depth(node) -> int:
    """Tree depth of a node (root has depth 0). Requires parent pointers."""
    d = 0
    cur = node
    while getattr(cur, "parent", None) is not None:
        cur = cur.parent
        d += 1
    return d


def _render_multi_path_section(cache_dir, config_name, results_file, row_meta) -> None:
    runs = _load_all_replications(cache_dir, config_name, results_file)
    if runs is None:
        st.warning("Could not locate the run_*.pkl companion file for this experiment.")
        return
    R = len(runs)

    ukey = f"mp_{config_name}_{results_file}"

    rep_idx = st.number_input(
        "Replication index", min_value=0, max_value=max(R - 1, 0),
        value=0, step=1, key=f"rep_{ukey}",
    )

    cols = st.columns(4)
    with cols[0]:
        K = st.number_input("K", min_value=2, value=int(row_meta.get("K") or 10),
                            step=1, key=f"K_{ukey}")
    with cols[1]:
        L = st.number_input("L", min_value=1, value=int(row_meta.get("L") or 10),
                            step=1, key=f"L_{ukey}")
    with cols[2]:
        default_centering = str(row_meta.get("centering") or "y")
        centering = st.selectbox(
            "centering", options=["y", "x"],
            index=0 if default_centering == "y" else 1,
            key=f"centering_{ukey}",
        )
    with cols[3]:
        interpolate = st.checkbox("interpolate", value=True, key=f"interp_{ukey}")

    with st.spinner("Fitting ALE…"):
        ale, X, model = _build_interactive_ale(
            cache_dir, config_name, results_file, int(rep_idx),
            int(K), int(L), centering, bool(interpolate), 0,
        )

    n = ale.X_values.shape[0]

    st.subheader("Query point")
    qcols = st.columns(2)
    with qcols[0]:
        x1q = st.number_input("x_1", value=0.0, step=0.1,
                              key=f"x1_{ukey}", format="%.4f")
    with qcols[1]:
        x2q = st.number_input("x_2", value=0.0, step=0.1,
                              key=f"x2_{ukey}", format="%.4f")
    x_query = np.array([[float(x1q), float(x2q)]])

    bg_idx = st.number_input(
        f"Background point index (0 – {n - 1})",
        min_value=0, max_value=n - 1, value=0, step=1,
        key=f"bg_{ukey}",
    )
    bg_idx = int(bg_idx)
    x_bg = ale.X_values[bg_idx]
    st.caption(
        f"x_bg = ({', '.join(f'{v:.4f}' for v in x_bg)})  "
        f"  |  x_query = ({x1q:.4f}, {x2q:.4f})"
    )

    with st.expander("Per-feature multi_path decomposition", expanded=True):
        for j in range(2):
            st.subheader(f"Feature {j}")
            try:
                with st.spinner(f"Computing feature {j}…"):
                    fig, info = _render_mp_feature_panel(ale, j, x_query, bg_idx)
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.warning(f"Feature {j} decomposition failed: {e}")

    with st.expander(
        "Full weighting over training observations (explain_local_weights)",
        expanded=False,
    ):
        weight_method = "multi_path_interpolate"
        try:
            with st.spinner(f"Computing per-observation weights ({weight_method})…"):
                fig_w = _render_mp_query_weight_panel(ale, x_query, weight_method)
            st.pyplot(fig_w)
            plt.close(fig_w)
        except Exception as e:
            st.warning(f"Full-weighting panel failed: {e}")

    st.subheader("Explanation comparison at query")
    comparison_rows = [
        {"feature": f"x_{j}", "multi_path_interpolate": float("nan"),
         "path_integral (bi=True)": float("nan"),
         "ALE (path_rep)": float("nan"), "SHAP": float("nan")}
        for j in range(2)
    ]

    try:
        mp_vals = ale.explain_local(x_query, local_method="multi_path_interpolate")[0]
        for j in range(2):
            comparison_rows[j]["multi_path_interpolate"] = float(mp_vals[j])
    except Exception as e:
        st.caption(f"explain_local (multi_path_interpolate) failed: {e}")

    try:
        pi_vals = ale.explain_local(
            x_query, local_method="path_integral", boundary_interp=True
        )[0]
        for j in range(2):
            comparison_rows[j]["path_integral (bi=True)"] = float(pi_vals[j])
    except Exception as e:
        st.caption(f"explain_local (path_integral) failed: {e}")

    try:
        pr_vals = ale.explain_local(x_query, local_method="path_rep")[0]
        for j in range(2):
            comparison_rows[j]["ALE (path_rep)"] = float(pr_vals[j])
    except Exception as e:
        st.caption(f"explain_local (path_rep) failed: {e}")

    try:
        shap_vals = _shap_at(cache_dir, config_name, results_file, int(rep_idx),
                             (float(x1q), float(x2q)))
        for j in range(2):
            comparison_rows[j]["SHAP"] = float(shap_vals[j])
    except Exception as e:
        st.caption(f"SHAP failed: {e}")

    st.dataframe(pd.DataFrame(comparison_rows).set_index("feature"))


def _render_interactive_section(cache_dir, config_name, results_file, row_meta) -> None:

    runs = _load_all_replications(cache_dir, config_name, results_file)
    if runs is None:
        st.warning("Could not locate the run_*.pkl companion file for this experiment.")
        return
    R = len(runs)

    ukey = f"{config_name}_{results_file}"

    rep_idx = st.number_input(
        "Replication index", min_value=0, max_value=max(R - 1, 0),
        value=0, step=1, key=f"rep_{ukey}",
    )

    cols = st.columns(4)
    with cols[0]:
        K = st.number_input("K", min_value=2, value=int(row_meta.get("K") or 10),
                            step=1, key=f"K_{ukey}")
    with cols[1]:
        L = st.number_input("L", min_value=1, value=int(row_meta.get("L") or 10),
                            step=1, key=f"L_{ukey}")
    with cols[2]:
        default_centering = str(row_meta.get("centering") or "y")
        centering = st.selectbox(
            "centering", options=["y", "x"],
            index=0 if default_centering == "y" else 1,
            key=f"centering_{ukey}",
        )
    with cols[3]:
        interpolate = st.checkbox("interpolate", value=True, key=f"interp_{ukey}")

    with st.spinner("Fitting ALE…"):
        ale, X, model = _build_interactive_ale(
            cache_dir, config_name, results_file, int(rep_idx),
            int(K), int(L), centering, bool(interpolate),
        )

    _X, y_rep, _m = runs[int(rep_idx)]
    r2_train = float(r2_score(y_rep, model.predict(X)))
    subdir = os.path.join(cache_dir, config_name)
    tune = _load_tune(row_meta, subdir) if row_meta else {}
    cv_r2 = tune.get("cv_r2")
    max_r2 = tune.get("max_r2")

    mcols = st.columns(3)
    mcols[0].metric("Model R² (train)", f"{r2_train:.4f}")
    mcols[1].metric("CV R² (cached)",
                    f"{cv_r2:.4f}" if cv_r2 is not None else "N/A")
    mcols[2].metric("Max R² (SNR bound)",
                    f"{max_r2:.4f}" if max_r2 is not None else "N/A")

    signal_fn = None
    signal_name = row_meta.get("signal")
    if signal_name:
        signal_fn = getattr(models, signal_name, None)
        if signal_fn is None:
            st.caption(f"True signal `{signal_name}` not found in `models`; showing trained f only.")

    with st.expander("Heatmaps: trained f(x) vs true signal", expanded=True):
        st.pyplot(_render_heatmaps(ale, signal_fn))

    st.subheader("Query point")
    qcols = st.columns(2)
    with qcols[0]:
        x1q = st.number_input("x_1", value=0.0, step=0.1,
                              key=f"x1_{ukey}", format="%.4f")
    with qcols[1]:
        x2q = st.number_input("x_2", value=0.0, step=0.1,
                              key=f"x2_{ukey}", format="%.4f")
    x_query = np.array([[float(x1q), float(x2q)]])

    with st.expander("Feature g-values and paths", expanded=True):
        for j in range(2):
            st.subheader(f"Feature {j}")
            st.pyplot(_render_feature_panel(ale, j, x_query))

    st.subheader("Local-effect terms at query")

    # Decompose per-feature terms
    decomp = []
    for j in range(2):
        try:
            decomp.append(_decompose_ale_terms(ale, j, x_query))
        except Exception as e:
            st.caption(f"Decomposition for feature {j} failed: {e}")
            decomp.append(None)

    # Show matched x* for each feature
    for j in range(2):
        if decomp[j] is not None:
            xs = decomp[j]["x_star"]
            st.caption(
                f"Feature {j} matched x\\* = "
                f"({', '.join(f'{v:.4f}' for v in xs)})"
            )

    rows = []

    # Final ALE values for surviving methods
    for method in ("path_rep", "path_integral"):
        try:
            terms = ale.explain_local(
                x_query, local_method=method
            )[0]
            rows.append({"method": f"ALE ({method})",
                         "feature_1": float(terms[0]),
                         "feature_2": float(terms[1])})
        except ValueError as e:
            rows.append({"method": f"ALE ({method})",
                         "feature_1": float("nan"),
                         "feature_2": float("nan")})
            st.caption(f"`{method}` raised: {e}")

    # Decomposition rows
    def _d(key):
        return [float(decomp[j][key]) if decomp[j] else float("nan") for j in range(2)]

    rows.append({"method": "g*(x_j_left)",
                 "feature_1": _d("g_left")[0], "feature_2": _d("g_left")[1]})
    rows.append({"method": "full delta: f(right, x*_{-j}) - f(left, x*_{-j})",
                 "feature_1": _d("full_delta")[0], "feature_2": _d("full_delta")[1]})
    rows.append({"method": "alpha * delta (interpolate term)",
                 "feature_1": _d("alpha_delta")[0], "feature_2": _d("alpha_delta")[1]})
    rows.append({"method": "path_rep term",
                 "feature_1": _d("term_path_rep")[0], "feature_2": _d("term_path_rep")[1]})
    rows.append({"method": "self term",
                 "feature_1": _d("term_self")[0], "feature_2": _d("term_self")[1]})

    try:
        shap_vals = _shap_at(cache_dir, config_name, results_file, int(rep_idx),
                             (float(x1q), float(x2q)))
        rows.append({"method": "SHAP (exact)",
                     "feature_1": float(shap_vals[0]),
                     "feature_2": float(shap_vals[1])})
    except Exception as e:
        rows.append({"method": "SHAP (exact)",
                     "feature_1": float("nan"),
                     "feature_2": float("nan")})
        st.caption(f"SHAP failed: {e}")

    true_fn = None
    if signal_name:
        true_fn = getattr(models, f"{signal_name}_explanation", None)
    if true_fn is not None:
        try:
            true_vals = np.asarray(true_fn(x_query))[0]
            rows.append({"method": "True explanation",
                         "feature_1": float(true_vals[0]),
                         "feature_2": float(true_vals[1])})
        except Exception as e:
            rows.append({"method": "True explanation",
                         "feature_1": float("nan"),
                         "feature_2": float("nan")})
            st.caption(f"True explanation failed: {e}")
    else:
        st.caption(f"No `{signal_name}_explanation` in `models`; true explanation unavailable.")

    st.dataframe(pd.DataFrame(rows).set_index("method"))


# ---------------------------------------------------------------------------
# Bias / stddev summary table (grid-point IQR)
# ---------------------------------------------------------------------------

def _plot_bias_hist(view: dict, true_explanation_fn):
    """Histogram of bias over grid points, one subplot per feature.

    exps has shape (R, n_grid, d). bias(g, j) = mean_R(exp[r, g, j]) - true(g, j).
    Returns a matplotlib figure, or None if true_explanation_fn is None.
    """
    if true_explanation_fn is None:
        return None
    ale_exps = view["ale_exps"]
    shap_exps = view["shap_exps"]
    explain_grid = view["explain_grid"]
    d = explain_grid.shape[1]
    true_exp = true_explanation_fn(explain_grid)

    fig, axes = plt.subplots(1, d, figsize=(6 * d, 4), squeeze=False)
    fig.suptitle("Bias (over grid points)")
    for j in range(d):
        ax = axes[0, j]
        for method, exps, color in [("ALE", ale_exps, "C0"), ("SHAP", shap_exps, "C1")]:
            bias = (exps.mean(axis=0) - true_exp)[:, j]  # (n_grid,)
            ax.hist(bias, bins=40, alpha=0.5, color=color, label=method)
            ax.axvline(bias.mean(), color=color, linestyle=":", linewidth=2)
        ax.set_xlabel("Bias")
        ax.set_ylabel("Count")
        ax.set_title(f"Feature {j}")
        ax.legend()
    fig.tight_layout()
    return fig


def _plot_stddev_hist(view: dict):
    """Histogram of std dev over grid points, one subplot per feature.

    exps has shape (R, n_grid, d). stddev(g, j) = std_R(exp[r, g, j]).
    Also overlays Std(f(x)) if available. Returns (fig, f_std_mean).
    """
    ale_exps = view["ale_exps"]
    shap_exps = view["shap_exps"]
    f_vals = view["f_vals"]
    d = ale_exps.shape[-1]

    fig, axes = plt.subplots(1, d, figsize=(6 * d, 4), squeeze=False)
    fig.suptitle("Std Dev (over grid points)")
    for j in range(d):
        ax = axes[0, j]
        for method, exps, color in [("ALE", ale_exps, "C0"), ("SHAP", shap_exps, "C1")]:
            std_r = exps.std(axis=0)[:, j]  # (n_grid,)
            ax.hist(std_r, bins=40, alpha=0.5, color=color, label=method)
            ax.axvline(std_r.mean(), color=color, linestyle=":", linewidth=2)
        if f_vals is not None:
            f_std = f_vals.std(axis=0)  # (n_grid,)
            ax.axvline(f_std.mean(), color="C2", linestyle=":", linewidth=2,
                       label="Std(f)")
        ax.set_xlabel("Std Dev")
        ax.set_ylabel("Count")
        ax.set_title(f"Feature {j}")
        ax.legend()
    fig.tight_layout()

    f_std_mean = float(f_vals.std(axis=0).mean()) if f_vals is not None else None
    return fig, f_std_mean


def _plot_additivity_hist(view: dict):
    """Histogram of the per-(replication, grid-point) additivity residual:
        residual = sum_d phi_d(x) - (f(x) - E[f(X)])
    One overlaid histogram for ALE and SHAP. Returns (fig, info_dict) or
    (None, None) if f_means is unavailable on the cached results.
    """
    f_vals = view.get("f_vals")
    f_means = view.get("f_means")
    if f_vals is None or f_means is None:
        return None, None

    target = f_vals - np.asarray(f_means)[:, None]  # (R, n_grid)
    fig, ax = plt.subplots(figsize=(8, 4))
    info = {}
    for method, exps, color in [
        ("ALE", view["ale_exps"], "C0"),
        ("SHAP", view["shap_exps"], "C1"),
    ]:
        residual = exps.sum(axis=-1) - target  # (R, n_grid)
        flat = residual.ravel()
        rmse = float(np.sqrt(np.mean(flat ** 2)))
        ax.hist(flat, bins=60, alpha=0.5, color=color,
                label=f"{method} (RMSE={rmse:.4f})")
        ax.axvline(flat.mean(), color=color, linestyle=":", linewidth=2)
        info[method] = rmse
    ax.set_xlabel("sum_d φ_d(x) − (f(x) − E[f(X)])")
    ax.set_ylabel("Count")
    ax.set_title("Additivity Residual (over replications × grid points)")
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax.legend()
    fig.tight_layout()
    return fig, info


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

_FILTER_SPECS = [
    ("Config",      "config_name"),
    ("Signal",      "signal"),
    ("SNR",         "snr"),
    ("Rho",         "rho"),
    ("Variant",     "variant"),
    ("LocalMethod", "local_method"),
    ("SHAP Tag",    "shap_tag"),
]


def show_summary_page(cache_dir: str) -> None:
    st.title("Experiment Summary")

    rows = load_all_rows(cache_dir)
    if not rows:
        st.error(f"No results_*.pkl files found under {cache_dir!r}")
        return

    st.subheader("Filters")
    filter_selections = {}
    cols = st.columns(3)
    for i, (label, key) in enumerate(_FILTER_SPECS):
        raw = {r.get(key) for r in rows if r.get(key) is not None}
        raw = {v for v in raw if not (isinstance(v, float) and np.isnan(v))}
        options = sorted(raw, key=lambda v: (isinstance(v, str), v))
        with cols[i % 3]:
            filter_selections[key] = st.multiselect(
                label, options=options, default=options, key=f"filter_{key}",
            )

    def _match(r):
        for _, key in _FILTER_SPECS:
            if r.get(key) not in filter_selections[key]:
                return False
        return True

    filtered = [r for r in rows if _match(r)]

    # Group filtered rows by (config_name, results_file)
    groups: OrderedDict[tuple, list] = OrderedDict()
    for r in filtered:
        gkey = (r["config_name"], r["_results_file"])
        groups.setdefault(gkey, []).append(r)

    # --- Sort control ---
    hdr_col, sort_col = st.columns([5, 2])
    with hdr_col:
        st.subheader(f"Experiments ({len(groups)} configs, {len(filtered)} method combinations)")
    with sort_col:
        sort_mode = st.radio(
            "Sort by", options=["rho", "signal"],
            horizontal=True, key="sort_mode",
        )

    # Re-sort groups based on selected mode
    if sort_mode == "signal":
        sorted_keys = sorted(
            groups.keys(),
            key=lambda k: (
                groups[k][0].get("signal", ""),
                -(groups[k][0].get("rho") or 0),
            ),
        )
    else:
        sorted_keys = sorted(
            groups.keys(),
            key=lambda k: (
                -(groups[k][0].get("rho") or 0),
                groups[k][0].get("signal", ""),
            ),
        )

    for gkey in sorted_keys:
        group_rows = groups[gkey]
        config_name = gkey[0]
        rep = group_rows[0]

        # --- Nicely formatted header ---
        header_parts = []
        signal = rep.get("signal", "?")
        header_parts.append(f"**{config_name}** — `{signal}`")
        meta_items = []
        for label, key, fmt in [
            ("n", "n", "{}"),
            ("R", "R", "{}"),
            ("ρ", "rho", None),
            ("SNR", "snr", None),
            ("Model", "model", "{}"),
        ]:
            v = rep.get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            if fmt is None:
                meta_items.append(f"{label} = {_fmt_cell(v, key)}")
            else:
                meta_items.append(f"{label} = {fmt.format(v)}")
        subtitle = " · ".join(meta_items)

        # --- Compute per-row mean stddev across features ---
        def _mean_stddev(r, prefix):
            d = r.get("dim", 1)
            vals = [r.get(f"{prefix}_stddev_d{i+1}", np.nan) for i in range(d)]
            return float(np.nanmean(vals))

        best_shap_std = min(_mean_stddev(r, "shap") for r in group_rows)
        worst_ale_std = max(_mean_stddev(r, "ale") for r in group_rows)
        f_std = rep.get("f_stddev")
        f_std_str = _fmt_cell(f_std, "f_stddev")
        shap_time = rep.get("shap_time_mean")
        shap_time_str = _fmt_cell(shap_time, "shap_time_mean")
        subtitle += (f"  ·  Best SHAP Std = {best_shap_std:.4f}"
                     f"  ·  Worst ALE Std = {worst_ale_std:.4f}"
                     f"  ·  Std(f) = {f_std_str}"
                     f"  ·  SHAP time/pt = {shap_time_str}")

        # --- Build comparison table ---
        table_rows = []
        for r in group_rows:
            table_rows.append({
                "ALE Tag": r.get("ale_tag", "?"),
                "SHAP Tag": r.get("shap_tag", "?"),
                "Variant": r.get("variant", "?"),
                "Local Method": r.get("local_method", "?"),
                "K": r.get("K", "?"),
                "L": r.get("L", "?"),
                "ALE time/pt": _fmt_cell(r.get("ale_time_mean"), "ale_time_mean"),
                "SHAP time/pt": _fmt_cell(r.get("shap_time_mean"), "shap_time_mean"),
                "StdRed": _fmt_cell(r.get("rel_stddev_reduction"), "rel_stddev_reduction"),
            })
        df = pd.DataFrame(table_rows)

        # --- Layout: info + table on left, button on right ---
        left_col, right_col = st.columns([5, 1])
        with left_col:
            st.markdown(f"#### {header_parts[0]}")
            st.caption(subtitle)
            st.dataframe(df, use_container_width=True, hide_index=True)
        with right_col:
            st.markdown("")  # vertical spacing
            st.markdown("")
            st.link_button(
                "More details",
                url=f"?selected_key={quote(rep['_key'])}",
                type="primary",
            )
        st.divider()


def show_detail_page(cache_dir: str, selected_key: str) -> None:
    rows = load_all_rows(cache_dir)
    row = next((r for r in rows if r["_key"] == selected_key), None)
    if row is None:
        st.error("Could not locate the selected experiment.")
        return

    config_name = row["config_name"]
    results_file = row["_results_file"]
    results_path = _results_path(cache_dir, config_name, results_file)

    results = _load_results_cached(results_path)
    ale_tags  = list(results["ale"].keys())
    shap_tags = list(results["shap"].keys())

    st.title("Experiment Details")
    st.caption(f"`{config_name}/{results_file}`")

    meta_items = []
    for label, key, fmt in [
        ("Signal", "signal", "{}"),
        ("n", "n", "{}"),
        ("R", "R", "{}"),
        ("ρ", "rho", None),
        ("SNR", "snr", None),
        ("Model", "model", "{}"),
    ]:
        v = row.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if fmt is None:
            meta_items.append(f"{label} = {_fmt_cell(v, key)}")
        else:
            meta_items.append(f"{label} = {fmt.format(v)}")
    st.caption(" · ".join(meta_items))

    cols = st.columns(2)
    with cols[0]:
        ale_tag = st.selectbox(
            "ALE tag", options=ale_tags,
            index=ale_tags.index(row["ale_tag"]) if row["ale_tag"] in ale_tags else 0,
            key=f"ale_{selected_key}",
        )
    with cols[1]:
        shap_tag = st.selectbox(
            "SHAP tag", options=shap_tags,
            index=shap_tags.index(row["shap_tag"]) if row["shap_tag"] in shap_tags else 0,
            key=f"shap_{selected_key}",
        )

    view = extract_view(results, ale_tag, shap_tag)
    true_fn = detect_explanation_fn(results)

    with st.expander("Bias (over grid points)", expanded=True):
        bias_fig = _plot_bias_hist(view, true_fn)
        if bias_fig is not None:
            st.pyplot(bias_fig)
        else:
            st.caption("No rho-independent true explanation available; bias histogram omitted.")

    with st.expander("Std Dev (over grid points)", expanded=True):
        stddev_fig, f_std_mean = _plot_stddev_hist(view)
        st.pyplot(stddev_fig)
        if f_std_mean is not None:
            st.caption(f"Mean Std(f(x)) across grid points: {f_std_mean:.4f}")

    with st.spinner("Generating plots…"):
        plot_dir = generate_plots(cache_dir, config_name, results_file, ale_tag, shap_tag)

    def _show(filename: str, title: str, expanded: bool = True) -> None:
        path = os.path.join(plot_dir, f"{filename}.png")
        if os.path.exists(path):
            with st.expander(title, expanded=expanded):
                st.image(path, width="stretch")

    _show("bias2",             "Bias²")
    _show("variance",          "Std Dev across Replications")
    _show("f_variance",        "Std(f(x)) across Replications")
    _show("mean_feature_exps",        "Mean Feature Explanations (ALE vs SHAP)")
    _show("mean_function_additivity", "Mean Function Value and Additivity")

    with st.expander("Additivity Residual", expanded=True):
        add_fig, add_info = _plot_additivity_hist(view)
        if add_fig is not None:
            st.pyplot(add_fig)
            st.caption(
                "Residual = sum_d φ_d(x) − (f(x) − E[f(X)]). "
                "Shapley efficiency ⇒ residual ≈ 0. "
                + "  ".join(f"{m} RMSE = {v:.4f}" for m, v in add_info.items())
            )
        else:
            st.caption(
                "Additivity histogram unavailable: this cached results pickle "
                "predates `f_means`. Rerun `python run_experiments.py <config>` "
                "to backfill."
            )

    d = int(results["explain_grid"].shape[1])
    if d == 2:
        with st.expander("Interactive Mode", expanded=False):
            _render_interactive_section(cache_dir, config_name, results_file, row)
        with st.expander("Path Integral Interactive Mode", expanded=False):
            _render_path_integral_section(cache_dir, config_name, results_file, row)
        with st.expander("Multi-Path Interactive Mode", expanded=False):
            _render_multi_path_section(cache_dir, config_name, results_file, row)
    else:
        st.caption(f"Interactive mode is only available for d=2 experiments (this one has d={d}).")

    if d > 2:
        with st.expander("ALE Paths Summary", expanded=False):
            feat_opts = list(range(d))
            pcols = st.columns(2)
            with pcols[0]:
                i_sel = st.selectbox(
                    "Feature i", options=feat_opts, index=0,
                    key=f"paths_i_{selected_key}_{ale_tag}",
                )
            with pcols[1]:
                j_opts = [j for j in feat_opts if j != i_sel]
                j_sel = st.selectbox(
                    "Feature j", options=j_opts, index=0,
                    key=f"paths_j_{selected_key}_{ale_tag}_{i_sel}",
                )

            with st.spinner("Building ALE for paths summary…"):
                ale_ps = _build_paths_ale(cache_dir, config_name, results_file, ale_tag)
            if ale_ps is None:
                st.warning("Could not build paths summary ALE (missing config or run_*.pkl).")
            else:
                with st.spinner(f"Rendering pair f{i_sel} → f{j_sel}…"):
                    fig, _ = plot_paths_summary(ale_ps, int(i_sel), int(j_sel), cmap="viridis")
                st.markdown(f"**Feature {i_sel} → Feature {j_sel}**")
                st.pyplot(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="ExplainableML Explorer", layout="wide")

    cache_dir = _parse_cache_dir()

    selected_key = st.query_params.get("selected_key")
    if selected_key:
        show_detail_page(cache_dir, unquote(selected_key))
    else:
        show_summary_page(cache_dir)


if __name__ == "__main__":
    main()
