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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import r2_score

import models
from ale import ALE, BootstrapALE
from shapley import SHAP
from summarize_experiments import (
    _walk_cache, _load_tune, _fmt_cell,
)
from visualize_experiments import (
    load_cache, extract_view, detect_explanation_fn,
    plot_bias2, plot_variance, plot_f_variability, plot_f_variance,
    plot_single_replication, plot_mean_explanations,
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
    plot_mean_explanations(view, save_dir=plot_dir)
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
    # `method` is a newer field; default to "connected" for older cached configs.
    method = getattr(ec, "method", "connected")
    random_seed = getattr(ec, "random_seed", 42)
    include_key = f"total_{method}"

    runs = _load_all_replications(cache_dir, config_name, results_file)
    if runs is None:
        return None
    X, _y, model = runs[0]

    if ec.variant == "bootstrap":
        ale_obj = BootstrapALE(
            model.predict, X,
            replications=ec.n_bootstrap,
            K=ec.K, L=ec.L, centering=ec.centering,
            interpolate=ec.interpolate,
            random_seed=random_seed,
            verbose=False,
        )
        ale_obj.explain(include=(include_key,))
        return ale_obj.ale_replications[0]
    ale = ALE(model.predict, X, K=ec.K, L=ec.L, centering=ec.centering,
              interpolate=ec.interpolate, random_seed=random_seed, verbose=False)
    ale.explain(include=(include_key,))
    return ale


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
                           K, L, centering, interpolate, levels_up):
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


def _matched_path(ale, j: int, x_query: np.ndarray, levels_up: int) -> int:
    x_row = x_query[0] if x_query.ndim == 2 else x_query
    indices = ale.connected_forest[j].route_and_pick_representative(
        x_row, ale.X_values, levels_up=levels_up
    )["indices"]
    return int(ale.observation_to_path[j][indices[0]])


def _decompose_ale_terms(ale, j: int, x_query: np.ndarray, levels_up: int) -> dict:
    """Decompose the local ALE effect for feature j into its component terms.

    Returns a dict with:
      x_star:        the matched training observation (array of length d)
      g_left:        g*(x_j_left) — centered g-value at the left bin edge
      full_delta:    f(right_edge, x*_{-j}) - f(left_edge, x*_{-j})
      alpha_delta:   alpha * (g*(right) - g*(left)), the interpolation term
      term_path_rep: mean_{x* in path,bin} [f(x_j, x*_{-j}) - f(left, x*_{-j})]
      term_self:     f(x_j, x_{-j}) - f(left, x_{-j})
    """
    from ale.ale_vim import _local_term_self, _local_term_path_rep, calculate_bin_index

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
        x_row, ale.X_values, levels_up=levels_up
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
    term_self = float(_local_term_self(ale.f, x_explain, j, edges[k]))

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


def _render_feature_panel(ale, j: int, x_query: np.ndarray, levels_up: int):
    from ale.shared import linear_interpolation

    edges = ale.edges[j]
    g = ale.centered_g_values[j].centered_g_values
    K = len(edges) - 1
    L = g.shape[1]
    interpolate = g.shape[0] == K + 1

    l_match = _matched_path(ale, j, x_query, levels_up)

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
        ax_g.scatter([xq], [yq], color="red", s=80, zorder=5, label=f"x_{j+1}={xq:.2f}")
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
    ax_g.set_xlabel(f"x_{j+1}")
    ax_g.set_ylabel("centered g-value")
    ax_g.set_title(f"g-values for feature {j+1}  (matched path: {l_match})")
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
    ax_sc.set_title(f"Training points — paths for feature {j+1}")
    ax_sc.legend(loc="best")

    fig.tight_layout()
    return fig


def _render_interactive_section(cache_dir, config_name, results_file, row_meta) -> None:
    st.header("Interactive Mode")

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
        levels_up = st.number_input(
            "levels_up", min_value=0, value=int(row_meta.get("levels_up") or 0),
            step=1, key=f"lu_{ukey}",
        )

    with st.spinner("Fitting ALE…"):
        ale, X, model = _build_interactive_ale(
            cache_dir, config_name, results_file, int(rep_idx),
            int(K), int(L), centering, bool(interpolate), int(levels_up),
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
            st.subheader(f"Feature {j+1}")
            st.pyplot(_render_feature_panel(ale, j, x_query, int(levels_up)))

    st.subheader("Local-effect terms at query")

    # Decompose per-feature terms
    decomp = []
    for j in range(2):
        try:
            decomp.append(_decompose_ale_terms(ale, j, x_query, int(levels_up)))
        except Exception as e:
            st.caption(f"Decomposition for feature {j+1} failed: {e}")
            decomp.append(None)

    # Show matched x* for each feature
    for j in range(2):
        if decomp[j] is not None:
            xs = decomp[j]["x_star"]
            st.caption(
                f"Feature {j+1} matched x\\* = "
                f"({', '.join(f'{v:.4f}' for v in xs)})"
            )

    rows = []

    # Final ALE values (all three methods)
    for method in ("interpolate", "path_rep", "self", "path_integral"):
        try:
            terms = ale.explain_local(
                x_query, levels_up=int(levels_up), local_method=method
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
        ax.set_title(f"Feature {j + 1}")
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
        ax.set_title(f"Feature {j + 1}")
        ax.legend()
    fig.tight_layout()

    f_std_mean = float(f_vals.std(axis=0).mean()) if f_vals is not None else None
    return fig, f_std_mean


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

_FILTER_SPECS = [
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

    bias_fig = _plot_bias_hist(view, true_fn)
    if bias_fig is not None:
        st.pyplot(bias_fig)
    else:
        st.caption("No rho-independent true explanation available; bias histogram omitted.")

    stddev_fig, f_std_mean = _plot_stddev_hist(view)
    st.pyplot(stddev_fig)
    if f_std_mean is not None:
        st.caption(f"Mean Std(f(x)) across grid points: {f_std_mean:.4f}")

    with st.spinner("Generating plots…"):
        plot_dir = generate_plots(cache_dir, config_name, results_file, ale_tag, shap_tag)

    def _show(filename: str, title: str) -> None:
        path = os.path.join(plot_dir, f"{filename}.png")
        if os.path.exists(path):
            st.subheader(title)
            st.image(path, width="stretch")

    _show("bias2",             "Bias²")
    _show("variance",          "Std Dev across Replications")
    _show("f_variance",        "Std(f(x)) across Replications")
    _show("mean_explanations", "Mean Explanations")

    d = int(results["explain_grid"].shape[1])
    if d == 2:
        st.divider()
        _render_interactive_section(cache_dir, config_name, results_file, row)
    else:
        st.divider()
        st.info(f"Interactive mode is only available for d=2 experiments (this one has d={d}).")

    if d > 2:
        st.divider()
        st.header("ALE Paths Summary")
        feat_opts = list(range(1, d + 1))
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
                fig, _ = ale_ps.plot_paths_summary(int(i_sel), int(j_sel), cmap="viridis")
            st.subheader(f"Feature {i_sel} → Feature {j_sel}")
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
