"""
explore_experiments.py

Streamlit app combining summarize_experiments.py (main page) and
visualize_experiments.py (detail page).

Main page  : interactive summary table of all full_*.npz experiments.
Detail page: full set of plots for a selected experiment.

Usage:
    streamlit run explore_experiments.py
    streamlit run explore_experiments.py -- --cache-dir /path/to/cache
"""

import glob
import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")  # non-interactive backend before any matplotlib import

import numpy as np
import pandas as pd
import streamlit as st

from summarize_experiments import _load_row, _BASE_COLS, _TAIL_COLS, _dim_cols, _fmt_cell
from visualize_experiments import (
    load_cache,
    detect_explanation_fn,
    plot_bias2,
    plot_variance,
    plot_f_variability,
    plot_single_replication,
    plot_mean_explanations,
    plot_paths_summary_all_pairs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_cache_dir() -> str:
    """Return --cache-dir arg if provided (after -- in streamlit run), else default."""
    for i, arg in enumerate(sys.argv):
        if arg == "--cache-dir" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return "cached_explanations"


@st.cache_data
def load_all_rows(cache_dir: str) -> list:
    paths = sorted(glob.glob(os.path.join(cache_dir, "full_*.npz")))
    rows = []
    for path in paths:
        row = _load_row(path, cache_dir)
        if row is not None:
            rows.append(row)
    rows.sort(
        key=lambda r: r["rel_var_reduction"]
        if not np.isnan(r["rel_var_reduction"])
        else -np.inf,
        reverse=True,
    )
    return rows


@st.cache_data(show_spinner=False)
def generate_plots(cache_file: str, cache_dir: str) -> str:
    """Render all plots for one experiment into a temp directory; return the path."""
    plot_dir = os.path.join(tempfile.gettempdir(), "explainableml_plots", cache_file)
    os.makedirs(plot_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_file)
    true_fn = detect_explanation_fn(cache_path)
    cache = load_cache(cache_path)
    plot_bias2(cache, true_explanation_fn=true_fn, save_dir=plot_dir)
    plot_variance(cache, save_dir=plot_dir)
    plot_f_variability(cache, save_dir=plot_dir)
    plot_single_replication(cache, r=0, save_dir=plot_dir)
    plot_mean_explanations(cache, save_dir=plot_dir)
    plot_paths_summary_all_pairs(cache_path, cache_dir=cache_dir, save_dir=plot_dir)
    return plot_dir


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def show_summary_page(cache_dir: str) -> None:
    st.title("Experiment Summary")

    rows = load_all_rows(cache_dir)
    if not rows:
        st.error(f"No full_*.npz files found in {cache_dir!r}")
        return

    max_dim = max(r.get("dim", 1) for r in rows)
    all_cols = _BASE_COLS + _dim_cols(max_dim) + _TAIL_COLS

    # Build display DataFrame (formatted strings, matching the TUI)
    records = []
    for r in rows:
        record = {header: _fmt_cell(r.get(key), key) for header, key in all_cols}
        record["_cache_file"] = r["_cache_file"]
        records.append(record)

    display_cols = [h for h, _ in all_cols]
    df = pd.DataFrame(records)
    st.dataframe(df[display_cols], use_container_width=True)

    st.divider()
    st.subheader("View Experiment Details")
    st.caption("Click an experiment below to open its detail view.")

    for rec in records:
        label = "  |  ".join([
            rec.get("Signal", "?"),
            f"ρ={rec.get('Rho', '?')}",
            f"SNR={rec.get('SNR', '?')}",
            rec.get("Model", "?"),
            f"K={rec.get('K', '?')}",
            f"L={rec.get('L', '?')}",
            rec.get("Variant", "standard"),
            f"lu={rec.get('LevelsUp', 0)}",
            f"VarRed={rec.get('VarRed%', '?')}",
        ])
        if st.button(label, key=rec["_cache_file"]):
            st.session_state.selected_file = rec["_cache_file"]
            st.rerun()


def show_detail_page(cache_file: str, cache_dir: str) -> None:
    if st.button("← Back to Summary"):
        st.session_state.selected_file = None
        st.rerun()

    st.title("Experiment Details")
    st.caption(f"`{cache_file}`")

    with st.spinner("Generating plots…"):
        plot_dir = generate_plots(cache_file, cache_dir)

    def _show(filename: str, title: str) -> None:
        path = os.path.join(plot_dir, f"{filename}.png")
        if os.path.exists(path):
            st.subheader(title)
            st.image(path, width="stretch")

    _show("bias2",                  "Bias²")
    _show("variance",               "Variance (Std Dev across Replications)")
    _show("f_variability",          "f(x) Variability")
    _show("single_replication_r0",  "Single Replication (r=0) Explanations")
    _show("mean_explanations",      "Mean Explanations")

    # ALE paths summary — one panel per ordered feature pair
    paths_files = sorted(
        f for f in os.listdir(plot_dir) if f.startswith("paths_summary_")
    )
    if paths_files:
        st.divider()
        st.header("ALE Paths Summary")
        for fname in paths_files:
            # fname like "paths_summary_f1_f2.png" -> title "Feature 1 → Feature 2"
            parts = fname.replace("paths_summary_", "").replace(".png", "").split("_")
            title = f"Feature {parts[0][1:]} → Feature {parts[1][1:]}"
            st.subheader(title)
            st.image(os.path.join(plot_dir, fname), width="stretch")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="ExplainableML Explorer", layout="wide")

    cache_dir = _parse_cache_dir()

    if "selected_file" not in st.session_state:
        st.session_state.selected_file = None

    if st.session_state.selected_file is None:
        show_summary_page(cache_dir)
    else:
        show_detail_page(st.session_state.selected_file, cache_dir)


if __name__ == "__main__":
    main()
