"""
Standalone visualization module for experiment results.

Usage (module):
    from visualize_experiments import load_cache, visualize
    cache = load_cache("cached_explanations/full_*.npz")
    visualize(cache, true_explanation_fn=signal_basic_explanation, save_dir="plots/")

Usage (CLI):
    python visualize_experiments.py cached_explanations/full_*.npz [--save-dir plots/]
"""

import os
import re
import argparse

import joblib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from ale import ALE, BootstrapALE

from models import (
    signal_basic_explanation,
    signal_nonlinear_explanation,
    signal_threshold_explanation,
)

# Maps signal function name (as it appears in the cache filename) to its
# true local explanation function, or None if no ground truth exists.
# signal_multiplicative omitted: its explanation depends on rho, which is
# not available here — bias² is pre-computed into the .npz at run time.
_SIGNAL_EXPLANATION_FNS = {
    "signal_basic":      signal_basic_explanation,
    "signal_nonlinear":  signal_nonlinear_explanation,
    "signal_threshold":  signal_threshold_explanation,
}


def detect_explanation_fn(cache_path: str):
    """Return the true explanation function inferred from the cache filename, or None."""
    basename = os.path.basename(cache_path)
    for signal_name, fn in _SIGNAL_EXPLANATION_FNS.items():
        if signal_name in basename:
            return fn
    return None


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

def load_cache(path: str) -> dict:
    """Load a full_*.npz cache file into a plain dict of arrays."""
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_heatmap(pts_i, pts_j, values, grid_res=100):
    """Interpolate scattered values onto a regular grid.

    Returns (masked_array, extent) where extent = [i_min, i_max, j_min, j_max].
    """
    xi = np.linspace(pts_i.min(), pts_i.max(), grid_res)
    xj = np.linspace(pts_j.min(), pts_j.max(), grid_res)
    xi_grid, xj_grid = np.meshgrid(xi, xj)
    z = griddata(
        np.column_stack([pts_i, pts_j]),
        values,
        (xi_grid, xj_grid),
        method="linear",
    )
    extent = [pts_i.min(), pts_i.max(), pts_j.min(), pts_j.max()]
    return np.ma.masked_invalid(z), extent


def _bin_line(x_vals, y_vals, n_bins=30):
    """Bin y_vals by x_vals into n_bins equal-width bins.

    Returns (bin_centers, bin_means, bin_stds).
    """
    edges = np.linspace(x_vals.min(), x_vals.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(n_bins, np.nan)
    stds = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = (x_vals >= edges[b]) & (x_vals < edges[b + 1])
        if mask.sum() > 1:
            means[b] = y_vals[mask].mean()
            stds[b] = y_vals[mask].std()
    return centers, means, stds


def _show_or_save(fig, name: str, save_dir):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"{name}.png"), bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Plot: Bias²
# ---------------------------------------------------------------------------

def plot_bias2(cache: dict, true_explanation_fn=None, save_dir=None):
    """Bias² of ALE and SHAP explanations vs. true explanation.

    Skipped if true_explanation_fn is None.
    """
    if true_explanation_fn is None:
        return

    shap_exps = cache["shap_exps"]        # (R, N, d)
    ale_exps  = cache["ale_exps"]         # (R, N, d)
    explain_grid = cache["explain_grid"]  # (N, d)
    d = explain_grid.shape[1]

    true_exp = true_explanation_fn(explain_grid)          # (N, d)
    shap_bias2 = (shap_exps.mean(axis=0) - true_exp) ** 2  # (N, d)
    ale_bias2  = (ale_exps.mean(axis=0)  - true_exp) ** 2  # (N, d)

    if d == 2:
        fig, axes = plt.subplots(d, 3, figsize=(21, 10))
        fig.suptitle("Bias²")
        x1, x2 = explain_grid[:, 0], explain_grid[:, 1]
        for feat in range(d):
            ale_hm, extent = _to_heatmap(x1, x2, ale_bias2[:, feat])
            shap_hm, _     = _to_heatmap(x1, x2, shap_bias2[:, feat])
            vmin = min(ale_hm.min(), shap_hm.min())
            vmax = max(ale_hm.max(), shap_hm.max())
            for col, (hm, label) in enumerate([(ale_hm, "ALE"), (shap_hm, "SHAP")]):
                ax = axes[feat, col]
                im = ax.imshow(hm, origin="lower", extent=extent,
                               cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
                ax.set_xlabel("X1"); ax.set_ylabel("X2")
                ax.set_title(f"{label} — Feature {feat + 1}")
                fig.colorbar(im, ax=ax, label="Bias²")
            diff_hm = ale_hm - shap_hm
            abs_max = np.nanmax(np.abs(diff_hm))
            ax = axes[feat, 2]
            im = ax.imshow(diff_hm, origin="lower", extent=extent,
                           cmap="RdBu", vmin=-abs_max, vmax=abs_max, aspect="auto")
            ax.set_xlabel("X1"); ax.set_ylabel("X2")
            ax.set_title(f"ALE − SHAP Bias² — Feature {feat + 1}")
            fig.colorbar(im, ax=ax, label="ALE − SHAP Bias²")
        fig.tight_layout()
    else:
        fig, axes = plt.subplots(1, d, figsize=(5 * d, 5))
        fig.suptitle("Bias²")
        if d == 1:
            axes = [axes]
        for feat in range(d):
            ax = axes[feat]
            xk = explain_grid[:, feat]
            for vals, label, color in [(ale_bias2[:, feat], "ALE", "C0"),
                                       (shap_bias2[:, feat], "SHAP", "C1")]:
                ax.scatter(xk, vals, alpha=0.15, s=6, color=color)
                centers, means, stds = _bin_line(xk, vals)
                ax.plot(centers, means, color=color, label=label, linewidth=2)
                ax.fill_between(centers, means - stds, means + stds,
                                alpha=0.2, color=color)
            ax.set_xlabel(f"X{feat + 1}"); ax.set_ylabel("Bias²")
            ax.set_title(f"Feature {feat + 1}")
            ax.legend()
        fig.tight_layout()

    _show_or_save(fig, "bias2", save_dir)


# ---------------------------------------------------------------------------
# Plot: Variance
# ---------------------------------------------------------------------------

def plot_variance(cache: dict, save_dir=None):
    """Std dev of ALE and SHAP explanations across replications."""
    shap_exps    = cache["shap_exps"]
    ale_exps     = cache["ale_exps"]
    explain_grid = cache["explain_grid"]
    d = explain_grid.shape[1]

    shap_std = shap_exps.std(axis=0)  # (N, d)
    ale_std  = ale_exps.std(axis=0)   # (N, d)

    if d == 2:
        fig, axes = plt.subplots(d, 3, figsize=(21, 10))
        fig.suptitle("Std Dev across Replications")
        x1, x2 = explain_grid[:, 0], explain_grid[:, 1]
        for feat in range(d):
            ale_hm, extent = _to_heatmap(x1, x2, ale_std[:, feat])
            shap_hm, _     = _to_heatmap(x1, x2, shap_std[:, feat])
            vmin = min(ale_hm.min(), shap_hm.min())
            vmax = max(ale_hm.max(), shap_hm.max())
            for col, (hm, label) in enumerate([(ale_hm, "ALE"), (shap_hm, "SHAP")]):
                ax = axes[feat, col]
                im = ax.imshow(hm, origin="lower", extent=extent,
                               cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
                ax.set_xlabel("X1"); ax.set_ylabel("X2")
                ax.set_title(f"{label} — Feature {feat + 1}")
                fig.colorbar(im, ax=ax, label="Std Dev")
            diff_hm = ale_hm - shap_hm
            abs_max = np.nanmax(np.abs(diff_hm))
            ax = axes[feat, 2]
            im = ax.imshow(diff_hm, origin="lower", extent=extent,
                           cmap="RdBu", vmin=-abs_max, vmax=abs_max, aspect="auto")
            ax.set_xlabel("X1"); ax.set_ylabel("X2")
            ax.set_title(f"ALE − SHAP Std Dev — Feature {feat + 1}")
            fig.colorbar(im, ax=ax, label="ALE − SHAP Std Dev")
        fig.tight_layout()
    else:
        fig, axes = plt.subplots(1, d, figsize=(5 * d, 5))
        fig.suptitle("Std Dev across Replications")
        if d == 1:
            axes = [axes]
        for feat in range(d):
            ax = axes[feat]
            xk = explain_grid[:, feat]
            for vals, label, color in [(ale_std[:, feat], "ALE", "C0"),
                                       (shap_std[:, feat], "SHAP", "C1")]:
                ax.scatter(xk, vals, alpha=0.15, s=6, color=color)
                centers, means, stds = _bin_line(xk, vals)
                ax.plot(centers, means, color=color, label=label, linewidth=2)
                ax.fill_between(centers, means - stds, means + stds,
                                alpha=0.2, color=color)
            ax.set_xlabel(f"X{feat + 1}"); ax.set_ylabel("Std Dev")
            ax.set_title(f"Feature {feat + 1}")
            ax.legend()
        fig.tight_layout()

    _show_or_save(fig, "variance", save_dir)


# ---------------------------------------------------------------------------
# Plot: Single replication explanations
# ---------------------------------------------------------------------------

def plot_single_replication(cache: dict, r: int = 0, save_dir=None):
    """Explanation values from a single replication."""
    shap_exps    = cache["shap_exps"]
    ale_exps     = cache["ale_exps"]
    explain_grid = cache["explain_grid"]
    d = explain_grid.shape[1]

    single_ale  = ale_exps[r]   # (N, d)
    single_shap = shap_exps[r]  # (N, d)

    if d == 2:
        fig, axes = plt.subplots(d, 2, figsize=(14, 10))
        fig.suptitle(f"Single Replication (r={r}) Explanations")
        x1, x2 = explain_grid[:, 0], explain_grid[:, 1]
        for feat in range(d):
            ale_hm, extent = _to_heatmap(x1, x2, single_ale[:, feat])
            shap_hm, _     = _to_heatmap(x1, x2, single_shap[:, feat])
            vmin = min(ale_hm.min(), shap_hm.min())
            vmax = max(ale_hm.max(), shap_hm.max())
            for col, (hm, label) in enumerate([(ale_hm, "ALE"), (shap_hm, "SHAP")]):
                ax = axes[feat, col]
                im = ax.imshow(hm, origin="lower", extent=extent,
                               cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
                ax.set_xlabel("X1"); ax.set_ylabel("X2")
                ax.set_title(f"{label} — Feature {feat + 1}")
                fig.colorbar(im, ax=ax, label="Explanation")
        fig.tight_layout()
    else:
        fig, axes = plt.subplots(1, d, figsize=(5 * d, 5))
        fig.suptitle(f"Single Replication (r={r}) Explanations")
        if d == 1:
            axes = [axes]
        for feat in range(d):
            ax = axes[feat]
            xk = explain_grid[:, feat]
            for vals, label, color in [(single_ale[:, feat], "ALE", "C0"),
                                       (single_shap[:, feat], "SHAP", "C1")]:
                ax.scatter(xk, vals, alpha=0.3, s=8, color=color, label=label)
            ax.set_xlabel(f"X{feat + 1}"); ax.set_ylabel("Explanation")
            ax.set_title(f"Feature {feat + 1}")
            ax.legend()
        fig.tight_layout()

    _show_or_save(fig, f"single_replication_r{r}", save_dir)


# ---------------------------------------------------------------------------
# Plot: Mean explanations + mean f(x)
# ---------------------------------------------------------------------------

def plot_mean_explanations(cache: dict, save_dir=None):
    """Mean ALE and SHAP explanations per feature, plus mean f(x), across replications."""
    ale_exps     = cache["ale_exps"]
    shap_exps    = cache["shap_exps"]
    f_vals       = cache["f_vals"]
    explain_grid = cache["explain_grid"]
    d = explain_grid.shape[1]

    mean_ale  = ale_exps.mean(axis=0)   # (N, d)
    mean_shap = shap_exps.mean(axis=0)  # (N, d)
    mean_f    = f_vals.mean(axis=0)     # (N,)

    if d == 2:
        # 2 methods × d features + 1 f(x) panel
        ncols = 2 * d + 1
        fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
        fig.suptitle("Mean Explanations and Mean f(x)")
        x1, x2 = explain_grid[:, 0], explain_grid[:, 1]
        for feat in range(d):
            for col_offset, (mean_vals, label, cbar_label) in enumerate([
                (mean_ale,  "ALE",  "Mean ALE"),
                (mean_shap, "SHAP", "Mean SHAP"),
            ]):
                ax = axes[feat * 2 + col_offset]
                hm, extent = _to_heatmap(x1, x2, mean_vals[:, feat])
                im = ax.imshow(hm, origin="lower", extent=extent,
                               cmap="viridis", aspect="auto")
                ax.set_xlabel("X1"); ax.set_ylabel("X2")
                ax.set_title(f"Mean {label} — Feature {feat + 1}")
                fig.colorbar(im, ax=ax, label=cbar_label)
        hm_f, extent = _to_heatmap(x1, x2, mean_f)
        im = axes[-1].imshow(hm_f, origin="lower", extent=extent,
                             cmap="viridis", aspect="auto")
        axes[-1].set_xlabel("X1"); axes[-1].set_ylabel("X2")
        axes[-1].set_title("Mean f(x)")
        fig.colorbar(im, ax=axes[-1], label="Mean f(x)")
        fig.tight_layout()
    else:
        # d panels (one per feature, ALE+SHAP overlaid) + 1 f(x) panel
        ncols = d + 1
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
        fig.suptitle("Mean Explanations and Mean f(x)")
        axes = list(axes)
        for feat in range(d):
            ax = axes[feat]
            xk = explain_grid[:, feat]
            for mean_vals, label, color in [
                (mean_ale[:, feat],  "ALE",  "C0"),
                (mean_shap[:, feat], "SHAP", "C1"),
            ]:
                centers, means, stds = _bin_line(xk, mean_vals)
                ax.scatter(xk, mean_vals, alpha=0.15, s=6, color=color)
                ax.plot(centers, means, color=color, label=label, linewidth=2)
                ax.fill_between(centers, means - stds, means + stds, alpha=0.2, color=color)
            ax.set_xlabel(f"X{feat + 1}"); ax.set_ylabel("Mean Explanation")
            ax.set_title(f"Mean Explanation — Feature {feat + 1}")
            ax.legend()
        ax = axes[d]
        xk = explain_grid[:, 0]
        centers, means, stds = _bin_line(xk, mean_f)
        ax.scatter(xk, mean_f, alpha=0.15, s=6, color="C2")
        ax.plot(centers, means, color="C2", linewidth=2)
        ax.fill_between(centers, means - stds, means + stds, alpha=0.2, color="C2")
        ax.set_xlabel("X1"); ax.set_ylabel("Mean f(x)")
        ax.set_title("Mean f(x)")
        fig.tight_layout()

    _show_or_save(fig, "mean_explanations", save_dir)


# ---------------------------------------------------------------------------
# Plot: f(x) variability across replications
# ---------------------------------------------------------------------------

def plot_f_variability(cache: dict, save_dir=None):
    """Std dev of f(x) predictions across replications."""
    f_vals       = cache["f_vals"]        # (R, N)
    explain_grid = cache["explain_grid"]  # (N, d)
    d = explain_grid.shape[1]

    f_std = f_vals.std(axis=0)  # (N,)

    if d == 2:
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.suptitle("f(x) Std Dev across Replications")
        x1, x2 = explain_grid[:, 0], explain_grid[:, 1]
        pts = np.column_stack([x1, x2])
        xi = np.linspace(x1.min(), x1.max(), 100)
        xj = np.linspace(x2.min(), x2.max(), 100)
        xi_grid, xj_grid = np.meshgrid(xi, xj)
        from scipy.interpolate import griddata
        z_lin = griddata(pts, f_std, (xi_grid, xj_grid), method="linear")
        z_nn  = griddata(pts, f_std, (xi_grid, xj_grid), method="nearest")
        z = np.where(np.isnan(z_lin), z_nn, z_lin)
        extent = [x1.min(), x1.max(), x2.min(), x2.max()]
        im = ax.imshow(z, origin="lower", extent=extent,
                       cmap="viridis", aspect="auto")
        ax.set_xlabel("X1"); ax.set_ylabel("X2")
        ax.set_title("f(x) Std Dev")
        fig.colorbar(im, ax=ax, label="f(x) stddev")
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.suptitle("f(x) Std Dev across Replications")
        xk = explain_grid[:, 0]
        ax.scatter(xk, f_std, alpha=0.15, s=6, color="C2")
        centers, means, stds = _bin_line(xk, f_std)
        ax.plot(centers, means, color="C2", linewidth=2)
        ax.fill_between(centers, means - stds, means + stds, alpha=0.2, color="C2")
        ax.set_xlabel("X1"); ax.set_ylabel("f(x) Std Dev")
        fig.tight_layout()

    _show_or_save(fig, "f_variability", save_dir)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(cache_path: str, true_explanation_fn=None):
    """Print a summary table of bias², variance, and timing for ALE and SHAP.

    ale_times lives in the companion ale_*.npz file; shap_times is in the
    full_*.npz file.  Both are in units of seconds per explained point.
    """
    if true_explanation_fn is None:
        true_explanation_fn = detect_explanation_fn(cache_path)

    cache = load_cache(cache_path)

    # Load ALE timing from the companion ale_*.npz
    cache_dir  = os.path.dirname(cache_path) or "."
    ale_fname  = os.path.basename(cache_path).replace("full_", "ale_", 1)
    ale_path   = os.path.join(cache_dir, ale_fname)
    ale_times  = None
    if os.path.exists(ale_path):
        ale_cache = np.load(ale_path, allow_pickle=True)
        if "ale_times" in ale_cache.files:
            ale_times = ale_cache["ale_times"]

    shap_times    = cache["shap_times"]                          # (R,) sec/point
    shap_variance = np.atleast_1d(cache["shap_variance"])       # (d,)
    ale_variance  = np.atleast_1d(cache["ale_variance"])        # (d,)
    d = len(shap_variance)

    has_bias = true_explanation_fn is not None
    if has_bias:
        shap_bias2 = np.atleast_1d(cache["shap_bias2"])         # (d,)
        ale_bias2  = np.atleast_1d(cache["ale_bias2"])          # (d,)

    col_w = 14
    sep   = "+" + ("-" * 18) + "+" + ("-" * col_w) + "+" + ("-" * col_w) + "+"

    def row(label, ale_val, shap_val, fmt=".6f"):
        return (
            f"| {label:<16} | {ale_val:{col_w}{fmt}} | {shap_val:{col_w}{fmt}} |"
            if ale_val is not None
            else f"| {label:<16} | {'N/A':>{col_w}} | {shap_val:{col_w}{fmt}} |"
        )

    print(sep)
    print(f"| {'Metric':<16} | {'ALE':>{col_w}} | {'SHAP':>{col_w}} |")
    print(sep)
    if has_bias:
        for i in range(d):
            print(row(f"Bias² (D{i+1})", float(ale_bias2[i]), float(shap_bias2[i])))
    for i in range(d):
        print(row(f"Variance (D{i+1})", float(ale_variance[i]), float(shap_variance[i])))
    if ale_times is not None:
        print(row("Time/pt mean (s)", ale_times.mean(), shap_times.mean()))
        print(row("Time/pt std  (s)", ale_times.std(),  shap_times.std()))
    else:
        print(row("Time/pt mean (s)", None, shap_times.mean()))
        print(row("Time/pt std  (s)", None, shap_times.std()))
    print(sep)


# ---------------------------------------------------------------------------
# Plot: ALE paths summary (all ordered feature pairs)
# ---------------------------------------------------------------------------

def plot_paths_summary_all_pairs(cache_path: str, cache_dir: str = None, save_dir=None):
    """Rebuild an ALE object from the companion run_*.pkl and call plot_paths_summary
    for every ordered pair (i, j) with i != j.

    ExplainerConfig params (K, L, centering, levels_up, variant) are read from
    meta_* fields embedded in the full_*.npz file.
    """
    if cache_dir is None:
        cache_dir = os.path.dirname(os.path.abspath(cache_path))

    # Load metadata from the npz
    npz = np.load(cache_path, allow_pickle=True)
    required_meta = {"meta_K", "meta_L", "meta_centering", "meta_variant"}
    if not required_meta.issubset(set(npz.files)):
        print(f"[paths_summary] no metadata in: {os.path.basename(cache_path)}")
        return

    K         = int(npz["meta_K"])
    L         = int(npz["meta_L"])
    centering = str(npz["meta_centering"])
    levels_up = int(npz["meta_levels_up"]) if "meta_levels_up" in npz.files else 0
    variant   = str(npz["meta_variant"])
    n_bootstrap = int(npz["meta_n_bootstrap"]) if "meta_n_bootstrap" in npz.files else 50

    # Derive run pkl path: strip everything after _n{n}_R{R}_
    basename = os.path.basename(cache_path)
    run_basename = re.sub(r"_K\d+.*\.npz$", ".pkl",
                          basename.replace("full_", "run_", 1))
    run_path = os.path.join(cache_dir, run_basename)
    if not os.path.exists(run_path):
        print(f"[paths_summary] run pkl not found: {run_path}")
        return

    data_and_fitted_models = joblib.load(run_path)
    X, _y, model = data_and_fitted_models[0]

    if variant == "bootstrap":
        ale_obj = BootstrapALE(
            model.predict, X,
            replications=n_bootstrap,
            K=K, L=L, centering=centering,
            interpolate=True, verbose=False,
        )
        ale_obj.explain(include=("total_connected",))
        # Use the first bootstrap replication for path plotting
        ale = ale_obj.ale_replications[0]
    else:
        ale = ALE(model.predict, X, K=K, L=L, centering=centering,
                  interpolate=True, verbose=False)
        ale.explain(include=("total_connected",))

    d = X.shape[1]
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            fig, _ = ale.plot_paths_summary(i + 1, j + 1, cmap="viridis")
            _show_or_save(fig, f"paths_summary_f{i+1}_f{j+1}", save_dir)


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def visualize(cache_path: str, true_explanation_fn=None, save_dir=None):
    """Load cache and produce all standard plots and summary table.

    If true_explanation_fn is not provided, it is inferred from the cache filename.
    """
    if true_explanation_fn is None:
        true_explanation_fn = detect_explanation_fn(cache_path)
    cache = load_cache(cache_path)
    print_summary_table(cache_path, true_explanation_fn=true_explanation_fn)
    plot_bias2(cache, true_explanation_fn=true_explanation_fn, save_dir=save_dir)
    plot_variance(cache, save_dir=save_dir)
    plot_f_variability(cache, save_dir=save_dir)
    plot_single_replication(cache, r=0, save_dir=save_dir)
    plot_mean_explanations(cache, save_dir=save_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize experiment results from a full_*.npz cache file."
    )
    parser.add_argument("cache_path", help="Path to the full_*.npz cache file.")
    parser.add_argument(
        "--save-dir", default=None,
        help="Directory to save figures as PNG. If omitted, figures are shown interactively.",
    )
    args = parser.parse_args()
    visualize(args.cache_path, save_dir=args.save_dir)
