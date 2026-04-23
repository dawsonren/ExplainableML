"""
Standalone visualization module for experiment results.

Usage (module):
    from visualize_experiments import load_cache, extract_view, visualize
    results = load_cache("cached_explanations/sparse_highdim/results_*.pkl")
    visualize(results, ale_tag="K40_L25_y", shap_tag="exact",
              true_explanation_fn=signal_basic_explanation, save_dir="plots/")

Usage (CLI):
    python visualize_experiments.py path/to/results_*.pkl --ale-tag ... --shap-tag ... [--save-dir plots/]
"""

import os
import argparse

import joblib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from ale import ALE, BootstrapALE
import models


def _true_explanation_fn(signal_name: str):
    """Return the *_explanation function for a signal name, if it's rho-independent."""
    fn = getattr(models, f"{signal_name}_explanation", None)
    if fn is None:
        return None
    try:
        import inspect
        nparams = len(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return None
    return fn if nparams == 1 else None


def detect_explanation_fn(results: dict):
    """Return the true explanation function inferred from the results metadata, or None."""
    signal = (results.get("experiment_meta") or {}).get("signal", "")
    return _true_explanation_fn(signal)


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

def load_cache(path: str) -> dict:
    """Load a results_*.pkl file into a dict. Returns the full nested structure."""
    return joblib.load(path)


def extract_view(results: dict, ale_tag: str, shap_tag: str) -> dict:
    """Flatten one (ALE, SHAP) pair into a view compatible with the plot_* functions."""
    ale_entry  = results["ale"][ale_tag]
    shap_entry = results["shap"][shap_tag]
    return {
        "ale_exps":     ale_entry["exps"],
        "shap_exps":    shap_entry["exps"],
        "f_vals":       results["f_vals"],
        "explain_grid": results["explain_grid"],
        "ale_config":   ale_entry.get("config"),
        "shap_config":  shap_entry.get("config"),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_heatmap(pts_i, pts_j, values, grid_res=100):
    xi = np.linspace(pts_i.min(), pts_i.max(), grid_res)
    xj = np.linspace(pts_j.min(), pts_j.max(), grid_res)
    xi_grid, xj_grid = np.meshgrid(xi, xj)
    z = griddata(
        np.column_stack([pts_i, pts_j]), values,
        (xi_grid, xj_grid), method="linear",
    )
    extent = [pts_i.min(), pts_i.max(), pts_j.min(), pts_j.max()]
    return np.ma.masked_invalid(z), extent


def _bin_line(x_vals, y_vals, n_bins=30):
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

def plot_bias2(view: dict, true_explanation_fn=None, save_dir=None):
    if true_explanation_fn is None:
        return

    shap_exps = view["shap_exps"]
    ale_exps  = view["ale_exps"]
    explain_grid = view["explain_grid"]
    d = explain_grid.shape[1]

    true_exp = true_explanation_fn(explain_grid)
    shap_bias2 = (shap_exps.mean(axis=0) - true_exp) ** 2
    ale_bias2  = (ale_exps.mean(axis=0)  - true_exp) ** 2

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

def plot_variance(view: dict, save_dir=None):
    shap_exps    = view["shap_exps"]
    ale_exps     = view["ale_exps"]
    explain_grid = view["explain_grid"]
    d = explain_grid.shape[1]

    shap_std = shap_exps.std(axis=0)
    ale_std  = ale_exps.std(axis=0)

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

def plot_single_replication(view: dict, r: int = 0, save_dir=None):
    shap_exps    = view["shap_exps"]
    ale_exps     = view["ale_exps"]
    explain_grid = view["explain_grid"]
    d = explain_grid.shape[1]

    single_ale  = ale_exps[r]
    single_shap = shap_exps[r]

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

def plot_mean_explanations(view: dict, save_dir=None):
    ale_exps     = view["ale_exps"]
    shap_exps    = view["shap_exps"]
    f_vals       = view["f_vals"]
    explain_grid = view["explain_grid"]
    d = explain_grid.shape[1]

    mean_ale  = ale_exps.mean(axis=0)
    mean_shap = shap_exps.mean(axis=0)
    mean_f    = f_vals.mean(axis=0)

    if d == 2:
        fig = plt.figure(figsize=(14, 5 * (d + 1)))
        gs = fig.add_gridspec(d + 1, 2, height_ratios=[1] * d + [1.2])
        fig.suptitle("Mean Explanations and Mean f(x)")
        x1, x2 = explain_grid[:, 0], explain_grid[:, 1]
        for feat in range(d):
            for col, (mean_vals, label, cbar_label) in enumerate([
                (mean_ale,  "ALE",  "Mean ALE"),
                (mean_shap, "SHAP", "Mean SHAP"),
            ]):
                ax = fig.add_subplot(gs[feat, col])
                hm, extent = _to_heatmap(x1, x2, mean_vals[:, feat])
                im = ax.imshow(hm, origin="lower", extent=extent,
                               cmap="viridis", aspect="auto")
                ax.set_xlabel("X1"); ax.set_ylabel("X2")
                ax.set_title(f"Mean {label} — Feature {feat + 1}")
                fig.colorbar(im, ax=ax, label=cbar_label)
        ax_f = fig.add_subplot(gs[d, :])
        hm_f, extent = _to_heatmap(x1, x2, mean_f)
        im = ax_f.imshow(hm_f, origin="lower", extent=extent,
                         cmap="viridis", aspect="auto")
        ax_f.set_xlabel("X1"); ax_f.set_ylabel("X2")
        ax_f.set_title("Mean f(x)")
        fig.colorbar(im, ax=ax_f, label="Mean f(x)")
        fig.tight_layout()
    else:
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

def plot_f_variability(view: dict, save_dir=None):
    f_vals       = view["f_vals"]
    explain_grid = view["explain_grid"]
    d = explain_grid.shape[1]

    f_std = f_vals.std(axis=0)

    if d == 2:
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.suptitle("f(x) Std Dev across Replications")
        x1, x2 = explain_grid[:, 0], explain_grid[:, 1]
        pts = np.column_stack([x1, x2])
        xi = np.linspace(x1.min(), x1.max(), 100)
        xj = np.linspace(x2.min(), x2.max(), 100)
        xi_grid, xj_grid = np.meshgrid(xi, xj)
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
# Plot: f(x) variance across replications (envelope + fill per feature)
# ---------------------------------------------------------------------------

def plot_f_variance(view: dict, save_dir=None):
    f_vals       = view["f_vals"]
    explain_grid = view["explain_grid"]
    d = explain_grid.shape[1]

    f_std = f_vals.std(axis=0)  # (explain_n,)

    fig, axes = plt.subplots(1, d, figsize=(5 * d, 5), squeeze=False)
    fig.suptitle("Std(f(x)) across Replications")
    for feat in range(d):
        ax = axes[0, feat]
        xk = explain_grid[:, feat]
        ax.scatter(xk, f_std, alpha=0.15, s=6, color="C2")
        centers, means, stds = _bin_line(xk, f_std)
        ax.plot(centers, means, color="C2", linewidth=2)
        ax.fill_between(centers, means - stds, means + stds, alpha=0.2, color="C2")
        ax.set_xlabel(f"X{feat + 1}"); ax.set_ylabel("Std(f)")
        ax.set_title(f"Feature {feat + 1}")
    fig.tight_layout()

    _show_or_save(fig, "f_variance", save_dir)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(results: dict, ale_tag: str, shap_tag: str, true_explanation_fn=None):
    """Print a summary table of bias², variance, and timing for one (ALE, SHAP) pair."""
    if true_explanation_fn is None:
        true_explanation_fn = detect_explanation_fn(results)

    ale_entry  = results["ale"][ale_tag]
    shap_entry = results["shap"][shap_tag]

    ale_exps  = ale_entry["exps"]
    shap_exps = shap_entry["exps"]
    ale_times  = ale_entry.get("times")
    shap_times = shap_entry.get("times")
    explain_grid = results["explain_grid"]
    d = ale_exps.shape[-1]

    shap_std = shap_exps.std(axis=0).mean(axis=0)
    ale_std  = ale_exps.std(axis=0).mean(axis=0)

    has_bias = true_explanation_fn is not None
    if has_bias:
        true_exp = true_explanation_fn(explain_grid)
        shap_bias2 = ((shap_exps.mean(axis=0) - true_exp) ** 2).mean(axis=0)
        ale_bias2  = ((ale_exps.mean(axis=0)  - true_exp) ** 2).mean(axis=0)

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
        print(row(f"Std Dev (D{i+1})", float(ale_std[i]), float(shap_std[i])))
    if ale_times is not None and shap_times is not None:
        print(row("Time/pt mean (s)", float(ale_times.mean()), float(shap_times.mean())))
        print(row("Time/pt std  (s)", float(ale_times.std()),  float(shap_times.std())))
    print(sep)


# ---------------------------------------------------------------------------
# Plot: ALE paths summary (all ordered feature pairs)
# ---------------------------------------------------------------------------

def plot_paths_summary_all_pairs(results: dict, ale_tag: str, cache_dir: str, save_dir=None):
    """Rebuild an ALE object from the companion run_*.pkl and call plot_paths_summary
    for every ordered pair (i, j) with i != j."""
    ale_entry = results["ale"].get(ale_tag)
    if ale_entry is None:
        print(f"[paths_summary] ALE tag not found: {ale_tag}")
        return
    ec = ale_entry.get("config")
    if ec is None:
        print(f"[paths_summary] no config in ALE entry for tag: {ale_tag}")
        return

    meta = results.get("experiment_meta", {})
    run_name = f"run_{meta['dgp_slug']}_{meta['fit_model_slug']}_n{meta['n']}_R{meta['replications']}.pkl"
    run_path = os.path.join(cache_dir, run_name)
    if not os.path.exists(run_path):
        print(f"[paths_summary] run pkl not found: {run_path}")
        return

    data_and_fitted_models = joblib.load(run_path)
    X, _y, model = data_and_fitted_models[0]

    if ec.variant == "bootstrap":
        ale_obj = BootstrapALE(
            model.predict, X,
            replications=ec.n_bootstrap,
            K=ec.K, L=ec.L, centering=ec.centering,
            interpolate=ec.interpolate, verbose=False,
        )
        ale_obj.explain(include=("total_connected",))
        ale = ale_obj.ale_replications[0]
    else:
        ale = ALE(model.predict, X, K=ec.K, L=ec.L, centering=ec.centering,
                  interpolate=ec.interpolate, verbose=False)
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

def visualize(results_path: str, ale_tag: str = None, shap_tag: str = None,
              true_explanation_fn=None, save_dir=None):
    """Load cache and produce all standard plots and summary table."""
    results = load_cache(results_path)
    cache_dir = os.path.dirname(os.path.abspath(results_path))

    if ale_tag is None:
        ale_tag = next(iter(results["ale"]))
    if shap_tag is None:
        shap_tag = next(iter(results["shap"]))

    if true_explanation_fn is None:
        true_explanation_fn = detect_explanation_fn(results)

    view = extract_view(results, ale_tag, shap_tag)
    print_summary_table(results, ale_tag, shap_tag, true_explanation_fn=true_explanation_fn)
    plot_bias2(view, true_explanation_fn=true_explanation_fn, save_dir=save_dir)
    plot_variance(view, save_dir=save_dir)
    plot_f_variability(view, save_dir=save_dir)
    plot_f_variance(view, save_dir=save_dir)
    plot_single_replication(view, r=0, save_dir=save_dir)
    plot_mean_explanations(view, save_dir=save_dir)
    plot_paths_summary_all_pairs(results, ale_tag, cache_dir=cache_dir, save_dir=save_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize experiment results from a results_*.pkl cache file."
    )
    parser.add_argument("cache_path", help="Path to the results_*.pkl cache file.")
    parser.add_argument("--ale-tag",  default=None, help="ALE tag to visualize (default: first)")
    parser.add_argument("--shap-tag", default=None, help="SHAP tag to visualize (default: first)")
    parser.add_argument(
        "--save-dir", default=None,
        help="Directory to save figures as PNG. If omitted, figures are shown interactively.",
    )
    args = parser.parse_args()
    visualize(args.cache_path, ale_tag=args.ale_tag, shap_tag=args.shap_tag, save_dir=args.save_dir)
