"""
Standalone diagnostic experiments to isolate ALE variance sources.

Experiment 4: Cumsum amplification -- compare pre-cumsum delta variance
              vs post-cumsum g-value variance across bins.
Experiment 5: True signal -- run ALE and SHAP with f=signal_basic
              (no model noise) to separate model noise from ALE machinery noise.

Usage:
    python diagnostic_scripts/isolate_variance.py
"""

import os
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt

# Allow imports from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import signal_basic, signal_basic_explanation, sample_X_gaussian
from experiments import DGP, Experiment, ExplainerConfig
from ale import ALE
from ale.ale_vim import _ale_total_vim
from ale.shared import calculate_edges, calculate_K, calculate_bins, calculate_deltas
from shapley import SHAP

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

REPLICATIONS = 100
N = 1000
EXPLAIN_N = 500
SNR = 9
K = 40
L = 25


# ---------------------------------------------------------------------------
# Experiment 4: Cumsum Amplification
# ---------------------------------------------------------------------------

def experiment_4_cumsum_amplification(rho=0.0):
    """
    For each replication, extract deltas_by_path (pre-cumsum) and g_values
    (post-cumsum). Compare variance by bin index to see if cumsum amplifies noise.
    """
    print(f"\n=== Experiment 4: Cumsum Amplification (rho={rho}) ===")

    sampler = sample_X_gaussian(cov=[[1, rho], [rho, 1]])
    dgp = DGP(snr=SNR, sample_X=sampler, signal=signal_basic)
    rng = np.random.default_rng(42)

    # We need the fitted models to match what run_experiments uses
    from models import NNModelTuner
    tuner = NNModelTuner(cv=5, n_iter=20, verbose=False, snr=SNR)
    X_tune, y_tune = dgp.sample(n=N, rng=rng)
    model = tuner.tune(X_tune, y_tune, rng, dgp_slug=dgp.slug, n=N)

    experiment = Experiment(
        dgp=dgp, fit_model=model, dgp_slug=dgp.slug,
        fit_model_slug=model.__name__, replications=REPLICATIONS, n=N, save=True,
    )
    data_and_models = experiment.fit_models(rng)

    # For each feature, collect deltas_by_path and g_values across replications
    for feature_idx in [1, 2]:  # 1-based
        idx = feature_idx - 1
        all_deltas_by_path = []  # (R, K_actual, L)
        all_g_values = []        # (R, K_actual, L)

        for r, (X, y, fitted_model) in enumerate(data_and_models[:REPLICATIONS]):
            f = fitted_model.predict
            categorical = [False] * X.shape[1]

            (
                total_vim, forest, paths, g_values,
                centered_g_values, l_x, deltas, k_x,
            ) = _ale_total_vim(
                f, X, feature_idx, K, L, categorical, {},
                method="connected", interpolate=True, centering="y",
                edges=None, knn_smooth=None,
            )

            # Recover deltas_by_path from g_values (g_values = cumsum of deltas_by_path)
            deltas_by_path = np.diff(g_values, axis=0, prepend=0)
            # g_values here is the raw (non-centered) accumulated values

            all_deltas_by_path.append(deltas_by_path)
            all_g_values.append(g_values)

        # Stack: all have shape (R, K_actual, L), but K_actual may vary slightly
        # Use the minimum K across replications
        min_K = min(d.shape[0] for d in all_deltas_by_path)
        deltas_stack = np.array([d[:min_K, :] for d in all_deltas_by_path])  # (R, K, L)
        gvals_stack = np.array([g[:min_K, :] for g in all_g_values])         # (R, K, L)

        # Variance across replications for each (k, l) cell
        delta_var = deltas_stack.var(axis=0)  # (K, L)
        gval_var = gvals_stack.var(axis=0)    # (K, L)

        # Average across paths
        delta_var_by_bin = delta_var.mean(axis=1)  # (K,)
        gval_var_by_bin = gval_var.mean(axis=1)    # (K,)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(range(min_K), delta_var_by_bin, 'b-o', markersize=3, label='Delta variance')
        axes[0].plot(range(min_K), gval_var_by_bin, 'r-o', markersize=3, label='G-value variance')
        axes[0].set_xlabel('Bin index k')
        axes[0].set_ylabel('Variance across replications')
        axes[0].set_title(f'Feature {feature_idx} (rho={rho})')
        axes[0].legend()
        axes[0].set_yscale('log')

        axes[1].plot(range(min_K), gval_var_by_bin / np.maximum(delta_var_by_bin, 1e-12),
                     'g-o', markersize=3)
        axes[1].set_xlabel('Bin index k')
        axes[1].set_ylabel('G-value var / Delta var')
        axes[1].set_title(f'Amplification ratio (feature {feature_idx})')

        fig.suptitle(f'Experiment 4: Cumsum Amplification (K={K}, L={L}, rho={rho})')
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f'exp4_cumsum_rho{rho}_f{feature_idx}.png'), dpi=150)
        plt.close(fig)

        print(f"  Feature {feature_idx}: delta_var range [{delta_var_by_bin.min():.6f}, {delta_var_by_bin.max():.6f}]")
        print(f"  Feature {feature_idx}: gval_var  range [{gval_var_by_bin.min():.6f}, {gval_var_by_bin.max():.6f}]")
        print(f"  Feature {feature_idx}: amplification ratio at last bin = {gval_var_by_bin[-1] / max(delta_var_by_bin[-1], 1e-12):.1f}x")


# ---------------------------------------------------------------------------
# Experiment 5: True Signal (no model noise)
# ---------------------------------------------------------------------------

def experiment_5_true_signal(rho=0.0):
    """
    Run ALE and SHAP with f=signal_basic instead of fitted models.
    This eliminates model fitting noise and isolates ALE's partitioning variance.
    """
    print(f"\n=== Experiment 5: True Signal (rho={rho}) ===")

    sampler = sample_X_gaussian(cov=[[1, rho], [rho, 1]])
    rng_base = np.random.default_rng(42)

    # Fixed explain grid
    explain_grid = sampler(n=EXPLAIN_N, rng=np.random.default_rng(99))

    # True explanations for reference
    true_exp = signal_basic_explanation(explain_grid)

    ale_exps_all = []
    shap_exps_all = []

    for r in range(REPLICATIONS):
        # Sample fresh training data (different each replication)
        X = sampler(n=N, rng=rng_base)

        # ALE with true signal
        ale = ALE(
            signal_basic, X, K=K, L=L,
            centering="y", interpolate=True, verbose=False,
        )
        ale.explain(include=("total_connected",))
        ale_exp = ale.explain_local(explain_grid)
        ale_exps_all.append(ale_exp)

        # SHAP with true signal
        shapley = SHAP(signal_basic, X)
        shap_exp = shapley.explain_local(explain_grid, method="exact_shap")
        shap_exps_all.append(shap_exp)

    ale_exps = np.array(ale_exps_all)   # (R, EXPLAIN_N, d)
    shap_exps = np.array(shap_exps_all) # (R, EXPLAIN_N, d)

    ale_var = ale_exps.var(axis=0).mean(axis=0)
    shap_var = shap_exps.var(axis=0).mean(axis=0)

    ale_bias2 = ((ale_exps.mean(axis=0) - true_exp) ** 2).mean(axis=0)
    shap_bias2 = ((shap_exps.mean(axis=0) - true_exp) ** 2).mean(axis=0)

    print(f"  ALE  variance: d1={ale_var[0]:.6f}, d2={ale_var[1]:.6f}")
    print(f"  SHAP variance: d1={shap_var[0]:.6f}, d2={shap_var[1]:.6f}")
    print(f"  ALE  bias^2:   d1={ale_bias2[0]:.6f}, d2={ale_bias2[1]:.6f}")
    print(f"  SHAP bias^2:   d1={shap_bias2[0]:.6f}, d2={shap_bias2[1]:.6f}")
    print(f"  Variance ratio (ALE/SHAP): d1={ale_var[0]/max(shap_var[0],1e-12):.2f}x, d2={ale_var[1]/max(shap_var[1],1e-12):.2f}x")

    # Now compare with fitted model results (load from cache if available)
    print("\n  --- Comparison with fitted model ---")

    # Plot: variance per explanation point for true signal
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for d_idx in range(2):
        ale_pointwise_var = ale_exps[:, :, d_idx].var(axis=0)   # (EXPLAIN_N,)
        shap_pointwise_var = shap_exps[:, :, d_idx].var(axis=0) # (EXPLAIN_N,)

        x_vals = explain_grid[:, d_idx]
        order = np.argsort(x_vals)

        axes[d_idx].scatter(x_vals[order], ale_pointwise_var[order], s=3, alpha=0.5, label='ALE')
        axes[d_idx].scatter(x_vals[order], shap_pointwise_var[order], s=3, alpha=0.5, label='SHAP')
        axes[d_idx].set_xlabel(f'x{d_idx+1}')
        axes[d_idx].set_ylabel('Pointwise variance')
        axes[d_idx].set_title(f'Feature {d_idx+1}')
        axes[d_idx].legend()

    fig.suptitle(f'Experiment 5: True Signal Variance (K={K}, L={L}, rho={rho})')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f'exp5_true_signal_rho{rho}.png'), dpi=150)
    plt.close(fig)

    # Save numerical results
    np.savez(
        os.path.join(OUTPUT_DIR, f'exp5_true_signal_rho{rho}.npz'),
        ale_exps=ale_exps, shap_exps=shap_exps,
        ale_var=ale_var, shap_var=shap_var,
        ale_bias2=ale_bias2, shap_bias2=shap_bias2,
        explain_grid=explain_grid, true_exp=true_exp,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("ALE Variance Diagnostic Experiments")
    print("=" * 60)

    for rho in [0.0, 0.7]:
        experiment_4_cumsum_amplification(rho=rho)

    for rho in [0.0, 0.7]:
        experiment_5_true_signal(rho=rho)

    print(f"\nPlots and data saved to {OUTPUT_DIR}/")
