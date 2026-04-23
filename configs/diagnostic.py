"""
Diagnostic config: isolate sources of ALE variance vs SHAP.

Focuses on signal_basic at rho=0 (where ALE is worst) and rho=0.7 (moderate).
Three experiment groups:
  1. L=1 vs L>1 -- does path partitioning cause the excess variance?
  2. K*L product sweep -- is variance driven by cell sparsity n/(K*L)?
  3. Fixed edges -- does bin-edge instability across replications matter?
"""

import numpy as np

from models import (
    NNModelTuner,
    sample_X_gaussian,
    signal_basic, signal_basic_explanation,
)
from experiments import ExplainerConfig, ShapConfig

REPLICATIONS = 100
EXPLAIN_N = 500

SIGNALS = [
    (signal_basic, signal_basic_explanation, None),
]

SNRS = [9]
NS = [1000]
RHOS = [0, 0.7]

SAMPLERS = [
    ("gaussian", lambda rho: sample_X_gaussian(cov=[[1, rho], [rho, 1]])),
]

MODEL_TYPES = [
    ("nn", lambda snr: NNModelTuner(cv=5, n_iter=20, verbose=False, snr=snr)),
]

# ---------------------------------------------------------------------------
# Precompute population-quantile edges for N(0,1) (Experiment 3)
# ---------------------------------------------------------------------------
_large_x = np.random.default_rng(0).standard_normal(1_000_000)
_fixed_edges_40 = np.quantile(_large_x, np.linspace(0, 1, 41))
_fixed_edges_10 = np.quantile(_large_x, np.linspace(0, 1, 11))

ALE_CONFIGS = [
    # --- Experiment 1: L=1 vs L>1 (path noise isolation) ---
    ExplainerConfig(K=10, L=1,   tag="exp1_K10_L1"),
    ExplainerConfig(K=10, L=25,  tag="exp1_K10_L25"),
    ExplainerConfig(K=10, L=100, tag="exp1_K10_L100"),
    ExplainerConfig(K=40, L=1,   tag="exp1_K40_L1"),
    ExplainerConfig(K=40, L=25,  tag="exp1_K40_L25"),

    # --- Experiment 2: K*L product sweep (cell sparsity) ---
    ExplainerConfig(K=5,  L=5,   tag="exp2_KL25"),
    ExplainerConfig(K=10, L=10,  tag="exp2_KL100"),
    ExplainerConfig(K=10, L=25,  tag="exp2_KL250"),
    ExplainerConfig(K=20, L=25,  tag="exp2_KL500"),
    # K=40,L=25 and K=10,L=100 already covered in Experiment 1

    # --- Experiment 3: Fixed edges (bin-edge variance) ---
    ExplainerConfig(K=40, L=25,  edges={1: _fixed_edges_40, 2: _fixed_edges_40}, tag="exp3_fixed_K40_L25"),
    ExplainerConfig(K=10, L=100, edges={1: _fixed_edges_10, 2: _fixed_edges_10}, tag="exp3_fixed_K10_L100"),
]

SHAP_CONFIGS = [
    ShapConfig(method="exact_shap"),
]
