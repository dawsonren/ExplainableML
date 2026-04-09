"""
Multiplicative config: f(x) = x1 + x2 + 4*x1*x2
True explanation (rho-adjusted Shapley):
    phi1(x) = x1 + 2*x1*x2 - 2*rho
    phi2(x) = x2 + 2*x1*x2 - 2*rho

Tests how ALE and SHAP handle a signal with a pure interaction term.
The rho correction accounts for E[X1*X2] = rho under the Gaussian sampler.
"""

from models import (
    NNModelTuner,
    sample_X_gaussian,
    signal_multiplicative, signal_multiplicative_explanation,
)
from experiments import ExplainerConfig

REPLICATIONS = 100
EXPLAIN_N = 500

# rho_explanation_factory: (rho) -> (X -> (N, d))
SIGNALS = [
    (
        signal_multiplicative,
        None,
        lambda rho: (lambda X: signal_multiplicative_explanation(X, rho)),
    ),
]

SNRS = [9]

NS = [1000]

ALE_CONFIGS = [
    ExplainerConfig(K=40, L=25),
]

RHOS = [0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]

SAMPLERS = [
    ("gaussian", lambda rho: sample_X_gaussian(cov=[[1, rho], [rho, 1]])),
]

MODEL_TYPES = [
    ("nn", lambda snr: NNModelTuner(cv=5, n_iter=20, verbose=False, snr=snr)),
]
