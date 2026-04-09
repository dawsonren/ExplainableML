"""
Threshold config: f(x) = sign(X1) + X2^2
True explanation: [sign(X1), X2^2]

Tests how ALE and SHAP handle a discontinuous (but additive) signal.
"""

from models import (
    NNModelTuner,
    sample_X_gaussian,
    signal_threshold, signal_threshold_explanation,
)
from experiments import ExplainerConfig

REPLICATIONS = 100
EXPLAIN_N = 500

SIGNALS = [
    (signal_threshold, signal_threshold_explanation, None),
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
