"""
Basic config: signal_basic and signal_nonlinear over 2D Gaussian features.

SIGNALS format: (signal_fn, static_explanation_fn_or_none, rho_explanation_factory_or_none)
  - static_explanation_fn : (X) -> (N, d) array, or None
  - rho_explanation_factory: (rho) -> ((X) -> (N, d) array), or None
"""

from models import (
    NNModelTuner,
    sample_X_gaussian,
    signal_basic,     signal_basic_explanation,
    signal_nonlinear, signal_nonlinear_explanation,
    signal_threshold, signal_threshold_explanation,
    signal_cubic,     signal_cubic_explanation,
    signal_abs,       signal_abs_explanation,
)
from experiments import ExplainerConfig

REPLICATIONS = 100
EXPLAIN_N = 500

SIGNALS = [
    (signal_basic,     signal_basic_explanation,     None),
    (signal_nonlinear, signal_nonlinear_explanation, None),
    (signal_threshold, signal_threshold_explanation, None),  # discontinuous at x1=0
    (signal_cubic,     signal_cubic_explanation,     None),  # high curvature
    (signal_abs,       signal_abs_explanation,       None),  # non-differentiable kink
]

SNRS = [9]

NS = [1000]

ALE_CONFIGS = [
    ExplainerConfig(K=40, L=25),
    ExplainerConfig(K=20, L=50),
    ExplainerConfig(K=10, L=100),
    ExplainerConfig(K=40, L=25, levels_up=2),
    ExplainerConfig(K=40, L=25, variant="bootstrap", n_bootstrap=5),
]

# TODO: try levels_up along with L larger, L "over-subscribing"

RHOS = [0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]

# (slug_prefix, sampler_factory(rho) -> sampler)
SAMPLERS = [
    ("gaussian", lambda rho: sample_X_gaussian(cov=[[1, rho], [rho, 1]])),
]

# (slug_prefix, tuner_factory(snr) -> tuner)
MODEL_TYPES = [
    ("nn", lambda snr: NNModelTuner(cv=5, n_iter=20, verbose=False, snr=snr)),
    # ("rf", lambda snr: RFModelTuner(cv=5, n_iter=20, verbose=False, snr=snr)),
]
