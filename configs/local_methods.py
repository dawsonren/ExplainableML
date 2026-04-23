"""
Local-method comparison config: signal_basic across full RHO sweep.

Compares the three explain_local variants ("interpolate", "path_rep", "self")
on the same fitted ALE so the only thing that varies is how the local effect
is computed.
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
from experiments import ExplainerConfig, ShapConfig

REPLICATIONS = 100
EXPLAIN_N = 500

SIGNALS = [
    (signal_basic,     signal_basic_explanation,     None),
    (signal_nonlinear, signal_nonlinear_explanation, None),
    (signal_threshold, signal_threshold_explanation, None),  # discontinuous at x1=0
    (signal_cubic,     signal_cubic_explanation,     None),  # high curvature
    # (signal_abs,       signal_abs_explanation,       None),  # non-differentiable kink
]

SNRS = [9]
NS = [1000]
RHOS = [0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]

SAMPLERS = [
    ("gaussian", lambda rho: sample_X_gaussian(cov=[[1, rho], [rho, 1]])),
]

MODEL_TYPES = [
    ("nn", lambda snr: NNModelTuner(cv=5, n_iter=20, verbose=False, snr=snr)),
]

ALE_CONFIGS = [
    ExplainerConfig(K=40, L=25, local_method="interpolate", tag="K40_L25_interpolate"),
    ExplainerConfig(K=40, L=25, local_method="path_rep",    tag="K40_L25_path_rep"),
    ExplainerConfig(K=40, L=25, local_method="self",        tag="K40_L25_self"),
]

SHAP_CONFIGS = [
    ShapConfig(method="exact_shap"),
]
