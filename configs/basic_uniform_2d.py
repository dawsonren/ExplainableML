"""
Local-method comparison config: signal_basic across full RHO sweep.

Compares the explain_local variants ("path_rep", "multi_path_interpolate")
on the same fitted ALE so the only thing that varies is how the local effect
is computed.
"""

from models import (
    NNModelTuner,
    sample_X_uniform,
    signal_basic,     signal_basic_explanation,
    signal_nonlinear, signal_nonlinear_explanation,
    signal_threshold, signal_threshold_explanation,
    signal_cubic,     signal_cubic_explanation,
    signal_abs,       signal_abs_explanation,
    signal_basic_interaction, signal_basic_interaction_explanation
)
from experiments import ExplainerConfig, ShapConfig

REPLICATIONS = 100
EXPLAIN_N = 100

SIGNALS = [
    (signal_basic,     signal_basic_explanation,     None),
    (signal_nonlinear, signal_nonlinear_explanation, None),
    (signal_threshold, signal_threshold_explanation, None),  # discontinuous at x1=0
    (signal_cubic,     signal_cubic_explanation,     None),  # high curvature
    (signal_abs,       signal_abs_explanation,       None),  # non-differentiable kink
    (signal_basic_interaction, signal_basic_interaction_explanation, None),  # interaction with no main effects
]

SNRS = [9]
NS = [1000]
RHOS = [0, 0.3, 0.5, 0.7, 0.9, 0.95]

SAMPLERS = [
    ("uniform", lambda rho: sample_X_uniform(corr=[[1, rho], [rho, 1]], scale=[-1, 1])),
]

MODEL_TYPES = [
    ("nn", lambda snr: NNModelTuner(cv=5, n_iter=20, verbose=False, snr=snr)),
]

ALE_CONFIGS = [
    ExplainerConfig(K=40, L=25, local_method="path_rep",    tag="K40_L25_path_rep"),
    ExplainerConfig(K=40, L=25, local_method="multi_path_interpolate",   tag="K40_L25_multi_path_interpolate", background_size=1000, background_seed=42, boundary_interp=False)
]

SHAP_CONFIGS = [
    ShapConfig(method="exact_shap"),
]

poopy_butthole = "hi im poopy butthole"