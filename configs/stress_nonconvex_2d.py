"""
Stress-test config: non-convex and multi-modal distributions.

Tests whether ALE path creation avoids extrapolation through low-density regions.
Distributions chosen because their support is non-convex (donut), strongly curved
(banana), or disconnected (mixture), so naive paths would cross empty regions.

Note: signal_*_explanation functions were derived under uniform/Gaussian marginals,
so error metrics relative to ground truth will not be exact here. The primary use
of this config is method comparison (path_rep vs multi_path_interpolate ALE) across shapes.
"""

from models import (
    NNModelTuner,
    sample_X_donut, sample_X_banana, sample_X_mixture, sample_X_gaussian,
    signal_basic,             signal_basic_explanation,
    signal_nonlinear,         signal_nonlinear_explanation,
    signal_threshold,         signal_threshold_explanation,
    signal_cubic,             signal_cubic_explanation,
    signal_abs,               signal_abs_explanation,
    signal_basic_interaction, signal_basic_interaction_explanation,
)
from experiments import ExplainerConfig, ShapConfig

REPLICATIONS = 100
EXPLAIN_N = 100

SIGNALS = [
    (signal_basic,             signal_basic_explanation,             None),
    (signal_nonlinear,         signal_nonlinear_explanation,         None),
    # (signal_threshold,         signal_threshold_explanation,         None),
    # (signal_cubic,             signal_cubic_explanation,             None),
    (signal_abs,               signal_abs_explanation,               None),
    (signal_basic_interaction, signal_basic_interaction_explanation, None),
]

SNRS = [9]
NS = [1000]
RHOS = [0]  # dummy — no correlation sweep; variation is across SAMPLERS

SAMPLERS = [
    ("donut",   lambda rho: sample_X_donut(r_min=1.0, r_max=2.0)),
    ("banana",  lambda rho: sample_X_banana(curvature=1.0, sigma_x=1.0, sigma_noise=0.2)),
    ("mixture", lambda rho: sample_X_mixture([
        (0.5, sample_X_gaussian(cov=[[1.0, 0], [0, 1.0]])),
        (0.5, sample_X_gaussian(cov=[[0.5, 0.3], [0.3, 0.5]], mean=[2.0, 2.0])),
    ])),
]

MODEL_TYPES = [
    ("nn", lambda snr: NNModelTuner(cv=5, n_iter=20, verbose=False, snr=snr)),
]

ALE_CONFIGS = [
    ExplainerConfig(K=40, L=25, local_method="path_rep",   tag="K40_L25_path_rep"),
    ExplainerConfig(K=40, L=25, local_method="multi_path_interpolate",   tag="K40_L25_multi_path_interpolate", background_size=1000, background_seed=42, boundary_interp=False)
]

SHAP_CONFIGS = [
    ShapConfig(method="exact_shap"),
]
