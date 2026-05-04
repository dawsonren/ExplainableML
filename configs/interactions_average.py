"""
Tests out functions with interactions.
"""

from models import (
    NNModelTuner,
    sample_X_gaussian,
    signal_basic_interaction
)
from experiments import ExplainerConfig, ShapConfig

REPLICATIONS = 100
EXPLAIN_N = 500

SIGNALS = [
    (signal_basic_interaction,     None,     None),
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
    ExplainerConfig(K=40, L=25, local_method="path_integral", background_size=1000, background_seed=42, boundary_interp=True, tag="K40_L25_path_integral"),
    ExplainerConfig(K=40, L=25, local_method="path_rep",    tag="K40_L25_path_rep"),
]

SHAP_CONFIGS = [
    ShapConfig(method="exact_shap"),
]
