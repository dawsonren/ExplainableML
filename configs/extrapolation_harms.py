"""
Extrapolation-harm sweep.

Story: as the fitted NN gets more "overfit" (larger width / smaller alpha),
predictions extrapolate further off the data manifold in correlated regions.
The proposed local-ALE method should stay close to the (interventional-Shapley)
ground truth, while exact SHAP — being computed *on the model* — inherits that
extrapolation. The widening gap as overfit increases is the headline.

Signal is additive `sum_j x_j^2` with N(0,1) marginals, so the interventional
Shapley target is `x_j^2 - 1` per feature for *any* correlation structure.

Three correlation shapes:
  - "pairs_k": k disjoint correlated pairs ((x0,x1), (x2,x3), ...), rest indep.
    1 pair always; 2 pairs when d>=4; 3 pairs when d>=6.
  - "ar1": cov[i,j] = rho^|i-j|.

NN sweep is a Cartesian product of widths x alphas using FixedNNTuner
(no CV — we want to walk the overfit axis on purpose).
"""

from models import (
    FixedNNTuner,
    sample_X_gaussian_pairs,
    sample_X_gaussian_ar1,
    signal_additive_quadratic, signal_additive_quadratic_explanation,
)
from experiments import ExplainerConfig, ShapConfig

REPLICATIONS = 25
EXPLAIN_N = 100

SIGNALS = [
    (signal_additive_quadratic, signal_additive_quadratic_explanation, None),
]

SNRS = [9]
NS = [1000]
RHOS = [0.5, 0.9]

# Each sampler factory bakes in the dimensionality. The signal adapts to d.
def _pairs(d, n_pairs):
    return (
        f"d{d}_pairs{n_pairs}",
        lambda rho, d=d, n_pairs=n_pairs: sample_X_gaussian_pairs(d=d, rho=rho, n_pairs=n_pairs),
    )

def _ar1(d):
    return (
        f"d{d}_ar1",
        lambda rho, d=d: sample_X_gaussian_ar1(d=d, rho=rho),
    )

SAMPLERS = [
    # d=2: only 1 pair possible
    _pairs(2, 1),
    # d=3
    _pairs(3, 1), # _ar1(3),
    # d=4: 1 or 2 pairs
    _pairs(4, 1), _pairs(4, 2), #  _ar1(4),
    # d=5
    # _pairs(5, 1), _pairs(5, 2), _ar1(5),
    # d=6: also 3 pairs
    # _pairs(6, 1), _pairs(6, 2), _pairs(6, 3), _ar1(6),
]

# Walk the overfit axis: wider/deeper + smaller alpha = more overfit.
_WIDTHS = [(100, )]
_ALPHAS = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]

MODEL_TYPES = [
    (
        f"nn_h{'x'.join(str(s) for s in w)}_a{a:g}",
        lambda snr, w=w, a=a: FixedNNTuner(hidden_layer_sizes=w, alpha=a, activation="tanh", snr=snr),
    )
    for w in _WIDTHS for a in _ALPHAS
]

ALE_CONFIGS = [
    ExplainerConfig(
        K=40, L=25, local_method="multi_path_interpolate",
        tag="K40_L25_multi_path_interpolate",
        background_size=1000, background_seed=42, boundary_interp=False,
    ),
]

SHAP_CONFIGS = [
    ShapConfig(method="exact_shap"),
]
