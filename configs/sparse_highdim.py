"""
Sparse high-dimensional config: f(x) = x1 + x2^2, d=10

Features 3-10 are pure noise. Correlation structure:
  - Block 1: {X1, X3, X5, X7, X9}  — within-block correlation = rho
  - Block 2: {X2, X4, X6, X8, X10} — within-block correlation = rho
  - Between blocks: 0

This tests whether ALE and SHAP variance inflates for noise features
that are correlated with the signal features (X1 and X2).
No true explanation provided since the ground truth is not well-defined
for the full 10-dimensional input under this correlation structure.
"""

import numpy as np
import hashlib

from models import (
    NNModelTuner,
    RFModelTuner,
    sample_X_uniform,
    signal_hooker_2021,
    signal_hooker_2021_explanation
)
from experiments import ExplainerConfig, ShapConfig


REPLICATIONS = 10
EXPLAIN_N = 500

SIGNALS = [
    (signal_hooker_2021, signal_hooker_2021_explanation, None),
]

SNRS = [9, 19, 49, 99]

NS = [1000]

ALE_CONFIGS = [
    ExplainerConfig(K=40, L=25),
]

SHAP_CONFIGS = [
    ShapConfig(method="tree_shap", kwargs={"npermutations": 10}),
]

RHOS = [0, 0.3, 0.5, 0.7, 0.9]

def make_corr(rho):
    corr = np.eye(10)
    # set X_1, X_2 correlated w/ rho
    corr[0, 1] = corr[1, 0] = rho
    return corr

SAMPLERS = [
    ("uniform", lambda rho: sample_X_uniform(corr=make_corr(rho), scale=[0, 1])),
]

MODEL_TYPES = [
    # ("nn", lambda snr: NNModelTuner(cv=5, n_iter=20, verbose=False, snr=snr)),
    ("rf", lambda snr: RFModelTuner(cv=5, n_iter=20, verbose=False, snr=snr)),
]
