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
    signal_basic,
    signal_basic_explanation
)
from experiments import ExplainerConfig


def _block_cov(rho: float) -> np.ndarray:
    """10x10 block-diagonal covariance with two blocks of 5, within-block corr = rho."""
    cov = np.zeros((10, 10))
    for block in [range(0, 10, 2), range(1, 10, 2)]:  # odd indices, even indices
        for i in block:
            for j in block:
                cov[i, j] = rho if i != j else 1.0
    return cov


def _make_sampler(rho: float):
    cov = _block_cov(rho)
    cov_hash = hashlib.md5(np.ascontiguousarray(cov).tobytes()).hexdigest()[:8]
    def sample(n, rng: np.random.Generator):
        return rng.multivariate_normal(mean=np.zeros(10), cov=cov, size=n)
    sample.__name__ = f"sample_X_gaussian_d10_{cov_hash}"
    return sample


REPLICATIONS = 100
EXPLAIN_N = 500

SIGNALS = [
    (signal_basic, signal_basic_explanation, None),
]

SNRS = [9]

NS = [1000]

ALE_CONFIGS = [
    ExplainerConfig(K=40, L=25),
]

RHOS = [0, 0.3, 0.5, 0.7, 0.9]

SAMPLERS = [
    ("gaussian", _make_sampler),
]

MODEL_TYPES = [
    ("nn", lambda snr: NNModelTuner(cv=5, n_iter=20, verbose=False, snr=snr)),
]
