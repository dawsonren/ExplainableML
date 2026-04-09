"""
Model factory functions and DGP sampling strategies for experiments.
Import these into notebooks instead of defining them inline.
"""

import hashlib
import json
import os

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


# ---------------------------------------------------------------------------
# DGP sampling strategies
# ---------------------------------------------------------------------------

def sample_X_uniform(rho, scale, d=2):
    """
    Sample d-dimensional features with controlled pairwise correlation via a shared uniform latent.

    rho controls correlation: Cor(Xi, Xj) ≈ rho for all i≠j
    scale controls feature spread: features are approximately in [-scale, scale]
    """
    def sample(n, rng: np.random.Generator):
        sigma_x = scale * np.sqrt((1 - rho) / (3 * rho))
        x = rng.uniform(-scale, scale, size=n)
        cols = [x + rng.normal(size=n, scale=sigma_x) for _ in range(d)]
        return np.column_stack(cols)
    sample.__name__ = f"sample_X_uniform_d{d}_rho{rho:g}_scale{scale:g}"
    return sample


def sample_X_gaussian(cov):
    """Sample d-dimensional features from a multivariate Gaussian with the given covariance matrix."""
    cov = np.array(cov, dtype=float)
    d = cov.shape[0]
    cov_hash = hashlib.md5(np.ascontiguousarray(cov).tobytes()).hexdigest()[:8]
    def sample(n, rng: np.random.Generator):
        return rng.multivariate_normal(mean=np.zeros(d), cov=cov, size=n)
    sample.__name__ = f"sample_X_gaussian_d{d}_{cov_hash}"
    return sample


# ---------------------------------------------------------------------------
# Signal functions
# ---------------------------------------------------------------------------

def signal_basic(X):
    return X[:, 0] + X[:, 1] ** 2

def signal_basic_interaction(X):
    return X[:, 0] + X[:, 1] + X[:, 0] * X[:, 1]

def signal_nonlinear(X):
    return X[:, 0] + np.sin(4 * X[:, 1])

def signal_nonlinear_interaction(X):
    return X[:, 0] + np.sin(4 * X[:, 1]) + np.sin(4 * X[:, 0] * X[:, 1])

def signal_tricky_valley_rho_99(X):
    return (X[:, 0] + X[:, 1]) + 10 * np.maximum(
        0, np.abs(X[:, 0] - X[:, 1]) - 0.15
) ** (1 / 3)

def signal_tricky_valley_rho_9(X):
    return (X[:, 0] + X[:, 1]) + 10 * np.maximum(
        0, np.abs(X[:, 0] - X[:, 1]) - 0.45
    ) ** (1 / 3)

def signal_threshold(X):
    return np.sign(X[:, 0]) + X[:, 1] ** 2

def signal_multiplicative(X):
    return X[:, 0] + X[:, 1] + 4 * X[:, 0] * X[:, 1]

def signal_cubic(X):
    """x1^3 + x2^2 — high curvature; stresses ALE finite-difference approximation."""
    return X[:, 0] ** 3 + X[:, 1] ** 2

def signal_abs(X):
    """abs(x1) + x2^2 — non-differentiable kink at x1=0."""
    return np.abs(X[:, 0]) + X[:, 1] ** 2


# ---------------------------------------------------------------------------
# True explanations (only for additive signals)
# ---------------------------------------------------------------------------

def signal_basic_explanation(X):
    return np.vstack([X[:, 0], X[:, 1] ** 2 - 1]).T

def signal_nonlinear_explanation(X):
    return np.vstack([X[:, 0], np.sin(4 * X[:, 1])]).T

def signal_threshold_explanation(X):
    return np.column_stack([np.sign(X[:, 0]), X[:, 1] ** 2 - 1])

def signal_cubic_explanation(X):
    """Shapley under N(0,1): E[X^3]=0 (odd moment), E[X^2]=1."""
    return np.column_stack([X[:, 0] ** 3, X[:, 1] ** 2 - 1])

def signal_abs_explanation(X):
    """Shapley under N(0,1): E[|X|]=sqrt(2/pi)."""
    return np.column_stack([np.abs(X[:, 0]) - np.sqrt(2 / np.pi), X[:, 1] ** 2 - 1])

def signal_multiplicative_explanation(X, rho):
    """Shapley-value explanation for signal_multiplicative under Gaussian(0, [[1,rho],[rho,1]])."""
    phi1 = X[:, 0] + 2 * X[:, 0] * X[:, 1] - 2 * rho
    phi2 = X[:, 1] + 2 * X[:, 0] * X[:, 1] - 2 * rho
    return np.column_stack([phi1, phi2])

# ---------------------------------------------------------------------------
# Model factory functions
# ---------------------------------------------------------------------------

def fit_linear_model(X, y, rng):
    lm = LinearRegression()
    lm.fit(X, y)
    return lm


def make_nn_model(hidden_layer_sizes=(40,), activation="tanh", alpha=1e-4, solver="lbfgs"):
    """Return a fit_model callable for MLPRegressor with the given hyperparameters."""
    def fit(X, y, rng: np.random.Generator):
        seed = int(rng.integers(0, 2**31 - 1))
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            alpha=alpha,
            solver=solver,
            max_iter=1000,
            random_state=seed,
        )
        model.fit(X, y)
        return model
    hidden_str = "x".join(str(s) for s in hidden_layer_sizes)
    fit.__name__ = f"nn_h{hidden_str}_{activation}_alpha{alpha:g}_{solver}"
    return fit


def make_rf_model(n_estimators=100, max_depth=None, min_samples_split=2,
                  max_features="sqrt", min_samples_leaf=1):
    """Return a fit_model callable for RandomForestRegressor."""
    def fit(X, y, rng: np.random.Generator):
        seed = int(rng.integers(0, 2**31 - 1))
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
        )
        model.fit(X, y)
        return model
    fit.__name__ = (
        f"rf_n{n_estimators}_depth{max_depth}_minsplit{min_samples_split}"
        f"_maxfeat{max_features}_minleaf{min_samples_leaf}"
    )
    return fit


class NNModelTuner:
    """
    Runs RandomizedSearchCV on one dataset to find the best MLPRegressor
    hyperparameters, then returns a fixed make_nn_model callable for use
    across all replications.

    Usage
    -----
    X_tune, y_tune = dgp.sample(n, rng)
    nn_model = NNModelTuner(cv=5, n_iter=20, verbose=True, snr=9).tune(X_tune, y_tune, rng)
    experiment = Experiment(..., fit_model=nn_model, fit_model_slug=nn_model.__name__)
    """
    _param_dist = {
        "hidden_layer_sizes": [(2,), (3,), (5,), (10,), (20,), (50,), (100,)],
        "activation": ["tanh", "relu"],
        "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        "solver": ["lbfgs"],
    }

    def __init__(self, cv=5, n_iter=20, verbose=False, snr=None):
        self.cv = cv
        self.n_iter = n_iter
        self.verbose = verbose
        self.snr = snr

    def tune(self, X, y, rng: np.random.Generator,
             dgp_slug=None, n=None, cache_dir="cached_explanations"):
        """
        Fit CV on (X, y) and return a fixed make_nn_model callable.

        If dgp_slug and n are provided, best params are cached to disk so
        subsequent calls with the same setup skip the search entirely.
        """
        cache_path = None
        if dgp_slug is not None and n is not None:
            fname = f"tune_nn_{dgp_slug}_n{n}_cv{self.cv}_niter{self.n_iter}.json"
            cache_path = os.path.join(cache_dir, fname)

        if cache_path is not None and os.path.exists(cache_path):
            with open(cache_path) as f:
                p = json.load(f)
            p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
            if self.verbose:
                print(f"[nn_cv] loaded cached params: {p}")
        else:
            seed = int(rng.integers(0, 2**31 - 1))
            search = RandomizedSearchCV(
                MLPRegressor(max_iter=1000, random_state=seed),
                self._param_dist,
                n_iter=self.n_iter, cv=self.cv, scoring="r2",
                random_state=seed, n_jobs=-1,
            )
            search.fit(X, y)
            p = search.best_params_
            if self.verbose:
                print(f"[nn_cv] best params : {p}")
                print(f"[nn_cv] CV R²       : {search.best_score_:.4f}")
                if self.snr is not None:
                    print(f"[nn_cv] theoretical R² (SNR={self.snr}): {1 - 1 / (1 + self.snr):.4f}")
            if cache_path is not None:
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump({
                        **p,
                        "hidden_layer_sizes": list(p["hidden_layer_sizes"]),
                        "cv_r2": search.best_score_,
                        "max_r2": 1 - 1 / (1 + self.snr) if self.snr is not None else None,
                    }, f, indent=2)

        return make_nn_model(
            hidden_layer_sizes=p["hidden_layer_sizes"],
            activation=p["activation"],
            alpha=p["alpha"],
            solver=p["solver"],
        )


class RFModelTuner:
    """
    Runs RandomizedSearchCV on one dataset to find the best RandomForestRegressor
    hyperparameters, then returns a fixed make_rf_model callable for use
    across all replications.

    Usage
    -----
    X_tune, y_tune = dgp.sample(n, rng)
    rf_model = RFModelTuner(cv=5, n_iter=20, verbose=True, snr=9).tune(X_tune, y_tune, rng)
    experiment = Experiment(..., fit_model=rf_model, fit_model_slug=rf_model.__name__)
    """
    _param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", 0.5, 1.0],
        "min_samples_leaf": [1, 2, 5],
    }

    def __init__(self, cv=5, n_iter=20, verbose=False, snr=None):
        self.cv = cv
        self.n_iter = n_iter
        self.verbose = verbose
        self.snr = snr

    def tune(self, X, y, rng: np.random.Generator,
             dgp_slug=None, n=None, cache_dir="cached_explanations"):
        """
        Fit CV on (X, y) and return a fixed make_rf_model callable.

        If dgp_slug and n are provided, best params are cached to disk so
        subsequent calls with the same setup skip the search entirely.
        """
        cache_path = None
        if dgp_slug is not None and n is not None:
            fname = f"tune_rf_{dgp_slug}_n{n}_cv{self.cv}_niter{self.n_iter}.json"
            cache_path = os.path.join(cache_dir, fname)

        if cache_path is not None and os.path.exists(cache_path):
            with open(cache_path) as f:
                p = json.load(f)
            if self.verbose:
                print(f"[rf_cv] loaded cached params: {p}")
        else:
            seed = int(rng.integers(0, 2**31 - 1))
            search = RandomizedSearchCV(
                RandomForestRegressor(random_state=seed),
                self._param_dist,
                n_iter=self.n_iter, cv=self.cv, scoring="r2",
                random_state=seed, n_jobs=-1,
            )
            search.fit(X, y)
            p = search.best_params_
            if self.verbose:
                print(f"[rf_cv] best params : {p}")
                print(f"[rf_cv] CV R²       : {search.best_score_:.4f}")
                if self.snr is not None:
                    print(f"[rf_cv] theoretical R² (SNR={self.snr}): {1 - 1 / (1 + self.snr):.4f}")
            if cache_path is not None:
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump({
                        **p,
                        "cv_r2": search.best_score_,
                        "max_r2": 1 - 1 / (1 + self.snr) if self.snr is not None else None,
                    }, f, indent=2)

        return make_rf_model(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            min_samples_split=p["min_samples_split"],
            max_features=p["max_features"],
            min_samples_leaf=p["min_samples_leaf"],
        )

