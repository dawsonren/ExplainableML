"""
Model factory functions and DGP sampling strategies for experiments.
Import these into notebooks instead of defining them inline.
"""

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

def sample_X_uniform(rho, scale):
    """
    Sample 2D features with controlled correlation via a shared uniform latent.

    rho controls correlation: Cov(X1, X2) / Var(X1) ≈ rho
    scale controls feature spread: features are approximately in [-scale, scale]
    """
    def sample(n, rng: np.random.Generator):
        sigma_x = scale * np.sqrt((1 - rho) / (3 * rho))
        x = rng.uniform(-scale, scale, size=n)
        x1 = x + rng.normal(size=n, scale=sigma_x)
        x2 = x + rng.normal(size=n, scale=sigma_x)
        return np.column_stack([x1, x2])
    sample.__name__ = f"sample_X_uniform_rho{rho:g}_scale{scale:g}"
    return sample


def sample_X_gaussian(rho, scale):
    """Sample 2D features from a bivariate Gaussian with correlation rho."""
    def sample(n, rng: np.random.Generator):
        cov = [[scale**2, rho * scale**2], [rho * scale**2, scale**2]]
        return rng.multivariate_normal(mean=[0, 0], cov=cov, size=n)
    sample.__name__ = f"sample_X_gaussian_rho{rho:g}_scale{scale:g}"
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

# ---------------------------------------------------------------------------
# True explanations (only for additive signals)
# ---------------------------------------------------------------------------

def signal_basic_explanation(X):
    return np.vstack([X[:, 0], X[:, 1] ** 2]).T

def signal_nonlinear_explanation(X):
    return np.vstack([X[:, 0], np.sin(4 * X[:, 1])]).T

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
        "hidden_layer_sizes": [(50,), (100,), (200,), (50, 50), (100, 50), (100, 100)],
        "activation": ["tanh", "relu"],
        "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        "solver": ["lbfgs", "adam"],
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
                    json.dump({**p, "hidden_layer_sizes": list(p["hidden_layer_sizes"])}, f, indent=2)

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
                    json.dump(p, f, indent=2)

        return make_rf_model(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            min_samples_split=p["min_samples_split"],
            max_features=p["max_features"],
            min_samples_leaf=p["min_samples_leaf"],
        )

