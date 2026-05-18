"""
Model factory functions and DGP sampling strategies for experiments.
Import these into notebooks instead of defining them inline.
"""

import hashlib
import json
import os

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


# ---------------------------------------------------------------------------
# DGP sampling strategies
# ---------------------------------------------------------------------------

def sample_X_uniform(corr, scale):
    """
    Sample d-dimensional uniform features via a Gaussian copula.

    `corr` is the *target* Pearson correlation matrix for the resulting
    uniform marginals. scale is [low, high]: each marginal is Uniform(low, high).

    Applying Φ to a Gaussian with correlation ρ_z produces uniforms with Pearson
    correlation (6/π)·arcsin(ρ_z/2) — slightly less than ρ_z. So we invert to
    pick the copula correlation ρ_z = 2·sin(π·ρ_target/6) that gives the
    requested ρ on the uniforms. Verified by sampling: target 0.9 → observed
    ≈0.900 with the inversion (vs ≈0.892 without).
    """
    corr = np.array(corr, dtype=float)
    d = corr.shape[0]
    low, high = float(scale[0]), float(scale[1])
    corr_hash = hashlib.md5(np.ascontiguousarray(corr).tobytes()).hexdigest()[:8]
    # Copula correlation that yields `corr` as the Pearson correlation of
    # the resulting uniforms. Diagonal of corr is 1, mapped to 1 exactly.
    corr_z = 2.0 * np.sin(np.pi * corr / 6.0)
    def sample(n, rng: np.random.Generator):
        z = rng.multivariate_normal(mean=np.zeros(d), cov=corr_z, size=n)
        u = norm.cdf(z)
        return low + (high - low) * u
    sample.__name__ = f"sample_X_uniform_d{d}_{corr_hash}_low{low:g}_high{high:g}"
    return sample


def sample_X_gaussian(cov, mean=None):
    """Sample d-dimensional features from a multivariate Gaussian with the given covariance matrix and optional mean."""
    cov = np.array(cov, dtype=float)
    d = cov.shape[0]
    mu = np.zeros(d) if mean is None else np.array(mean, dtype=float)
    cov_hash = hashlib.md5(np.ascontiguousarray(cov).tobytes()).hexdigest()[:8]
    mean_hash = hashlib.md5(np.ascontiguousarray(mu).tobytes()).hexdigest()[:8]
    def sample(n, rng: np.random.Generator):
        return rng.multivariate_normal(mean=mu, cov=cov, size=n)
    sample.__name__ = f"sample_X_gaussian_d{d}_{cov_hash}_mu{mean_hash}"
    return sample


def sample_X_gaussian_pairs(d, rho, n_pairs):
    """
    d-dim Gaussian with N(0,1) marginals, where `n_pairs` disjoint consecutive
    pairs (0,1), (2,3), ... have off-diagonal covariance `rho`; remaining features
    are independent. Requires d >= 2*n_pairs.
    """
    d, n_pairs = int(d), int(n_pairs)
    rho = float(rho)
    if d < 2 * n_pairs:
        raise ValueError(f"d={d} too small for {n_pairs} pairs")
    cov = np.eye(d)
    for k in range(n_pairs):
        i, j = 2 * k, 2 * k + 1
        cov[i, j] = cov[j, i] = rho
    def sample(n, rng: np.random.Generator):
        return rng.multivariate_normal(mean=np.zeros(d), cov=cov, size=n)
    sample.__name__ = f"sample_X_gauss_pairs_d{d}_p{n_pairs}_rho{rho:g}"
    return sample


def sample_X_gaussian_ar1(d, rho):
    """d-dim Gaussian with N(0,1) marginals and AR(1) covariance cov[i,j] = rho^|i-j|."""
    d = int(d)
    rho = float(rho)
    idx = np.arange(d)
    cov = rho ** np.abs(idx[:, None] - idx[None, :])
    def sample(n, rng: np.random.Generator):
        return rng.multivariate_normal(mean=np.zeros(d), cov=cov, size=n)
    sample.__name__ = f"sample_X_gauss_ar1_d{d}_rho{rho:g}"
    return sample


def sample_X_donut(r_min, r_max):
    """
    Sample 2D features uniformly from an annulus with inner radius r_min and outer radius r_max.

    Radius is drawn via inverse-CDF of the annular area measure so density is
    uniform over the ring: r = sqrt(U*(r_max^2 - r_min^2) + r_min^2).
    This distribution has a non-convex support, making it useful for stress-testing
    path-based explainability methods that can extrapolate through the hole.
    """
    r_min, r_max = float(r_min), float(r_max)
    def sample(n, rng: np.random.Generator):
        theta = rng.uniform(0, 2 * np.pi, n)
        r = np.sqrt(rng.uniform(r_min ** 2, r_max ** 2, n))
        return np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    sample.__name__ = f"sample_X_donut_rmin{r_min:g}_rmax{r_max:g}"
    return sample


def sample_X_banana(curvature, sigma_x, sigma_noise):
    """
    Sample 2D features from a banana-shaped distribution via a nonlinear Gaussian transform.

    x1 ~ N(0, sigma_x^2), x2 = curvature * x1^2 + eps, eps ~ N(0, sigma_noise^2).
    The curved manifold stresses path-based methods because a straight-line path
    between two points on the banana crosses through low-density regions.
    """
    curvature, sigma_x, sigma_noise = float(curvature), float(sigma_x), float(sigma_noise)
    def sample(n, rng: np.random.Generator):
        x1 = rng.normal(0, sigma_x, n)
        x2 = curvature * x1 ** 2 + rng.normal(0, sigma_noise, n)
        return np.column_stack([x1, x2])
    sample.__name__ = f"sample_X_banana_curv{curvature:g}_sx{sigma_x:g}_sn{sigma_noise:g}"
    return sample


def sample_X_mixture(components):
    """
    Sample from a mixture of distributions.

    components is a list of (weight, sample_fn) pairs where each sample_fn is a
    callable(n, rng) -> (n, d) array (e.g. the result of sample_X_gaussian).
    Weights are automatically normalised. The mixture creates disconnected or
    multi-modal support, stressing path-based methods that must traverse the gaps.
    """
    weights = np.array([w for w, _ in components], dtype=float)
    weights /= weights.sum()
    fns = [fn for _, fn in components]
    name_parts = "_".join(f"({w:g}_{fn.__name__})" for w, fn in zip(weights, fns))
    def sample(n, rng: np.random.Generator):
        counts = rng.multinomial(n, weights)
        parts = [fn(k, rng) for fn, k in zip(fns, counts) if k > 0]
        return rng.permutation(np.vstack(parts))
    sample.__name__ = f"sample_X_mixture_{name_parts}"
    return sample


# ---------------------------------------------------------------------------
# Signal functions
# ---------------------------------------------------------------------------

def signal_basic(X):
    return X[:, 0] + X[:, 1] ** 2

def signal_basic_interaction(X):
    return X[:, 0] + X[:, 1] + 2 * X[:, 0] * X[:, 1]

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

def signal_additive_quadratic(X):
    """f = sum_j X_j^2. Works for any d; ground truth per feature is x_j^2 (centered: x_j^2 - 1 under N(0,1))."""
    return (X ** 2).sum(axis=1)


def signal_hooker_2021(X):
    """From Hooker et al. 2021"""
    beta = np.array([1, 1, 1, 1, 1, 0, 0.5, 0.8, 1.2, 1.5])
    return X @ beta


# ---------------------------------------------------------------------------
# True explanations (only for additive signals and simple interaction ones)
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

def signal_additive_quadratic_explanation(X):
    """Interventional Shapley for sum_j X_j^2 with N(0,1) marginals: x_j^2 - 1 per feature, any correlation."""
    return X ** 2 - 1


def signal_hooker_2021_explanation(X):
    """Shapley-value explanation for signal_hooker_2021 under U[0, 1]."""
    beta = np.array([1, 1, 1, 1, 1, 0, 0.5, 0.8, 1.2, 1.5])
    return X * beta - 0.5 * beta

def signal_basic_interaction_explanation(X):
    """Shapley-value explanation for signal_basic_interaction under U[0, 1]."""
    return np.column_stack([X[:, 0] ** 2 + X[:, 0] * X[:, 1], X[:, 1] ** 2 + X[:, 0] * X[:, 1]])

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


class FixedNNTuner:
    """
    Drop-in for NNModelTuner that skips CV and returns a fixed make_nn_model
    with the given hyperparameters. Use to sweep architectures/regularization
    explicitly via MODEL_TYPES.
    """
    def __init__(self, hidden_layer_sizes, alpha, activation="tanh", solver="lbfgs", snr=None):
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.alpha = float(alpha)
        self.activation = activation
        self.solver = solver
        self.snr = snr

    def tune(self, X, y, rng: np.random.Generator,
             dgp_slug=None, n=None, cache_dir="cached_explanations"):
        return make_nn_model(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            alpha=self.alpha,
            solver=self.solver,
        )


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
        "n_estimators": [100, 500, 1000, 2000],
        "max_depth": [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16],
        "min_samples_split": [2, 5, 10, 20],
        "max_features": ["sqrt", "log2", 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
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

