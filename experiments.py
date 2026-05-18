from dataclasses import dataclass, field
from typing import List, Optional
import hashlib
import os
import time
import joblib

import numpy as np

from ale import ALE, BootstrapALE
from shapley import SHAP


@dataclass
class DGP:
    """Data Generating Process (DGP) for the experiments. Handles adding Gaussian noise."""
    snr: float
    sample_X: callable  # (n: int, rng: np.random.Generator) -> np.ndarray
    signal: callable    # (X: np.ndarray) -> np.ndarray
    sigma_eps: float = field(init=False)
    d: int = field(init=False)

    def __post_init__(self):
        X_probe = self.sample_X(n=10000, rng=np.random.default_rng())
        var_fX = float(np.var(self.signal(X_probe)))
        self.sigma_eps = float(np.sqrt(var_fX / self.snr))
        self.d = X_probe.shape[1]

    @property
    def slug(self) -> str:
        """Unique string describing the DGP: sampler, SNR, and signal."""
        return f"{self.sample_X.__name__}_snr{self.snr:g}_{self.signal.__name__}"

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Sample from the DGP."""
        X = self.sample_X(n=n, rng=rng)
        y = self.signal(X) + rng.normal(scale=self.sigma_eps, size=n)
        return X, y


@dataclass(frozen=True)
class Experiment:
    """
    Container for experiment settings + data generating process.
    """
    dgp: DGP
    fit_model: callable  # (X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> model
    dgp_slug: str        # exhaustively describes all DGP details
    fit_model_slug: str  # exhaustively describes the model algorithm
    n: int               # sample size (optimal hyperparameters may depend on n)
    replications: int = 100
    save: bool = True

    def slug(self) -> str:
        return f"{self.dgp_slug}_{self.fit_model_slug}_n{self.n}_R{self.replications}"

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Sample from the DGP."""
        return self.dgp.sample(n=self.n, rng=rng)

    def fit_models(self, rng: np.random.Generator, cache_dir: str) -> list[tuple[np.ndarray, np.ndarray, object]]:
        """Fit models for each replication. Returns list of (X, y, fitted_model)."""
        if self.save:
            path = os.path.join(cache_dir, f"run_{self.slug()}.pkl")
            try:
                return joblib.load(path)
            except FileNotFoundError:
                pass
        results = []
        for _ in range(self.replications):
            X, y = self.dgp.sample(n=self.n, rng=rng)
            model = self.fit_model(X, y, rng=rng)
            results.append((X, y, model))

        if self.save:
            self.save_run(results, cache_dir)
        return results

    def save_run(self, results: list[tuple[np.ndarray, np.ndarray, object]], cache_dir: str) -> None:
        """Save all replications to a single file."""
        os.makedirs(cache_dir, exist_ok=True)
        joblib.dump(results, os.path.join(cache_dir, f"run_{self.slug()}.pkl"))


def _md5_of(parts: dict) -> str:
    key_str = "__".join(f"{k}={v}" for k, v in sorted(parts.items()))
    return hashlib.md5(key_str.encode()).hexdigest()[:12]


@dataclass
class ExplainerConfig:
    """
    All hyperparameters that control how ALE explanations are produced.
    Changing any field (except tag) produces a different explanation, so all
    fields except tag participate in the cache key.
    """
    K: int = 10
    L: int = 10
    centering: str = "y"
    interpolate: bool = True
    edges: Optional[dict] = None  # dict mapping 0-based feature index -> edges array
    variant: str = "standard"     # "standard" or "bootstrap"
    n_bootstrap: int = 50         # bootstrap replications (variant="bootstrap" only)
    local_method: str = "path_rep"  # "path_rep", "path_integral", or "multi_path_interpolate"
    method: str = "connected"     # path-generation method: "connected", "quantile", "random"
    random_seed: int = 42         # seed for method="random" path partitioning
    background_size: Optional[int] = None  # path_integral / multi_path_interpolate only: size of background subsample (None = full X)
    background_seed: Optional[int] = None  # path_integral / multi_path_interpolate only: RNG seed for the subsample (None = use random_seed)
    boundary_interp: bool = False  # path_integral only (ignored by multi_path_interpolate): replace boundary f-evals with linear-interp of routed deltas
    tag: Optional[str] = None     # human-readable filename label; auto-generated if None

    @property
    def auto_tag(self) -> str:
        base = f"K{self.K}_L{self.L}_{self.centering}"
        if self.method == "quantile":
            base += "_quant"
        elif self.method == "random":
            base += f"_rand_s{self.random_seed}"
        base += f"_{self.local_method}"
        if self.local_method in ("path_integral", "multi_path_interpolate") and self.background_size is not None:
            base += f"_bg{self.background_size}"
        if self.local_method == "path_integral" and self.boundary_interp:
            base += "_binterp"
        if self.variant == "bootstrap":
            base = f"bale_{base}_nb{self.n_bootstrap}"
        return base

    def get_tag(self) -> str:
        return self.tag if self.tag is not None else self.auto_tag

    def cache_key(self) -> str:
        return _md5_of({
            "K": self.K,
            "L": self.L,
            "centering": self.centering,
            "interpolate": self.interpolate,
            "edges": str(self.edges),
            "variant": self.variant,
            "n_bootstrap": self.n_bootstrap,
            "local_method": self.local_method,
            "method": self.method,
            "random_seed": self.random_seed,
            "background_size": self.background_size,
            "background_seed": self.background_seed,
            "boundary_interp": self.boundary_interp,
        })


@dataclass
class ShapConfig:
    """
    All hyperparameters that control how SHAP explanations are produced.
    """
    method: str = "exact_shap"            # permutation_shap / kernel_shap / tree_shap / exact_shap / linear_shap
    kwargs: dict = field(default_factory=dict)
    sample_method: Optional[str] = None   # None / "kmeans" / "sample"
    sample_size: int = 1000
    random_state: int = 42
    tag: Optional[str] = None

    @property
    def auto_tag(self) -> str:
        short = {
            "exact_shap": "exact",
            "permutation_shap": "perm",
            "kernel_shap": "kernel",
            "tree_shap": "tree",
            "linear_shap": "linear",
        }.get(self.method, self.method)
        base = short
        if self.sample_method is not None:
            base += f"_{self.sample_method}{self.sample_size}"
        return base

    def get_tag(self) -> str:
        return self.tag if self.tag is not None else self.auto_tag

    def cache_key(self) -> str:
        return _md5_of({
            "method": self.method,
            "kwargs": str(sorted((self.kwargs or {}).items())),
            "sample_method": self.sample_method,
            "sample_size": self.sample_size,
            "random_state": self.random_state,
        })


# ---------------------------------------------------------------------------
# Merged results pickle
# ---------------------------------------------------------------------------

def results_path(experiment: "Experiment", cache_dir: str) -> str:
    return os.path.join(cache_dir, f"results_{experiment.slug()}.pkl")


def load_results(experiment: "Experiment", cache_dir: str) -> dict:
    """Load the merged results pickle for an experiment, or an empty shell if missing."""
    path = results_path(experiment, cache_dir)
    if os.path.exists(path):
        return joblib.load(path)
    return {
        "experiment_meta": {
            "dgp_slug": experiment.dgp_slug,
            "fit_model_slug": experiment.fit_model_slug,
            "n": experiment.n,
            "replications": experiment.replications,
        },
        "explain_grid": None,
        "f_vals": None,
        "f_means": None,
        "ale": {},
        "shap": {},
    }


def save_results(results: dict, experiment: "Experiment", cache_dir: str) -> None:
    """Atomically write the merged results pickle."""
    os.makedirs(cache_dir, exist_ok=True)
    path = results_path(experiment, cache_dir)
    tmp = path + ".tmp"
    joblib.dump(results, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Replication runners
# ---------------------------------------------------------------------------

def compute_ale(experiment: "Experiment", ec: ExplainerConfig, explain_grid: np.ndarray,
                cache_dir: str, seed: int = 42) -> dict:
    """Run ALE over all replications and return a sub-dict for the results pickle."""
    from tqdm import tqdm

    rng = np.random.default_rng(seed)
    runs = experiment.fit_models(rng, cache_dir)[: experiment.replications]

    include_key = f"total_{ec.method}"

    exps, times, tree_times = [], [], []
    for X, _, model in tqdm(runs, desc=f"ALE[{ec.get_tag()}]", position=1, leave=False):
        f = model.predict

        t0 = time.perf_counter()
        if ec.variant == "bootstrap":
            ale = BootstrapALE(
                f, X,
                replications=ec.n_bootstrap,
                K=ec.K, L=ec.L,
                centering=ec.centering,
                interpolate=ec.interpolate,
                random_seed=ec.random_seed,
                verbose=False,
            )
            ale.explain(include=(include_key,))
        else:
            ale = ALE(
                f, X,
                K=ec.K, L=ec.L,
                centering=ec.centering,
                interpolate=ec.interpolate,
                edges=ec.edges,
                random_seed=ec.random_seed,
                verbose=False,
            )
            ale.explain(include=(include_key,))
        tree_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        if ec.method == "random":
            # Local explanations are not supported for random paths; fill NaN so
            # downstream bias/variance code still has a tensor of the right shape.
            exp_r = np.full((explain_grid.shape[0], X.shape[1]), np.nan)
        else:
            exp_r = ale.explain_local(
                explain_grid,
                local_method=ec.local_method,
                background_size=ec.background_size,
                background_seed=ec.background_seed,
                boundary_interp=ec.boundary_interp,
            )
        explain_time = time.perf_counter() - t0

        exps.append(exp_r)
        times.append((tree_time + explain_time) / explain_grid.shape[0])
        tree_times.append(tree_time)

    return {
        "exps": np.asarray(exps),
        "times": np.asarray(times),
        "tree_times": np.asarray(tree_times),
        "config": ec,
        "cache_key": ec.cache_key(),
    }


def compute_shap(experiment: "Experiment", sc: ShapConfig, explain_grid: np.ndarray,
                 cache_dir: str, seed: int = 42) -> dict:
    """Run SHAP over all replications and return a sub-dict for the results pickle."""
    from tqdm import tqdm

    rng = np.random.default_rng(seed)
    runs = experiment.fit_models(rng, cache_dir)[: experiment.replications]

    exps, times = [], []
    for X, _, model in tqdm(runs, desc=f"SHAP[{sc.get_tag()}]", position=1, leave=False):
        # TreeExplainer and LinearExplainer need the sklearn model object,
        # not the predict function.
        if sc.method in ("tree_shap", "linear_shap"):
            f = model
        else:
            f = model.predict
        shapley = SHAP(f, X, verbose=False)
        t0 = time.perf_counter()
        exp_r = shapley.explain_local(
            explain_grid,
            method=sc.method,
            kwargs=sc.kwargs,
            sample_method=sc.sample_method,
            sample_size=sc.sample_size,
            random_state=sc.random_state,
        )
        elapsed = time.perf_counter() - t0

        exps.append(np.asarray(exp_r))
        times.append(elapsed / explain_grid.shape[0])

    return {
        "exps": np.asarray(exps),
        "times": np.asarray(times),
        "config": sc,
        "cache_key": sc.cache_key(),
    }


def compute_f_vals(experiment: "Experiment", explain_grid: np.ndarray,
                   cache_dir: str, seed: int = 42) -> np.ndarray:
    """Cheap pass: model predictions on the explain grid, (R, explain_n)."""
    rng = np.random.default_rng(seed)
    runs = experiment.fit_models(rng, cache_dir)[: experiment.replications]
    return np.asarray([model.predict(explain_grid) for _, _, model in runs])


def compute_f_means(experiment: "Experiment", cache_dir: str,
                    seed: int = 42) -> np.ndarray:
    """E[f(X)] per replication, taken over each replication's training X. Shape (R,)."""
    rng = np.random.default_rng(seed)
    runs = experiment.fit_models(rng, cache_dir)[: experiment.replications]
    return np.asarray([float(model.predict(X).mean()) for X, _, model in runs])


def compute_r2(experiment: "Experiment", cache_dir: str,
               n_test: int = 5000, seed: int = 12345) -> dict:
    """
    Per-replication fit diagnostics, each shape (R,):
      r2_train        : R² of model on its training (X, y_noisy)
      r2_test_noisy   : R² on a fresh (X', y'_noisy) draw — ceiling is 1 - 1/(1+snr)
      r2_test_signal  : R² on a fresh X' against the noiseless signal f(X') — ceiling is 1
      gap             : r2_train - r2_test_noisy
    """
    from sklearn.metrics import r2_score
    runs = experiment.fit_models(np.random.default_rng(seed), cache_dir)[: experiment.replications]
    rng = np.random.default_rng(seed + 1)
    r2_train, r2_test_noisy, r2_test_signal = [], [], []
    for X_tr, y_tr, model in runs:
        r2_train.append(float(r2_score(y_tr, model.predict(X_tr))))
        X_te, y_te = experiment.dgp.sample(n=n_test, rng=rng)
        y_te_signal = experiment.dgp.signal(X_te)
        preds = model.predict(X_te)
        r2_test_noisy.append(float(r2_score(y_te, preds)))
        r2_test_signal.append(float(r2_score(y_te_signal, preds)))
    r2_train = np.asarray(r2_train)
    r2_test_noisy = np.asarray(r2_test_noisy)
    r2_test_signal = np.asarray(r2_test_signal)
    return {
        "r2_train": r2_train,
        "r2_test_noisy": r2_test_noisy,
        "r2_test_signal": r2_test_signal,
        "gap": r2_train - r2_test_noisy,
    }


# ---------------------------------------------------------------------------
# Bias/variance (computed on demand from stored explanations)
# ---------------------------------------------------------------------------

def compute_bias_variance(exps: np.ndarray, true_explanation: Optional[np.ndarray]) -> dict:
    """
    Given explanations of shape (R, explain_n, d) and optional ground-truth
    explanations of shape (explain_n, d), return mean bias² and stddev per dim,
    averaged over the explain grid.
    """
    stddev = exps.std(axis=0).mean(axis=0)  # (d,)
    if true_explanation is None:
        d = exps.shape[-1]
        bias2 = np.full(d, np.nan)
    else:
        bias2 = ((exps.mean(axis=0) - true_explanation) ** 2).mean(axis=0)
    return {"bias2": np.asarray(bias2), "stddev": np.asarray(stddev)}


def compute_additivity(exps: np.ndarray, f_vals: Optional[np.ndarray],
                       f_means: Optional[np.ndarray]) -> dict:
    """
    Test the Shapley efficiency property: sum_d phi_d(x) == f(x) - E[f(X)].

    `exps`    : (R, n_grid, d) attributions.
    `f_vals`  : (R, n_grid) model predictions on the explain grid.
    `f_means` : (R,) E[f(X)] per replication (mean over training X).

    Returns RMSE of the residual over (R, n_grid), plus a normalized
    version (RMSE / std of the target f(x) - E[f(X)]) and the mean abs
    residual. NaN if f_vals or f_means is missing.
    """
    if f_vals is None or f_means is None:
        return {"rmse": np.nan, "rmse_norm": np.nan, "mean_abs": np.nan}
    target = f_vals - f_means[:, None]            # (R, n_grid)
    residual = exps.sum(axis=-1) - target         # (R, n_grid)
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    target_std = float(target.std())
    rmse_norm = rmse / target_std if target_std > 0 else np.nan
    mean_abs = float(np.mean(np.abs(residual)))
    return {"rmse": rmse, "rmse_norm": rmse_norm, "mean_abs": mean_abs}
