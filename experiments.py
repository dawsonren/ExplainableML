from dataclasses import dataclass, field
from typing import List, Optional
import hashlib
import os
import joblib

import numpy as np

from ale import ALE, BootstrapALE

SAVE_DIR = "cached_explanations"


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

    def fit_models(self, rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray, object]]:
        """Fit models for each replication. Returns list of (X, y, fitted_model)."""
        if self.save:
            path = f"{SAVE_DIR}/run_{self.slug()}.pkl"
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
            self.save_run(results)
        return results

    def save_run(self, results: list[tuple[np.ndarray, np.ndarray, object]]) -> None:
        """Save all replications to a single file."""
        os.makedirs(SAVE_DIR, exist_ok=True)
        joblib.dump(results, f"{SAVE_DIR}/run_{self.slug()}.pkl")


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
    knn_smooth: Optional[int] = None
    levels_up: int = 0
    edges: Optional[dict] = None  # dict mapping 1-based feature index -> edges array
    variant: str = "standard"     # "standard" or "bootstrap"
    n_bootstrap: int = 50         # bootstrap replications (variant="bootstrap" only)
    tag: Optional[str] = None     # human-readable filename label; auto-generated if None

    @property
    def auto_tag(self) -> str:
        base = f"K{self.K}_L{self.L}_{self.centering}"
        if self.levels_up != 0:
            base += f"_lu{self.levels_up}"
        if self.variant == "bootstrap":
            base = f"bale_{base}_nb{self.n_bootstrap}"
        return base

    def get_tag(self) -> str:
        return self.tag if self.tag is not None else self.auto_tag


@dataclass
class RunConfig:
    """
    Ties together an Experiment, an ExplainerConfig, and grid settings for
    a single run of the ALE bias/variance experiment.

    Usage
    -----
    config = RunConfig(experiment=exp, explainer_config=ec, replications=25)
    results = config.run_ale()
    """
    experiment: Experiment
    explainer_config: ExplainerConfig
    seed: int = 42

    def cache_key(self, grid: "np.ndarray | None" = None) -> str:
        """Deterministic MD5 hash of all fields that affect output arrays."""
        ec = self.explainer_config
        parts = {
            "dgp": self.experiment.dgp_slug,
            "fit": self.experiment.fit_model_slug,
            "n": self.experiment.n,
            "R": self.experiment.replications,
            "K": ec.K,
            "L": ec.L,
            "centering": ec.centering,
            "interpolate": ec.interpolate,
            "knn_smooth": ec.knn_smooth,
            "levels_up": ec.levels_up,
            "edges": str(ec.edges),
            "variant": ec.variant,
            "n_bootstrap": ec.n_bootstrap,
            "seed": self.seed,
        }
        if grid is not None:
            parts["grid"] = hashlib.md5(np.ascontiguousarray(grid).tobytes()).hexdigest()[:8]
        key_str = "__".join(f"{k}={v}" for k, v in parts.items())
        return hashlib.md5(key_str.encode()).hexdigest()[:12]

    def cache_path(self, grid: "np.ndarray | None" = None, cache_dir: str = SAVE_DIR) -> str:
        ec = self.explainer_config
        name = (
            f"ale_{self.experiment.dgp_slug}_{self.experiment.fit_model_slug}"
            f"_n{self.experiment.n}_R{self.experiment.replications}"
            f"_{ec.get_tag()}"
            f"_{self.cache_key(grid)}.npz"
        )
        return os.path.join(cache_dir, name)

    def run_ale(self, grid, cache_dir: str = SAVE_DIR) -> dict:
        """
        Run the ALE bias/variance experiment and return a dict of result arrays.

        Loads from cache if available; otherwise runs all replications and saves.

        Returned keys
        -------------
        ale_exps : (replications, grid_resolution, d)  local explanations per replication
        ale_exps_mean : (grid_resolution, d)            mean across replications
        ale_exps_std  : (grid_resolution, d)            std across replications
        ale_times : (replications,)                     time per explained point
        ale_tree_times : (replications,)                tree construction time
        grid : (grid_resolution, d)                     the diagonal grid used
        edges : dict mapping feature index (0-based) -> edges array from last replication
        """
        path = self.cache_path(grid, cache_dir)
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            return {k: data[k] for k in data.files}

        os.makedirs(cache_dir, exist_ok=True)
        ec = self.explainer_config

        rng = np.random.default_rng(self.seed)
        data_and_models = self.experiment.fit_models(rng)

        # Use only the requested number of replications (may be < experiment.replications)
        runs = data_and_models[: self.experiment.replications]

        ale_exps = []
        ale_times = []
        ale_tree_times = []
        last_edges = {}

        import time
        from tqdm import tqdm

        for r, (X, y, model) in enumerate(tqdm(runs, desc="ALE replications", position=1, leave=False)):
            f = model.predict

            t0 = time.perf_counter()
            if ec.variant == "bootstrap":
                ale = BootstrapALE(
                    f, X,
                    replications=ec.n_bootstrap,
                    K=ec.K,
                    L=ec.L,
                    centering=ec.centering,
                    interpolate=ec.interpolate,
                    knn_smooth=ec.knn_smooth,
                    verbose=False,
                )
                ale.explain(include=("total_connected",))
                last_edges = ale.ale_replications[0].edges
            else:
                ale = ALE(
                    f, X,
                    K=ec.K,
                    L=ec.L,
                    centering=ec.centering,
                    interpolate=ec.interpolate,
                    knn_smooth=ec.knn_smooth,
                    edges=ec.edges,
                    verbose=False,
                )
                ale.explain(include=("total_connected",))
                last_edges = ale.edges
            tree_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            exps = ale.explain_local(grid, levels_up=ec.levels_up)
            explain_time = time.perf_counter() - t0

            ale_exps.append(exps)
            ale_times.append((tree_time + explain_time) / grid.shape[0])
            ale_tree_times.append(tree_time)

        ale_exps = np.array(ale_exps)

        results = {
            "ale_exps": ale_exps,
            "ale_exps_mean": ale_exps.mean(axis=0),
            "ale_exps_std": ale_exps.std(axis=0),
            "ale_times": np.array(ale_times),
            "ale_tree_times": np.array(ale_tree_times),
            "grid": grid,
        }
        np.savez(
            path, **results,
            edges=np.array([last_edges], dtype=object),
            meta_K=np.array(ec.K),
            meta_L=np.array(ec.L),
            meta_centering=np.array(ec.centering),
            meta_levels_up=np.array(ec.levels_up),
            meta_variant=np.array(ec.variant),
            meta_n_bootstrap=np.array(ec.n_bootstrap),
            meta_tag=np.array(ec.get_tag()),
        )
        return results
