import hashlib
import importlib.util
import itertools
import time
import os
import sys
import warnings

import numpy as np
from tqdm import tqdm

from shapley import SHAP

warnings.filterwarnings("ignore")

from experiments import DGP, Experiment, RunConfig


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str):
    """Load a config module from a file path and return it."""
    spec = importlib.util.spec_from_file_location("config", path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


# ---------------------------------------------------------------------------
# SHAP cache helpers
# ---------------------------------------------------------------------------

def _shap_cache_path(experiment: "Experiment", explain_grid: np.ndarray,
                     cache_dir: str = "cached_explanations") -> str:
    grid_hash = hashlib.md5(np.ascontiguousarray(explain_grid).tobytes()).hexdigest()[:8]
    name = f"shap_{experiment.slug()}_eg{grid_hash}.npz"
    return os.path.join(cache_dir, name)


def run_shap(experiment: "Experiment", explain_grid: np.ndarray,
             replications: int, cache_dir: str = "cached_explanations") -> dict:
    """Compute SHAP explanations for all replications, or load from cache."""
    path = _shap_cache_path(experiment, explain_grid, cache_dir)
    if os.path.exists(path):
        data = np.load(path)
        return {k: data[k] for k in data.files}

    data_and_fitted_models = experiment.fit_models(np.random.default_rng(42))
    runs = data_and_fitted_models[:replications]

    shap_exps, f_vals, shap_times = [], [], []
    for X, _, fitted_model in tqdm(runs, desc="SHAP replications", position=1, leave=False):
        f = fitted_model.predict
        shapley = SHAP(f, X)
        t0 = time.perf_counter()
        exps = shapley.explain_local(explain_grid, method="exact_shap")
        shap_times.append((time.perf_counter() - t0) / explain_grid.shape[0])
        shap_exps.append(exps)
        f_vals.append(f(explain_grid))

    result = {
        "shap_exps": np.array(shap_exps),
        "f_vals": np.array(f_vals),
        "shap_times": np.array(shap_times),
    }
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(path, **result)
    return result


# ---------------------------------------------------------------------------
# Base experiment setup (model tuning + SHAP)
# ---------------------------------------------------------------------------

def run_base(signal, snr, n, rho, sampler_factory, tuner_factory,
             replications, explain_n, pbar=None, base_desc="",
             cache_dir="cached_explanations"):
    """Tune model and compute SHAP once for a given (signal, snr, n, rho, model) combo."""
    rng = np.random.default_rng(42)

    dgp = DGP(snr=snr, sample_X=sampler_factory(rho), signal=signal)

    X_tune, y_tune = dgp.sample(n=n, rng=rng)
    tuner = tuner_factory(snr)
    if pbar is not None:
        pbar.set_description(f"{base_desc} [tuning]")
    model = tuner.tune(X_tune, y_tune, rng, dgp_slug=dgp.slug, n=n)
    if pbar is not None:
        pbar.set_description(base_desc)

    experiment = Experiment(
        dgp=dgp,
        fit_model=model,
        dgp_slug=dgp.slug,
        fit_model_slug=model.__name__,
        replications=replications,
        n=n,
        save=True,
    )

    explain_grid = experiment.dgp.sample_X(n=explain_n, rng=np.random.default_rng(99))

    if pbar is not None:
        pbar.set_description(f"{base_desc} [SHAP]")
    shap_data = run_shap(experiment, explain_grid, replications, cache_dir)
    if pbar is not None:
        pbar.set_description(base_desc)

    return experiment, explain_grid, shap_data


# ---------------------------------------------------------------------------
# Per-ALE-config runner
# ---------------------------------------------------------------------------

def run_ale_config(experiment, ale_config, explain_grid, shap_data,
                   true_explanation_fn, signal, snr, rho, replications,
                   cache_dir="cached_explanations"):
    """Run one ALE config and write full_*.npz, reusing pre-computed SHAP."""
    run_config = RunConfig(experiment=experiment, explainer_config=ale_config)
    ale_results = run_config.run_ale(explain_grid, cache_dir=cache_dir)

    ale_cache_name = os.path.basename(run_config.cache_path(explain_grid, cache_dir))
    full_path = os.path.join(cache_dir, ale_cache_name.replace("ale_", "full_", 1))

    if os.path.exists(full_path):
        return

    shap_exps = shap_data["shap_exps"]   # (R, explain_n, d)
    ale_exps  = ale_results["ale_exps"]  # (R, explain_n, d)

    shap_variance = shap_exps.var(axis=0).mean(axis=0)   # (d,)
    ale_variance  = ale_exps.var(axis=0).mean(axis=0)    # (d,)

    if true_explanation_fn is not None:
        true_exp  = true_explanation_fn(explain_grid)
        shap_bias2 = ((shap_exps.mean(axis=0) - true_exp) ** 2).mean(axis=0)
        ale_bias2  = ((ale_exps.mean(axis=0)  - true_exp) ** 2).mean(axis=0)
    else:
        d = shap_exps.shape[2]
        shap_bias2 = np.full(d, np.nan)
        ale_bias2  = np.full(d, np.nan)

    ec = ale_config
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(
        full_path,
        shap_exps=shap_exps,
        f_vals=shap_data["f_vals"],
        shap_globals=np.zeros(replications),
        shap_times=shap_data["shap_times"],
        shap_variance=np.array(shap_variance),
        ale_variance=np.array(ale_variance),
        shap_bias2=np.array(shap_bias2),
        ale_bias2=np.array(ale_bias2),
        ale_exps=ale_exps,
        explain_grid=explain_grid,
        # experiment metadata
        meta_signal=np.array(signal.__name__),
        meta_snr=np.array(snr),
        meta_rho=np.array(rho),
        meta_n=np.array(experiment.n),
        meta_R=np.array(replications),
        meta_model=np.array(experiment.fit_model_slug),
        # ALE config metadata
        meta_K=np.array(ec.K),
        meta_L=np.array(ec.L),
        meta_centering=np.array(ec.centering),
        meta_levels_up=np.array(ec.levels_up),
        meta_variant=np.array(ec.variant),
        meta_n_bootstrap=np.array(ec.n_bootstrap),
        meta_tag=np.array(ec.get_tag()),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_experiments.py <config_file>")
        print("Example: python run_experiments.py configs/basic.py")
        sys.exit(1)

    cfg = load_config(sys.argv[1])

    # Base combos: no ALE_CONFIGS — model fitting and SHAP are independent of ALE config
    base_combos = list(itertools.product(
        cfg.SIGNALS, cfg.SNRS, cfg.NS, cfg.RHOS, cfg.SAMPLERS, cfg.MODEL_TYPES,
    ))

    pbar = tqdm(base_combos, desc="experiments", position=0, leave=True)
    for combo in pbar:
        (signal, static_explanation_fn, rho_explanation_factory), snr, n, rho, \
            (sampler_slug, sampler_factory), (model_slug, tuner_factory) = combo

        true_explanation_fn = static_explanation_fn
        if rho_explanation_factory is not None:
            true_explanation_fn = rho_explanation_factory(rho)

        base_desc = f"{signal.__name__} | snr={snr} | n={n} | rho={rho} | {model_slug}"
        pbar.set_description(base_desc)

        experiment, explain_grid, shap_data = run_base(
            signal=signal,
            snr=snr,
            n=n,
            rho=rho,
            sampler_factory=sampler_factory,
            tuner_factory=tuner_factory,
            replications=cfg.REPLICATIONS,
            explain_n=cfg.EXPLAIN_N,
            pbar=pbar,
            base_desc=base_desc,
        )

        for ale_config in cfg.ALE_CONFIGS:
            run_ale_config(
                experiment=experiment,
                ale_config=ale_config,
                explain_grid=explain_grid,
                shap_data=shap_data,
                true_explanation_fn=true_explanation_fn,
                signal=signal,
                snr=snr,
                rho=rho,
                replications=cfg.REPLICATIONS,
            )
