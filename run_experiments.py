import importlib.util
import itertools
import os
import sys
import warnings

import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")

from experiments import (
    DGP, Experiment,
    ExplainerConfig, ShapConfig,
    load_results, save_results,
    compute_ale, compute_shap, compute_f_vals,
)


CACHE_ROOT = "cached_explanations"


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str):
    """Load a config module from a file path and return it."""
    spec = importlib.util.spec_from_file_location("config", path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def config_cache_dir(config_path: str) -> str:
    stem = os.path.splitext(os.path.basename(config_path))[0]
    return os.path.join(CACHE_ROOT, stem)


# ---------------------------------------------------------------------------
# Base setup (tune + build Experiment + pick explain grid)
# ---------------------------------------------------------------------------

def setup_experiment(signal, snr, n, rho, sampler_factory, tuner_factory,
                     replications, explain_n, cache_dir, pbar=None, base_desc=""):
    """Tune the model for this DGP and return (experiment, explain_grid)."""
    rng = np.random.default_rng(42)
    dgp = DGP(snr=snr, sample_X=sampler_factory(rho), signal=signal)

    X_tune, y_tune = dgp.sample(n=n, rng=rng)
    tuner = tuner_factory(snr)
    if pbar is not None:
        pbar.set_description(f"{base_desc} [tuning]")
    model = tuner.tune(X_tune, y_tune, rng, dgp_slug=dgp.slug, n=n, cache_dir=cache_dir)
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
    return experiment, explain_grid


# ---------------------------------------------------------------------------
# Per-experiment runner: load/init results pickle, compute missing tags, save.
# ---------------------------------------------------------------------------

def run_experiment(experiment: Experiment, explain_grid: np.ndarray,
                   ale_configs, shap_configs,
                   signal, snr, rho, cache_dir: str,
                   pbar=None, base_desc=""):
    results = load_results(experiment, cache_dir)

    # Stamp/refresh metadata on each run (cheap; keeps the pickle self-describing).
    results["experiment_meta"] = {
        "dgp_slug": experiment.dgp_slug,
        "fit_model_slug": experiment.fit_model_slug,
        "n": experiment.n,
        "replications": experiment.replications,
        "signal": signal.__name__,
        "snr": float(snr),
        "rho": float(rho),
        "model": experiment.fit_model_slug,
    }
    results["explain_grid"] = explain_grid

    dirty = False

    if results.get("f_vals") is None:
        if pbar is not None:
            pbar.set_description(f"{base_desc} [f_vals]")
        results["f_vals"] = compute_f_vals(experiment, explain_grid, cache_dir)
        dirty = True

    # ALE configs
    ale_store = results.setdefault("ale", {})
    for ec in ale_configs:
        tag = ec.get_tag()
        key = ec.cache_key()
        existing = ale_store.get(tag)
        if existing is not None and existing.get("cache_key") == key:
            continue
        if pbar is not None:
            pbar.set_description(f"{base_desc} [ALE {tag}]")
        ale_store[tag] = compute_ale(experiment, ec, explain_grid, cache_dir)
        dirty = True

    # SHAP configs
    shap_store = results.setdefault("shap", {})
    for sc in shap_configs:
        tag = sc.get_tag()
        key = sc.cache_key()
        existing = shap_store.get(tag)
        if existing is not None and existing.get("cache_key") == key:
            continue
        if pbar is not None:
            pbar.set_description(f"{base_desc} [SHAP {tag}]")
        shap_store[tag] = compute_shap(experiment, sc, explain_grid, cache_dir)
        dirty = True

    if pbar is not None:
        pbar.set_description(base_desc)

    if dirty:
        save_results(results, experiment, cache_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_experiments.py <config_file>")
        print("Example: python run_experiments.py configs/sparse_highdim.py")
        sys.exit(1)

    config_path = sys.argv[1]
    cfg = load_config(config_path)
    cache_dir = config_cache_dir(config_path)
    os.makedirs(cache_dir, exist_ok=True)

    shap_configs = getattr(cfg, "SHAP_CONFIGS", [ShapConfig()])

    base_combos = list(itertools.product(
        cfg.SIGNALS, cfg.SNRS, cfg.NS, cfg.RHOS, cfg.SAMPLERS, cfg.MODEL_TYPES,
    ))

    pbar = tqdm(base_combos, desc="experiments", position=0, leave=True)
    for combo in pbar:
        (signal, _static_expl_fn, _rho_expl_factory), snr, n, rho, \
            (sampler_slug, sampler_factory), (model_slug, tuner_factory) = combo

        base_desc = f"{signal.__name__} | snr={snr} | n={n} | rho={rho} | {model_slug}"
        pbar.set_description(base_desc)

        experiment, explain_grid = setup_experiment(
            signal=signal, snr=snr, n=n, rho=rho,
            sampler_factory=sampler_factory, tuner_factory=tuner_factory,
            replications=cfg.REPLICATIONS, explain_n=cfg.EXPLAIN_N,
            cache_dir=cache_dir, pbar=pbar, base_desc=base_desc,
        )

        run_experiment(
            experiment=experiment,
            explain_grid=explain_grid,
            ale_configs=cfg.ALE_CONFIGS,
            shap_configs=shap_configs,
            signal=signal, snr=snr, rho=rho,
            cache_dir=cache_dir,
            pbar=pbar, base_desc=base_desc,
        )
