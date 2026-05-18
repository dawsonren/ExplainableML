"""
Shared read-side helpers used by summarize_experiments, visualize_experiments,
and explore_experiments.

These functions were duplicated across the three modules; consolidating them
here keeps the analysis layer DRY without splitting the runner/dashboard.
"""

import inspect
import os

import joblib

import models
from ale import ALE, BootstrapALE


def true_explanation_fn(signal_name: str):
    """Return the *_explanation function for a signal name, or None.

    Rho-dependent explanations (e.g. signal_multiplicative) take two args; we
    can't evaluate them here without knowing rho, so they are skipped.
    """
    fn = getattr(models, f"{signal_name}_explanation", None)
    if fn is None:
        return None
    try:
        nparams = len(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return None
    return fn if nparams == 1 else None


def run_path_for(results: dict, cache_dir: str) -> str:
    """Locate the run_*.pkl companion to a results pickle."""
    meta = results.get("experiment_meta", {})
    name = f"run_{meta['dgp_slug']}_{meta['fit_model_slug']}_n{meta['n']}_R{meta['replications']}.pkl"
    return os.path.join(cache_dir, name)


def rebuild_ale_from_run(run_pkl_path: str, ec, rep_idx: int = 0):
    """
    Rebuild an ALE/BootstrapALE object on a specific replication from a run
    pickle, using the ExplainerConfig that produced the corresponding results.

    Returns the ALE instance with `explain(include=("total_<method>",))`
    already invoked.
    """
    data_and_fitted_models = joblib.load(run_pkl_path)
    X, _y, model = data_and_fitted_models[rep_idx]

    method = getattr(ec, "method", "connected")
    random_seed = getattr(ec, "random_seed", 42)
    include_key = f"total_{method}"

    if ec.variant == "bootstrap":
        ale_obj = BootstrapALE(
            model.predict, X,
            replications=ec.n_bootstrap,
            K=ec.K, L=ec.L, centering=ec.centering,
            interpolate=ec.interpolate,
            random_seed=random_seed,
            verbose=False,
        )
        ale_obj.explain(include=(include_key,))
        return ale_obj.ale_replications[0]

    ale = ALE(
        model.predict, X,
        K=ec.K, L=ec.L, centering=ec.centering,
        interpolate=ec.interpolate,
        random_seed=random_seed,
        verbose=False,
    )
    ale.explain(include=(include_key,))
    return ale
