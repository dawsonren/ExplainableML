import itertools
import time
import os
import warnings

import numpy as np
from tqdm import tqdm

from ale import ALE
from shapley import SHAP
from plots import get_full_bounding_box

warnings.filterwarnings("ignore")

from experiments import DGP, Experiment, RunConfig, ExplainerConfig
from models import (
    NNModelTuner, RFModelTuner,
    sample_X_gaussian, sample_X_uniform,
    signal_basic, signal_basic_interaction,
    signal_nonlinear, signal_nonlinear_interaction,
    signal_tricky_valley_rho_9, signal_tricky_valley_rho_99,
    signal_basic_explanation, signal_nonlinear_explanation,
)

# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------

REPLICATIONS = 50

# (signal_fn, true_explanation_fn, true_explanation_y_fn)
SIGNALS = [
    (signal_basic,                signal_basic_explanation,    signal_basic_explanation),
    (signal_basic_interaction,    None,                        None),
    (signal_nonlinear,            signal_nonlinear_explanation, signal_nonlinear_explanation),
    (signal_nonlinear_interaction, None,                       None),
    (signal_tricky_valley_rho_9,  None,                        None),
    (signal_tricky_valley_rho_99, None,                        None),
]

SNRS = [4, 9, 19, 99]

# (n, K, L) — K and L scale with sample size
N_KL = [
    (100,  10, 10),
    (200,  20, 10),
    (500,  25, 20),
    (1000, 50, 20),
]

RHOS = [0.7, 0.9, 0.95]

# (slug_prefix, sampler_factory)
SAMPLERS = [
    ("uniform", sample_X_uniform),
    ("gaussian", sample_X_gaussian),
]

# (slug_prefix, tuner_factory(snr) -> tuner)
MODEL_TYPES = [
    ("nn", lambda snr: NNModelTuner(cv=5, n_iter=20, verbose=False, snr=snr)),
    ("rf", lambda snr: RFModelTuner(cv=5, n_iter=20, verbose=False, snr=snr)),
]

# ---------------------------------------------------------------------------
# Per-configuration runner
# ---------------------------------------------------------------------------

def run_one(signal, snr, n, K, L, rho, sampler_factory, tuner_factory):
    rng = np.random.default_rng(42)

    dgp = DGP(
        snr=snr,
        sample_X=sampler_factory(rho=rho, scale=1),
        signal=signal,
    )

    # Tune once; result is cached to disk by dgp_slug + n
    X_tune, y_tune = dgp.sample(n=n, rng=rng)
    tuner = tuner_factory(snr)
    model = tuner.tune(X_tune, y_tune, rng, dgp_slug=dgp.slug, n=n)

    experiment = Experiment(
        dgp=dgp,
        fit_model=model,
        dgp_slug=dgp.slug,
        fit_model_slug=model.__name__,
        replications=REPLICATIONS,
        n=n,
        save=True,
    )

    explainer_config = ExplainerConfig(K=K, L=L, centering="y", interpolate=True)
    run_config = RunConfig(experiment=experiment, explainer_config=explainer_config)

    # Fit all replications (cached via joblib)
    data_and_fitted_models = experiment.fit_models(rng)

    # explain_grid and var_grid
    explain_grid = experiment.dgp.sample_X(n=n, rng=np.random.default_rng(99))
    bounding_box = get_full_bounding_box([d[0] for d in data_and_fitted_models])
    x1_min, x1_max, x2_min, x2_max = bounding_box
    x1_lin = np.linspace(x1_min, x1_max, 50)
    x2_lin = np.linspace(x2_min, x2_max, 50)
    xx, yy = np.meshgrid(x1_lin, x2_lin)
    var_grid = np.column_stack([xx.ravel(), yy.ravel()])

    # ALE (cached internally by run_config)
    ale_results = run_config.run_ale(explain_grid)

    # Full (ALE + SHAP) cache
    _ale_cache_name = os.path.basename(run_config.cache_path(explain_grid))
    cache_path = os.path.join("cached_explanations", _ale_cache_name.replace("ale_", "full_", 1))

    if os.path.exists(cache_path):
        return  # already done

    ale_queried_points = ale_results.get("ale_queried_points")

    # Single model for query-point precomputation
    X0, y0 = experiment.dgp.sample(n, rng=np.random.default_rng(seed=42))
    model0 = experiment.fit_model(X0, y0, rng=np.random.default_rng(0))
    f0 = model0.predict

    if ale_queried_points is None:
        ale_obj = ALE(f0, X0, K=K, L=L, centering="y", verbose=False, interpolate=True)
        ale_obj.explain()
        ale_queried_points = ale_obj.get_query_points()

    shap_queried_points = []
    for i in range(explain_grid.shape[0]):
        shap = SHAP(f0, X0)
        shap.shim(explain_grid[i:i+1])
        shap_queried_points.append(shap.get_query_points())

    shap_queried_f = {i: [] for i in range(explain_grid.shape[0])}
    ale_queried_f  = []
    zz_f           = np.zeros((REPLICATIONS, var_grid.shape[0]))
    shap_exps    = []
    f_vals       = []
    shap_globals = []
    shap_times   = []

    for r in range(REPLICATIONS):
        X, _, fitted_model = data_and_fitted_models[r]
        f = fitted_model.predict
        total_explained_points = explain_grid.shape[0] + X.shape[0]

        shapley = SHAP(f, X)
        t0 = time.perf_counter()
        shap_explanations = shapley.shim(explain_grid)
        shap_time = time.perf_counter() - t0

        for i in range(explain_grid.shape[0]):
            shap_queried_f[i].append(f(shap_queried_points[i]))
        ale_queried_f.append(f(ale_queried_points))

        shap_exps.append(shap_explanations)
        f_vals.append(f(explain_grid))
        shap_globals.append(0)
        shap_times.append(shap_time / total_explained_points)
        zz_f[r, :] = f(var_grid)

    shap_exps    = np.array(shap_exps)
    f_vals       = np.array(f_vals)
    shap_globals = np.array(shap_globals)
    shap_times   = np.array(shap_times)
    average_variability_shap = np.array([
        np.array(shap_queried_f[i]).std(axis=0).mean()
        for i in range(explain_grid.shape[0])
    ])
    average_variability_ale = float(np.array(ale_queried_f).std(axis=0).mean())
    f_variability = zz_f.std(axis=0)

    os.makedirs("cached_explanations", exist_ok=True)
    np.savez(
        cache_path,
        shap_exps=shap_exps,
        f_vals=f_vals,
        shap_globals=shap_globals,
        shap_times=shap_times,
        f_variability=f_variability,
        ale_queried_points=ale_queried_points,
        shap_queried_points=np.array(shap_queried_points),
        average_variability_shap=average_variability_shap,
        average_variability_ale=np.array(average_variability_ale),
    )


# ---------------------------------------------------------------------------
# Main: cartesian product
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    combos = list(itertools.product(SIGNALS, SNRS, N_KL, RHOS, SAMPLERS, MODEL_TYPES))

    for (signal, _, _), snr, (n, K, L), rho, (_, sampler_factory), (_, tuner_factory) in tqdm(combos, desc="experiments"):
        run_one(
            signal=signal,
            snr=snr,
            n=n, K=K, L=L,
            rho=rho,
            sampler_factory=sampler_factory,
            tuner_factory=tuner_factory,
        )
