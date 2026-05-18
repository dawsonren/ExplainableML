"""
Recreating ALE VIM Experiments.

- Example 4. Theoretical Linear Model with Correlation
- Example 5. Uniform DGP with Squared Term and Correlation on NN
- Example 6. Gaussian Copula DGP with Dummy and Correlation on NN, GBT, and RF
- Example 7. Real-world Bike Sharing Data on NN

Refactored from reimplement_ale_vim.ipynb. The old `replicate_ale_vim` /
`replicate_ale_vim_training` helpers in utils.py have been deleted, so they're
re-derived here against the current ALE API (which uses K=... rather than
bins=...).
"""

import argparse

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm

from ale.ale import ALE
from utils import bin_selection

pd.options.mode.copy_on_write = True


# ---------------------------------------------------------------------------
# Replication helpers (re-derived from old utils.py)
# ---------------------------------------------------------------------------

def _summarize(vim, label):
    for j in range(vim.shape[1]):
        col = vim[:, j]
        ci = np.percentile(col, [2.5, 97.5])
        print(f"VIM {j + 1} {label} - Mean: {col.mean():.4f}  "
              f"CI: [{ci[0]:.4f}, {ci[1]:.4f}]  SD: {col.std():.4f}")


def _check_values(vim, expected, label, *, k_se=4.0, abs_tol=0.02, rel_tol=0.05,
                  strict=True):
    """Compare per-variable means against expected reference values.

    For replicated runs (vim.shape[0] > 1), the tolerance is
        max(abs_tol, rel_tol * |expected|, k_se * SE)
    where SE = SD / sqrt(reps) is the Monte Carlo standard error of the mean.

    For single-run results, the SE term collapses to 0 and we fall back on
    abs_tol / rel_tol only; pass strict=False to downgrade failures to warnings
    (used for examples whose underlying fitted model can vary run-to-run).
    """
    reps, nvars = vim.shape
    expected = np.asarray(expected, dtype=float)
    assert expected.shape == (nvars,), \
        f"expected length {nvars}, got {expected.shape}"
    means = vim.mean(axis=0)
    se = vim.std(axis=0, ddof=1) / np.sqrt(reps) if reps > 1 else np.zeros(nvars)
    all_ok = True
    for j in range(nvars):
        tol = max(abs_tol, rel_tol * abs(expected[j]), k_se * se[j])
        diff = abs(means[j] - expected[j])
        ok = diff <= tol
        all_ok &= ok
        tag = "OK  " if ok else "FAIL"
        print(f"  [{tag}] {label} x{j + 1}: mean={means[j]:.4f} "
              f"expected={expected[j]:.4f} diff={diff:.4f} tol={tol:.4f}")
    if not all_ok and strict:
        raise AssertionError(f"{label} numerical check failed")
    return all_ok 


def _run_one(f, X, K, categorical=None, check_invariant=False):
    ale = ALE(f, X, K=K, categorical=categorical, interpolate=False, verbose=False)
    explanation = ale.explain()
    if check_invariant:
        ale.check_main_total_invariant(explanation, verbose=True)
    return (
        explanation.loc["main"].values,
        explanation.loc["total_connected"].values,
        explanation.loc["total_quantile"].values,
    )


def _invariant_summary(vim_mains, vim_connected, vim_quantile, tol=1e-6):
    """Summarize how often `main <= total_*` is violated across replications."""
    reps, nvars = vim_mains.shape
    excess_conn = np.maximum(vim_mains - vim_connected, 0)
    excess_quant = np.maximum(vim_mains - vim_quantile, 0)
    print("Invariant main <= total (per feature, across replications):")
    for j in range(nvars):
        vc = (excess_conn[:, j] > tol).sum()
        vq = (excess_quant[:, j] > tol).sum()
        print(
            f"  x{j + 1}: connected violated {vc}/{reps} "
            f"(max excess={excess_conn[:, j].max():.4f}), "
            f"quantile violated {vq}/{reps} "
            f"(max excess={excess_quant[:, j].max():.4f})"
        )


def replicate_ale_vim(dgp, f, n, K, replications=100, categorical=None,
                      check_invariant=False):
    """Replicate ALE variable importance for a fixed signal `f`."""
    nvars = dgp(1).shape[1]
    vim_mains = np.zeros((replications, nvars))
    vim_connected = np.zeros((replications, nvars))
    vim_quantile = np.zeros((replications, nvars))
    for i in tqdm(range(replications), desc="Replicating ALE VIMs"):
        X = dgp(n)
        m, c, q = _run_one(f, X, K, categorical=categorical,
                           check_invariant=check_invariant)
        vim_mains[i] = m
        vim_connected[i] = c
        vim_quantile[i] = q
    _summarize(vim_mains, "Main")
    _summarize(vim_connected, "Connected")
    _summarize(vim_quantile, "Quantile")
    _invariant_summary(vim_mains, vim_connected, vim_quantile)
    return vim_mains, vim_connected, vim_quantile


def replicate_ale_vim_training(dgp, f_factory, n, K, replications=100,
                               categorical=None, check_invariant=False):
    """Replicate ALE variable importance, refitting the model each draw."""
    nvars = dgp(1)[0].shape[1]
    vim_mains = np.zeros((replications, nvars))
    vim_connected = np.zeros((replications, nvars))
    vim_quantile = np.zeros((replications, nvars))
    for i in tqdm(range(replications), desc="Replicating ALE VIMs"):
        X, y = dgp(n)
        f = f_factory(X, y)
        m, c, q = _run_one(f, X, K, categorical=categorical,
                           check_invariant=check_invariant)
        vim_mains[i] = m
        vim_connected[i] = c
        vim_quantile[i] = q
    _summarize(vim_mains, "Main")
    _summarize(vim_connected, "Connected")
    _summarize(vim_quantile, "Quantile")
    _invariant_summary(vim_mains, vim_connected, vim_quantile)
    return vim_mains, vim_connected, vim_quantile


# ---------------------------------------------------------------------------
# Example 4: theoretical linear model with correlation
# ---------------------------------------------------------------------------

def example_4_dgp(n, rho):
    mean = [0, 0, 0]
    cov = [[1, rho, 0], [rho, 1, 0], [0, 0, 1]]
    return np.random.multivariate_normal(mean, cov, n)


def example_4_f(X, betas):
    return betas[0] * X[:, 0] + betas[1] * X[:, 1] + betas[2] * X[:, 2]


def run_example_4(replications=5):
    n = 1000
    K = 30
    dgp = lambda n: example_4_dgp(n, rho=0.5)
    f = lambda X: example_4_f(X, betas=[3, 2, 1])
    return replicate_ale_vim(dgp, f, n=n, K=K, replications=replications)


# ---------------------------------------------------------------------------
# Example 5: uniform DGP with squared term and correlation on NN
# ---------------------------------------------------------------------------

def example_5_dgp(n):
    u = np.random.uniform(0, 1, n)
    x1 = u + np.random.normal(0, 0.05, n)
    x2 = u + np.random.normal(0, 0.05, n)
    X = np.column_stack((x1, x2))
    y = x1 + x2 ** 2 + np.random.normal(0, 0.1, n)
    return X, y


def get_mlp_hyperparams(X, y):
    param_grid = {
        "hidden_layer_sizes": [(x,) for x in np.arange(50, 101, 5)],
        "activation": ["relu"],
        "solver": ["lbfgs"],
        "alpha": 0.00001 * np.arange(1, 11),
        "learning_rate": ["constant"],
    }
    mlp = MLPRegressor(max_iter=1000, random_state=42)
    grid_search = GridSearchCV(mlp, param_grid, cv=3,
                               scoring="neg_mean_squared_error", n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_params_


def example_5_f_factory(X, y):
    mlp = MLPRegressor(
        hidden_layer_sizes=(80,),
        activation="relu",
        solver="lbfgs",
        alpha=0.00001,
        learning_rate="constant",
        max_iter=1000,
        random_state=42,
    )
    mlp.fit(X, y)
    return lambda X: mlp.predict(X)


def run_example_5(replications=100, tune=False):
    n = 200
    K = bin_selection(n)
    if tune:
        X_tune, y_tune = example_5_dgp(n)
        print("Best hyperparameters:", get_mlp_hyperparams(X_tune, y_tune))
    return replicate_ale_vim_training(
        example_5_dgp, example_5_f_factory, n=n, K=K, replications=replications
    )


# ---------------------------------------------------------------------------
# Example 6: Gaussian copula DGP on NN, GBT, RF
# ---------------------------------------------------------------------------

def example_6_dgp(n):
    R = np.array([
        [1.0, 0.0, 0.2, 0.0],
        [0.0, 1.0, 0.9, 0.0],
        [0.2, 0.9, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    L = np.linalg.cholesky(R)
    Z = np.random.normal(size=(n, 4)) @ L.T
    X = stats.norm.cdf(Z)
    y = (4 * X[:, 0]
         + 3.87 * X[:, 1] ** 2
         + 2.97 * np.exp(-5 + 10 * X[:, 2]) / (1 + np.exp(-5 + 10 * X[:, 2]))
         + 13.86 * (X[:, 0] - 0.5) * (X[:, 1] - 0.5)
         + np.random.normal(0, 0.5, X.shape[0]))
    return X, y


def example_6_f_factory_mlp(X, y):
    mlp = MLPRegressor(
        hidden_layer_sizes=(80,), activation="relu", solver="lbfgs",
        alpha=0.00001, learning_rate="constant",
        max_iter=1000, random_state=42,
    )
    mlp.fit(X, y)
    print(mlp.score(X, y))
    return lambda X: mlp.predict(X)


def example_6_f_factory_gradient_boosted_tree(X, y):
    hgbr = HistGradientBoostingRegressor(
        max_iter=1000, learning_rate=0.1, max_depth=3, random_state=42,
    )
    hgbr.fit(X, y)
    print(hgbr.score(X, y))
    return lambda X: hgbr.predict(X)


def example_6_f_factory_random_forest(X, y):
    rf = RandomForestRegressor(n_estimators=500, max_depth=5, random_state=42)
    rf.fit(X, y)
    print(rf.score(X, y))
    return lambda X: rf.predict(X)


def run_example_6(model="mlp", replications=1):
    n = 10000
    K = bin_selection(n)
    factory = {
        "mlp": example_6_f_factory_mlp,
        "gbt": example_6_f_factory_gradient_boosted_tree,
        "rf": example_6_f_factory_random_forest,
    }[model]
    return replicate_ale_vim_training(
        example_6_dgp, factory, n=n, K=K, replications=replications
    )


# ---------------------------------------------------------------------------
# Example 7: UCI Bike Sharing on NN
# ---------------------------------------------------------------------------

def load_bikesharing(path="data/uci_bikesharing.csv"):
    df = pd.read_csv(path)
    df = df.dropna()
    # observations with 0 humidity / atemp==0.2424 & low temp are sensor errors
    df = df[df["hum"] != 0]
    df = df[(df["atemp"] != 0.2424) | (df["temp"] <= 0.5)]
    df["quarter"] = 1 + 4 * df["yr"] + df["mnth"] // 4
    df = df.rename(columns={"mnth": "month", "hr": "hour",
                            "weathersit": "weather_situation"})
    X = df[["quarter", "month", "hour", "holiday", "weekday", "workingday",
            "weather_situation", "atemp", "hum", "windspeed"]].copy()
    X["holiday"] = X["holiday"].astype("category")
    X["workingday"] = X["workingday"].astype("category")
    y = df["cnt"]
    return X, y


def summarize_model_performance(model, X_test, y_test):
    y_pred = np.exp(model.predict(X_test))
    y_true = np.exp(y_test)
    mse = mean_squared_error(y_true, y_pred)
    print("Mean Squared Error:", mse)
    print("R^2:", model.score(X_test, y_test))


def plot_correlation(X):
    corr = X.corr()
    plt.figure(figsize=(10, 8))
    plt.matshow(corr, cmap="coolwarm", fignum=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.colorbar()
    plt.title("Correlation Matrix of Covariates", pad=20)
    plt.show()


def run_example_7(plot_corr=False, K=300):
    X, y = load_bikesharing()
    if plot_corr:
        plot_correlation(X)

    est = MLPRegressor(hidden_layer_sizes=(25,), activation="logistic", 
                       alpha=0.05, max_iter=1000, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in cv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = np.log(y.iloc[train_index]), np.log(y.iloc[test_index])
        est.fit(X_train, y_train)
        print("Training set performance:")
        summarize_model_performance(est, X_train, y_train)

    est.fit(X, np.log(y))
    f = lambda x: est.predict(x)

    ale = ALE(f, X, K=K)
    explanation = ale.explain()
    print(explanation)
    return explanation


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", choices=["4", "5", "6", "7", "all"],
                        help="Which example to run")
    parser.add_argument("--replications", type=int, default=None,
                        help="Number of replications (default per-example)")
    parser.add_argument("--ex6-model", choices=["mlp", "gbt", "rf"], default="mlp",
                        help="Model to use for example 6")
    parser.add_argument("--tune", action="store_true",
                        help="Run hyperparameter tuning where applicable (ex5)")
    parser.add_argument("--plot-corr", action="store_true",
                        help="Show correlation heatmap in example 7")
    args = parser.parse_args()

    # Expected reference values for numerical checks (from original paper /
    # prior known-good runs). Tolerances account for Monte Carlo error via SE.
    if args.example in ("4", "all"):
        print("\n=== Example 4 ===")
        m, c, q = run_example_4(replications=args.replications or 100)
        # betas = [3, 2, 1]; main and total importances should recover these.
        _check_values(m, [3.0, 2.0, 1.0], "Ex4 Main")
        _check_values(c, [3.0, 2.0, 1.0], "Ex4 Connected")
        _check_values(q, [3.0, 2.0, 1.0], "Ex4 Quantile")
    if args.example in ("5", "all"):
        print("\n=== Example 5 ===")
        m, c, q = run_example_5(replications=args.replications or 100,
                                tune=args.tune)
        _check_values(m, [0.288, 0.307], "Ex5 Main")
        _check_values(c, [0.289, 0.308], "Ex5 Connected")
        _check_values(q, [0.290, 0.309], "Ex5 Quantile")
    if args.example in ("6", "all"):
        print("\n=== Example 6 ===")
        reps = args.replications or 10
        m, c, q = run_example_6(model=args.ex6_model, replications=reps)
        # Single-run model fits vary; check is informational (non-strict).
        strict = reps > 1
        # NOTE: reference values are reported for the "total" / main metric in
        # the docstring TODO -- those are the "main" outputs from ALE here.
        _check_values(m, [1.156, 1.174, 1.139, 0.014], "Ex6 Main",
                      abs_tol=0.1, rel_tol=0.15, strict=strict)
        _check_values(c, [1.633, 1.633, 1.14, 0.021], "Ex6 Connected",
                      abs_tol=0.1, rel_tol=0.15, strict=strict)
        _check_values(q, [1.627, 1.627, 1.14, 0.022], "Ex6 Quantile",
                      abs_tol=0.1, rel_tol=0.15, strict=strict)
    if args.example in ("7", "all"):
        print("\n=== Example 7 ===")
        explanation = run_example_7(plot_corr=args.plot_corr)
        # Single run on real data; fitted MLP can vary across environments.
        main_exp = [0.261, 0.109, 1.233, 0.025, 0.071,
                    0.09, 0.061, 0.262, 0.076, 0.034]
        conn_exp = [0.287, 0.223, 1.983, 0.268, 0.223,
                    0.561, 0.074, 0.363, 0.135, 0.082]
        quant_exp = [0.289, 0.222, 1.982, 0.265, 0.226,
                     0.559, 0.076, 0.363, 0.136, 0.079]
        _check_values(explanation.loc["main"].values[None, :], main_exp,
                      "Ex7 Main", abs_tol=0.1, rel_tol=0.25, strict=False)
        _check_values(explanation.loc["total_connected"].values[None, :],
                      conn_exp, "Ex7 Connected",
                      abs_tol=0.1, rel_tol=0.25, strict=False)
        _check_values(explanation.loc["total_quantile"].values[None, :],
                      quant_exp, "Ex7 Quantile",
                      abs_tol=0.1, rel_tol=0.25, strict=False)
