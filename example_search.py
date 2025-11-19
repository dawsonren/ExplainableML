import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
import matplotlib.pyplot as plt
# from scalene import scalene_profiler

from ale import ALE
from shapley import SHAP

DEBUG = False

# vary rho, sigma_eps, f, n, p
def f_1(X):
    # linear model with beta = 1
    return X.sum(axis=1)

def f_2(X):
    # return sum of first feature, square of second and third,
    # cube of fourth and fifth, etc.
    result = X[:, 0].copy()
    # make sure number of columns is odd
    if X.shape[1] % 2 == 0:
        raise ValueError("Number of columns must be odd for f_2")
    for j in range(1, X.shape[1]):
        result += X[:, j] ** ((j + 1) // 2 + 1)
    return result

def f_3(X):
    # exponential functions
    return np.exp(X).sum(axis=1)

def f_4(X):
    # sine and cosine functions, alternating
    # and increasing frequency
    # first feature is sin(x)
    # second and third feature are cos(2x), sin(2x)
    # fourth and fifth feature are cos(3x), sin(3x), etc.
    result = np.sin(X[:, 0])
    for j in range(1, X.shape[1], 2):
        freq = ((j + 1) // 2) + 1
        result += np.sin(freq * X[:, j])
        if j + 1 < X.shape[1]:
            result += np.cos(freq * X[:, j + 1])
    return result
    
def explain_f_1(x_explain):
    # x_explain is 1D array of shape (p,)
    # explanations for additive functions are
    # just \phi_j(x) = f_j(x_j) - E[f_j(X_j)]
    # which for f_1 is just x_j - E[X_j] = x_j - 0 = x_j
    explanation = x_explain.copy()
    return explanation

def explain_f_2(x_explain):
    explanation = np.zeros_like(x_explain)
    explanation[0] = x_explain[0]  # first feature is linear
    for j in range(1, len(x_explain)):
        p = ((j + 1) // 2) + 1
        # for even, x_j^p - E[X_j^p] = x_j^p - 1 / (p + 1)
        # for odd, x_j^p - E[X_j^p] = x_j^p
        if p % 2 == 0:
            explanation[j] = x_explain[j] ** p - 1 / (p + 1)
        else:
            explanation[j] = x_explain[j] ** p
    return explanation

def explain_f_3(x_explain):
    explanation = np.exp(x_explain) - (np.exp(1) - np.exp(-1)) / 2
    return explanation

def explain_f_4(x_explain):
    explanation = np.zeros_like(x_explain)
    explanation[0] = np.sin(x_explain[0])  # first feature
    for j in range(1, len(x_explain), 2):
        freq = ((j + 1) // 2) + 1
        explanation[j] = np.sin(freq * x_explain[j])  # E[sin(freq * X_j)] = 0
        if j + 1 < len(x_explain):
            explanation[j + 1] = np.cos(freq * x_explain[j + 1]) - np.sin(freq) / freq  # E[cos(freq * X_j)] = sin(freq) / freq
    return explanation


f_list = [f_1, f_2, f_3, f_4]
explain_f_list = [explain_f_1, explain_f_2, explain_f_3, explain_f_4]

rhos = [0.9, 0.99]
sigma_epss = [0.25, 1, 2]
ns = [1000]
ps = [5, 11]
f_idxs = [0, 1, 3]

# rhos = [0.99]
# sigma_epss = [0.5]
# ns = [100]
# ps = [3]
# f_idxs = [3] # zero-indexed

REPLICATIONS = 40
M = 1000

NN_PARAM_GRID = {
    "hidden_layer_sizes": [(20,), (40, ), (60, ), (80, ), (100, ), (150, ), (20, 20), (40, 40), (60, 60)],
    "activation": ['relu', 'tanh'],
    "solver": ['adam'],
    "alpha": [0.0001, 0.00025, 0.0005, 0.001],
    "learning_rate": ['constant'],
    "max_iter": [1000, 5000]
}

RF_PARAM_GRID = {
    "n_estimators": [100, 300, 1000, 3000],
    "max_depth": [4, 8, 12, 16, 20, 24, 28, 32]
}

GB_PARAM_GRID = {
    "n_estimators": [100, 300, 1000],
    "learning_rate": [0.01, 0.1],
    "max_depth": [2, 4, 6, 8, 10, 12, 14]
}

def gaussian_copula_uniform(normal_variates):
    return norm.cdf(normal_variates) * 2 - 1

def dgp(n, p, f, rho, sigma_eps):
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    pairs = p // 2
    # construct pairs of correlated normal r.v.
    Zs = [np.random.multivariate_normal(mean, cov, n) for _ in range(pairs)]
    # convert to uniform r.v. using the CDF of the standard normal
    u1 = np.random.uniform(-1, 1, n)
    us = [(gaussian_copula_uniform(Zs[i][:, 0]), gaussian_copula_uniform(Zs[i][:, 1])) for i in range(pairs)]
    # flatten us
    us = [item for sublist in us for item in sublist]
    # combine into X
    X = np.column_stack((u1, *us))
    # additive noise model
    y = f(X) + np.random.normal(0, sigma_eps, n)
    return X, y

def hyperparameter_tuning(X, y, sigma_eps):
    # NN
    nn_grid_search = GridSearchCV(MLPRegressor(random_state=42), NN_PARAM_GRID, cv=5, n_jobs=-1)
    nn_grid_search.fit(X, y)
    print(f"Best NN parameters: {nn_grid_search.best_params_}")
    print(f"R^2: {nn_grid_search.best_score_}")

    # RF
    rf_grid_search = GridSearchCV(RandomForestRegressor(random_state=42), RF_PARAM_GRID, cv=5, n_jobs=-1)
    rf_grid_search.fit(X, y)
    print(f"Best RF parameters: {rf_grid_search.best_params_}")
    print(f"R^2: {rf_grid_search.best_score_}")

    # GB
    gb_grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), GB_PARAM_GRID, cv=5, n_jobs=-1)
    gb_grid_search.fit(X, y)
    print(f"Best GB parameters: {gb_grid_search.best_params_}")
    print(f"R^2: {gb_grid_search.best_score_}")

    print("Theoretical R^2:", 1 - (sigma_eps ** 2) / np.var(y))

    return {
        "nn": nn_grid_search.best_params_,
        "rf": rf_grid_search.best_params_,
        "gb": gb_grid_search.best_params_
    }

def f_factory(X, y, sigma_eps, params, type="nn", verbose=False):
    # train a black-box model
    if type == "nn":
        model = MLPRegressor(
            hidden_layer_sizes=params["hidden_layer_sizes"],
            activation=params["activation"],
            solver=params["solver"],
            alpha=params["alpha"],
            learning_rate=params["learning_rate"],
            max_iter=params["max_iter"],
            random_state=42
        )
    elif type == "rf":
        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42
        )
    elif type == "gb":
        model = GradientBoostingRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            random_state=42
        )
        
    model.fit(X, y)
    # show R^2 on training data
    if verbose:
        # get CV R^2
        cv = KFold(n_splits=10, shuffle=True, random_state=42)
        r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1)
        print(f"CV R^2 scores for ({type}): {r2_scores}, mean: {r2_scores.mean():.4f}, std: {r2_scores.std():.4f}")
        print(f"Theoretical R^2: {1 - (sigma_eps ** 2) / np.var(y):.4f}")

    return lambda X: model.predict(X)

def folder_name(n, p, rho, sigma_eps, f_index):
    return Path(f"figures/example_search/n{n}_p{p}_rho{int(rho*100)}_sigma{int(sigma_eps*100)}_f{f_index}")

def ale_shap_barplot(x, x_idx, p, ale_importances, shap_importances):
    features = [f"X{j+1}" for j in range(p)]
    plt.figure(figsize=(6, 4 * p))
    x_explain = [x[x_idx]] * p
    for j in range(p):
        plt.subplot(p, 1, j + 1)
        data = [
            ale_importances[:, x_idx, j, 0],  # nn
            shap_importances[:, x_idx, j, 0],  # nn
            ale_importances[:, x_idx, j, 1],  # rf
            shap_importances[:, x_idx, j, 1],  # rf
            ale_importances[:, x_idx, j, 2],  # gb
            shap_importances[:, x_idx, j, 2]   # gb
        ]
        plt.boxplot(data, tick_labels=['ALE NN', 'SHAP NN', 'ALE RF', 'SHAP RF', 'ALE GB', 'SHAP GB'])
        plt.title(f"Local VI x={list(x_explain)}, {features[j]}")
        plt.ylabel("Importance Value")
        plt.grid()

def ale_shap_lineplot(x, j, ale_importances, shap_importances, true_importances, model_type):
    # plot 50% and 90% CI for ale and shap importances
    plt.figure(figsize=(8, 6))
    ale_mean = np.mean(ale_importances, axis=0)
    ale_50_lower = np.percentile(ale_importances, 25, axis=0)
    ale_50_upper = np.percentile(ale_importances, 75, axis=0)
    ale_90_lower = np.percentile(ale_importances, 5, axis=0)
    ale_90_upper = np.percentile(ale_importances, 95, axis=0)

    shap_mean = np.mean(shap_importances, axis=0)
    shap_50_lower = np.percentile(shap_importances, 25, axis=0)
    shap_50_upper = np.percentile(shap_importances, 75, axis=0)
    shap_90_lower = np.percentile(shap_importances, 5, axis=0)
    shap_90_upper = np.percentile(shap_importances, 95, axis=0)

    # ALE = orange, SHAP = blue
    plt.plot(x, ale_mean, label=f"ALE {model_type}", color=f"orange")
    plt.fill_between(x, ale_50_lower, ale_50_upper, color=f"orange", alpha=0.3)
    plt.fill_between(x, ale_90_lower, ale_90_upper, color=f"orange", alpha=0.1)

    plt.plot(x, shap_mean, label=f"SHAP {model_type}", color=f"blue")
    plt.fill_between(x, shap_50_lower, shap_50_upper, color=f"blue", alpha=0.3)
    plt.fill_between(x, shap_90_lower, shap_90_upper, color=f"blue", alpha=0.1)

    # plot true importances as black
    plt.plot(x, true_importances, label="True Importance", color="black")

    plt.title(f"ALE vs SHAP for Feature {j + 1}")
    plt.xlabel("x value")
    plt.ylabel("Importance Value")
    plt.grid()
    plt.legend()

def f_variability_lineplot(x, f_values):
    # plot std dev of f_values across replications
    f_std = np.std(f_values, axis=0)
    plt.figure(figsize=(10, 4))
    for m, model_type in enumerate(["nn", "rf", "gb"]):
        plt.plot(x, f_std[:, m], label=model_type)
    plt.title("Standard Deviation of f(x,-x,x,x,-x) Across Replications")
    plt.xlabel("x value")
    plt.ylabel("Standard Deviation of f")
    plt.grid()
    plt.legend()

def run_replication(n, p, f, rho, sigma_eps, params, f_index, xs, replication_index):
    print(f"Running replication with n={n}, p={p}, rho={rho}, sigma_eps={sigma_eps}, f=f_{f_index + 1}")
    X, y = dgp(n, p, f, rho, sigma_eps)

    time_ale_tree = 0
    time_ale_local = 0
    time_shap_local = 0

    # each replication, train nn, rf, gb models and keep track of local importances
    # at x_explain = [x, x, x, ... , x] for each dimension

    # dimensions: len(x), p features, 3 models
    ale_importances = np.zeros((len(xs), p, 3))
    shap_importances = np.zeros((len(xs), p, 3))
    # also keep track of trained function f's variability at [x, -x, x, x, -x]
    f_values = np.zeros((len(xs), 3))

    X, y = dgp(n, p, f, rho, sigma_eps)

    for m, model_type in enumerate(["nn", "rf", "gb"]):
        f = f_factory(X, y, sigma_eps, params[model_type], type=model_type)
        ale_explainer = ALE(f, X, verbose=False, interpolate=True, centering="y")
        t = time.perf_counter()
        # scalene_profiler.start()
        ale_explainer.explain()
        # scalene_profiler.stop()

        image_slug = folder_name(n, p, rho, sigma_eps, f_index + 1)

        if not image_slug.exists():
            image_slug.mkdir(parents=True, exist_ok=True)

        if replication_index in [0, 1]:
            # save ale_ice plots for feature 1
            if not DEBUG:
                ale_explainer.plot_ale_ice(1, True)
                plt.savefig(image_slug / f"model_type{model_type}_replication{replication_index}_ice_plot.png")
                plt.clf()

            # save connected paths plot for features 2 and 3
            if not DEBUG:
                ale_explainer.plot_connected_paths(2, 3)
                plt.savefig(image_slug / f"model_type{model_type}_replication{replication_index}_connected_paths_plot.png")
                plt.clf()

        time_ale_tree += time.perf_counter() - t
        shap_explainer = SHAP(f, X, verbose=False)

        for i, xi in enumerate(xs):
            x_explain = np.array([xi] * p)
            # keep track of trained function f's variability at [x, -x, x, x, -x]
            f_values[i, m] = f(np.array([xi] + [xi, -xi] * (p // 2)).reshape(1, -1))[0]

            t = time.perf_counter()
            ale_local = ale_explainer.explain_local(x_explain, method="tree")
            time_ale_local += time.perf_counter() - t
            for j in range(p):
                ale_importances[i, j, m] = ale_local[f'X{j+1}']

            t = time.perf_counter()
            shap_local = shap_explainer.explain_local(x_explain, method="permutation", num_samples=M)
            time_shap_local += time.perf_counter() - t
            for j in range(p):
                shap_importances[i, j, m] = shap_local[f'X{j+1}']

    # print average times for replications * len(x) * 3 models
    total_explanations = len(xs) * 3
    total_trees = 3
    timing = {
        "ale_tree_time": time_ale_tree / total_trees,
        "ale_local_time": time_ale_local / total_explanations,
        "shap_local_time": time_shap_local / total_explanations
    }

    return ale_importances, shap_importances, f_values, timing

if __name__ == "__main__":
    print("Starting example search...")
    # plot instead of bar plot, line plot for each variable with 50% CI and 90% CI (requires at least 20 replications)
    xs = np.arange(-1, 1.05, 0.05)

    timing_df = []

    # iterate over all combinations of rhos, sigma_epss, ns, ps, fs
    for i, (rho, sigma_eps, n, p, f_index) in enumerate(product(rhos, sigma_epss, ns, ps, f_idxs)):
        slug = folder_name(n, p, rho, sigma_eps, f_index + 1)
        already_done = slug.exists() and (slug / "ale_importances.npy").exists() and (slug / "shap_importances.npy").exists() and (slug / "f_values.npy").exists()

        if already_done:
            print(f"Skipping already done example {i+1}/{len(rhos) * len(sigma_epss) * len(ns) * len(ps) * len(f_idxs)}")
            continue

        if not slug.exists():
            slug.mkdir(parents=True, exist_ok=True)

        f = f_list[f_index]
        print(f"\n\n\n\n\nRunning example {i+1}/{len(rhos) * len(sigma_epss) * len(ns) * len(ps) * len(f_idxs)}")
        print(f"Parameters: rho={rho}, sigma_eps={sigma_eps}, n={n}, p={p}, f=f_{f_index + 1}")

        # generate hyperparameters
        print("\n\n\nGenerating hyperparameters...")
        X, y = dgp(n, p, f, rho, sigma_eps)
        params = hyperparameter_tuning(X, y, sigma_eps)
        print("Hyperparameters generated...\n\n\n")

        # store results over replications
        ale_importances_list = []
        shap_importances_list = []
        true_importances_list = []
        f_values_list = []
        timings_list = []

        for i in range(REPLICATIONS):
            print(f"Replication {i+1}/{REPLICATIONS}...")
            ale_importances, shap_importances, f_values, timing = run_replication(n, p, f, rho, sigma_eps, params, f_index, xs, i)

            ale_importances_list.append(ale_importances)
            shap_importances_list.append(shap_importances)
            f_values_list.append(f_values)
            timings_list.append(timing)

        ale_importances_array = np.stack(ale_importances_list, axis=0)
        shap_importances_array = np.stack(shap_importances_list, axis=0)
        f_values_array = np.stack(f_values_list, axis=0)

        true_importances = np.zeros((len(xs), p))
        for i, xi in enumerate(xs):
            x_explain = np.array([xi] * p)
            true_importances[i, :] = explain_f_list[f_index](x_explain)

        # save figure of ale and shap importances barplots for each x in xs
        for j in range(p):
            for m, model_type in enumerate(["nn", "rf", "gb"]):
                if not DEBUG:
                    ale_shap_lineplot(xs, j, ale_importances_array[:, :, j, m], shap_importances_array[:, :, j, m], true_importances[:, j], model_type)
                    plt.savefig(slug / f"ale_shap_feature{j+1}_model{model_type}.png")
                    plt.clf()

        # for x_idx in range(len(xs)):
        #     if not DEBUG:
        #         ale_shap_barplot(xs, x_idx=x_idx, p=p, ale_importances=ale_importances_array, shap_importances=shap_importances_array)
        #         plt.savefig(slug / f"barplot_x{xs[x_idx]:.1f}.png")
        #         plt.clf()

        # save figure of f variability lineplot
        if not DEBUG:
            f_variability_lineplot(xs, f_values_array)
            plt.savefig(slug / f"f_variability.png")
            plt.clf()

        # save npy files of ale_importances_array, shap_importances_array, f_values_array
        if not DEBUG:
            np.save(slug / "ale_importances.npy", ale_importances_array)
            np.save(slug / "shap_importances.npy", shap_importances_array)
            np.save(slug / "f_values.npy", f_values_array)

        # save csv of timing averaged over replications
        avg_timing = {
            "ale_tree_time": np.mean([t["ale_tree_time"] for t in timings_list]),
            "ale_local_time": np.mean([t["ale_local_time"] for t in timings_list]),
            "shap_local_time": np.mean([t["shap_local_time"] for t in timings_list])
        }

        timing_df.append({
            "n": n,
            "p": p,
            "rho": rho,
            "sigma_eps": sigma_eps,
            "f_index": f_index,
            **avg_timing,
            "hyperparameters_nn": params["nn"],
            "hyperparameters_rf": params["rf"],
            "hyperparameters_gb": params["gb"]
        })

        # save timing df each iteration so we don't lose data
        timing_df = pd.DataFrame(timing_df)
        timing_df.to_csv("example_search_timing.csv", index=False)

        # clear all figures
        plt.close('all')
    
    print("Example search complete.")