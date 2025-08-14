import numpy as np
from tqdm.auto import tqdm

from algorithms.ale_vim import ale_main_vim, ale_connected_total, ale_quantile_total


def replicate_ale_vims(dgp, f, nvars, n, bins=20, replications=100, categorical=False):
    """Replicate ALE variable importance."""
    vim_mains = np.zeros((replications, nvars))
    vim_connected = np.zeros((replications, nvars))
    vim_quantiles = np.zeros((replications, nvars))
    for i in tqdm(range(replications), desc="Replicating ALE VIMs"):
        X = dgp(n)
        for j in range(nvars):
            vim_mains[i, j] = ale_main_vim(f, X, j + 1, bins, categorical=categorical)
            vim_connected[i, j] = ale_connected_total(
                f, X, j + 1, bins, categorical=categorical
            )
            vim_quantiles[i, j] = ale_quantile_total(
                f, X, j + 1, bins, categorical=categorical
            )

    for j in range(nvars):
        print(
            f"VIM {j + 1} Main Importance - Mean:",
            np.mean(vim_mains[:, j]),
            "CI:",
            np.percentile(vim_mains[:, j], [2.5, 97.5]),
            "SD:",
            np.std(vim_mains[:, j]),
        )
        print(
            f"VIM {j + 1} Connected Importance - Mean:",
            np.mean(vim_connected[:, j]),
            "CI:",
            np.percentile(vim_connected[:, j], [2.5, 97.5]),
            "SD:",
            np.std(vim_connected[:, j]),
        )
        print(
            f"VIM {j + 1} Quantile Importance - Mean:",
            np.mean(vim_quantiles[:, j]),
            "CI:",
            np.percentile(vim_quantiles[:, j], [2.5, 97.5]),
            "SD:",
            np.std(vim_quantiles[:, j]),
        )

    return vim_mains, vim_connected, vim_quantiles


def replicate_ale_vim_training(
    dgp, f_factory, nvars, n, bins=20, replications=100, categorical=False
):
    """Replicate ALE variable importance and train f on X."""
    vim_mains = np.zeros((replications, nvars))
    vim_connected = np.zeros((replications, nvars))
    vim_quantiles = np.zeros((replications, nvars))
    for i in tqdm(range(replications), desc="Replicating ALE VIMs"):
        X, y = dgp(n)
        # maps training data to a function
        f = f_factory(X, y)
        for j in range(nvars):
            vim_mains[i, j] = ale_main_vim(f, X, j + 1, bins, categorical=categorical)
            vim_connected[i, j] = ale_connected_total(
                f, X, j + 1, bins, categorical=categorical
            )
            vim_quantiles[i, j] = ale_quantile_total(
                f, X, j + 1, bins, categorical=categorical
            )

    for j in range(nvars):
        print(
            f"VIM {j + 1} Main Importance - Mean:",
            np.mean(vim_mains[:, j]),
            "CI:",
            np.percentile(vim_mains[:, j], [2.5, 97.5]),
            "SD:",
            np.std(vim_mains[:, j]),
        )
        print(
            f"VIM {j + 1} Connected Importance - Mean:",
            np.mean(vim_connected[:, j]),
            "CI:",
            np.percentile(vim_connected[:, j], [2.5, 97.5]),
            "SD:",
            np.std(vim_connected[:, j]),
        )
        print(
            f"VIM {j + 1} Quantile Importance - Mean:",
            np.mean(vim_quantiles[:, j]),
            "CI:",
            np.percentile(vim_quantiles[:, j], [2.5, 97.5]),
            "SD:",
            np.std(vim_quantiles[:, j]),
        )

    return vim_mains, vim_connected, vim_quantiles
