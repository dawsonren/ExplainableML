import numpy as np
from tqdm.auto import tqdm

from ale.ale import ALE


def replicate_ale_vim(dgp, f, n, bins=20, replications=100, categorical=None):
    """Replicate ALE variable importance."""
    nvars = dgp(1).shape[1]  # number of variables in the DGP
    vim_mains = np.zeros((replications, nvars))
    vim_connected = np.zeros((replications, nvars))
    vim_quantiles = np.zeros((replications, nvars))
    for i in tqdm(range(replications), desc="Replicating ALE VIMs"):
        X = dgp(n)
        ale = ALE(f, X, bins=bins, categorical=categorical)
        explanation = ale.explain()
        vim_mains[i, :] = explanation.loc["main"].values
        vim_connected[i, :] = explanation.loc["total_connected"].values
        vim_quantiles[i, :] = explanation.loc["total_quantile"].values

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
    dgp, f_factory, n, bins=20, replications=100, categorical=None
):
    """Replicate ALE variable importance and train f on X."""
    nvars = dgp(1)[0].shape[1]  # number of variables in the DGP
    vim_mains = np.zeros((replications, nvars))
    vim_connected = np.zeros((replications, nvars))
    vim_quantiles = np.zeros((replications, nvars))
    for i in tqdm(range(replications), desc="Replicating ALE VIMs"):
        X, y = dgp(n)
        # maps training data to a function
        f = f_factory(X, y)
        ale = ALE(f, X, bins=bins, categorical=categorical)
        explanation = ale.explain()
        vim_mains[i, :] = explanation.loc["main"].values
        vim_connected[i, :] = explanation.loc["total_connected"].values
        vim_quantiles[i, :] = explanation.loc["total_quantile"].values

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
