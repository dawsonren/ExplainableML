import numpy as np

from algorithms.ale_vim import ale_global_main, ale_connected_total, ale_quantile_total


def replicate_ale_vims(dgp, f, nvars, n, bins=20, replications=100, categorical=False):
    """Replicate ALE variable importance."""
    vim_mains = np.zeros((replications, nvars))
    vim_connected = np.zeros((replications, nvars))
    vim_quantiles = np.zeros((replications, nvars))
    for i in range(replications):
        X = dgp(n)
        for j in range(nvars):
            vim_mains[i, j] = ale_global_main(
                f, X, j + 1, bins, categorical=categorical
            )
            vim_connected[i, j] = ale_connected_total(
                f, X, j + 1, bins, categorical=categorical
            )
            vim_quantiles[i, j] = ale_quantile_total(
                f, X, j + 1, bins, categorical=categorical
            )
            print(
                f"Replication {i + 1}, Variable {j + 1}: VIM Main: {vim_mains[i, j]}, VIM Connected: {vim_connected[i, j]}, VIM Quantile: {vim_quantiles[i, j]}"
            )

    for j in range(nvars):
        print(
            f"VIM {j + 1} Main Importance - Mean:",
            np.mean(vim_mains[:, j]),
            "CI:",
            np.percentile(vim_mains[:, j], [2.5, 97.5]),
        )
        print(
            f"VIM {j + 1} Connected Importance - Mean:",
            np.mean(vim_connected[:, j]),
            "CI:",
            np.percentile(vim_connected[:, j], [2.5, 97.5]),
        )
        print(
            f"VIM {j + 1} Quantile Importance - Mean:",
            np.mean(vim_quantiles[:, j]),
            "CI:",
            np.percentile(vim_quantiles[:, j], [2.5, 97.5]),
        )

    return vim_mains, vim_connected, vim_quantiles
