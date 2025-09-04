"""
Tests for consistency of ALE VIMs.
"""

import numpy as np
import scipy.stats as stats
import unittest
from utils import bin_selection
from ale import ALE


def generate_2d_data_normal(n, rho=0.5):
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]  # covariance matrix
    data = np.random.multivariate_normal(mean, cov, n)
    return data


def generate_2d_data_uniform(n, sigma=0.5):
    x1 = np.random.uniform(-np.sqrt(3), np.sqrt(3), n)
    x2 = x1 + np.random.multivariate_normal([0], [[sigma]], n).flatten()
    return np.column_stack((x1, x2))


def check_in_confidence_interval(samples, theoretical_value, alpha=0.05):
    """
    Check if the theoretical value is within the confidence interval of the samples.
    """
    mean = np.mean(samples)
    std = np.std(samples, ddof=1)
    n = len(samples)
    z_score = stats.norm.ppf(1 - alpha / 2)  # for 1 - alpha confidence interval
    margin_of_error = z_score * (std / np.sqrt(n))
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    return lower_bound <= theoretical_value <= upper_bound


class TestALEVIMs(unittest.TestCase):
    def uniform_data(self, n, sigma=0.5):
        X = generate_2d_data_uniform(n, sigma)

        def f(x):
            return x[:, 0] + x[:, 1] + x[:, 0] * x[:, 1]

        K = bin_selection(n)
        replications = 10

        ale_global_main_vals = np.zeros(replications)
        ale_connected_total_vals = np.zeros(replications)
        ale_quantile_total_vals = np.zeros(replications)

        for r in range(replications):
            ale = ALE(f, X, bins=K)
            ale_global_main_vals[r] = ale.ale_main_vim(1)
            ale_connected_total_vals[r] = ale.ale_total_vim(1, method="connected")
            ale_quantile_total_vals[r] = ale.ale_total_vim(1, method="quantile")

        theoretical_main = 6 / 5
        theoretical_total = 6 / 5 + sigma**2

        print(
            f"n={n}, ale_global_main={ale_global_main_vals.mean()}, ale_connected_total={ale_connected_total_vals.mean()}, ale_quantile_total={ale_quantile_total_vals.mean()}"
        )

        self.assertTrue(
            check_in_confidence_interval(ale_global_main_vals, theoretical_main)
        )
        self.assertTrue(
            check_in_confidence_interval(ale_connected_total_vals, theoretical_total)
        )
        self.assertTrue(
            check_in_confidence_interval(ale_quantile_total_vals, theoretical_total)
        )

    def test_uniform_data(self):
        n = 10000
        for sigma in np.arange(0.1, 1.0, 0.1):
            with self.subTest(sigma=sigma):
                self.uniform_data(n, sigma)
