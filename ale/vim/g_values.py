from dataclasses import dataclass
from typing import Optional

import numpy as np

from ale.shared import (
    calculate_bin_index,
    linear_interpolation,
)


@dataclass
class GValues:
    """
    The g_values are the accumulated values for each of L paths along K bins.
    g_values has shape (K, L)
    Nkl has shape (K, L) and contains the number of observations in each bin for each path
    centering can be "x" or "y"
    interpolate is a boolean indicating whether to interpolate g_values

    centered_g_values, K and L should not be modified and will be computed
    when get_centered is called
    """
    g_values: np.ndarray
    xj: np.ndarray
    bins: np.ndarray
    k_x: np.ndarray  # 0-indexed bin indices for each observation
    l_x: np.ndarray  # 0-indexed path indices for each observation
    centering: str
    interpolate: bool
    categorical: bool

    n: Optional[int] = None
    K: Optional[int] = None
    L: Optional[int] = None
    centered_g_values: Optional[np.ndarray] = None

    def __post_init__(self):
        self.xj_mean = self.xj.mean()
        self.K, self.L = self.g_values.shape
        self.n = len(self.xj)
        self._compute_centered()

    def _compute_centered(self):
        if self.centering == "x":
            k_bar = calculate_bin_index(self.xj_mean, self.bins, self.K, self.categorical)

            if self.interpolate:
                centering = linear_interpolation(
                    x=self.xj_mean,
                    x0=self.bins[k_bar],
                    x1=self.bins[k_bar + 1],
                    y0=self.g_values[k_bar - 1, :] if k_bar != 0 else np.zeros(self.L),
                    y1=self.g_values[k_bar, :],
                )
            else:
                centering = self.g_values[k_bar, :]
        elif self.centering == "y":
            if self.interpolate:
                padded_g_values = np.pad(self.g_values, ((1, 0), (0, 0)), mode='constant', constant_values=0)
                g_values_per_observation = linear_interpolation(
                    x=self.xj,
                    x0=self.bins[self.k_x],
                    x1=self.bins[self.k_x + 1],
                    y0=padded_g_values[self.k_x, self.l_x],
                    y1=padded_g_values[self.k_x + 1, self.l_x],
                )
                centering = np.zeros(self.L)
                for l in range(self.L):
                    centering[l] = np.mean(g_values_per_observation[self.l_x == l])
            else:
                g_values_per_observation = self.g_values[self.k_x, self.l_x]
                centering = np.zeros(self.L)
                for l in range(self.L):
                    centering[l] = np.mean(g_values_per_observation[self.l_x == l])

        if self.interpolate:
            self.centered_g_values = np.pad(self.g_values, ((1, 0), (0, 0)), mode='constant', constant_values=0) - centering
        else:
            self.centered_g_values = self.g_values - centering

    def lookup_locals(self, k_idxs, l_idxs, x_explain_j):
        if not self.interpolate:
            return self.centered_g_values[k_idxs, l_idxs]
        return linear_interpolation(
            x=x_explain_j,
            x0=self.bins[k_idxs],
            x1=self.bins[k_idxs + 1],
            y0=self.centered_g_values[k_idxs, l_idxs],
            y1=self.centered_g_values[k_idxs + 1, l_idxs],
        )

    def plot_centered(self, ax):
        K, L = self.K, self.L
        if self.interpolate:
            for l in range(L):
                ax.plot(self.bins, self.centered_g_values[:, l], color=f"C{l}")
            for bin_edge in self.bins:
                ax.axvline(bin_edge, color='gray', linestyle='--')
        else:
            for l in range(L):
                for k in range(K):
                    ax.hlines(self.centered_g_values[k, l], self.bins[k], self.bins[k + 1], colors=f"C{l}")
            for bin_edge in self.bins:
                ax.axvline(bin_edge, color='gray', linestyle='--')

        ax.set_title(f"Centered g values (centering={self.centering}, interpolate={self.interpolate})")
        return ax
