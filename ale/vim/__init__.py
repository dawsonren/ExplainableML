"""
ale.vim — variable-importance and local-explanation computation for ALE.

Stable internal API consumed by `ale.ale.ALE` and by diagnostic/dashboard code.
"""

from ale.vim.g_values import GValues
from ale.vim.paths import (
    _ale_main_vim,
    _ale_total_vim,
    _generate_random_paths,
    _generate_quantile_delta_values,
    calculate_g_values,
    observation_to_path,
)
from ale.vim.local import _ale_local_vim
from ale.vim.local_path_integral import _ale_local_vim_path_integral
from ale.vim.local_multi_path import _ale_local_vim_multi_path
from ale.vim.local_terms import (
    _local_term_path_rep,
    route_first_index_at_bin,
)

# Public, stable aliases for cross-package consumers (diagnostics, dashboard).
compute_total_vim = _ale_total_vim
compute_main_vim = _ale_main_vim
local_term_path_rep = _local_term_path_rep

__all__ = [
    # Stable public API
    "compute_total_vim",
    "compute_main_vim",
    "local_term_path_rep",
    "route_first_index_at_bin",
    "GValues",
    # Private-but-imported (kept for ale.ale and tests)
    "_ale_main_vim",
    "_ale_total_vim",
    "_ale_local_vim",
    "_ale_local_vim_path_integral",
    "_ale_local_vim_multi_path",
    "_local_term_path_rep",
    "_generate_random_paths",
    "_generate_quantile_delta_values",
    "calculate_g_values",
    "observation_to_path",
]
