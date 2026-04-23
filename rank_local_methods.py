"""
rank_local_methods.py

For each experiment group (same DGP, model, SHAP config), rank the local
methods (interpolate, path_rep, self) by StdRed (higher = better).
Print average rank per local method across all groups.

Usage:
    python rank_local_methods.py [--cache-dir cached_explanations]
"""

import argparse
from collections import defaultdict

import numpy as np

from summarize_experiments import _walk_cache


def main():
    parser = argparse.ArgumentParser(
        description="Rank local methods by StdRed across experiments."
    )
    parser.add_argument("--cache-dir", default="cached_explanations",
                        help="Root directory containing per-config sub-folders.")
    args = parser.parse_args()

    rows = _walk_cache(args.cache_dir)
    if not rows:
        print(f"No results found under {args.cache_dir!r}")
        return

    # Group rows by (config_name, results_file, shap_tag) — everything except
    # the ALE config / local_method.
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        gkey = (r["config_name"], r["_results_file"], r.get("shap_tag", ""))
        groups[gkey].append(r)

    # Only keep groups that have more than one local_method to compare.
    groups = {k: v for k, v in groups.items()
              if len({r.get("local_method") for r in v}) > 1}

    if not groups:
        print("No groups with multiple local methods found.")
        return

    # For each group, rank local methods by rel_stddev_reduction (descending).
    # Rank 1 = highest StdRed (best).
    rank_sums: dict[str, list[float]] = defaultdict(list)

    print(f"{'Group':<70} {'Local Method':<15} {'StdRed':>10} {'Rank':>6}")
    print("-" * 105)

    for gkey, group_rows in sorted(groups.items()):
        config_name, results_file, shap_tag = gkey
        group_label = f"{config_name}/{results_file} [{shap_tag}]"

        # Sort descending by StdRed (best first)
        scored = sorted(
            group_rows,
            key=lambda r: r["rel_stddev_reduction"]
            if not np.isnan(r["rel_stddev_reduction"])
            else -np.inf,
            reverse=True,
        )

        for rank, r in enumerate(scored, start=1):
            lm = r.get("local_method", "?")
            stdred = r["rel_stddev_reduction"]
            rank_sums[lm].append(rank)
            stdred_str = f"{stdred:+.1%}" if not np.isnan(stdred) else "N/A"
            print(f"{group_label:<70} {lm:<15} {stdred_str:>10} {rank:>6}")
        print()

    # Summary
    print("=" * 60)
    print(f"{'Local Method':<20} {'Avg Rank':>10} {'Groups':>8}")
    print("-" * 60)
    for lm in sorted(rank_sums, key=lambda k: np.mean(rank_sums[k])):
        ranks = rank_sums[lm]
        print(f"{lm:<20} {np.mean(ranks):>10.2f} {len(ranks):>8}")
    print("=" * 60)


if __name__ == "__main__":
    main()
