"""
Plotting helpers for an `ALE` instance. Free functions taking an ALE first arg
so the core class stays focused on computation.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_connected_paths(ale, feature_1, feature_2):
    """
    Plot the connected paths for a pair of feature indices (0-based).

    Parameters:
    - feature_1: 0-based index or feature name of the examined feature.
    - feature_2: 0-based index or feature name of the secondary feature
      (plotted on y axis).
    """
    idx_1 = ale._get_feature_index(feature_1)
    idx_2 = ale._get_feature_index(feature_2)
    if idx_1 == idx_2:
        raise ValueError(
            "Feature indices must be different for connected paths plot."
        )

    if idx_1 not in ale.connected_paths or idx_2 not in ale.connected_paths:
        raise ValueError(
            f"Connected paths for feature indices {idx_1} and/or "
            f"{idx_2} not found. Please run ale_total_vim with "
            "method='connected' first."
        )

    paths = ale.connected_paths[idx_1]
    if not paths:
        raise ValueError(
            f"No connected path found between features {idx_1} and {idx_2}."
        )

    categorical_1 = ale.categorical[idx_1]
    edges_1 = ale.edges[idx_1]

    if not categorical_1:
        for e in edges_1:
            plt.axvline(x=e, color="black", linestyle="--", alpha=0.4)

    X = ale.X_values
    plt.scatter(X[:, idx_1], X[:, idx_2], alpha=0.3)
    for path in paths:
        flat_path = [item for interval in path for item in interval]
        plt.plot(X[flat_path, idx_1], X[flat_path, idx_2])

    plt.xlabel(f"{ale.feature_names[idx_1]}")
    plt.ylabel(f"{ale.feature_names[idx_2]}")
    plt.title(f"Connected Paths for {ale.feature_names[idx_1]}")


def plot_ale_ice(ale, feature):
    idx = ale._get_feature_index(feature)
    categorical = ale.categorical[idx]

    if idx not in ale.connected_paths or idx not in ale.connected_forest:
        raise ValueError(
            f"Connected paths/forest for feature index {idx} not found. "
            "Please run ale_total_vim with method='connected' first."
        )

    y_axis = ale.centered_g_values[idx].centered_g_values
    edges = ale.edges[idx]
    for l in range(y_axis.shape[1]):
        if categorical:
            original_edges = [ale.num_to_label[idx][int(e)] for e in edges]
            plt.bar(
                original_edges, y_axis[:, l], width=0.5, align="center", alpha=0.7
            )
            plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        else:
            plt.plot(edges, y_axis[:, l], color="blue", alpha=0.8)
    plt.xlabel(f"{ale.feature_names[idx]}")
    plt.ylabel("Centered g-values")
    plt.title(f"Centered g-values for {ale.feature_names[idx]}")
    if not categorical:
        for e in edges:
            plt.axvline(x=e, color="black", linestyle="--", alpha=0.4)


def _plot_paths_scatter(ale, idx_1, idx_2, paths, edges, colors, ax_data, ax_heatmap):
    """Left subplot: scatter of (feature_1, feature_2) colored by path, with
    centroid trajectories drawn on both the scatter and the f-heatmap axes."""
    ax_data.scatter(
        ale.X_values[:, idx_1], ale.X_values[:, idx_2],
        color="lightgray", alpha=0.3, s=5, zorder=1,
    )
    for l, path in enumerate(paths):
        c = colors[l]
        centroids_x, centroids_y = [], []
        for k, interval in enumerate(path):
            if len(interval) > 0:
                pts = ale.X_values[interval]
                assert np.all(pts[:, idx_1] >= edges[k]) and np.all(pts[:, idx_1] <= edges[k + 1]), \
                    "Points are not within bin edges"
                ax_data.scatter(
                    pts[:, idx_1], pts[:, idx_2],
                    color=c, alpha=0.6, s=10, zorder=2,
                )
                centroids_x.append(pts[:, idx_1].mean())
                centroids_y.append(pts[:, idx_2].mean())
        if centroids_x:
            ax_data.plot(centroids_x, centroids_y, color=c, alpha=0.5, linewidth=1)
            ax_heatmap.plot(centroids_x, centroids_y, color=c, alpha=0.8,
                            linewidth=2, marker='o', markersize=3)
    for e in edges:
        ax_data.axvline(e, color="black", linestyle="--", alpha=0.3)
    ax_data.set_xlabel(ale.feature_names[idx_1])
    ax_data.set_ylabel(ale.feature_names[idx_2])
    ax_data.set_title(
        f"Paths: {ale.feature_names[idx_1]} vs {ale.feature_names[idx_2]}"
    )


def _plot_f_heatmap(ale, idx_1, idx_2, edges, fig, ax_heatmap):
    """Middle subplot: heatmap of f over the (feature_1, feature_2) grid,
    holding other features at their mean (or most-common value for categoricals)."""
    grid_size = 50
    x1_min, x1_max = ale.X_values[:, idx_1].min(), ale.X_values[:, idx_1].max()
    x2_min, x2_max = ale.X_values[:, idx_2].min(), ale.X_values[:, idx_2].max()
    x1_grid = np.linspace(x1_min, x1_max, grid_size)
    x2_grid = np.linspace(x2_min, x2_max, grid_size)
    xx1, xx2 = np.meshgrid(x1_grid, x2_grid)

    X_grid = np.zeros((grid_size * grid_size, ale.d))
    for j in range(ale.d):
        if ale.categorical[j]:
            vals, counts = np.unique(ale.X_values[:, j], return_counts=True)
            X_grid[:, j] = vals[np.argmax(counts)]
        else:
            X_grid[:, j] = ale.X_values[:, j].mean()
    X_grid[:, idx_1] = xx1.ravel()
    X_grid[:, idx_2] = xx2.ravel()

    z = ale.f(X_grid).reshape(grid_size, grid_size)

    mesh = ax_heatmap.pcolormesh(xx1, xx2, z, shading="auto", cmap="viridis", alpha=0.5)
    fig.colorbar(mesh, ax=ax_heatmap, label="f(x)")

    for e in edges:
        ax_heatmap.axvline(e, color="black", linestyle="--", alpha=0.3)
    ax_heatmap.set_xlabel(ale.feature_names[idx_1])
    ax_heatmap.set_ylabel(ale.feature_names[idx_2])
    f_name = ale.f.__name__ if hasattr(ale.f, '__name__') else 'model'
    ax_heatmap.set_title(f"Heatmap of {f_name}")


def _plot_g_value_curves(ale, idx_1, edges, g_vals, L, colors, ax_effect):
    """Right subplot: centered g-value curve per path."""
    for l in range(L):
        ax_effect.plot(edges, g_vals[:, l], color=colors[l], label=f"Path {l}")
    for e in edges:
        ax_effect.axvline(e, color="black", linestyle="--", alpha=0.3)
    ax_effect.axhline(0, color="black", alpha=0.2)
    ax_effect.set_xlabel(ale.feature_names[idx_1])
    ax_effect.set_ylabel("Centered g-value")
    ax_effect.set_title(f"G-value curves for {ale.feature_names[idx_1]}")
    ax_effect.legend(loc="best", fontsize="small")


def plot_paths_summary(ale, feature_1, feature_2, figsize=(21, 5), cmap="tab10"):
    """
    Combined three-subplot visualization linking data partitions to g-value
    curves.

    Left: scatter of (feature_1, feature_2) with each path's observations
    colored uniquely; bin centroid trajectories overlaid.
    Middle: heatmap of f over the (feature_1, feature_2) grid, with the same
    path centroids.
    Right: centered g-value curve for each path vs feature_1 edges, using the
    same per-path colors as the left subplot.
    """
    idx_1 = ale._get_feature_index(feature_1)
    idx_2 = ale._get_feature_index(feature_2)

    if idx_1 == idx_2:
        raise ValueError("feature_1 and feature_2 must be different.")
    if idx_1 not in ale.connected_paths:
        raise ValueError(
            f"Connected paths for feature {idx_1} not found. "
            "Run ale_total_vim(method='connected') first."
        )

    paths = ale.connected_paths[idx_1]
    edges = ale.edges[idx_1]
    g_vals = ale.centered_g_values[idx_1].centered_g_values  # (K+1, L) or (K, L)
    L = g_vals.shape[1]

    cmap_obj = plt.get_cmap(cmap)
    colors = [cmap_obj(l / max(L - 1, 1)) for l in range(L)]

    fig, (ax_data, ax_heatmap, ax_effect) = plt.subplots(1, 3, figsize=figsize)
    _plot_paths_scatter(ale, idx_1, idx_2, paths, edges, colors, ax_data, ax_heatmap)
    _plot_f_heatmap(ale, idx_1, idx_2, edges, fig, ax_heatmap)
    _plot_g_value_curves(ale, idx_1, edges, g_vals, L, colors, ax_effect)

    fig.tight_layout()
    return fig, (ax_data, ax_heatmap, ax_effect)
