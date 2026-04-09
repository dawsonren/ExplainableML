import numpy as np
import matplotlib.pyplot as plt

from experiments import Experiment
from matplotlib.patches import Patch

def get_full_bounding_box(Xs):
    """Returns a list of (min, max) tuples, one per feature dimension."""
    d = Xs[0].shape[1]
    return [
        (min(X[:, j].min() for X in Xs), max(X[:, j].max() for X in Xs))
        for j in range(d)
    ]

def create_grid(X, resolution=50):
    assert X.shape[1] == 2, f"create_grid only supports d=2, got d={X.shape[1]}"
    # create 2D grid
    x1_lin = np.linspace(X[:, 0].min(), X[:, 0].max(), resolution)
    x2_lin = np.linspace(X[:, 1].min(), X[:, 1].max(), resolution)
    xx, yy = np.meshgrid(x1_lin, x2_lin)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    return xx, yy, grid

def plot_replication(experiment: Experiment, rng=None):
    # if rng is None, set seed at random
    rng = np.random.default_rng()
    # generate data
    X, y = experiment.sample(rng=rng)

    # fit model
    model = experiment.fit_model(X, y, rng=rng)
    f = model.predict

    # plot training data vs. predictions (3D)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # create grid for surface
    xx, yy, grid = create_grid(X)
    zz_f = f(grid).reshape(xx.shape)
    zz_true = experiment.dgp.signal(grid).reshape(xx.shape)

    # plot surfaces
    surf = ax.plot_surface(xx, yy, zz_f, cmap="viridis", alpha=0.6, linewidth=0, antialiased=True)
    surf_true = ax.plot_surface(xx, yy, zz_true, cmap="plasma", alpha=0.6, linewidth=0, antialiased=True)

    # add a legend (3D surfaces don't create legend entries automatically)
    legend_handles = [
        Patch(facecolor=plt.cm.viridis(0.6), edgecolor="none", label="Model prediction"),
        Patch(facecolor=plt.cm.plasma(0.6), edgecolor="none", label="True signal"),
    ]
    ax.legend(handles=legend_handles, loc="upper left")

    # change view angle
    ax.view_init(elev=20, azim=70)

    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("y")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    fig.colorbar(surf_true, ax=ax, shrink=0.5, aspect=10)
    plt.show()

def plot_variability(data_and_fitted_models, bounding_box):
    x1_low, x1_hi = bounding_box[0]
    x2_low, x2_hi = bounding_box[1]
    x1_lin = np.linspace(x1_low, x1_hi, 50)
    x2_lin = np.linspace(x2_low, x2_hi, 50)
    xx, yy = np.meshgrid(x1_lin, x2_lin)
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    zz_f = np.zeros((len(data_and_fitted_models), grid.shape[0]))

    for i in range(len(data_and_fitted_models)):
        X, y, model = data_and_fitted_models[i]
        f = model.predict
        zz_f[i, :] = f(grid)

    X = data_and_fitted_models[0][0]

    f_variability = zz_f.std(axis=0).reshape(xx.shape)

    # heatmap plot
    fig, ax = plt.subplots()
    im = ax.imshow(
        f_variability,
        origin="lower",
        extent=[x1_low, x1_hi, x2_low, x2_hi],
        cmap="viridis",
        aspect="auto",
    )
    # scatterplot
    ax.scatter(X[:, 0], X[:, 1], color='red', s=20, label='Training data')
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.legend()
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("f(x) stddev")
    plt.show()


