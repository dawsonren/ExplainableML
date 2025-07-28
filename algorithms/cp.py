import numpy as np

# NOTE: I think all ALE is doing is basically CP
# but we're going "along the data" instead of "along the axis".

def ceteris_paribus(f, X, feature_idx, explain_idx, bins=10):
    """
    Calculate Ceteris Paribus (CP) values for a given feature.

    Parameters:
    - f: function to evaluate the model
    - X: numpy array of shape (n, p)
    - feature_idx: 1-based index of the feature
    - explain_idx: 0-based index of the observation to explain
    - bins: number of bins for CP calculation

    Returns:
    - edges: bin edges
    - predictions: CP values at bin edges
    """
    idx = feature_idx - 1  # convert to 0-based index
    x = X[:, idx]

    # equal-width bin edges
    edges = np.linspace(x.min(), x.max(), bins + 1)
    edges[0], edges[-1] = x.min(), x.max()

    # calculate predictions for each edge
    predictions = np.zeros(bins)
    for i in range(bins):
        X_temp = X[explain_idx, :].copy()
        X_temp[idx] = edges[i]
        predictions[i] = f(X_temp)

    return edges, predictions
