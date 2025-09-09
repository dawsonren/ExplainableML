import numpy as np
import pandas as pd
from tqdm.auto import tqdm


class Explanation:
    def __init__(self, f, X, feature_names=None, categorical=None):
        """
        Initialize an Explanation. This automatically handles
        numpy arrays and pandas DataFrames, and populates
        some useful attributes.

        Parameters:
        - f: The model function that takes a 2D numpy array and returns predictions.
        - X: The input data as a 2D numpy array or pandas DataFrame.
        - feature_names: List of feature names. If None and X is a DataFrame, use its columns.
        - bins: Number of bins for ALE calculation.
        - categorical: List of booleans indicating if each feature is categorical.
                       If None, all features are treated as continuous.
        """
        if not callable(f):
            raise ValueError("f must be a callable function.")
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise ValueError("X must be a 2D numpy array or pandas DataFrame.")
        if categorical is not None and not isinstance(categorical, list):
            raise ValueError("categorical must be a list of booleans or None.")

        self.is_dataframe = isinstance(X, pd.DataFrame)
        self.n, self.d = X.shape
        self.f = self._log_query_points(f)
        self.query_log = set()

        if self.is_dataframe:
            # store the DataFrame and its values
            self.X = X.copy()
            self.X_values = X.values

            # if feature_names is None, use DataFrame columns
            if feature_names is None:
                self.feature_names = X.columns.tolist()
            else:
                raise ValueError("If X is a DataFrame, feature_names must be None.")

            # get categorical features from DataFrame
            self.categorical = [X[col].dtype == "category" for col in X.columns]
        else:
            self.X_values = X.copy()
            if feature_names is None:
                self.feature_names = [f"X{i+1}" for i in range(self.d)]
            else:
                self.feature_names = feature_names
            self.X = pd.DataFrame(X, columns=self.feature_names)

            if categorical is None:
                self.categorical = [False] * self.d
            else:
                if len(categorical) != X.shape[1]:
                    raise ValueError(
                        "Length of categorical must match number of features in X."
                    )
                self.categorical = categorical

    def _log_query_points(self, f):
        """
        Wrap the model function f to log its query points.
        """
        def wrapper(X, log=True):
            # Log the query points
            if log:
                for i in range(X.shape[0]):
                    self.query_log.add(tuple(X[i, :]))
            return f(X)
        return wrapper
    
    def get_query_points(self):
        return np.array(list(self.query_log))

def bin_selection(n):
    # choose closest divisor to sqrt(n)
    # list of divisors of n
    divisors = [i for i in range(1, n + 1) if n % i == 0]
    closest_divisor = min(divisors, key=lambda x: abs(x - np.sqrt(n)))
    return closest_divisor
