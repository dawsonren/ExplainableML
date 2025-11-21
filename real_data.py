"""
Study variability of ALE and SHAP explanations on real data.

3 Datasets:
1. Breast Cancer Wisconsin (Diagnostic) Data Set
2. Statlog (German Credit Data) Data Set
3. Bike Sharing Demand Data Set
"""
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import (
    KFold,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)

# pandas copy warning ignore
pd.options.mode.chained_assignment = None  # default='warn'

from shapley import SHAP
from ale import ALE


# ----------------------------------------------------------------------
# Data loading & cleaning
# ----------------------------------------------------------------------

def load_breast_cancer():
    """
    Load Breast Cancer Wisconsin (Diagnostic) from UCI zip and prepare X, y.

    - Drops ID column
    - y = 1 for malignant (M), 0 for benign (B)
    """
    # wdbc.data has no header
    df = pd.read_csv(
        "data/wdbc.data",
        header=None,
    )

    # Column names: id, diagnosis, then 30 features
    # (mean, se, worst for 10 basic measurements)
    feature_bases = [
        "radius", "texture", "perimeter", "area", "smoothness",
        "compactness", "concavity", "concave_points", "symmetry",
        "fractal_dimension",
    ]
    feature_names = (
        [f"{b}_mean" for b in feature_bases] +
        [f"{b}_se" for b in feature_bases] +
        [f"{b}_worst" for b in feature_bases]
    )
    cols = ["id", "diagnosis"] + feature_names
    df.columns = cols

    X = df.drop(columns=["id", "diagnosis"])
    # Diagnosis: M = malignant, B = benign
    y = (df["diagnosis"] == "M").astype(int)

    return X, y


def load_german_credit():
    """
    Load Statlog (German Credit Data) from UCI zip and prepare X, y.

    - Uses german.data (categorical + numeric attributes)
    - y = 1 for class "good" (1), 0 for "bad" (2)
    """
    # german.data has 20 attributes + class label, whitespace-separated, no header
    df = pd.read_csv(
        "data/german.data",
        header=None,
        delim_whitespace=True,
    )

    # Give simple names: attr_1 ... attr_20, target
    n_features = df.shape[1] - 1
    feature_cols = [f"attr_{i}" for i in range(1, n_features + 1)]
    cols = feature_cols + ["target"]
    df.columns = cols

    X = df[feature_cols]
    y_raw = df["target"].astype(int)

    # Map labels: 1 = good, 2 = bad -> 1 and 0 (or vice versa; we choose this convention)
    y = y_raw.map({1: 1, 2: 0}).astype(int)

    return X, y


def load_bike_sharing():
    """
    Load Bike Sharing Demand from UCI zip and prepare X, y.
    """
    # clean data
    df = pd.read_csv("data/bikesharing.csv")
    df = df.dropna()

    # remove observations with missing feeling temperature or humidity values
    # these observations are recorded as 0 humidity and 0.2424 feeling temperature
    df = df[df["hum"] != 0]
    df = df[(df["atemp"] != 0.2424) | (df["temp"] <= 0.5)]
    # create quarter column
    df["quarter"] = 1 + 4 * df["yr"] + df["mnth"] // 4
    # rename mnth to month, hr to hour
    df = df.rename(columns={"mnth": "month", "hr": "hour", "weathersit": "weather_situation"})
    # keep only relevant columns
    X = df[["quarter", "month", "hour", "holiday", "weekday", "workingday", "weather_situation", "atemp", "hum", "windspeed"]]
    X["holiday"] = X["holiday"].astype("category")
    X["workingday"] = X["workingday"].astype("category")
    y = df["cnt"]

    return X, y


def run_experiments(EXPLAIN_NAME, K_FOLDS, REPLICATIONS, M, K, subsample=None):
    # # 1) Breast Cancer – classification
    # X_bc, y_bc = load_breast_cancer()

    # # 2) German Credit – classification
    # X_gc, y_gc = load_german_credit()

    # 3) Bike Sharing – regression
    if EXPLAIN_NAME == "bike_sharing":
        X_bike, y_bike = load_bike_sharing()

    # subsample data for speed
    if subsample is not None:
        X_bike = X_bike.sample(n=subsample, random_state=42)
        y_bike = y_bike.loc[X_bike.index]

    n, p = X_bike.shape
    est = MLPRegressor(hidden_layer_sizes=(25,), activation='logistic', alpha=0.05, max_iter=1000, random_state=42)

    # provide explanations for x_explain where not part of training data
    # an explanation is a vector of length p
    # since we have K folds, we will have K explanations for each of the n data points
    # we sum over replications and will average at the end
    shap_explanations_sum = np.zeros((K_FOLDS, n, p))
    ale_explanations_sum = np.zeros((K_FOLDS, n, p))

    for r in range(REPLICATIONS):
        print(f"Replication {r + 1}/{REPLICATIONS}")

        shap_explanations = np.zeros((K_FOLDS, n, p))
        ale_explanations = np.zeros((K_FOLDS, n, p))

        # create K folds
        kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
        for k, (train_index, test_index) in enumerate(kf.split(X_bike)):
            print(f"Fold {k + 1}...")

            X_train, _ = X_bike.iloc[train_index], X_bike.iloc[test_index]
            y_train, _ = y_bike.iloc[train_index], y_bike.iloc[test_index]

            # fit model on training data
            est.fit(X_train, y_train)
            f = lambda X: est.predict(X)

            # SHAP explanation
            shap_explainer = SHAP(f, X_train, verbose=False)
            for i in tqdm(range(n)):
                # select single observation to explain
                x_explain = X_bike.iloc[i, :]
                shap_values = shap_explainer.explain_local(x_explain, num_samples=M)
                for j in range(p):
                    shap_explanations[k, i, j] = shap_values[X_bike.columns[j]]

            # ALE explanation
            ale_explainer = ALE(f, X_train, verbose=False, centering="y")
            ale_explainer.explain()
            for i in tqdm(range(n)):
                # select single observation to explain
                x_explain = X_bike.iloc[i, :]
                ale_values = ale_explainer.explain_local(x_explain)
                for j in range(p):
                    ale_explanations[k, i, j] = ale_values[X_bike.columns[j]]

        shap_explanations_sum += shap_explanations
        ale_explanations_sum += ale_explanations

    # average over replications
    shap_explanations_avg = shap_explanations_sum / REPLICATIONS
    ale_explanations_avg = ale_explanations_sum / REPLICATIONS

    slug = f"{EXPLAIN_NAME}_K{K}_M{M}_KF{K_FOLDS}_REP{REPLICATIONS}"

    # save explanations
    np.savez(f"figures/{slug}/real_data_explanations.npz", shap=shap_explanations_avg, ale=ale_explanations_avg)

if __name__ == "__main__":
    SAVE_IMAGE = False
    EXPLAIN_NAME = "bike_sharing"
    K_FOLDS = 5
    REPLICATIONS = 1
    M = 10
    K = 30

    slug = f"{EXPLAIN_NAME}_K{K}_M{M}_KF{K_FOLDS}_REP{REPLICATIONS}"

    explanations_file = f"figures/{slug}/real_data_explanations.npz"

    # check if explanations already exist
    # if so, skip computation
    if os.path.exists(explanations_file):
        print("Explanations already exist. Skipping computation.")
    else:
        # create directory if it doesn't exist
        if not os.path.exists(f"figures/{slug}"):
            os.makedirs(f"figures/{slug}")
        run_experiments(EXPLAIN_NAME, K_FOLDS, REPLICATIONS, M, K)