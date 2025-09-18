import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class AdversarialModel:
    def __init__(self, f, psi):
        """
        Base class for adversarial models.

        Parameters
        - f : biased model
        - psi : innocuous model
        """
        self.f = f
        self.psi = psi
        self.ood_classifier = None

    def predict_proba(self, X: pd.DataFrame, ood_confidence_threshold=0.5):
        if self.ood_classifier is None:
            raise NameError("OOD classifier is not trained yet.")

        pred_f = self.f.predict_proba(X)
        pred_psi = self.psi.predict_proba(X)

        # allow threshold for confidence in perturbation detection
        pred_probs = self.ood_classifier.predict_proba(X)
        ood = (pred_probs[:, 1] >= ood_confidence_threshold).astype(int)

        # if ood == 1, use f's prediction, else use psi's prediction
        return np.where(np.array([ood == 1, ood == 1]).transpose(), pred_f, pred_psi)

    def predict(self, X: pd.DataFrame):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X_test: pd.DataFrame, y_test: pd.Series):
        return np.sum(self.predict(X_test) == y_test) / y_test.size

    def fidelity(self, X: pd.DataFrame):
        return np.sum(self.predict(X) == self.f_obscure.predict(X)) / X.shape[0]


class AdversarialSHAPModel(AdversarialModel):
    def __init__(self, f, psi, categorical, perturbation_std=0.3):
        super().__init__(f, psi)
        self.perturbation_std = perturbation_std
        self.categorical = categorical
        self.ood_classifier = None

    def train_ood_classifier(
        self,
        X: pd.DataFrame,
        perturbation_multiplier=30,
        rf_estimators=100,
        estimator=None,
    ):
        print("Training OOD classifier...")
        n = X.shape[0] * perturbation_multiplier
        # generate augmented data, label perturbed data as 1, original data as 0
        # only perturb continuous features
        x_augmented = np.vstack(
            np.repeat(X.values[:, ~np.array(self.categorical)], perturbation_multiplier, axis=0),
            np.repeat(X.values[:, ~np.array(self.categorical)], perturbation_multiplier, axis=0),
        )
        y_augmented = np.hstack(
            np.zeros(n),
            np.ones(n),
        )
        noise = np.random.normal(
            loc=0,
            scale=self.perturbation_std,
            size=x_augmented[n:, :].shape
        )
        # only add noise to the bottom part of the array (the perturbed data)
        x_augmented[n:, ~np.array(self.categorical)] += noise
        
        xtrain, xtest, ytrain, ytest = train_test_split(x_augmented, y_augmented, test_size=0.2)

        if estimator is not None:
            self.ood_classifier = estimator.fit(xtrain, ytrain)
        else:
            self.ood_classifier = RandomForestClassifier(
                n_estimators=rf_estimators
            ).fit(xtrain, ytrain)
        
        print(f"OOD classifier accuracy: {self.ood_classifier.score(xtest, ytest)}")
