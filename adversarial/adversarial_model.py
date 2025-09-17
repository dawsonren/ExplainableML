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

        # in the case that we're only considering numerical columns
        if self.numerical_cols:
            X = X[:, self.numerical_cols]

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
    """SHAP adversarial model.  Generates an adversarial model for SHAP style explainers using the Adversarial Model
        base class.

        Parameters:
        - f : biased model
    - psi : innocuous model
        - perturbation_std : standard deviation of Gaussian noise added to create perturbations
    """

    def __init__(self, f, psi, categorical, perturbation_std=0.3):
        super().__init__(f, psi)
        self.perturbation_std = perturbation_std
        self.categorical = categorical

    def train_ood_classifier(
        self,
        X: pd.DataFrame,
        perturbation_multiplier=30,
        rf_estimators=100,
        estimator=None,
    ):
        all_x, all_y = [], []

        # loop over perturbation data to create larger data set
        for _ in range(perturbation_multiplier):
            perturbed_xtrain = np.random.normal(0, self.perturbation_std, size=X.shape)
            p_train_x = np.vstack((X, X + perturbed_xtrain))
            p_train_y = np.concatenate((np.ones(X.shape[0]), np.zeros(X.shape[0])))

            all_x.append(p_train_x)
            all_y.append(p_train_y)

        all_x = np.vstack(all_x)
        all_y = np.concatenate(all_y)

        if all(self.categorical):
            raise NotImplementedError(
                "We currently only support numerical column data. If your data set is all categorical, consider using SHAP adversarial model."
            )

        # generate perturbation detection model as RF
        xtrain = all_x[:, self.numerical_cols]
        xtrain, xtest, ytrain, ytest = train_test_split(xtrain, all_y, test_size=0.2)

        if estimator is not None:
            self.perturbation_identifier = estimator.fit(xtrain, ytrain)
        else:
            self.perturbation_identifier = RandomForestClassifier(
                n_estimators=rf_estimators
            ).fit(xtrain, ytrain)

        ypred = self.perturbation_identifier.predict(xtest)
        self.ood_training_task_ability = (ytest, ypred)

        return self
