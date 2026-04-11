"""SVM baseline model for EEG emotion classification using DE features."""

import pickle
from typing import Optional

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from base_model import BaseModel


class SVMModel(BaseModel):
    """SVM classifier with DE features.

    Pipeline: StandardScaler -> SVM (RBF kernel)
    """

    def __init__(self, kernel: str = "rbf", C: float = 1.0,
                 gamma: str = "scale", feature_set: str = "de"):
        super().__init__(name="SVM_DE")
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.feature_set = feature_set

        self.scaler = StandardScaler()
        self.svm = SVC(
            kernel=kernel, C=C, gamma=gamma,
            probability=True, random_state=42
        )

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_test: Optional[np.ndarray] = None) -> None:
        """Train SVM on DE features.

        Args:
            X: DE features, shape (n_samples, n_de_features)
            y: labels, shape (n_samples,)
            X_test: ignored (SVM doesn't support domain adaptation directly)
        """
        X_scaled = self.scaler.fit_transform(X)
        self.svm.fit(X_scaled, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels.

        Args:
            X: DE features, shape (n_samples, n_de_features)

        Returns:
            predictions: shape (n_samples,), values 0 or 1
        """
        X_scaled = self.scaler.transform(X)
        return self.svm.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: DE features, shape (n_samples, n_de_features)

        Returns:
            probabilities: shape (n_samples, 2)
        """
        X_scaled = self.scaler.transform(X)
        return self.svm.predict_proba(X_scaled)

    def save(self, path: str) -> None:
        """Save SVM model and scaler."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.svm,
                "scaler": self.scaler,
                "name": self.name,
                "feature_set": self.feature_set,
            }, f)
        print(f"  Saved SVM to: {path}")

    @classmethod
    def load(cls, path: str) -> "SVMModel":
        """Load SVM model and scaler."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls(feature_set=data.get("feature_set", "de"))
        model.svm = data["model"]
        model.scaler = data["scaler"]
        model.is_fitted = True
        return model

    def fit_with_history(self, X: np.ndarray, y: np.ndarray,
                         X_test: Optional[np.ndarray] = None) -> dict:
        """Train SVM and return per-epoch metrics (1 epoch = 1 fit for SVM).

        Returns:
            history: dict with 'loss', 'train_acc', 'test_acc'
        """
        X_scaled = self.scaler.fit_transform(X)
        self.svm.fit(X_scaled, y)
        self.is_fitted = True

        train_preds = self.svm.predict(X_scaled)
        train_acc = (train_preds == y).mean()

        test_acc = 0.0
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            test_preds = self.svm.predict(X_test_scaled)
            test_acc = (test_preds == y_test).mean() if y_test is not None else 0.0

        return {"loss": [0.0], "train_acc": [train_acc], "test_acc": [test_acc]}
