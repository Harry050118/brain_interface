"""Base model interface for all EEG emotion classifiers."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseModel(ABC):
    """Abstract base class for EEG emotion classifiers.

    All models must implement fit() and predict() with this interface
    to enable unified LOSO training and ensemble prediction.
    """

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.use_raw = False  # True if model uses raw EEG, False if DE features

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray,
            X_test: Optional[np.ndarray] = None) -> None:
        """Train the model.

        Args:
            X: training features, shape (n_samples, n_features) for DE
               or (n_samples, n_channels, window_size) for raw
            y: training labels, shape (n_samples,)
            X_test: optional unlabeled test-domain data for domain adaptation
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict emotion labels.

        Args:
            X: features, same format as fit()

        Returns:
            predictions: shape (n_samples,), values 0 or 1
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: features

        Returns:
            probabilities: shape (n_samples, 2)
        """
        pass

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: file path to save the model
        """
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "BaseModel":
        """Load model from disk.

        Args:
            path: file path to load the model from

        Returns:
            model: loaded model instance
        """
        raise NotImplementedError
