"""Model ensemble for improved prediction accuracy.

Supports weighted voting and majority voting across multiple models.
"""

from typing import Dict, List, Optional

import numpy as np

from models.base_model import BaseModel


def ensemble_predict(
    models: List[BaseModel],
    X: np.ndarray,
    weights: Optional[Dict[str, float]] = None,
    method: str = "weighted_vote"
) -> np.ndarray:
    """Ensemble predictions from multiple models.

    Args:
        models: list of trained BaseModel instances
        X: features for prediction
        weights: model weights for weighted voting
        method: 'weighted_vote' or 'majority_vote'

    Returns:
        predictions: shape (n_samples,), values 0 or 1
    """
    all_probas = []

    for model in models:
        probas = model.predict_proba(X)  # (n_samples, 2)
        all_probas.append(probas)

    if method == "weighted_vote" and weights:
        # Weighted average of probabilities
        ensemble_proba = np.zeros_like(all_probas[0])
        for model, proba in zip(models, all_probas):
            w = weights.get(model.name, 1.0 / len(models))
            ensemble_proba += w * proba
        return ensemble_proba.argmax(axis=1)

    else:
        # Majority voting
        all_preds = np.array([model.predict(X) for model in models])
        # Majority vote: average and threshold
        avg_proba = np.mean(all_probas, axis=0)
        return avg_proba.argmax(axis=1)
