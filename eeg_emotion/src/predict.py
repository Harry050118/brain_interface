"""Prediction pipeline for test set submission generation.

Generates submission.xlsx with columns: user_id, trial_id, Emotion_label
"""

import os
import logging
from typing import Dict, List, Optional

import numpy as np
import openpyxl

from data_loader import load_test_data
from features import extract_de_batch, DEFAULT_BANDS
from models.base_model import BaseModel


def predict_single_model(
    model: BaseModel,
    test_dir: str,
    bands: Dict = None,
    use_raw: bool = None
) -> List[tuple]:
    """Predict emotions for all test subjects using a single model.

    Args:
        model: trained BaseModel
        test_dir: path to 公开测试集 directory
        bands: frequency band definitions
        use_raw: if None, auto-detect from model.use_raw

    Returns:
        predictions: list of (user_id, trial_id, emotion_label) tuples
    """
    if bands is None:
        bands = DEFAULT_BANDS
    if use_raw is None:
        use_raw = getattr(model, "use_raw", False)

    log = logging.getLogger("eeg_emotion")
    test_data = load_test_data(test_dir)

    predictions = []

    for user_id, trials in test_data.items():
        # trials: shape (8, 30, 2500)
        if use_raw:
            X = trials
        else:
            X = extract_de_batch(trials, bands)

        preds = model.predict(X)  # (8,)

        for trial_idx, pred in enumerate(preds, start=1):
            predictions.append((user_id, trial_idx, int(pred)))

        log.info(f"  {user_id}: {preds} (trials 1-8)")

    return predictions


def save_submission(
    predictions: List[tuple],
    output_path: str
):
    """Save predictions to submission Excel file.

    Args:
        predictions: list of (user_id, trial_id, emotion_label) tuples
        output_path: path to save submission.xlsx
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"

    # Header
    ws.append(["user_id", "trial_id", "Emotion_label"])

    # Data
    for user_id, trial_id, label in predictions:
        ws.append([user_id, trial_id, label])

    wb.save(output_path)
    print(f"Submission saved to: {output_path}")
    print(f"Total predictions: {len(predictions)}")


def predict_ensemble(
    models: List[BaseModel],
    test_dir: str,
    bands: Dict = None,
    use_raw: bool = None,
    weights: Dict[str, float] = None,
    method: str = "weighted_vote"
) -> List[tuple]:
    """Predict using ensemble of models.

    Each model receives its preferred input type (raw EEG or DE features)
    based on its `use_raw` attribute.
    """
    from ensemble import ensemble_predict

    if bands is None:
        bands = DEFAULT_BANDS

    log = logging.getLogger("eeg_emotion")
    test_data = load_test_data(test_dir)

    predictions = []

    for user_id, trials in test_data.items():
        # Collect per-model probabilities
        all_probas = []
        for model in models:
            model_use_raw = getattr(model, "use_raw", False)
            if model_use_raw:
                X = trials
            else:
                X = extract_de_batch(trials, bands)
            probas = model.predict_proba(X)
            all_probas.append((model, probas))

        preds = ensemble_predict_from_probas(all_probas, weights, method)

        for trial_idx, pred in enumerate(preds, start=1):
            predictions.append((user_id, trial_idx, int(pred)))

        model_names = [m.name for m in models]
        log.info(f"  {user_id}: ensemble={preds}")

    return predictions


def ensemble_predict_from_probas(
    all_probas: List[tuple],
    weights: Dict[str, float] = None,
    method: str = "weighted_vote"
) -> np.ndarray:
    """Ensemble from pre-computed probabilities.

    Args:
        all_probas: list of (model, proba_array) tuples
        weights: model weights for weighted voting
        method: 'weighted_vote' or 'majority_vote'

    Returns:
        predictions: shape (n_samples,)
    """
    if method == "weighted_vote" and weights:
        ensemble_proba = np.zeros_like(all_probas[0][1])
        for model, proba in all_probas:
            w = weights.get(model.name, 1.0 / len(all_probas))
            ensemble_proba += w * proba
        return ensemble_proba.argmax(axis=1)
    else:
        avg_proba = np.mean([p for _, p in all_probas], axis=0)
        return avg_proba.argmax(axis=1)
