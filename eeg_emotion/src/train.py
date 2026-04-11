"""Leave-One-Subject-Out (LOSO) cross-validation training loop.

Trains and evaluates models by iteratively leaving one subject out
as validation, training on all others. Supports stratified subject
sampling for faster evaluation.
"""

import os
import time
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm

from data_loader import load_train_data
from features import extract_de_batch, DEFAULT_BANDS
from utils import save_run_summary
from domain_adapt import apply_coral_for_loso


def run_loso(
    model,
    train_dir: str,
    bands: Dict[str, Tuple[float, float]] = None,
    use_domain_adapt: bool = False,
    use_raw: bool = False,
    n_eval_subjects: int = None,
    seed: int = 42,
    logger: logging.Logger = None
) -> Tuple[float, List[float], Dict[str, float]]:
    """Run LOSO cross-validation.

    Args:
        model: a BaseModel instance
        train_dir: path to 训练集 directory
        bands: frequency band definitions
        use_domain_adapt: whether to apply CORAL alignment
        use_raw: if True, use raw EEG instead of DE features
        n_eval_subjects: number of subjects for evaluation.
            None = all subjects (full 60-fold).
            When set, stratified sampling (HC/DEP proportional).
        seed: random seed for stratified sampling
        logger: optional logger

    Returns:
        mean_accuracy: average accuracy across all subjects
        subject_accuracies: list of per-subject accuracies
        per_subject_results: dict mapping subject_id -> accuracy
    """
    if bands is None:
        bands = DEFAULT_BANDS

    log = logger or logging.getLogger("eeg_emotion")
    log.info("=" * 60)
    log.info(f"Starting LOSO evaluation for: {model.name}")
    log.info("=" * 60)

    # Load all training data
    X_all, y_all, subjects_all = load_train_data(train_dir)
    log.info(f"Loaded {len(X_all)} windows from {len(set(subjects_all))} subjects")

    # Extract DE features or use raw
    if use_raw:
        X_features = X_all  # (n_windows, 30, 2500)
    else:
        X_features = extract_de_batch(X_all, bands)
        log.info(f"DE features shape: {X_features.shape}")

    # Get unique subjects
    unique_subjects = sorted(set(subjects_all))

    # Stratified sampling: select a subset of subjects for evaluation
    if n_eval_subjects is not None and n_eval_subjects < len(unique_subjects):
        rng = np.random.RandomState(seed)

        # Split subjects by class
        dep_subjects = [s for s in unique_subjects if s.startswith("DEP")]
        hc_subjects = [s for s in unique_subjects if s.startswith("HC")]

        # Proportional allocation: 20 DEP : 40 HC = 1 : 2
        n_dep = max(1, int(n_eval_subjects * len(dep_subjects) / len(unique_subjects)))
        n_hc = n_eval_subjects - n_dep

        # Clamp to available
        n_dep = min(n_dep, len(dep_subjects))
        n_hc = min(n_hc, len(hc_subjects))

        # Stratified sample
        eval_dep = rng.choice(dep_subjects, n_dep, replace=False).tolist()
        eval_hc = rng.choice(hc_subjects, n_hc, replace=False).tolist()
        eval_subjects = sorted(eval_dep + eval_hc)

        log.info(f"Stratified LOSO: {len(eval_subjects)} subjects "
                 f"({n_hc} HC + {n_dep} DEP) out of {len(unique_subjects)} total")
        log.info(f"Eval subjects: {eval_subjects}")
    else:
        eval_subjects = unique_subjects
        log.info(f"Full LOSO: {len(unique_subjects)} subjects")

    n_subjects = len(eval_subjects)

    per_subject_results = {}
    subject_accuracies = []

    for i, test_subject in enumerate(eval_subjects):
        # Split
        test_mask = np.array([s == test_subject for s in subjects_all])
        train_mask = ~test_mask

        X_train = X_features[train_mask]
        y_train = y_all[train_mask]
        X_val = X_features[test_mask]
        y_val = y_all[test_mask]

        # Domain adaptation: align source to target distribution
        if use_domain_adapt and not use_raw:
            X_train = apply_coral_for_loso(X_train, y_train, X_val)

        # Train and evaluate
        model.fit(X_train, y_train, X_test=X_val)
        preds = model.predict(X_val)
        acc = (preds == y_val).mean()

        subject_accuracies.append(acc)
        per_subject_results[test_subject] = acc

        step = max(1, n_subjects // 10)
        if (i + 1) % step == 0 or i == n_subjects - 1:
            log.info(f"  [{i+1}/{n_subjects}] {test_subject}: acc={acc:.4f}")

    mean_acc = np.mean(subject_accuracies)
    std_acc = np.std(subject_accuracies)

    log.info(f"\n{'='*60}")
    log.info(f"Model: {model.name}")
    log.info(f"LOSO Mean Accuracy: {mean_acc:.4f} ({mean_acc*100:.2f}%)")
    log.info(f"LOSO Std: {std_acc:.4f}")
    log.info(f"{'='*60}")

    return mean_acc, subject_accuracies, per_subject_results


def select_eval_subjects(subjects_all, n_eval_subjects=None, seed=42):
    """Select LOSO evaluation subjects with the same stratified rule as run_loso."""
    unique_subjects = sorted(set(subjects_all))
    if n_eval_subjects is None or n_eval_subjects >= len(unique_subjects):
        return unique_subjects

    rng = np.random.RandomState(seed)
    dep_subjects = [s for s in unique_subjects if s.startswith("DEP")]
    hc_subjects = [s for s in unique_subjects if s.startswith("HC")]

    n_dep = max(1, int(n_eval_subjects * len(dep_subjects) / len(unique_subjects)))
    n_hc = n_eval_subjects - n_dep
    n_dep = min(n_dep, len(dep_subjects))
    n_hc = min(n_hc, len(hc_subjects))

    eval_dep = rng.choice(dep_subjects, n_dep, replace=False).tolist()
    eval_hc = rng.choice(hc_subjects, n_hc, replace=False).tolist()
    return sorted(eval_dep + eval_hc)


def run_loso_features(
    model_factory,
    X_features: np.ndarray,
    y_all: np.ndarray,
    subjects_all,
    n_eval_subjects: int = None,
    seed: int = 42,
    logger: logging.Logger = None,
):
    """Run LOSO over precomputed features.

    This is useful for SVM grid search, because feature extraction is the slow
    part and should not be repeated for every C/gamma pair.
    """
    log = logger or logging.getLogger("eeg_emotion")
    subjects_array = np.asarray(subjects_all)
    eval_subjects = select_eval_subjects(subjects_all, n_eval_subjects, seed)
    subject_accuracies = []
    per_subject_results = {}

    pbar = tqdm(enumerate(eval_subjects), total=len(eval_subjects), desc="LOSO subjects", unit="subj")
    for i, test_subject in pbar:
        test_mask = subjects_array == test_subject
        train_mask = ~test_mask

        model = model_factory()
        model.fit(X_features[train_mask], y_all[train_mask])
        preds = model.predict(X_features[test_mask])
        acc = (preds == y_all[test_mask]).mean()
        subject_accuracies.append(acc)
        per_subject_results[test_subject] = float(acc)
        
        pbar.set_postfix({"subject": test_subject, "acc": f"{acc:.4f}"})

        step = max(1, len(eval_subjects) // 10)
        if (i + 1) % step == 0 or i == len(eval_subjects) - 1:
            log.info(f"  [{i+1}/{len(eval_subjects)}] {test_subject}: acc={acc:.4f}")

    mean_acc = float(np.mean(subject_accuracies))
    std_acc = float(np.std(subject_accuracies))
    log.info(f"LOSO Mean Accuracy: {mean_acc:.4f} ({mean_acc*100:.2f}%), std={std_acc:.4f}")
    return mean_acc, subject_accuracies, per_subject_results
