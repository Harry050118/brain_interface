#!/usr/bin/env python3
"""Search sklearn baselines on cached EEG features.

This runner is intentionally separate from the main pipeline. It lets us test
traditional classifiers on the same LOSO split without re-extracting enhanced
features for every model.
"""

import argparse
import hashlib
import os
import pickle
import sys
import time
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import yaml
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

_base = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, _base)
sys.path.insert(0, os.path.join(_base, "models"))

from data_loader import load_train_data
from features import extract_feature_batch
from predict import predict_single_model, save_submission
from train import run_loso_features
from utils import set_seed, setup_logging


class SklearnFeatureModel:
    """Small adapter so sklearn estimators work with predict_single_model."""

    def __init__(self, name: str, estimator, feature_set: str):
        self.name = name
        self.estimator = estimator
        self.feature_set = feature_set

    def fit(self, X: np.ndarray, y: np.ndarray, X_test=None) -> None:
        self.estimator.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X)

        if hasattr(self.estimator, "decision_function"):
            scores = self.estimator.decision_function(X)
            scores = np.asarray(scores)
            if scores.ndim == 1:
                scores = np.column_stack([-scores, scores])
            scores = scores - scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            return exp_scores / exp_scores.sum(axis=1, keepdims=True)

        preds = self.predict(X).astype(int)
        proba = np.zeros((len(preds), 2), dtype=np.float32)
        proba[np.arange(len(preds)), preds] = 1.0
        return proba


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", choices=["de", "enhanced"], default="enhanced")
    parser.add_argument("--n-eval-subjects", type=int, default=None)
    parser.add_argument("--full-loso", action="store_true")
    parser.add_argument("--repeat-seeds", nargs="*", type=int, default=None)
    parser.add_argument("--preset", choices=["quick", "standard"], default="quick")
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Optional candidate names to run, e.g. svm_rbf_C0.03_gscale svm_rbf_C0.1_gscale",
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--save-best-submission", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def cache_key(cfg: Dict, feature_set: str) -> str:
    parts = [
        feature_set,
        str(cfg["signal"]["sample_rate"]),
        str(cfg["signal"]["window_size_sec"]),
        str(cfg["signal"]["train_stride_sec"]),
        str(cfg["signal"]["clip_sigma"]),
        repr(cfg["features"]["bands"]),
        cfg["data"]["train_dir"],
    ]
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"{feature_set}_{digest}.npz"


def load_or_extract_features(cfg: Dict, feature_set: str, use_cache: bool, logger):
    cache_dir = os.path.join(os.path.dirname(__file__), "outputs", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_key(cfg, feature_set))

    if use_cache and os.path.exists(cache_path):
        logger.info(f"Loading cached {feature_set} features: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["X_features"], data["y"], data["subjects"].tolist()

    logger.info("Loading training data...")
    X, y, subjects = load_train_data(
        train_dir=cfg["data"]["train_dir"],
        window_size=cfg["signal"]["sample_rate"] * cfg["signal"]["window_size_sec"],
        stride=cfg["signal"]["sample_rate"] * cfg["signal"]["train_stride_sec"],
        clip_sigma=cfg["signal"]["clip_sigma"],
    )
    logger.info(f"Raw windows: {X.shape}")

    logger.info(f"Extracting {feature_set} features...")
    X_features = extract_feature_batch(
        X,
        bands=cfg["features"]["bands"],
        sample_rate=cfg["signal"]["sample_rate"],
        feature_set=feature_set,
    )
    logger.info(f"Features: {X_features.shape}")

    if use_cache:
        np.savez_compressed(
            cache_path,
            X_features=X_features,
            y=y,
            subjects=np.asarray(subjects, dtype=object),
        )
        logger.info(f"Saved feature cache: {cache_path}")

    return X_features, y, subjects


def candidate_factories(preset: str, seed: int, feature_set: str):
    candidates: List[Tuple[str, Callable[[], SklearnFeatureModel]]] = []

    svm_grid = [
        (0.1, "scale"),
        (0.3, "scale"),
        (1.0, "scale"),
        (0.1, 0.003),
        (0.3, 0.003),
    ]
    if preset == "standard":
        svm_grid.extend([
            (0.03, "scale"),
            (3.0, "scale"),
            (0.1, 0.001),
            (0.3, 0.001),
            (1.0, 0.001),
            (1.0, 0.003),
            (0.1, 0.01),
            (0.3, 0.01),
        ])

    for C, gamma in svm_grid:
        name = f"svm_rbf_C{C}_g{gamma}"
        candidates.append((
            name,
            lambda C=C, gamma=gamma, name=name: SklearnFeatureModel(
                name,
                make_pipeline(
                    StandardScaler(),
                    SVC(C=C, gamma=gamma, kernel="rbf", probability=True, random_state=seed),
                ),
                feature_set,
            ),
        ))

    for C in ([0.1, 1.0, 3.0] if preset == "quick" else [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]):
        name = f"logreg_C{C}"
        candidates.append((
            name,
            lambda C=C, name=name: SklearnFeatureModel(
                name,
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=C,
                        max_iter=3000,
                        solver="lbfgs",
                        class_weight=None,
                        random_state=seed,
                    ),
                ),
                feature_set,
            ),
        ))

    for C in ([0.1, 1.0] if preset == "quick" else [0.03, 0.1, 0.3, 1.0, 3.0]):
        name = f"linear_svc_C{C}"
        candidates.append((
            name,
            lambda C=C, name=name: SklearnFeatureModel(
                name,
                make_pipeline(
                    StandardScaler(),
                    LinearSVC(C=C, dual="auto", max_iter=5000, random_state=seed),
                ),
                feature_set,
            ),
        ))

    candidates.append((
        "ridge",
        lambda: SklearnFeatureModel(
            "ridge",
            make_pipeline(StandardScaler(), RidgeClassifier()),
            feature_set,
        ),
    ))

    tree_sizes = [300] if preset == "quick" else [300, 600]
    for n_estimators in tree_sizes:
        for max_features in ["sqrt", 0.5]:
            name = f"extra_trees_{n_estimators}_mf{max_features}"
            candidates.append((
                name,
                lambda n_estimators=n_estimators, max_features=max_features, name=name: SklearnFeatureModel(
                    name,
                    ExtraTreesClassifier(
                        n_estimators=n_estimators,
                        max_features=max_features,
                        min_samples_leaf=2,
                        class_weight=None,
                        n_jobs=-1,
                        random_state=seed,
                    ),
                    feature_set,
                ),
            ))

    if preset == "standard":
        candidates.append((
            "random_forest_500",
            lambda: SklearnFeatureModel(
                "random_forest_500",
                RandomForestClassifier(
                    n_estimators=500,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=seed,
                ),
                feature_set,
            ),
        ))

    return candidates


def main():
    args = parse_args()
    start = time.time()

    config_path = os.path.join(os.path.dirname(__file__), "configs", "config.yaml")
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_dir = os.path.dirname(__file__)
    for key in ["train_dir", "test_dir"]:
        if not os.path.isabs(cfg["data"][key]):
            cfg["data"][key] = os.path.abspath(os.path.join(base_dir, cfg["data"][key]))
    for key in ["model_dir", "log_dir", "submission_path"]:
        if not os.path.isabs(cfg["output"][key]):
            cfg["output"][key] = os.path.abspath(os.path.join(base_dir, cfg["output"][key]))

    set_seed(cfg["training"]["random_seed"])
    logger, log_file = setup_logging(cfg["output"]["log_dir"])

    n_eval_subjects = None if args.full_loso else (
        args.n_eval_subjects if args.n_eval_subjects is not None else cfg["training"].get("n_eval_subjects")
    )
    seeds = args.repeat_seeds or [cfg["training"]["random_seed"]]

    logger.info("=" * 60)
    logger.info(
        f"Model search: feature_set={args.feature_set}, preset={args.preset}, "
        f"n_eval_subjects={n_eval_subjects or 'full'}, seeds={seeds}"
    )
    logger.info("=" * 60)

    X_features, y, subjects = load_or_extract_features(
        cfg,
        args.feature_set,
        use_cache=not args.no_cache,
        logger=logger,
    )

    results = []
    candidates = candidate_factories(args.preset, cfg["training"]["random_seed"], args.feature_set)
    if args.include:
        include_names = set(args.include)
        candidates = [(name, factory) for name, factory in candidates if name in include_names]
        missing = sorted(include_names - {name for name, _ in candidates})
        if missing:
            raise ValueError(f"Unknown candidate(s) for preset={args.preset}: {missing}")

    for name, factory in candidates:
        seed_scores = []
        logger.info("=" * 60)
        logger.info(f"Candidate: {name}")
        for seed in seeds:
            logger.info(f"Seed {seed}, n_eval_subjects={n_eval_subjects or 'full'}")
            mean_acc, _, _ = run_loso_features(
                model_factory=factory,
                X_features=X_features,
                y_all=y,
                subjects_all=subjects,
                n_eval_subjects=n_eval_subjects,
                seed=seed,
                logger=logger,
            )
            seed_scores.append(mean_acc)

        mean_score = float(np.mean(seed_scores))
        std_score = float(np.std(seed_scores))
        results.append((mean_score, std_score, name, seed_scores, factory))
        logger.info(
            f"Candidate result: {name}, mean={mean_score:.4f}, "
            f"std={std_score:.4f}, scores={seed_scores}"
        )

    results.sort(key=lambda item: item[0], reverse=True)
    logger.info("=" * 60)
    logger.info("Leaderboard")
    for rank, (mean_score, std_score, name, seed_scores, _) in enumerate(results, start=1):
        logger.info(f"{rank:02d}. {name}: mean={mean_score:.4f}, std={std_score:.4f}, scores={seed_scores}")

    best_mean, best_std, best_name, best_scores, best_factory = results[0]
    logger.info(
        f"Best candidate: {best_name}, mean={best_mean:.4f}, "
        f"std={best_std:.4f}, scores={best_scores}"
    )

    if args.save_best_submission:
        output_path = args.output or os.path.join(
            os.path.dirname(__file__),
            "outputs",
            f"submission_{best_name}.xlsx",
        )
        logger.info(f"Training best candidate on all data and saving submission: {output_path}")
        best_model = best_factory()
        best_model.fit(X_features, y)
        predictions = predict_single_model(
            best_model,
            test_dir=cfg["data"]["test_dir"],
            bands=cfg["features"]["bands"],
        )
        save_submission(predictions, output_path)

        model_path = os.path.join(os.path.dirname(__file__), "outputs", "models", f"{best_name}.pkl")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        logger.info(f"Saved best model: {model_path}")

    logger.info(f"Model search complete in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
