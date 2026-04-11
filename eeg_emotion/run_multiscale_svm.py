#!/usr/bin/env python3
"""Train and submit a strict 5s/10s SVM feature ensemble.

This script intentionally does not use 15s windows: each public/private test
trial is 10 seconds, so 15s windows would cross trial boundaries and leak mixed
emotion content. For 5s models, each 10s test trial is split into two non-
overlapping 5s sub-windows and their probabilities are averaged back to the
trial level.
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import yaml

_base = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, _base)
sys.path.insert(0, os.path.join(_base, "models"))

from data_loader import load_train_data, load_test_data
from features import extract_feature_batch
from models.svm_model import SVMModel
from predict import save_submission
from train import run_loso_features
from utils import set_seed, setup_logging


@dataclass
class TrainedMember:
    scale_sec: int
    feature_set: str
    weight: float
    model: SVMModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", nargs="+", type=int, default=[5, 10], choices=[5, 10],
                        help="Window sizes in seconds. Only 5 and 10 are valid for 10s test trials.")
    parser.add_argument("--feature-sets", nargs="+", default=["enhanced", "riemannian"],
                        choices=["de", "enhanced", "riemannian", "hybrid"])
    parser.add_argument("--C", type=float, default=0.1)
    parser.add_argument("--gamma", default="scale")
    parser.add_argument("--n-eval-subjects", type=int, default=None)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--full-loso", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def absolutize_paths(cfg: Dict, base_dir: str) -> Dict:
    for key in ["train_dir", "test_dir"]:
        if not os.path.isabs(cfg["data"][key]):
            cfg["data"][key] = os.path.abspath(os.path.join(base_dir, cfg["data"][key]))
    for key in ["model_dir", "log_dir", "submission_path"]:
        if not os.path.isabs(cfg["output"][key]):
            cfg["output"][key] = os.path.abspath(os.path.join(base_dir, cfg["output"][key]))
    return cfg


def load_scale_training(cfg: Dict, scale_sec: int):
    sample_rate = cfg["signal"]["sample_rate"]
    window_size = sample_rate * scale_sec
    stride = window_size // 2
    return load_train_data(
        train_dir=cfg["data"]["train_dir"],
        window_size=window_size,
        stride=stride,
        clip_sigma=cfg["signal"]["clip_sigma"],
    )


def split_trials_to_scale(trials: np.ndarray, scale_sec: int, sample_rate: int) -> Tuple[np.ndarray, List[slice]]:
    """Split 10s test trials into scale windows without crossing trial boundaries."""
    trial_len = trials.shape[-1]
    window_size = sample_rate * scale_sec
    if window_size > trial_len:
        raise ValueError(f"scale_sec={scale_sec} exceeds test trial length")

    windows = []
    trial_slices = []
    for trial in trials:
        start_idx = len(windows)
        for start in range(0, trial_len - window_size + 1, window_size):
            windows.append(trial[:, start:start + window_size])
        trial_slices.append(slice(start_idx, len(windows)))
    return np.asarray(windows, dtype=np.float32), trial_slices


def train_member(cfg: Dict, scale_sec: int, feature_set: str, C: float, gamma, logger,
                 n_eval_subjects=None, skip_eval=False) -> TrainedMember:
    logger.info("=" * 60)
    logger.info(f"Training member: scale={scale_sec}s, feature_set={feature_set}, C={C}, gamma={gamma}")
    X, y, subjects = load_scale_training(cfg, scale_sec)
    logger.info(f"Scale {scale_sec}s raw windows: {X.shape}")

    X_features = extract_feature_batch(
        X,
        bands=cfg["features"]["bands"],
        sample_rate=cfg["signal"]["sample_rate"],
        feature_set=feature_set,
    )
    logger.info(f"Scale {scale_sec}s {feature_set} features: {X_features.shape}")

    if not skip_eval:
        run_loso_features(
            model_factory=lambda: SVMModel(
                kernel=cfg["svm"].get("kernel", "rbf"),
                C=C,
                gamma=gamma,
                feature_set=feature_set,
            ),
            X_features=X_features,
            y_all=y,
            subjects_all=subjects,
            n_eval_subjects=n_eval_subjects,
            seed=cfg["training"]["random_seed"],
            logger=logger,
        )

    model = SVMModel(
        kernel=cfg["svm"].get("kernel", "rbf"),
        C=C,
        gamma=gamma,
        feature_set=feature_set,
    )
    model.fit(X_features, y)
    return TrainedMember(scale_sec=scale_sec, feature_set=feature_set, weight=1.0, model=model)


def predict_multiscale(members: List[TrainedMember], cfg: Dict, logger) -> List[tuple]:
    test_data = load_test_data(
        cfg["data"]["test_dir"],
        window_size=cfg["signal"]["sample_rate"] * 10,
        clip_sigma=cfg["signal"]["clip_sigma"],
    )
    predictions = []

    for user_id, trials in test_data.items():
        trial_proba = np.zeros((len(trials), 2), dtype=np.float64)
        total_weight = 0.0

        for member in members:
            windows, trial_slices = split_trials_to_scale(
                trials,
                scale_sec=member.scale_sec,
                sample_rate=cfg["signal"]["sample_rate"],
            )
            X_features = extract_feature_batch(
                windows,
                bands=cfg["features"]["bands"],
                sample_rate=cfg["signal"]["sample_rate"],
                feature_set=member.feature_set,
            )
            window_proba = member.model.predict_proba(X_features)
            for trial_idx, slc in enumerate(trial_slices):
                trial_proba[trial_idx] += member.weight * window_proba[slc].mean(axis=0)
            total_weight += member.weight

        trial_proba /= max(total_weight, 1e-12)
        labels = trial_proba.argmax(axis=1)
        logger.info(f"{user_id}: multiscale={labels}")
        for trial_idx, label in enumerate(labels, start=1):
            predictions.append((user_id, trial_idx, int(label)))
    return predictions


def main():
    args = parse_args()
    start = time.time()
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, "configs", "config.yaml")
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = absolutize_paths(cfg, base_dir)

    if args.output:
        cfg["output"]["submission_path"] = os.path.abspath(args.output)

    n_eval_subjects = None if args.full_loso else args.n_eval_subjects
    if n_eval_subjects is None and not args.full_loso:
        n_eval_subjects = cfg["training"].get("n_eval_subjects")

    set_seed(cfg["training"]["random_seed"])
    logger, _ = setup_logging(cfg["output"]["log_dir"])
    logger.info("Strict multiscale SVM: scales=%s, feature_sets=%s", args.scales, args.feature_sets)

    members = []
    for scale_sec in args.scales:
        for feature_set in args.feature_sets:
            members.append(train_member(
                cfg=cfg,
                scale_sec=scale_sec,
                feature_set=feature_set,
                C=args.C,
                gamma=args.gamma,
                logger=logger,
                n_eval_subjects=n_eval_subjects,
                skip_eval=args.skip_eval,
            ))

    predictions = predict_multiscale(members, cfg, logger)
    save_submission(predictions, cfg["output"]["submission_path"])
    logger.info(f"Multiscale SVM complete in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
