#!/usr/bin/env python3
"""Run SVM baseline with DE or enhanced EEG features."""

import argparse
import os
import sys
import time

import yaml

_base = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, _base)
sys.path.insert(0, os.path.join(_base, "models"))

from utils import set_seed, setup_logging, save_run_summary
from data_loader import load_train_data
from features import extract_feature_batch
from models.svm_model import SVMModel
from train import run_loso_features
from predict import predict_single_model, save_submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", choices=["de", "enhanced"], default=None)
    parser.add_argument("--C", type=float, default=None)
    parser.add_argument("--gamma", default=None)
    parser.add_argument("--full-loso", action="store_true")
    parser.add_argument("--skip-loso", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    config_path = os.path.join(os.path.dirname(__file__), "configs", "config.yaml")
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_dir = os.path.dirname(__file__)
    for key in ["train_dir", "test_dir"]:
        if not os.path.isabs(cfg["data"][key]):
            cfg["data"][key] = os.path.abspath(os.path.join(base_dir, cfg["data"][key]))

    feature_set = args.feature_set or (
        "enhanced" if cfg.get("evaluation", {}).get("use_enhanced_features", False) else "de"
    )
    svm_C = args.C if args.C is not None else cfg["svm"]["C"]
    svm_gamma = args.gamma if args.gamma is not None else cfg["svm"]["gamma"]
    n_eval_subjects = None if args.full_loso else cfg["training"].get("n_eval_subjects")

    set_seed(cfg["training"]["random_seed"])
    logger, log_file = setup_logging(cfg["output"]["log_dir"])

    logger.info("=" * 60)
    logger.info(f"SVM Baseline: feature_set={feature_set}, C={svm_C}, gamma={svm_gamma}")
    logger.info("=" * 60)

    logger.info("Loading training data...")
    X, y, subjects = load_train_data(
        train_dir=cfg["data"]["train_dir"],
        window_size=cfg["signal"]["sample_rate"] * cfg["signal"]["window_size_sec"],
        stride=cfg["signal"]["sample_rate"] * cfg["signal"]["train_stride_sec"],
        clip_sigma=cfg["signal"]["clip_sigma"],
    )
    logger.info(f"Data: {X.shape}, labels: {y.shape}")

    logger.info(f"Extracting {feature_set} features...")
    X_features = extract_feature_batch(
        X,
        bands=cfg["features"]["bands"],
        sample_rate=cfg["signal"]["sample_rate"],
        feature_set=feature_set,
    )
    logger.info(f"Features: {X_features.shape}")

    mean_acc = 0.0
    if not args.skip_loso:
        mean_acc, _, _ = run_loso_features(
            model_factory=lambda: SVMModel(
                kernel=cfg["svm"]["kernel"],
                C=svm_C,
                gamma=svm_gamma,
                feature_set=feature_set,
            ),
            X_features=X_features,
            y_all=y,
            subjects_all=subjects,
            n_eval_subjects=n_eval_subjects,
            seed=cfg["training"]["random_seed"],
            logger=logger,
        )

    elapsed = time.time() - start_time
    if not args.skip_loso:
        save_run_summary(
            cfg["output"]["log_dir"],
            model_name=f"SVM_{feature_set}",
            accuracy=mean_acc,
            elapsed_sec=elapsed,
            log_file=log_file,
        )

    logger.info("Generating submission...")
    model_final = SVMModel(
        kernel=cfg["svm"]["kernel"],
        C=svm_C,
        gamma=svm_gamma,
        feature_set=feature_set,
    )
    model_final.fit(X_features, y)
    predictions = predict_single_model(
        model_final,
        test_dir=cfg["data"]["test_dir"],
        bands=cfg["features"]["bands"],
    )
    save_submission(predictions, cfg["output"]["submission_path"])
    logger.info(f"SVM baseline complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
