#!/usr/bin/env python3
"""Run SVM baseline with DE features and LOSO cross-validation."""

import os
import sys
import time

import yaml

# Add src and src/models to path
_base = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, _base)
sys.path.insert(0, os.path.join(_base, "models"))

from utils import set_seed, setup_logging, save_run_summary
from data_loader import load_train_data
from features import extract_de_batch
from models.svm_model import SVMModel
from train import run_loso
from predict import predict_single_model, save_submission


def main():
    start_time = time.time()

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "configs", "config.yaml")
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # Setup
    set_seed(cfg["training"]["random_seed"])
    logger, log_file = setup_logging(cfg["output"]["log_dir"])

    logger.info("=" * 60)
    logger.info("PHASE 1: SVM Baseline with DE Features")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading training data...")
    X, y, subjects = load_train_data(
        train_dir=cfg["data"]["train_dir"],
        window_size=cfg["signal"]["sample_rate"] * cfg["signal"]["window_size_sec"],
        stride=cfg["signal"]["sample_rate"] * cfg["signal"]["train_stride_sec"],
        clip_sigma=cfg["signal"]["clip_sigma"]
    )
    logger.info(f"Data: {X.shape}, labels: {y.shape}")

    # Extract DE features
    logger.info("Extracting DE features...")
    X_de = extract_de_batch(X, bands=cfg["features"]["bands"])
    logger.info(f"DE features: {X_de.shape}")

    # LOSO evaluation
    model = SVMModel(
        kernel=cfg["svm"]["kernel"],
        C=cfg["svm"]["C"],
        gamma=cfg["svm"]["gamma"]
    )

    mean_acc, subj_acc, per_subj = run_loso(
        model=model,
        train_dir=cfg["data"]["train_dir"],
        bands=cfg["features"]["bands"],
        use_domain_adapt=cfg["domain_adapt"]["use_domain_adapt"],
        n_eval_subjects=cfg["training"].get("n_eval_subjects"),
        seed=cfg["training"]["random_seed"],
        logger=logger
    )

    # Save summary
    elapsed = time.time() - start_time
    summary_file = save_run_summary(
        cfg["output"]["log_dir"],
        model_name="SVM_DE_Baseline",
        accuracy=mean_acc,
        elapsed_sec=elapsed,
        log_file=log_file
    )

    logger.info(f"\nRun summary saved to: {summary_file}")
    logger.info(f"Total time: {elapsed:.1f}s")

    # Generate submission
    logger.info("\nGenerating submission...")
    # Retrain on all data for final submission
    model_final = SVMModel(
        kernel=cfg["svm"]["kernel"],
        C=cfg["svm"]["C"],
        gamma=cfg["svm"]["gamma"]
    )
    model_final.fit(X_de, y)

    predictions = predict_single_model(
        model_final,
        test_dir=cfg["data"]["test_dir"],
        bands=cfg["features"]["bands"]
    )

    save_submission(predictions, cfg["output"]["submission_path"])

    logger.info("\nBaseline complete!")


if __name__ == "__main__":
    main()
