#!/usr/bin/env python3
"""Run deep learning models (DGCNN and/or EEG-Conformer)."""

import os
import sys
import time
import argparse

import yaml

_base = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, _base)
sys.path.insert(0, os.path.join(_base, "models"))

from utils import set_seed, setup_logging, save_run_summary
from data_loader import load_train_data
from features import extract_de_batch
from train import run_loso
from predict import predict_single_model, save_submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["dgcnn", "conformer", "all"], default="all")
    args = parser.parse_args()

    start_time = time.time()

    config_path = os.path.join(os.path.dirname(__file__), "configs", "config.yaml")
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["random_seed"])
    logger, log_file = setup_logging(cfg["output"]["log_dir"])

    # Load data
    logger.info("Loading training data...")
    X, y, subjects = load_train_data(
        train_dir=cfg["data"]["train_dir"],
        window_size=cfg["signal"]["sample_rate"] * cfg["signal"]["window_size_sec"],
        stride=cfg["signal"]["sample_rate"] * cfg["signal"]["train_stride_sec"],
        clip_sigma=cfg["signal"]["clip_sigma"]
    )
    logger.info(f"Data: {X.shape}")

    # Extract DE features for DGCNN
    logger.info("Extracting DE features...")
    X_de = extract_de_batch(X, bands=cfg["features"]["bands"])
    logger.info(f"DE features: {X_de.shape}")

    _loso_kwargs = {
        "train_dir": cfg["data"]["train_dir"],
        "bands": cfg["features"]["bands"],
        "use_domain_adapt": cfg["domain_adapt"]["use_domain_adapt"],
        "n_eval_subjects": cfg["training"].get("n_eval_subjects"),
        "seed": cfg["training"]["random_seed"],
        "logger": logger,
    }

    results = {}
    models_for_ensemble = []

    # DGCNN
    if args.model in ["dgcnn", "all"]:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: DGCNN")
        logger.info("=" * 60)

        from models.dgcnn import DGCNNModel
        model = DGCNNModel(
            n_bands=len(cfg["features"]["bands"]),
            **{k: v for k, v in cfg["dgcnn"].items() if k in [
                "hidden_dim", "num_layers", "dropout",
                "learning_rate", "epochs", "batch_size", "weight_decay"
            ]}
        )

        mean_acc, subj_acc, per_subj = run_loso(model=model, **_loso_kwargs)
        results["DGCNN"] = mean_acc

        # Retrain on all data
        model_final = DGCNNModel(
            n_bands=len(cfg["features"]["bands"]),
            **{k: v for k, v in cfg["dgcnn"].items() if k in [
                "hidden_dim", "num_layers", "dropout",
                "learning_rate", "epochs", "batch_size", "weight_decay"
            ]}
        )
        model_final.fit(X_de, y)
        models_for_ensemble.append(model_final)

    # EEG Conformer
    if args.model in ["conformer", "all"]:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 3: EEG-Conformer")
        logger.info("=" * 60)

        from models.eeg_conformer import EEGConformerModel
        model = EEGConformerModel(
            n_channels=cfg["signal"]["n_channels"],
            window_size=cfg["signal"]["sample_rate"] * cfg["signal"]["window_size_sec"],
            **{k: v for k, v in cfg["conformer"].items() if k in [
                "hidden_dim", "num_heads", "num_layers", "dropout",
                "learning_rate", "epochs", "batch_size", "weight_decay",
                "time_pool"
            ]}
        )

        mean_acc, subj_acc, per_subj = run_loso(
            model=model, use_raw=True, **_loso_kwargs
        )
        results["EEG-Conformer"] = mean_acc

        # Retrain on all data
        model_final = EEGConformerModel(
            n_channels=cfg["signal"]["n_channels"],
            window_size=cfg["signal"]["sample_rate"] * cfg["signal"]["window_size_sec"],
            **{k: v for k, v in cfg["conformer"].items() if k in [
                "hidden_dim", "num_heads", "num_layers", "dropout",
                "learning_rate", "epochs", "batch_size", "weight_decay",
                "time_pool"
            ]}
        )
        model_final.fit(X, y)
        models_for_ensemble.append(model_final)

    # Save results
    elapsed = time.time() - start_time
    for name, acc in results.items():
        summary_file = save_run_summary(
            cfg["output"]["log_dir"],
            model_name=name,
            accuracy=acc,
            elapsed_sec=elapsed,
        )
        logger.info(f"{name} LOSO Acc: {acc:.4f} ({acc*100:.2f}%)")

    # Save ensemble submission
    if len(models_for_ensemble) > 1:
        from predict import predict_ensemble
        logger.info("\nGenerating ensemble submission...")
        predictions = predict_ensemble(
            models_for_ensemble,
            test_dir=cfg["data"]["test_dir"],
            bands=cfg["features"]["bands"],
            weights=cfg["ensemble"].get("weights"),
            method=cfg["ensemble"].get("method", "weighted_vote")
        )
        save_submission(predictions, cfg["output"]["submission_path"])
    elif len(models_for_ensemble) == 1:
        logger.info("\nGenerating submission...")
        predictions = predict_single_model(
            models_for_ensemble[0],
            test_dir=cfg["data"]["test_dir"],
            bands=cfg["features"]["bands"]
        )
        save_submission(predictions, cfg["output"]["submission_path"])

    logger.info(f"\nDeep learning complete! Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
