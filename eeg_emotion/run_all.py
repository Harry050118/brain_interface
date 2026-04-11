#!/usr/bin/env python3
"""Full pipeline: SVM Baseline → DGCNN → EEG-Conformer → Ensemble."""

import os
import sys
import time

import yaml

_base = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, _base)
sys.path.insert(0, os.path.join(_base, "models"))

from utils import set_seed, setup_logging, save_run_summary
from data_loader import load_train_data
from features import extract_de_batch
from train import run_loso
from predict import predict_single_model, predict_ensemble, save_submission


def main():
    start_time = time.time()

    config_path = os.path.join(os.path.dirname(__file__), "configs", "config.yaml")
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["random_seed"])
    logger, log_file = setup_logging(cfg["output"]["log_dir"])

    logger.info("=" * 60)
    logger.info("FULL PIPELINE: SVM → DGCNN → Conformer → Ensemble")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading training data...")
    X, y, subjects = load_train_data(
        train_dir=cfg["data"]["train_dir"],
        window_size=cfg["signal"]["sample_rate"] * cfg["signal"]["window_size_sec"],
        stride=cfg["signal"]["sample_rate"] * cfg["signal"]["train_stride_sec"],
        clip_sigma=cfg["signal"]["clip_sigma"]
    )
    logger.info(f"Data: {X.shape}")
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

    all_models = []

    # Phase 1: SVM
    logger.info("\n--- Phase 1: SVM Baseline ---")
    from models.svm_model import SVMModel
    svm = SVMModel(**cfg["svm"])
    acc_svm, _, _ = run_loso(svm, **_loso_kwargs)
    svm.fit(X_de, y)
    svm_path = os.path.join(cfg["output"]["model_dir"], "svm.pkl")
    svm.save(svm_path)
    all_models.append(svm)

    # Phase 2: DGCNN
    logger.info("\n--- Phase 2: DGCNN ---")
    from models.dgcnn import DGCNNModel
    dgcnn = DGCNNModel(
        n_bands=len(cfg["features"]["bands"]),
        **{k: v for k, v in cfg["dgcnn"].items() if k in [
            "hidden_dim", "num_layers", "dropout",
            "learning_rate", "epochs", "batch_size", "weight_decay"
        ]}
    )
    acc_dgcnn, _, _ = run_loso(dgcnn, **_loso_kwargs)
    dgcnn.fit(X_de, y)
    dgcnn_path = os.path.join(cfg["output"]["model_dir"], "dgcnn.pt")
    dgcnn.save(dgcnn_path)
    all_models.append(dgcnn)

    # Phase 3: EEG Conformer
    logger.info("\n--- Phase 3: EEG-Conformer ---")
    from models.eeg_conformer import EEGConformerModel
    conformer = EEGConformerModel(
        n_channels=cfg["signal"]["n_channels"],
        window_size=cfg["signal"]["sample_rate"] * cfg["signal"]["window_size_sec"],
        **{k: v for k, v in cfg["conformer"].items() if k in [
            "hidden_dim", "num_heads", "num_layers", "dropout",
            "learning_rate", "epochs", "batch_size", "weight_decay",
            "time_pool"
        ]}
    )
    acc_conf, _, _ = run_loso(conformer, use_raw=True, **_loso_kwargs)
    conformer.fit(X, y)
    conf_path = os.path.join(cfg["output"]["model_dir"], "conformer.pt")
    conformer.save(conf_path)
    all_models.append(conformer)

    # Summary
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"SVM Baseline:    {acc_svm:.4f} ({acc_svm*100:.2f}%)")
    logger.info(f"DGCNN:           {acc_dgcnn:.4f} ({acc_dgcnn*100:.2f}%)")
    logger.info(f"EEG-Conformer:   {acc_conf:.4f} ({acc_conf*100:.2f}%)")

    # Ensemble submission
    logger.info("\nGenerating ensemble submission...")
    predictions = predict_ensemble(
        all_models,
        test_dir=cfg["data"]["test_dir"],
        bands=cfg["features"]["bands"],
        weights=cfg["ensemble"].get("weights"),
        method=cfg["ensemble"].get("method", "weighted_vote")
    )
    save_submission(predictions, cfg["output"]["submission_path"])

    # Save summary
    summary = (
        f"\n{'='*60}\n"
        f"Full Pipeline Results\n"
        f"{'='*60}\n"
        f"SVM Baseline:    {acc_svm:.4f} ({acc_svm*100:.2f}%)\n"
        f"DGCNN:           {acc_dgcnn:.4f} ({acc_dgcnn*100:.2f}%)\n"
        f"EEG-Conformer:   {acc_conf:.4f} ({acc_conf*100:.2f}%)\n"
        f"Ensemble:        submission generated\n"
        f"Total time:      {elapsed:.1f}s ({elapsed/60:.1f} min)\n"
        f"{'='*60}\n"
    )
    summary_file = save_run_summary(
        cfg["output"]["log_dir"], "Full_Pipeline_Ensemble",
        accuracy=max(acc_svm, acc_dgcnn, acc_conf),
        elapsed_sec=elapsed, log_file=log_file
    )

    logger.info(summary)
    logger.info(f"\nFull pipeline complete! Total time: {elapsed:.1f}s")

    # Verify saved models can be loaded
    logger.info("\n--- Verifying saved models ---")
    svm_loaded = SVMModel.load(svm_path)
    logger.info(f"  SVM loaded OK from {svm_path}")
    dgcnn_loaded = DGCNNModel.load(dgcnn_path)
    logger.info(f"  DGCNN loaded OK from {dgcnn_path}")
    conformer_loaded = EEGConformerModel.load(conf_path)
    logger.info(f"  Conformer loaded OK from {conf_path}")


if __name__ == "__main__":
    main()
