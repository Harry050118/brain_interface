#!/usr/bin/env python3
"""Test CORAL domain adaptation effect.

Runs LOSO evaluation for each model with and without CORAL
and compares results.
"""

import os
import sys

import yaml

_base = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, _base)
sys.path.insert(0, os.path.join(_base, "models"))

from utils import set_seed, setup_logging
from data_loader import load_train_data
from features import extract_de_batch
from train import run_loso

config_path = os.path.join(os.path.dirname(__file__), "configs", "config.yaml")
with open(config_path, encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

set_seed(cfg["training"]["random_seed"])
logger, _ = setup_logging(cfg["output"]["log_dir"])

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
    "n_eval_subjects": cfg["training"].get("n_eval_subjects"),
    "seed": cfg["training"]["random_seed"],
    "logger": logger,
}

results = {}

# --- Without CORAL ---
logger.info("\n" + "=" * 60)
logger.info("WITHOUT CORAL")
logger.info("=" * 60)

# SVM
from models.svm_model import SVMModel
svm = SVMModel(**cfg["svm"])
acc_svm_no, _, _ = run_loso(svm, use_domain_adapt=False, **_loso_kwargs)
results["SVM (no CORAL)"] = acc_svm_no

# DGCNN
from models.dgcnn import DGCNNModel
dgcnn = DGCNNModel(
    n_bands=len(cfg["features"]["bands"]),
    **{k: v for k, v in cfg["dgcnn"].items() if k in [
        "hidden_dim", "num_layers", "dropout",
        "learning_rate", "epochs", "batch_size", "weight_decay"
    ]}
)
acc_dgcnn_no, _, _ = run_loso(dgcnn, use_domain_adapt=False, **_loso_kwargs)
results["DGCNN (no CORAL)"] = acc_dgcnn_no

# Conformer
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
acc_conf_no, _, _ = run_loso(conformer, use_raw=True, use_domain_adapt=False, **_loso_kwargs)
results["Conformer (no CORAL)"] = acc_conf_no

# --- With CORAL ---
logger.info("\n" + "=" * 60)
logger.info("WITH CORAL")
logger.info("=" * 60)

# SVM
svm2 = SVMModel(**cfg["svm"])
acc_svm_yes, _, _ = run_loso(svm2, use_domain_adapt=True, **_loso_kwargs)
results["SVM (with CORAL)"] = acc_svm_yes

# DGCNN
dgcnn2 = DGCNNModel(
    n_bands=len(cfg["features"]["bands"]),
    **{k: v for k, v in cfg["dgcnn"].items() if k in [
        "hidden_dim", "num_layers", "dropout",
        "learning_rate", "epochs", "batch_size", "weight_decay"
    ]}
)
acc_dgcnn_yes, _, _ = run_loso(dgcnn2, use_domain_adapt=True, **_loso_kwargs)
results["DGCNN (with CORAL)"] = acc_dgcnn_yes

# Conformer (CORAL not applicable for raw EEG, skip)
results["Conformer (with CORAL)"] = "N/A (raw EEG)"

# Summary
logger.info("\n" + "=" * 60)
logger.info("CORAL COMPARISON")
logger.info("=" * 60)
for name, acc in results.items():
    if isinstance(acc, float):
        logger.info(f"  {name:30s}: {acc:.4f} ({acc*100:.2f}%)")
    else:
        logger.info(f"  {name:30s}: {acc}")

logger.info(f"\n{'='*60}")
logger.info(f"  SVM delta:      {results['SVM (with CORAL)'] - results['SVM (no CORAL)']:+.4f}")
logger.info(f"  DGCNN delta:    {results['DGCNN (with CORAL)'] - results['DGCNN (no CORAL)']:+.4f}")
logger.info(f"{'='*60}")
