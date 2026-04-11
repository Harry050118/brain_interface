#!/usr/bin/env python3
"""Train all 3 models on full data and save them to outputs/models/."""

import os
import sys
import time

import yaml

_base = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, _base)
sys.path.insert(0, os.path.join(_base, "models"))

from utils import set_seed, setup_logging
from data_loader import load_train_data
from features import extract_de_batch

start = time.time()

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

model_dir = cfg["output"]["model_dir"]
os.makedirs(model_dir, exist_ok=True)

# 1. SVM
logger.info("\n--- Training SVM ---")
from models.svm_model import SVMModel
svm = SVMModel(**cfg["svm"])
svm.fit(X_de, y)
svm.save(os.path.join(model_dir, "svm.pkl"))

# 2. DGCNN
logger.info("\n--- Training DGCNN ---")
from models.dgcnn import DGCNNModel
dgcnn = DGCNNModel(
    n_bands=len(cfg["features"]["bands"]),
    **{k: v for k, v in cfg["dgcnn"].items() if k in [
        "hidden_dim", "num_layers", "dropout",
        "learning_rate", "epochs", "batch_size", "weight_decay"
    ]}
)
dgcnn.fit(X_de, y)
dgcnn.save(os.path.join(model_dir, "dgcnn.pt"))

# 3. EEG-Conformer
logger.info("\n--- Training EEG-Conformer ---")
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
conformer.fit(X, y)
conformer.save(os.path.join(model_dir, "conformer.pt"))

logger.info(f"\nAll models saved to {model_dir}")
logger.info(f"Total time: {time.time()-start:.0f}s")
