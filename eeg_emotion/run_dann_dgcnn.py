#!/usr/bin/env python3
"""Run domain-adversarial DGCNN for cross-subject EEG emotion recognition."""

import argparse
import os
import sys
import time

import yaml

_base = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, _base)
sys.path.insert(0, os.path.join(_base, "models"))

from data_loader import load_train_data
from features import extract_de_batch
from models.domain_adversarial_dgcnn import DomainAdversarialDGCNNModel
from predict import predict_single_model, save_submission
from dann_train import run_loso_dann_features, train_domain_adversarial_dgcnn
from utils import set_seed, setup_logging, save_run_summary


def _abs_path(base_dir, path):
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(base_dir, path))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-eval-subjects", type=int, default=None)
    parser.add_argument("--full-loso", action="store_true")
    parser.add_argument("--skip-loso", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--domain-loss-weight", type=float, default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def build_model_factory(cfg):
    model_cfg = cfg["dann_dgcnn"]

    def factory(n_domains):
        return DomainAdversarialDGCNNModel(
            n_channels=cfg["signal"]["n_channels"],
            n_bands=len(cfg["features"]["bands"]),
            n_domains=n_domains,
            hidden_dim=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
            domain_hidden_dim=model_cfg["domain_hidden_dim"],
        )

    return factory


def main():
    args = parse_args()
    start = time.time()
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, "configs", "config.yaml")
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["data"]["train_dir"] = _abs_path(base_dir, cfg["data"]["train_dir"])
    cfg["data"]["test_dir"] = _abs_path(base_dir, cfg["data"]["test_dir"])
    cfg["output"]["log_dir"] = _abs_path(base_dir, cfg["output"]["log_dir"])
    cfg["output"]["model_dir"] = _abs_path(base_dir, cfg["output"]["model_dir"])
    cfg["output"]["submission_path"] = _abs_path(base_dir, cfg["output"]["submission_path"])
    if args.output:
        cfg["output"]["submission_path"] = os.path.abspath(args.output)

    set_seed(cfg["training"]["random_seed"])
    logger, log_file = setup_logging(cfg["output"]["log_dir"])
    model_cfg = cfg["dann_dgcnn"]
    epochs = args.epochs if args.epochs is not None else model_cfg["epochs"]
    domain_loss_weight = (
        args.domain_loss_weight
        if args.domain_loss_weight is not None
        else model_cfg["domain_loss_weight"]
    )
    n_eval_subjects = None if args.full_loso else args.n_eval_subjects
    if n_eval_subjects is None and not args.full_loso:
        n_eval_subjects = cfg["training"].get("n_eval_subjects")

    logger.info("=" * 60)
    logger.info("DANN-DGCNN: DE features + subject-domain adversarial training")
    logger.info("=" * 60)
    logger.info("epochs=%s, domain_loss_weight=%s, n_eval_subjects=%s",
                epochs, domain_loss_weight, n_eval_subjects or "full")

    X, y, subjects = load_train_data(
        train_dir=cfg["data"]["train_dir"],
        window_size=cfg["signal"]["sample_rate"] * cfg["signal"]["window_size_sec"],
        stride=cfg["signal"]["sample_rate"] * cfg["signal"]["train_stride_sec"],
        clip_sigma=cfg["signal"]["clip_sigma"],
    )
    logger.info("Raw windows: %s", X.shape)
    X_de = extract_de_batch(X, bands=cfg["features"]["bands"])
    logger.info("DE features: %s", X_de.shape)

    factory = build_model_factory(cfg)
    mean_acc = 0.0
    if not args.skip_loso:
        mean_acc, _, _ = run_loso_dann_features(
            model_factory=factory,
            X_features=X_de,
            y_all=y,
            subjects_all=subjects,
            epochs=epochs,
            batch_size=model_cfg["batch_size"],
            learning_rate=model_cfg["learning_rate"],
            weight_decay=model_cfg["weight_decay"],
            domain_loss_weight=domain_loss_weight,
            n_eval_subjects=n_eval_subjects,
            seed=cfg["training"]["random_seed"],
            logger=logger,
        )

    logger.info("Training final model on all labeled subjects...")
    final_model = factory(len(set(subjects)))
    train_domain_adversarial_dgcnn(
        model=final_model,
        X_train=X_de,
        y_train=y,
        subjects_train=subjects,
        epochs=epochs,
        batch_size=model_cfg["batch_size"],
        learning_rate=model_cfg["learning_rate"],
        weight_decay=model_cfg["weight_decay"],
        domain_loss_weight=domain_loss_weight,
        logger=logger,
    )

    model_path = os.path.join(cfg["output"]["model_dir"], "dann_dgcnn.pt")
    final_model.save(model_path)
    logger.info("Saved final DANN-DGCNN to %s", model_path)

    predictions = predict_single_model(
        final_model,
        test_dir=cfg["data"]["test_dir"],
        bands=cfg["features"]["bands"],
    )
    save_submission(predictions, cfg["output"]["submission_path"])

    elapsed = time.time() - start
    save_run_summary(
        cfg["output"]["log_dir"],
        model_name="DANN-DGCNN",
        accuracy=mean_acc,
        elapsed_sec=elapsed,
        log_file=log_file,
    )
    logger.info("DANN-DGCNN complete in %.1f min", elapsed / 60)


if __name__ == "__main__":
    main()
