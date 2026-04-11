#!/usr/bin/env python3
"""Evaluation utilities for improving the EEG emotion pipeline.

Examples:
  python run_eval.py --feature-set enhanced --svm-grid --n-eval-subjects 10
  python run_eval.py --feature-set enhanced --svm-grid --repeat-seeds 42 7 13 21 84
  python run_eval.py --feature-set enhanced --svm-grid --full-loso
  python run_eval.py --compare-dgcnn --n-eval-subjects 10
"""

import argparse
import os
import sys
import time

import numpy as np
import yaml

_base = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, _base)
sys.path.insert(0, os.path.join(_base, "models"))

from data_loader import load_train_data
from features import extract_feature_batch, extract_de_batch
from models.svm_model import SVMModel
from models.dgcnn import DGCNNModel
from train import run_loso, run_loso_features
from utils import set_seed, setup_logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", choices=["de", "enhanced"], default="enhanced")
    parser.add_argument("--svm-grid", action="store_true", help="Run C/gamma grid search for SVM")
    parser.add_argument("--compare-dgcnn", action="store_true", help="Evaluate the current lightweight DGCNN")
    parser.add_argument("--n-eval-subjects", type=int, default=None,
                        help="Number of LOSO subjects to sample; omit with --full-loso for all 60")
    parser.add_argument("--full-loso", action="store_true", help="Use all subjects for LOSO")
    parser.add_argument("--repeat-seeds", nargs="*", type=int, default=None,
                        help="Repeat sampled LOSO with these seeds")
    return parser.parse_args()


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

    set_seed(cfg["training"]["random_seed"])
    logger, _ = setup_logging(cfg["output"]["log_dir"])

    if args.full_loso:
        n_eval_subjects = None
    elif args.n_eval_subjects is not None:
        n_eval_subjects = args.n_eval_subjects
    else:
        n_eval_subjects = cfg["training"].get("n_eval_subjects")

    if args.repeat_seeds is not None and len(args.repeat_seeds) > 0:
        seeds = args.repeat_seeds
    else:
        seeds = [cfg["training"]["random_seed"]]

    logger.info("Loading training data...")
    X, y, subjects = load_train_data(
        train_dir=cfg["data"]["train_dir"],
        window_size=cfg["signal"]["sample_rate"] * cfg["signal"]["window_size_sec"],
        stride=cfg["signal"]["sample_rate"] * cfg["signal"]["train_stride_sec"],
        clip_sigma=cfg["signal"]["clip_sigma"],
    )
    logger.info(f"Raw windows: {X.shape}")

    if not args.svm_grid and not args.compare_dgcnn:
        args.svm_grid = True

    if args.svm_grid:
        logger.info(f"Extracting {args.feature_set} features for SVM...")
        X_features = extract_feature_batch(
            X,
            bands=cfg["features"]["bands"],
            sample_rate=cfg["signal"]["sample_rate"],
            feature_set=args.feature_set,
        )
        logger.info(f"SVM features: {X_features.shape}")

        Cs = cfg.get("svm_grid", {}).get("C", [cfg["svm"].get("C", 1.0)])
        gammas = cfg.get("svm_grid", {}).get("gamma", [cfg["svm"].get("gamma", "scale")])

        best = None
        for C in Cs:
            for gamma in gammas:
                seed_scores = []
                logger.info("=" * 60)
                logger.info(f"SVM grid: feature_set={args.feature_set}, C={C}, gamma={gamma}")
                for seed in seeds:
                    logger.info(f"Seed {seed}, n_eval_subjects={n_eval_subjects or 'full'}")
                    mean_acc, _, _ = run_loso_features(
                        model_factory=lambda C=C, gamma=gamma: SVMModel(
                            kernel=cfg["svm"].get("kernel", "rbf"),
                            C=C,
                            gamma=gamma,
                            feature_set=args.feature_set,
                        ),
                        X_features=X_features,
                        y_all=y,
                        subjects_all=subjects,
                        n_eval_subjects=n_eval_subjects,
                        seed=seed,
                        logger=logger,
                    )
                    seed_scores.append(mean_acc)

                avg = float(np.mean(seed_scores))
                std = float(np.std(seed_scores))
                logger.info(f"SVM grid result: C={C}, gamma={gamma}, mean={avg:.4f}, std={std:.4f}")
                if best is None or avg > best[0]:
                    best = (avg, std, C, gamma, seed_scores)

        logger.info("=" * 60)
        logger.info(
            f"Best SVM: feature_set={args.feature_set}, C={best[2]}, gamma={best[3]}, "
            f"mean={best[0]:.4f}, std={best[1]:.4f}, scores={best[4]}"
        )

    if args.compare_dgcnn:
        logger.info("=" * 60)
        logger.info("Evaluating lightweight DGCNN with DE features")
        # run_loso extracts DE internally and keeps the same train/test behavior as run_all.py.
        for seed in seeds:
            set_seed(seed)
            model = DGCNNModel(
                n_bands=len(cfg["features"]["bands"]),
                **{k: v for k, v in cfg["dgcnn"].items() if k in [
                    "hidden_dim", "num_layers", "dropout", "learning_rate",
                    "epochs", "batch_size", "weight_decay"
                ]}
            )
            acc, _, _ = run_loso(
                model=model,
                train_dir=cfg["data"]["train_dir"],
                bands=cfg["features"]["bands"],
                use_domain_adapt=cfg["domain_adapt"]["use_domain_adapt"],
                use_raw=False,
                n_eval_subjects=n_eval_subjects,
                seed=seed,
                logger=logger,
            )
            logger.info(f"DGCNN seed={seed}: {acc:.4f} ({acc*100:.2f}%)")

    logger.info(f"Evaluation complete in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
