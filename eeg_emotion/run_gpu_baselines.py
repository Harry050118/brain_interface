#!/usr/bin/env python3
"""GPU-oriented raw EEG baseline runner."""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

_base = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, _base)
sys.path.insert(0, os.path.join(_base, "models"))

from data_loader import load_train_data
from models.eeg_token_transformer import EEGTokenTransformer
from raw_dataset import iter_unique_subjects, make_data_loader, make_loso_raw_datasets
from train import select_eval_subjects
from utils import set_seed, setup_logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["token_transformer"], default="token_transformer")
    parser.add_argument("--n-eval-subjects", type=int, default=None)
    parser.add_argument("--full-loso", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embed-dim", type=int, default=48)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patch-size", type=int, default=125)
    parser.add_argument("--patch-stride", type=int, default=125)
    return parser.parse_args()


def build_model(args, cfg):
    window_size = cfg["signal"]["sample_rate"] * cfg["signal"]["window_size_sec"]
    return EEGTokenTransformer(
        n_channels=cfg["signal"]["n_channels"],
        window_size=window_size,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    losses = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            losses.append(float(criterion(logits, y).item()))
            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum().item())
            total += int(y.numel())
    return correct / max(1, total), float(np.mean(losses)) if losses else 0.0


def train_fold(args, cfg, X, y, subjects, test_subject, logger):
    device = torch.device(args.device)
    train_ds, val_ds, _ = make_loso_raw_datasets(X, y, subjects, test_subject)
    train_loader = make_data_loader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = make_data_loader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(args, cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    best_acc = 0.0
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y_batch in train_loader:
            x = x.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * int(y_batch.numel())

        val_acc, val_loss = evaluate(model, val_loader, device)
        train_loss = running_loss / max(1, len(train_ds))
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 5) == 0:
            logger.info(
                f"    epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
        if stale_epochs >= args.patience:
            break

    return best_acc, best_epoch


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

    seed = args.seed if args.seed is not None else cfg["training"]["random_seed"]
    set_seed(seed)
    logger, _ = setup_logging(cfg["output"]["log_dir"])
    logger.info("=" * 60)
    logger.info(f"GPU baseline: {args.model}, device={args.device}, amp={args.amp}")
    logger.info("=" * 60)

    X, y, subjects = load_train_data(
        train_dir=cfg["data"]["train_dir"],
        window_size=cfg["signal"]["sample_rate"] * cfg["signal"]["window_size_sec"],
        stride=cfg["signal"]["sample_rate"] * cfg["signal"]["train_stride_sec"],
        clip_sigma=cfg["signal"]["clip_sigma"],
    )
    logger.info(f"Raw windows: {X.shape}, subjects={len(iter_unique_subjects(subjects))}")

    if args.full_loso:
        eval_subjects = iter_unique_subjects(subjects)
    else:
        n_eval_subjects = args.n_eval_subjects or cfg["training"].get("n_eval_subjects")
        eval_subjects = select_eval_subjects(subjects, n_eval_subjects, seed)
    logger.info(f"Eval subjects: {list(eval_subjects)}")

    scores = []
    for i, test_subject in enumerate(eval_subjects, start=1):
        acc, best_epoch = train_fold(args, cfg, X, y, subjects, test_subject, logger)
        scores.append(acc)
        logger.info(f"  [{i}/{len(eval_subjects)}] {test_subject}: best_acc={acc:.4f} epoch={best_epoch}")

    mean_acc = float(np.mean(scores))
    std_acc = float(np.std(scores))
    logger.info(f"GPU baseline LOSO Mean Accuracy: {mean_acc:.4f} ({mean_acc*100:.2f}%), std={std_acc:.4f}")
    logger.info(f"GPU baseline complete in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
