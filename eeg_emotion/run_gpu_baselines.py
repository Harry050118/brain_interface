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
from braindecode.models import BIOT, EEGConformer, Labram

_base = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, _base)
sys.path.insert(0, os.path.join(_base, "models"))

from data_loader import load_test_data, load_train_data
from models.eeg_token_transformer import EEGFactorizedTransformer, EEGTokenTransformer
from predict import save_submission
from raw_dataset import RawEEGDataset, compute_channel_stats, iter_unique_subjects, make_data_loader, make_loso_raw_datasets
from train import select_eval_subjects
from utils import set_seed, setup_logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["token_transformer", "factorized_transformer", "biot", "labram", "bd_conformer"],
        default="token_transformer",
    )
    parser.add_argument("--n-eval-subjects", type=int, default=None)
    parser.add_argument("--full-loso", action="store_true")
    parser.add_argument("--skip-loso", action="store_true")
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
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--channel-drop-prob", type=float, default=0.0)
    parser.add_argument("--max-time-shift", type=int, default=0)
    parser.add_argument("--save-submission", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--model-output", default=None)
    parser.add_argument("--final-epochs", type=int, default=None)
    return parser.parse_args()


def build_model(args, cfg):
    window_size = cfg["signal"]["sample_rate"] * cfg["signal"]["window_size_sec"]
    common = dict(
        n_channels=cfg["signal"]["n_channels"],
        window_size=window_size,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    if args.model == "token_transformer":
        return EEGTokenTransformer(num_layers=args.num_layers, **common)
    if args.model == "factorized_transformer":
        return EEGFactorizedTransformer(
            temporal_layers=max(1, args.num_layers // 2),
            channel_layers=args.num_layers,
            **common,
        )
    if args.model == "biot":
        return BIOT(
            n_chans=cfg["signal"]["n_channels"],
            n_times=window_size,
            sfreq=cfg["signal"]["sample_rate"],
            n_outputs=2,
            emb_size=args.embed_dim,
            att_num_heads=args.num_heads,
            n_layers=args.num_layers,
            drop_prob=args.dropout,
        )
    if args.model == "labram":
        return Labram(
            n_chans=cfg["signal"]["n_channels"],
            n_times=window_size,
            sfreq=cfg["signal"]["sample_rate"],
            n_outputs=2,
            patch_size=args.patch_size,
            emb_size=args.embed_dim,
            att_num_heads=args.num_heads,
            n_layers=args.num_layers,
            drop_prob=args.dropout,
            attn_drop_prob=args.dropout,
        )
    if args.model == "bd_conformer":
        return EEGConformer(
            n_chans=cfg["signal"]["n_channels"],
            n_times=window_size,
            sfreq=cfg["signal"]["sample_rate"],
            n_outputs=2,
            att_depth=args.num_layers,
            att_heads=args.num_heads,
            drop_prob=args.dropout,
            att_drop_prob=args.dropout,
        )
    raise ValueError(f"Unknown model: {args.model}")


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
    train_ds, val_ds, _ = make_loso_raw_datasets(
        X,
        y,
        subjects,
        test_subject,
        train_noise_std=args.noise_std,
        train_channel_drop_prob=args.channel_drop_prob,
        train_max_time_shift=args.max_time_shift,
    )
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


def train_final_model(args, cfg, X, y, logger):
    """Train one model on all labeled windows for final test prediction."""
    device = torch.device(args.device)
    stats = compute_channel_stats(X)
    train_ds = RawEEGDataset(
        X,
        y,
        stats,
        noise_std=args.noise_std,
        channel_drop_prob=args.channel_drop_prob,
        max_time_shift=args.max_time_shift,
    )
    train_loader = make_data_loader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    model = build_model(args, cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    final_epochs = args.final_epochs or args.epochs

    for epoch in range(1, final_epochs + 1):
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
        logger.info(f"    final epoch {epoch}/{final_epochs}: train_loss={running_loss / len(train_ds):.4f}")

    return model, stats


def predict_test_raw(model, stats, test_dir, args, logger):
    """Predict all public test trials with a raw EEG model."""
    device = torch.device(args.device)
    model.eval()
    test_data = load_test_data(test_dir)
    predictions = []
    with torch.no_grad():
        for user_id, trials in test_data.items():
            test_ds = RawEEGDataset(trials, np.zeros(len(trials), dtype=np.int64), stats)
            test_loader = make_data_loader(
                test_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            subject_preds = []
            for x, _ in test_loader:
                x = x.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                    logits = model(x)
                subject_preds.extend(logits.argmax(dim=1).detach().cpu().numpy().astype(int).tolist())

            for trial_idx, pred in enumerate(subject_preds, start=1):
                predictions.append((user_id, trial_idx, int(pred)))
            logger.info(f"  {user_id}: {np.asarray(subject_preds, dtype=int)} (trials 1-8)")
    return predictions


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

    if not args.skip_loso:
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
    else:
        logger.info("Skipping LOSO evaluation by request.")

    if args.save_submission:
        output_path = args.output or os.path.join(
            cfg["output"]["model_dir"],
            "..",
            f"submission_{args.model}.xlsx",
        )
        model_output = args.model_output or os.path.join(cfg["output"]["model_dir"], f"{args.model}.pt")
        logger.info("Training final raw EEG model on all labeled subjects...")
        model, stats = train_final_model(args, cfg, X, y, logger)
        predictions = predict_test_raw(model, stats, cfg["data"]["test_dir"], args, logger)
        save_submission(predictions, output_path)
        os.makedirs(os.path.dirname(model_output), exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "model_name": args.model,
                "channel_mean": stats.mean,
                "channel_std": stats.std,
                "args": vars(args),
            },
            model_output,
        )
        logger.info(f"Saved final raw EEG model: {model_output}")

    logger.info(f"GPU baseline complete in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
