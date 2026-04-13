#!/usr/bin/env python3
"""GPU-oriented raw EEG baseline runner."""

import argparse
import inspect
import os
import sys
import time
from copy import deepcopy

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
from raw_dataset import (
    RawEEGDataset,
    compute_channel_stats,
    iter_unique_subjects,
    loso_masks,
    make_data_loader,
    make_loso_raw_datasets,
    standardize_by_window,
)
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
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--ensemble-seeds", default=None, help="Comma-separated seeds for probability averaging.")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embed-dim", type=int, default=48)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--train-stride-sec", type=float, default=None)
    parser.add_argument("--patch-size", type=int, default=125)
    parser.add_argument("--patch-stride", type=int, default=125)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--channel-drop-prob", type=float, default=0.0)
    parser.add_argument("--max-time-shift", type=int, default=0)
    parser.add_argument("--norm-mode", choices=["fold", "window"], default="fold")
    parser.add_argument("--balanced-rank", action="store_true")
    parser.add_argument("--save-submission", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--model-output", default=None)
    parser.add_argument("--final-epochs", type=int, default=None)
    return parser.parse_args()


def parse_ensemble_seeds(seed_text, default_seed):
    """Parse optional comma-separated ensemble seeds."""
    if seed_text is None or str(seed_text).strip() == "":
        return (int(default_seed),)
    seeds = []
    for part in str(seed_text).split(","):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    if not seeds:
        raise ValueError("--ensemble-seeds must contain at least one integer seed")
    return tuple(seeds)


def resolve_eval_seed(args, training_seed):
    """Return the seed used to choose the screening subject subset."""
    return int(training_seed) if args.eval_seed is None else int(args.eval_seed)


def uses_window_normalization(norm_mode: str) -> bool:
    """Return whether data is already normalized window-by-window."""
    return norm_mode == "window"


def standardize_by_norm_mode(X: np.ndarray, norm_mode: str) -> np.ndarray:
    """Apply the requested pre-dataset normalization mode."""
    if norm_mode == "window":
        return standardize_by_window(X)
    return X


def average_probabilities(probability_arrays):
    """Average probability arrays from ensemble members."""
    if not probability_arrays:
        raise ValueError("At least one probability array is required")
    return np.mean(np.stack(probability_arrays, axis=0), axis=0)


def balanced_rank_predictions(probas):
    """Predict exactly half positive samples by positive-class probability rank."""
    probas = np.asarray(probas)
    if probas.ndim != 2 or probas.shape[1] != 2:
        raise ValueError(f"Expected probability shape (n, 2), got {probas.shape}")
    n_positive = probas.shape[0] // 2
    preds = np.zeros(probas.shape[0], dtype=np.int64)
    if n_positive == 0:
        return preds
    positive_rank = np.argsort(probas[:, 1], kind="mergesort")
    preds[positive_rank[-n_positive:]] = 1
    return preds


def make_criterion(label_smoothing=0.0):
    """Create the classification criterion used by GPU baselines."""
    return nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))


def resolve_signal_samples(cfg, train_stride_sec=None):
    """Resolve window and training stride durations to sample counts."""
    sample_rate = cfg["signal"]["sample_rate"]
    stride_sec = cfg["signal"]["train_stride_sec"] if train_stride_sec is None else train_stride_sec
    return int(round(sample_rate * cfg["signal"]["window_size_sec"])), int(round(sample_rate * stride_sec))


def make_bd_conformer_kwargs(args, cfg, window_size, signature_parameters=None):
    """Build EEGConformer kwargs across Braindecode parameter name changes."""
    params = set(signature_parameters or inspect.signature(EEGConformer).parameters)
    kwargs = {
        "n_chans": cfg["signal"]["n_channels"],
        "n_times": window_size,
        "sfreq": cfg["signal"]["sample_rate"],
        "n_outputs": 2,
        "drop_prob": args.dropout,
        "att_drop_prob": args.dropout,
    }
    if "num_layers" in params:
        kwargs["num_layers"] = args.num_layers
    else:
        kwargs["att_depth"] = args.num_layers
    if "num_heads" in params:
        kwargs["num_heads"] = args.num_heads
    else:
        kwargs["att_heads"] = args.num_heads
    return kwargs


def build_model(args, cfg):
    window_size, _ = resolve_signal_samples(cfg, args.train_stride_sec)
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
        return EEGConformer(**make_bd_conformer_kwargs(args, cfg, window_size))
    raise ValueError(f"Unknown model: {args.model}")


def predict_proba_loader(model, loader, device, amp=False):
    """Return class probabilities for all samples in a loader."""
    model.eval()
    probas = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                logits = model(x)
            probas.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    return np.concatenate(probas, axis=0) if probas else np.empty((0, 2), dtype=np.float32)


def evaluate(model, loader, device, amp=False, label_smoothing=0.0, balanced_rank=False):
    model.eval()
    correct = 0
    total = 0
    losses = []
    probas = []
    targets = []
    criterion = make_criterion(label_smoothing)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                logits = model(x)
            losses.append(float(criterion(logits, y).item()))
            proba = torch.softmax(logits, dim=1)
            probas.append(proba.detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())
            preds = proba.argmax(dim=1)
            correct += int((preds == y).sum().item())
            total += int(y.numel())
    if balanced_rank and probas:
        balanced_preds = balanced_rank_predictions(np.concatenate(probas, axis=0))
        target_arr = np.concatenate(targets, axis=0)
        correct = int((balanced_preds == target_arr).sum())
        total = int(target_arr.shape[0])
    return correct / max(1, total), float(np.mean(losses)) if losses else 0.0


def train_fold_model(args, cfg, X, y, subjects, test_subject, logger):
    device = torch.device(args.device)
    if uses_window_normalization(args.norm_mode):
        train_mask, test_mask = loso_masks(subjects, test_subject)
        train_ds = RawEEGDataset(
            X[train_mask],
            y[train_mask],
            None,
            noise_std=args.noise_std,
            channel_drop_prob=args.channel_drop_prob,
            max_time_shift=args.max_time_shift,
        )
        val_ds = RawEEGDataset(X[test_mask], y[test_mask], None)
    else:
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
    criterion = make_criterion(args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    best_acc = 0.0
    best_epoch = 0
    best_state = None
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

        val_acc, val_loss = evaluate(
            model,
            val_loader,
            device,
            amp=args.amp,
            label_smoothing=args.label_smoothing,
            balanced_rank=args.balanced_rank,
        )
        train_loss = running_loss / max(1, len(train_ds))
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
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

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, val_ds, best_acc, best_epoch


def train_fold(args, cfg, X, y, subjects, test_subject, logger):
    _, _, best_acc, best_epoch = train_fold_model(args, cfg, X, y, subjects, test_subject, logger)
    return best_acc, best_epoch


def train_fold_ensemble(args, cfg, X, y, subjects, test_subject, ensemble_seeds, logger):
    """Train multiple seed members for one fold and average validation probabilities."""
    device = torch.device(args.device)
    member_probas = []
    member_summaries = []
    val_targets = None
    for member_idx, member_seed in enumerate(ensemble_seeds, start=1):
        set_seed(member_seed)
        logger.info(f"    ensemble member {member_idx}/{len(ensemble_seeds)} seed={member_seed}")
        model, val_ds, best_acc, best_epoch = train_fold_model(args, cfg, X, y, subjects, test_subject, logger)
        val_loader = make_data_loader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        member_probas.append(predict_proba_loader(model, val_loader, device, amp=args.amp))
        member_summaries.append((member_seed, best_acc, best_epoch))
        if val_targets is None:
            val_targets = val_ds.y

    ensemble_proba = average_probabilities(member_probas)
    preds = balanced_rank_predictions(ensemble_proba) if args.balanced_rank else ensemble_proba.argmax(axis=1)
    ensemble_acc = float(np.mean(preds == val_targets))
    return ensemble_acc, member_summaries


def train_final_model(args, cfg, X, y, logger):
    """Train one model on all labeled windows for final test prediction."""
    device = torch.device(args.device)
    stats = None if uses_window_normalization(args.norm_mode) else compute_channel_stats(X)
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
    criterion = make_criterion(args.label_smoothing)
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


def predict_test_raw_proba(model, stats, test_dir, args, logger):
    """Predict class probabilities for all public test trials with a raw EEG model."""
    device = torch.device(args.device)
    model.eval()
    test_data = load_test_data(test_dir)
    rows = []
    probas = []
    with torch.no_grad():
        for user_id, trials in test_data.items():
            if uses_window_normalization(args.norm_mode):
                trials = standardize_by_norm_mode(trials, args.norm_mode)
            test_ds = RawEEGDataset(trials, np.zeros(len(trials), dtype=np.int64), stats)
            test_loader = make_data_loader(
                test_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            subject_proba = predict_proba_loader(model, test_loader, device, amp=args.amp)
            if args.balanced_rank:
                subject_preds = balanced_rank_predictions(subject_proba).astype(int).tolist()
            else:
                subject_preds = subject_proba.argmax(axis=1).astype(int).tolist()
            probas.append(subject_proba)
            for trial_idx in range(1, len(subject_preds) + 1):
                rows.append((user_id, trial_idx))
            logger.info(f"  {user_id}: {np.asarray(subject_preds, dtype=int)} (trials 1-8)")
    return rows, np.concatenate(probas, axis=0) if probas else np.empty((0, 2), dtype=np.float32)


def rows_to_predictions(rows, probas, balanced_rank=False):
    if balanced_rank:
        preds = np.zeros(probas.shape[0], dtype=np.int64)
        start = 0
        while start < len(rows):
            user_id = rows[start][0]
            end = start + 1
            while end < len(rows) and rows[end][0] == user_id:
                end += 1
            preds[start:end] = balanced_rank_predictions(probas[start:end])
            start = end
        preds = preds.tolist()
    else:
        preds = probas.argmax(axis=1).astype(int).tolist()
    return [(user_id, trial_idx, int(pred)) for (user_id, trial_idx), pred in zip(rows, preds)]


def predict_test_raw(model, stats, test_dir, args, logger):
    """Predict all public test trials with a raw EEG model."""
    rows, probas = predict_test_raw_proba(model, stats, test_dir, args, logger)
    predictions = rows_to_predictions(rows, probas, balanced_rank=args.balanced_rank)
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
    eval_seed = resolve_eval_seed(args, seed)
    ensemble_seeds = parse_ensemble_seeds(args.ensemble_seeds, seed)
    set_seed(seed)
    logger, _ = setup_logging(cfg["output"]["log_dir"])
    logger.info("=" * 60)
    logger.info(f"GPU baseline: {args.model}, device={args.device}, amp={args.amp}")
    if len(ensemble_seeds) > 1:
        logger.info(f"Ensemble seeds: {list(ensemble_seeds)}")
    logger.info("=" * 60)

    X, y, subjects = load_train_data(
        train_dir=cfg["data"]["train_dir"],
        window_size=resolve_signal_samples(cfg, args.train_stride_sec)[0],
        stride=resolve_signal_samples(cfg, args.train_stride_sec)[1],
        clip_sigma=cfg["signal"]["clip_sigma"],
    )
    if uses_window_normalization(args.norm_mode):
        X = standardize_by_norm_mode(X, args.norm_mode)
    logger.info(f"Raw windows: {X.shape}, subjects={len(iter_unique_subjects(subjects))}")

    if not args.skip_loso:
        if args.full_loso:
            eval_subjects = iter_unique_subjects(subjects)
        else:
            n_eval_subjects = args.n_eval_subjects or cfg["training"].get("n_eval_subjects")
            eval_subjects = select_eval_subjects(subjects, n_eval_subjects, eval_seed)
        logger.info(f"Eval subjects: {list(eval_subjects)}")

        scores = []
        for i, test_subject in enumerate(eval_subjects, start=1):
            if len(ensemble_seeds) > 1:
                acc, member_summaries = train_fold_ensemble(
                    args, cfg, X, y, subjects, test_subject, ensemble_seeds, logger
                )
                summary = ", ".join(
                    f"seed={member_seed}:acc={member_acc:.4f}@{member_epoch}"
                    for member_seed, member_acc, member_epoch in member_summaries
                )
                best_epoch = f"ensemble({summary})"
            else:
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
        model_states = []
        test_probas = []
        test_rows = None
        stats = None
        for member_idx, member_seed in enumerate(ensemble_seeds, start=1):
            set_seed(member_seed)
            if len(ensemble_seeds) > 1:
                logger.info(f"  final ensemble member {member_idx}/{len(ensemble_seeds)} seed={member_seed}")
            model, stats = train_final_model(args, cfg, X, y, logger)
            rows, probas = predict_test_raw_proba(model, stats, cfg["data"]["test_dir"], args, logger)
            if test_rows is None:
                test_rows = rows
            test_probas.append(probas)
            model_states.append({"seed": member_seed, "state_dict": model.state_dict()})

        predictions = rows_to_predictions(
            test_rows,
            average_probabilities(test_probas),
            balanced_rank=args.balanced_rank,
        )
        save_submission(predictions, output_path)
        os.makedirs(os.path.dirname(model_output), exist_ok=True)
        torch.save(
            {
                "model": model_states[0]["state_dict"] if len(model_states) == 1 else model_states,
                "model_name": args.model,
                "channel_mean": stats.mean,
                "channel_std": stats.std,
                "ensemble_seeds": ensemble_seeds,
                "args": vars(args),
            },
            model_output,
        )
        logger.info(f"Saved final raw EEG model: {model_output}")

    logger.info(f"GPU baseline complete in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
