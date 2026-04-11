"""Training helpers for domain-adversarial DGCNN.

The domain label is the training subject ID. In LOSO, the held-out subject is
not used in the domain classifier, avoiding leakage from validation labels.
"""

import logging
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.domain_adversarial_dgcnn import DomainAdversarialDGCNNModel
from train import select_eval_subjects


def make_domain_labels(subjects) -> Tuple[np.ndarray, Dict[str, int]]:
    """Map subject IDs to contiguous domain labels."""
    unique_subjects = sorted(set(subjects))
    subject_to_domain = {subject: idx for idx, subject in enumerate(unique_subjects)}
    domain_labels = np.array([subject_to_domain[subject] for subject in subjects], dtype=np.int64)
    return domain_labels, subject_to_domain


def grl_lambda_schedule(epoch: int, total_epochs: int) -> float:
    """DANN schedule from Ganin et al.; ramps from near 0 to near 1."""
    if total_epochs <= 1:
        return 1.0
    progress = epoch / float(total_epochs - 1)
    return float(2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0)


def train_domain_adversarial_dgcnn(
    model: DomainAdversarialDGCNNModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    subjects_train,
    epochs: int = 80,
    batch_size: int = 256,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-3,
    domain_loss_weight: float = 0.2,
    logger: logging.Logger = None,
) -> None:
    """Train a DANN-DGCNN model on precomputed DE features."""
    log = logger or logging.getLogger("eeg_emotion")
    domain_labels, subject_to_domain = make_domain_labels(subjects_train)
    model.n_domains = len(subject_to_domain)

    X_norm = model._normalize(model._reshape_features(X_train), fit=True)
    X_tensor = torch.FloatTensor(X_norm)
    y_tensor = torch.LongTensor(y_train)
    d_tensor = torch.LongTensor(domain_labels)

    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor, d_tensor),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    net = model.build_model()
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    class_loss = nn.CrossEntropyLoss()
    domain_loss = nn.CrossEntropyLoss()

    net.train()
    for epoch in range(epochs):
        lambda_ = grl_lambda_schedule(epoch, epochs)
        total_cls = 0.0
        total_dom = 0.0
        total = 0

        for batch_X, batch_y, batch_domain in loader:
            batch_X = batch_X.to(model.device)
            batch_y = batch_y.to(model.device)
            batch_domain = batch_domain.to(model.device)

            emotion_logits, domain_logits = net(batch_X, grl_lambda=lambda_)
            loss_cls = class_loss(emotion_logits, batch_y)
            loss_domain = domain_loss(domain_logits, batch_domain)
            loss = loss_cls + domain_loss_weight * loss_domain

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            total_cls += loss_cls.item() * len(batch_y)
            total_dom += loss_domain.item() * len(batch_y)
            total += len(batch_y)

        scheduler.step()
        if (epoch + 1) == epochs or (epoch + 1) % max(1, epochs // 5) == 0:
            log.info(
                "  epoch %d/%d: cls_loss=%.4f domain_loss=%.4f grl=%.3f",
                epoch + 1,
                epochs,
                total_cls / max(total, 1),
                total_dom / max(total, 1),
                lambda_,
            )

    model.is_fitted = True


def run_loso_dann_features(
    model_factory: Callable[[int], DomainAdversarialDGCNNModel],
    X_features: np.ndarray,
    y_all: np.ndarray,
    subjects_all,
    epochs: int = 80,
    batch_size: int = 256,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-3,
    domain_loss_weight: float = 0.2,
    n_eval_subjects: int = None,
    seed: int = 42,
    logger: logging.Logger = None,
) -> Tuple[float, List[float], Dict[str, float]]:
    """Run LOSO for DANN-DGCNN over precomputed DE features."""
    log = logger or logging.getLogger("eeg_emotion")
    subjects_array = np.asarray(subjects_all)
    eval_subjects = select_eval_subjects(subjects_all, n_eval_subjects, seed)
    log.info("DANN-DGCNN LOSO: %d eval subjects", len(eval_subjects))

    subject_accuracies = []
    per_subject_results = {}

    for i, test_subject in enumerate(eval_subjects):
        test_mask = subjects_array == test_subject
        train_mask = ~test_mask
        train_subjects = subjects_array[train_mask].tolist()
        n_domains = len(set(train_subjects))
        model = model_factory(n_domains)

        train_domain_adversarial_dgcnn(
            model=model,
            X_train=X_features[train_mask],
            y_train=y_all[train_mask],
            subjects_train=train_subjects,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            domain_loss_weight=domain_loss_weight,
            logger=log,
        )

        preds = model.predict(X_features[test_mask])
        acc = float((preds == y_all[test_mask]).mean())
        subject_accuracies.append(acc)
        per_subject_results[test_subject] = acc
        log.info("  [%d/%d] %s: acc=%.4f", i + 1, len(eval_subjects), test_subject, acc)

    mean_acc = float(np.mean(subject_accuracies))
    std_acc = float(np.std(subject_accuracies))
    log.info("DANN-DGCNN LOSO Mean Accuracy: %.4f (%.2f%%), std=%.4f", mean_acc, mean_acc * 100, std_acc)
    return mean_acc, subject_accuracies, per_subject_results
