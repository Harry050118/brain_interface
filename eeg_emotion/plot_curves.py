#!/usr/bin/env python3
"""Train all models with per-epoch logging and generate accuracy/loss curves.

Saves plots to outputs/plots/ directory.
"""

import os
import sys

import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_base = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, _base)
sys.path.insert(0, os.path.join(_base, "models"))

from utils import set_seed
from data_loader import load_train_data
from features import extract_de_batch


def train_svm_with_history(X_de, y, cfg):
    """Train SVM and return history."""
    from models.svm_model import SVMModel
    model = SVMModel(**cfg["svm"])
    model.fit(X_de, y)
    # SVM doesn't have epochs, so just one point
    X_scaled = model.scaler.transform(X_de)
    train_preds = model.svm.predict(X_scaled)
    train_acc = (train_preds == y).mean()
    return model, {"train_acc": [train_acc]}, "SVM"


def train_dgcnn_with_history(X_de, y, cfg):
    """Train DGCNN with per-epoch logging."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from models.dgcnn import DGCNN, DGCNNModel

    n_bands = len(cfg["features"]["bands"])
    wrapper = DGCNNModel(
        n_bands=n_bands,
        **{k: v for k, v in cfg["dgcnn"].items() if k in [
            "hidden_dim", "num_layers", "dropout",
            "learning_rate", "epochs", "batch_size", "weight_decay"
        ]}
    )

    X_reshaped = wrapper._reshape_features(X_de)
    X_norm = wrapper._normalize(X_reshaped, fit=True)
    X_tensor = torch.FloatTensor(X_norm)
    y_tensor = torch.LongTensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=wrapper.batch_size, shuffle=True)

    wrapper.model = DGCNN(
        n_channels=wrapper.n_channels,
        n_bands=n_bands,
        hidden_dim=wrapper.hidden_dim,
        num_layers=wrapper.num_layers,
        dropout=wrapper.dropout
    ).to(wrapper.device)

    optimizer = torch.optim.Adam(
        wrapper.model.parameters(), lr=wrapper.lr, weight_decay=wrapper.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=wrapper.epochs)
    criterion = torch.nn.CrossEntropyLoss()

    history = {"train_acc": [], "loss": [], "test_acc": []}
    wrapper.model.train()
    for epoch in range(wrapper.epochs):
        total_loss = 0
        correct = 0
        total = 0
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(wrapper.device)
            batch_y = batch_y.to(wrapper.device)
            logits = wrapper.model(batch_X)
            loss = criterion(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_y)
            correct += (logits.argmax(dim=1) == batch_y).sum().item()
            total += len(batch_y)
        scheduler.step()
        epoch_acc = correct / total
        epoch_loss = total_loss / total
        history["train_acc"].append(epoch_acc)
        history["loss"].append(epoch_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{wrapper.epochs}: acc={epoch_acc:.4f}, loss={epoch_loss:.4f}")

    wrapper.is_fitted = True
    return wrapper, history, "DGCNN"


def train_conformer_with_history(X, y, cfg):
    """Train EEG-Conformer with per-epoch logging."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from models.eeg_conformer import EEGConformer, EEGConformerModel

    wrapper = EEGConformerModel(
        n_channels=cfg["signal"]["n_channels"],
        window_size=cfg["signal"]["sample_rate"] * cfg["signal"]["window_size_sec"],
        **{k: v for k, v in cfg["conformer"].items() if k in [
            "hidden_dim", "num_heads", "num_layers", "dropout",
            "learning_rate", "epochs", "batch_size", "weight_decay",
            "time_pool"
        ]}
    )

    X_norm = wrapper._normalize(X, fit=True)
    X_tensor = torch.FloatTensor(X_norm)
    y_tensor = torch.LongTensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=wrapper.batch_size, shuffle=True)

    wrapper.model = EEGConformer(
        n_channels=wrapper.n_channels,
        window_size=wrapper.window_size,
        hidden_dim=wrapper.hidden_dim,
        num_heads=wrapper.num_heads,
        num_layers=wrapper.num_layers,
        dropout=wrapper.dropout,
        time_pool=wrapper.time_pool
    ).to(wrapper.device)

    optimizer = torch.optim.AdamW(
        wrapper.model.parameters(), lr=wrapper.lr, weight_decay=wrapper.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=wrapper.lr,
        total_steps=len(loader) * wrapper.epochs,
        pct_start=0.1
    )
    criterion = torch.nn.CrossEntropyLoss()

    history = {"train_acc": [], "loss": [], "test_acc": []}
    wrapper.model.train()
    for epoch in range(wrapper.epochs):
        total_loss = 0
        correct = 0
        total = 0
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(wrapper.device)
            batch_y = batch_y.to(wrapper.device)
            logits = wrapper.model(batch_X)
            loss = criterion(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wrapper.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(batch_y)
            correct += (logits.argmax(dim=1) == batch_y).sum().item()
            total += len(batch_y)
        epoch_acc = correct / total
        epoch_loss = total_loss / total
        history["train_acc"].append(epoch_acc)
        history["loss"].append(epoch_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{wrapper.epochs}: acc={epoch_acc:.4f}, loss={epoch_loss:.4f}")

    wrapper.is_fitted = True
    return wrapper, history, "EEG-Conformer"


def plot_curves(histories, output_dir):
    """Generate accuracy and loss curve plots."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy curves
    ax = axes[0]
    for name, history, _ in histories:
        epochs = range(1, len(history["train_acc"]) + 1)
        ax.plot(epochs, history["train_acc"], label=name, linewidth=2)
        best_acc = max(history["train_acc"])
        best_epoch = history["train_acc"].index(best_acc) + 1
        ax.scatter([best_epoch], [best_acc], zorder=5)
        ax.annotate(f"{best_acc:.4f}", (best_epoch, best_acc),
                     textcoords="offset points", xytext=(8, 8), fontsize=8)
    ax.set_title("Training Accuracy", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.45, 1.0)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Loss curves
    ax2 = axes[1]
    for name, history, _ in histories:
        if "loss" in history and history["loss"]:
            epochs = range(1, len(history["loss"]) + 1)
            ax2.plot(epochs, history["loss"], label=name, linewidth=2)
    ax2.set_title("Training Loss", fontsize=14)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cross-Entropy Loss")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    acc_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(acc_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved training curves to: {acc_path}")
    plt.close()

    # Individual model plots
    for name, history, _ in histories:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        epochs = range(1, len(history["train_acc"]) + 1)

        ax = axes[0]
        ax.plot(epochs, history["train_acc"], color="#2196F3", linewidth=2)
        ax.set_title(f"{name} - Training Accuracy", fontsize=14)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.3)

        ax2 = axes[1]
        if "loss" in history and history["loss"]:
            ax2.plot(epochs, history["loss"], color="#F44336", linewidth=2)
            ax2.set_title(f"{name} - Training Loss", fontsize=14)
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Loss")
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        individual_path = os.path.join(output_dir, f"training_{name.replace(' ', '_').lower()}.png")
        plt.savefig(individual_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {individual_path}")
        plt.close()


def main():
    config_path = os.path.join(os.path.dirname(__file__), "configs", "config.yaml")
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["random_seed"])

    print("Loading training data...")
    X, y, subjects = load_train_data(
        train_dir=cfg["data"]["train_dir"],
        window_size=cfg["signal"]["sample_rate"] * cfg["signal"]["window_size_sec"],
        stride=cfg["signal"]["sample_rate"] * cfg["signal"]["train_stride_sec"],
        clip_sigma=cfg["signal"]["clip_sigma"]
    )
    print(f"Data: {X.shape}")
    X_de = extract_de_batch(X, bands=cfg["features"]["bands"])
    print(f"DE features: {X_de.shape}")

    histories = []

    # SVM
    print("\n--- Training SVM ---")
    model, history, name = train_svm_with_history(X_de, y, cfg)
    histories.append((name, history, model))

    # DGCNN
    print("\n--- Training DGCNN ---")
    model, history, name = train_dgcnn_with_history(X_de, y, cfg)
    histories.append((name, history, model))

    # EEG-Conformer
    print("\n--- Training EEG-Conformer ---")
    model, history, name = train_conformer_with_history(X, y, cfg)
    histories.append((name, history, model))

    # Plot
    print("\n--- Generating plots ---")
    plot_dir = os.path.join(os.path.dirname(__file__), "outputs", "plots")
    plot_curves(histories, plot_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()
