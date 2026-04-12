"""Raw EEG dataset utilities for GPU baselines."""

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class ChannelStats:
    """Per-channel normalization statistics estimated from training windows."""

    mean: np.ndarray
    std: np.ndarray


def compute_channel_stats(X: np.ndarray, eps: float = 1e-6) -> ChannelStats:
    """Compute per-channel mean/std for raw EEG windows.

    Args:
        X: raw EEG windows, shape (n_windows, n_channels, n_times).
        eps: minimum std for numerical stability.
    """
    mean = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = X.std(axis=(0, 2), keepdims=True).astype(np.float32)
    std = np.maximum(std, eps)
    return ChannelStats(mean=mean, std=std)


def standardize_raw_eeg(X: np.ndarray, stats: ChannelStats) -> np.ndarray:
    """Apply per-channel normalization using precomputed training statistics."""
    return ((X - stats.mean) / stats.std).astype(np.float32, copy=False)


class RawEEGDataset(Dataset):
    """Torch dataset for raw EEG windows.

    Samples are returned as (x, y), where x has shape (n_channels, n_times).
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        stats: Optional[ChannelStats] = None,
        noise_std: float = 0.0,
        channel_drop_prob: float = 0.0,
        max_time_shift: int = 0,
    ):
        if X.ndim != 3:
            raise ValueError(f"Expected X shape (n, channels, time), got {X.shape}")
        if len(X) != len(y):
            raise ValueError(f"X/y length mismatch: {len(X)} != {len(y)}")

        self.X = standardize_raw_eeg(X, stats) if stats is not None else X.astype(np.float32, copy=False)
        self.y = np.asarray(y, dtype=np.int64)
        self.noise_std = noise_std
        self.channel_drop_prob = channel_drop_prob
        self.max_time_shift = max_time_shift

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_np = self.X[idx]
        if self.noise_std > 0.0 or self.channel_drop_prob > 0.0 or self.max_time_shift > 0:
            x_np = x_np.copy()
            if self.max_time_shift > 0:
                shift = np.random.randint(-self.max_time_shift, self.max_time_shift + 1)
                if shift != 0:
                    x_np = np.roll(x_np, shift=shift, axis=-1)
            if self.channel_drop_prob > 0.0:
                mask = np.random.rand(x_np.shape[0]) < self.channel_drop_prob
                x_np[mask] = 0.0
            if self.noise_std > 0.0:
                x_np = x_np + np.random.normal(0.0, self.noise_std, size=x_np.shape).astype(np.float32)

        x = torch.from_numpy(x_np)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


def loso_masks(subjects: Sequence[str], test_subject: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return train/test boolean masks for one LOSO subject."""
    subjects_array = np.asarray(subjects)
    test_mask = subjects_array == test_subject
    if not test_mask.any():
        raise ValueError(f"Unknown test_subject: {test_subject}")
    train_mask = ~test_mask
    return train_mask, test_mask


def make_loso_raw_datasets(
    X: np.ndarray,
    y: np.ndarray,
    subjects: Sequence[str],
    test_subject: str,
    train_noise_std: float = 0.0,
    train_channel_drop_prob: float = 0.0,
    train_max_time_shift: int = 0,
) -> Tuple[RawEEGDataset, RawEEGDataset, ChannelStats]:
    """Build normalized train/validation datasets for one LOSO fold."""
    train_mask, test_mask = loso_masks(subjects, test_subject)
    stats = compute_channel_stats(X[train_mask])
    train_ds = RawEEGDataset(
        X[train_mask],
        y[train_mask],
        stats,
        noise_std=train_noise_std,
        channel_drop_prob=train_channel_drop_prob,
        max_time_shift=train_max_time_shift,
    )
    val_ds = RawEEGDataset(X[test_mask], y[test_mask], stats)
    return train_ds, val_ds, stats


def make_data_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
) -> DataLoader:
    """Create a DataLoader with CUDA-friendly defaults when available."""
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def iter_unique_subjects(subjects: Iterable[str]) -> Tuple[str, ...]:
    """Stable sorted subject list for LOSO loops."""
    return tuple(sorted(set(subjects)))
