"""Utility functions for EEG emotion recognition."""

import os
import random
import time
import logging
from datetime import datetime

import numpy as np


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def clip_outliers(data: np.ndarray, clip_sigma: float = 5.0) -> np.ndarray:
    """Clip outliers to mean ± clip_sigma * std, channel-wise.

    Args:
        data: shape (n_channels, n_samples) or (n_windows, n_channels, n_samples)
        clip_sigma: number of standard deviations for clipping

    Returns:
        Clipped data array
    """
    result = data.copy()
    if result.ndim == 2:
        # (channels, samples)
        for ch in range(result.shape[0]):
            ch_data = result[ch]
            mean = np.mean(ch_data)
            std = np.std(ch_data)
            lower = mean - clip_sigma * std
            upper = mean + clip_sigma * std
            result[ch] = np.clip(ch_data, lower, upper)
    elif result.ndim == 3:
        # (windows, channels, samples)
        for ch in range(result.shape[1]):
            ch_data = result[:, ch, :]
            mean = np.mean(ch_data)
            std = np.std(ch_data)
            lower = mean - clip_sigma * std
            upper = mean + clip_sigma * std
            result[:, ch, :] = np.clip(ch_data, lower, upper)
    return result


def setup_logging(log_dir: str = "outputs/logs"):
    """Setup logging to both file and console."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_{timestamp}.log")

    logger = logging.getLogger("eeg_emotion")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    ))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    ))
    logger.addHandler(ch)

    return logger, log_file


def save_run_summary(log_dir: str, model_name: str, accuracy: float,
                     elapsed_sec: float, log_file: str = ""):
    """Save a human-readable run summary for reporting."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(log_dir, f"summary_{timestamp}.txt")

    summary = (
        f"{'='*50}\n"
        f"Run Summary\n"
        f"{'='*50}\n"
        f"Model:       {model_name}\n"
        f"LOSO Acc:    {accuracy:.4f} ({accuracy*100:.2f}%)\n"
        f"Time:        {elapsed_sec:.1f}s ({elapsed_sec/60:.1f} min)\n"
        f"Timestamp:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Log file:    {log_file or 'N/A'}\n"
        f"{'='*50}\n"
    )

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary)

    return summary_file
