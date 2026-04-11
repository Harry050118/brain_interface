"""Differential Entropy (DE) feature extraction for EEG signals.

DE features are computed per channel per frequency band.
For a Gaussian signal, DE = 0.5 * log(2 * pi * e * sigma^2).
"""

from typing import Dict, Tuple

import numpy as np
from scipy.signal import butter, filtfilt


# Default frequency bands for emotion recognition
DEFAULT_BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (31, 49),
}


def bandpass_filter(signal: np.ndarray, low: float, high: float,
                    sample_rate: float = 250.0, order: int = 4) -> np.ndarray:
    """Apply bandpass filter to a 1D signal.

    Args:
        signal: 1D array of EEG samples
        low: low cutoff frequency (Hz)
        high: high cutoff frequency (Hz)
        sample_rate: sampling rate (Hz)
        order: filter order

    Returns:
        filtered signal
    """
    nyquist = sample_rate / 2.0
    low_norm = low / nyquist
    high_norm = high / nyquist

    # Ensure frequencies are valid
    if low_norm <= 0:
        low_norm = 0.001
    if high_norm >= 1.0:
        high_norm = 0.999
    if low_norm >= high_norm:
        return np.zeros_like(signal)

    b, a = butter(order, [low_norm, high_norm], btype="band")
    return filtfilt(b, a, signal)


def differential_entropy(signal: np.ndarray) -> float:
    """Compute differential entropy of a signal.

    For a Gaussian distribution: DE = 0.5 * log(2 * pi * e * variance)

    Args:
        signal: 1D array

    Returns:
        DE value (scalar)
    """
    variance = np.var(signal)
    if variance <= 1e-10:
        variance = 1e-10
    return 0.5 * np.log(2 * np.pi * np.e * variance)


def extract_de_features(
    window: np.ndarray,
    bands: Dict[str, Tuple[float, float]] = None,
    sample_rate: float = 250.0
) -> np.ndarray:
    """Extract DE features from a single EEG window.

    Args:
        window: shape (n_channels, n_samples)
        bands: frequency band definitions
        sample_rate: sampling rate in Hz

    Returns:
        DE features: shape (n_channels * n_bands,)
    """
    if bands is None:
        bands = DEFAULT_BANDS

    n_channels = window.shape[0]
    features = []

    for ch_idx in range(n_channels):
        ch_signal = window[ch_idx]
        for band_name, (low, high) in bands.items():
            filtered = bandpass_filter(ch_signal, low, high, sample_rate)
            de = differential_entropy(filtered)
            features.append(de)

    return np.array(features, dtype=np.float32)


def extract_de_batch(
    windows: np.ndarray,
    bands: Dict[str, Tuple[float, float]] = None,
    sample_rate: float = 250.0
) -> np.ndarray:
    """Extract DE features from a batch of windows.

    Args:
        windows: shape (n_windows, n_channels, n_samples)
        bands: frequency band definitions
        sample_rate: sampling rate in Hz

    Returns:
        DE features: shape (n_windows, n_channels * n_bands)
    """
    n_windows = windows.shape[0]
    n_bands = len(bands) if bands else len(DEFAULT_BANDS)
    n_channels = windows.shape[1]
    feature_dim = n_channels * n_bands

    features = np.empty((n_windows, feature_dim), dtype=np.float32)

    for i in range(n_windows):
        features[i] = extract_de_features(windows[i], bands, sample_rate)

    return features
