"""Feature extraction for EEG emotion recognition.

The original baseline uses differential entropy (DE) per channel and band.
This module also exposes an enhanced feature set for small-sample SVM baselines:
DE + Welch PSD band power + multitaper band power + simple asymmetry + Hjorth.
"""

from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.signal.windows import dpss


DEFAULT_BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (31, 49),
}

# Without an official channel-name map in code, use adjacent channel pairs as a
# conservative asymmetry proxy. If the exact montage is added later, replace
# these with true left/right homologous pairs.
DEFAULT_ASYMMETRY_PAIRS = tuple((i, i + 1) for i in range(0, 30, 2))


def bandpass_filter(signal: np.ndarray, low: float, high: float,
                    sample_rate: float = 250.0, order: int = 4) -> np.ndarray:
    """Apply a Butterworth bandpass filter to a 1D signal."""
    nyquist = sample_rate / 2.0
    low_norm = max(low / nyquist, 0.001)
    high_norm = min(high / nyquist, 0.999)
    if low_norm >= high_norm:
        return np.zeros_like(signal)

    b, a = butter(order, [low_norm, high_norm], btype="band")
    return filtfilt(b, a, signal)


def differential_entropy(signal: np.ndarray) -> float:
    """Compute Gaussian differential entropy from signal variance."""
    variance = np.var(signal)
    if variance <= 1e-10:
        variance = 1e-10
    return 0.5 * np.log(2 * np.pi * np.e * variance)


def _band_mask(freqs: np.ndarray, low: float, high: float) -> np.ndarray:
    return (freqs >= low) & (freqs <= high)


def _safe_log_power(power: float) -> float:
    return float(np.log(max(power, 1e-12)))


def welch_bandpowers(signal: np.ndarray, bands: Dict[str, Tuple[float, float]],
                     sample_rate: float = 250.0) -> np.ndarray:
    """Log band powers from Welch PSD for one channel."""
    nperseg = min(len(signal), int(sample_rate * 2))
    freqs, psd = welch(signal, fs=sample_rate, nperseg=nperseg)
    feats = []
    for low, high in bands.values():
        mask = _band_mask(freqs, low, high)
        power = np.trapz(psd[mask], freqs[mask]) if np.any(mask) else 0.0
        feats.append(_safe_log_power(power))
    return np.array(feats, dtype=np.float32)


def multitaper_bandpowers(signal: np.ndarray, bands: Dict[str, Tuple[float, float]],
                          sample_rate: float = 250.0, time_bandwidth: float = 3.0,
                          n_tapers: int = 5) -> np.ndarray:
    """Log band powers from a simple DPSS multitaper PSD estimate."""
    n = len(signal)
    x = signal - np.mean(signal)
    tapers = dpss(n, time_bandwidth, Kmax=n_tapers, sym=False)
    spectra = np.fft.rfft(tapers * x[None, :], axis=1)
    psd = (np.abs(spectra) ** 2).mean(axis=0) / (sample_rate * n)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    feats = []
    for low, high in bands.values():
        mask = _band_mask(freqs, low, high)
        power = np.trapz(psd[mask], freqs[mask]) if np.any(mask) else 0.0
        feats.append(_safe_log_power(power))
    return np.array(feats, dtype=np.float32)


def hjorth_features(signal: np.ndarray) -> np.ndarray:
    """Return Hjorth activity, mobility, and complexity for one channel."""
    x = signal - np.mean(signal)
    dx = np.diff(x)
    ddx = np.diff(dx)

    var_x = np.var(x) + 1e-12
    var_dx = np.var(dx) + 1e-12
    var_ddx = np.var(ddx) + 1e-12

    activity = var_x
    mobility = np.sqrt(var_dx / var_x)
    complexity = np.sqrt(var_ddx / var_dx) / mobility
    return np.log(np.array([activity, mobility, complexity], dtype=np.float32) + 1e-12)


def extract_de_features(window: np.ndarray,
                        bands: Dict[str, Tuple[float, float]] = None,
                        sample_rate: float = 250.0) -> np.ndarray:
    """Extract DE features from one EEG window, shape (channels * bands,)."""
    if bands is None:
        bands = DEFAULT_BANDS

    features = []
    for ch_signal in window:
        for low, high in bands.values():
            filtered = bandpass_filter(ch_signal, low, high, sample_rate)
            features.append(differential_entropy(filtered))
    return np.array(features, dtype=np.float32)


def extract_enhanced_features(
    window: np.ndarray,
    bands: Dict[str, Tuple[float, float]] = None,
    sample_rate: float = 250.0,
    asymmetry_pairs: Iterable[Tuple[int, int]] = DEFAULT_ASYMMETRY_PAIRS,
) -> np.ndarray:
    """Extract a stronger SVM feature vector from one EEG window.

    Layout:
    - DE features: channels x bands
    - Welch log band powers: channels x bands
    - multitaper log band powers: channels x bands
    - asymmetry on DE: pair x band for difference and ratio
    - Hjorth features: channels x 3
    """
    if bands is None:
        bands = DEFAULT_BANDS

    n_channels = window.shape[0]
    de = extract_de_features(window, bands, sample_rate).reshape(n_channels, len(bands))

    welch_feats = []
    mt_feats = []
    hjorth = []
    for ch_signal in window:
        welch_feats.append(welch_bandpowers(ch_signal, bands, sample_rate))
        mt_feats.append(multitaper_bandpowers(ch_signal, bands, sample_rate))
        hjorth.append(hjorth_features(ch_signal))

    welch_arr = np.vstack(welch_feats)
    mt_arr = np.vstack(mt_feats)
    hjorth_arr = np.vstack(hjorth)

    asym = []
    for left, right in asymmetry_pairs:
        if left >= n_channels or right >= n_channels:
            continue
        diff = de[left] - de[right]
        ratio = diff / (np.abs(de[left]) + np.abs(de[right]) + 1e-6)
        asym.extend(diff.tolist())
        asym.extend(ratio.tolist())

    return np.concatenate([
        de.ravel(),
        welch_arr.ravel(),
        mt_arr.ravel(),
        np.array(asym, dtype=np.float32),
        hjorth_arr.ravel(),
    ]).astype(np.float32)


def extract_de_batch(windows: np.ndarray,
                     bands: Dict[str, Tuple[float, float]] = None,
                     sample_rate: float = 250.0) -> np.ndarray:
    """Extract DE features from a batch of EEG windows."""
    if bands is None:
        bands = DEFAULT_BANDS

    n_windows = windows.shape[0]
    feature_dim = windows.shape[1] * len(bands)
    features = np.empty((n_windows, feature_dim), dtype=np.float32)
    for i in range(n_windows):
        features[i] = extract_de_features(windows[i], bands, sample_rate)
    return features


def extract_enhanced_batch(windows: np.ndarray,
                           bands: Dict[str, Tuple[float, float]] = None,
                           sample_rate: float = 250.0) -> np.ndarray:
    """Extract enhanced SVM features from a batch of EEG windows."""
    if bands is None:
        bands = DEFAULT_BANDS

    first = extract_enhanced_features(windows[0], bands, sample_rate)
    features = np.empty((windows.shape[0], len(first)), dtype=np.float32)
    features[0] = first
    for i in range(1, windows.shape[0]):
        features[i] = extract_enhanced_features(windows[i], bands, sample_rate)
    return features


def extract_feature_batch(windows: np.ndarray,
                          bands: Dict[str, Tuple[float, float]] = None,
                          sample_rate: float = 250.0,
                          feature_set: str = "de") -> np.ndarray:
    """Dispatch feature extraction by name: 'de' or 'enhanced'."""
    if feature_set == "de":
        return extract_de_batch(windows, bands, sample_rate)
    if feature_set == "enhanced":
        return extract_enhanced_batch(windows, bands, sample_rate)
    raise ValueError(f"Unknown feature_set: {feature_set}")
