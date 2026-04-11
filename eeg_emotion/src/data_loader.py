"""Data loader for EEG emotion recognition dataset.

Handles reading MATLAB .mat files (both v7.3 training and v7 test formats),
sliding window extraction, and train/test data organization.
"""

import os
import glob
import time
from typing import Dict, List, Tuple

import h5py
import numpy as np
import scipy.io
from tqdm import tqdm

from utils import clip_outliers


def read_train_subject(filepath: str) -> Dict[str, np.ndarray]:
    """Read a single training subject .mat file (MATLAB v7.3 format).

    Args:
        filepath: path to DEP*timedata.mat or HC*timedata.mat

    Returns:
        dict with 'EEG_data_neu' and 'EEG_data_pos' keys,
        each shape (30, 50000), dtype float32
    """
    data = {}
    with h5py.File(filepath, 'r') as f:
        for key in ['EEG_data_neu', 'EEG_data_pos']:
            if key in f:
                arr = f[key][()]
                # h5py returns transposed data from MATLAB
                if arr.ndim == 2:
                    arr = arr.T
                data[key] = arr.astype(np.float32)
    return data


def read_test_subject(filepath: str) -> np.ndarray:
    """Read a single test subject .mat file (MATLAB v7 format).

    Args:
        filepath: path to P_test*.mat

    Returns:
        numpy array shape (30, 20000), dtype float32
    """
    raw = scipy.io.loadmat(filepath)
    arr = raw['test_eeg_c']
    return arr.astype(np.float32)


def sliding_window(data: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """Extract sliding windows from EEG data.

    Args:
        data: shape (n_channels, n_samples)
        window_size: window size in samples
        stride: stride in samples

    Returns:
        windows: shape (n_windows, n_channels, window_size)
    """
    n_channels, n_samples = data.shape
    n_windows = (n_samples - window_size) // stride + 1

    windows = []
    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        windows.append(data[:, start:end])

    return np.array(windows)


def load_train_data(
    train_dir: str,
    window_size: int = 2500,
    stride: int = 1250,
    clip_sigma: float = 5.0
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load all training data with sliding window augmentation.

    Each 50s video segment (12500 samples per emotion per subject)
    is split into sliding windows.

    Args:
        train_dir: path to 训练集 directory
        window_size: window size in samples (default 2500 = 10s)
        stride: stride in samples (default 1250 = 5s)
        clip_sigma: outlier clipping threshold

    Returns:
        X: shape (n_windows, 30, window_size), float32
        y: shape (n_windows,), int (0=neutral, 1=positive)
        subject_ids: list of subject IDs corresponding to each window
    """
    all_X = []
    all_y = []
    all_subjects = []

    file_jobs = []
    for subdir in ["正常人", "抑郁症患者"]:
        pattern = os.path.join(train_dir, subdir, "*timedata.mat")
        for fpath in sorted(glob.glob(pattern)):
            file_jobs.append((subdir, fpath))

    # Show progress while reading all subject files.
    pbar = tqdm(file_jobs, desc="Loading train subjects", unit="file")
    for _, fpath in pbar:
        subject_name = os.path.basename(fpath).replace("timedata.mat", "")
        pbar.set_postfix_str(subject_name)
        data = read_train_subject(fpath)

        for emotion_key, label in [("EEG_data_neu", 0), ("EEG_data_pos", 1)]:
            if emotion_key not in data:
                continue

            eeg = data[emotion_key]  # (30, 50000)
            eeg = clip_outliers(eeg, clip_sigma)

            # Each 50s segment = 12500 samples
            segment_size = 12500
            n_segments = eeg.shape[1] // segment_size

            for seg_idx in range(n_segments):
                start = seg_idx * segment_size
                end = start + segment_size
                segment = eeg[:, start:end]

                windows = sliding_window(segment, window_size, stride)
                all_X.append(windows)
                all_y.append(np.full(len(windows), label, dtype=np.int64))
                all_subjects.extend([subject_name] * len(windows))

    t_concat = time.time()
    tqdm.write("Concatenating windows and labels...")
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    tqdm.write(f"Concatenate done: X={X.shape}, y={y.shape}, time={time.time() - t_concat:.1f}s")

    return X, y, all_subjects


def load_test_data(
    test_dir: str,
    window_size: int = 2500,
    clip_sigma: float = 5.0
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[int]]]:
    """Load all test data.

    Each test subject has 20000 samples = 8 trials * 2500 samples.

    Args:
        test_dir: path to 公开测试集 directory
        window_size: window size in samples (default 2500 = 10s)
        clip_sigma: outlier clipping threshold

    Returns:
        test_data: dict mapping user_id -> (n_trials, 30, window_size)
        trial_info: dict mapping user_id -> list of trial indices
    """
    test_data = {}

    pattern = os.path.join(test_dir, "P_test*.mat")
    files = sorted(glob.glob(pattern))

    for fpath in files:
        user_id = os.path.basename(fpath).replace(".mat", "")
        eeg = read_test_subject(fpath)  # (30, 20000)
        eeg = clip_outliers(eeg, clip_sigma)

        # Split into 8 trials of 2500 samples each
        segment_size = 2500
        n_trials = eeg.shape[1] // segment_size

        trials = []
        for t in range(n_trials):
            start = t * segment_size
            end = start + segment_size
            trial = eeg[:, start:end]  # (30, 2500)
            trials.append(trial)

        test_data[user_id] = np.array(trials)  # (8, 30, 2500)

    return test_data
