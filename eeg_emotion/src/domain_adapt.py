"""Domain adaptation methods for cross-subject EEG emotion recognition.

Currently implements CORAL (CORrelation ALignment), which aligns
source and target feature distributions by matching second-order statistics.

Usage in config.yaml:
    domain_adapt:
        use_domain_adapt: true
        method: "CORAL"
"""

import numpy as np


def coral_align(X_source: np.ndarray, X_target: np.ndarray,
                lambda_reg: float = 1e-5) -> np.ndarray:
    """CORAL: align source features to target distribution.

    Transforms source features so their covariance matches target covariance.
    X_source_aligned = X_source @ Cs^{-1/2} @ Ct^{1/2}

    Args:
        X_source: source features, shape (n_source, n_features)
        X_target: target features, shape (n_target, n_features)
        lambda_reg: regularization parameter for covariance stability

    Returns:
        X_source_aligned: transformed source features, same shape as X_source
    """
    n_src, dim = X_source.shape
    n_tgt = X_target.shape[0]

    # Source covariance
    mean_src = X_source.mean(axis=0)
    Cs = ((X_source - mean_src).T @ (X_source - mean_src)) / (n_src - 1)
    Cs += lambda_reg * np.eye(dim)

    # Target covariance
    mean_tgt = X_target.mean(axis=0)
    Ct = ((X_target - mean_tgt).T @ (X_target - mean_tgt)) / (n_tgt - 1)
    Ct += lambda_reg * np.eye(dim)

    # Matrix square roots via eigendecomposition
    eig_vals_s, eig_vecs_s = np.linalg.eigh(Cs)
    eig_vals_t, eig_vecs_t = np.linalg.eigh(Ct)

    # Clamp negative eigenvalues from numerical errors
    eig_vals_s = np.maximum(eig_vals_s, 1e-10)
    eig_vals_t = np.maximum(eig_vals_t, 1e-10)

    # Cs^{-1/2}
    Cs_inv_sqrt = eig_vecs_s @ np.diag(eig_vals_s ** -0.5) @ eig_vecs_s.T
    # Ct^{1/2}
    Ct_sqrt = eig_vecs_t @ np.diag(eig_vals_t ** 0.5) @ eig_vecs_t.T

    # Transform
    X_aligned = X_source @ Cs_inv_sqrt @ Ct_sqrt

    return X_aligned


def apply_coral_for_loso(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray
) -> np.ndarray:
    """Apply CORAL alignment for LOSO evaluation.

    The test subject's (unlabeled) data serves as the target domain.

    Args:
        X_train: source (all other subjects) DE features
        y_train: source labels
        X_test: target (left-out subject) DE features, unlabeled

    Returns:
        X_train_aligned: aligned source features
    """
    return coral_align(X_train, X_test)
