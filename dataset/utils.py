"""
utils.py — helper functions for dataset management and preprocessing.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# -------------------------------------------------------------------------
# 1. Label Cleaning
# -------------------------------------------------------------------------
def clean_labels(X, y, sbp_range=(80, 180), dbp_range=(50, 120), verbose=True):
    """
    Remove segments with unrealistic SBP/DBP values.

    Args:
        X (np.ndarray): input signals [N, C, L] or [N, L]
        y (np.ndarray): labels [N, 2] → columns: [SBP, DBP]
        sbp_range (tuple): acceptable range for SBP (min, max)
        dbp_range (tuple): acceptable range for DBP (min, max)
        verbose (bool): print summary if True

    Returns:
        X_clean, y_clean (np.ndarray, np.ndarray)
    """
    sbp_min, sbp_max = sbp_range
    dbp_min, dbp_max = dbp_range

    mask = (
        (y[:, 0] >= sbp_min) & (y[:, 0] <= sbp_max) &
        (y[:, 1] >= dbp_min) & (y[:, 1] <= dbp_max)
    )

    X_clean = X[mask]
    y_clean = y[mask]

    if verbose:
        removed = len(X) - len(X_clean)
        pct = 100 * removed / len(X)
        print(f"🧹 Removed {removed} outlier segments ({pct:.2f}%)")
        print(f"✅ Remaining: {len(X_clean)}")

    return X_clean, y_clean


# -------------------------------------------------------------------------
# 2. Train/Validation/Test Split
# -------------------------------------------------------------------------
def split_dataset(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42, verbose=True):
    """
    Split dataset into train, val, test sets (segment-level).

    Args:
        X (np.ndarray): inputs
        y (np.ndarray): labels
        train_ratio (float): fraction of total data for training
        val_ratio (float): fraction for validation
        test_ratio (float): fraction for testing
        seed (int): random seed
        verbose (bool): print split summary

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1 - train_ratio), random_state=seed)
    rel_val_ratio = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1 - rel_val_ratio), random_state=seed)

    if verbose:
        print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# -------------------------------------------------------------------------
# 3. Visualization Helper
# -------------------------------------------------------------------------
def plot_bp_distribution(y, title="BP Distribution", bins=50):
    """
    Plot histogram of SBP and DBP to visualize label distribution.

    Args:
        y (np.ndarray): labels [N, 2]
        title (str): plot title
        bins (int): number of histogram bins
    """
    sbp, dbp = y[:, 0], y[:, 1]

    plt.figure(figsize=(8, 4))
    plt.hist(sbp, bins=bins, alpha=0.7, label="SBP")
    plt.hist(dbp, bins=bins, alpha=0.7, label="DBP")
    plt.xlabel("Blood Pressure (mmHg)")
    plt.ylabel("Count")
    plt.legend()
    plt.title(title)
    plt.show()
