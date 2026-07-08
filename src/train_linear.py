"""
train_linear.py
Train Linear Regression baseline model.
"""

import numpy as np
from sklearn.linear_model import LinearRegression


def build_linear() -> LinearRegression:
    return LinearRegression()


def train_linear(X_train: np.ndarray, y_train: np.ndarray):
    """
    Fit Linear Regression model.

    Args:
        X_train: shape (samples, seq_length)  — 2D, no timestep dim
        y_train: shape (samples,)

    Returns:
        Fitted LinearRegression model
    """
    model = build_linear()
    print(f"\n[Linear] Training on {len(X_train)} samples ...")
    model.fit(X_train, y_train)
    print("[Linear] Training done.")
    return model