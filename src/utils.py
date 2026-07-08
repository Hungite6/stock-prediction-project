"""
utils.py
Plotting helpers + save/load models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib

from tensorflow.keras.models import load_model as keras_load_model


# ─────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────

def save_keras_model(model, path: str) -> None:
    """Save LSTM or GRU model to .keras file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"[Utils] Keras model saved → {path}")


def load_keras_model(path: str):
    """Load LSTM or GRU model from .keras file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    model = keras_load_model(path)
    print(f"[Utils] Keras model loaded ← {path}")
    return model


def save_sklearn_model(model, path: str) -> None:
    """Save Linear Regression model with joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[Utils] Sklearn model saved → {path}")


def load_sklearn_model(path: str):
    """Load Linear Regression model with joblib."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    model = joblib.load(path)
    print(f"[Utils] Sklearn model loaded ← {path}")
    return model


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────

def plot_stock_price(df, ticker: str, save_path: str = None):
    """Plot historical Close price."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["Close"], color="#1f77b4", linewidth=1.5)
    ax.set_title(f"{ticker} — Historical Close Price", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (VND)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Utils] Plot saved → {save_path}")
    plt.show()


def plot_predictions(
    y_true: np.ndarray,
    predictions: dict,
    ticker: str,
    save_path: str = None,
):
    """
    Plot actual vs predicted prices for multiple models.

    Args:
        y_true:      actual prices (original scale)
        predictions: {"LSTM": array, "GRU": array, "Linear": array}
        ticker:      stock ticker for title
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(y_true, label="Actual", color="black", linewidth=2)

    colors = {"LSTM": "#e74c3c", "GRU": "#2ecc71", "Linear": "#3498db"}
    styles = {"LSTM": "-",       "GRU": "--",      "Linear": "-."}

    for name, preds in predictions.items():
        color = colors.get(name, "gray")
        style = styles.get(name, "-")
        ax.plot(preds, label=name, color=color, linestyle=style, linewidth=1.5)

    ax.set_title(f"{ticker} — Model Predictions vs Actual", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Price (VND)")
    ax.legend(fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Utils] Plot saved → {save_path}")
    plt.show()


def plot_loss(histories: dict, save_path: str = None):
    """
    Plot training loss curves for Keras models.

    Args:
        histories: {"LSTM": history_object, "GRU": history_object}
    """
    fig, axes = plt.subplots(1, len(histories), figsize=(6 * len(histories), 4))
    if len(histories) == 1:
        axes = [axes]

    for ax, (name, history) in zip(axes, histories.items()):
        ax.plot(history.history["loss"],     label="Train Loss", color="#e74c3c")
        if "val_loss" in history.history:
            ax.plot(history.history["val_loss"], label="Val Loss", color="#3498db", linestyle="--")
        ax.set_title(f"{name} — Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Utils] Plot saved → {save_path}")
    plt.show()


def plot_comparison_bar(comparison_df, save_path: str = None):
    """
    Bar chart comparing RMSE across models.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    bars = ax.bar(comparison_df["Model"], comparison_df["RMSE"], color=colors[:len(comparison_df)], width=0.5)

    # Add value labels on top of bars
    for bar, val in zip(bars, comparison_df["RMSE"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    ax.set_title("Model Comparison — RMSE (lower is better)", fontsize=13, fontweight="bold")
    ax.set_ylabel("RMSE")
    ax.set_ylim(0, comparison_df["RMSE"].max() * 1.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Utils] Plot saved → {save_path}")
    plt.show()
