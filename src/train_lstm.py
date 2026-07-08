"""
train_lstm.py
Build and train LSTM model for stock price prediction.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from debug_config import DEBUG, DEBUG_DIR, OVERFIT_DEBUG


class DebugTrainingCallback(Callback):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.epoch_start = None
        self.rows = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        elapsed = time.time() - self.epoch_start if self.epoch_start else 0.0
        lr_value = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        row = {
            "epoch": epoch + 1,
            "loss": logs.get("loss"),
            "val_loss": logs.get("val_loss"),
            "learning_rate": lr_value,
            "epoch_seconds": elapsed,
        }
        self.rows.append(row)
        if DEBUG:
            print(
                f"[DEBUG][{self.model_name}] epoch={epoch + 1} "
                f"loss={row['loss']:.8f} "
                f"val_loss={row['val_loss'] if row['val_loss'] is not None else 'NA'} "
                f"lr={lr_value:.8g} seconds={elapsed:.2f}"
            )


def _weight_stats(weights):
    rows = []
    for idx, w in enumerate(weights):
        arr = np.asarray(w)
        rows.append({
            "weight_index": idx,
            "shape": str(arr.shape),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        })
    return pd.DataFrame(rows)


def _print_weight_debug(model, initial_weights, model_name: str) -> None:
    if not DEBUG:
        return
    os.makedirs(DEBUG_DIR, exist_ok=True)
    final_weights = model.get_weights()
    stats = _weight_stats(final_weights)
    stats.to_csv(os.path.join(DEBUG_DIR, f"{model_name.lower()}_weight_stats.csv"), index=False)
    print(f"\n[DEBUG][{model_name}] Weight stats:\n{stats.to_string(index=False)}")

    deltas = []
    for idx, (before, after) in enumerate(zip(initial_weights, final_weights)):
        diff = np.asarray(after) - np.asarray(before)
        deltas.append({
            "weight_index": idx,
            "mean_abs_delta": float(np.mean(np.abs(diff))),
            "max_abs_delta": float(np.max(np.abs(diff))),
            "l2_delta": float(np.linalg.norm(diff)),
        })
    delta_df = pd.DataFrame(deltas)
    delta_df.to_csv(os.path.join(DEBUG_DIR, f"{model_name.lower()}_weight_delta.csv"), index=False)
    print(f"\n[DEBUG][{model_name}] Weight deltas:\n{delta_df.to_string(index=False)}")
    if (delta_df["max_abs_delta"] < 1e-10).all():
        print(f"[DEBUG][WARNING][{model_name}] Weights barely changed. Model may not be learning.")


def _print_gradient_norm(model, X_train: np.ndarray, y_train: np.ndarray, model_name: str) -> None:
    if not DEBUG:
        return
    sample_count = min(32, len(X_train))
    if sample_count == 0:
        return
    x = tf.convert_to_tensor(X_train[:sample_count], dtype=tf.float32)
    y = tf.convert_to_tensor(y_train[:sample_count].reshape(-1, 1), dtype=tf.float32)
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss = tf.reduce_mean(tf.keras.losses.mse(y, pred))
    grads = tape.gradient(loss, model.trainable_weights)
    norms = [float(tf.norm(g).numpy()) for g in grads if g is not None]
    print(f"[DEBUG][{model_name}] Gradient norms sample: {norms}")


def _save_history(history, debug_callback: DebugTrainingCallback, model_name: str) -> None:
    if not DEBUG:
        return
    os.makedirs(DEBUG_DIR, exist_ok=True)
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    timing_df = pd.DataFrame(debug_callback.rows)
    if not timing_df.empty:
        history_df = history_df.merge(timing_df, on="epoch", how="left", suffixes=("", "_debug"))
    path = os.path.join(DEBUG_DIR, f"{model_name.lower()}_history.csv")
    history_df.to_csv(path, index=False)
    print(f"[DEBUG][{model_name}] History saved -> {path}")


def _save_overfit_loss_curve(history, model_name: str) -> None:
    os.makedirs(DEBUG_DIR, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.title(f"{model_name} Overfit Debug Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(DEBUG_DIR, f"{model_name.lower()}_overfit_loss.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[DEBUG][{model_name}] Overfit loss curve saved -> {path}")


def build_lstm(input_shape: tuple) -> Sequential:
    """
    Build LSTM model.
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.15),
        LSTM(32, return_sequences=False),
        Dropout(0.15),
        Dense(16, activation="relu"),
        Dense(1),
    ])

    model.compile(optimizer=Adam(learning_rate=3e-4, clipnorm=1.0), loss="mse")
    model.summary()
    return model


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 150,
    batch_size: int = 16,
    validation_split: float = 0.2,
):
    """
    Train LSTM model with early stopping.
    """
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm(input_shape)
    initial_weights = [w.copy() for w in model.get_weights()]

    if OVERFIT_DEBUG:
        X_train = X_train[:200]
        y_train = y_train[:200]
        validation_split = 0.0
        epochs = max(epochs, 300)
        print("[DEBUG][LSTM][OVERFIT] Enabled: using first 200 samples, no validation, >=300 epochs.")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=25,
        min_delta=1e-4,
        restore_best_weights=True,
        verbose=1,
    )

    os.makedirs("models/lstm", exist_ok=True)
    checkpoint = ModelCheckpoint(
        filepath="models/lstm/lstm_checkpoint.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=0,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=10,
        min_lr=1e-6,
        verbose=1,
    )

    debug_callback = DebugTrainingCallback("LSTM")
    callbacks = [debug_callback]
    if not OVERFIT_DEBUG:
        callbacks = [early_stop, checkpoint, reduce_lr, debug_callback]

    print(f"\n[LSTM] Training on {len(X_train)} samples ...")
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle=False,
        callbacks=callbacks,
        verbose=1,
    )

    if OVERFIT_DEBUG:
        train_pred = model.predict(X_train, verbose=0).flatten()
        rmse = float(np.sqrt(np.mean((train_pred - y_train) ** 2)))
        print(f"[DEBUG][LSTM][OVERFIT] Train RMSE on scaled target: {rmse:.10f}")
        if rmse > 0.02:
            print("[DEBUG][WARNING][LSTM][OVERFIT] RMSE did not approach zero. Model or pipeline may be faulty.")
        _save_overfit_loss_curve(history, "LSTM")
    elif "val_loss" in history.history:
        best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
        print(f"[LSTM] Training done. Best val_loss: {min(history.history['val_loss']):.6f}")
        if early_stop.stopped_epoch > 0:
            print(f"[DEBUG][LSTM] EarlyStopping stopped at epoch {early_stop.stopped_epoch + 1}; best epoch {best_epoch}.")
    else:
        print(f"[LSTM] Training done. Final loss: {history.history['loss'][-1]:.6f}")

    _save_history(history, debug_callback, "LSTM")
    _print_weight_debug(model, initial_weights, "LSTM")
    _print_gradient_norm(model, X_train, y_train, "LSTM")
    return model, history

