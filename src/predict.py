"""
predict.py
Generate predictions from trained models.
"""

import numpy as np
import pandas as pd

from debug_config import DEBUG


def _debug_prediction_table(
    model_name: str,
    predictions_scaled: np.ndarray,
    y_true_scaled=None,
    timestamps=None,
    scaler=None,
) -> None:
    if not DEBUG:
        return
    predictions_scaled = np.asarray(predictions_scaled).flatten()
    y_scaled = np.asarray(y_true_scaled).flatten() if y_true_scaled is not None else None

    print(
        f"\n[DEBUG][Prediction][{model_name}] prediction_scaled stats: "
        f"min={predictions_scaled.min():.8f}, max={predictions_scaled.max():.8f}, "
        f"mean={predictions_scaled.mean():.8f}, std={predictions_scaled.std():.8f}"
    )
    if y_scaled is not None:
        print(
            f"[DEBUG][Prediction][{model_name}] y_true_scaled stats: "
            f"min={y_scaled.min():.8f}, max={y_scaled.max():.8f}, "
            f"mean={y_scaled.mean():.8f}, std={y_scaled.std():.8f}"
        )
        pred_range = float(predictions_scaled.max() - predictions_scaled.min())
        true_range = float(y_scaled.max() - y_scaled.min())
        if true_range > 0 and pred_range < true_range * 0.25:
            print(
                f"[DEBUG][WARNING][Prediction][{model_name}] Predictions look flat: "
                f"pred_range={pred_range:.8f}, y_true_range={true_range:.8f}"
            )

    starts = [
        ("first", 0),
        ("middle", max(0, len(predictions_scaled) // 2 - 10)),
        ("last", max(0, len(predictions_scaled) - 20)),
    ]
    for label, start in starts:
        end = min(start + 20, len(predictions_scaled))
        table = pd.DataFrame({
            "index": np.arange(start, end),
            "timestamp": pd.to_datetime(timestamps[start:end]) if timestamps is not None else np.arange(start, end),
            "prediction_scaled": predictions_scaled[start:end],
        })
        if scaler is not None:
            table["prediction_inverse"] = scaler.inverse_transform(
                predictions_scaled[start:end].reshape(-1, 1)
            ).flatten()
        if y_scaled is not None:
            table["y_true_scaled"] = y_scaled[start:end]
            table["error_scaled"] = table["prediction_scaled"] - table["y_true_scaled"]
            if scaler is not None:
                table["y_true_inverse"] = scaler.inverse_transform(y_scaled[start:end].reshape(-1, 1)).flatten()
                table["error_inverse"] = table["prediction_inverse"] - table["y_true_inverse"]
        print(f"\n[DEBUG][Prediction][{model_name}] {label} rows {start}:{end}:\n{table.to_string(index=False)}")


def predict_keras(
    model,
    X_test: np.ndarray,
    y_true=None,
    timestamps=None,
    scaler=None,
    model_name: str = "Keras",
) -> np.ndarray:
    """
    Predict with LSTM or GRU (Keras model).

    Returns:
        predictions: shape (samples,)
    """
    preds = model.predict(X_test, verbose=0).flatten()
    _debug_prediction_table(model_name, preds, y_true_scaled=y_true, timestamps=timestamps, scaler=scaler)
    return preds


def predict_linear(
    model,
    X_test: np.ndarray,
    y_true=None,
    timestamps=None,
    scaler=None,
    model_name: str = "Linear",
) -> np.ndarray:
    """
    Predict with sklearn LinearRegression.

    Returns:
        predictions: shape (samples,)
    """
    preds = model.predict(X_test)
    _debug_prediction_table(model_name, preds, y_true_scaled=y_true, timestamps=timestamps, scaler=scaler)
    return preds
