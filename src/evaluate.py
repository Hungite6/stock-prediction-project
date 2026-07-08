"""
evaluate.py
Evaluate and compare model predictions.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from debug_config import DEBUG, DEBUG_DIR


def inverse_scale(predictions: np.ndarray, scaler) -> np.ndarray:
    """
    Inverse-transform normalized predictions back to original price scale.
    """
    return scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()


def _assert_aligned(y_true, y_pred, timestamps_true=None, timestamps_pred=None, model_name: str = "model") -> None:
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"[DEBUG][Evaluation][{model_name}] Length mismatch: "
            f"len(y_true)={len(y_true)}, len(y_pred)={len(y_pred)}"
        )
    if timestamps_true is not None and timestamps_pred is not None:
        true_ts = pd.to_datetime(timestamps_true)
        pred_ts = pd.to_datetime(timestamps_pred)
        if len(true_ts) != len(pred_ts) or not np.array_equal(true_ts.values, pred_ts.values):
            mismatch = np.where(true_ts.values != pred_ts.values)[0]
            first = int(mismatch[0]) if len(mismatch) else "length-only"
            raise ValueError(f"[DEBUG][Evaluation][{model_name}] Timestamp mismatch at index {first}")


def _save_debug_plots(model_name: str, y_true: np.ndarray, y_pred: np.ndarray, residual: np.ndarray) -> None:
    if not DEBUG:
        return
    os.makedirs(DEBUG_DIR, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_")

    plots = [
        ("prediction_hist", y_pred, "Prediction Histogram"),
        ("residual_hist", residual, "Residual Histogram"),
    ]
    for suffix, values, title in plots:
        plt.figure(figsize=(7, 4))
        plt.hist(values, bins=40, alpha=0.85)
        plt.title(f"{model_name} {title}")
        plt.tight_layout()
        plt.savefig(os.path.join(DEBUG_DIR, f"{safe_name}_{suffix}.png"), dpi=150)
        plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=12, alpha=0.7)
    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([lo, hi], [lo, hi], color="red", linestyle="--")
    plt.xlabel("Actual")
    plt.ylabel("Prediction")
    plt.title(f"{model_name} Prediction vs Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(DEBUG_DIR, f"{safe_name}_prediction_vs_actual.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.scatter(np.arange(len(residual)), residual, s=12, alpha=0.75)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Sample")
    plt.ylabel("Prediction - Actual")
    plt.title(f"{model_name} Residual Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(DEBUG_DIR, f"{safe_name}_residual_plot.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label="Actual", color="black", linewidth=1.8)
    plt.plot(y_pred, label="Prediction", linewidth=1.5)
    plt.title(f"{model_name} Prediction vs Actual - Full Test")
    plt.xlabel("Sample")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DEBUG_DIR, f"{safe_name}_prediction_vs_actual_full.png"), dpi=150)
    plt.close()

    window = 30 if len(residual) >= 30 else max(2, len(residual) // 3)
    rolling_rmse = pd.Series(residual ** 2).rolling(window=window).mean().pow(0.5)
    spike_threshold = rolling_rmse.mean() + 2 * rolling_rmse.std()
    spike_idx = np.where(rolling_rmse.values > spike_threshold)[0]
    plt.figure(figsize=(12, 4))
    plt.plot(rolling_rmse, label=f"Rolling RMSE (window={window})")
    if len(spike_idx):
        plt.scatter(spike_idx, rolling_rmse.iloc[spike_idx], color="red", s=18, label="RMSE spike")
    plt.title(f"{model_name} Rolling RMSE")
    plt.xlabel("Sample")
    plt.ylabel("RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DEBUG_DIR, f"{safe_name}_rolling_rmse.png"), dpi=150)
    plt.close()

    try:
        from scipy import stats

        plt.figure(figsize=(6, 6))
        stats.probplot(residual, dist="norm", plot=plt)
        plt.title(f"{model_name} Residual QQ Plot")
        plt.tight_layout()
        plt.savefig(os.path.join(DEBUG_DIR, f"{safe_name}_qq_plot.png"), dpi=150)
        plt.close()
    except Exception as exc:
        print(f"[DEBUG][Evaluation][{model_name}] QQ plot skipped: {exc}")


def _debug_sample_alignment(model_name: str, y_true: np.ndarray, y_pred: np.ndarray, timestamps) -> None:
    if not DEBUG:
        return
    starts = [
        ("first", 0),
        ("middle", max(0, len(y_true) // 2 - 2)),
        ("last", max(0, len(y_true) - 5)),
    ]
    for label, start in starts:
        end = min(start + 5, len(y_true))
        sample = pd.DataFrame({
            "index": np.arange(start, end),
            "timestamp": pd.to_datetime(timestamps[start:end]) if timestamps is not None else np.arange(start, end),
            "y_true": y_true[start:end],
            "prediction": y_pred[start:end],
            "error": y_pred[start:end] - y_true[start:end],
        })
        print(f"\n[DEBUG][Alignment][{model_name}] {label} samples:\n{sample.to_string(index=False)}")


def _debug_segment_metrics(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if not DEBUG:
        return
    rows = []
    for n in [50, 100, 150, 200, len(y_true)]:
        if n <= len(y_true):
            err_true = y_true[:n]
            err_pred = y_pred[:n]
            rows.append({
                "segment": f"first_{n}" if n != len(y_true) else "full_test",
                "start": 0,
                "end": n - 1,
                "rmse": float(np.sqrt(mean_squared_error(err_true, err_pred))),
                "mae": float(mean_absolute_error(err_true, err_pred)),
            })

    edges = np.linspace(0, len(y_true), 5, dtype=int)
    for i in range(4):
        start, end = int(edges[i]), int(edges[i + 1])
        if end > start:
            rows.append({
                "segment": f"Q{i + 1}",
                "start": start,
                "end": end - 1,
                "rmse": float(np.sqrt(mean_squared_error(y_true[start:end], y_pred[start:end]))),
                "mae": float(mean_absolute_error(y_true[start:end], y_pred[start:end])),
            })

    df = pd.DataFrame(rows)
    path = os.path.join(DEBUG_DIR, f"{model_name.lower()}_segment_metrics.csv")
    df.to_csv(path, index=False)
    print(f"\n[DEBUG][Evaluation][{model_name}] Segment RMSE/MAE:\n{df.to_string(index=False)}")
    rmse_median = float(df["rmse"].median())
    for _, row in df.iterrows():
        if rmse_median > 0 and row["rmse"] > rmse_median * 1.75:
            print(
                f"[DEBUG][WARNING][Evaluation][{model_name}] RMSE spike in {row['segment']}: "
                f"{row['rmse']:.4f} vs median segment RMSE {rmse_median:.4f}"
            )


def _debug_scaled_inverse_metrics(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_true_scaled,
    y_pred_scaled,
    scaler,
) -> None:
    if not DEBUG or y_true_scaled is None or y_pred_scaled is None or scaler is None:
        return
    y_true_scaled = np.asarray(y_true_scaled).flatten()
    y_pred_scaled = np.asarray(y_pred_scaled).flatten()
    if y_true_scaled.shape != y_pred_scaled.shape:
        raise ValueError(f"[DEBUG][Evaluation][{model_name}] Scaled shape mismatch.")

    rmse_scaled = float(np.sqrt(mean_squared_error(y_true_scaled, y_pred_scaled)))
    rmse_inverse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    scale_factor = float(scaler.data_max_[0] - scaler.data_min_[0])
    expected_inverse = rmse_scaled * scale_factor
    print(
        f"\n[DEBUG][Evaluation][{model_name}] RMSE scaled={rmse_scaled:.8f}, "
        f"inverse={rmse_inverse:.8f}, scaled*target_range={expected_inverse:.8f}"
    )
    if not np.isclose(rmse_inverse, expected_inverse, rtol=1e-5, atol=1e-5):
        print(f"[DEBUG][WARNING][Evaluation][{model_name}] Inverse RMSE does not match scaler target range.")

    rng = np.random.default_rng(123)
    n = min(10, len(y_true))
    indices = rng.choice(len(y_true), size=n, replace=False)
    original = y_true[indices].reshape(-1, 1)
    scaled = scaler.transform(original)
    restored = scaler.inverse_transform(scaled).flatten()
    max_abs_diff = float(np.max(np.abs(restored - y_true[indices])))
    check = pd.DataFrame({
        "index": indices,
        "original": y_true[indices],
        "scaled": scaled.flatten(),
        "inverse": restored,
        "abs_diff": np.abs(restored - y_true[indices]),
    })
    print(f"[DEBUG][Evaluation][{model_name}] Scaler roundtrip random samples:\n{check.to_string(index=False)}")
    assert np.allclose(restored, y_true[indices], atol=1e-8), (
        f"[DEBUG][Evaluation][{model_name}] scaler roundtrip failed, max_abs_diff={max_abs_diff}"
    )


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps_true=None,
    timestamps_pred=None,
    model_name: str = "model",
    y_true_scaled=None,
    y_pred_scaled=None,
    scaler=None,
) -> dict:
    """
    Calculate RMSE, MAE, MAPE for a single model.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"[DEBUG][Evaluation][{model_name}] Shape mismatch: "
            f"y_true.shape={y_true.shape}, y_pred.shape={y_pred.shape}"
        )
    _assert_aligned(y_true, y_pred, timestamps_true, timestamps_pred, model_name)

    residual = y_pred - y_true
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    if DEBUG:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        print(f"\n[DEBUG][Evaluation][{model_name}] RMSE: {rmse:.8f}")
        print(f"[DEBUG][Evaluation][{model_name}] MAE: {mae:.8f}")
        print(f"[DEBUG][Evaluation][{model_name}] MAPE: {mape:.8f}")
        print(f"[DEBUG][Evaluation][{model_name}] Mean Error: {np.mean(residual):.8f}")
        print(f"[DEBUG][Evaluation][{model_name}] Median Error: {np.median(residual):.8f}")
        print(f"[DEBUG][Evaluation][{model_name}] Max Error: {np.max(residual):.8f}")
        print(f"[DEBUG][Evaluation][{model_name}] Min Error: {np.min(residual):.8f}")
        print(f"[DEBUG][Evaluation][{model_name}] Std Error: {np.std(residual):.8f}")

        out = pd.DataFrame({
            "timestamp": pd.to_datetime(timestamps_true) if timestamps_true is not None else np.arange(len(y_true)),
            "y_true": y_true,
            "prediction": y_pred,
            "error": residual,
        })
        csv_path = os.path.join(DEBUG_DIR, f"{model_name.lower()}_predictions_errors.csv")
        out.to_csv(csv_path, index=False)
        print(f"[DEBUG][Evaluation][{model_name}] Prediction/error CSV saved -> {csv_path}")
        print(f"[DEBUG][Evaluation][{model_name}] First 20 rows:\n{out.head(20).to_string(index=False)}")
        _debug_sample_alignment(model_name, y_true, y_pred, timestamps_true)
        _debug_segment_metrics(model_name, y_true, y_pred)
        _debug_scaled_inverse_metrics(model_name, y_true, y_pred, y_true_scaled, y_pred_scaled, scaler)
        _save_debug_plots(model_name, y_true, y_pred, residual)

    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "mape": round(mape, 4)}


def compare_models(results: dict) -> pd.DataFrame:
    """
    Build a comparison DataFrame from multiple models.
    """
    rows = []
    for model_name, metrics in results.items():
        rows.append({
            "Model": model_name,
            "RMSE": metrics["rmse"],
            "MAE": metrics["mae"],
            "MAPE (%)": metrics["mape"],
        })

    df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    df.index = df.index + 1
    return df


def print_comparison(df: pd.DataFrame) -> None:
    """Pretty-print comparison table."""
    print("\n" + "=" * 50)
    print("         MODEL COMPARISON RESULTS")
    print("=" * 50)
    print(df.to_string(index=True))
    print("=" * 50)
    best = df.iloc[0]["Model"]
    print(f"Best model: {best}")
    print("=" * 50 + "\n")
