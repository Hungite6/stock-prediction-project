"""
preprocessing.py
Normalize data, create train/test split, generate sequences for LSTM/GRU.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from debug_config import DEBUG, DEBUG_DIR


def get_close_prices(df: pd.DataFrame) -> np.ndarray:
    """Extract Close prices as numpy array."""
    return df["Close"].values.reshape(-1, 1)


FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "MA5", "MA10", "MA20", "EMA12", "EMA26",
    "RSI14", "MACD", "MACD_Signal",
    "Bollinger_Upper", "Bollinger_Lower", "Daily_Return",
]


def _debug(message: str) -> None:
    if DEBUG:
        print(message)


def _ensure_debug_dir() -> None:
    if DEBUG:
        os.makedirs(DEBUG_DIR, exist_ok=True)


def _as_timestamp(value) -> pd.Timestamp:
    return pd.Timestamp(value)


def _check_finite_df(df: pd.DataFrame, name: str) -> None:
    if not DEBUG:
        return

    nan_mask = df.isna()
    if nan_mask.any().any():
        rows, cols = np.where(nan_mask.values)
        details = [
            f"row={df.index[r]}, column={df.columns[c]}"
            for r, c in zip(rows[:20], cols[:20])
        ]
        raise ValueError(f"[DEBUG][{name}] NaN detected: {details}")

    numeric = df.select_dtypes(include=[np.number])
    inf_mask = np.isinf(numeric.values)
    if inf_mask.any():
        rows, cols = np.where(inf_mask)
        details = [
            f"row={numeric.index[r]}, column={numeric.columns[c]}"
            for r, c in zip(rows[:20], cols[:20])
        ]
        raise ValueError(f"[DEBUG][{name}] Inf detected: {details}")


def _check_finite_array(arr: np.ndarray, name: str) -> None:
    if not DEBUG:
        return
    arr = np.asarray(arr)
    if np.isnan(arr).any():
        first = np.argwhere(np.isnan(arr))[0].tolist()
        raise ValueError(f"[DEBUG][{name}] NaN detected at index {first}")
    if np.isinf(arr).any():
        first = np.argwhere(np.isinf(arr))[0].tolist()
        raise ValueError(f"[DEBUG][{name}] Inf detected at index {first}")


def _print_scaled_stats(values: np.ndarray, columns: list[str], name: str) -> None:
    if not DEBUG:
        return
    stats = pd.DataFrame(values, columns=columns).agg(["min", "max", "mean", "std"]).T
    _debug(f"\n[DEBUG][{name}] scaled feature stats:\n{stats.to_string()}")


def _save_feature_engineering_debug(feat_df: pd.DataFrame) -> None:
    if not DEBUG:
        return
    _ensure_debug_dir()

    _debug("\n[DEBUG][Feature Engineering] Final features:")
    for col in FEATURE_COLUMNS:
        _debug(f"  - {col}")

    for col in FEATURE_COLUMNS:
        nan_count = int(feat_df[col].isna().sum())
        variance = float(feat_df[col].var())
        _debug(f"[DEBUG][Feature] {col}: nan={nan_count}, variance={variance:.10f}")
        if np.isclose(variance, 0.0, atol=1e-12):
            _debug(f"[DEBUG][WARNING] Feature variance near zero: {col}")

    corr = feat_df[FEATURE_COLUMNS].corrwith(feat_df["Close"]).sort_values(ascending=False)
    _debug(f"\n[DEBUG][Feature Engineering] Correlation with target Close:\n{corr.to_string()}")
    corr.to_csv(os.path.join(DEBUG_DIR, "feature_target_correlation.csv"), header=["correlation"])

    plt.figure(figsize=(12, 10))
    sns.heatmap(feat_df[FEATURE_COLUMNS].corr(), cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    path = os.path.join(DEBUG_DIR, "feature_correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    _debug(f"[DEBUG] Saved heatmap -> {path}")


def _save_scaled_histograms(values: np.ndarray, columns: list[str], prefix: str) -> None:
    if not DEBUG:
        return
    _ensure_debug_dir()
    frame = pd.DataFrame(values, columns=columns)
    for col in columns:
        plt.figure(figsize=(7, 4))
        plt.hist(frame[col].dropna(), bins=40, color="#3498db", alpha=0.85)
        plt.title(f"{prefix} {col} histogram")
        plt.tight_layout()
        safe_col = col.replace("/", "_").replace(" ", "_")
        plt.savefig(os.path.join(DEBUG_DIR, f"{prefix.lower()}_{safe_col}_hist.png"), dpi=150)
        plt.close()


def _debug_distribution_shift(
    train_features_raw: np.ndarray,
    test_features_raw: np.ndarray,
    train_target_raw: np.ndarray,
    test_target_raw: np.ndarray,
    train_target_scaled: np.ndarray,
    test_target_scaled: np.ndarray,
) -> None:
    if not DEBUG:
        return
    _ensure_debug_dir()

    train_stats = pd.DataFrame(train_features_raw, columns=FEATURE_COLUMNS).agg(["min", "max", "mean", "std"]).T
    test_stats = pd.DataFrame(test_features_raw, columns=FEATURE_COLUMNS).agg(["min", "max", "mean", "std"]).T
    shift = train_stats.add_prefix("train_").join(test_stats.add_prefix("test_"))
    shift["test_min_below_train"] = shift["test_min"] < shift["train_min"]
    shift["test_max_above_train"] = shift["test_max"] > shift["train_max"]
    shift_path = os.path.join(DEBUG_DIR, "train_test_feature_distribution_shift.csv")
    shift.to_csv(shift_path)
    _debug(f"\n[DEBUG][DistributionShift] Feature train/test stats:\n{shift.to_string()}")
    for feature, row in shift.iterrows():
        if row["test_min_below_train"] or row["test_max_above_train"]:
            _debug(
                f"[DEBUG][WARNING][DistributionShift] {feature} test range outside train range: "
                f"train=[{row['train_min']:.6f}, {row['train_max']:.6f}], "
                f"test=[{row['test_min']:.6f}, {row['test_max']:.6f}]"
            )

    target_stats = pd.DataFrame({
        "split": ["train_raw", "test_raw", "train_scaled", "test_scaled"],
        "min": [
            float(train_target_raw.min()),
            float(test_target_raw.min()),
            float(train_target_scaled.min()),
            float(test_target_scaled.min()),
        ],
        "max": [
            float(train_target_raw.max()),
            float(test_target_raw.max()),
            float(train_target_scaled.max()),
            float(test_target_scaled.max()),
        ],
        "mean": [
            float(train_target_raw.mean()),
            float(test_target_raw.mean()),
            float(train_target_scaled.mean()),
            float(test_target_scaled.mean()),
        ],
        "std": [
            float(train_target_raw.std()),
            float(test_target_raw.std()),
            float(train_target_scaled.std()),
            float(test_target_scaled.std()),
        ],
    })
    target_stats.to_csv(os.path.join(DEBUG_DIR, "train_test_target_distribution_shift.csv"), index=False)
    _debug(f"\n[DEBUG][DistributionShift] Target train/test stats:\n{target_stats.to_string(index=False)}")
    if test_target_raw.min() < train_target_raw.min() or test_target_raw.max() > train_target_raw.max():
        _debug(
            "[DEBUG][WARNING][DistributionShift] Test Close is outside train Close range: "
            f"train=[{float(train_target_raw.min()):.6f}, {float(train_target_raw.max()):.6f}], "
            f"test=[{float(test_target_raw.min()):.6f}, {float(test_target_raw.max()):.6f}]"
        )
    if test_target_scaled.min() < 0 or test_target_scaled.max() > 1:
        _debug(
            "[DEBUG][WARNING][DistributionShift] Test target scaled outside [0, 1]: "
            f"min={float(test_target_scaled.min()):.6f}, max={float(test_target_scaled.max()):.6f}"
        )

    plt.figure(figsize=(8, 4))
    plt.hist(train_target_raw.flatten(), bins=40, alpha=0.65, label="Train Close")
    plt.hist(test_target_raw.flatten(), bins=40, alpha=0.65, label="Test Close")
    plt.title("Train/Test Close Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DEBUG_DIR, "train_test_close_hist.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(train_target_scaled.flatten(), bins=40, alpha=0.65, label="Train target scaled")
    plt.hist(test_target_scaled.flatten(), bins=40, alpha=0.65, label="Test target scaled")
    plt.title("Train/Test Target Scaled Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DEBUG_DIR, "train_test_target_scaled_hist.png"), dpi=150)
    plt.close()


def _debug_scaler_roundtrip(raw_values: np.ndarray, scaler, name: str, columns: list[str]) -> None:
    if not DEBUG:
        return
    _debug(f"\n[DEBUG][{name}] data_min_: {scaler.data_min_}")
    _debug(f"[DEBUG][{name}] data_max_: {scaler.data_max_}")
    _debug(f"[DEBUG][{name}] scale_: {scaler.scale_}")

    sample_count = min(5, len(raw_values))
    if sample_count == 0:
        return
    rng = np.random.default_rng(42)
    indices = rng.choice(len(raw_values), size=sample_count, replace=False)
    scaled = scaler.transform(raw_values[indices])
    restored = scaler.inverse_transform(scaled)
    max_abs_diff = float(np.max(np.abs(restored - raw_values[indices])))
    _debug(f"[DEBUG][{name}] roundtrip indices: {indices.tolist()}")
    _debug(f"[DEBUG][{name}] original:\n{pd.DataFrame(raw_values[indices], columns=columns).to_string(index=False)}")
    _debug(f"[DEBUG][{name}] scaled:\n{pd.DataFrame(scaled, columns=columns).to_string(index=False)}")
    _debug(f"[DEBUG][{name}] inverse:\n{pd.DataFrame(restored, columns=columns).to_string(index=False)}")
    assert np.allclose(restored, raw_values[indices], atol=1e-8), (
        f"[DEBUG][{name}] inverse(scale(x)) mismatch, max_abs_diff={max_abs_diff}"
    )


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add OHLCV + technical indicators to the raw DataFrame.
    Drops warm-up rows containing NaN.
    """
    if DEBUG:
        _debug(f"[DEBUG][Preprocessing] Raw shape: {df.shape}")
        _check_finite_df(df, "raw_df")

    out = df.copy()

    out["MA5"] = out["Close"].rolling(5).mean()
    out["MA10"] = out["Close"].rolling(10).mean()
    out["MA20"] = out["Close"].rolling(20).mean()
    out["EMA12"] = out["Close"].ewm(span=12, adjust=False).mean()
    out["EMA26"] = out["Close"].ewm(span=26, adjust=False).mean()

    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI14"] = (100 - (100 / (1 + rs))).fillna(50)

    out["MACD"] = out["EMA12"] - out["EMA26"]
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()

    ma20_std = out["Close"].rolling(20).std()
    out["Bollinger_Upper"] = out["MA20"] + 2 * ma20_std
    out["Bollinger_Lower"] = out["MA20"] - 2 * ma20_std

    out["Daily_Return"] = out["Close"].pct_change()

    out = out.dropna()
    print(f"[Preprocessing] Feature engineering done. Rows after dropna: {len(out)}")

    if DEBUG:
        _debug(f"[DEBUG][Preprocessing] Shape after feature engineering: {out.shape}")
        _debug(f"[DEBUG][Preprocessing] Final feature columns: {FEATURE_COLUMNS}")
        _check_finite_df(out[FEATURE_COLUMNS], "feature_engineered_df")
        _save_feature_engineering_debug(out)

    return out


def create_sequences(
    features: np.ndarray,
    target: np.ndarray,
    seq_length: int = 20,
    timestamps=None,
    return_timestamps: bool = False,
):
    """
    Convert multivariate time series into (X, y) sequences.
    """
    X, y = [], []
    target_timestamps, sequence_start_timestamps, sequence_end_timestamps = [], [], []
    for i in range(seq_length, len(features)):
        X.append(features[i - seq_length:i, :])
        y.append(target[i, 0])
        if timestamps is not None:
            target_timestamps.append(_as_timestamp(timestamps[i]))
            sequence_start_timestamps.append(_as_timestamp(timestamps[i - seq_length]))
            sequence_end_timestamps.append(_as_timestamp(timestamps[i - 1]))

    X = np.array(X)
    y = np.array(y)
    print(f"[Preprocessing] Created sequences: X={X.shape}, y={y.shape}")

    if return_timestamps:
        return (
            X,
            y,
            np.array(target_timestamps, dtype="datetime64[ns]"),
            np.array(sequence_start_timestamps, dtype="datetime64[ns]"),
            np.array(sequence_end_timestamps, dtype="datetime64[ns]"),
        )
    return X, y


def _debug_sequence_timestamps(label: str, ts, seq_start_ts, seq_end_ts) -> None:
    if not DEBUG or len(ts) == 0:
        return
    _debug(f"\n[DEBUG][Timestamp] {label} first target timestamp: {pd.Timestamp(ts[0])}")
    _debug(f"[DEBUG][Timestamp] {label} last target timestamp: {pd.Timestamp(ts[-1])}")
    for i in range(min(5, len(ts))):
        _debug(
            f"[DEBUG][Timestamp] {label} sample {i}: "
            f"sequence_start={pd.Timestamp(seq_start_ts[i])}, "
            f"sequence_end={pd.Timestamp(seq_end_ts[i])}, "
            f"target={pd.Timestamp(ts[i])}"
        )


def _assert_next_row_targets(all_timestamps, target_ts, seq_end_ts, label: str) -> None:
    if not DEBUG:
        return
    timestamp_to_pos = {pd.Timestamp(ts): pos for pos, ts in enumerate(all_timestamps)}
    for i, (end_ts, tgt_ts) in enumerate(zip(seq_end_ts, target_ts)):
        end_pos = timestamp_to_pos[pd.Timestamp(end_ts)]
        expected = pd.Timestamp(all_timestamps[end_pos + 1])
        actual = pd.Timestamp(tgt_ts)
        if actual != expected:
            raise AssertionError(
                f"[DEBUG][Timestamp] {label} sample {i} target misaligned: "
                f"sequence_end={pd.Timestamp(end_ts)}, expected_next_row={expected}, actual={actual}"
            )


def prepare_data_for_training(df: pd.DataFrame, seq_length: int = 20, train_ratio: float = 0.8):
    """
    Full pipeline: raw DataFrame -> ready-to-train arrays.
    Correct time-series order: feature engineering -> time split ->
    fit scaler on TRAIN ONLY -> transform train/test -> create sequences.
    """
    _ensure_debug_dir()
    feat_df = add_technical_indicators(df)

    split_idx = int(len(feat_df) * train_ratio)
    train_df = feat_df.iloc[:split_idx]
    test_df = feat_df.iloc[split_idx:]
    print(f"[Preprocessing] Train size: {len(train_df)} | Test size: {len(test_df)}")

    if DEBUG:
        _debug(f"[DEBUG][Split] feat_df shape before split: {feat_df.shape}")
        _debug(f"[DEBUG][Split] train_df shape: {train_df.shape}")
        _debug(f"[DEBUG][Split] test_df shape: {test_df.shape}")
        _check_finite_df(train_df[FEATURE_COLUMNS], "train_features_before_scaling")
        _check_finite_df(test_df[FEATURE_COLUMNS], "test_features_before_scaling")

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    train_features_raw = train_df[FEATURE_COLUMNS].values
    test_features_raw = test_df[FEATURE_COLUMNS].values
    train_features_scaled = feature_scaler.fit_transform(train_features_raw)
    test_features_scaled = feature_scaler.transform(test_features_raw)

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    train_target_raw = train_df[["Close"]].values
    test_target_raw = test_df[["Close"]].values
    train_target_scaled = target_scaler.fit_transform(train_target_raw)
    test_target_scaled = target_scaler.transform(test_target_raw)

    if DEBUG:
        _debug(f"[DEBUG][Scaling] train_features raw -> scaled: {train_features_raw.shape} -> {train_features_scaled.shape}")
        _debug(f"[DEBUG][Scaling] test_features raw -> scaled: {test_features_raw.shape} -> {test_features_scaled.shape}")
        _debug(f"[DEBUG][Scaling] train_target raw -> scaled: {train_target_raw.shape} -> {train_target_scaled.shape}")
        _debug(f"[DEBUG][Scaling] test_target raw -> scaled: {test_target_raw.shape} -> {test_target_scaled.shape}")
        _check_finite_array(train_features_scaled, "train_features_scaled")
        _check_finite_array(test_features_scaled, "test_features_scaled")
        _check_finite_array(train_target_scaled, "train_target_scaled")
        _check_finite_array(test_target_scaled, "test_target_scaled")
        _print_scaled_stats(train_features_scaled, FEATURE_COLUMNS, "train_features")
        _print_scaled_stats(test_features_scaled, FEATURE_COLUMNS, "test_features")
        _print_scaled_stats(train_target_scaled, ["Close"], "train_target")
        _print_scaled_stats(test_target_scaled, ["Close"], "test_target")
        _debug_scaler_roundtrip(train_features_raw, feature_scaler, "feature_scaler", FEATURE_COLUMNS)
        _debug_scaler_roundtrip(train_target_raw, target_scaler, "target_scaler", ["Close"])
        _debug_distribution_shift(
            train_features_raw,
            test_features_raw,
            train_target_raw,
            test_target_raw,
            train_target_scaled,
            test_target_scaled,
        )
        _save_scaled_histograms(train_features_scaled, FEATURE_COLUMNS, "scaled_feature")
        _save_scaled_histograms(train_target_scaled, ["Close"], "scaled_target")

    train_ts = train_df.index.to_numpy()
    test_ts = test_df.index.to_numpy()

    X_train, y_train, y_train_ts, train_seq_start_ts, train_seq_end_ts = create_sequences(
        train_features_scaled,
        train_target_scaled,
        seq_length,
        timestamps=train_ts,
        return_timestamps=True,
    )

    test_features_bridged = np.vstack([train_features_scaled[-seq_length:], test_features_scaled])
    test_target_bridged = np.vstack([train_target_scaled[-seq_length:], test_target_scaled])
    test_timestamps_bridged = np.concatenate([train_ts[-seq_length:], test_ts])
    X_test, y_test, y_test_ts, test_seq_start_ts, test_seq_end_ts = create_sequences(
        test_features_bridged,
        test_target_bridged,
        seq_length,
        timestamps=test_timestamps_bridged,
        return_timestamps=True,
    )

    X_train_lin = X_train.reshape(X_train.shape[0], -1)
    X_test_lin = X_test.reshape(X_test.shape[0], -1)
    y_train_lin = y_train
    y_test_lin = y_test

    if DEBUG:
        _check_finite_array(X_train, "X_train")
        _check_finite_array(y_train, "y_train")
        _check_finite_array(X_test, "X_test")
        _check_finite_array(y_test, "y_test")
        _debug(f"\n[DEBUG][Sequences] X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        _debug(f"[DEBUG][Sequences] X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        _debug(f"[DEBUG][Linear] X_train_lin shape: {X_train_lin.shape}, X_test_lin shape: {X_test_lin.shape}")
        _debug(f"[DEBUG][Samples] X_train first 5:\n{X_train[:5]}")
        _debug(f"[DEBUG][Samples] y_train first 5: {y_train[:5]}")
        _debug(f"[DEBUG][Samples] X_test first 5:\n{X_test[:5]}")
        _debug(f"[DEBUG][Samples] y_test first 5: {y_test[:5]}")
        if len(X_train):
            first_sample = pd.DataFrame(X_train[0], columns=FEATURE_COLUMNS)
            _debug(f"[DEBUG][Samples] Full first X_train sample features:\n{first_sample.to_string(index=False)}")

        _debug_sequence_timestamps("train", y_train_ts, train_seq_start_ts, train_seq_end_ts)
        _debug_sequence_timestamps("test", y_test_ts, test_seq_start_ts, test_seq_end_ts)
        _assert_next_row_targets(train_ts, y_train_ts, train_seq_end_ts, "train")
        _assert_next_row_targets(test_timestamps_bridged, y_test_ts, test_seq_end_ts, "test")

        _debug("\n[DEBUG][Bridge]")
        _debug(f"Timestamp last train row: {pd.Timestamp(train_ts[-1])}")
        _debug(f"Timestamp first test row: {pd.Timestamp(test_ts[0]) if len(test_ts) else 'EMPTY'}")
        first_test_sequence_ts = test_timestamps_bridged[:seq_length]
        _debug(f"First test sequence uses: {pd.Timestamp(first_test_sequence_ts[0])} -> {pd.Timestamp(first_test_sequence_ts[-1])}")
        _debug(f"First test target timestamp: {pd.Timestamp(y_test_ts[0]) if len(y_test_ts) else 'EMPTY'}")
        expected_bridge = train_ts[-seq_length:]
        if not np.array_equal(first_test_sequence_ts, expected_bridge):
            raise AssertionError("[DEBUG][Bridge] First test sequence does not contain the last seq_length train rows.")
        _debug("[DEBUG][Bridge] First test sequence contains exactly the last seq_length train rows.")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_lin": X_train_lin,
        "y_train_lin": y_train_lin,
        "X_test_lin": X_test_lin,
        "y_test_lin": y_test_lin,
        "scaler": target_scaler,
        "feature_scaler": feature_scaler,
        "scaled_data": np.vstack([train_features_scaled, test_features_scaled]),
        "train_size": len(train_df),
        "seq_length": seq_length,
        "num_features": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "y_train_timestamps": y_train_ts,
        "y_test_timestamps": y_test_ts,
        "train_sequence_start_timestamps": train_seq_start_ts,
        "train_sequence_end_timestamps": train_seq_end_ts,
        "test_sequence_start_timestamps": test_seq_start_ts,
        "test_sequence_end_timestamps": test_seq_end_ts,
    }
