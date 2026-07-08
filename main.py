"""
main.py
Entry point: chạy toàn bộ pipeline từ download → train → evaluate → save.
Usage: python main.py
"""

import sys
import os

# Add src/ to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader   import load_stock_data, save_raw_data
from preprocessing import prepare_data_for_training
from train_lstm    import train_lstm
from train_gru     import train_gru
from train_linear  import train_linear
from predict       import predict_keras, predict_linear
from evaluate      import inverse_scale, calculate_metrics, compare_models, print_comparison
from utils         import (
    save_keras_model, save_sklearn_model,
    plot_stock_price, plot_predictions, plot_loss, plot_comparison_bar,
)

# ──────────────────────────────────────────────
# CONFIG — chỉnh ở đây nếu muốn thay đổi
# ──────────────────────────────────────────────
TICKER      = "FPT.VN"
START_DATE  = "2020-01-01"
END_DATE    = "2026-01-01"
SEQ_LENGTH  = 20          # look-back window (days)
EPOCHS      = 150
BATCH_SIZE  = 16


def main():
    print("\n" + "=" * 60)
    print("   AI STOCK PREDICTION — FULL PIPELINE")
    print("=" * 60)

    # ── PHASE 1: Load Data ──────────────────────────────────────
    print("\n[Phase 1] Loading data ...")
    df = load_stock_data(TICKER, START_DATE, END_DATE)
    save_raw_data(df, TICKER, save_dir="data/raw")
    plot_stock_price(df, TICKER, save_path=f"reports/{TICKER}_price.png")

    # ── PHASE 2: Preprocessing ──────────────────────────────────
    print("\n[Phase 2] Preprocessing ...")
    data = prepare_data_for_training(df, seq_length=SEQ_LENGTH)

    X_train     = data["X_train"]
    y_train     = data["y_train"]
    X_test      = data["X_test"]
    y_test      = data["y_test"]
    X_train_lin = data["X_train_lin"]
    y_train_lin = data["y_train_lin"]
    X_test_lin  = data["X_test_lin"]
    y_test_lin  = data["y_test_lin"]
    scaler      = data["scaler"]
    y_test_timestamps = data.get("y_test_timestamps")

    # ── PHASE 3: Train LSTM ──────────────────────────────────────
    print("\n[Phase 3] Training LSTM ...")
    lstm_model, lstm_history = train_lstm(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
    save_keras_model(lstm_model, "models/lstm/lstm_model.keras")

    # ── PHASE 4: Train GRU ───────────────────────────────────────
    print("\n[Phase 4] Training GRU ...")
    gru_model, gru_history = train_gru(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
    save_keras_model(gru_model, "models/gru/gru_model.keras")

    # ── PHASE 5: Train Linear Regression ─────────────────────────
    print("\n[Phase 5] Training Linear Regression ...")
    lin_model = train_linear(X_train_lin, y_train_lin)
    save_sklearn_model(lin_model, "models/linear/linear.pkl")

    # ── PHASE 6: Predict ─────────────────────────────────────────
    print("\n[Phase 6] Generating predictions ...")
    lstm_preds_scaled  = predict_keras(
        lstm_model, X_test, y_true=y_test, timestamps=y_test_timestamps, scaler=scaler, model_name="LSTM"
    )
    gru_preds_scaled   = predict_keras(
        gru_model, X_test, y_true=y_test, timestamps=y_test_timestamps, scaler=scaler, model_name="GRU"
    )
    linear_preds_scaled = predict_linear(
        lin_model, X_test_lin, y_true=y_test_lin, timestamps=y_test_timestamps, scaler=scaler, model_name="Linear"
    )

    # Inverse transform back to original price scale
    y_true      = inverse_scale(y_test,              scaler)
    lstm_preds  = inverse_scale(lstm_preds_scaled,   scaler)
    gru_preds   = inverse_scale(gru_preds_scaled,    scaler)
    lin_preds   = inverse_scale(linear_preds_scaled, scaler)

    # ── PHASE 7: Evaluate ────────────────────────────────────────
    print("\n[Phase 7] Evaluating models ...")
    results = {
        "LSTM": calculate_metrics(
            y_true,
            lstm_preds,
            timestamps_true=y_test_timestamps,
            timestamps_pred=y_test_timestamps,
            model_name="LSTM",
            y_true_scaled=y_test,
            y_pred_scaled=lstm_preds_scaled,
            scaler=scaler,
        ),
        "GRU": calculate_metrics(
            y_true,
            gru_preds,
            timestamps_true=y_test_timestamps,
            timestamps_pred=y_test_timestamps,
            model_name="GRU",
            y_true_scaled=y_test,
            y_pred_scaled=gru_preds_scaled,
            scaler=scaler,
        ),
        "Linear": calculate_metrics(
            y_true,
            lin_preds,
            timestamps_true=y_test_timestamps,
            timestamps_pred=y_test_timestamps,
            model_name="Linear",
            y_true_scaled=y_test_lin,
            y_pred_scaled=linear_preds_scaled,
            scaler=scaler,
        ),
    }

    comparison_df = compare_models(results)
    print_comparison(comparison_df)

    # Save comparison table to CSV
    comparison_df.to_csv("reports/model_comparison.csv", index=False)
    print("[main] Comparison saved → reports/model_comparison.csv")

    # ── PHASE 8: Plots ───────────────────────────────────────────
    plot_loss(
        {"LSTM": lstm_history, "GRU": gru_history},
        save_path="reports/training_loss.png"
    )

    plot_predictions(
        y_true,
        {"LSTM": lstm_preds, "GRU": gru_preds, "Linear": lin_preds},
        TICKER,
        save_path=f"reports/{TICKER}_predictions.png"
    )

    plot_comparison_bar(comparison_df, save_path="reports/model_comparison_bar.png")

    print("\n✅ Pipeline complete! All outputs saved to reports/ and models/")


if __name__ == "__main__":
    main()
