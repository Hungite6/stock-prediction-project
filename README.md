# 📈 AI Stock Price Prediction

Dự đoán giá cổ phiếu Việt Nam sử dụng LSTM, GRU và Linear Regression.

---

## Tech Stack

| Layer | Library |
|-------|---------|
| Data | `yfinance` |
| ML/DL | `TensorFlow/Keras`, `scikit-learn` |
| Visualization | `matplotlib`, `seaborn` |
| Notebook | `Jupyter` |

---

## Project Structure

```
stock-prediction-project/
│
├── data/
│   ├── raw/          ← CSV files từ Yahoo Finance
│   └── processed/    ← (dùng nếu cần cache thêm)
│
├── models/
│   ├── lstm/         ← lstm_model.keras
│   ├── gru/          ← gru_model.keras
│   └── linear/       ← linear.pkl
│
├── notebooks/
│   └── Stock_Prediction_Models.ipynb   ← Main notebook
│
├── src/
│   ├── data_loader.py      ← Download từ yfinance
│   ├── preprocessing.py    ← Normalize, split, sequences
│   ├── train_lstm.py       ← Build + train LSTM
│   ├── train_gru.py        ← Build + train GRU
│   ├── train_linear.py     ← Linear Regression baseline
│   ├── evaluate.py         ← RMSE, MAE, MAPE + comparison
│   ├── predict.py          ← Inference helpers
│   └── utils.py            ← Plots + save/load models
│
├── reports/          ← Auto-generated charts + CSV
├── requirements.txt
├── README.md
├── main.py  ← Run full pipeline
└── app.py   ← Run streamlit app
```

---

## Models

### LSTM
- 2 LSTM layers (64 → 32 units) + Dropout(0.2)
- EarlyStopping on val_loss

### GRU
- Kiến trúc giống LSTM, dùng GRU layers
- Train nhanh hơn, ít params hơn

### Linear Regression
- Baseline model
- Input: 20-day look-back window (flattened)

---

## Dataset

| Mã | Công ty |
|----|---------|
| FPT.VN | FPT Corporation |
| VNM.VN | Vinamilk |
| HPG.VN | Hòa Phát Group |

Nguồn: Yahoo Finance (`yfinance`)  
Thời gian: 2020-01-01 → 2026-01-01

---

## Cách chạy

### 1. Cài dependencies

```bash
pip install -r requirements.txt
```

### 2. Chạy notebook 

```bash
cd notebooks
jupyter notebook Stock_Prediction_Models.ipynb
```

### 3. Hoặc chạy full pipeline

```bash
python main.py
```

### 4. Hoặc chạy streamlit app

```bash
streamlit run app.py
```

---

## Outputs

Sau khi chạy, các file được tạo trong `reports/`:

| File | Nội dung |
|------|----------|
| `{TICKER}_price.png` | Historical price chart |
| `{TICKER}_eda.png` | EDA dashboard (4 panels) |
| `{TICKER}_train_test_split.png` | Train/test split visual |
| `lstm_loss.png` / `gru_loss.png` | Loss curves |
| `training_loss_comparison.png` | LSTM vs GRU loss |
| `{TICKER}_predictions.png` | All models vs actual |
| `model_comparison_bar.png` | RMSE bar chart |
| `model_comparison.csv` | Metrics table |

---

## Kết quả mẫu (FPT.VN)

| Rank | Model | RMSE | MAE | MAPE (%) |
|------|-------|------|-----|----------|
| 1 | Linear | ~2271 | ~1698 | ~1.65 |
| 2 | GRU | ~6711 | ~5351 | ~4.85 |
| 3 | LSTM | ~15591 | ~13988 | ~12.76 |

> Kết quả thực tế sẽ khác tùy dữ liệu và lần train.
