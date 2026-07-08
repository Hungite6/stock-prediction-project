This file is a merged representation of a subset of the codebase, containing specifically included files and files not matching ignore patterns, combined into a single document by Repomix.

<file_summary>
This section contains a summary of this file.

<purpose>
This file contains a packed representation of a subset of the repository's contents that is considered the most important context.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.
</purpose>

<file_format>
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  - File path as an attribute
  - Full contents of the file
</file_format>

<usage_guidelines>
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.
</usage_guidelines>

<notes>
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Only files matching these patterns are included: README.md, requirements.txt, main.py, app.py, src/**/*.py, notebooks/*.ipynb
- Files matching these patterns are excluded: **/__pycache__/**, **/.ipynb_checkpoints/**, data/**, models/**, reports/**, *.png, *.csv, *.h5, *.pth, *.pt, *.ckpt
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)
</notes>

</file_summary>

<directory_structure>
app.py
main.py
notebooks/Stock_Prediction_Models.ipynb
README.md
requirements.txt
src/data_loader.py
src/debug_config.py
src/evaluate.py
src/predict.py
src/preprocessing.py
src/train_gru.py
src/train_linear.py
src/train_lstm.py
src/utils.py
</directory_structure>

<files>
This section contains the contents of the repository's files.

<file path="app.py">
"""
app.py
AI Stock Prediction Dashboard — Streamlit frontend.

Run with:
    streamlit run app.py

Requires the src/ directory to be on the Python path.
All heavy ML logic lives in src/; this file is UI-only.
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── global ─────────────────────────────── */
[data-testid="stAppViewContainer"] {
    background: #0d1117;
    color: #e6edf3;
}
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * {
    color: #c9d1d9 !important;
}

/* ── metric cards ───────────────────────── */
[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 12px; }
[data-testid="stMetricValue"] { color: #58a6ff !important; font-size: 28px; font-weight: 700; }
[data-testid="stMetricDelta"] { font-size: 13px; }

/* ── tabs ───────────────────────────────── */
[data-testid="stTabs"] button {
    color: #8b949e;
    border-radius: 6px 6px 0 0;
    font-weight: 500;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
    background: transparent !important;
}

/* ── buttons ────────────────────────────── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 8px 24px;
    transition: opacity .2s;
}
[data-testid="stButton"] > button:hover { opacity: .85; }

/* ── dataframe ──────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* ── section headers ────────────────────── */
.section-header {
    font-size: 18px;
    font-weight: 700;
    color: #e6edf3;
    padding: 4px 0 12px;
    border-bottom: 1px solid #21262d;
    margin-bottom: 20px;
}
.badge {
    display: inline-block;
    background: #1f6feb22;
    color: #58a6ff;
    border: 1px solid #1f6feb55;
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 12px;
    font-weight: 600;
    margin-right: 6px;
}
.info-box {
    background: #161b22;
    border: 1px solid #21262d;
    border-left: 3px solid #58a6ff;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 13px;
    color: #8b949e;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

AVAILABLE_TICKERS = {
    "FPT.VN  — FPT Corporation":   "FPT.VN",
    "VNM.VN  — Vinamilk":          "VNM.VN",
    "HPG.VN  — Hòa Phát Group":    "HPG.VN",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c9d1d9", family="Inter, sans-serif"),
    legend=dict(
        bgcolor="rgba(22,27,34,0.8)",
        bordercolor="#21262d",
        borderwidth=1,
    ),
    xaxis=dict(gridcolor="#21262d", zeroline=False),
    yaxis=dict(gridcolor="#21262d", zeroline=False),
    margin=dict(l=0, r=0, t=40, b=0),
)


# ═══════════════════════════════════════════════════════════════════════════════
#  CACHED HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def cached_load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    from data_loader import load_stock_data
    return load_stock_data(ticker, start, end)


@st.cache_data(show_spinner=False)
def cached_prepare_data(ticker: str, start: str, end: str, seq_length: int, train_ratio: float):
    from preprocessing import prepare_data_for_training
    df = cached_load_data(ticker, start, end)
    return prepare_data_for_training(df, seq_length=seq_length, train_ratio=train_ratio)


@st.cache_resource(show_spinner=False)
def load_existing_keras(path: str):
    from utils import load_keras_model
    return load_keras_model(path)


@st.cache_resource(show_spinner=False)
def load_existing_sklearn(path: str):
    from utils import load_sklearn_model
    return load_sklearn_model(path)


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTLY CHART BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_price_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        mode="lines",
        name="Close",
        line=dict(color="#58a6ff", width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>Price: %{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"{ticker} — Close Price", font=dict(size=15)),
        xaxis_title="Date",
        yaxis_title="Price (VND)",
        **PLOTLY_LAYOUT,
    )
    return fig


def build_eda_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    df = df.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    daily_ret  = df["Close"].pct_change().dropna()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Close Price + Moving Averages",
            "Trading Volume",
            "Daily Return Distribution",
            "Price vs Volume Correlation",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )

    # 1 — Close + MA
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close",
                             line=dict(color="#58a6ff", width=1.5), opacity=0.7), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20",
                             line=dict(color="#e74c3c", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50",
                             line=dict(color="#2ecc71", width=2)), row=1, col=1)

    # 2 — Volume
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                         marker_color="#8b949e", opacity=0.6), row=1, col=2)

    # 3 — Daily return histogram
    fig.add_trace(go.Histogram(x=daily_ret, name="Daily Return",
                               marker_color="#9b59b6", nbinsx=60, opacity=0.8), row=2, col=1)
    fig.add_vline(x=0, line_dash="dash", line_color="red", row=2, col=1)

    # 4 — Price vs Volume scatter
    fig.add_trace(go.Scatter(x=df["Volume"], y=df["Close"],
                             mode="markers", name="Price/Vol",
                             marker=dict(color="#f39c12", size=4, opacity=0.5)), row=2, col=2)

    fig.update_layout(
        height=600,
        title=dict(text=f"{ticker} — Exploratory Analysis", font=dict(size=15)),
        showlegend=False,
        **PLOTLY_LAYOUT,
    )
    return fig


def build_split_chart(df: pd.DataFrame, train_ratio: float, ticker: str) -> go.Figure:
    close = df["Close"].values
    split = int(len(close) * train_ratio)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index[:split], y=close[:split],
        name="Train", line=dict(color="#58a6ff", width=1.8),
    ))
    fig.add_trace(go.Scatter(
        x=df.index[split:], y=close[split:],
        name="Test", line=dict(color="#e74c3c", width=1.8),
    ))
    fig.add_vline(x=df.index[split], line_dash="dot", line_color="#8b949e",
                  annotation_text=" Split", annotation_font_color="#8b949e")
    fig.update_layout(
        title=dict(text=f"{ticker} — Train / Test Split ({int(train_ratio*100)}/{int((1-train_ratio)*100)})",
                   font=dict(size=15)),
        xaxis_title="Date", yaxis_title="Price (VND)",
        **PLOTLY_LAYOUT,
    )
    return fig


def build_loss_chart(histories: dict) -> go.Figure:
    colors = {"LSTM": "#e74c3c", "GRU": "#2ecc71"}
    n = len(histories)
    fig = make_subplots(rows=1, cols=n,
                        subplot_titles=[f"{k} — Loss" for k in histories])
    for col, (name, hist) in enumerate(histories.items(), 1):
        c = colors.get(name, "#58a6ff")
        fig.add_trace(go.Scatter(y=hist.history["loss"],     name=f"{name} Train",
                                 line=dict(color=c, width=2)), row=1, col=col)
        fig.add_trace(go.Scatter(y=hist.history["val_loss"], name=f"{name} Val",
                                 line=dict(color=c, width=2, dash="dot")), row=1, col=col)
    fig.update_layout(height=320, title=dict(text="Training Loss Curves", font=dict(size=15)),
                      **PLOTLY_LAYOUT)
    return fig


def build_predictions_chart(y_true, predictions: dict, ticker: str, results: dict) -> go.Figure:
    colors = {"LSTM": "#e74c3c", "GRU": "#2ecc71", "Linear": "#f39c12"}
    dashes  = {"LSTM": "solid",  "GRU": "dot",     "Linear": "dash"}
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_true, name="Actual",
                             line=dict(color="#e6edf3", width=2.5)))
    for name, preds in predictions.items():
        rmse = results[name]["rmse"] if name in results else 0
        fig.add_trace(go.Scatter(
            y=preds,
            name=f"{name} (RMSE={rmse:.2f})",
            line=dict(color=colors.get(name, "#58a6ff"),
                      width=1.8,
                      dash=dashes.get(name, "solid")),
            opacity=0.9,
        ))
    fig.update_layout(
        title=dict(text=f"{ticker} — Model Predictions vs Actual", font=dict(size=15)),
        xaxis_title="Time Steps (Test Period)",
        yaxis_title="Price (VND)",
        **PLOTLY_LAYOUT,
    )
    return fig


def build_rmse_bar(comparison_df: pd.DataFrame) -> go.Figure:
    colors = ["#e74c3c", "#2ecc71", "#f39c12"]
    fig = go.Figure(go.Bar(
        x=comparison_df["Model"],
        y=comparison_df["RMSE"],
        marker_color=colors[:len(comparison_df)],
        text=comparison_df["RMSE"].round(4),
        textposition="outside",
        width=0.45,
    ))
    fig.update_layout(
        title=dict(text="Model Comparison — RMSE (lower = better)", font=dict(size=15)),
        **PLOTLY_LAYOUT,
    )
    fig.update_yaxes(title="RMSE", range=[0, comparison_df["RMSE"].max() * 1.35])
    return fig


def build_multi_bar(multi_results: dict) -> go.Figure:
    tickers = list(multi_results.keys())
    metrics = ["rmse", "mae", "mape"]
    labels  = ["RMSE", "MAE", "MAPE (%)"]
    colors  = ["#58a6ff", "#2ecc71", "#f39c12"]

    fig = make_subplots(rows=1, cols=3, subplot_titles=labels)
    for col, (metric, label, color) in enumerate(zip(metrics, labels, colors), 1):
        vals = [multi_results[t][metric] for t in tickers]
        fig.add_trace(go.Bar(x=tickers, y=vals, name=label,
                             marker_color=color, text=[round(v, 2) for v in vals],
                             textposition="outside"), row=1, col=col)
    fig.update_layout(height=360, showlegend=False,
                      title=dict(text="Multi-Stock LSTM Metrics", font=dict(size=15)),
                      **PLOTLY_LAYOUT)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.divider()

    # Stock selection
    st.markdown("**Stock Ticker**")
    ticker_label = st.selectbox("", list(AVAILABLE_TICKERS.keys()), label_visibility="collapsed")
    TICKER = AVAILABLE_TICKERS[ticker_label]

    st.markdown("**Date Range**")
    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input("Start", value=pd.Timestamp("2020-01-01"),
                                   min_value=pd.Timestamp("2015-01-01"),
                                   label_visibility="visible")
    with col_e:
        end_date = st.date_input("End", value=pd.Timestamp("2025-01-01"),
                                 label_visibility="visible")

    st.markdown("**Model Parameters**")
    seq_length   = st.slider("Sequence Length (days)", 20, 120, 60, step=10)
    train_ratio  = st.slider("Train Ratio", 0.6, 0.9, 0.8, step=0.05)
    epochs       = st.slider("Epochs", 10, 200, 150, step=10)
    batch_size   = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)

    st.markdown("**Models to Train**")
    use_lstm   = st.checkbox("LSTM",              value=True)
    use_gru    = st.checkbox("GRU",               value=True)
    use_linear = st.checkbox("Linear Regression", value=True)

    st.divider()
    st.markdown(
        '<div style="font-size:11px;color:#484f58;text-align:center;">'
        'AI Stock Prediction · Vietnamese Markets</div>',
        unsafe_allow_html=True,
    )

START = str(start_date)
END   = str(end_date)


# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="padding:8px 0 4px">
  <span style="font-size:34px;font-weight:800;color:#e6edf3;letter-spacing:-1px;">
    📈 AI Stock Predictor
  </span>
  <br>
  <span style="font-size:14px;color:#8b949e;">
    Deep learning forecasts for Vietnamese equities — LSTM · GRU · Linear Regression
  </span>
</div>
""", unsafe_allow_html=True)

st.markdown(
    f'<span class="badge">LSTM</span>'
    f'<span class="badge">GRU</span>'
    f'<span class="badge">Linear</span>'
    f'<span class="badge">yfinance</span>'
    f'<span class="badge">TensorFlow</span>',
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

data_ok = False
df = None

with st.spinner(f"Loading {TICKER} data…"):
    try:
        df = cached_load_data(TICKER, START, END)
        data_ok = True
    except Exception as e:
        st.error(f"⚠️ Could not load data for **{TICKER}**: {e}")

if data_ok and df is not None and not df.empty:
    # ── KPI row ───────────────────────────────────────────────────────────────
    latest_close  = float(df["Close"].iloc[-1])
    prev_close    = float(df["Close"].iloc[-2]) if len(df) > 1 else latest_close
    delta_pct     = (latest_close - prev_close) / prev_close * 100
    avg_volume    = int(df["Volume"].mean())
    total_rows    = len(df)

    split_idx     = int(total_rows * train_ratio)
    n_train       = split_idx
    n_test        = total_rows - split_idx

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Ticker",      TICKER)
    k2.metric("Latest Close", f"{latest_close:,.0f}", f"{delta_pct:+.2f}%")
    k3.metric("Total Rows",   f"{total_rows:,}")
    k4.metric("Train / Test", f"{n_train} / {n_test}")
    k5.metric("Avg Volume",   f"{avg_volume:,}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    #  TABS
    # ═══════════════════════════════════════════════════════════════════════════
    tab_data, tab_train, tab_results, tab_multi = st.tabs([
        "📊 Data & EDA",
        "🏋️ Train Models",
        "📉 Results",
        "🌐 Multi-Stock",
    ])

    # ─────────────────────────────────────────────────────────────────────────
    #  TAB 1 — DATA & EDA
    # ─────────────────────────────────────────────────────────────────────────
    with tab_data:
        st.markdown('<div class="section-header">Historical Price</div>', unsafe_allow_html=True)
        st.plotly_chart(build_price_chart(df, TICKER), use_container_width=True)

        st.markdown('<div class="section-header">Train / Test Split</div>', unsafe_allow_html=True)
        st.plotly_chart(build_split_chart(df, train_ratio, TICKER), use_container_width=True)

        st.markdown('<div class="section-header">Exploratory Analysis</div>', unsafe_allow_html=True)
        st.plotly_chart(build_eda_chart(df, TICKER), use_container_width=True)

        with st.expander("📋 Raw Data Sample"):
            st.dataframe(
                df.tail(50).style.format("{:,.2f}"),
                use_container_width=True,
            )

    # ─────────────────────────────────────────────────────────────────────────
    #  TAB 2 — TRAINING
    # ─────────────────────────────────────────────────────────────────────────
    with tab_train:
        st.markdown('<div class="section-header">Model Training</div>', unsafe_allow_html=True)

        # Check for existing saved models
        lstm_path   = os.path.join(ROOT, "models", "lstm",   "lstm_model.keras")
        gru_path    = os.path.join(ROOT, "models", "gru",    "gru_model.keras")
        linear_path = os.path.join(ROOT, "models", "linear", "linear.pkl")

        models_exist = {
            "LSTM":   os.path.exists(lstm_path),
            "GRU":    os.path.exists(gru_path),
            "Linear": os.path.exists(linear_path),
        }

        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(
                '<div class="info-box">'
                'Click <b>Train Models</b> to run the full pipeline. '
                'Training uses the parameters set in the sidebar. '
                'Saved models will be overwritten.'
                '</div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown("**Saved models found:**")
            for name, found in models_exist.items():
                icon = "✅" if found else "❌"
                st.markdown(f"{icon} &nbsp; {name}", unsafe_allow_html=True)

        train_btn = st.button("🏋️ Train Models", type="primary")

        if train_btn:
            if not (use_lstm or use_gru or use_linear):
                st.warning("Select at least one model in the sidebar.")
            else:
                from preprocessing import prepare_data_for_training
                from predict       import predict_keras, predict_linear
                from evaluate      import inverse_scale, calculate_metrics, compare_models
                from utils         import save_keras_model, save_sklearn_model

                # ── preprocess ────────────────────────────────────────────
                with st.spinner("Preprocessing data…"):
                    data = prepare_data_for_training(df, seq_length=seq_length,
                                                     train_ratio=train_ratio)
                st.success("✅ Preprocessing done")

                histories      = {}
                trained_models = {}

                # ── LSTM ──────────────────────────────────────────────────
                if use_lstm:
                    from train_lstm import train_lstm
                    with st.spinner("Training LSTM…"):
                        model_lstm, hist_lstm = train_lstm(
                            data["X_train"], data["y_train"],
                            epochs=epochs, batch_size=batch_size,
                        )
                    save_keras_model(model_lstm, lstm_path)
                    histories["LSTM"]      = hist_lstm
                    trained_models["LSTM"] = model_lstm
                    st.success(f"✅ LSTM trained — best val_loss: {min(hist_lstm.history['val_loss']):.6f}")

                # ── GRU ───────────────────────────────────────────────────
                if use_gru:
                    from train_gru import train_gru
                    with st.spinner("Training GRU…"):
                        model_gru, hist_gru = train_gru(
                            data["X_train"], data["y_train"],
                            epochs=epochs, batch_size=batch_size,
                        )
                    save_keras_model(model_gru, gru_path)
                    histories["GRU"]      = hist_gru
                    trained_models["GRU"] = model_gru
                    st.success(f"✅ GRU trained — best val_loss: {min(hist_gru.history['val_loss']):.6f}")

                # ── Linear ────────────────────────────────────────────────
                if use_linear:
                    from train_linear import train_linear
                    with st.spinner("Training Linear Regression…"):
                        model_lin = train_linear(data["X_train_lin"], data["y_train_lin"])
                    save_sklearn_model(model_lin, linear_path)
                    trained_models["Linear"] = model_lin
                    st.success("✅ Linear Regression trained")

                # ── loss chart ────────────────────────────────────────────
                if histories:
                    st.markdown('<div class="section-header">Training Loss</div>',
                                unsafe_allow_html=True)
                    st.plotly_chart(build_loss_chart(histories), use_container_width=True)

                # ── store for results tab via session_state ───────────────
                preds_dict = {}
                results    = {}

                if "LSTM" in trained_models:
                    p = predict_keras(trained_models["LSTM"], data["X_test"])
                    preds_dict["LSTM"] = inverse_scale(p, data["scaler"])
                if "GRU" in trained_models:
                    p = predict_keras(trained_models["GRU"], data["X_test"])
                    preds_dict["GRU"] = inverse_scale(p, data["scaler"])
                if "Linear" in trained_models:
                    p = trained_models["Linear"].predict(data["X_test_lin"])
                    preds_dict["Linear"] = inverse_scale(p, data["scaler"])

                y_true = inverse_scale(data["y_test"], data["scaler"])
                for name, preds in preds_dict.items():
                    results[name] = calculate_metrics(y_true, preds)

                comparison_df = compare_models(results)
                os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
                comparison_df.to_csv(os.path.join(ROOT, "reports", "model_comparison.csv"), index=False)

                st.session_state["trained_results"]  = results
                st.session_state["trained_preds"]    = preds_dict
                st.session_state["y_true"]           = y_true
                st.session_state["comparison_df"]    = comparison_df
                st.session_state["ticker_trained"]   = TICKER

                best = comparison_df.iloc[0]["Model"]
                st.balloons()
                st.success(f"🏆 Training complete! Best model: **{best}**")
                st.info("Switch to the **Results** tab to see predictions.")

    # ─────────────────────────────────────────────────────────────────────────
    #  TAB 3 — RESULTS
    # ─────────────────────────────────────────────────────────────────────────
    with tab_results:
        st.markdown('<div class="section-header">Evaluation & Predictions</div>',
                    unsafe_allow_html=True)

        # Load from session (just trained) OR from saved models on disk
        results       = st.session_state.get("trained_results")
        preds_dict    = st.session_state.get("trained_preds")
        y_true        = st.session_state.get("y_true")
        comparison_df = st.session_state.get("comparison_df")

        # Try loading saved models if nothing in session
        if results is None:
            from predict  import predict_keras, predict_linear
            from evaluate import inverse_scale, calculate_metrics, compare_models
            from preprocessing import prepare_data_for_training

            with st.spinner("Loading saved models and running predictions…"):
                try:
                    data = prepare_data_for_training(df, seq_length=seq_length,
                                                     train_ratio=train_ratio)
                    y_true     = inverse_scale(data["y_test"], data["scaler"])
                    preds_dict = {}
                    results    = {}

                    if os.path.exists(lstm_path):
                        m = load_existing_keras(lstm_path)
                        p = predict_keras(m, data["X_test"])
                        preds_dict["LSTM"]  = inverse_scale(p, data["scaler"])

                    if os.path.exists(gru_path):
                        m = load_existing_keras(gru_path)
                        p = predict_keras(m, data["X_test"])
                        preds_dict["GRU"] = inverse_scale(p, data["scaler"])

                    if os.path.exists(linear_path):
                        m = load_existing_sklearn(linear_path)
                        p = m.predict(data["X_test_lin"])
                        preds_dict["Linear"] = inverse_scale(p, data["scaler"])

                    if preds_dict:
                        for name, preds in preds_dict.items():
                            results[name] = calculate_metrics(y_true, preds)
                        comparison_df = compare_models(results)
                    else:
                        st.info("No saved models found. Train models first in the **Train Models** tab.")

                except Exception as e:
                    st.error(f"Error loading models: {e}")

        if results and preds_dict is not None and y_true is not None:
            # Metrics cards
            st.markdown("**Performance Metrics**")
            mcols = st.columns(len(results))
            for col, (name, m) in zip(mcols, results.items()):
                col.metric(f"{name} — RMSE", f"{m['rmse']:.4f}")
                col.metric(f"{name} — MAE",  f"{m['mae']:.4f}")
                col.metric(f"{name} — MAPE", f"{m['mape']:.2f}%")

            st.markdown("<br>", unsafe_allow_html=True)

            # Comparison table
            if comparison_df is not None:
                st.markdown('<div class="section-header">Model Ranking</div>',
                            unsafe_allow_html=True)
                st.dataframe(
                    comparison_df.style.highlight_min(subset=["RMSE", "MAE", "MAPE (%)"],
                                                      color="#1f3a1f"),
                    use_container_width=True,
                )

                st.plotly_chart(build_rmse_bar(comparison_df), use_container_width=True)

            # Predictions chart
            used_ticker = st.session_state.get("ticker_trained", TICKER)
            st.markdown('<div class="section-header">Prediction vs Actual</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(
                build_predictions_chart(y_true, preds_dict, used_ticker, results),
                use_container_width=True,
            )

    # ─────────────────────────────────────────────────────────────────────────
    #  TAB 4 — MULTI-STOCK
    # ─────────────────────────────────────────────────────────────────────────
    with tab_multi:
        st.markdown('<div class="section-header">Multi-Stock LSTM Comparison</div>',
                    unsafe_allow_html=True)

        st.markdown(
            '<div class="info-box">'
            'Trains a lightweight LSTM (20 epochs) on each ticker to compare performance. '
            'This may take a few minutes.'
            '</div>',
            unsafe_allow_html=True,
        )

        multi_btn = st.button("🌐 Run Multi-Stock Comparison", type="primary")

        if multi_btn:
            from data_loader    import load_stock_data
            from preprocessing  import prepare_data_for_training
            from train_lstm     import train_lstm
            from predict        import predict_keras
            from evaluate       import inverse_scale, calculate_metrics

            multi_results = {}
            progress_bar  = st.progress(0)
            status_text   = st.empty()
            tickers_list  = list(AVAILABLE_TICKERS.values())

            for i, t in enumerate(tickers_list):
                status_text.markdown(f"Processing **{t}** ({i+1}/{len(tickers_list)})…")
                try:
                    _df   = load_stock_data(t, START, END)
                    _data = prepare_data_for_training(_df, seq_length=seq_length,
                                                      train_ratio=train_ratio)
                    _model, _ = train_lstm(_data["X_train"], _data["y_train"],
                                           epochs=20, batch_size=32)
                    _preds_sc = predict_keras(_model, _data["X_test"])
                    _y_true   = inverse_scale(_data["y_test"],  _data["scaler"])
                    _y_pred   = inverse_scale(_preds_sc,         _data["scaler"])
                    multi_results[t] = calculate_metrics(_y_true, _y_pred)
                except Exception as e:
                    st.warning(f"⚠️ {t}: {e}")
                progress_bar.progress((i + 1) / len(tickers_list))

            status_text.empty()
            progress_bar.empty()

            if multi_results:
                st.success("✅ Multi-stock run complete!")
                st.plotly_chart(build_multi_bar(multi_results), use_container_width=True)

                rows = [{"Ticker": t, **m} for t, m in multi_results.items()]
                mdf  = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
                mdf.columns = ["Ticker", "RMSE", "MAE", "MAPE (%)"]
                st.dataframe(
                    mdf.style.highlight_min(subset=["RMSE", "MAE", "MAPE (%)"], color="#1f3a1f"),
                    use_container_width=True,
                )

                # Save to CSV
                os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
                mdf.to_csv(os.path.join(ROOT, "reports", "multi_stock_comparison.csv"), index=False)
                st.caption("Results saved to `reports/multi_stock_comparison.csv`")

        else:
            # Show price overlay if we're not running
            st.markdown('<div class="section-header">Historical Prices — All Stocks</div>',
                        unsafe_allow_html=True)
            with st.spinner("Loading price data for all tickers…"):
                try:
                    from data_loader import load_stock_data
                    fig_multi = go.Figure()
                    colors_m  = ["#58a6ff", "#2ecc71", "#f39c12"]
                    for idx_t, (label, t) in enumerate(AVAILABLE_TICKERS.items()):
                        try:
                            _df = cached_load_data(t, START, END)
                            # Normalize to 100 for comparison
                            norm = _df["Close"] / _df["Close"].iloc[0] * 100
                            fig_multi.add_trace(go.Scatter(
                                x=_df.index, y=norm,
                                name=t,
                                line=dict(color=colors_m[idx_t], width=1.8),
                            ))
                        except Exception:
                            pass
                    fig_multi.update_layout(
                        title=dict(text="Normalised Price Comparison (base = 100)",
                                   font=dict(size=15)),
                        xaxis_title="Date",
                        yaxis_title="Indexed Price",
                        **PLOTLY_LAYOUT,
                    )
                    st.plotly_chart(fig_multi, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not load multi-ticker prices: {e}")

else:
    st.warning(
        "No data loaded. Check the ticker / date range in the sidebar, "
        "or verify your internet connection."
    )
</file>

<file path="main.py">
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
END_DATE    = "2025-01-01"
SEQ_LENGTH  = 60          # look-back window (days)
EPOCHS      = 150
BATCH_SIZE  = 32


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
</file>

<file path="notebooks/Stock_Prediction_Models.ipynb">
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "491b03cd",
   "metadata": {},
   "source": [
    "# 📈 Stock Price Prediction using Deep Learning\n",
    "\n",
    "## Objective\n",
    "Dự án này xây dựng và so sánh 3 mô hình dự đoán giá cổ phiếu:\n",
    "- **LSTM** (Long Short-Term Memory)\n",
    "- **GRU** (Gated Recurrent Unit)\n",
    "- **Linear Regression** (baseline)\n",
    "\n",
    "## Dataset\n",
    "Dữ liệu từ Yahoo Finance API, tập trung vào các mã: `FPT.VN`, `VNM.VN`, `HPG.VN`\n",
    "\n",
    "## Tech Stack\n",
    "`Python` · `TensorFlow/Keras` · `yfinance` · `scikit-learn` · `matplotlib`\n",
    "\n",
    "---\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d41dd37a",
   "metadata": {},
   "source": [
    "## Section 1 — Import Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7f482b77",
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "import os\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "\n",
    "# Thêm src/ vào path để import các module\n",
    "sys.path.insert(0, \"../src\")\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.dates as mdates\n",
    "import seaborn as sns\n",
    "import yfinance as yf\n",
    "\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.metrics import mean_squared_error, mean_absolute_error\n",
    "\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout\n",
    "from tensorflow.keras.callbacks import EarlyStopping\n",
    "\n",
    "print(f\"TensorFlow version: {tf.__version__}\")\n",
    "print(f\"NumPy version: {np.__version__}\")\n",
    "print(f\"Pandas version: {pd.__version__}\")\n",
    "\n",
    "# Style cho plot đẹp hơn\n",
    "plt.style.use(\"seaborn-v0_8-darkgrid\")\n",
    "plt.rcParams[\"figure.dpi\"] = 100\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8971415c",
   "metadata": {},
   "source": [
    "## Section 2 — Configuration"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "84904a49",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ─── Chỉnh tại đây để thay đổi stock / thời gian ───\n",
    "TICKER     = \"FPT.VN\"\n",
    "START_DATE = \"2020-01-01\"\n",
    "END_DATE   = \"2025-01-01\"\n",
    "SEQ_LENGTH = 60     # số ngày look-back\n",
    "TRAIN_RATIO = 0.8   # 80% train, 20% test\n",
    "EPOCHS      = 150\n",
    "BATCH_SIZE  = 32\n",
    "\n",
    "print(f\"Config: {TICKER} | {START_DATE} → {END_DATE}\")\n",
    "print(f\"Sequence length: {SEQ_LENGTH} days | Train ratio: {TRAIN_RATIO}\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2de93db8",
   "metadata": {},
   "source": [
    "## Section 3 — Load Data\n",
    "\n",
    "Tải dữ liệu từ Yahoo Finance. Sử dụng hàm từ `src/data_loader.py`.\n",
    "\n",
    "> **Note:** Cổ phiếu Việt Nam dùng suffix `.VN` (e.g. `FPT.VN`).\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "79196102",
   "metadata": {},
   "outputs": [],
   "source": [
    "from data_loader import load_stock_data, save_raw_data\n",
    "\n",
    "# Tải và lưu dữ liệu\n",
    "df = load_stock_data(TICKER, START_DATE, END_DATE)\n",
    "save_raw_data(df, TICKER, save_dir=\"../data/raw\")\n",
    "\n",
    "# Preview\n",
    "print(f\"\\nShape: {df.shape}\")\n",
    "print(f\"Date range: {df.index[0].date()} → {df.index[-1].date()}\")\n",
    "print(f\"Columns: {list(df.columns)}\")\n",
    "df.head(10)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "616f3e34",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Thống kê cơ bản\n",
    "df[\"Close\"].describe()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4393c654",
   "metadata": {},
   "source": [
    "## Section 4 — Exploratory Data Analysis (EDA)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "41dd02ba",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(2, 2, figsize=(16, 10))\n",
    "fig.suptitle(f\"{TICKER} — Stock Analysis\", fontsize=16, fontweight=\"bold\")\n",
    "\n",
    "# Plot 1: Close price\n",
    "ax1 = axes[0, 0]\n",
    "ax1.plot(df.index, df[\"Close\"], color=\"#1f77b4\", linewidth=1.5)\n",
    "ax1.set_title(\"Close Price\")\n",
    "ax1.set_ylabel(\"Price (VND)\")\n",
    "ax1.xaxis.set_major_formatter(mdates.DateFormatter(\"%Y-%m\"))\n",
    "ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))\n",
    "plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)\n",
    "\n",
    "# Plot 2: Volume\n",
    "ax2 = axes[0, 1]\n",
    "ax2.bar(df.index, df[\"Volume\"], color=\"#95a5a6\", alpha=0.7, width=2)\n",
    "ax2.set_title(\"Trading Volume\")\n",
    "ax2.set_ylabel(\"Volume\")\n",
    "ax2.xaxis.set_major_formatter(mdates.DateFormatter(\"%Y-%m\"))\n",
    "plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)\n",
    "\n",
    "# Plot 3: 20-day & 50-day Moving Average\n",
    "ax3 = axes[1, 0]\n",
    "df[\"MA20\"] = df[\"Close\"].rolling(20).mean()\n",
    "df[\"MA50\"] = df[\"Close\"].rolling(50).mean()\n",
    "ax3.plot(df.index, df[\"Close\"], label=\"Close\",  color=\"#1f77b4\", alpha=0.6)\n",
    "ax3.plot(df.index, df[\"MA20\"],  label=\"MA20\",   color=\"#e74c3c\", linewidth=2)\n",
    "ax3.plot(df.index, df[\"MA50\"],  label=\"MA50\",   color=\"#2ecc71\", linewidth=2)\n",
    "ax3.set_title(\"Moving Averages\")\n",
    "ax3.legend()\n",
    "ax3.xaxis.set_major_formatter(mdates.DateFormatter(\"%Y-%m\"))\n",
    "plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)\n",
    "\n",
    "# Plot 4: Daily return distribution\n",
    "ax4 = axes[1, 1]\n",
    "daily_return = df[\"Close\"].pct_change().dropna()\n",
    "ax4.hist(daily_return, bins=50, color=\"#9b59b6\", edgecolor=\"white\", alpha=0.8)\n",
    "ax4.set_title(\"Daily Return Distribution\")\n",
    "ax4.set_xlabel(\"Daily Return (%)\")\n",
    "ax4.set_ylabel(\"Frequency\")\n",
    "ax4.axvline(0, color=\"red\", linestyle=\"--\")\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig(f\"../reports/{TICKER}_eda.png\", dpi=150, bbox_inches=\"tight\")\n",
    "plt.show()\n",
    "print(\"EDA plot saved to reports/\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bbb92abc",
   "metadata": {},
   "source": [
    "## Section 5 — Preprocessing\n",
    "\n",
    "Steps:\n",
    "1. Lấy cột **Close price**\n",
    "2. **Normalize** về [0, 1] dùng MinMaxScaler\n",
    "3. **Train/test split** theo thời gian (không shuffle)\n",
    "4. Tạo **sequences** (X, y) với look-back window = 60 ngày\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e3f8a406",
   "metadata": {},
   "outputs": [],
   "source": [
    "from preprocessing import prepare_data_for_training\n",
    "\n",
    "data = prepare_data_for_training(df, seq_length=SEQ_LENGTH, train_ratio=TRAIN_RATIO)\n",
    "\n",
    "X_train      = data[\"X_train\"]\n",
    "y_train      = data[\"y_train\"]\n",
    "X_test       = data[\"X_test\"]\n",
    "y_test       = data[\"y_test\"]\n",
    "X_train_lin  = data[\"X_train_lin\"]\n",
    "y_train_lin  = data[\"y_train_lin\"]\n",
    "X_test_lin   = data[\"X_test_lin\"]\n",
    "y_test_lin   = data[\"y_test_lin\"]\n",
    "scaler       = data[\"scaler\"]\n",
    "\n",
    "print(f\"LSTM/GRU — X_train: {X_train.shape} | y_train: {y_train.shape}\")\n",
    "print(f\"LSTM/GRU — X_test:  {X_test.shape}  | y_test:  {y_test.shape}\")\n",
    "print(f\"Linear   — X_train: {X_train_lin.shape}\")\n",
    "print(f\"Linear   — X_test:  {X_test_lin.shape}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2d37ac9a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize normalized data + train/test split\n",
    "close_prices = df[\"Close\"].values\n",
    "split_idx = int(len(close_prices) * TRAIN_RATIO)\n",
    "\n",
    "plt.figure(figsize=(13, 4))\n",
    "plt.plot(df.index[:split_idx], close_prices[:split_idx], \n",
    "         label=\"Train\", color=\"#1f77b4\", linewidth=1.5)\n",
    "plt.plot(df.index[split_idx:], close_prices[split_idx:], \n",
    "         label=\"Test\", color=\"#e74c3c\", linewidth=1.5)\n",
    "plt.axvline(df.index[split_idx], color=\"gray\", linestyle=\"--\", alpha=0.8, label=\"Split\")\n",
    "plt.title(f\"{TICKER} — Train / Test Split (80/20)\")\n",
    "plt.ylabel(\"Price (VND)\")\n",
    "plt.legend()\n",
    "plt.tight_layout()\n",
    "plt.savefig(f\"../reports/{TICKER}_train_test_split.png\", dpi=150, bbox_inches=\"tight\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d7238f09",
   "metadata": {},
   "source": [
    "## Section 6 — Train LSTM\n",
    "\n",
    "**LSTM (Long Short-Term Memory)** phù hợp với chuỗi thời gian nhờ cơ chế gates kiểm soát thông tin.\n",
    "\n",
    "Architecture:\n",
    "- LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16) → Dense(1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8071c1a5",
   "metadata": {},
   "outputs": [],
   "source": [
    "from train_lstm import train_lstm\n",
    "from utils import save_keras_model\n",
    "\n",
    "lstm_model, lstm_history = train_lstm(\n",
    "    X_train, y_train,\n",
    "    epochs=EPOCHS,\n",
    "    batch_size=BATCH_SIZE\n",
    ")\n",
    "\n",
    "save_keras_model(lstm_model, \"../models/lstm/lstm_model.keras\")\n",
    "print(\"LSTM model saved.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d0639b71",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Loss curve\n",
    "plt.figure(figsize=(10, 4))\n",
    "plt.plot(lstm_history.history[\"loss\"],     label=\"Train Loss\", color=\"#e74c3c\")\n",
    "plt.plot(lstm_history.history[\"val_loss\"], label=\"Val Loss\",   color=\"#3498db\", linestyle=\"--\")\n",
    "plt.title(\"LSTM — Training Loss\")\n",
    "plt.xlabel(\"Epoch\")\n",
    "plt.ylabel(\"MSE Loss\")\n",
    "plt.legend()\n",
    "plt.tight_layout()\n",
    "plt.savefig(\"../reports/lstm_loss.png\", dpi=150, bbox_inches=\"tight\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "91df0111",
   "metadata": {},
   "source": [
    "## Section 7 — Train GRU\n",
    "\n",
    "**GRU (Gated Recurrent Unit)** tương tự LSTM nhưng đơn giản hơn (ít tham số hơn), thường train nhanh hơn.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "99bafc8c",
   "metadata": {},
   "outputs": [],
   "source": [
    "from train_gru import train_gru\n",
    "\n",
    "gru_model, gru_history = train_gru(\n",
    "    X_train, y_train,\n",
    "    epochs=EPOCHS,\n",
    "    batch_size=BATCH_SIZE\n",
    ")\n",
    "\n",
    "save_keras_model(gru_model, \"../models/gru/gru_model.keras\")\n",
    "print(\"GRU model saved.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "de9b5a31",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(10, 4))\n",
    "plt.plot(gru_history.history[\"loss\"],     label=\"Train Loss\", color=\"#e74c3c\")\n",
    "plt.plot(gru_history.history[\"val_loss\"], label=\"Val Loss\",   color=\"#2ecc71\", linestyle=\"--\")\n",
    "plt.title(\"GRU — Training Loss\")\n",
    "plt.xlabel(\"Epoch\")\n",
    "plt.ylabel(\"MSE Loss\")\n",
    "plt.legend()\n",
    "plt.tight_layout()\n",
    "plt.savefig(\"../reports/gru_loss.png\", dpi=150, bbox_inches=\"tight\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e5638ac2",
   "metadata": {},
   "source": [
    "## Section 8 — Train Linear Regression (Baseline)\n",
    "\n",
    "Model đơn giản nhất để làm **baseline** so sánh — nếu LSTM/GRU không beat được Linear thì cần xem lại.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "56aa0473",
   "metadata": {},
   "outputs": [],
   "source": [
    "from train_linear import train_linear\n",
    "from utils import save_sklearn_model\n",
    "\n",
    "lin_model = train_linear(X_train_lin, y_train_lin)\n",
    "save_sklearn_model(lin_model, \"../models/linear/linear.pkl\")\n",
    "print(\"Linear model saved.\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5fc2e37a",
   "metadata": {},
   "source": [
    "## Section 9 — Generate Predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "23dda716",
   "metadata": {},
   "outputs": [],
   "source": [
    "from predict import predict_keras, predict_linear\n",
    "from evaluate import inverse_scale\n",
    "\n",
    "# Raw predictions (normalized scale)\n",
    "lstm_preds_scaled  = predict_keras(lstm_model, X_test)\n",
    "gru_preds_scaled   = predict_keras(gru_model,  X_test)\n",
    "lin_preds_scaled   = predict_linear(lin_model, X_test_lin)\n",
    "\n",
    "# Inverse transform → original price\n",
    "y_true     = inverse_scale(y_test,            scaler)\n",
    "lstm_preds = inverse_scale(lstm_preds_scaled, scaler)\n",
    "gru_preds  = inverse_scale(gru_preds_scaled,  scaler)\n",
    "lin_preds  = inverse_scale(lin_preds_scaled,  scaler)\n",
    "\n",
    "print(f\"Predictions generated. Sample size: {len(y_true)}\")\n",
    "print(f\"Price range — Actual:  {y_true.min():.0f} → {y_true.max():.0f}\")\n",
    "print(f\"Price range — LSTM:    {lstm_preds.min():.0f} → {lstm_preds.max():.0f}\")\n",
    "print(f\"Price range — GRU:     {gru_preds.min():.0f} → {gru_preds.max():.0f}\")\n",
    "print(f\"Price range — Linear:  {lin_preds.min():.0f} → {lin_preds.max():.0f}\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9a2a2a8f",
   "metadata": {},
   "source": [
    "## Section 10 — Evaluation & Model Comparison\n",
    "\n",
    "Metrics:\n",
    "- **RMSE** (Root Mean Squared Error) — penalizes large errors, lower is better\n",
    "- **MAE** (Mean Absolute Error) — average absolute error\n",
    "- **MAPE** (Mean Absolute Percentage Error) — error in %\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3d9688a4",
   "metadata": {},
   "outputs": [],
   "source": [
    "from evaluate import calculate_metrics, compare_models, print_comparison\n",
    "\n",
    "results = {\n",
    "    \"LSTM\":   calculate_metrics(y_true, lstm_preds),\n",
    "    \"GRU\":    calculate_metrics(y_true, gru_preds),\n",
    "    \"Linear\": calculate_metrics(y_true, lin_preds),\n",
    "}\n",
    "\n",
    "comparison_df = compare_models(results)\n",
    "print_comparison(comparison_df)\n",
    "\n",
    "# Save\n",
    "comparison_df.to_csv(\"../reports/model_comparison.csv\", index=False)\n",
    "comparison_df\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ce45abf8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Prediction Chart ──────────────────────────────────────────────────────\n",
    "fig, ax = plt.subplots(figsize=(14, 6))\n",
    "\n",
    "ax.plot(y_true, label=\"Actual\", color=\"black\", linewidth=2, zorder=5)\n",
    "ax.plot(lstm_preds, label=f\"LSTM  (RMSE={results['LSTM']['rmse']:.2f})\",\n",
    "        color=\"#e74c3c\", linestyle=\"-\", linewidth=1.5, alpha=0.9)\n",
    "ax.plot(gru_preds,  label=f\"GRU   (RMSE={results['GRU']['rmse']:.2f})\",\n",
    "        color=\"#2ecc71\", linestyle=\"--\", linewidth=1.5, alpha=0.9)\n",
    "ax.plot(lin_preds,  label=f\"Linear (RMSE={results['Linear']['rmse']:.2f})\",\n",
    "        color=\"#3498db\", linestyle=\"-.\", linewidth=1.5, alpha=0.9)\n",
    "\n",
    "ax.set_title(f\"{TICKER} — Model Predictions vs Actual Prices\", fontsize=14, fontweight=\"bold\")\n",
    "ax.set_xlabel(\"Time Steps (Test Period)\")\n",
    "ax.set_ylabel(\"Price (VND)\")\n",
    "ax.legend(fontsize=11)\n",
    "plt.tight_layout()\n",
    "plt.savefig(f\"../reports/{TICKER}_predictions.png\", dpi=150, bbox_inches=\"tight\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4f86cf68",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── RMSE Bar Chart ────────────────────────────────────────────────────────\n",
    "from utils import plot_comparison_bar\n",
    "plot_comparison_bar(comparison_df, save_path=\"../reports/model_comparison_bar.png\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9ac74e0b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Loss curves side-by-side ─────────────────────────────────────────────\n",
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "ax1.plot(lstm_history.history[\"loss\"],     label=\"Train\", color=\"#e74c3c\")\n",
    "ax1.plot(lstm_history.history[\"val_loss\"], label=\"Val\",   color=\"#e74c3c\", linestyle=\"--\")\n",
    "ax1.set_title(\"LSTM — Loss\")\n",
    "ax1.set_xlabel(\"Epoch\"); ax1.set_ylabel(\"MSE Loss\"); ax1.legend()\n",
    "\n",
    "ax2.plot(gru_history.history[\"loss\"],     label=\"Train\", color=\"#2ecc71\")\n",
    "ax2.plot(gru_history.history[\"val_loss\"], label=\"Val\",   color=\"#2ecc71\", linestyle=\"--\")\n",
    "ax2.set_title(\"GRU — Loss\")\n",
    "ax2.set_xlabel(\"Epoch\"); ax2.set_ylabel(\"MSE Loss\"); ax2.legend()\n",
    "\n",
    "plt.suptitle(\"Training Loss Curves\", fontsize=14, fontweight=\"bold\")\n",
    "plt.tight_layout()\n",
    "plt.savefig(\"../reports/training_loss_comparison.png\", dpi=150, bbox_inches=\"tight\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "24c18eab",
   "metadata": {},
   "source": [
    "## Section 11 — Multi-Stock Quick Compare\n",
    "\n",
    "Chạy nhanh pipeline cho 3 mã cổ phiếu, so sánh RMSE của LSTM.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fec31fca",
   "metadata": {},
   "outputs": [],
   "source": [
    "TICKERS = [\"FPT.VN\", \"VNM.VN\", \"HPG.VN\"]\n",
    "multi_results = {}\n",
    "\n",
    "for t in TICKERS:\n",
    "    try:\n",
    "        print(f\"\\n{'─'*40}\")\n",
    "        print(f\"Processing: {t}\")\n",
    "        _df = load_stock_data(t, START_DATE, END_DATE)\n",
    "        _data = prepare_data_for_training(_df, seq_length=SEQ_LENGTH)\n",
    "        \n",
    "        _lstm, _ = train_lstm(_data[\"X_train\"], _data[\"y_train\"], epochs=20, batch_size=32)\n",
    "        _preds_sc = predict_keras(_lstm, _data[\"X_test\"])\n",
    "        _y_true   = inverse_scale(_data[\"y_test\"],   _data[\"scaler\"])\n",
    "        _y_pred   = inverse_scale(_preds_sc,          _data[\"scaler\"])\n",
    "        \n",
    "        m = calculate_metrics(_y_true, _y_pred)\n",
    "        multi_results[t] = m\n",
    "        print(f\"  → RMSE: {m['rmse']:.2f} | MAE: {m['mae']:.2f} | MAPE: {m['mape']:.2f}%\")\n",
    "    except Exception as e:\n",
    "        print(f\"  [ERROR] {t}: {e}\")\n",
    "\n",
    "print(\"\\n=== Multi-Stock LSTM RMSE Summary ===\")\n",
    "for t, m in multi_results.items():\n",
    "    print(f\"  {t:12s}  RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  MAPE={m['mape']:.2f}%\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9b8a67eb",
   "metadata": {},
   "source": [
    "## Section 12 — Save / Load Model Demo"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "75fa0fa9",
   "metadata": {},
   "outputs": [],
   "source": [
    "from utils import load_keras_model, load_sklearn_model\n",
    "\n",
    "# Load models từ disk\n",
    "loaded_lstm   = load_keras_model(\"../models/lstm/lstm_model.keras\")\n",
    "loaded_gru    = load_keras_model(\"../models/gru/gru_model.keras\")\n",
    "loaded_linear = load_sklearn_model(\"../models/linear/linear.pkl\")\n",
    "\n",
    "# Verify bằng cách predict lại\n",
    "verify_preds = predict_keras(loaded_lstm, X_test[:5])\n",
    "original_preds = predict_keras(lstm_model, X_test[:5])\n",
    "\n",
    "print(\"\\nVerification (first 5 predictions):\")\n",
    "print(f\"  Original:  {original_preds}\")\n",
    "print(f\"  Reloaded:  {verify_preds}\")\n",
    "match = np.allclose(verify_preds, original_preds, atol=1e-5)\n",
    "print(f\"  Match: {'✅ YES' if match else '❌ NO'}\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "151fc0b2",
   "metadata": {},
   "source": [
    "## Section 13 — Conclusion\n",
    "\n",
    "### Tóm tắt kết quả\n",
    "\n",
    "| Model | Đặc điểm | Khi nào dùng |\n",
    "|-------|----------|--------------|\n",
    "| **LSTM** | Tốt nhất với long-term dependencies | Khi cần độ chính xác cao |\n",
    "| **GRU** | Tương đương LSTM, train nhanh hơn | Khi cần trade-off speed/accuracy |\n",
    "| **Linear** | Baseline đơn giản | Để so sánh chuẩn |\n",
    "\n",
    "### Hạn chế\n",
    "- Mô hình chỉ dùng **Close price** — có thể bổ sung Volume, Open, High, Low\n",
    "- Chưa tích hợp **sentiment analysis** từ tin tức\n",
    "- Stock market có tính **non-stationary** cao → accuracy bị giới hạn\n",
    "\n",
    "### Future Work\n",
    "- Thêm **Transformer/Attention** model\n",
    "- Feature engineering: technical indicators (RSI, MACD, Bollinger Bands)\n",
    "- Deploy lên Streamlit dashboard\n",
    "\n",
    "---\n",
    "*Generated by AI Stock Prediction Project*\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
</file>

<file path="README.md">
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
└── main.py           ← Run full pipeline
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
- Input: 60-day look-back window (flattened)

---

## Dataset

| Mã | Công ty |
|----|---------|
| FPT.VN | FPT Corporation |
| VNM.VN | Vinamilk |
| HPG.VN | Hòa Phát Group |

Nguồn: Yahoo Finance (`yfinance`)  
Thời gian: 2020-01-01 → 2025-01-01

---

## Cách chạy

### 1. Cài dependencies

```bash
pip install -r requirements.txt
```

### 2. Chạy notebook (khuyến nghị)

```bash
cd notebooks
jupyter notebook Stock_Prediction_Models.ipynb
```

### 3. Hoặc chạy full pipeline

```bash
python main.py
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
| 1 | LSTM | ~2.3 | ~1.8 | ~2.1 |
| 2 | GRU | ~2.5 | ~2.0 | ~2.3 |
| 3 | Linear | ~4.8 | ~3.9 | ~4.5 |

> Kết quả thực tế sẽ khác tùy dữ liệu và lần train.
</file>

<file path="requirements.txt">
tensorflow>=2.12.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
scikit-learn>=1.2.0
yfinance>=0.2.18
joblib>=1.2.0
seaborn>=0.12.0
streamlit>=1.32.0
plotly>=5.18.0
</file>

<file path="src/data_loader.py">
"""
data_loader.py
Load stock data from Yahoo Finance and save to data/raw/
"""

import yfinance as yf
import pandas as pd
import os


def load_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download stock data from Yahoo Finance.

    Args:
        ticker: e.g. "FPT.VN", "VNM.VN"
        start:  "YYYY-MM-DD"
        end:    "YYYY-MM-DD"

    Returns:
        DataFrame with OHLCV columns
    """
    print(f"[DataLoader] Downloading {ticker} from {start} to {end} ...")
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    # Flatten MultiIndex columns if present (yfinance sometimes returns them)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print(f"[DataLoader] Downloaded {len(df)} rows. Columns: {list(df.columns)}")
    return df


def save_raw_data(df: pd.DataFrame, ticker: str, save_dir: str = "data/raw") -> str:
    """
    Save raw DataFrame to CSV.

    Returns:
        Path to saved file
    """
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{ticker.replace('.', '_')}_raw.csv"
    path = os.path.join(save_dir, filename)
    df.to_csv(path)
    print(f"[DataLoader] Saved raw data → {path}")
    return path


def load_raw_data(ticker: str, save_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load previously saved raw CSV.
    """
    filename = f"{ticker.replace('.', '_')}_raw.csv"
    path = os.path.join(save_dir, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data not found: {path}. Run load_stock_data() first.")

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"[DataLoader] Loaded {len(df)} rows from {path}")
    return df
</file>

<file path="src/debug_config.py">
"""
Shared debug switches for the stock forecasting pipeline.

Set these variables directly, or override them without editing code:
    $env:STOCK_DEBUG="1"
    $env:STOCK_OVERFIT_DEBUG="1"
"""

import os


DEBUG = os.getenv("STOCK_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
OVERFIT_DEBUG = os.getenv("STOCK_OVERFIT_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
DEBUG_DIR = os.path.join("reports", "debug")
</file>

<file path="src/evaluate.py">
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
</file>

<file path="src/predict.py">
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
</file>

<file path="src/preprocessing.py">
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
    seq_length: int = 60,
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


def prepare_data_for_training(df: pd.DataFrame, seq_length: int = 60, train_ratio: float = 0.8):
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
</file>

<file path="src/train_gru.py">
"""
train_gru.py
Build and train GRU model for stock price prediction.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.models import Sequential

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


def build_gru(input_shape: tuple) -> Sequential:
    """
    Build GRU model.
    """
    model = Sequential([
        GRU(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        GRU(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1),
    ])

    model.compile(optimizer="adam", loss="mse")
    model.summary()
    return model


def train_gru(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 150,
    batch_size: int = 32,
    validation_split: float = 0.1,
):
    """
    Train GRU model with early stopping.
    """
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_gru(input_shape)
    initial_weights = [w.copy() for w in model.get_weights()]

    if OVERFIT_DEBUG:
        X_train = X_train[:200]
        y_train = y_train[:200]
        validation_split = 0.0
        epochs = max(epochs, 300)
        print("[DEBUG][GRU][OVERFIT] Enabled: using first 200 samples, no validation, >=300 epochs.")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1,
    )

    os.makedirs("models/gru", exist_ok=True)
    checkpoint = ModelCheckpoint(
        filepath="models/gru/gru_checkpoint.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=0,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1,
    )

    debug_callback = DebugTrainingCallback("GRU")
    callbacks = [debug_callback]
    if not OVERFIT_DEBUG:
        callbacks = [early_stop, checkpoint, reduce_lr, debug_callback]

    print(f"\n[GRU] Training on {len(X_train)} samples ...")
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
        print(f"[DEBUG][GRU][OVERFIT] Train RMSE on scaled target: {rmse:.10f}")
        if rmse > 0.02:
            print("[DEBUG][WARNING][GRU][OVERFIT] RMSE did not approach zero. Model or pipeline may be faulty.")
        _save_overfit_loss_curve(history, "GRU")
    elif "val_loss" in history.history:
        best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
        print(f"[GRU] Training done. Best val_loss: {min(history.history['val_loss']):.6f}")
        if early_stop.stopped_epoch > 0:
            print(f"[DEBUG][GRU] EarlyStopping stopped at epoch {early_stop.stopped_epoch + 1}; best epoch {best_epoch}.")
    else:
        print(f"[GRU] Training done. Final loss: {history.history['loss'][-1]:.6f}")

    _save_history(history, debug_callback, "GRU")
    _print_weight_debug(model, initial_weights, "GRU")
    _print_gradient_norm(model, X_train, y_train, "GRU")
    return model, history
</file>

<file path="src/train_linear.py">
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
</file>

<file path="src/train_lstm.py">
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
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1),
    ])

    model.compile(optimizer="adam", loss="mse")
    model.summary()
    return model


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 150,
    batch_size: int = 32,
    validation_split: float = 0.1,
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
        patience=15,
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
        factor=0.5,
        patience=7,
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
</file>

<file path="src/utils.py">
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
</file>

</files>
