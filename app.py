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
        end_date = st.date_input("End", value=pd.Timestamp("2026-01-01"),
                                 label_visibility="visible")

    st.markdown("**Model Parameters**")
    seq_length   = st.slider("Sequence Length (days)", 20, 120, 20, step=10)
    train_ratio  = st.slider("Train Ratio", 0.6, 0.9, 0.8, step=0.05)
    epochs       = st.slider("Epochs", 10, 200, 150, step=10)
    batch_size   = st.selectbox("Batch Size", [16, 32, 64, 128], index=0)

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
