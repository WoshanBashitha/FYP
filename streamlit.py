import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Crypto Forecast Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .main { background-color: #0d0f14; }
    .block-container { padding-top: 1.5rem; }

    h1, h2, h3 { font-family: 'Space Mono', monospace; }

    .metric-card {
        background: linear-gradient(135deg, #1a1d26 0%, #12151e 100%);
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 10px;
    }
    .metric-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #6b7280;
        margin-bottom: 4px;
    }
    .metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 22px;
        font-weight: 700;
        color: #f0f4ff;
    }
    .metric-change-pos { color: #22c55e; font-size: 13px; }
    .metric-change-neg { color: #ef4444; font-size: 13px; }

    .section-title {
        font-family: 'Space Mono', monospace;
        font-size: 14px;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #7c86ff;
        border-left: 3px solid #7c86ff;
        padding-left: 10px;
        margin: 28px 0 14px 0;
    }

    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background-color: #1a1d26 !important;
        border: 1px solid #2a2d3e !important;
        color: #f0f4ff !important;
    }

    .warning-box {
        background: #1f1a10;
        border: 1px solid #854d0e;
        border-radius: 8px;
        padding: 12px 16px;
        color: #fbbf24;
        font-size: 13px;
    }
    .info-box {
        background: #0f1f2e;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 12px 16px;
        color: #60a5fa;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# COIN CONFIG
# ─────────────────────────────────────────────
COIN_CONFIG = {
    "Bitcoin (BTC)":  {"ticker": "BTC-USD",  "key": "BTC",  "color": "#f59e0b"},
    "Ethereum (ETH)": {"ticker": "ETH-USD",  "key": "ETH",  "color": "#6366f1"},
    "Solana (SOL)":   {"ticker": "SOL-USD",  "key": "SOL",  "color": "#22d3ee"},
    "BNB (BNB)":      {"ticker": "BNB-USD",  "key": "BNB",  "color": "#eab308"},
    "Ripple (XRP)":   {"ticker": "XRP-USD",  "key": "XRP",  "color": "#10b981"},
}

FORECAST_HORIZONS = {
    "7 days — Short-term":      7,
    "30 days — Monthly trend":  30,
    "90 days — Quarterly view": 90,
    "180 days — Longer outlook": 180,
}

# ─────────────────────────────────────────────
# BiLSTM MODEL DEFINITION  (exact copy from Bi_LSTM.py)
# ─────────────────────────────────────────────
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 because bidirectional

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]   # last time step
        out = self.fc(out)
        return out

# ─────────────────────────────────────────────
# FEATURE ENGINEERING  (your feature_engineering.py)
# ─────────────────────────────────────────────
from feature_engineering import calculate_features

# ─────────────────────────────────────────────
# LOADERS (cached)
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

@st.cache_resource
def load_model_and_scalers(coin_key: str):
    """
    Loads the BiLSTM model + x/y scalers for a given coin key (e.g. 'BTC').

    ⚠️  UPDATE MODEL_DIR and SCALER_DIR below if your files are in a subfolder.
    Expected filenames:
        {coin_key}_x_scaler.pkl
        {coin_key}_y_scaler.pkl
        {coin_key}_final_model.pth
    """
    MODEL_DIR  = "."   # ← change to e.g. "models/"  if needed
    SCALER_DIR = "."   # ← change to e.g. "scalers/" if needed

    x_scaler_path = f"{SCALER_DIR}/{coin_key}_x_scaler.pkl"
    y_scaler_path = f"{SCALER_DIR}/{coin_key}_y_scaler.pkl"
    model_path    = f"{MODEL_DIR}/{coin_key}_final_model.pth"

    with open(x_scaler_path, "rb") as f:
        x_scaler = pickle.load(f)
    with open(y_scaler_path, "rb") as f:
        y_scaler = pickle.load(f)

    # input_size inferred from the fitted x_scaler
    input_size = x_scaler.n_features_in_

    # Try loading as state_dict first (torch.save(model.state_dict(), path))
    # Falls back to full-model load (torch.save(model, path))
    raw = torch.load(model_path, map_location=torch.device("cpu"))

    if isinstance(raw, dict):
        # state_dict save
        model = BiLSTM(input_size=input_size)
        model.load_state_dict(raw)
    else:
        # full model save — use directly
        model = raw

    model.eval()
    return model, x_scaler, y_scaler

# ─────────────────────────────────────────────
# PREPROCESSING & FORECASTING
# ─────────────────────────────────────────────
TIMESTEP = 30  # must match training

def preprocess(df: pd.DataFrame, x_scaler):
    """Apply feature engineering → drop NaN → scale."""
    featured = calculate_features(df.copy())
    featured.dropna(inplace=True)
    # All columns except 'Close' are features — matches training setup
    feature_cols = [c for c in featured.columns if c != "Close"]
    X = featured[feature_cols].values
    X_scaled = x_scaler.transform(X)
    return X_scaled, featured

def make_sequences(X_scaled: np.ndarray, timestep: int = TIMESTEP):
    sequences = []
    for i in range(len(X_scaled) - timestep):
        sequences.append(X_scaled[i: i + timestep])
    return np.array(sequences)

def forecast_future(model, x_scaler, y_scaler, df: pd.DataFrame, horizon: int):
    """
    Iterative one-step-ahead forecast for `horizon` days.
    Returns a list of predicted closing prices (inverse-scaled).
    """
    featured = calculate_features(df.copy())
    featured.dropna(inplace=True)

    feature_cols = [c for c in featured.columns if c != "Close"]
    X_all = featured[feature_cols].values
    X_scaled = x_scaler.transform(X_all)

    # Seed window = last TIMESTEP rows
    window = X_scaled[-TIMESTEP:].copy()
    predictions = []

    with torch.no_grad():
        for _ in range(horizon):
            seq = torch.tensor(window[np.newaxis, :, :], dtype=torch.float32)
            pred_scaled = model(seq).item()

            # Inverse-transform the predicted close price
            pred_price = y_scaler.inverse_transform([[pred_scaled]])[0][0]
            predictions.append(pred_price)

            # Roll the window: drop oldest row, append new row
            # For new row we reuse the last known feature vector
            # with the close-price feature updated via y_scaler
            new_row = window[-1].copy()
            window = np.vstack([window[1:], new_row])

    return predictions

def get_actual_vs_predicted(model, x_scaler, y_scaler, df: pd.DataFrame):
    """Run model over historical data to compare actual vs predicted."""
    featured = calculate_features(df.copy())
    featured.dropna(inplace=True)

    feature_cols = [c for c in featured.columns if c != "Close"]
    X_all = featured[feature_cols].values
    X_scaled = x_scaler.transform(X_all)

    sequences = make_sequences(X_scaled, TIMESTEP)
    if len(sequences) == 0:
        return None, None

    X_tensor = torch.tensor(sequences, dtype=torch.float32)
    with torch.no_grad():
        preds_scaled = model(X_tensor).numpy().flatten()

    preds = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    actuals = featured["Close"].values[TIMESTEP:]
    dates   = featured.index[TIMESTEP:]
    return dates, actuals, preds

# ─────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d0f14",
    plot_bgcolor="#0d0f14",
    font=dict(color="#9ca3af", family="DM Sans"),
    xaxis=dict(gridcolor="#1e2130", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#1e2130", showgrid=True, zeroline=False),
    legend=dict(bgcolor="#12151e", bordercolor="#2a2d3e", borderwidth=1),
    margin=dict(l=10, r=10, t=40, b=10),
)

def apply_layout(fig, title=""):
    fig.update_layout(title=dict(text=title, font=dict(size=13, family="Space Mono")), **PLOTLY_LAYOUT)
    return fig

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⬡ CRYPTO FORECAST")
    st.markdown("---")

    st.markdown("### Coin Selection")
    selected_coins = st.multiselect(
        "Select coins",
        options=list(COIN_CONFIG.keys()),
        default=["Bitcoin (BTC)"],
    )

    st.markdown("### Date Range")
    min_date = datetime(2017, 1, 1)
    max_date = datetime.today()

    start_date = st.date_input("Start date", value=datetime(2022, 1, 1),
                               min_value=min_date, max_value=max_date)
    end_date   = st.date_input("End date",   value=max_date,
                               min_value=min_date, max_value=max_date)

    if start_date >= end_date:
        st.error("Start date must be before end date.")

    st.markdown("### Forecast Horizon")
    horizon_label = st.selectbox("Select horizon", list(FORECAST_HORIZONS.keys()))
    forecast_days = FORECAST_HORIZONS[horizon_label]

    st.markdown("### Historical View Window")
    hist_period = st.selectbox(
        "Chart period",
        ["1 Month", "3 Months", "6 Months", "1 Year", "Full Range"],
        index=3,
    )

    st.markdown("---")
    run_forecast = st.button("🚀  Run Forecast", use_container_width=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(
    "<h1 style='font-family:Space Mono;font-size:26px;color:#f0f4ff;"
    "letter-spacing:2px;margin-bottom:0'>CRYPTO INTELLIGENCE DASHBOARD</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<p style='color:#4b5563;font-size:12px;margin-top:4px'>"
    f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; "
    f"Data: Yahoo Finance &nbsp;|&nbsp; Model: BiLSTM</p>",
    unsafe_allow_html=True,
)

if not selected_coins:
    st.markdown('<div class="info-box">👈 Select at least one coin from the sidebar to get started.</div>',
                unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
coin_data = {}
unavailable = []

for coin_name in selected_coins:
    cfg = COIN_CONFIG[coin_name]
    df = load_price_data(cfg["ticker"], str(start_date), str(end_date))
    if df.empty:
        unavailable.append(coin_name)
    else:
        coin_data[coin_name] = df

if unavailable:
    st.markdown(
        f'<div class="warning-box">⚠️ Data not available for the selected period: '
        f'<b>{", ".join(unavailable)}</b>. '
        f'These coins may not have been listed during that date range.</div>',
        unsafe_allow_html=True,
    )

if not coin_data:
    st.stop()

# Trim to hist_period for chart display
def trim_to_period(df, period):
    today = df.index.max()
    if period == "1 Month":
        return df[df.index >= today - timedelta(days=30)]
    elif period == "3 Months":
        return df[df.index >= today - timedelta(days=90)]
    elif period == "6 Months":
        return df[df.index >= today - timedelta(days=180)]
    elif period == "1 Year":
        return df[df.index >= today - timedelta(days=365)]
    return df

# ─────────────────────────────────────────────
# SUMMARY METRIC CARDS
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">Summary Metrics</div>', unsafe_allow_html=True)

forecast_cache = {}  # store forecasts for reuse below

cols = st.columns(len(coin_data))
for i, (coin_name, df) in enumerate(coin_data.items()):
    cfg = COIN_CONFIG[coin_name]
    current_price = float(df["Close"].iloc[-1])
    prev_price    = float(df["Close"].iloc[-2]) if len(df) > 1 else current_price
    pct_change    = (current_price - prev_price) / prev_price * 100

    period_start_price = float(df["Close"].iloc[0])
    period_return = (current_price - period_start_price) / period_start_price * 100

    # Forecast
    forecast_end_price  = None
    forecast_pct_change = None
    if run_forecast:
        try:
            model, x_sc, y_sc = load_model_and_scalers(cfg["key"])
            preds = forecast_future(model, x_sc, y_sc, df, forecast_days)
            forecast_end_price  = preds[-1]
            forecast_pct_change = (forecast_end_price - current_price) / current_price * 100
            forecast_cache[coin_name] = preds
        except Exception as e:
            st.warning(f"Could not load model for {coin_name}: {e}")

    sign     = "▲" if pct_change >= 0 else "▼"
    sign_cls = "metric-change-pos" if pct_change >= 0 else "metric-change-neg"

    fcast_html = ""
    if forecast_end_price is not None:
        fs = "metric-change-pos" if forecast_pct_change >= 0 else "metric-change-neg"
        fcast_html = (f"<div class='metric-label' style='margin-top:8px'>Forecast {forecast_days}d</div>"
                      f"<div class='metric-value' style='font-size:16px'>${forecast_end_price:,.2f}</div>"
                      f"<span class='{fs}'>{forecast_pct_change:+.2f}%</span>")

    with cols[i]:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>{coin_name}</div>
            <div class='metric-value'>${current_price:,.2f}</div>
            <span class='{sign_cls}'>{sign} {abs(pct_change):.2f}% (24h)</span>
            <div class='metric-label' style='margin-top:8px'>Period Return</div>
            <div style='color:#f0f4ff;font-size:14px;font-family:Space Mono'>{period_return:+.2f}%</div>
            {fcast_html}
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 1. HISTORICAL PRICE CHART
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">Historical Price Comparison</div>', unsafe_allow_html=True)

norm_toggle = st.toggle("Normalize prices (compare performance %)", value=False)

fig_hist = go.Figure()
for coin_name, df in coin_data.items():
    cfg = COIN_CONFIG[coin_name]
    trimmed = trim_to_period(df, hist_period)
    if trimmed.empty:
        continue
    y = trimmed["Close"]
    if norm_toggle:
        y = (y / y.iloc[0] - 1) * 100
    fig_hist.add_trace(go.Scatter(
        x=trimmed.index, y=y,
        name=coin_name,
        line=dict(color=cfg["color"], width=2),
        mode="lines",
        hovertemplate="%{y:.2f}" + ("%" if norm_toggle else " USD") + "<extra>" + coin_name + "</extra>",
    ))

apply_layout(fig_hist, "Historical Closing Price")
fig_hist.update_yaxes(title_text="Return (%)" if norm_toggle else "Price (USD)")
st.plotly_chart(fig_hist, use_container_width=True)

# ─────────────────────────────────────────────
# 2. FORECAST CHART
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">Price Forecast</div>', unsafe_allow_html=True)

if not run_forecast:
    st.markdown('<div class="info-box">Click <b>🚀 Run Forecast</b> in the sidebar to generate predictions.</div>',
                unsafe_allow_html=True)
else:
    fig_fcast = go.Figure()
    last_date = max(df.index.max() for df in coin_data.values())

    for coin_name, df in coin_data.items():
        cfg = COIN_CONFIG[coin_name]
        if coin_name not in forecast_cache:
            continue

        # Historical tail (last 60 days for context)
        tail = df["Close"].tail(60)
        fig_fcast.add_trace(go.Scatter(
            x=tail.index, y=tail,
            name=f"{coin_name} (actual)",
            line=dict(color=cfg["color"], width=1.5, dash="dot"),
            opacity=0.5,
        ))

        preds = forecast_cache[coin_name]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
        fig_fcast.add_trace(go.Scatter(
            x=future_dates, y=preds,
            name=f"{coin_name} (forecast)",
            line=dict(color=cfg["color"], width=2.5),
        ))

    apply_layout(fig_fcast, f"Forecast — Next {forecast_days} Days")
    fig_fcast.update_yaxes(title_text="Price (USD)")
    st.plotly_chart(fig_fcast, use_container_width=True)

# ─────────────────────────────────────────────
# 3. ACTUAL vs PREDICTED
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">Actual vs Predicted (Historical Backtest)</div>',
            unsafe_allow_html=True)

avp_coin = st.selectbox("Select coin for backtest chart", list(coin_data.keys()), key="avp")

if run_forecast:
    cfg = COIN_CONFIG[avp_coin]
    try:
        model, x_sc, y_sc = load_model_and_scalers(cfg["key"])
        result = get_actual_vs_predicted(model, x_sc, y_sc, coin_data[avp_coin])
        if result[0] is not None:
            dates, actuals, preds = result
            fig_avp = go.Figure()
            fig_avp.add_trace(go.Scatter(x=dates, y=actuals, name="Actual",
                                         line=dict(color="#f0f4ff", width=1.5)))
            fig_avp.add_trace(go.Scatter(x=dates, y=preds, name="Predicted",
                                         line=dict(color=cfg["color"], width=1.5, dash="dash")))
            apply_layout(fig_avp, f"{avp_coin} — Actual vs Predicted")
            st.plotly_chart(fig_avp, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not generate backtest for {avp_coin}: {e}")
else:
    st.markdown('<div class="info-box">Run forecast to see the backtest chart.</div>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 4. CANDLESTICK + VOLUME
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">Candlestick & Volume</div>', unsafe_allow_html=True)

candle_coin = st.selectbox("Select coin", list(coin_data.keys()), key="candle")
candle_df   = trim_to_period(coin_data[candle_coin], hist_period)

if not candle_df.empty:
    fig_candle = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               row_heights=[0.7, 0.3], vertical_spacing=0.04)

    fig_candle.add_trace(go.Candlestick(
        x=candle_df.index,
        open=candle_df["Open"], high=candle_df["High"],
        low=candle_df["Low"],   close=candle_df["Close"],
        increasing_line_color="#22c55e",
        decreasing_line_color="#ef4444",
        name="OHLC",
    ), row=1, col=1)

    colors = ["#22c55e" if c >= o else "#ef4444"
              for c, o in zip(candle_df["Close"], candle_df["Open"])]
    fig_candle.add_trace(go.Bar(
        x=candle_df.index, y=candle_df["Volume"],
        marker_color=colors, name="Volume", opacity=0.7,
    ), row=2, col=1)

    fig_candle.update_layout(
        xaxis_rangeslider_visible=False,
        **PLOTLY_LAYOUT,
        title=dict(text=f"{candle_coin} — Candlestick & Volume",
                   font=dict(size=13, family="Space Mono")),
    )
    fig_candle.update_xaxes(gridcolor="#1e2130")
    fig_candle.update_yaxes(gridcolor="#1e2130")
    st.plotly_chart(fig_candle, use_container_width=True)

# ─────────────────────────────────────────────
# 5. RETURN COMPARISON
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">Period Return Comparison</div>', unsafe_allow_html=True)

returns = {}
for coin_name, df in coin_data.items():
    trimmed = trim_to_period(df, hist_period)
    if len(trimmed) >= 2:
        r = (float(trimmed["Close"].iloc[-1]) - float(trimmed["Close"].iloc[0])) / float(trimmed["Close"].iloc[0]) * 100
        returns[coin_name] = r

if returns:
    fig_ret = go.Figure(go.Bar(
        x=list(returns.keys()),
        y=list(returns.values()),
        marker_color=["#22c55e" if v >= 0 else "#ef4444" for v in returns.values()],
        text=[f"{v:+.2f}%" for v in returns.values()],
        textposition="outside",
    ))
    apply_layout(fig_ret, f"Return Comparison — {hist_period}")
    fig_ret.update_yaxes(title_text="Return (%)")
    st.plotly_chart(fig_ret, use_container_width=True)

# ─────────────────────────────────────────────
# 6. VOLATILITY COMPARISON
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">Volatility Comparison</div>', unsafe_allow_html=True)

vols = {}
for coin_name, df in coin_data.items():
    trimmed = trim_to_period(df, hist_period)
    if len(trimmed) >= 5:
        daily_returns = trimmed["Close"].pct_change().dropna()
        vols[coin_name] = float(daily_returns.std() * np.sqrt(365) * 100)

if vols:
    fig_vol = go.Figure(go.Bar(
        x=list(vols.keys()),
        y=list(vols.values()),
        marker_color=[COIN_CONFIG[c]["color"] for c in vols],
        text=[f"{v:.2f}%" for v in vols.values()],
        textposition="outside",
    ))
    apply_layout(fig_vol, "Annualised Volatility (%)")
    fig_vol.update_yaxes(title_text="Volatility (%)")
    st.plotly_chart(fig_vol, use_container_width=True)

# ─────────────────────────────────────────────
# 7. CORRELATION HEATMAP
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)

if len(coin_data) >= 2:
    close_df = pd.DataFrame({
        coin_name: trim_to_period(df, hist_period)["Close"]
        for coin_name, df in coin_data.items()
    }).dropna()

    if close_df.shape[0] > 10:
        corr = close_df.corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu",
            zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            colorbar=dict(tickfont=dict(color="#9ca3af")),
        ))
        apply_layout(fig_corr, "Price Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough overlapping data for correlation.")
else:
    st.markdown('<div class="info-box">Select at least 2 coins to see the correlation heatmap.</div>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='color:#374151;font-size:11px;text-align:center'>"
    "For research and educational purposes only. Not financial advice.</p>",
    unsafe_allow_html=True,
)