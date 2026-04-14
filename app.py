"""
Cryptocurrency Price Forecasting Dashboard
==========================================
A Streamlit dashboard for forecasting BTC, ETH, XRP, SOL, BNB prices
using BiLSTM (Bitcoin, Binance) and XGBoost (Ethereum, Ripple, Solana) models.

Author: University Project Prototype
Models: BiLSTM (.pth) and XGBoost (.pkl)
Data: yfinance (live historical data)
AI Reports: Google Gemini API
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import os
import warnings
import joblib
from datetime import datetime, timedelta

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CryptoForecast Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  – dark financial aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-primary: #0d1117;
    --bg-card: #161b22;
    --bg-card2: #1c2128;
    --accent: #f0b429;
    --accent2: #58a6ff;
    --green: #3fb950;
    --red: #f85149;
    --text: #e6edf3;
    --muted: #8b949e;
    --border: #30363d;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
    background-color: var(--bg-primary);
}

.stApp { background-color: var(--bg-primary); }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}

/* Headers */
h1, h2, h3 { font-family: 'Space Mono', monospace; }
h1 { color: var(--accent) !important; letter-spacing: -1px; }
h2 { color: var(--accent2) !important; }
h3 { color: var(--text) !important; }

/* Cards */
.crypto-card {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 22px;
    margin: 8px 0;
}
.metric-row {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin: 12px 0;
}
.metric-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 20px;
    flex: 1;
    min-width: 140px;
}
.metric-label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { font-family: 'Space Mono', monospace; font-size: 22px; font-weight: 700; color: var(--accent); }
.metric-value.green { color: var(--green); }
.metric-value.red { color: var(--red); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: var(--bg-card2);
    border-radius: 8px;
    gap: 4px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: var(--muted) !important;
    background: transparent !important;
    border-radius: 6px;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: #0d1117 !important;
}

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: #0d1117 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 8px 20px !important;
    letter-spacing: 0.5px;
}
.stButton > button:hover { opacity: 0.85; }

/* Info/warning boxes */
.stAlert { border-radius: 8px !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; }

/* Selectbox, multiselect */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background-color: var(--bg-card2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

/* Section divider */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
    margin: 20px 0 14px 0;
}

/* AI report box */
.ai-report {
    background: linear-gradient(135deg, #1c2128, #161b22);
    border: 1px solid var(--accent2);
    border-left: 4px solid var(--accent2);
    border-radius: 8px;
    padding: 18px 22px;
    margin: 12px 0;
    font-size: 14px;
    line-height: 1.7;
    color: var(--text);
}

/* Note box */
.note-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 3px solid var(--accent);
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 13px;
    color: var(--muted);
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS & MAPPINGS
# ─────────────────────────────────────────────
COINS = {
    "Bitcoin":     {"ticker": "BTC-USD",  "model": "BiLSTM",   "model_file": "Bitcoin_Final.pth",  "color": "#f7931a"},
    "Ethereum":    {"ticker": "ETH-USD",  "model": "XGBoost",  "model_file": "Ethereum_Final.pkl", "color": "#627eea"},
    "Ripple":      {"ticker": "XRP-USD",  "model": "XGBoost",  "model_file": "Ripple_Final.pkl",   "color": "#346aa9"},
    "Solana":      {"ticker": "SOL-USD",  "model": "XGBoost",  "model_file": "Solana_Final.pkl",   "color": "#9945ff"},
    "Binance Coin":{"ticker": "BNB-USD",  "model": "BiLSTM",   "model_file": "Binance_Final.pth",  "color": "#f3ba2f"},
}

FEATURE_COLS = [
    "High",
    "Low",
    "Open",
    "Volume",
    "Daily_Return",
    "Close_Lag_1",
    "Close_Lag_7",
    "Close_Lag_14",
    "Close_Lag_30",
    "Close_Lag_90",
    "Close_Lag_180",
    "MA_7",
    "MA_21",
    "MA_60",
    "MA_180",
    "Volatility_7",
    "Volatility_21",
    "Volatility_60",
    "Volatility_180",
    "HL_Spread",
    "Volume_MA_7",
    "Volume_MA_21",
    "Volume_MA_60",
    "Volume_MA_180",
    "Volume_Change",
    "Target_Return_7",
    "Target_Return_30",
    "Target_Return_365",
    "Target_Up",
]

HORIZONS = [7, 30, 90, 180]
SEQ_LEN   = 30     # BiLSTM sequence length
HIDDEN    = 78     # BiLSTM hidden size

# ─────────────────────────────────────────────
# BILSTM ARCHITECTURE  (must match saved weights)
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
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        return self.fc(out)

# ─────────────────────────────────────────────
# FEATURE ENGINEERING  (mirrors feature_engineering.py)
# ─────────────────────────────────────────────
def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    if 'Date' in data.columns:
        data.set_index('Date', inplace=True)

    data.sort_index(inplace=True)

    # Price-based features
    data['Daily_Return'] = data['Close'].pct_change()
    for lag in [1, 7, 14, 30, 90, 180]:
        data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)

    # Moving averages
    for window in [7, 21, 60, 180]:
        data[f'MA_{window}'] = data['Close'].rolling(window).mean()

    # Volatility
    for window in [7, 21, 60, 180]:
        data[f'Volatility_{window}'] = data['Daily_Return'].rolling(window).std()

    # High-Low spread
    data['HL_Spread'] = data['High'] - data['Low']

    # Volume-based features
    for window in [7, 21, 60, 180]:
        data[f'Volume_MA_{window}'] = data['Volume'].rolling(window).mean()
    data['Volume_Change'] = data['Volume'].pct_change()

    # Placeholder columns for inference compatibility
    data['Target_Return_7'] = 0.0
    data['Target_Return_30'] = 0.0
    data['Target_Return_365'] = 0.0
    data['Target_Up'] = 0

    return data

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_historical_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data from yfinance for the given ticker and date range."""
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


@st.cache_data(ttl=86400, show_spinner=False)
def get_earliest_date(ticker: str) -> str:
    """Return the earliest available date for a given yfinance ticker."""
    try:
        info = yf.Ticker(ticker).history(period="max", progress=False)
        if not info.empty:
            return str(info.index.min().date())
    except Exception:
        pass
    # Fallback known dates
    defaults = {
        "BTC-USD": "2014-09-17", "ETH-USD": "2015-08-07",
        "XRP-USD": "2013-08-04", "SOL-USD": "2020-04-10",
        "BNB-USD": "2017-11-09",
    }
    return defaults.get(ticker, "2017-01-01")

# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_bilstm_model(model_path: str, input_size: int) -> BiLSTM:
    """Load a BiLSTM .pth file onto CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTM(input_size=input_size, hidden_size=HIDDEN, num_layers=2, output_size=1)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def load_xgboost_model(model_path: str):
    """Load an XGBoost .pkl file saved with joblib."""
    return joblib.load(model_path)


@st.cache_resource(show_spinner=False)
def load_scaler(scaler_path: str):
    """Load a scaler .pkl file saved with joblib."""
    return joblib.load(scaler_path)

# ─────────────────────────────────────────────
# FORECASTING LOGIC
# ─────────────────────────────────────────────
def recursive_forecast_bilstm(coin: str, df_raw: pd.DataFrame, horizon: int) -> pd.Series:
    """
    Recursively forecast `horizon` future Close prices using the BiLSTM model.
    Close-derived features are updated each step; High/Low/Volume features
    are carried forward from the last observed values.
    """
    # Load model and scalers
    coin_key = coin.replace(" ", "").lower()
    if coin == "Bitcoin":
        prefix = "btc"
    else:
        prefix = "bnb"

    x_scaler_path = f"{prefix}_final_x_scaler.pkl"
    y_scaler_path = f"{prefix}_final_y_scaler.pkl"
    model_path    = COINS[coin]["model_file"]

    x_scaler = load_scaler(x_scaler_path)
    y_scaler = load_scaler(y_scaler_path)

    # Prepare feature dataframe
    df = calculate_features(df_raw.copy())
    df = df.dropna(subset=FEATURE_COLS)

    input_size = len(FEATURE_COLS)
    model = load_bilstm_model(model_path, input_size)
    device = next(model.parameters()).device

    # Working buffer – we append predicted Close values here
    close_series  = list(df["Close"].values)
    high_series   = list(df["High"].values)
    low_series    = list(df["Low"].values)
    volume_series = list(df["Volume"].values)

    # Last-known carry-forward values for non-target inputs
    last_hl   = df["HL_Spread"].iloc[-1]
    last_vol  = df["Volume"].iloc[-1]

    forecasts = []
    last_date = df.index[-1]

    for step in range(horizon):
        # Build a temporary extended series for feature calculation
        n = len(close_series)

        def _roll_mean(arr, w):
            return np.mean(arr[-w:]) if len(arr) >= w else np.mean(arr)

        def _roll_std(arr, w):
            return np.std(arr[-w:], ddof=1) if len(arr) >= w else 0.0

        def _lag(arr, k):
            return arr[-k] if len(arr) >= k else arr[0]

        feat = {
            "High":           high_series[-1],
            "Low":            low_series[-1],
            "Open":           close_series[-1],
            "Volume":         volume_series[-1],

            "Daily_Return":   (close_series[-1] - close_series[-2]) / close_series[-2] if n >= 2 else 0.0,
            "Close_Lag_1":    _lag(close_series, 1),
            "Close_Lag_7":    _lag(close_series, 7),
            "Close_Lag_14":   _lag(close_series, 14),
            "Close_Lag_30":   _lag(close_series, 30),
            "Close_Lag_90":   _lag(close_series, 90),
            "Close_Lag_180":  _lag(close_series, 180),
            "MA_7":           _roll_mean(close_series, 7),
            "MA_21":          _roll_mean(close_series, 21),
            "MA_60":          _roll_mean(close_series, 60),
            "MA_180":         _roll_mean(close_series, 180),
            "Volatility_7":   _roll_std(close_series, 7),
            "Volatility_21":  _roll_std(close_series, 21),
            "Volatility_60":  _roll_std(close_series, 60),
            "Volatility_180": _roll_std(close_series, 180),
            "HL_Spread":      last_hl,
            "Volume_MA_7":    np.mean(volume_series[-7:])   if n >= 7   else np.mean(volume_series),
            "Volume_MA_21":   np.mean(volume_series[-21:])  if n >= 21  else np.mean(volume_series),
            "Volume_MA_60":   np.mean(volume_series[-60:])  if n >= 60  else np.mean(volume_series),
            "Volume_MA_180":  np.mean(volume_series[-180:]) if n >= 180 else np.mean(volume_series),
            "Volume_Change":  (volume_series[-1] - volume_series[-2]) / volume_series[-2] if n >= 2 else 0.0,

            # Approximated target/helper features carried forward for inference compatibility
            "Target_Return_7":   0.0,
            "Target_Return_30":  0.0,
            "Target_Return_365": 0.0,
            "Target_Up":         0,
        }

        row = np.array([[feat[c] for c in FEATURE_COLS]], dtype=np.float32)

        # Build sequence: use the last SEQ_LEN scaled feature rows
        # We scale the new row and append to the scaled history
        row_scaled = x_scaler.transform(row)

        # On first step, pre-build scaled history from historical data
        if step == 0:
            hist_feats = df[FEATURE_COLS].values[-SEQ_LEN:].astype(np.float32)
            hist_scaled = x_scaler.transform(hist_feats)
            seq_buffer  = list(hist_scaled)

        seq_buffer.append(row_scaled[0])
        seq = np.array(seq_buffer[-SEQ_LEN:], dtype=np.float32)
        seq_tensor = torch.tensor(seq).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_scaled = model(seq_tensor).cpu().numpy()

        pred_close = float(y_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0])

        # Advance buffers
        close_series.append(pred_close)
        volume_series.append(last_vol)

        next_date = last_date + timedelta(days=step + 1)
        forecasts.append((next_date, pred_close))

    dates = [f[0] for f in forecasts]
    vals  = [f[1] for f in forecasts]
    return pd.Series(vals, index=pd.DatetimeIndex(dates), name="Forecast_Close")


def recursive_forecast_xgboost(coin: str, df_raw: pd.DataFrame, horizon: int) -> pd.Series:
    """
    Recursively forecast `horizon` future Close prices using the XGBoost model.
    Same carry-forward logic as BiLSTM version.
    """
    model_path = COINS[coin]["model_file"]
    model = load_xgboost_model(model_path)

    df = calculate_features(df_raw.copy())
    df = df.dropna(subset=FEATURE_COLS)

    close_series  = list(df["Close"].values)
    volume_series = list(df["Volume"].values)
    last_hl  = df["HL_Spread"].iloc[-1]
    last_vol = df["Volume"].iloc[-1]

    forecasts = []
    last_date = df.index[-1]

    for step in range(horizon):
        n = len(close_series)

        def _roll_mean(arr, w):
            return np.mean(arr[-w:]) if len(arr) >= w else np.mean(arr)

        def _roll_std(arr, w):
            return np.std(arr[-w:], ddof=1) if len(arr) >= w else 0.0

        def _lag(arr, k):
            return arr[-k] if len(arr) >= k else arr[0]

        feat = {
            "Daily_Return":   (close_series[-1] - close_series[-2]) / close_series[-2] if n >= 2 else 0.0,
            "Close_Lag_1":    _lag(close_series, 1),
            "Close_Lag_7":    _lag(close_series, 7),
            "Close_Lag_14":   _lag(close_series, 14),
            "Close_Lag_30":   _lag(close_series, 30),
            "Close_Lag_90":   _lag(close_series, 90),
            "Close_Lag_180":  _lag(close_series, 180),
            "MA_7":           _roll_mean(close_series, 7),
            "MA_21":          _roll_mean(close_series, 21),
            "MA_60":          _roll_mean(close_series, 60),
            "MA_180":         _roll_mean(close_series, 180),
            "Volatility_7":   _roll_std(close_series, 7),
            "Volatility_21":  _roll_std(close_series, 21),
            "Volatility_60":  _roll_std(close_series, 60),
            "Volatility_180": _roll_std(close_series, 180),
            "HL_Spread":      last_hl,
            "Volume_MA_7":    np.mean(volume_series[-7:])   if n >= 7   else np.mean(volume_series),
            "Volume_MA_21":   np.mean(volume_series[-21:])  if n >= 21  else np.mean(volume_series),
            "Volume_MA_60":   np.mean(volume_series[-60:])  if n >= 60  else np.mean(volume_series),
            "Volume_MA_180":  np.mean(volume_series[-180:]) if n >= 180 else np.mean(volume_series),
            "Volume_Change":  (volume_series[-1] - volume_series[-2]) / volume_series[-2] if n >= 2 else 0.0,
        }

        row = np.array([[feat[c] for c in FEATURE_COLS]])
        pred_close = float(model.predict(row)[0])

        close_series.append(pred_close)
        volume_series.append(last_vol)

        next_date = last_date + timedelta(days=step + 1)
        forecasts.append((next_date, pred_close))

    dates = [f[0] for f in forecasts]
    vals  = [f[1] for f in forecasts]
    return pd.Series(vals, index=pd.DatetimeIndex(dates), name="Forecast_Close")


def generate_forecast(coin: str, df_raw: pd.DataFrame, horizon: int) -> pd.Series:
    """Route to the correct forecasting function based on coin model type."""
    try:
        model_type = COINS[coin]["model"]
        if model_type == "BiLSTM":
            return recursive_forecast_bilstm(coin, df_raw, horizon)
        else:
            return recursive_forecast_xgboost(coin, df_raw, horizon)
    except FileNotFoundError as e:
        st.error(f"⚠️ Model file not found: {e}. Please ensure all model files are in the same directory as this app.")
        return pd.Series(dtype=float)
    except Exception as e:
        st.error(f"⚠️ Forecasting error for {coin}: {e}")
        return pd.Series(dtype=float)

# ─────────────────────────────────────────────
# AI REPORT (Gemini)
# ─────────────────────────────────────────────
def generate_ai_report(prompt: str) -> str:
    """
    Generate a short AI report using the Gemini API.
    Returns an empty string if unavailable or rate-limited.
    """
    try:
        import google.generativeai as genai

        # Retrieve key from Streamlit secrets or environment variable
        api_key = None
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            api_key = os.environ.get("GEMINI_API_KEY", "")

        if not api_key:
            return ""

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            f"You are a concise financial analyst. {prompt} "
            "Write a clear, informative paragraph of 80–150 words. "
            "Avoid speculation. Focus only on the data provided."
        )
        return response.text.strip()
    except Exception:
        return ""

# ─────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────
DARK_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(color="#e6edf3", family="DM Sans"),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
    margin=dict(l=10, r=10, t=40, b=10),
)

def line_chart(series_dict: dict, title: str, yaxis_title: str = "Price (USD)") -> go.Figure:
    """Create a multi-series line chart."""
    fig = go.Figure()
    for label, (s, color) in series_dict.items():
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=label,
                                  line=dict(color=color, width=2)))
    fig.update_layout(title=title, yaxis_title=yaxis_title, **DARK_LAYOUT)
    return fig


def candlestick_chart(df: pd.DataFrame, title: str) -> go.Figure:
    """Create a candlestick OHLC chart."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.04)
    fig.add_trace(
        go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                       low=df["Low"], close=df["Close"],
                       increasing_line_color="#3fb950",
                       decreasing_line_color="#f85149", name="OHLC"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"],
               marker_color=["#3fb950" if c >= o else "#f85149"
                             for c, o in zip(df["Close"], df["Open"])],
               name="Volume", opacity=0.7),
        row=2, col=1,
    )
    fig.update_layout(title=title, xaxis_rangeslider_visible=False,
                      yaxis_title="Price (USD)", yaxis2_title="Volume",
                      **DARK_LAYOUT)
    return fig

# ─────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────
def coin_info_box(coin: str) -> None:
    """Render a styled info card for a coin in Overview."""
    info = COINS[coin]
    earliest = get_earliest_date(info["ticker"])
    color = info["color"]
    model_type = info["model"]
    model_badge = "🔵 BiLSTM" if model_type == "BiLSTM" else "🟢 XGBoost"
    st.markdown(f"""
    <div class="crypto-card" style="border-left: 4px solid {color};">
        <b style="color:{color}; font-family:Space Mono,monospace; font-size:15px;">{coin}</b>
        &nbsp;&nbsp;
        <span style="color:#8b949e; font-size:12px;">{info['ticker']}</span>
        <br/>
        <span style="font-size:13px; color:#8b949e;">Model: {model_badge}</span>
        &nbsp;&nbsp;
        <span style="font-size:13px; color:#8b949e;">Target: Close Price</span>
        <br/>
        <span style="font-size:12px; color:#58a6ff;">📅 Data available from: {earliest}</span>
    </div>
    """, unsafe_allow_html=True)


def metric_row(metrics: list) -> None:
    """
    Render a horizontal row of metric boxes.
    metrics: list of (label, value, css_class) tuples
    """
    html = '<div class="metric-row">'
    for label, value, cls in metrics:
        html += f'''
        <div class="metric-box">
            <div class="metric-label">{label}</div>
            <div class="metric-value {cls}">{value}</div>
        </div>'''
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def format_price(v: float) -> str:
    return f"${v:,.2f}"


def format_pct(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}%"

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("# 📈 CryptoForecast")
        st.markdown("<div style='color:#8b949e;font-size:12px;'>University Research Dashboard</div>", unsafe_allow_html=True)
        st.divider()
        st.markdown("### Model Info")
        st.markdown("""
        <div class="note-box">
        <b>BiLSTM</b> → Bitcoin, Binance Coin<br>
        <b>XGBoost</b> → Ethereum, Ripple, Solana<br><br>
        • Target variable: <b>Close Price</b><br>
        • BiLSTM sequence length: <b>30 days</b><br>
        • Forecasting: <b>Recursive one-step-ahead</b><br>
        • Long-horizon forecasts are approximate
        </div>
        """, unsafe_allow_html=True)
        st.divider()
        st.markdown("<div style='color:#8b949e;font-size:11px;'>⚠️ For academic use only. Not financial advice.</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TAB 1 – OVERVIEW
# ─────────────────────────────────────────────
def tab_overview():
    st.markdown("# CryptoForecast Dashboard")
    st.markdown(
        "<div style='color:#8b949e; font-size:14px; margin-bottom:20px;'>"
        "Explore historical data, generate price forecasts, and compare trends for "
        "the top 5 cryptocurrencies using state-of-the-art machine learning models."
        "</div>",
        unsafe_allow_html=True,
    )

    selected = st.multiselect(
        "Select coins to display",
        list(COINS.keys()),
        default=list(COINS.keys()),
        key="overview_coins",
    )

    if not selected:
        st.warning("Please select at least one coin.")
        return

    cols = st.columns(min(len(selected), 3))
    for i, coin in enumerate(selected):
        with cols[i % 3]:
            coin_info_box(coin)

    st.markdown("<div class='section-title'>AI Coin Summaries</div>", unsafe_allow_html=True)
    if st.button("✨ Generate AI Coin Summaries", key="overview_ai"):
        with st.spinner("Generating summaries via Gemini…"):
            for coin in selected:
                info = COINS[coin]
                earliest = get_earliest_date(info["ticker"])
                prompt = (
                    f"Give a brief academic overview of {coin} ({info['ticker']}) cryptocurrency. "
                    f"Mention its use case, market position, and the fact that data is available "
                    f"from {earliest}. Model used: {info['model']}."
                )
                report = generate_ai_report(prompt)
                if report:
                    st.markdown(f"**{coin}**")
                    st.markdown(f'<div class="ai-report">{report}</div>', unsafe_allow_html=True)
                else:
                    st.info(f"AI summary unavailable for {coin}. Check your GEMINI_API_KEY in Streamlit secrets.")


# ─────────────────────────────────────────────
# TAB 2 – HISTORICAL DATA
# ─────────────────────────────────────────────
def tab_historical():
    st.markdown("## Historical Market Data")

    coin = st.selectbox("Select Coin", list(COINS.keys()), key="hist_coin")
    ticker = COINS[coin]["ticker"]
    earliest = get_earliest_date(ticker)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.strptime(earliest, "%Y-%m-%d").date(),
            min_value=datetime.strptime(earliest, "%Y-%m-%d").date(),
            max_value=datetime.today().date() - timedelta(days=1),
            key="hist_start",
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.today().date(),
            min_value=datetime.strptime(earliest, "%Y-%m-%d").date(),
            max_value=datetime.today().date(),
            key="hist_end",
        )

    if start_date >= end_date:
        st.error("❌ Invalid date selection. Please choose a valid period based on the available historical data for the selected coin.")
        return

    with st.spinner(f"Fetching {coin} data from yfinance…"):
        df = load_historical_data(ticker, str(start_date), str(end_date))

    if df.empty:
        st.error("No data returned for the selected range. Please adjust the dates.")
        return

    # Metrics
    latest_close = float(df["Close"].iloc[-1])
    first_close  = float(df["Close"].iloc[0])
    chg_abs  = latest_close - first_close
    chg_pct  = chg_abs / first_close * 100
    chg_cls  = "green" if chg_abs >= 0 else "red"

    metric_row([
        ("Latest Close", format_price(latest_close), ""),
        ("Period Change", format_price(chg_abs), chg_cls),
        ("% Change", format_pct(chg_pct), chg_cls),
        ("Data Points", f"{len(df):,}", ""),
    ])

    # Tabs within section
    t1, t2, t3 = st.tabs(["📈 Line Chart", "🕯️ Candlestick", "📋 Data Table"])

    with t1:
        fig = line_chart(
            {f"{coin} Close": (df["Close"], COINS[coin]["color"])},
            f"{coin} — Historical Close Price",
        )
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        fig_c = candlestick_chart(df, f"{coin} — OHLC + Volume")
        st.plotly_chart(fig_c, use_container_width=True)

    with t3:
        st.dataframe(df.reset_index().rename(columns={"index": "Date"}), use_container_width=True)
        csv = df.reset_index().to_csv(index=False).encode()
        st.download_button("⬇️ Download CSV", csv, f"{coin}_historical.csv", "text/csv")

    # AI Report
    st.markdown("---")
    if st.button("✨ Generate AI Historical Report", key="hist_ai"):
        with st.spinner("Generating report…"):
            summary = (
                f"Coin: {coin}, Ticker: {ticker}. "
                f"Period: {start_date} to {end_date}. "
                f"Open: ${first_close:,.2f}, Latest Close: ${latest_close:,.2f}. "
                f"Change: {format_pct(chg_pct)}. "
                f"Max: ${float(df['Close'].max()):,.2f}, Min: ${float(df['Close'].min()):,.2f}. "
                f"Avg volume: {df['Volume'].mean():.0f}. "
                "Summarise the historical price behaviour observed."
            )
            report = generate_ai_report(summary)
            if report:
                st.markdown('<div class="ai-report">' + report + '</div>', unsafe_allow_html=True)
            else:
                st.info("AI report unavailable. Check your GEMINI_API_KEY in Streamlit secrets.")


# ─────────────────────────────────────────────
# TAB 3 – FORECASTING
# ─────────────────────────────────────────────
def tab_forecasting():
    st.markdown("## Price Forecasting")

    st.markdown("""
    <div class="note-box">
    <b>Forecasting Methodology:</b> Models are trained to predict one day ahead. For multi-day forecasts,
    predictions are made recursively — each day's predicted Close price feeds back into the next step's features.
    High, Low, Volume, and related features are <b>carried forward</b> from the last observed values.
    Long-horizon forecasts (90–180 days) are approximate.
    </div>
    """, unsafe_allow_html=True)

    selected_coins = st.multiselect(
        "Select Coin(s)", list(COINS.keys()), default=["Bitcoin"], key="fc_coins"
    )
    horizon = st.selectbox("Forecast Horizon (days)", HORIZONS, key="fc_horizon")

    if not selected_coins:
        st.warning("Please select at least one coin.")
        return

    if st.button("🚀 Generate Forecast", key="fc_run"):
        all_forecasts = {}
        for coin in selected_coins:
            ticker = COINS[coin]["ticker"]
            earliest = get_earliest_date(ticker)
            # Fetch enough data for feature warmup (at least 200+ days)
            warmup_start = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
            with st.spinner(f"Forecasting {coin}…"):
                df_raw = load_historical_data(ticker, warmup_start, datetime.today().strftime("%Y-%m-%d"))
                if df_raw.empty or len(df_raw) < 200:
                    st.error(f"Not enough data for {coin}. Try a longer warmup period.")
                    continue
                fc = generate_forecast(coin, df_raw, horizon)
                if not fc.empty:
                    all_forecasts[coin] = (df_raw, fc)

        if not all_forecasts:
            return

        for coin, (df_raw, fc) in all_forecasts.items():
            st.markdown(f"### {coin}")
            color = COINS[coin]["color"]
            latest_actual = float(df_raw["Close"].iloc[-1])
            final_fc      = float(fc.iloc[-1])
            chg_abs = final_fc - latest_actual
            chg_pct = chg_abs / latest_actual * 100
            chg_cls = "green" if chg_abs >= 0 else "red"

            metric_row([
                ("Latest Actual Close", format_price(latest_actual), ""),
                ("Forecast Final Close", format_price(final_fc), chg_cls),
                ("Absolute Change", format_price(chg_abs), chg_cls),
                ("% Change", format_pct(chg_pct), chg_cls),
            ])

            # Chart: last 60 actual + forecast
            recent_actual = df_raw["Close"].tail(60)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=recent_actual.index, y=recent_actual.values,
                                      name="Historical", line=dict(color=color, width=2)))
            fig.add_trace(go.Scatter(x=fc.index, y=fc.values,
                                      name="Forecast",
                                      line=dict(color="#f0b429", width=2, dash="dash")))
            split_x = df_raw.index[-1]
            split_y = max(float(recent_actual.max()), float(fc.max()))

            fig.add_vline(x=split_x, line_dash="dot", line_color="#8b949e")
            fig.add_annotation(
                    x=split_x,
                    y=split_y,
                    text="Today",
                    showarrow=False,
                    yshift=10,
                    font=dict(color="#8b949e")
            )
                  
            fig.update_layout(title=f"{coin} — {horizon}-Day Forecast", **DARK_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

            # Table
            fc_df = fc.reset_index()
            fc_df.columns = ["Date", "Forecast Close (USD)"]
            fc_df["Date"] = fc_df["Date"].dt.strftime("%Y-%m-%d")
            fc_df["Forecast Close (USD)"] = fc_df["Forecast Close (USD)"].round(2)
            st.dataframe(fc_df, use_container_width=True)
            csv = fc_df.to_csv(index=False).encode()
            st.download_button(f"⬇️ Download {coin} Forecast CSV",
                               csv, f"{coin}_forecast_{horizon}d.csv", "text/csv",
                               key=f"dl_{coin}_{horizon}")

        # AI Report
        st.markdown("---")
        if st.button("✨ Generate AI Forecast Report", key="fc_ai"):
            with st.spinner("Generating AI report…"):
                summary_parts = []
                for coin, (df_raw, fc) in all_forecasts.items():
                    la = float(df_raw["Close"].iloc[-1])
                    ff = float(fc.iloc[-1])
                    chg = (ff - la) / la * 100
                    summary_parts.append(
                        f"{coin}: Latest close=${la:,.2f}, {horizon}-day forecast=${ff:,.2f}, change={chg:+.2f}%"
                    )
                prompt = (
                    f"Cryptocurrency {horizon}-day price forecast summary. "
                    + "; ".join(summary_parts)
                    + ". Provide a brief analytical commentary on these forecasts."
                )
                report = generate_ai_report(prompt)
                if report:
                    st.markdown('<div class="ai-report">' + report + '</div>', unsafe_allow_html=True)
                else:
                    st.info("AI report unavailable. Check your GEMINI_API_KEY.")


# ─────────────────────────────────────────────
# TAB 4 – PAST vs FORECAST COMPARISON
# ─────────────────────────────────────────────
def tab_comparison():
    st.markdown("## Past vs Forecast Comparison")
    st.markdown("<div style='color:#8b949e;font-size:13px;'>Compare the most recent historical window against the forecasted future window of the same length.</div>", unsafe_allow_html=True)

    coin    = st.selectbox("Select Coin", list(COINS.keys()), key="cmp_coin")
    horizon = st.selectbox("Horizon (days)", HORIZONS, key="cmp_horizon")

    if st.button("🔍 Run Comparison", key="cmp_run"):
        ticker  = COINS[coin]["ticker"]
        color   = COINS[coin]["color"]
        warmup  = (datetime.today() - timedelta(days=500)).strftime("%Y-%m-%d")

        with st.spinner("Fetching data and generating forecast…"):
            df_raw = load_historical_data(ticker, warmup, datetime.today().strftime("%Y-%m-%d"))

        if df_raw.empty or len(df_raw) < 200:
            st.error("Not enough historical data.")
            return

        past_window = df_raw["Close"].tail(horizon)

        with st.spinner("Generating forecast…"):
            fc = generate_forecast(coin, df_raw, horizon)

        if fc.empty:
            return

        # Combined chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=past_window.index, y=past_window.values,
            name=f"Past {horizon} Days", line=dict(color=color, width=2)
        ))
        fig.add_trace(go.Scatter(
            x=fc.index, y=fc.values,
            name=f"Next {horizon} Days (Forecast)",
            line=dict(color="#f0b429", width=2, dash="dash")
        ))
        split_x = df_raw.index[-1]
        split_y = max(float(past_window.max()), float(fc.max()))

        fig.add_vline(x=split_x, line_dash="dot", line_color="#8b949e")
        fig.add_annotation(
                x=split_x,
                y=split_y, 
                text="Today",
                showarrow=False,
                yshift=10,
                font=dict(color="#8b949e")
        )

        fig.update_layout(title=f"{coin} — Past {horizon}d vs Forecast {horizon}d", **DARK_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        # Metrics
        past_start = float(past_window.iloc[0])
        past_end   = float(past_window.iloc[-1])
        fc_end     = float(fc.iloc[-1])
        past_chg   = (past_end - past_start) / past_start * 100
        fc_chg     = (fc_end - past_end) / past_end * 100

        metric_row([
            ("Past Period Start", format_price(past_start), ""),
            ("Past Period End",   format_price(past_end),   ""),
            ("Past % Change",     format_pct(past_chg),    "green" if past_chg >= 0 else "red"),
            ("Forecast % Change", format_pct(fc_chg),      "green" if fc_chg >= 0 else "red"),
        ])

        # Side-by-side table
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            st.markdown("**Past Window**")
            past_df = past_window.reset_index()
            past_df.columns = ["Date", "Close (USD)"]
            past_df["Date"] = past_df["Date"].dt.strftime("%Y-%m-%d")
            past_df["Close (USD)"] = past_df["Close (USD)"].round(2)
            st.dataframe(past_df, use_container_width=True)
        with t_col2:
            st.markdown("**Forecast Window**")
            fc_df = fc.reset_index()
            fc_df.columns = ["Date", "Forecast Close (USD)"]
            fc_df["Date"] = fc_df["Date"].dt.strftime("%Y-%m-%d")
            fc_df["Forecast Close (USD)"] = fc_df["Forecast Close (USD)"].round(2)
            st.dataframe(fc_df, use_container_width=True)

        # AI Report
        st.markdown("---")
        if st.button("✨ Generate AI Comparison Report", key="cmp_ai"):
            with st.spinner("Generating…"):
                prompt = (
                    f"Crypto comparison for {coin}. "
                    f"Past {horizon} days: start=${past_start:,.2f}, end=${past_end:,.2f}, change={past_chg:+.2f}%. "
                    f"Forecast next {horizon} days: end=${fc_end:,.2f}, change={fc_chg:+.2f}%. "
                    "Briefly compare the past trend and the forecasted trend."
                )
                report = generate_ai_report(prompt)
                if report:
                    st.markdown('<div class="ai-report">' + report + '</div>', unsafe_allow_html=True)
                else:
                    st.info("AI report unavailable. Check your GEMINI_API_KEY.")


# ─────────────────────────────────────────────
# TAB 5 – MULTI-COIN COMPARISON
# ─────────────────────────────────────────────
def tab_multi_coin():
    st.markdown("## Multi-Coin Forecast Comparison")

    selected = st.multiselect(
        "Select Coins to Compare",
        list(COINS.keys()),
        default=["Bitcoin", "Ethereum"],
        key="multi_coins",
    )
    horizon = st.selectbox("Forecast Horizon (days)", HORIZONS, key="multi_horizon")

    if len(selected) < 2:
        st.warning("Please select at least two coins for comparison.")
        return

    if st.button("🔍 Compare Forecasts", key="multi_run"):
        warmup = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
        all_fc = {}

        for coin in selected:
            ticker = COINS[coin]["ticker"]
            with st.spinner(f"Forecasting {coin}…"):
                df_raw = load_historical_data(ticker, warmup, datetime.today().strftime("%Y-%m-%d"))
                if df_raw.empty or len(df_raw) < 200:
                    st.error(f"Insufficient data for {coin}.")
                    continue
                fc = generate_forecast(coin, df_raw, horizon)
                if not fc.empty:
                    all_fc[coin] = (float(df_raw["Close"].iloc[-1]), fc)

        if not all_fc:
            return

        # Raw price chart
        series_dict = {
            coin: (data[1], COINS[coin]["color"])
            for coin, data in all_fc.items()
        }
        fig_raw = line_chart(series_dict, f"Raw Forecast Prices — {horizon} Days")
        st.plotly_chart(fig_raw, use_container_width=True)

        # Normalised chart (base 100)
        fig_norm = go.Figure()
        for coin, (_, fc) in all_fc.items():
            normalised = (fc / fc.iloc[0]) * 100
            fig_norm.add_trace(go.Scatter(
                x=normalised.index, y=normalised.values,
                name=coin, line=dict(color=COINS[coin]["color"], width=2)
            ))
        fig_norm.update_layout(
            title=f"Normalised Forecast Comparison (Base = 100) — {horizon} Days",
            yaxis_title="Indexed Price",
            **DARK_LAYOUT,
        )
        st.plotly_chart(fig_norm, use_container_width=True)

        # Summary metrics
        metric_parts = []
        for coin, (latest_actual, fc) in all_fc.items():
            fc_end  = float(fc.iloc[-1])
            chg_pct = (fc_end - latest_actual) / latest_actual * 100
            chg_cls = "green" if chg_pct >= 0 else "red"
            metric_parts.append(
                (f"{coin} ({horizon}d)", format_price(fc_end), chg_cls)
            )
        metric_row(metric_parts)

        # Combined table
        fc_table = pd.DataFrame({
            coin: all_fc[coin][1].values for coin in all_fc
        }, index=all_fc[list(all_fc.keys())[0]][1].index)
        fc_table.index = fc_table.index.strftime("%Y-%m-%d")
        fc_table = fc_table.round(2)
        st.dataframe(fc_table, use_container_width=True)

        csv = fc_table.reset_index().to_csv(index=False).encode()
        st.download_button("⬇️ Download Comparison CSV", csv,
                           f"multi_coin_forecast_{horizon}d.csv", "text/csv")

        # AI Report
        st.markdown("---")
        if st.button("✨ Generate AI Multi-Coin Report", key="multi_ai"):
            with st.spinner("Generating…"):
                parts = []
                for coin, (la, fc) in all_fc.items():
                    fe = float(fc.iloc[-1])
                    chg = (fe - la) / la * 100
                    parts.append(f"{coin}: ${la:,.2f} → ${fe:,.2f} ({chg:+.2f}%)")
                prompt = (
                    f"Multi-cryptocurrency {horizon}-day forecast comparison. "
                    + "; ".join(parts)
                    + ". Provide a brief comparative analysis highlighting relative performance."
                )
                report = generate_ai_report(prompt)
                if report:
                    st.markdown('<div class="ai-report">' + report + '</div>', unsafe_allow_html=True)
                else:
                    st.info("AI report unavailable. Check your GEMINI_API_KEY.")


# ─────────────────────────────────────────────
# TAB 6 – AI REPORT HUB
# ─────────────────────────────────────────────
def tab_ai_report():
    st.markdown("## AI Report Hub")
    st.markdown(
        "<div style='color:#8b949e;font-size:13px;'>Generate a custom AI-written report on any coin and horizon. "
        "Reports are powered by Google Gemini and are ~80–150 words.</div>",
        unsafe_allow_html=True,
    )

    coin    = st.selectbox("Select Coin", list(COINS.keys()), key="ai_hub_coin")
    horizon = st.selectbox("Forecast Horizon", HORIZONS, key="ai_hub_horizon")
    report_type = st.selectbox(
        "Report Type",
        ["Overall Summary", "Risk Analysis", "Market Context", "Investment Considerations"],
        key="ai_hub_type",
    )

    if st.button("✨ Generate Full AI Report", key="ai_hub_run"):
        ticker  = COINS[coin]["ticker"]
        warmup  = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
        model_t = COINS[coin]["model"]

        with st.spinner("Fetching data…"):
            df_raw = load_historical_data(ticker, warmup, datetime.today().strftime("%Y-%m-%d"))

        if df_raw.empty or len(df_raw) < 200:
            st.error("Not enough data.")
            return

        latest_close = float(df_raw["Close"].iloc[-1])
        high_30 = float(df_raw["Close"].tail(30).max())
        low_30  = float(df_raw["Close"].tail(30).min())
        vol_30  = float(df_raw["Close"].tail(30).std())

        with st.spinner("Generating forecast…"):
            fc = generate_forecast(coin, df_raw, horizon)

        if fc.empty:
            return

        fc_end = float(fc.iloc[-1])
        fc_chg = (fc_end - latest_close) / latest_close * 100

        prompt = (
            f"Report type: {report_type}. "
            f"Coin: {coin} ({ticker}), Model: {model_t}. "
            f"Latest close: ${latest_close:,.2f}. "
            f"30-day range: ${low_30:,.2f}–${high_30:,.2f}, Volatility: ${vol_30:,.2f}. "
            f"{horizon}-day forecast: ${fc_end:,.2f} ({fc_chg:+.2f}% change). "
            f"Write a {report_type.lower()} for this cryptocurrency."
        )

        with st.spinner("Generating AI report via Gemini…"):
            report = generate_ai_report(prompt)

        if report:
            st.markdown(f"### {report_type}: {coin} ({horizon}-Day Horizon)")
            st.markdown(f'<div class="ai-report">{report}</div>', unsafe_allow_html=True)
        else:
            st.info(
                "AI report unavailable. Ensure your GEMINI_API_KEY is set in "
                "`.streamlit/secrets.toml` as:\n\n```\nGEMINI_API_KEY = 'your_key_here'\n```"
            )


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    render_sidebar()

    tab_labels = [
        "🏠 Overview",
        "📊 Historical Data",
        "🔮 Forecasting",
        "⚖️ Past vs Forecast",
        "🌐 Multi-Coin",
        "🤖 AI Report",
    ]

    tabs = st.tabs(tab_labels)

    with tabs[0]:
        tab_overview()
    with tabs[1]:
        tab_historical()
    with tabs[2]:
        tab_forecasting()
    with tabs[3]:
        tab_comparison()
    with tabs[4]:
        tab_multi_coin()
    with tabs[5]:
        tab_ai_report()


if __name__ == "__main__":
    main()