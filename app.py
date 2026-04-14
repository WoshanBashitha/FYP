import os
import io
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import yfinance as yf
from google import genai

from feature_engineering import calculate_features


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Cryptocurrency Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
)


# =========================
# Constants and config
# =========================
TICKER_MAP = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Ripple": "XRP-USD",
    "Solana": "SOL-USD",
    "Binance Coin": "BNB-USD",
}

MODEL_META = {
    "Bitcoin": {
        "type": "BiLSTM",
        "model_file": "Bitcoin_Final.pth",
        "x_scaler": "btc_final_x_scaler.pkl",
        "y_scaler": "btc_final_Y_scaler.pkl",
        "hidden_size": 78,
        "num_layers": 2,
        "sequence_length": 30,
    },
    "Binance Coin": {
        "type": "BiLSTM",
        "model_file": "Binance_Final.pth",
        "x_scaler": "bnb_final_x_scaler.pkl",
        "y_scaler": "bnb_final_Y_scaler.pkl",
        "hidden_size": 78,
        "num_layers": 2,
        "sequence_length": 30,
    },
    "Ripple": {
        "type": "XGBoost",
        "model_file": "Ripple_Final.pkl",
    },
    "Ethereum": {
        "type": "XGBoost",
        "model_file": "Ethereum_Final.pkl",
    },
    "Solana": {
        "type": "XGBoost",
        "model_file": "Solana_Final.pkl",
    },
}

FORECAST_HORIZONS = [7, 30, 90, 180]
SECTION_REPORT_WORDS = 120
MIN_HISTORY_BUFFER = 240  # Enough room for lag 180 + moving features + sequence

# Input features used at inference. These exclude helper/target columns.
FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
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
]

HELPER_TARGET_COLUMNS = [
    "Target_Return_7",
    "Target_Return_30",
    "Target_Return_365",
    "Target_Up",
]


# =========================
# Model classes
# =========================
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

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
        out = self.fc(out)
        return out


# =========================
# Utility helpers
# =========================
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_gemini_api_key() -> Optional[str]:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY")


def get_gemini_client() -> Optional[genai.Client]:
    api_key = get_gemini_api_key()
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def download_coin_data(coin_name: str, period: str = "max") -> pd.DataFrame:
    ticker = TICKER_MAP[coin_name]
    df = yf.download(ticker, period=period, auto_adjust=False, progress=False)

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])

    keep_cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep_cols].copy()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().sort_values("Date").reset_index(drop=True)
    return df


@st.cache_resource(show_spinner=False)
def load_pickle_file(filepath: str):
    try:
        return joblib.load(filepath)
    except Exception:
        with open(filepath, "rb") as f:
            return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_model_for_coin(coin_name: str):
    meta = MODEL_META[coin_name]

    if meta["type"] == "XGBoost":
        return load_pickle_file(meta["model_file"])

    input_size = len(FEATURE_COLUMNS)
    model = BiLSTM(
        input_size=input_size,
        hidden_size=meta["hidden_size"],
        num_layers=meta["num_layers"],
        output_size=1,
    )
    state_dict = torch.load(meta["model_file"], map_location=get_device())
    model.load_state_dict(state_dict)
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def load_scalers_for_coin(coin_name: str):
    meta = MODEL_META[coin_name]
    if meta["type"] != "BiLSTM":
        return None, None
    x_scaler = load_pickle_file(meta["x_scaler"])
    y_scaler = load_pickle_file(meta["y_scaler"])
    return x_scaler, y_scaler


def prepare_feature_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    work = raw_df.copy()
    work["Date"] = pd.to_datetime(work["Date"])
    work = work.sort_values("Date")
    engineered = calculate_features(work)
    engineered = engineered.reset_index().rename(columns={"index": "Date"})
    for helper_col in HELPER_TARGET_COLUMNS:
        if helper_col in engineered.columns:
            engineered = engineered.drop(columns=helper_col)
    return engineered


def validate_date_range(df: pd.DataFrame, start_date, end_date) -> Tuple[bool, str]:
    if df.empty:
        return False, "No data available for the selected coin."
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()

    if start_date > end_date:
        return False, "Invalid date selection. Start date must be before end date."
    if start_date < min_date or end_date > max_date:
        return False, (
            "Invalid date selection. Please choose a valid period based on the available "
            "historical data for the selected coin."
        )
    return True, ""


def get_latest_info_box(coin_name: str, df: pd.DataFrame) -> Dict[str, str]:
    latest = df.iloc[-1]
    return {
        "Coin": coin_name,
        "Ticker": TICKER_MAP[coin_name],
        "Earliest Available Date": df["Date"].min().date().isoformat(),
        "Latest Available Date": df["Date"].max().date().isoformat(),
        "Latest Close": f"{latest['Close']:.4f}",
        "Model Type": MODEL_META[coin_name]["type"],
    }


def build_bilstm_sequence(feature_df: pd.DataFrame, sequence_length: int, x_scaler) -> np.ndarray:
    seq_df = feature_df[FEATURE_COLUMNS].tail(sequence_length).copy()
    if len(seq_df) < sequence_length:
        raise ValueError("Not enough rows to build the BiLSTM input sequence.")
    seq_scaled = x_scaler.transform(seq_df.values)
    return np.expand_dims(seq_scaled, axis=0)


def predict_next_close_bilstm(feature_df: pd.DataFrame, coin_name: str) -> float:
    meta = MODEL_META[coin_name]
    model = load_model_for_coin(coin_name)
    x_scaler, y_scaler = load_scalers_for_coin(coin_name)
    sequence = build_bilstm_sequence(feature_df, meta["sequence_length"], x_scaler)

    device = get_device()
    model = model.to(device)
    with torch.no_grad():
        X_tensor = torch.tensor(sequence, dtype=torch.float32).to(device)
        pred_scaled = model(X_tensor).cpu().numpy().reshape(-1, 1)
    pred = y_scaler.inverse_transform(pred_scaled).flatten()[0]
    return float(pred)


def predict_next_close_xgb(feature_df: pd.DataFrame, coin_name: str) -> float:
    model = load_model_for_coin(coin_name)
    latest_row = feature_df[FEATURE_COLUMNS].tail(1)
    pred = model.predict(latest_row)[0]
    return float(pred)


def predict_next_close(feature_df: pd.DataFrame, coin_name: str) -> float:
    model_type = MODEL_META[coin_name]["type"]
    if model_type == "BiLSTM":
        return predict_next_close_bilstm(feature_df, coin_name)
    return predict_next_close_xgb(feature_df, coin_name)


def append_synthetic_future_row(raw_df: pd.DataFrame, predicted_close: float) -> pd.DataFrame:
    last_row = raw_df.iloc[-1].copy()
    next_date = pd.to_datetime(last_row["Date"]) + pd.Timedelta(days=1)

    synthetic = {
        "Date": next_date,
        "Open": float(last_row["Close"]),
        "High": float(last_row["High"]),
        "Low": float(last_row["Low"]),
        "Close": float(predicted_close),
        "Volume": float(last_row["Volume"]),
    }

    new_df = pd.concat([raw_df, pd.DataFrame([synthetic])], ignore_index=True)
    return new_df


def recursive_forecast(coin_name: str, raw_history_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if raw_history_df.empty:
        raise ValueError("No historical data available for forecasting.")

    raw_working = raw_history_df.copy().sort_values("Date").reset_index(drop=True)
    forecasts = []

    needed_rows = max(MIN_HISTORY_BUFFER, MODEL_META.get(coin_name, {}).get("sequence_length", 30) + 5)
    if len(raw_working) < needed_rows:
        raise ValueError(
            f"Not enough historical rows for {coin_name}. Need at least {needed_rows} rows after download."
        )

    for step in range(horizon):
        feature_df = prepare_feature_dataframe(raw_working)
        predicted_close = predict_next_close(feature_df, coin_name)
        raw_working = append_synthetic_future_row(raw_working, predicted_close)
        forecasts.append(
            {
                "Date": raw_working.iloc[-1]["Date"],
                "Forecast_Close": float(predicted_close),
                "Step": step + 1,
            }
        )

    forecast_df = pd.DataFrame(forecasts)
    forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])
    return forecast_df


def historical_chart(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price", height=430)
    return fig


def candlestick_chart(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="OHLC",
            )
        ]
    )
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price", height=500)
    return fig


def volume_chart(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="Volume"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Volume", height=280)
    return fig


def forecast_chart(forecast_df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Forecast_Close"],
            mode="lines+markers",
            name="Forecast Close",
        )
    )
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Forecasted Close", height=430)
    return fig


def past_vs_future_chart(past_df: pd.DataFrame, forecast_df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=past_df["Date"], y=past_df["Close"], mode="lines+markers", name="Past Close"))
    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Forecast_Close"],
            mode="lines+markers",
            name="Forecast Close",
        )
    )
    split_date = forecast_df["Date"].min()
    fig.add_vline(x=split_date, line_dash="dash", annotation_text="Forecast Start")
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price", height=450)
    return fig


def normalized_multicoin_chart(forecast_results: Dict[str, pd.DataFrame], title: str) -> go.Figure:
    fig = go.Figure()
    for coin, fdf in forecast_results.items():
        series = fdf["Forecast_Close"]
        if len(series) == 0:
            continue
        base = series.iloc[0]
        normalized = (series / base) * 100
        fig.add_trace(go.Scatter(x=fdf["Date"], y=normalized, mode="lines+markers", name=coin))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Normalized Index (Base=100)", height=450)
    return fig


def raw_multicoin_chart(forecast_results: Dict[str, pd.DataFrame], title: str) -> go.Figure:
    fig = go.Figure()
    for coin, fdf in forecast_results.items():
        fig.add_trace(go.Scatter(x=fdf["Date"], y=fdf["Forecast_Close"], mode="lines+markers", name=coin))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Forecasted Close", height=450)
    return fig


def dataframe_to_csv_download(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")


def format_metrics(latest_actual: float, final_forecast: float) -> Tuple[float, float]:
    absolute_change = final_forecast - latest_actual
    percentage_change = (absolute_change / latest_actual) * 100 if latest_actual != 0 else np.nan
    return absolute_change, percentage_change


def generate_ai_report(section_name: str, prompt_body: str) -> str:
    client = get_gemini_client()
    if client is None:
        raise RuntimeError("Gemini API key not found. Add GEMINI_API_KEY to Streamlit secrets or environment variables.")

    full_prompt = f"""
You are writing a concise cryptocurrency dashboard report.
Write a clear, natural, neutral report in about {SECTION_REPORT_WORDS} words.
Do not use bullet points.
Focus only on the provided data for the section: {section_name}.
Avoid financial advice language.

Data:
{prompt_body}
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=full_prompt,
    )
    return response.text.strip()


def safe_report_button(button_key: str, section_name: str, prompt_body: str):
    if st.button("Generate Report", key=button_key):
        try:
            report = generate_ai_report(section_name, prompt_body)
            st.success("Report generated successfully.")
            st.write(report)
        except Exception as e:
            st.warning(f"AI report could not be generated right now. {e}")


def summarize_coin_for_prompt(coin_name: str, df: pd.DataFrame) -> str:
    latest = df.iloc[-1]
    earliest_date = df["Date"].min().date().isoformat()
    latest_date = df["Date"].max().date().isoformat()
    return (
        f"Coin: {coin_name}\n"
        f"Ticker: {TICKER_MAP[coin_name]}\n"
        f"Model Type: {MODEL_META[coin_name]['type']}\n"
        f"Earliest Available Date: {earliest_date}\n"
        f"Latest Available Date: {latest_date}\n"
        f"Latest Close: {latest['Close']:.4f}"
    )


# =========================
# Sidebar
# =========================
st.sidebar.title("Dashboard Controls")
st.sidebar.info(
    "Target variable: Close\n\n"
    "BiLSTM sequence length: 30\n\n"
    "Long-horizon forecasts are recursive. Unknown future High, Low, and Volume inputs are carried forward from the latest observed values."
)


# =========================
# Main title
# =========================
st.title("Cryptocurrency Price Forecasting Dashboard")
st.caption(
    "Explore historical cryptocurrency prices, generate recursive future Close forecasts, compare coins, and create short AI summaries for each section."
)


tab_overview, tab_history, tab_forecast, tab_compare, tab_multi, tab_ai = st.tabs(
    [
        "Overview",
        "Historical Data",
        "Forecasting",
        "Past vs Forecast Comparison",
        "Multi-Coin Comparison",
        "AI Report Notes",
    ]
)


# =========================
# Overview tab
# =========================
with tab_overview:
    st.subheader("Project Overview")
    selected_overview_coins = st.multiselect(
        "Select one or more coins",
        options=list(TICKER_MAP.keys()),
        default=["Bitcoin", "Ethereum"],
        key="overview_coins",
    )

    if selected_overview_coins:
        cols = st.columns(min(3, len(selected_overview_coins)))
        coin_prompts = []
        for idx, coin in enumerate(selected_overview_coins):
            df = download_coin_data(coin)
            if df.empty:
                cols[idx % len(cols)].error(f"No data available for {coin}.")
                continue

            info = get_latest_info_box(coin, df)
            with cols[idx % len(cols)]:
                st.markdown(f"### {coin}")
                st.write(f"**Ticker:** {info['Ticker']}")
                st.write(f"**Model Type:** {info['Model Type']}")
                st.write(f"**Earliest Available Date:** {info['Earliest Available Date']}")
                st.write(f"**Latest Available Date:** {info['Latest Available Date']}")
                st.write(f"**Latest Close:** {info['Latest Close']}")
            coin_prompts.append(summarize_coin_for_prompt(coin, df))

        if coin_prompts:
            safe_report_button(
                "overview_ai_report",
                "Overview Coin Summary",
                "\n\n".join(coin_prompts),
            )
    else:
        st.info("Select at least one coin to view the dashboard overview.")


# =========================
# Historical Data tab
# =========================
with tab_history:
    st.subheader("Historical Data")
    hist_coin = st.selectbox("Select a coin", options=list(TICKER_MAP.keys()), key="hist_coin")
    hist_df = download_coin_data(hist_coin)

    if hist_df.empty:
        st.error("No historical data could be loaded for the selected coin.")
    else:
        min_date = hist_df["Date"].min().date()
        max_date = hist_df["Date"].max().date()
        st.write(f"Available period: **{min_date}** to **{max_date}**")

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start date", value=max(min_date, max_date - pd.Timedelta(days=365)), min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("End date", value=max_date, min_value=min_date, max_value=max_date)

        valid, msg = validate_date_range(hist_df, start_date, end_date)
        if not valid:
            st.error(msg)
        else:
            filtered_hist = hist_df[(hist_df["Date"].dt.date >= start_date) & (hist_df["Date"].dt.date <= end_date)].copy()
            if filtered_hist.empty:
                st.warning("No data found for the selected period.")
            else:
                top_col1, top_col2, top_col3 = st.columns(3)
                top_col1.metric("Rows", len(filtered_hist))
                top_col2.metric("First Close", f"{filtered_hist['Close'].iloc[0]:.4f}")
                top_col3.metric("Last Close", f"{filtered_hist['Close'].iloc[-1]:.4f}")

                st.dataframe(filtered_hist, use_container_width=True)
                dataframe_to_csv_download(filtered_hist, f"{hist_coin.lower().replace(' ', '_')}_historical_data.csv", "Download Historical Data CSV")

                st.plotly_chart(historical_chart(filtered_hist, f"{hist_coin} Close Price"), use_container_width=True)
                st.plotly_chart(candlestick_chart(filtered_hist, f"{hist_coin} Candlestick Chart"), use_container_width=True)
                st.plotly_chart(volume_chart(filtered_hist, f"{hist_coin} Volume"), use_container_width=True)

                prompt_text = (
                    f"Coin: {hist_coin}\n"
                    f"Period: {start_date} to {end_date}\n"
                    f"Rows: {len(filtered_hist)}\n"
                    f"Start Close: {filtered_hist['Close'].iloc[0]:.4f}\n"
                    f"End Close: {filtered_hist['Close'].iloc[-1]:.4f}\n"
                    f"Highest Close: {filtered_hist['Close'].max():.4f}\n"
                    f"Lowest Close: {filtered_hist['Close'].min():.4f}\n"
                    f"Average Volume: {filtered_hist['Volume'].mean():.2f}"
                )
                safe_report_button("hist_ai_report", "Historical Data", prompt_text)


# =========================
# Forecasting tab
# =========================
with tab_forecast:
    st.subheader("Future Forecasting")
    st.info(
        "Forecasts are generated recursively from one-day-ahead models. Close-derived features are updated using predicted Close values. Future High, Low, and Volume-related inputs are carried forward using the latest observed values."
    )

    forecast_coins = st.multiselect(
        "Select one or more coins for forecasting",
        options=list(TICKER_MAP.keys()),
        default=["Bitcoin"],
        key="forecast_coins",
    )
    horizon = st.selectbox("Select forecast horizon (days)", options=FORECAST_HORIZONS, index=0)

    if st.button("Run Forecast", key="run_forecast"):
        if not forecast_coins:
            st.warning("Select at least one coin to forecast.")
        else:
            forecast_prompt_parts = []
            for coin in forecast_coins:
                with st.spinner(f"Generating forecast for {coin}..."):
                    try:
                        raw_df = download_coin_data(coin)
                        forecast_df = recursive_forecast(coin, raw_df, horizon)
                        latest_actual = raw_df["Close"].iloc[-1]
                        final_forecast = forecast_df["Forecast_Close"].iloc[-1]
                        absolute_change, pct_change = format_metrics(latest_actual, final_forecast)

                        st.markdown(f"### {coin}")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Latest Actual Close", f"{latest_actual:.4f}")
                        m2.metric("Final Forecast Close", f"{final_forecast:.4f}")
                        m3.metric("Absolute Change", f"{absolute_change:.4f}")
                        m4.metric("Percentage Change", f"{pct_change:.2f}%")

                        st.dataframe(forecast_df, use_container_width=True)
                        dataframe_to_csv_download(
                            forecast_df,
                            f"{coin.lower().replace(' ', '_')}_{horizon}_day_forecast.csv",
                            f"Download {coin} Forecast CSV",
                        )
                        st.plotly_chart(
                            forecast_chart(forecast_df, f"{coin} {horizon}-Day Forecast"),
                            use_container_width=True,
                        )

                        forecast_prompt_parts.append(
                            f"Coin: {coin}\n"
                            f"Horizon: {horizon} days\n"
                            f"Latest Actual Close: {latest_actual:.4f}\n"
                            f"Final Forecast Close: {final_forecast:.4f}\n"
                            f"Absolute Change: {absolute_change:.4f}\n"
                            f"Percentage Change: {pct_change:.2f}%"
                        )
                    except Exception as e:
                        st.error(f"Forecast failed for {coin}: {e}")

            if forecast_prompt_parts:
                safe_report_button(
                    "forecast_ai_report",
                    "Forecasting",
                    "\n\n".join(forecast_prompt_parts),
                )


# =========================
# Past vs Forecast Comparison tab
# =========================
with tab_compare:
    st.subheader("Past vs Forecast Comparison")
    compare_coin = st.selectbox("Select a coin for comparison", options=list(TICKER_MAP.keys()), key="compare_coin")
    compare_horizon = st.selectbox("Select comparison horizon", options=FORECAST_HORIZONS, index=0, key="compare_horizon")

    if st.button("Compare Past and Forecast", key="compare_button"):
        try:
            raw_df = download_coin_data(compare_coin)
            past_df = raw_df.tail(compare_horizon).copy()
            forecast_df = recursive_forecast(compare_coin, raw_df, compare_horizon)

            st.dataframe(
                pd.concat(
                    [
                        past_df[["Date", "Close"]].assign(Series="Past").rename(columns={"Close": "Value"}),
                        forecast_df[["Date", "Forecast_Close"]].assign(Series="Forecast").rename(columns={"Forecast_Close": "Value"}),
                    ],
                    ignore_index=True,
                ),
                use_container_width=True,
            )

            st.plotly_chart(
                past_vs_future_chart(past_df, forecast_df, f"{compare_coin}: Recent Past vs Future Forecast"),
                use_container_width=True,
            )

            prompt_text = (
                f"Coin: {compare_coin}\n"
                f"Comparison Horizon: {compare_horizon} days\n"
                f"Past Start Close: {past_df['Close'].iloc[0]:.4f}\n"
                f"Past End Close: {past_df['Close'].iloc[-1]:.4f}\n"
                f"Forecast Start Close: {forecast_df['Forecast_Close'].iloc[0]:.4f}\n"
                f"Forecast End Close: {forecast_df['Forecast_Close'].iloc[-1]:.4f}"
            )
            safe_report_button("compare_ai_report", "Past vs Forecast Comparison", prompt_text)
        except Exception as e:
            st.error(f"Comparison could not be generated: {e}")


# =========================
# Multi-Coin Comparison tab
# =========================
with tab_multi:
    st.subheader("Multi-Coin Forecast Comparison")
    multi_coins = st.multiselect(
        "Select multiple coins",
        options=list(TICKER_MAP.keys()),
        default=["Bitcoin", "Ethereum", "Solana"],
        key="multi_coins",
    )
    multi_horizon = st.selectbox("Select common forecast horizon", options=FORECAST_HORIZONS, index=0, key="multi_horizon")

    if st.button("Run Multi-Coin Comparison", key="multi_button"):
        if len(multi_coins) < 2:
            st.warning("Select at least two coins for multi-coin comparison.")
        else:
            forecast_results = {}
            summary_rows = []
            prompt_parts = []

            for coin in multi_coins:
                try:
                    raw_df = download_coin_data(coin)
                    fdf = recursive_forecast(coin, raw_df, multi_horizon)
                    forecast_results[coin] = fdf

                    latest_actual = raw_df["Close"].iloc[-1]
                    final_forecast = fdf["Forecast_Close"].iloc[-1]
                    absolute_change, pct_change = format_metrics(latest_actual, final_forecast)
                    summary_rows.append(
                        {
                            "Coin": coin,
                            "Model Type": MODEL_META[coin]["type"],
                            "Latest Actual Close": latest_actual,
                            "Final Forecast Close": final_forecast,
                            "Absolute Change": absolute_change,
                            "Percentage Change": pct_change,
                        }
                    )
                    prompt_parts.append(
                        f"Coin: {coin}\n"
                        f"Horizon: {multi_horizon} days\n"
                        f"Latest Actual Close: {latest_actual:.4f}\n"
                        f"Final Forecast Close: {final_forecast:.4f}\n"
                        f"Percentage Change: {pct_change:.2f}%"
                    )
                except Exception as e:
                    st.error(f"{coin} comparison forecast failed: {e}")

            if forecast_results:
                summary_df = pd.DataFrame(summary_rows)
                st.dataframe(summary_df, use_container_width=True)
                dataframe_to_csv_download(summary_df, "multi_coin_forecast_summary.csv", "Download Multi-Coin Summary CSV")

                st.plotly_chart(raw_multicoin_chart(forecast_results, f"Raw {multi_horizon}-Day Forecast Comparison"), use_container_width=True)
                st.plotly_chart(normalized_multicoin_chart(forecast_results, f"Normalized {multi_horizon}-Day Forecast Comparison"), use_container_width=True)

                safe_report_button(
                    "multi_ai_report",
                    "Multi-Coin Comparison",
                    "\n\n".join(prompt_parts),
                )


# =========================
# AI Notes tab
# =========================
with tab_ai:
    st.subheader("AI Report Feature")
    st.write(
        "Each section has its own report button. The report is generated only from the data currently shown in that section. "
        "If the Gemini API key is missing, invalid, or rate-limited, the dashboard will continue to work without the AI summary."
    )
    st.code(
        """
# .streamlit/secrets.toml
GEMINI_API_KEY = "your_gemini_api_key_here"
        """.strip(),
        language="toml",
    )
    st.warning(
        "Because the raw API key was pasted into chat, rotate that key in Google AI Studio and create a new one before deploying this app."
    )
    st.markdown("**Model Information**")
    st.write("- Target variable: Close")
    st.write("- BiLSTM sequence length: 30")
    st.write("- Multi-step forecasts are recursive")
    st.write("- Long-horizon forecasts are approximate and use carried-forward non-target inputs")
