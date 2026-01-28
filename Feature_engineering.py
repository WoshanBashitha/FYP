import pandas as pd
import numpy as np

def apply_feature_engineering(df, save_name=None):
    # Feature parameters
    LAGS = [1, 7, 14, 30, 90, 180]
    MAS = [7, 21, 60, 180]
    TARGET_RETURNS = [7, 30, 365]

    # ---------- Price-based features ----------
    
    # Daily return
    df["Daily_Return"] = df["Close"].pct_change()

    # Close price lags
    for lag in LAGS:
        df[f"Close_Lag_{lag}"] = df["Close"].shift(lag)

    # ---------- Moving averages ----------
    for window in MAS:
        df[f"MA_{window}"] = df["Close"].rolling(window).mean()

    # ---------- Volatility ----------
    for window in MAS:
        df[f"Volatility_{window}"] = df["Daily_Return"].rolling(window).std()

    # ---------- High-Low spread ----------
    df["HL_Spread"] = df["High"] - df["Low"]

    # ---------- Volume-based features ----------
    for window in MAS:
        df[f"Volume_MA_{window}"] = df["Volume"].rolling(window).mean()
    df["Volume_Change"] = df["Volume"].pct_change()

    # ---------- Targets ----------
    for horizon in TARGET_RETURNS:
        df[f"Target_Return_{horizon}"] = df["Close"].shift(-horizon) / df["Close"] - 1

    # Binary target (price up or not)
    df["Target_Up"] = (df["Target_Return_7"] > 0).astype(int)

    # ---------- Drop rows with NaNs caused by rolling & shifting ----------
    df = df.dropna()

    # ---------- Save engineered dataset if save_name provided ----------
    if save_name:
        df.to_csv(f"preprocessed_datasets/{save_name}_features.csv")
        print(f"\nSaved: preprocessed_datasets/{save_name}_features.csv")

    # ---------- Print full dataframe ----------
    print(df)

    return df