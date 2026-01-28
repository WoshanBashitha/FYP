import pandas as pd

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all features for a cryptocurrency DataFrame."""
    data = df.copy()

    # Ensure Date is index
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

    # Target variables
    for period in [7, 30, 365]:
        data[f'Target_Return_{period}'] = data['Close'].shift(-period) / data['Close'] - 1
    data['Target_Up'] = (data['Target_Return_7'] > 0).astype(int)

    # Drop NaNs from rolling and shifting
    data.dropna(inplace=True)

    return data

import os
import pandas as pd

def process_and_save(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Calculate features and save the DataFrame to CSV in feature_datasets folder."""
    
    # Ensure the folder exists
    os.makedirs("feature_datasets", exist_ok=True)
    
    # Calculate features
    processed = calculate_features(df)
    
    # Full path to save
    full_path = os.path.join("feature_datasets", filename)
    processed.to_csv(full_path)
    
    return processed