"""
Cryptocurrency Feature Engineering Module

This module provides feature engineering functions for cryptocurrency price data.
It calculates price-based features, moving averages, volatility metrics, volume features,
and target variables for machine learning models.
"""

import pandas as pd
import numpy as np
from typing import Optional


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all feature engineering features for cryptocurrency data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with Date as index and columns: Open, High, Low, Close, Volume
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with all original columns plus engineered features
    """
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Ensure Date is the index
    if 'Date' in data.columns:
        data.set_index('Date', inplace=True)
    
    # Sort by date to ensure proper calculations
    data.sort_index(inplace=True)
    
    # ==================== PRICE-BASED FEATURES ====================
    
    # Daily Return (percentage)
    data['Daily_Return'] = data['Close'].pct_change() * 100
    
    # Lagged Close Prices
    lag_periods = [1, 7, 14, 30, 90, 180]
    for lag in lag_periods:
        data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
    
    # ==================== MOVING AVERAGES ====================
    
    ma_windows = [7, 21, 60, 180]
    for window in ma_windows:
        data[f'MA_{window}'] = data['Close'].rolling(window=window).mean()
    
    # ==================== VOLATILITY ====================
    
    volatility_windows = [7, 21, 60, 180]
    for window in volatility_windows:
        data[f'Volatility_{window}'] = data['Daily_Return'].rolling(window=window).std()
    
    # ==================== HIGH-LOW SPREAD ====================
    
    data['HL_Spread'] = (data['High'] - data['Low']) / data['Close']
    
    # ==================== VOLUME-BASED FEATURES ====================
    
    # Volume Moving Averages
    volume_ma_windows = [7, 21, 60, 180]
    for window in volume_ma_windows:
        data[f'Volume_MA_{window}'] = data['Volume'].rolling(window=window).mean()
    
    # Volume Change (percentage)
    data['Volume_Change'] = data['Volume'].pct_change() * 100
    
    # ==================== TARGET VARIABLES ====================
    
    # Forward Returns (future returns)
    target_periods = [7, 30, 365]
    for period in target_periods:
        data[f'Target_Return_{period}'] = (
            data['Close'].shift(-period) / data['Close'] - 1
        ) * 100
    
    # Binary Target: 1 if 7-day forward return is positive, else 0
    data['Target_Up'] = (data['Target_Return_7'] > 0).astype(int)
    
    # ==================== DROP ROWS WITH NaN ====================
    
    data.dropna(inplace=True)
    
    return data


def process_and_save(df: pd.DataFrame, 
                     output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Process features and optionally save to CSV.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with Date as index and OHLCV columns
    output_path : str, optional
        Path to save the processed data (e.g., 'BTC_features.csv')
        
    Returns:
    --------
    pd.DataFrame
        Processed dataframe with all features
    """
    
    # Calculate features
    processed_df = calculate_features(df)
    
    # Save to CSV if path provided
    if output_path:
        processed_df.to_csv(output_path)
    
    return processed_df