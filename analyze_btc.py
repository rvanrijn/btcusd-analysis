#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bitcoin USD Price Analysis Script

This script analyzes Bitcoin USD price data, calculates technical indicators,
and generates visualizations to help identify patterns and trends.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
import os
from datetime import datetime
import matplotlib.dates as mdates

# Set the style for plots
sns.set(style="darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)

def load_data(file_path):
    """
    Load and preprocess the OHLC data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Preprocessed dataframe
    """
    print(f"Loading data from {file_path}...")
    
    try:
        # Load data
        df = pd.read_csv(file_path)
        
        # Convert datetime column to datetime type
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Set datetime as index
        df.set_index('datetime', inplace=True)
        
        # Make sure numeric columns are numeric
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaN values
        df.dropna(inplace=True)
        
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def add_technical_indicators(df):
    """
    Add technical indicators to the dataframe.
    
    Args:
        df (pandas.DataFrame): OHLC dataframe
        
    Returns:
        pandas.DataFrame: Dataframe with technical indicators
    """
    print("Adding technical indicators...")
    
    # MACD
    macd = MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # RSI
    rsi = RSIIndicator(close=df['close'])
    df['rsi'] = rsi.rsi()
    
    # Bollinger Bands
    bollinger = BollingerBands(close=df['close'])
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    df['bb_mid'] = bollinger.bollinger_mavg()
    
    # Moving Averages
    df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
    df['sma_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
    df['sma_200'] = SMAIndicator(close=df['close'], window=200).sma_indicator()
    df['ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
    
    # Stochastic Oscillator
    stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # Average True Range - volatility measure
    df['atr'] = calculate_atr(df, 14)
    
    return df

def calculate_atr(df, window=14):
    """
    Calculate Average True Range (ATR).
    
    Args:
        df (pandas.DataFrame): OHLC dataframe
        window (int): Window size for ATR calculation
        
    Returns:
        pandas.Series: ATR values
    """
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    
    return atr

def plot_price_with_indicators(df, save_path=None):
    """
    Create a plot showing price and key indicators.
    
    Args:
        df (pandas.DataFrame): Dataframe with OHLC and indicators
        save_path (str, optional): Path to save the plot. If None, plot will be shown.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Plot price and moving averages on top subplot
    ax1.plot(df.index, df['close'], label='Close Price', color='black', alpha=0.75)
    ax1.plot(df.index, df['sma_20'], label='SMA 20', color='blue', alpha=0.6)
    ax1.plot(df.index, df['sma_50'], label='SMA 50', color='green', alpha=0.6)
    ax1.plot(df.index, df['sma_200'], label='SMA 200', color='red', alpha=0.6)
    ax1.plot(df.index, df['bb_high'], label='BB Upper', color='gray', linestyle='--', alpha=0.5)
    ax1.plot(df.index, df['bb_low'], label='BB Lower', color='gray', linestyle='--', alpha=0.5)
    ax1.fill_between(df.index, df['bb_high'], df['bb_low'], color='gray', alpha=0.1)
    
    # Format the x axis to show dates nicely
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    ax1.set_title('Bitcoin USD Price with Indicators')
    ax1.set_ylabel('Price (USD)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot MACD on middle subplot
    ax2.plot(df.index, df['macd'], label='MACD', color='blue', alpha=0.75)
    ax2.plot(df.index, df['macd_signal'], label='Signal', color='red', alpha=0.75)
    ax2.bar(df.index, df['macd_diff'], label='Histogram', color='green', alpha=0.5)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_ylabel('MACD')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot RSI on bottom subplot
    ax3.plot(df.index, df['rsi'], label='RSI', color='purple', alpha=0.75)
    ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5)
    ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5)
    ax3.fill_between(df.index, df['rsi'], 70, where=(df['rsi'] >= 70), color='red', alpha=0.3)
    ax3.fill_between(df.index, df['rsi'], 30, where=(df['rsi'] <= 30), color='green', alpha=0.3)
    ax3.set_ylabel('RSI')
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def generate_signals(df):
    """
    Generate buy/sell signals based on indicators.
    This is a simple example - real trading would use more sophisticated methods.
    
    Args:
        df (pandas.DataFrame): Dataframe with indicators
        
    Returns:
        pandas.DataFrame: Dataframe with signals
    """
    # Create a copy to avoid SettingWithCopyWarning
    signals_df = df.copy()
    
    # Initialize signal columns
    signals_df['signal'] = 0
    signals_df['signal_macd'] = 0
    signals_df['signal_rsi'] = 0
    signals_df['signal_bb'] = 0
    
    # MACD signal: 1 for buy, -1 for sell
    signals_df.loc[signals_df['macd'] > signals_df['macd_signal'], 'signal_macd'] = 1
    signals_df.loc[signals_df['macd'] < signals_df['macd_signal'], 'signal_macd'] = -1
    
    # RSI signals
    signals_df.loc[signals_df['rsi'] < 30, 'signal_rsi'] = 1
    signals_df.loc[signals_df['rsi'] > 70, 'signal_rsi'] = -1
    
    # Bollinger Band signals
    signals_df.loc[signals_df['close'] < signals_df['bb_low'], 'signal_bb'] = 1
    signals_df.loc[signals_df['close'] > signals_df['bb_high'], 'signal_bb'] = -1
    
    # Combine signals - simple majority rule
    signals_df['signal'] = signals_df[['signal_macd', 'signal_rsi', 'signal_bb']].sum(axis=1)
    signals_df.loc[signals_df['signal'] >= 2, 'signal'] = 1
    signals_df.loc[signals_df['signal'] <= -2, 'signal'] = -1
    signals_df.loc[(signals_df['signal'] > -2) & (signals_df['signal'] < 2), 'signal'] = 0
    
    return signals_df

def backtest_strategy(df, initial_capital=1000, position_size=0.1):
    """
    Simple backtest for the generated signals.
    
    Args:
        df (pandas.DataFrame): Dataframe with signals
        initial_capital (float): Starting capital
        position_size (float): Portion of capital to use for each trade
        
    Returns:
        pandas.DataFrame: Dataframe with backtest results
    """
    backtest_df = df.copy()
    
    # Initialize backtest columns
    backtest_df['position'] = 0
    backtest_df['capital'] = initial_capital
    backtest_df['holdings'] = 0
    backtest_df['total_value'] = initial_capital
    
    # Track position
    position = 0
    capital = initial_capital
    holdings = 0
    
    for i in range(1, len(backtest_df)):
        prev_signal = backtest_df['signal'].iloc[i-1]
        current_price = backtest_df['close'].iloc[i]
        prev_price = backtest_df['close'].iloc[i-1]
        
        # No change in position
        if prev_signal == 0 or prev_signal == position:
            position = position
        # Buy signal
        elif prev_signal == 1 and position <= 0:
            # Calculate how much to buy
            buy_amount = capital * position_size
            holdings_change = buy_amount / prev_price
            
            capital -= buy_amount
            holdings += holdings_change
            position = 1
        # Sell signal
        elif prev_signal == -1 and position >= 0:
            # Calculate how much to sell
            sell_amount = holdings * position_size * prev_price
            holdings_change = holdings * position_size
            
            capital += sell_amount
            holdings -= holdings_change
            position = -1
        
        # Update dataframe
        backtest_df['position'].iloc[i] = position
        backtest_df['capital'].iloc[i] = capital
        backtest_df['holdings'].iloc[i] = holdings
        backtest_df['total_value'].iloc[i] = capital + (holdings * current_price)
    
    # Calculate performance metrics
    initial_value = backtest_df['total_value'].iloc[0]
    final_value = backtest_df['total_value'].iloc[-1]
    returns = (final_value - initial_value) / initial_value * 100
    
    print(f"Backtest Results:")
    print(f"Initial Value: ${initial_value:.2f}")
    print(f"Final Value: ${final_value:.2f}")
    print(f"Returns: {returns:.2f}%")
    
    return backtest_df

def plot_backtest_results(df, save_path=None):
    """
    Plot backtest results.
    
    Args:
        df (pandas.DataFrame): Dataframe with backtest results
        save_path (str, optional): Path to save the plot. If None, plot will be shown.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot total value on top subplot
    ax1.plot(df.index, df['total_value'], label='Portfolio Value', color='blue')
    
    # Mark buy and sell signals
    buy_signals = df[(df['signal'] == 1) & (df['signal'].shift(1) != 1)]
    sell_signals = df[(df['signal'] == -1) & (df['signal'].shift(1) != -1)]
    
    ax1.scatter(buy_signals.index, buy_signals['close'], color='green', s=50, marker='^', label='Buy Signal')
    ax1.scatter(sell_signals.index, sell_signals['close'], color='red', s=50, marker='v', label='Sell Signal')
    
    ax1.set_title('Backtest Results')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot position on bottom subplot
    ax2.plot(df.index, df['position'], label='Position', color='purple')
    ax2.set_ylabel('Position')
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(['Short', 'Neutral', 'Long'])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    """
    Main function to run the analysis.
    """
    # Check if data file exists, use a placeholder path for now
    # In a real scenario, you would provide the actual file path
    file_path = "data/BTCUSD15m2023101T00_00.csv"
    
    # For demo purposes, we'll just print a message
    print("Note: This script is designed to work with a CSV file at: " + file_path)
    print("Since we don't have access to the file in this environment, " +
          "this script serves as a template that you can run after downloading the file.")
    
    print("\nSample Analysis Flow:")
    print("1. Load data from CSV file")
    print("2. Calculate technical indicators")
    print("3. Generate buy/sell signals")
    print("4. Backtest the strategy")
    print("5. Visualize results")
    
    print("\nTo run this script with your data:")
    print("1. Save your BTCUSD15m2023101T00_00.csv file in the 'data' directory")
    print("2. Run: python analyze_btc.py")
    
    # Create output directory if it doesn't exist
    if not os.path.exists('output'):
        os.makedirs('output')
        print("\nCreated 'output' directory for visualizations")
    
    print("\nThis script will generate analysis visualizations in the 'output' directory")
    
    # Example code for when data is available:
    """
    # Load data
    df = load_data(file_path)
    
    if df is not None:
        # Add technical indicators
        df = add_technical_indicators(df)
        
        # Plot price with indicators
        plot_price_with_indicators(df, save_path='output/btc_price_indicators.png')
        
        # Generate signals
        signals_df = generate_signals(df)
        
        # Backtest strategy
        backtest_df = backtest_strategy(signals_df)
        
        # Plot backtest results
        plot_backtest_results(backtest_df, save_path='output/btc_backtest_results.png')
        
        print("Analysis complete. Check the output directory for visualizations.")
    """

if __name__ == "__main__":
    main()
