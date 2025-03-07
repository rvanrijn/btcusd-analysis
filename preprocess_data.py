#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bitcoin USD Data Preprocessing Script

This script preprocesses the raw BTCUSD data, fixing common issues and
preparing it for analysis.
"""

import pandas as pd
import os
import argparse
from datetime import datetime

def preprocess_btc_data(input_file, output_file=None):
    """
    Preprocess Bitcoin price data.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str, optional): Path to save the preprocessed data.
                                   If None, will use input_file with '_preprocessed' suffix.
    
    Returns:
        pandas.DataFrame: Preprocessed dataframe
    """
    print(f"Preprocessing data from {input_file}...")
    
    try:
        # Load the data
        df = pd.read_csv(input_file)
        
        # Print original data info
        print("\nOriginal data info:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Sample data:\n{df.head()}")
        
        # Handle column names (in case they have whitespace)
        df.columns = [col.strip() for col in df.columns]
        
        # Check if datetime column exists
        if 'datetime' not in df.columns:
            # Try to find a date/time column
            date_cols = [col for col in df.columns if any(x in col.lower() for x in ['date', 'time'])]
            if date_cols:
                print(f"Renaming column '{date_cols[0]}' to 'datetime'")
                df.rename(columns={date_cols[0]: 'datetime'}, inplace=True)
        
        # Convert datetime to proper format
        if 'datetime' in df.columns:
            try:
                df['datetime'] = pd.to_datetime(df['datetime'])
            except:
                print("Warning: Could not convert datetime column. Trying different formats...")
                try:
                    # Try different formats
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', 
                                '%d-%m-%Y %H:%M:%S', '%d/%m/%Y %H:%M:%S',
                                '%Y-%m-%dT%H:%M:%S', '%Y%m%d%H%M%S']:
                        try:
                            df['datetime'] = pd.to_datetime(df['datetime'], format=fmt)
                            print(f"Successful format: {fmt}")
                            break
                        except:
                            continue
                except:
                    print("Warning: Could not parse datetime column. Keeping as is.")
        
        # Convert price columns to numeric
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert volume to numeric
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Remove rows with NaN values
        original_rows = len(df)
        df.dropna(inplace=True)
        print(f"Removed {original_rows - len(df)} rows with NaN values")
        
        # Sort by datetime
        if 'datetime' in df.columns:
            df.sort_values('datetime', inplace=True)
            
        # Check for duplicates
        duplicates = df.duplicated(subset=['datetime'], keep='first')
        if duplicates.sum() > 0:
            print(f"Found {duplicates.sum()} duplicate timestamps. Keeping first occurrence.")
            df = df[~duplicates]
        
        # Save preprocessed data
        if output_file is None:
            base, ext = os.path.splitext(input_file)
            output_file = f"{base}_preprocessed{ext}"
        
        df.to_csv(output_file, index=False)
        print(f"Preprocessed data saved to {output_file}")
        
        # Print preprocessed data info
        print("\nPreprocessed data info:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Sample data:\n{df.head()}")
        print(f"Data range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        return df
    
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return None

def main():
    """
    Main function to run preprocessing with command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Preprocess Bitcoin price data')
    parser.add_argument('input_file', help='Path to the input CSV file')
    parser.add_argument('--output_file', '-o', help='Path to save the preprocessed data')
    
    args = parser.parse_args()
    
    # Run preprocessing
    preprocess_btc_data(args.input_file, args.output_file)

if __name__ == "__main__":
    # When running as script, print usage example
    if not os.isatty(0):  # If piped or redirected
        main()
    else:
        print("Bitcoin USD Data Preprocessing Script")
        print("\nUSAGE:")
        print("  python preprocess_data.py INPUT_FILE [--output_file OUTPUT_FILE]")
        print("\nExample:")
        print("  python preprocess_data.py data/BTCUSD15m2023101T00_00.csv")
