# Step 2: Add Calculated Metrics to Price Data
# Calculates: Returns, Volatility, Moving Averages, Volume metrics

import pandas as pd
import numpy as np

def add_metrics():
    """
    Load raw price data and add calculated metrics
    Saves to: data/processed/stock_prices_with_metrics.csv
    """
    
    print("=" * 60)
    print("CALCULATING STOCK METRICS")
    print("=" * 60)
    
    # Load raw data
    print("\nLoading raw price data...", end=" ")
    df = pd.read_csv('data/raw/stock_prices_raw.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"✅ {len(df)} records loaded")
    
    # Sort by ticker and date
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    print("\nCalculating metrics...\n")
    
    # 1. DAILY RETURNS
    print("  → Daily Returns (% change)...", end=" ")
    df['Return_Pct'] = df.groupby('Ticker')['Close'].pct_change() * 100
    print("✅")
    
    # 2. PRICE RANGE (High-Low spread)
    print("  → Daily Price Range...", end=" ")
    df['Price_Range'] = df['High'] - df['Low']
    df['Price_Range_Pct'] = (df['Price_Range'] / df['Close']) * 100
    print("✅")
    
    # 3. VOLATILITY (Rolling Standard Deviation)
    print("  → 10-day Rolling Volatility...", end=" ")
    df['Realized_Vol_10d'] = df.groupby('Ticker')['Return_Pct'].transform(
        lambda x: x.rolling(10, min_periods=5).std()
    )
    print("✅")
    
    print("  → 20-day Rolling Volatility...", end=" ")
    df['Realized_Vol_20d'] = df.groupby('Ticker')['Return_Pct'].transform(
        lambda x: x.rolling(20, min_periods=10).std()
    )
    print("✅")
    
    # 4. MOVING AVERAGES
    print("  → Moving Averages (20, 50, 200 day)...", end=" ")
    df['MA_20'] = df.groupby('Ticker')['Close'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    df['MA_50'] = df.groupby('Ticker')['Close'].transform(
        lambda x: x.rolling(50, min_periods=25).mean()
    )
    df['MA_200'] = df.groupby('Ticker')['Close'].transform(
        lambda x: x.rolling(200, min_periods=100).mean()
    )
    print("✅")
    
    # 5. VOLUME METRICS
    print("  → Volume Metrics...", end=" ")
    df['Volume_MA_20'] = df.groupby('Ticker')['Volume'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
    print("✅")
    
    # 6. MOMENTUM INDICATORS
    print("  → Momentum (5-day change)...", end=" ")
    df['Momentum_5d'] = df.groupby('Ticker')['Close'].pct_change(5) * 100
    print("✅")
    
    # 7. TARGET VARIABLES (for ML later)
    print("  → Creating Target Variables...", end=" ")
    # Next day price (for price prediction)
    df['Target_Price_Next'] = df.groupby('Ticker')['Close'].shift(-1)
    # Next day return (for return prediction)
    df['Target_Return_Next'] = df.groupby('Ticker')['Return_Pct'].shift(-1)
    # Next day volatility (for volatility prediction)
    df['Target_Vol_Next'] = df.groupby('Ticker')['Realized_Vol_10d'].shift(-1)
    print("✅")
    
    # Save processed data
    output_file = 'data/processed/stock_prices_with_metrics.csv'
    df.to_csv(output_file, index=False)
    
    # Summary
    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)
    print(f"Total records: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"\nNew columns added:")
    new_cols = ['Return_Pct', 'Price_Range', 'Price_Range_Pct', 
                'Realized_Vol_10d', 'Realized_Vol_20d',
                'MA_20', 'MA_50', 'MA_200', 
                'Volume_MA_20', 'Volume_Ratio', 'Momentum_5d',
                'Target_Price_Next', 'Target_Return_Next', 'Target_Vol_Next']
    for col in new_cols:
        print(f"  • {col}")
    
    print(f"\n✅ Data saved to: {output_file}")
    
    # Data quality check
    print("\n" + "=" * 60)
    print("DATA QUALITY CHECK")
    print("=" * 60)
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    print("\nMissing values (%):")
    print(missing_pct[missing_pct > 0].to_string())
    
    # Preview by ticker
    print("\n" + "=" * 60)
    print("PREVIEW (Latest 5 records per stock)")
    print("=" * 60)
    preview_cols = ['Date', 'Ticker', 'Close', 'Return_Pct', 'Realized_Vol_20d', 'MA_50']
    for ticker in df['Ticker'].unique():
        print(f"\n{ticker}:")
        print(df[df['Ticker']==ticker][preview_cols].tail(5).to_string(index=False))
    
    return df


if __name__ == "__main__":
    df = add_metrics()
    
    print("\n" + "=" * 60)
    print("✅ STEP 2 COMPLETE!")
    print("=" * 60)
    print("\nYou now have:")
    print("  • Raw price data: data/raw/stock_prices_raw.csv")
    print("  • Processed data with metrics: data/processed/stock_prices_with_metrics.csv")