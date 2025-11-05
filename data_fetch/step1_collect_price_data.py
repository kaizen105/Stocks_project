# Step 1: Collect Yahoo Finance Price Data
# This gets OHLCV data for 3 stocks over 15 years

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def collect_price_data():
    """
    Download historical stock price data from Yahoo Finance
    Saves to: data/raw/stock_prices_raw.csv
    """
    
    # Configuration
    TICKERS = ['AAPL', 'GOOGL', 'MSFT']  # <-- CHANGED: Kept only the first three stocks
    
    # Get dates (last 15 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=15*365) # <-- CHANGED: Set to 15 years
    
    print("=" * 60)
    print("YAHOO FINANCE DATA COLLECTION")
    print("=" * 60)
    print(f"\nTickers: {', '.join(TICKERS)}")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Expected: ~15 years of daily data\n")
    
    all_data = []
    
    # Download data for each ticker
    for ticker in TICKERS:
        print(f"Downloading {ticker}...", end=" ")
        
        try:
            # Download data
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            # Check if data was retrieved
            if df.empty:
                print(f"❌ No data retrieved")
                continue
            
            # Add ticker column
            df['Ticker'] = ticker
            
            # Reset index to make Date a column
            df.reset_index(inplace=True)
            
            # Keep only relevant columns
            df = df[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            all_data.append(df)
            print(f"✅ {len(df)} days collected")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Combine all data
    if not all_data:
        print("\n❌ No data collected! Check your internet connection.")
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by date and ticker
    combined_df = combined_df.sort_values(['Date', 'Ticker']).reset_index(drop=True)
    
    # Create directories if they don't exist (optional but good practice)
    import os
    os.makedirs('data/raw', exist_ok=True)
    
    # Save to CSV
    output_file = 'data/raw/stock_prices_raw.csv'
    combined_df.to_csv(output_file, index=False)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total records collected: {len(combined_df):,}")
    # Use .strftime('%Y-%m-%d') to avoid printing time part of the timestamp
    print(f"Date range: {combined_df['Date'].min().strftime('%Y-%m-%d')} to {combined_df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"\nRecords per stock:")
    print(combined_df.groupby('Ticker').size())
    print(f"\n✅ Data saved to: {output_file}")
    
    # Show first few rows
    print("\n" + "=" * 60)
    print("PREVIEW (First 10 rows)")
    print("=" * 60)
    print(combined_df.head(10).to_string(index=False))
    
    return combined_df


if __name__ == "__main__":
    df = collect_price_data()
    
    if df is not None:
        print("\n" + "=" * 60)
        print("✅ STEP 1 COMPLETE!")
        print("=" * 60)
    else:
        print("\n❌ Data collection failed. Please check errors above.")