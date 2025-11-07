import yfinance as yf
import pandas as pd
import numpy as np
import requests
from fredapi import Fred
from datetime import datetime, timedelta
import time
import os

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
ALL_TICKERS = ['AAPL', 'GOOGL', 'MSFT'] # Using the 3 tickers as requested
YEARS_OF_DATA = 15

# ==============================================================================
# --- STEP 1: YAHOO FINANCE PRICE DATA ---
# ==============================================================================
def collect_price_data():
    """
    Download historical stock price data from Yahoo Finance.
    Saves to: data/raw/stock_prices_raw.csv
    """
    print("=" * 60)
    print("STEP 1: COLLECTING YAHOO FINANCE PRICE DATA")
    print("=" * 60)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=YEARS_OF_DATA * 365)
    
    print(f"\nTickers: {', '.join(ALL_TICKERS)}")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n")
    
    all_data = []
    for ticker in ALL_TICKERS:
        print(f"Downloading {ticker}...", end=" ")
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty:
                print(f"❌ No data retrieved")
                continue
            df['Ticker'] = ticker
            all_data.append(df)
            print(f"✅ {len(df)} days collected")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    if not all_data:
        print("\n❌ No price data collected! Check tickers or internet connection.")
        return None
        
    combined_df = pd.concat(all_data)
    combined_df.reset_index(inplace=True)
    combined_df = combined_df[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    os.makedirs('data/raw', exist_ok=True)
    output_file = 'data/raw/stock_prices_raw.csv'
    combined_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Price data for {len(ALL_TICKERS)} stocks saved to: {output_file}")
    return combined_df

# ==============================================================================
# --- STEP 2: CALCULATE TECHNICAL METRICS (WITH FIX) ---
# ==============================================================================
def add_metrics():
    """
    Load raw price data and add calculated metrics.
    Saves to: data/processed/stock_prices_with_metrics.csv
    """
    print("\n" + "=" * 60)
    print("STEP 2: CALCULATING STOCK METRICS")
    print("=" * 60)
    
    try:
        # FIX #1: Use low_memory=False to handle the DtypeWarning from the start.
        df = pd.read_csv('data/raw/stock_prices_raw.csv', low_memory=False)
        df['Date'] = pd.to_datetime(df['Date'])
    except FileNotFoundError:
        print("\n❌ Raw price data not found. Please run Step 1 first.")
        return None

    # FIX #2: Force price/volume columns to numeric, coercing errors to NaN.
    # This is the most important fix for the TypeError.
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows where the Close price is now NaN after coercion
    df.dropna(subset=['Close'], inplace=True)
        
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    # FIX #3: Added fill_method=None to address the FutureWarning.
    df['Return_Pct'] = df.groupby('Ticker')['Close'].pct_change(fill_method=None) * 100
    
    # Rest of the calculations
    df['Price_Range'] = df['High'] - df['Low']
    df['Realized_Vol_10d'] = df.groupby('Ticker')['Return_Pct'].transform(lambda x: x.rolling(10).std())
    df['MA_50'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(50).mean())
    df['MA_200'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(200).mean())
    
    # Target variables for ML
    df['Target_Price_Next'] = df.groupby('Ticker')['Close'].shift(-1)
    df['Target_Return_Next'] = df.groupby('Ticker')['Return_Pct'].shift(-1)
    
    os.makedirs('data/processed', exist_ok=True)
    output_file = 'data/processed/stock_prices_with_metrics.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Data with technical metrics saved to: {output_file}")
    return df

# ==============================================================================
# --- STEP 3: FUNDAMENTALS DATA (ALPHA VANTAGE) ---
# (No changes needed here)
# ==============================================================================
def get_quarterly_income_statement(ticker):
    url = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}'
    try:
        data = requests.get(url).json()
        if 'quarterlyReports' not in data: return None
        quarterly = [{'Date': r.get('fiscalDateEnding'), 'Ticker': ticker, 'Revenue': r.get('totalRevenue'), 'Net_Income': r.get('netIncome')} for r in data['quarterlyReports'][:60]]
        return pd.DataFrame(quarterly)
    except Exception: return None

def get_quarterly_balance_sheet(ticker):
    url = f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}'
    try:
        data = requests.get(url).json()
        if 'quarterlyReports' not in data: return None
        quarterly = [{'Date': r.get('fiscalDateEnding'), 'Ticker': ticker, 'Total_Assets': r.get('totalAssets'), 'Total_Debt': float(r.get('shortTermDebt',0) or 0) + float(r.get('longTermDebt',0) or 0)} for r in data['quarterlyReports'][:60]]
        return pd.DataFrame(quarterly)
    except Exception: return None

def get_quarterly_earnings(ticker):
    url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}'
    try:
        data = requests.get(url).json()
        if 'quarterlyEarnings' not in data: return None
        quarterly = [{'Date': q.get('fiscalDateEnding'), 'Ticker': ticker, 'Reported_EPS': q.get('reportedEPS')} for q in data['quarterlyEarnings']]
        return pd.DataFrame(quarterly)
    except Exception: return None

def collect_fundamentals():
    print("\n" + "=" * 60)
    print("STEP 3: COLLECTING FUNDAMENTALS DATA (APPEND MODE)")
    print("=" * 60)
    output_file = 'data/raw/fundamentals_quarterly.csv'
    existing_df = pd.DataFrame()
    processed_tickers = []
    if os.path.exists(output_file):
        print(f"Loading existing data from {output_file}...")
        existing_df = pd.read_csv(output_file)
        processed_tickers = existing_df['Ticker'].unique().tolist()
        print(f"✅ Found {len(processed_tickers)} previously processed tickers.")
    tickers_to_process = [t for t in ALL_TICKERS if t not in processed_tickers]
    if not tickers_to_process:
        print("\nAll tickers have already been processed for fundamentals. Skipping.")
        return existing_df
    tickers_this_run = tickers_to_process[:6]
    print(f"\nWill process up to 6 new tickers. This run: {', '.join(tickers_this_run)}\n")
    all_fundamentals = []
    for i, ticker in enumerate(tickers_this_run, 1):
        print(f"[{i}/{len(tickers_this_run)}] Processing {ticker}...")
        time.sleep(12)
        earnings_df = get_quarterly_earnings(ticker)
        time.sleep(12)
        income_df = get_quarterly_income_statement(ticker)
        time.sleep(12)
        balance_df = get_quarterly_balance_sheet(ticker)
        if earnings_df is not None and income_df is not None and balance_df is not None:
            merged = earnings_df.merge(income_df, on=['Date', 'Ticker'], how='outer').merge(balance_df, on=['Date', 'Ticker'], how='outer')
            all_fundamentals.append(merged)
            print(f"    ✅ Data collected for {ticker}")
        else:
            print(f"    ❌ Failed to collect complete data for {ticker}")
    if not all_fundamentals:
        print("\n❌ No new fundamental data was collected in this run.")
        return existing_df
    new_df = pd.concat(all_fundamentals, ignore_index=True)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    combined_df = combined_df.sort_values(['Date', 'Ticker']).reset_index(drop=True)
    combined_df.to_csv(output_file, index=False)
    print(f"\n✅ Fundamentals data updated in: {output_file}")
    print(f"Total unique tickers in file: {len(combined_df['Ticker'].unique())}")
    return combined_df

# ==============================================================================
# --- STEP 4: MACROECONOMIC DATA (FRED) ---
# (No changes needed here)
# ==============================================================================
def get_macro_data():
    print("\n" + "=" * 60)
    print("STEP 4: COLLECTING MACROECONOMIC DATA (FRED)")
    print("=" * 60)
    fred = Fred(api_key=FRED_API_KEY)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=YEARS_OF_DATA * 365)
    print(f"\nPeriod: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n")
    indicators = {'DFF': 'Fed_Funds_Rate', 'CPIAUCSL': 'CPI', 'UNRATE': 'Unemployment_Rate', 'GDP': 'GDP', 'T10Y2Y': 'Yield_Curve_10Y_2Y', 'VIXCLS': 'VIX'}
    all_data = {}
    for series_id, name in indicators.items():
        print(f"Fetching {name}...", end=" ")
        try:
            data = fred.get_series(series_id, start_date, end_date)
            data.dropna(inplace=True)
            all_data[name] = data
            print(f"✅")
        except Exception as e:
            print(f"❌ Error: {e}")
    df = pd.DataFrame(all_data).resample('D').ffill().reset_index()
    df.rename(columns={'index': 'Date'}, inplace=True)
    df = df.fillna(method='ffill').fillna(method='bfill')
    output_file = 'data/raw/macro_data_raw.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ Macroeconomic data saved to: {output_file}")
    return df

# ==============================================================================
# --- MAIN EXECUTION ---
# ==============================================================================
if __name__ == "__main__":
    print("STARTING DATA COLLECTION PIPELINE\n")
    collect_price_data()
    add_metrics()
    collect_fundamentals()
    get_macro_data()
    print("\n" + "=" * 60)
    print("✅ DATA COLLECTION PIPELINE COMPLETE!")
    print("=" * 60)