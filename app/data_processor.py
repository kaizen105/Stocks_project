import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import requests
import json  # <-- 1. ADDED: For loading the cache
import time  # <-- 2. ADDED: For throttling *live* search/fallback calls
import os  # <-- 3. ADDED: For environment variables
from dotenv import load_dotenv  # <-- 4. ADDED: To load the .env file

# --- Import only what's needed ---
from alpha_vantage.timeseries import TimeSeries
# FundamentalData is no longer called from this file, so we can remove it
# from alpha_vantage.fundamentaldata import FundamentalData 

# Load environment variables from .env file (must be at the top)
load_dotenv()

# =============================================================================
# ALPHA VANTAGE SETUP
# =============================================================================
# 5. CHANGED: Loads key securely from the system environment or your .env file.
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'PLACEHOLDER_KEY')

# =============================================================================
# 6. NEW: LOAD COMPANY DATA CACHE
# =============================================================================
# This loads your JSON file into a variable when the app starts.
COMPANY_CACHE = {}
CACHE_FILENAME = 'company_data_cache.json'
try:
    with open(CACHE_FILENAME, 'r') as f:
        COMPANY_CACHE = json.load(f)
    print(f"Successfully loaded {len(COMPANY_CACHE)} tickers from {CACHE_FILENAME}")
except FileNotFoundError:
    print(f"WARNING: {CACHE_FILENAME} not found. Company data will be empty.")
    print("Run 'create_dataset.py' to generate this file.")
except Exception as e:
    print(f"Error loading {CACHE_FILENAME}: {e}. Company data will be empty.")

# =============================================================================
# 1. LIVE STOCK SEARCH (Still needs throttling)
# =============================================================================
def search_stocks(keywords):
    """
    Search for stocks using Alpha Vantage SYMBOL_SEARCH
    Returns: List of dicts with symbol, name, type, region
    """
    # --- 7. ADDED: This is still a live API call, so it needs the 15s delay ---
    # This prevents the "Too Many Requests" error on your home page search.
    time.sleep(15)

    try:
        url = f'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={keywords}&apikey={ALPHA_VANTAGE_API_KEY}'
        response = requests.get(url)
        data = response.json()

        if 'bestMatches' in data:
            results = []
            for match in data['bestMatches'][:10]:  # Top 10 results
                results.append({
                    'symbol': match['1. symbol'],
                    'name': match['2. name'],
                    'type': match['3. type'],
                    'region': match['4. region'],
                    'currency': match.get('8. currency', 'USD')
                })
            return results
        return []
    except Exception as e:
        print(f"Search error: {e}")
        return []

# =============================================================================
# 2. LIVE MARKET DATA (OVERVIEW)
# =============================================================================
def get_market_overview():
    """
    Fetch live market indices data (Uses yfinance, no throttling needed)
    Returns: Dict with S&P500, NASDAQ, DOW data
    """
    # ... (This function is unchanged) ...
    try:
        indices = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ',
            'DIA': 'Dow Jones'
        }

        market_data = {}
        for symbol, name in indices.items():
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period='2d')

            if len(hist) >= 2:
                current = hist['Close'].iloc[-1]
                previous = hist['Close'].iloc[-2]
                change = current - previous
                change_pct = (change / previous) * 100

                market_data[name] = {
                    'price': current,
                    'change': change,
                    'change_pct': change_pct,
                    'volume': hist['Volume'].iloc[-1]
                }

        return market_data
    except Exception as e:
        print(f"Market overview error: {e}")
        return {}

# =============================================================================
# 3. COMPANY OVERVIEW (FUNDAMENTALS) - *** THE BIG CHANGE ***
# =============================================================================
def get_company_overview(ticker):
    """
    Fetch company fundamentals from the LOCAL CACHE.
    This is now instantaneous and requires no API call.
    """
    # --- 8. MODIFIED: All API logic is replaced with a simple cache lookup ---
    try:
        # .get() is safe. It returns the data if 'ticker' is a key,
        # or the default value ({}) if it's not.
        return COMPANY_CACHE.get(ticker.upper(), {})
    except Exception as e:
        print(f"Cache lookup error for {ticker}: {e}")
        return {}

# =============================================================================
# 4. DATA FETCHING (Fallback still needs throttling)
# =============================================================================
def fetch_stock_data(ticker, days_needed=252, use_alpha_vantage=False):
    """
    Fetches stock data with Alpha Vantage as fallback
    """
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=int(days_needed * 1.5))).strftime('%Y-%m-%d')

    try:
        if use_alpha_vantage:
            # --- 9. ADDED: This is also a live API call, so it needs the 15s delay ---
            time.sleep(15)

            # Use Alpha Vantage for data
            ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
            data, _ = ts.get_daily(symbol=ticker, outputsize='full')
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            data.index = pd.to_datetime(data.index)
            data = data.sort_index()
            data = data[data.index >= start_date]
        else:
            # Use yfinance (default)
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if data.empty:
                return pd.DataFrame()

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)

            if 'Adj Close' in data.columns:
                if 'Close' in data.columns:
                    data = data.drop(columns=['Close'])
                data = data.rename(columns={'Adj Close': 'Close'})

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required_cols if col not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        data['Ticker'] = ticker
        return data[required_cols + ['Ticker']]

    except Exception as e:
        print(f"Data fetch error: {e}")
        return pd.DataFrame()

# =============================================================================
# 5. FEATURE ENGINEERING (Same as before)
# =============================================================================
def calculate_technical_features(df):
    """Calculates technical features"""
    # ... (This function is unchanged) ...
    if df.empty:
        return df

    df['Return_Pct'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Realized_Vol_10d'] = df['Log_Return'].rolling(window=10).std() * np.sqrt(252)
    df['Realized_Vol_20d'] = df['Log_Return'].rolling(window=20).std() * np.sqrt(252)

    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()

    df['Price_Range'] = df['High'] - df['Low']
    price_range_pct_calc = df['Price_Range'].div(df['Close'].shift(1))
    if isinstance(price_range_pct_calc, pd.DataFrame) and price_range_pct_calc.shape[1] == 1:
        price_range_pct_calc = price_range_pct_calc.squeeze()
    df['Price_Range_Pct'] = price_range_pct_calc.replace([np.inf, -np.inf], 0).astype(float)

    df['Volume_Ratio'] = df['Volume'].div(df['Volume'].shift(1)).replace([np.inf, -np.inf], 0).astype(float)
    df['Momentum_5d'] = df['Close'].pct_change(periods=5).astype(float)

    df['Momentum_20d'] = df['Close'].pct_change(periods=20).astype(float)
    df['Momentum_60d'] = df['Close'].pct_change(periods=60).astype(float)

    df['Momentum_Ratio_S_M'] = df['Momentum_5d'].div(df['Momentum_20d']).replace([np.inf, -np.inf], 0).astype(float)
    df['Momentum_Ratio_M_L'] = df['Momentum_20d'].div(df['Momentum_60d']).replace([np.inf, -np.inf], 0).astype(float)

    df['Volatility_Ratio'] = df['Realized_Vol_10d'].div(df['Realized_Vol_20d']).replace([np.inf, -np.inf], 0).astype(float)

    for i in range(1, 4):
        df[f'Return_Pct_lag_{i}'] = df['Return_Pct'].shift(i).astype(float)

    std_20 = df['Close'].rolling(window=20).std()
    zscore_calc = (df['Close'] - df['MA_20']).div(std_20)
    if isinstance(zscore_calc, pd.DataFrame) and zscore_calc.shape[1] == 1:
        zscore_calc = zscore_calc.squeeze()
    df['Price_ZScore'] = zscore_calc.replace([np.inf, -np.inf], 0).astype(float)
    # --- Additional aliases and features expected by the Dash app ---
    # Create MA aliases without underscores (app expects 'MA50', 'MA200', 'MA20')
    df['MA50'] = df['MA_50']
    df['MA200'] = df['MA_200']
    df['MA20'] = df['MA_20']

    # 20-day rolling std of log returns (daily volatility, non-annualized)
    df['rolling_std_20'] = df['Log_Return'].rolling(window=20).std().astype(float)

    # RSI (14-day) - typical implementation
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1.0 + rs))
    df['RSI'] = df['RSI'].fillna(50).astype(float)

    # Keep a datetime index for plotting in the Dash app and drop NA rows
    return df.dropna()

# =============================================================================
# 6. MACRO DATA FETCHING
# =============================================================================
def fetch_macro_data():
    """Fetches latest macro data"""
    macro_defaults = {
        'Fed_Funds_Rate': 3.90,
        'CPI': 3.0,
        'Unemployment_Rate': 4.35,
        'GDP': 4.0,
        'Yield_Curve_10Y_2Y': 0.52,
        'VIX': 16.37
    }

    try:
        # Fetch VIX data
        vix_data = yf.download('^VIX', period='5d', progress=False)
        if not vix_data.empty:
            vix_value = vix_data['Close'].iloc[-1]
            if isinstance(vix_value, (int, float, np.number)):
                macro_defaults['VIX'] = float(vix_value)

        # Fetch yield curve data
        tnx_data = yf.download('^TNX', period='5d', progress=False)
        irx_data = yf.download('^IRX', period='5d', progress=False)
        
        if not tnx_data.empty and not irx_data.empty:
            tnx_value = tnx_data['Close'].iloc[-1]
            irx_value = irx_data['Close'].iloc[-1]
            
            if isinstance(tnx_value, (int, float, np.number)) and isinstance(irx_value, (int, float, np.number)):
                macro_defaults['Yield_Curve_10Y_2Y'] = float(tnx_value) - float(irx_value/10)

    except Exception as e:
        print(f"Macro data fetch error: {e}")
    
    # Ensure all values are plain floats
    return {k: float(v) for k, v in macro_defaults.items()}

# =============================================================================
# 7. FINAL INPUT PREPARATION
# =============================================================================
def prepare_model_input(df_features, macro_inputs):
    """Prepares final model input"""
    # ... (This function is unchanged) ...
    latest_data = df_features.tail(1).copy()

    if latest_data.empty:
        raise ValueError("Insufficient data for features")

    for feature, value in macro_inputs.items():
        latest_data[feature] = float(value)

    hmm_input_features = ['Return_Pct', 'Realized_Vol_10d']
    hmm_input = latest_data[hmm_input_features].values

    xgb_features = [
        'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'Return_Pct',
        'Price_Range', 'Price_Range_Pct', 'Realized_Vol_10d', 'Realized_Vol_20d',
        'MA_20', 'MA_50', 'MA_200', 'Volume_MA_20', 'Volume_Ratio', 'Momentum_5d',
        'Fed_Funds_Rate', 'CPI', 'Unemployment_Rate', 'GDP', 'Yield_Curve_10Y_2Y',
        'VIX', 'Return_Pct_lag_1', 'Return_Pct_lag_2', 'Return_Pct_lag_3',
        'Momentum_Ratio_S_M', 'Momentum_Ratio_M_L', 'Volatility_Ratio',
        'Price_ZScore'
    ]
    
    # Ensure all required columns exist before filtering
    for col in xgb_features:
        if col not in latest_data.columns:
            if col == 'Ticker':
                 latest_data[col] = df_features['Ticker'].iloc[-1] if 'Ticker' in df_features.columns else 'N/A'
            else:
                 latest_data[col] = np.nan # Or 0, depending on model training

    xgb_input = latest_data[xgb_features]

    return hmm_input, xgb_input