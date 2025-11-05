import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import requests
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData

# =============================================================================
# ALPHA VANTAGE SETUP
# =============================================================================
ALPHA_VANTAGE_API_KEY = 'YOUR_API_KEY_HERE'  # Get free key from https://www.alphavantage.co/support/#api-key

# =============================================================================
# 1. LIVE STOCK SEARCH
# =============================================================================
def search_stocks(keywords):
    """
    Search for stocks using Alpha Vantage SYMBOL_SEARCH
    Returns: List of dicts with symbol, name, type, region
    """
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
    Fetch live market indices data
    Returns: Dict with S&P500, NASDAQ, DOW data
    """
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
# 3. COMPANY OVERVIEW (FUNDAMENTALS)
# =============================================================================
def get_company_overview(ticker):
    """
    Fetch company fundamentals using Alpha Vantage
    """
    try:
        fd = FundamentalData(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        data, _ = fd.get_company_overview(ticker)
        
        if not data.empty:
            return {
                'Name': data.get('Name', ['N/A'])[0],
                'Sector': data.get('Sector', ['N/A'])[0],
                'Industry': data.get('Industry', ['N/A'])[0],
                'Market Cap': data.get('MarketCapitalization', ['N/A'])[0],
                'PE Ratio': data.get('PERatio', ['N/A'])[0],
                '52 Week High': data.get('52WeekHigh', ['N/A'])[0],
                '52 Week Low': data.get('52WeekLow', ['N/A'])[0],
                'Dividend Yield': data.get('DividendYield', ['N/A'])[0]
            }
        return {}
    except Exception as e:
        print(f"Company overview error: {e}")
        return {}

# =============================================================================
# 4. DATA FETCHING (Enhanced with Alpha Vantage fallback)
# =============================================================================
def fetch_stock_data(ticker, days_needed=252, use_alpha_vantage=False):
    """
    Fetches stock data with Alpha Vantage as fallback
    """
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=int(days_needed * 1.5))).strftime('%Y-%m-%d')
   
    try:
        if use_alpha_vantage:
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
        vix_data = yf.download('^VIX', period='5d', progress=False)
        if not vix_data.empty:
            macro_defaults['VIX'] = vix_data['Close'].iloc[-1]
        
        tnx = yf.download('^TNX', period='5d', progress=False)['Close'].iloc[-1]
        irx = yf.download('^IRX', period='5d', progress=False)['Close'].iloc[-1] / 10
        macro_defaults['Yield_Curve_10Y_2Y'] = tnx - irx
        
        return macro_defaults
    except Exception as e:
        return macro_defaults

# =============================================================================
# 7. FINAL INPUT PREPARATION
# =============================================================================
def prepare_model_input(df_features, macro_inputs):
    """Prepares final model input"""
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
   
    xgb_input = latest_data[xgb_features]
   
    return hmm_input, xgb_input