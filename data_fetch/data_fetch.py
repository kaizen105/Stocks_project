import json
import time
from alpha_vantage.fundamentaldata import FundamentalData
import os
from dotenv import load_dotenv

# Load the API key from your .env file
load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

# --- List of stocks you want to pre-fetch ---
# You can add as many as you want here. The script will only fetch 25 per day.
# This is a sample of the 50 largest S&P 500 companies
TICKERS_TO_FETCH = [
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B', 
    'LLY', 'V', 'JPM', 'XOM', 'WMT', 'JNJ', 'UNH', 'MA', 'PG', 'COST', 'HD', 
    'ORCL', 'MRK', 'CVX', 'AVGO', 'CRM', 'KO', 'PEP', 'ADBE', 'BAC', 'NFLX',
    'MCD', 'TMO', 'CSCO', 'DIS', 'LIN', 'ABT', 'ACN', 'WFC', 'PFE', 'INTC',
    'AMD', 'SBUX', 'QCOM', 'CAT', 'GILD', 'AMGN', 'GM', 'F', 'UBER'
]

# --- API Limits ---
API_CALLS_PER_DAY = 25
WAIT_TIME_SECONDS = 15 # Wait 15s to stay under 5 calls/min limit

def create_company_dataset():
    """
    Fetches company overview data for the tickers list and saves it to a JSON file,
    respecting the 25-call daily limit.
    """
    print("--- Starting dataset creation... ---")
    
    # Initialize the fundamental data fetcher
    fd = FundamentalData(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    
    company_data_cache = {}
    
    # Try to load existing data so we can add to it
    output_filename = 'company_data_cache.json'
    if os.path.exists(output_filename):
        try:
            with open(output_filename, 'r') as f:
                company_data_cache = json.load(f)
                print(f"Loaded {len(company_data_cache)} tickers from existing cache.")
        except Exception as e:
            print(f"Could not read cache file, starting fresh. Error: {e}")
            company_data_cache = {}

    call_count = 0
    for ticker in TICKERS_TO_FETCH:
        # Check if we've already fetched this ticker
        if ticker in company_data_cache:
            print(f"Skipping {ticker}, already in cache.")
            continue

        # Check if we've hit the daily API limit
        if call_count >= API_CALLS_PER_DAY:
            print(f"\n--- Hit daily limit of {API_CALLS_PER_DAY} calls. ---")
            print("Run this script again tomorrow to fetch more data.")
            break # Stop the loop

        print(f"Fetching {ticker} (Call {call_count + 1} of {API_CALLS_PER_DAY})...")
        try:
            # Call the API
            data, _ = fd.get_company_overview(ticker)
            call_count += 1 # Increment *after* the call
            
            if not data.empty:
                # Convert the pandas data to a simple dictionary
                overview_dict = {
                    'Name': data.get('Name', ['N/A'])[0],
                    'Sector': data.get('Sector', ['N/A'])[0],
                    'Industry': data.get('Industry', ['N/A'])[0],
                    'Market Cap': data.get('MarketCapitalization', ['N/A'])[0],
                    'PE Ratio': data.get('PERatio', ['N/A'])[0],
                    '52 Week High': data.get('52WeekHigh', ['N/A'])[0],
                    '52 Week Low': data.get('52WeekLow', ['N/A'])[0],
                    'Dividend Yield': data.get('DividendYield', ['N/A'])[0]
                }
                company_data_cache[ticker] = overview_dict
                print(f"Successfully fetched {ticker}.")
            else:
                print(f"No data returned for {ticker}.")
                company_data_cache[ticker] = {} # Save empty dict to avoid re-fetching

        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            # Check for rate limit error specifically
            if "Our standard API call frequency is" in str(e):
                print("Hit rate limit. Exiting. Run again later.")
                break # Stop the script
            company_data_cache[ticker] = {} # Save empty on error
        
        # --- This is the rate limit fix ---
        if call_count < API_CALLS_PER_DAY: # Don't wait after the last call
            print(f"Waiting {WAIT_TIME_SECONDS} seconds for next API call...")
            time.sleep(WAIT_TIME_SECONDS) 

    # Save the final dictionary to a JSON file
    with open(output_filename, 'w') as f:
        json.dump(company_data_cache, f, indent=4)
        
    print(f"\n--- All done! {len(company_data_cache)} total tickers saved to {output_filename} ---")

# --- Run the function ---
if __name__ == "__main__":
    create_company_dataset()