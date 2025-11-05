# ==============================================================================
# --- Combine Price and Macro Datasets ---
# This script has ONLY ONE JOB:
# 1. Load the existing 'stock_prices_with_metrics.csv'.
# 2. Load the existing 'macro_data_daily.csv'.
# 3. Merge them into a single, clean file.
# IT DOES NOT DOWNLOAD ANY NEW DATA.
# ==============================================================================

import pandas as pd
import os

def combine_existing_datasets():
    """
    Loads and merges the two specified data files without re-downloading anything.
    """
    print("=" * 60)
    print("COMBINING EXISTING PRICE AND MACRO DATASETS")
    print("=" * 60)

    # --- Define File Paths ---
    price_file = 'data/processed/stock_prices_with_metrics.csv'
    macro_file = 'data/processed/macro_data_daily.csv'
    output_file = 'data/features/Stocks_dataset.csv'

    # --- 1. Load the two datasets ---
    print("\n--> STEP 1: Loading your existing files...")
    try:
        price_df = pd.read_csv(price_file, parse_dates=['Date'])
        macro_df = pd.read_csv(macro_file, parse_dates=['Date'])
        print(f"   ✅ Successfully loaded {len(price_df)} price records.")
        print(f"   ✅ Successfully loaded {len(macro_df)} macro records.")
    except FileNotFoundError as e:
        print(f"\n   ❌ CRITICAL ERROR: A required file is missing.")
        print(f"   - {e}")
        print(f"\n   Please make sure both '{price_file}' and '{macro_file}' exist before running.")
        return

    # --- 2. Merge the datasets ---
    print("\n--> STEP 2: Merging the two datasets...")
    
    # A robust 'merge_asof' is the best method. It finds the most recent macro
    # data for each trading day, which handles weekends and holidays perfectly.
    master_df = pd.merge_asof(
        price_df.sort_values('Date'),
        macro_df.sort_values('Date'),
        on='Date',
        direction='backward'  # Use the latest macro data available for a given day
    )
    print("   ✅ Datasets merged successfully.")

    # --- 3. Final Cleanup & Save ---
    print("\n--> STEP 3: Cleaning and saving the final file...")
    
    # Drop any initial rows that might have missing values from rolling calculations
    master_df.dropna(inplace=True)
    
    # Ensure the final file is sorted
    master_df = master_df.sort_values(by='Date').reset_index(drop=True)

    os.makedirs('data/features', exist_ok=True)
    master_df.to_csv(output_file, index=False)

    print(f"   ✅ Final dataset saved to: {output_file}")

    # --- 4. Summary ---
    print("\n" + "=" * 60)
    print("✅ COMBINE COMPLETE!")
    print("=" * 60)
    print(f"\n   - Final Shape: {master_df.shape[0]} rows, {master_df.shape[1]} columns")
    print(f"   - Tickers: {', '.join(master_df['Ticker'].unique())}")
    print(f"   - Date Range: {master_df['Date'].min().strftime('%Y-%m-%d')} to {master_df['Date'].max().strftime('%Y-%m-%d')}")


if __name__ == "__main__":
    combine_existing_datasets()

