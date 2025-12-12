# Stock Analytics - End-to-End Project

A comprehensive stock market analytics project covering data acquisition, exploratory analysis, SQL querying, business intelligence dashboards, time series forecasting, and an interactive web application.

## Project Overview

This project performs multi-layered analysis on stock market data (AAPL, MSFT, GOOGL) combining traditional analytics with machine learning approaches to forecast returns, volatility, and market regimes. Data is fetched programmatically using yfinance and Alpha Vantage APIs.

## Data Acquisition

**APIs Used:**
- **yfinance**: Historical stock prices, volume, and technical indicators
- **Alpha Vantage**: Fundamental data (EPS, PE Ratio, Revenue, Debt-to-Equity) and macroeconomic indicators (Fed Funds Rate, CPI, Unemployment, GDP)

## Dataset

**Initial Analysis (3 Years)**
- **Tickers**: AAPL, GOOGL, MSFT
- **Period**: September 2022 - September 2025
- **Features**: Date, Ticker, Close, Volume, Return_Pct, Realized_Vol_20d, EPS, PE_Ratio, Debt_to_Equity, Revenue, Fed_Funds_Rate, CPI, VIX, GoogleTrends, Target_Price, Target_Vol

**ML Analysis (15 Years)**
- **Tickers**: AAPL, GOOGL, MSFT
- **Period**: 2010 - 2025
- **Features**: 30+ technical, fundamental, and macroeconomic indicators including:
  - Price metrics (Open, High, Low, Close, Volume)
  - Technical indicators (MA_20, MA_50, MA_200, Momentum)
  - Volatility measures (Realized_Vol_10d, Realized_Vol_20d)
  - Macroeconomic data (Fed_Funds_Rate, CPI, Unemployment_Rate, GDP, Yield_Curve)
  - Market sentiment (VIX, Volume_Ratio)
  - Derived features (Price_ZScore, Momentum_Ratios, Volatility_Ratio)

## Project Pipeline

### 1. Data Collection
- Automated data fetching using yfinance and Alpha Vantage APIs
- Data cleaning and preprocessing
- Feature engineering

### 2. Excel Analysis
- Descriptive statistics
- Trend visualization
- Correlation matrix analysis

### 3. SQL Analysis
- 5 custom queries for data exploration
- Stock performance metrics
- Comparative analysis across tickers

### 4. Power BI Dashboard
- 2-page interactive dashboard
- Visual insights on stock trends
- Performance metrics visualization

### 5. Machine Learning & Forecasting

**Time Series Analysis (TSA)**
- Stationarity tests
- Seasonality decomposition
- Autocorrelation analysis

**Forecasting Models**
- **Returns Prediction**: Multi-step ahead return forecasts
- **Realized Volatility**: Volatility forecasting models
- **Hidden Markov Model (HMM)**: Market regime detection and prediction

### 6. Dash Application
- Interactive web app for real-time exploration
- Dynamic visualizations
- Model predictions interface

## Technologies Used

- **Data Collection**: yfinance, Alpha Vantage API
- **Data Analysis**: Python (Pandas, NumPy)
- **Visualization**: Matplotlib, Seaborn, Power BI
- **Database**: SQL
- **Machine Learning**: Scikit-learn, Statsmodels
- **Time Series**: ARIMA, GARCH, HMM
- **Web App**: Dash, Plotly
- **Office Suite**: Microsoft Excel

## Key Features

- **Automated CI/CD Pipeline**: GitHub Actions workflow for continuous integration and deployment
- Automated data pipeline using financial APIs
- Comprehensive exploratory data analysis across multiple dimensions
- Integration of technical, fundamental, and macroeconomic indicators
- Multi-model forecasting approach for returns and volatility
- Regime detection using Hidden Markov Models
- Interactive dashboards for business intelligence
- End-to-end deployment with Dash web application
- Model persistence using joblib for trained models

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/stock-analytics.git
cd stock-analytics

# Install required packages
pip install -r requirements.txt
```

## API Setup

**Alpha Vantage:**
1. Get your free API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Create a `.env` file in the project root
3. Add your API key:
```
ALPHA_VANTAGE_API_KEY=your_api_key_here
```

**yfinance:**
- No API key required
- Automatically fetches data from Yahoo Finance

## Usage

```bash
# 1. Fetch data from APIs
cd data_fetch
python [your_data_fetch_script].py

# 2. Create SQL tables and clean data
cd sql
python create_tables.py
# Run SQL queries for insights
python idk.py

# 3. Run notebooks for analysis
cd notebooks
jupyter notebook
# Open and run: ml_model.ipynb, Models_checking.ipynb

# 4. Launch Power BI dashboard
cd powerbi
# Open the .pbix file in Power BI Desktop

# 5. Start Dash application
cd app
python app.py
# Access the dashboard at http://localhost:8050
```

## Project Structure

```
YFINANCE_STOCKS_PROJECT/
├── .github/
│   └── workflows/              # CI/CD pipeline
├── app/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── app.py                  # Main Dash application
│   ├── data_processor.py       # Data processing utilities
│   ├── hmm_regime_classifier_v1.joblib
│   ├── returns_xgb_global_tuned_v1.joblib
│   └── volatility_xgb_no_persistence_v1.joblib
├── data/                       # Processed datasets
├── data_fetch/                 # API data fetching scripts
├── excel/                      # Excel analysis files
├── notebooks/
│   ├── .ipynb_checkpoints/
│   ├── keras_tuner_dir/
│   ├── hmm_regime_classifier_v1.joblib
│   ├── ml_model.ipynb          # Main ML modeling notebook
│   ├── Models_checking.ipynb   # Model validation notebook
│   ├── returns_xgb_global_tuned_v1.joblib
│   └── volatility_xgb_no_persistence_v1.joblib
├── powerbi/                    # Power BI dashboard
├── sql/
│   ├── create_tables.py        # Database table creation
│   ├── data_cleaning.sql       # SQL data cleaning scripts
│   ├── idk.py                  # SQL utilities
│   └── insights_queries.sql    # 5 analytical queries
├── .env                        # API keys (not committed)
├── .gitignore
├── company_data_cache.json     # Cached company fundamentals
├── profile.txt
├── requirements.txt
└── README.md
```

## Models Implemented

### Forecasting Models
- **XGBoost (Global Tuned)**: Returns forecasting with hyperparameter optimization
- **XGBoost (No Persistence)**: Volatility prediction model
- **Hidden Markov Model (HMM)**: Market regime classification

### Model Files
All trained models are saved as `.joblib` files for persistence and deployment:
- `returns_xgb_global_tuned_v1.joblib`
- `volatility_xgb_no_persistence_v1.joblib`
- `hmm_regime_classifier_v1.joblib`

## Results

- Returns forecasting accuracy metrics
- Volatility prediction performance
- Regime classification results
- Interactive visualizations in Dash app

## Future Enhancements

- Real-time data streaming
- Additional tickers and asset classes
- Deep learning models (Transformers)
- Portfolio optimization module
- Trading strategy backtesting

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data provided by yfinance and Alpha Vantage
- Inspired by quantitative finance research and industry practices

## Contact

Your Name - Yash sharma

Project Link: https://stocks-project-82n7.onrender.com
