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
- **Period**: 2011 - 2025
- **Total Records**: 11,001 observations
- **Features (33 total)**: 
  - Price metrics (Open, High, Low, Close, Volume)
  - Technical indicators (MA_20, MA_50, MA_200, Momentum_5d, Price_Range, Price_Range_Pct)
  - Volatility measures (Realized_Vol_10d, Realized_Vol_20d, Volatility_Ratio)
  - Volume metrics (Volume_MA_20, Volume_Ratio)
  - Macroeconomic data (Fed_Funds_Rate, CPI, Unemployment_Rate, GDP, Yield_Curve_10Y_2Y)
  - Market sentiment (VIX)
  - Lagged features (Return_Pct_lag_1, Return_Pct_lag_2, Return_Pct_lag_3)
  - Derived features (Price_ZScore, Momentum_Ratio_S_M, Momentum_Ratio_M_L)
  - Target variables (Target_Return_Next, Target_Vol_Next)
  - **HMM Regime** (bull/bear/neutral states from unsupervised learning)

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
- **5 comprehensive analytical queries** for deep insights:
  1. **Stock Performance Summary**: Overall returns, price movements, volume analysis
  2. **High Volatility & Volume Spike Detection**: Identifying significant market events
  3. **Value vs Growth Categorization**: Fundamental-based stock classification
  4. **Market Regime Analysis**: VIX-based performance comparison across volatility regimes
  5. **Advanced Multi-Factor Performance Attribution**: Risk-adjusted returns across Fed rate environments with Sharpe ratios

**Key Findings:**
- GOOGL outperformed with +89.2% returns over 3 years
- MSFT demonstrated best risk-adjusted returns (highest Sharpe ratio)
- GOOGL showed highest volatility sensitivity (55 high-vol days)
- MSFT proved most defensive in high-VIX environments
- Fed rate regime analysis revealed optimal allocation strategies

### 4. Power BI Dashboard
- **2-page interactive dashboard** with comprehensive visualizations

**Page 1 - Overview:**
- Key metrics cards: Total Revenue ($267B), Total Return (121%), Latest Close Price ($338), Average Daily Volume (39M), Average Volatility (1.67)
- Price trend analysis across all three tickers
- Total return comparison by stock (GOOGL: 162.15%, MSFT: 125.42%, AAPL: 86.31%)
- Average volume by stock breakdown
- Volatility comparison (GOOGL: 1.9, AAPL: 1.6, MSFT: 1.5)
- Fundamental metrics table (Debt-to-Equity, EPS, PE Ratio)
- Interactive filters: Month, Quarter, VIX Regime

**Page 2 - Detailed Analysis:**
- 52-week high/low tracking
- Volatility trend over time
- Volume vs 50-day moving average analysis
- Risk vs Return scatter plot
- Price vs Volatility trend correlation
- Daily metrics table with return %, volume, and 20-day volatility
- Quarter-by-quarter performance breakdown

### 5. Machine Learning & Forecasting

**Dataset Split:**
- Training set: 8,800 observations (80%)
- Test set: 2,201 observations (20%)
- Time-series validation with grouped sequences (time_steps=10)

**Time Series Analysis (TSA)**
- Stationarity tests (ADF, KPSS)
- Seasonality decomposition
- Autocorrelation analysis (ACF/PACF)

**Volatility Forecasting Models**

Compared multiple approaches for realized volatility prediction:

| Model | RMSE | MAE | QLIKE | Directional Acc | R² |
|-------|------|-----|-------|-----------------|-----|
| Naive (Global) | 0.0028 | 0.0014 | 0.0141 | 87.77% | 0.8686 |
| EWMA (Global) | 0.0095 | 0.0069 | 0.1337 | 63.59% | -0.5075 |
| GARCH(1,1) | 0.0083 | 0.0056 | 0.1195 | 67.55% | -0.1567 |
| EGARCH | 0.0086 | 0.0058 | 0.1361 | 66.73% | -0.2437 |
| XGBoost (Tuned) | 0.0027 | 0.0016 | 0.0132 | 86.91% | 0.8811 |
| **XGBoost (No Persistence)** | **0.0055** | **0.0040** | **0.0501** | **73.86%** | **0.4847** |
| GRU (LSTM) | 0.0850 | 0.0763 | 329,983 | 49.79% | -108.06 |
| GRU (Tuned) | 0.0040 | 0.0027 | 0.0303 | 53.32% | 0.7049 |

**Model Selection Decision:**
- **Deployed Model**: XGBoost (No Persistence) - R²=0.4847
- **Why Not Tuned?** The tuned model (R²=0.8811) uses lagged volatility features, creating **data leakage** in real-world predictions
- **Data Leakage Prevention**: Excluded these features to ensure true out-of-sample forecasting:
  - `Realized_Vol_10d`, `Realized_Vol_20d` (directly related to target)
  - `Volatility_Ratio` (derived from volatility lags)
  - All `_CLEAN` variants of volatility features
- **Trade-off**: Lower R² (0.48 vs 0.88) but **more realistic and deployable** predictions
- **Result**: Model still beats GARCH baseline (DM test p=0.0000) without data leakage

**Returns Forecasting Models**

| Model | Directional Accuracy | Annualized Sharpe | Information Coefficient | RMSE | MAE |
|-------|---------------------|-------------------|------------------------|------|-----|
| Naive | 50.25% | 0.87 | -0.0253 | 0.0244 | 0.0178 |
| MA-20 | 49.47% | 0.75 | -0.0571 | 0.0177 | 0.0125 |
| Linear Regression | 49.29% | 0.70 | 0.0334 | 0.0182 | 0.0131 |
| XGBoost (Default) | 49.20% | 0.70 | -0.0135 | 0.0247 | 0.0192 |
| **XGBoost (Tuned)** | **53.54%** | **1.08** | **0.0186** | **0.0177** | **0.0127** |

**Key Improvements:**
- Tuned XGBoost achieved 53.54% directional accuracy (actual test set had 54.36% up days)
- Annualized Sharpe ratio: 1.08 (significantly better than baselines)
- Model shows 92.14% "Up" predictions (optimistic bias)
- Classification metrics: 68% F1-score for "Up" predictions

**Regime Detection**
- **Hidden Markov Model (HMM)**: Identifies bull/bear/neutral market regimes
- Regime feature added to final dataset (33 features total)
- Improves XGBoost model performance when included as categorical feature

### 6. Web Application & Deployment
- **Dash Application**: Interactive web app for exploration and visualization
- **Flask API**: RESTful API for model predictions and data access
- **CI/CD Pipeline**: Automated testing and deployment via GitHub Actions

## Technologies Used

- **Data Collection**: yfinance, Alpha Vantage API
- **Data Analysis**: Python (Pandas, NumPy)
- **Visualization**: Matplotlib, Seaborn, Plotly, Power BI
- **Database**: SQL (MySQL)
- **Machine Learning**: 
  - Scikit-learn (preprocessing, metrics)
  - XGBoost (gradient boosting)
  - Statsmodels (GARCH, EGARCH)
  - TensorFlow/Keras (GRU, LSTM)
  - hmmlearn (Hidden Markov Models)
- **Model Persistence**: joblib
- **Web Frameworks**: 
  - Dash (interactive dashboard)
  - Flask (REST API)
- **DevOps**: GitHub Actions (CI/CD)
- **Office Suite**: Microsoft Excel

## Key Features

- **Automated CI/CD Pipeline**: GitHub Actions workflow for continuous integration and deployment
- **Automated data pipeline** using financial APIs (yfinance, Alpha Vantage)
- **Comprehensive EDA** across technical, fundamental, and macro dimensions
- **Advanced SQL analytics** with 5 in-depth queries revealing regime-based insights
- **Interactive Power BI dashboards** (2 pages) with filtering and drill-down capabilities
- **State-of-the-art ML models**:
  - XGBoost with hyperparameter tuning (R²=0.88 for volatility, Sharpe=1.08 for returns)
  - Deep learning with GRU/LSTM architectures
  - Regime detection using Hidden Markov Models
  - Diebold-Mariano tests for statistical validation
- **Model persistence** using joblib for production deployment
- **Dual web interfaces**:
  - Dash for interactive exploration
  - Flask REST API for programmatic access
- **End-to-end reproducibility** with virtual environments and version control

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
- **XGBoost (Returns - Global Tuned)**: Next-day returns forecasting with hyperparameter optimization
- **XGBoost (Volatility - No Persistence)**: Next-day volatility prediction **without lagged volatility features** to prevent data leakage
- **Hidden Markov Model (HMM)**: Market regime classification (bull/bear/neutral)

### Model Files & Data Leakage Prevention
All trained models are saved as `.joblib` files for persistence and deployment:
- `returns_xgb_global_tuned_v1.joblib` - Returns forecasting model
- `volatility_xgb_no_persistence_v1.joblib` - **No-persistence volatility model (deployed)**
- `hmm_regime_classifier_v1.joblib` - Regime detection model

**Volatility Model Choice:**
The "no persistence" model excludes these features to prevent data leakage:
```python
vol_features_to_drop = [
    'Realized_Vol_10d', 
    'Realized_Vol_20d', 
    'Volatility_Ratio',
    'Realized_Vol_10d_CLEAN',  
    'Realized_Vol_20d_CLEAN',
    'Volatility_Ratio_CLEAN'
]
```
This ensures the model can make true out-of-sample predictions in production without access to future volatility information.

## Results

### SQL Analysis Highlights
- **Performance**: GOOGL led with +89.2% returns, followed by AAPL (+74.0%) and MSFT (+69.5%)
- **Volatility**: 122 high-volatility days detected; GOOGL most volatile with 55 days
- **Risk-Adjusted**: MSFT achieved best Sharpe ratios across Fed rate regimes
- **Market Regimes**: MSFT most defensive in high-VIX (-0.218%), GOOGL best in low-VIX (+0.287%)
- **Fed Rate Impact**: GOOGL dominates in high-rate environments; MSFT excels in medium rates

### Machine Learning Performance

**Volatility Forecasting:**
- **Best Model**: XGBoost (No Persistence) - **Deployed Version**
- **R² Score**: 0.4847 (realistic performance without data leakage)
- **RMSE**: 0.0055
- **Directional Accuracy**: 73.86%
- **Statistical Significance**: Beat GARCH baseline (DM test p<0.0001)
- **Key Achievement**: Successfully predicts volatility using only price/volume features and macro indicators
- **Data Leakage Prevention**: Deliberately excluded lagged volatility features (`Realized_Vol_10d/20d`, `Volatility_Ratio`) that would create unrealistic in-sample performance but fail in production
- **SHAP Analysis**: SHAP output for the ML model reveals that the top drivers of volatility are actually **Fed_Funds_Rate** and **Price_Range_Pct**, which both rank higher in importance than the **VIX**. Other features like Momentum_5d and Volume_MA_20 also play a significant role.

**Returns Forecasting:**
- **Best Model**: XGBoost (Tuned)
- **Directional Accuracy**: 53.54%
- **Annualized Sharpe Ratio**: 1.08
- **Information Coefficient**: 0.0186 (positive skill)
- **RMSE**: 0.0177
- Significantly outperforms naive and MA-20 baselines

**Regime Classification:**
- HMM successfully identifies market regimes (bull/bear/neutral)
- Regime feature improves XGBoost performance when included
- 33-feature engineered dataset with regime as categorical variable

### Power BI Insights
- Portfolio generated 121% total return across $267B revenue
- Average daily volume: 39M shares with AAPL leading liquidity
- GOOGL shows highest volatility (1.9) but best absolute returns
- Clear risk-return trade-offs visible across all metrics
- 52-week range: High $535, Low $144
- Average portfolio Sharpe ratio: 0.07

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
