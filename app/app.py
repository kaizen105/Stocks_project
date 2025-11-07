import dash
from dash import dcc, html, Input, Output, State, dash_table, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime, timedelta
import yfinance as yf
from dash_extensions.caching import Cache, FileSystemStore # <--- MODIFIED 1: Added import

warnings.filterwarnings('ignore')
# Required imports for model loading
import xgboost
import hmmlearn
from hmmlearn.hmm import GaussianHMM
# Import data processor
from app.data_processor import (
    fetch_stock_data,
    calculate_technical_features,
    prepare_model_input,
    fetch_macro_data,
    search_stocks,
    get_market_overview,
    get_company_overview
)
# =============================================================================
# 1. LOAD MODELS
# =============================================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS = {}
MODELS_LOADED = False
print("--- Loading models... ---")
try:
    MODELS = {
        "hmm": joblib.load(os.path.join(APP_DIR, 'hmm_regime_classifier_v1.joblib')),
        # This model predicts NEXT-DAY volatility
        "volatility": joblib.load(os.path.join(APP_DIR, 'volatility_xgb_no_persistence_v1.joblib')),
        # This model predicts NEXT-DAY return
        "returns_1d": joblib.load(os.path.join(APP_DIR, 'returns_xgb_global_tuned_v1.joblib'))
    }
    MODELS_LOADED = True
    print("✅ Models loaded")
except FileNotFoundError as e:
    print(f"❌ Model error: {e}")
    MODELS_LOADED = False
# Constants
MACRO_FEATURES = ['Fed_Funds_Rate', 'CPI', 'Unemployment_Rate', 'GDP', 'Yield_Curve_10Y_2Y', 'VIX']
MACRO_DEFAULTS = {
    'Fed_Funds_Rate': 3.90, 'CPI': 3.0, 'Unemployment_Rate': 4.35,
    'GDP': 4.0, 'Yield_Curve_10Y_2Y': 0.52, 'VIX': 16.37
}
CHART_HEIGHT = 400
# A larger set of popular tickers (used to generate the Popular Stocks cards)
POPULAR_TICKERS = [
    'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'BRK-B', 'JPM',
    'V', 'MA', 'DIS', 'NFLX', 'XOM', 'PFE', 'INTC', 'ORCL', 'BABA', 'BAC', 'CSCO'
]

def create_popular_cards():
    """Builds a responsive row of popular stock cards using COMPANY_CACHE when available."""
    colors = ['primary', 'success', 'info', 'danger', 'warning', 'secondary', 'dark']
    cols = []
    for i, symbol in enumerate(POPULAR_TICKERS):
        # Try to get the company name from the cache via get_company_overview
        try:
            company = get_company_overview(symbol).get('Name', '')
        except Exception:
            company = ''

        color = colors[i % len(colors)]
        cols.append(
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H5(symbol, className=f"text-{color}"),
                    html.P(company, className="text-muted small"),
                    dbc.Button("Analyze", href=f"/analysis?ticker={symbol}", color=color, size="sm", className="w-100")
                ])
            ], className="h-100"), md=2, className="mb-3")
        )

    return dbc.Row(cols)
# =============================================================================
# 2. APP INITIALIZATION
# =============================================================================
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True
)
server = app.server

# <--- MODIFIED 2: Initialize the cache ---
cache = Cache(app.server, config={
    'CACHE_TYPE': 'FileSystemCache',
    # This folder will be created to store cache files
    'CACHE_DIR': 'app-cache-directory' 
})
# ----------------------------------------

# =============================================================================
# 3. NAVBAR
# =============================================================================
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink([html.I(className="fas fa-home me-1"), "Home"], href="/", active="exact")),
        dbc.NavItem(dbc.NavLink([html.I(className="fas fa-chart-line me-1"), "Analysis"], href="/analysis", active="exact")),
        dbc.NavItem(dbc.NavLink([html.I(className="fas fa-crystal-ball me-1"), "Predict"], href="/prediction", active="exact")),
    ],
    brand=[html.I(className="fas fa-chart-pie me-2"), "StockPredict Pro"],
    brand_href="/",
    color="primary",
    dark=True,
    className="mb-4 shadow-sm"
)
# =============================================================================
# 4. HOME PAGE LAYOUT
# =============================================================================
home_layout = dbc.Container([
    # Hero Section
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("📈 Stock Prediction Platform", className="display-3 text-white mb-4"),
                html.P("Real-time stock analysis with AI-powered predictions and live market data",
                        className="lead text-white-50 mb-4"),
                dbc.Button("Start Analyzing", href="/analysis", color="light", size="lg", className="me-3"),
                dbc.Button("Run Predictions", href="/prediction", color="outline-light", size="lg")
            ], style={
                'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                'padding': '60px',
                'borderRadius': '15px',
                'textAlign': 'center',
                'minHeight': '50vh'
            })
        ])
    ], className="mb-5"),
    
    # Live Market Overview
    html.H3([html.I(className="fas fa-globe me-2"), "Live Market Overview"], className="mb-4"),
    dbc.Row(id='market-overview-cards', className="mb-5"),
    
    # Live Stock Search
    html.H3([html.I(className="fas fa-search me-2"), "Real-Time Stock Search"], className="mb-4"),
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Input(
                        id='live-search-input',
                        type='text',
                        placeholder='Search stocks by symbol or company name...',
                        debounce=True
                    )
                ], md=10),
                dbc.Col([
                    dbc.Button([html.I(className="fas fa-search")], id='search-btn', color="primary")
                ], md=2)
            ]),
            html.Div(id='search-results', className="mt-3")
        ])
    ], className="mb-5"),
    
    # Popular Stocks (dynamically generated)
    html.H3([html.I(className="fas fa-star me-2"), "Popular Stocks"], className="mb-4"),
    create_popular_cards(),
], fluid=True, className="py-4")
# =============================================================================
# 5. ANALYSIS PAGE LAYOUT
# =============================================================================
analysis_layout = dbc.Container([
    html.H2([html.I(className="fas fa-chart-bar me-2"), "Advanced Stock Analysis"], className="mb-4"),
    
    # Search Bar
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col(dbc.Input(id='ticker-input', type='text', placeholder='Enter Ticker (e.g., AAPL)', value='AAPL'), md=4),
                dbc.Col(dbc.Button([html.I(className="fas fa-download me-1"), "Fetch Data"], id='fetch-button', color="success"), md=2),
            ])
        ])
    ], className="mb-4"),
    
    # Status
    html.Div(id='analysis-status'),
    
    # Company Info Card
    dbc.Row([
        dbc.Col(html.Div(id='company-info-card'), md=12)
    ], className="mb-4"),
    
    # ... (all your chart rows) ...
    # Charts Row 1: Price Chart
    dbc.Row([
        dbc.Col(dcc.Loading(html.Div(id='price-chart')), md=12),
    ], className="mb-4"),
    
    # Charts Row 2: Volume + Volatility
    dbc.Row([
        dbc.Col(dcc.Loading(html.Div(id='volume-chart')), md=6),
        dbc.Col(dcc.Loading(html.Div(id='volatility-chart')), md=6),
    ], className="mb-4"),
    
    # Charts Row 3: Momentum + MA
    dbc.Row([
        dbc.Col(dcc.Loading(html.Div(id='momentum-chart')), md=6),
        dbc.Col(dcc.Loading(html.Div(id='ma-chart')), md=6),
    ], className="mb-4"),
    
    # Charts Row 4: Returns Dist
    dbc.Row([
        dbc.Col(dcc.Loading(html.Div(id='returns-dist-chart')), md=12),
    ], className="mb-4"),
    
    # Summary Stats
    dbc.Row([
        dbc.Col(html.Div(id='summary-stats'), md=12)
    ])
], fluid=True)
# =============================================================================
# 6. PREDICTION PAGE LAYOUT
# =============================================================================
def create_macro_inputs():
    inputs = []
    for feature in MACRO_FEATURES:
        tooltip = dbc.Tooltip(f"Enter expected {feature.replace('_', ' ').lower()}", target=f"tooltip-{feature}")
        row = dbc.Row([
            dbc.Label([
                feature.replace('_', ' ').title(),
                dbc.Badge(f"ℹ️", id=f"tooltip-{feature}", color="info", pill=True, className="ms-1")
            ], width=4, html_for=f"input-{feature}"),
            dbc.Col(dbc.Input(id=f"input-{feature}", type="number", value=MACRO_DEFAULTS.get(feature), step=0.01), width=8)
        ], className="mb-2")
        inputs.append(row)
        inputs.append(tooltip)
    return inputs

prediction_layout = dbc.Container([
    html.H2([html.I(className="fas fa-magic me-2"), "AI-Powered Prediction Engine"], className="mb-4"),
    
    dbc.Row([
        # Left: Inputs
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Prediction Inputs")),
                dbc.CardBody([
                    dbc.Label("Stock Ticker"),
                    dbc.Input(id='predict-ticker-input', type='text', placeholder='e.g., TSLA', value='TSLA', className="mb-3"),
                    html.Hr(),
                    html.H6("Macroeconomic Inputs"),
                    html.Div(create_macro_inputs()),
                    dbc.Button([html.I(className="fas fa-sync me-1"), "Load Latest"], id='auto-macro-btn', color="info", className="w-100 mb-2"),
                    dbc.Button([html.I(className="fas fa-play me-1"), "Run Prediction"], id='predict-button', color="danger", className="w-100"),
                ])
            ], className="shadow")
        ], md=4),
        
        # Right: Results
        dbc.Col([
            html.Div(id='prediction-status'),
            dcc.Loading(id='prediction-loading', children=html.Div(id='prediction-output'), type="circle")
        ], md=8)
    ])
], fluid=True)
# =============================================================================
# 7. MAIN LAYOUT
# =============================================================================
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Interval(id='market-update-interval', interval=300000, n_intervals=0),  # 5min update
    navbar,
    html.Div(id='page-content')
])
# =============================================================================
# 8. URL HANDLING & TICKER FROM PARAMS
# =============================================================================
@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/analysis':
        return analysis_layout
    elif pathname == '/prediction':
        return prediction_layout
    return home_layout

@app.callback(
    Output('ticker-input', 'value'),
    Input('url', 'search')
)
def set_ticker_from_url(search):
    if search and 'ticker=' in search:
        return search.split('ticker=')[1].split('&')[0].upper()
    return 'AAPL'
# =============================================================================
# 9. HOME PAGE CALLBACKS
# =============================================================================
@app.callback(
    Output('market-overview-cards', 'children'),
    Input('market-update-interval', 'n_intervals')
)
def update_market_overview(n):
    # This function is fine, uses yfinance for live data
    market_data = get_market_overview()
    cards = []
    for name, data in market_data.items():
        color = 'success' if data['change'] >= 0 else 'danger'
        icon = 'fa-arrow-up' if data['change'] >= 0 else 'fa-arrow-down'
        card = dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6(name, className="text-muted"),
                    html.H4(f"${data['price']:.2f}", className="mb-0"),
                    html.P([
                        html.I(className=f"fas {icon} me-1"),
                        f"{data['change']:+.2f} ({data['change_pct']:+.2f}%)"
                    ], className=f"text-{color} mb-0")
                ])
            ], className="shadow-sm")
        ], md=4)
        cards.append(card)
    return cards

@app.callback(
    Output('search-results', 'children'),
    [Input('search-btn', 'n_clicks'), Input('live-search-input', 'value')],
    prevent_initial_call=True
)
def live_search(n_clicks, search_value):
    # This function is fine, uses the rate-limited search
    if not search_value or len(search_value) < 2:
        return ""
    results = search_stocks(search_value)
    if not results:
        return dbc.Alert("No results found", color="info")
    result_cards = []
    for stock in results:
        card = dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5(stock['symbol'], className="text-primary mb-1"),
                    html.P(stock['name'], className="text-muted small mb-2"),
                    html.P([
                        html.Small(f"{stock['type']} | {stock['region']}", className="text-secondary")
                    ]),
                    dbc.Button("Analyze", href=f"/analysis?ticker={stock['symbol']}",
                                color="primary", size="sm", className="w-100")
                ])
            ], className="h-100")
        ], md=3, className="mb-3")
        result_cards.append(card)
    return dbc.Row(result_cards)

# <--- MODIFIED 3: Added new cached helper function ---
@cache.memoize(timeout=300) # Caches for 300 seconds (5 minutes)
def get_yfinance_info(ticker):
    """
    Safely fetches and caches yf.Ticker.info data.
    Returns None if the ticker is invalid.
    """
    print(f"--- [Cache MISS] Fetching yf.info for {ticker} ---")
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        # Check for a valid price to confirm it's a real ticker
        if info and info.get('regularMarketPrice') is not None:
            return info
        return None
    except Exception as e:
        print(f"Error fetching yf.info for {ticker}: {e}")
        return None
# ----------------------------------------------------

# =============================================================================
# 10. ANALYSIS PAGE CALLBACKS (DEBUGGED)
# =============================================================================
@app.callback(
    [
        Output('analysis-status', 'children'),
        Output('company-info-card', 'children'),
        Output('price-chart', 'children'),
        Output('volume-chart', 'children'),
        Output('volatility-chart', 'children'),
        Output('momentum-chart', 'children'),
        Output('ma-chart', 'children'),
        Output('returns-dist-chart', 'children'),
        Output('summary-stats', 'children')
    ],
    Input('fetch-button', 'n_clicks'),
    State('ticker-input', 'value')
)
def update_analysis(n_clicks, ticker):
    if not n_clicks or not ticker:
        return [""] * 9
    
    print(f"\n--- [Analysis Callback Triggered] Ticker: {ticker} ---")
    ticker = ticker.upper()
    
    try:
        # <--- MODIFIED 4: Replaced direct .info call with cached function ---
        print(f"Step 1: Validating ticker via cache...")
        ticker_info = get_yfinance_info(ticker)
        
        if not ticker_info:
            print("Step 1 FAILED: Ticker validation failed or is invalid.")
            return [dbc.Alert(f"Invalid or delisted ticker: {ticker}", color="danger")] + [""] * 8
        # --- END OF MODIFICATION ---
        
        print(f"Step 2: Ticker valid. Fetching stock data...")
        df = fetch_stock_data(ticker, days_needed=252)
        if df.empty:
            print("Step 2 FAILED: fetch_stock_data returned an empty DataFrame.")
            return [dbc.Alert(f"No data for {ticker}", color="danger")] + [""] * 8
        
        print("Step 3: Data fetched. Calculating technical features...")
        df_features = calculate_technical_features(df.copy())
        
        if len(df_features) == 0:
            print("Step 3 FAILED: Feature calculation resulted in empty data.")
            return [dbc.Alert("Insufficient history", color="warning")] + [""] * 8
        
        print("Step 4: Features calculated. Getting company overview from CACHE...")
        # This now reads from the JSON cache, so it's instant
        company_info = get_company_overview(ticker)
        
        if not company_info:
            print(f"Step 4 WARNING: No cache data for {ticker}. Proceeding with N/A.")
            company_info = {} # Ensure it's a dict
        
        print("Step 5: Building info card...")
        info_card = dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H3(f"{ticker} - {company_info.get('Name', 'N/A')}", className="mb-2"),
                        html.P([
                            html.Strong("Sector: "), company_info.get('Sector', 'N/A'), " | ",
                            html.Strong("Industry: "), company_info.get('Industry', 'N/A')
                        ], className="text-muted")
                    ], md=8),
                    dbc.Col([
                        html.H5(f"Market Cap: {company_info.get('Market Cap', 'N/A')}", className="text-end"),
                        html.P(f"P/E Ratio: {company_info.get('PE Ratio', 'N/A')}", className="text-muted text-end")
                    ], md=4)
                ])
            ])
        ], color="light", className="mb-3")
        
        print("Step 6: Building charts...")
        # Chart 1: Candlestick Price Chart
        price_hovertext = [
            f"<b>Open</b>: {o:.2f}<br><b>High</b>: {h:.2f}<br><b>Low</b>: {l:.2f}<br><b>Close</b>: {c:.2f}"
            for o, h, l, c in zip(df['Open'], df['High'], df['Low'], df['Close'])
        ]
        fig_price = go.Figure(go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='Price', hovertext=price_hovertext, hoverinfo='text'
        ))
        fig_price.update_layout(
            title=f'{ticker} Price', xaxis_rangeslider_visible=False,
            showlegend=False, height=CHART_HEIGHT * 1.5, template='plotly_white'
        )
        
        # Chart 2: Volume Chart
        colors = ['red' if row['Close'] < row['Open'] else 'green' for _, row in df.iterrows()]
        fig_volume = go.Figure(go.Bar(
            x=df.index, y=df['Volume'], name='Volume', marker_color=colors,
            hovertemplate='<b>Volume</b>: %{y:,.0f}<extra></extra>'
        ))
        fig_volume.update_layout(title='Trading Volume', template='plotly_white', height=CHART_HEIGHT)
        
        # Chart 3: Volatility Chart
        fig_volatility = go.Figure()
        fig_volatility.add_trace(go.Scatter(
            x=df_features.index, y=df_features['rolling_std_20'],
            name='20-day Volatility', line=dict(color='orange'),
            hovertemplate='<b>Volatility</b>: %{y:.4f}<extra></extra>' # Changed to .4f
        ))
        fig_volatility.update_layout(
            title='Rolling Volatility (20-day)', template='plotly_white', height=CHART_HEIGHT
        )
        
        # Chart 4: Momentum Chart (RSI)
        fig_momentum = go.Figure()
        fig_momentum.add_trace(go.Scatter(
            x=df_features.index, y=df_features['RSI'],
            name='RSI', line=dict(color='purple'),
            hovertemplate='<b>RSI</b>: %{y:.2f}<extra></extra>'
        ))
        fig_momentum.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, annotation_text="Overbought")
        fig_momentum.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, annotation_text="Oversold")
        fig_momentum.update_layout(title='Relative Strength Index (RSI)', template='plotly_white', height=CHART_HEIGHT)
        
        # Chart 5: Moving Averages
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=df_features.index, y=df['Close'], name='Price'))
        fig_ma.add_trace(go.Scatter(x=df_features.index, y=df_features['MA50'], name='MA50'))
        fig_ma.add_trace(go.Scatter(x=df_features.index, y=df_features['MA200'], name='MA200'))
        fig_ma.update_layout(title='Moving Averages', template='plotly_white', height=CHART_HEIGHT)
        
        # Chart 6: Returns Distribution
        returns = df_features['Return_Pct'].dropna() # Use features to respect dropna
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=returns, nbinsx=50, name='Returns Distribution',
            hovertemplate='<b>Return</b>: %{x:.2%}<extra></extra>' # This is fine as %
        ))
        fig_dist.add_vline(x=returns.mean(), line_dash="dash", line_color="red", annotation_text="Mean")
        fig_dist.update_layout(title='Daily Returns Distribution', template='plotly_white', height=CHART_HEIGHT)
        
        print("Step 7: Building summary stats...")
        summary_stats = dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Price Statistics"),
                        html.P(f"Current Price: ${df['Close'].iloc[-1]:.2f}"),
                        html.P(f"52-Week High: ${df['High'].max():.2f}"),
                        html.P(f"52-Week Low: ${df['Low'].min():.2f}"),
                    ], md=4),
                    dbc.Col([
                        html.H6("Returns Statistics"),
                        html.P(f"Daily Volatility (Std): {returns.std():.4f}"),
                        html.P(f"Annual Volatility: {returns.std()*np.sqrt(252):.4f}"),
                        html.P(f"Sharpe Ratio (Annualized): {(returns.mean()/returns.std())*np.sqrt(252):.2f}"),
                    ], md=4),
                    dbc.Col([
                        html.H6("Technical Indicators"),
                        html.P(f"RSI: {df_features['RSI'].iloc[-1]:.2f}"),
                        html.P(f"Above MA50: {'Yes' if df['Close'].iloc[-1] > df_features['MA50'].iloc[-1] else 'No'}"),
                        html.P(f"Above MA200: {'Yes' if df['Close'].iloc[-1] > df_features['MA200'].iloc[-1] else 'No'}"),
                    ], md=4),
                ])
            ])
        ])
        
        print("Step 8: All components built. Returning to browser.")
        
        return [
            dbc.Alert(f"Analysis complete for {ticker}", color="success"),
            info_card,
            dcc.Graph(figure=fig_price),
            dcc.Graph(figure=fig_volume),
            dcc.Graph(figure=fig_volatility),
            dcc.Graph(figure=fig_momentum),
            dcc.Graph(figure=fig_ma),
            dcc.Graph(figure=fig_dist),
            summary_stats
        ]
        
    except Exception as e:
        print("\n" + "="*50)
        print("--- ANALYSIS CALLBACK CRASHED! ---")
        print(f"An unexpected error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        print("="*50 + "\n")
        return [dbc.Alert(f"Error: {str(e)}", color="danger")] + [""] * 8
# =============================================================================
# 11. PREDICTION PAGE CALLBACKS (*** THIS IS THE FINAL FIX ***)
# =============================================================================

# --- FIX 1: Renamed function to reflect 1-DAY forecast ---
def create_forecast_plot_1day(df, returns_pred, current_price):
    fig = go.Figure()
    
    # Historical (last 30 days)
    hist_start = max(len(df)-30, 0)
    hist_dates = df.index[hist_start:]
    fig.add_trace(go.Scatter(
        x=hist_dates, y=df['Close'].iloc[hist_start:],
        name='Historical Price',
        line=dict(color='blue'),
        hovertemplate='<b>Date</b>: %{x}<br><b>Price</b>: $%{y:.2f}<extra></extra>'
    ))
    
    # --- FIX 2: Only predict ONE business day into the future ---
    last_date = df.index[-1]
    future_date = pd.bdate_range(start=last_date, periods=2)[-1] # Next business day
    predicted_price = current_price * (1 + returns_pred)
    
    fig.add_trace(go.Scatter(
        x=[last_date, future_date], 
        y=[current_price, predicted_price],
        name=f'Forecast ({returns_pred*100:+.2f}%)',
        line=dict(color='red', dash='dash'),
        hovertemplate='<b>Date</b>: %{x}<br><b>Forecast Price</b>: $%{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        # --- FIX 3: Update title ---
        title='Next-Day Price Forecast',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_white',
        height=CHART_HEIGHT * 1.2,
        hovermode='x unified'
    )
    
    return fig

@app.callback(
    [Output('prediction-status', 'children'),
     Output('prediction-output', 'children')] +
    [Output(f'input-{feature}', 'value') for feature in MACRO_FEATURES],
    [Input('predict-button', 'n_clicks'),
     Input('auto-macro-btn', 'n_clicks')],
    [State('predict-ticker-input', 'value')] +
    [State(f'input-{feature}', 'value') for feature in MACRO_FEATURES]
)
def run_prediction(n_clicks, auto_clicks, ticker, *macro_values):
    ctx = dash.callback_context
    
    def _to_float_or_default(val, default):
        if val is None: return float(default)
        if isinstance(val, (pd.Series, pd.Index, list, tuple, np.ndarray)):
            try: val = val[0]
            except Exception: return float(default)
        try:
            f = float(val)
            if np.isnan(f): return float(default)
            return f
        except Exception: return float(default)
            
    if not ctx.triggered:
        return [""] + [""] + [None] * len(MACRO_FEATURES)
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Auto-load macro data
    if button_id == 'auto-macro-btn':
        print("\n--- [Prediction Callback] Loading macro data... ---")
        macro_data = fetch_macro_data()
        if macro_data is None:
            status = dbc.Alert("Failed to fetch macro data", color="danger")
            return [status, ""] + [None] * len(MACRO_FEATURES)
        status = dbc.Alert("Macro data loaded successfully", color="success")
        input_values = []
        for i, feature in enumerate(MACRO_FEATURES):
            raw = macro_data.get(feature, None)
            if raw is None:
                raw = macro_values[i] if i < len(macro_values) else None
            input_values.append(_to_float_or_default(raw, MACRO_DEFAULTS[feature]))
        return [status, "", *input_values]
    
    # Prediction run
    if not n_clicks or not ticker or not MODELS_LOADED:
        status = ""
        if not MODELS_LOADED:
            status = dbc.Alert("Models not loaded. Check console.", color="danger")
        return [status, ""] + [None] * len(MACRO_FEATURES)
    
    print(f"\n--- [Prediction Callback Triggered] Ticker: {ticker} ---")

    try:
        ticker = ticker.upper()
        print("Step 1: Fetching stock data for prediction...")
        df = fetch_stock_data(ticker)
        if df.empty:
            print("Step 1 FAILED: No data found.")
            return [dbc.Alert(f"No data found for {ticker}", color="danger"), ""] + [None] * len(MACRO_FEATURES)
        
        macro_dict = dict(zip(MACRO_FEATURES, [_to_float_or_default(v if (v is not None and v != '') else None, MACRO_DEFAULTS[f]) for v, f in zip(macro_values, MACRO_FEATURES)]))
        
        print("Step 2: Calculating features for prediction...")
        df_features = calculate_technical_features(df.copy())
        hmm_input, xgb_input = prepare_model_input(df_features, macro_dict)
        
        print("Step 3: Running models (HMM, Volatility, Returns)...")
        # Predictions
        regimes = MODELS['hmm'].predict(hmm_input)
        regime = regimes[-1]
        
        # This now predicts the NEXT-DAY volatility (e.g., 0.02)
        volatility_pred = MODELS['volatility'].predict(xgb_input)[0] 
        
        # --- FIX 4: Use the correct model name ---
        # This now predicts the NEXT-DAY return (e.g., 0.005)
        returns_pred = MODELS['returns_1d'].predict(xgb_input)[0]
        
        current_price = df['Close'].iloc[-1]
        predicted_price = current_price * (1 + returns_pred)
        
        print("Step 4: Building prediction cards and plots...")
        # Prediction cards
        regime_color = 'success' if regime == 1 else 'danger'
        regime_text = "Bullish" if regime == 1 else "Bearish"
        returns_color = 'success' if returns_pred > 0 else 'danger'
        
        prediction_cards = dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Market Regime"),
                dbc.CardBody([
                    html.H4(regime_text, className=f"text-{regime_color}"),
                    html.P("HMM regime (medium / long-term signal)", className="text-muted small mt-2")
                ])
            ]), md=4),
            dbc.Col(dbc.Card([
                # --- FIX 5: Update card title and value ---
                dbc.CardHeader("Next-Day Volatility"),
                # Display as a sane decimal
                dbc.CardBody(html.H4(f"{volatility_pred:.4f}", className="text-warning")) 
            ]), md=4),
            dbc.Col(dbc.Card([
                # --- FIX 6: Update card title ---
                dbc.CardHeader("Next-Day Price Forecast"),
                dbc.CardBody([
                    html.H4(f"${predicted_price:.2f}"),
                    # Display the 1-DAY return
                    html.P(f"({returns_pred*100:+.2f}%)", className=f"text-{returns_color}") 
                ])
            ]), md=4)
        ])
        
        # Plot 1: HMM Regime Evolution
        fig_regime = go.Figure()
        # ... (HMM plot code is fine) ...
        fig_regime.add_trace(go.Scatter(
            x=df_features.index, y=regimes, mode='lines', name='Historical Regimes',
            line=dict(color='purple'), hovertemplate='<b>Date</b>: %{x}<br><b>Regime</b>: %{y}<extra></extra>'
        ))
        fig_regime.add_trace(go.Scatter(
            x=[df_features.index[-1]], y=[regime], mode='markers', marker=dict(size=12, color='red'),
            name='Current Regime', hovertemplate='<b>Current</b>: %{y}<extra></extra>'
        ))
        fig_regime.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Bearish / Bullish")
        fig_regime.update_layout(
            title='Market Regime Evolution (HMM)',
            yaxis=dict(tickvals=[0, 1], ticktext=['Bearish', 'Bullish']),
            template='plotly_white', height=CHART_HEIGHT, hovermode='x unified'
        )
        
        # Plot 2: Volatility Trend & Prediction
        recent_vol_idx = -min(30, len(df_features))
        recent_vol = df_features['rolling_std_20'].iloc[recent_vol_idx:]
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=recent_vol.index, y=recent_vol, mode='lines', name='Recent Volatility',
            line=dict(color='orange'), hovertemplate='<b>Date</b>: %{x}<br><b>Vol</b>: %{y:.4f}<extra></extra>'
        ))
        # --- FIX 7: Update volatility plot to show NEXT-DAY prediction ---
        fig_vol.add_hline(y=volatility_pred, line_dash="dash", line_color="red",
                          annotation_text=f'Predicted Next-Day Vol: {volatility_pred:.4f}')
        fig_vol.update_layout(
            title='Volatility Trend & Next-Day Prediction (XGBoost)',
            yaxis_title='Volatility (Std. Dev of Daily Returns)', 
            template='plotly_white', height=CHART_HEIGHT, hovermode='x unified'
        )
        
        # Plot 3: Returns Trend & Expected
        recent_returns = df_features['Return_Pct'].iloc[-30:].dropna()
        fig_returns = go.Figure()
        fig_returns.add_trace(go.Scatter(
            x=recent_returns.index, y=recent_returns, mode='lines+markers', name='Recent Daily Returns',
            line=dict(color='green'), hovertemplate='<b>Date</b>: %{x}<br><b>Return</b>: %{y:.2%}<extra></extra>'
        ))
        # --- FIX 8: Update returns plot to show NEXT-DAY prediction ---
        fig_returns.add_hline(y=returns_pred, line_dash="dash", line_color="blue",
                              annotation_text=f'Predicted Next-Day Return: {returns_pred:.2%}')
        fig_returns.update_layout(
            title='Recent Returns & Next-Day Prediction (XGBoost)',
            yaxis_title='Daily Return', template='plotly_white', height=CHART_HEIGHT, hovermode='x unified'
        )
        
        # --- FIX 9: Call the new 1-day plot function ---
        fig_forecast = create_forecast_plot_1day(df_features, returns_pred, current_price)
        
        # Layout plots in two rows
        plots_row1 = dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_regime), md=6),
            dbc.Col(dcc.Graph(figure=fig_vol), md=6)
        ], className="mb-4")
        plots_row2 = dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_returns), md=6),
            dbc.Col(dcc.Graph(figure=fig_forecast), md=6)
        ], className="mb-4")
        
        output_content = html.Div([prediction_cards, plots_row1, plots_row2])
        input_values_after = [_to_float_or_default(macro_dict.get(f, None), MACRO_DEFAULTS[f]) for f in MACRO_FEATURES]
        
        print("Step 5: Prediction complete. Returning to browser.")
        return [dbc.Alert("Prediction complete", color="success"), output_content] + input_values_after
    
    except Exception as e:
        print("\n" + "="*50)
        print("--- PREDICTION CALLBACK CRASHED! ---")
        print(f"An unexpected error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        print("="*50 + "\n")
        
        return [dbc.Alert(f"Error: {str(e)}", color="danger"), ""] + [None] * len(MACRO_FEATURES)

if __name__ == '__main__':
    app.run(debug=True)