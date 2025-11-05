# dashboard_app.py
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime

# Initialize app
app = dash.Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    pages_folder="",  # 👈 disables need for /pages directory
    external_stylesheets=[
        'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap'
    ]
)
app.title = "Stock Intelligence Demo"

server = app.server

# Enhanced Custom index HTML (improved dark aesthetic with animations)
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Stock Intelligence Demo</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                margin: 0;
                background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
                font-family: 'Inter', sans-serif;
                color: #fff;
                overflow-x: hidden;
                transition: background 0.5s ease-in-out;
            }
            .navbar {
                display: flex;
                justify-content: center;
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(20px);
                padding: 15px 0;
                border-bottom: 1px solid rgba(255,255,255,0.1);
                position: sticky;
                top: 0;
                z-index: 1000;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
            }
            .nav-link {
                margin: 0 25px;
                font-weight: 600;
                font-size: 16px;
                color: rgba(255, 255, 255, 0.7);
                text-decoration: none;
                padding: 10px 20px;
                border-radius: 50px;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                overflow: hidden;
            }
            .nav-link::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
                transition: left 0.5s;
            }
            .nav-link:hover::before {
                left: 100%;
            }
            .nav-link:hover {
                color: #00d2ff;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0, 210, 255, 0.3);
                background: rgba(0, 210, 255, 0.1);
            }
            .nav-link.active {
                color: #00d2ff;
                background: rgba(0, 210, 255, 0.15);
                box-shadow: 0 0 20px rgba(0, 210, 255, 0.2);
            }
            .page-content {
                opacity: 0;
                transform: translateY(20px);
                animation: fadeInUp 0.6s cubic-bezier(0.4, 0, 0.2, 1) forwards;
                padding: 40px;
                max-width: 1200px;
                margin: 0 auto;
            }
            @keyframes fadeInUp {
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            .card {
                background: rgba(255, 255, 255, 0.08);
                backdrop-filter: blur(20px);
                border-radius: 24px;
                padding: 25px;
                border: 1px solid rgba(255, 255, 255, 0.12);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                overflow: hidden;
            }
            .card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 1px;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            }
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
                border-color: rgba(0, 210, 255, 0.3);
            }
            .metric-icon {
                font-size: 24px;
                margin-bottom: 10px;
            }
            .positive { color: #00ff88; }
            .negative { color: #ff6b6b; }
            .neutral { color: #ffd93d; }
            .loading {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid rgba(255,255,255,.3);
                border-radius: 50%;
                border-top-color: #00d2ff;
                animation: spin 1s ease-in-out infinite;
            }
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            .dropdown {
                background: rgba(255, 255, 255, 0.08) !important;
                border: 1px solid rgba(255, 255, 255, 0.12) !important;
                color: #fff !important;
                border-radius: 12px !important;
                backdrop-filter: blur(10px) !important;
            }
            .dropdown:hover {
                border-color: rgba(0, 210, 255, 0.3) !important;
            }
            .select-option {
                background: rgba(0, 0, 0, 0.8) !important;
                color: #fff !important;
            }
            footer {
                text-align: center;
                padding: 20px;
                color: rgba(255,255,255,0.5);
                font-size: 12px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>Built with ❤️ using Dash & Plotly</footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </body>
</html>
'''

# Generate mock stock data function (Fixed returns calculation)
def generate_mock_stock_data(days=100):
    dates = pd.date_range(end=datetime.now(), periods=days)
    # Simulate stock price with trend and noise
    price = 100 + np.cumsum(np.random.randn(days) * 0.5)
    price = np.maximum(price, 50)  # Floor at 50
    volume = np.random.randint(1000000, 10000000, days)
    # Fixed returns calculation
    returns = np.zeros(len(price))
    returns[1:] = np.diff(price) / price[:-1] * 100
    df = pd.DataFrame({
        'Date': dates,
        'Close': price,
        'Volume': volume,
        'Returns': returns
    })
    return df

# Pre-generate data for dashboard to avoid re-generation issues
mock_data = generate_mock_stock_data()

# ---------------------------
# PAGE 1 — DASHBOARD (Enhanced with realistic mock data and icons)
# ---------------------------
dash.register_page(
    "home",
    path="/",
    title="Dashboard",
    layout=html.Div([
        html.H1("📈 Stock Intelligence Dashboard",
                style={'textAlign': 'center', 'marginBottom': '40px', 'fontWeight': '700', 'letterSpacing': '1px'}),
        html.Div([
            html.Div([
                html.Div("📊", className='metric-icon'),
                html.H3("Portfolio Yield", style={'color': '#a0a0a0', 'marginBottom': '10px'}),
                html.H1("7.23%", style={'color': '#00d2ff', 'marginBottom': '5px'}),
                html.P("↑ +0.32% MoM", className='positive', style={'margin': '0'})
            ], className='card', style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
            html.Div([
                html.Div("🔮", className='metric-icon'),
                html.H3("Prediction Accuracy", style={'color': '#a0a0a0', 'marginBottom': '10px'}),
                html.H1("0.87", style={'color': '#ffd93d', 'marginBottom': '5px'}),
                html.P("R² Score", style={'color': '#aaa', 'margin': '0'})
            ], className='card', style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
            html.Div([
                html.Div("🐂", className='metric-icon'),
                html.H3("Market Regime", style={'color': '#a0a0a0', 'marginBottom': '10px'}),
                html.H1("Bullish", style={'color': '#00ff88', 'marginBottom': '5px'}),
                html.P("Confidence: 78%", className='positive', style={'margin': '0'})
            ], className='card', style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
            html.Div([
                html.Div("⚡", className='metric-icon'),
                html.H3("Volatility", style={'color': '#a0a0a0', 'marginBottom': '10px'}),
                html.H1("15.4%", style={'color': '#ff6b6b', 'marginBottom': '5px'}),
                html.P("VIX Equivalent", className='negative', style={'margin': '0'})
            ], className='card', style={'width': '23%', 'display': 'inline-block'})
        ], style={'marginBottom': '50px'}),
        html.Div([
            html.Div([
                dcc.Graph(
                    id='line-graph',
                    figure=px.line(
                        mock_data,
                        x='Date', y='Close',
                        title="AAPL Stock Price Forecast (with Confidence Bands)",
                        labels={'Close': 'Price ($)', 'Date': 'Date'}
                    ).add_hline(y=mock_data['Close'].mean(), line_dash="dash", line_color="gray", annotation_text="Mean Price")
                      .update_traces(line_color='#00d2ff', line_width=2)
                      .update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white', family='Inter'),
                        title_font=dict(size=16, family='Inter'),
                        margin=dict(l=20, r=20, t=40, b=20),
                        hovermode='x unified'
                    )
                )
            ], className='card', style={'marginBottom': '30px'})
        ])
    ])
)

# ---------------------------
# PAGE 2 — ANALYTICS (Enhanced with more options, sliders, and smooth updates)
# ---------------------------
dash.register_page(
    "analytics",
    path="/analytics",
    title="Analytics",
    layout=html.Div([
        html.H1("📊 Interactive Analytics Suite",
                style={'textAlign': 'center', 'marginBottom': '40px', 'fontWeight': '700'}),
        html.Div([
            html.Div([
                dcc.Dropdown(
                    id='chart-type',
                    options=[
                        {'label': 'Bar Chart', 'value': 'bar'},
                        {'label': 'Scatter Plot', 'value': 'scatter'},
                        {'label': 'Pie Chart', 'value': 'pie'},
                        {'label': 'Line Chart', 'value': 'line'}
                    ],
                    value='bar',
                    clearable=False,
                    className='dropdown'
                ),
                dcc.Slider(
                    id='data-range',
                    min=10,
                    max=100,
                    step=10,
                    value=50,
                    marks={i: f'{i}%' for i in range(10, 101, 20)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                html.Div(id='loading-indicator', children=[html.Span(className='loading', style={'marginLeft': '10px'})], style={'display': 'none'})
            ], style={'marginBottom': '30px'}),
            dcc.Graph(id='chart-display', style={'marginTop': '20px'})
        ], className='card', style={'padding': '30px'})
    ])
)

@app.callback(
    [Output('chart-display', 'figure'),
     Output('loading-indicator', 'style')],
    [Input('chart-type', 'value'),
     Input('data-range', 'value')]
)
def update_chart(chart_type, range_val):
    # Simulate loading delay for smoothness demo - but since callback is sync, show briefly or not
    df_size = int(4 * (range_val / 100 * 25))  # Scale to 4-100 items
    df = pd.DataFrame({
        "Category": [chr(65 + i) for i in range(df_size)],
        "Value": np.random.randint(10, 100, df_size),
        "Group": np.random.choice(['Tech', 'Finance', 'Health'], df_size)
    })
    
    if chart_type == 'bar':
        fig = px.bar(df, x='Category', y='Value', color='Group', title=f'Bar Chart (Data Range: {range_val}%)')
    elif chart_type == 'scatter':
        fig = px.scatter(df, x='Category', y='Value', size='Value', color='Group', title=f'Scatter Plot (Data Range: {range_val}%)')
    elif chart_type == 'pie':
        fig = px.pie(df.head(6), names='Category', values='Value', title=f'Pie Chart (Top 6, Range: {range_val}%)')
    else:  # line
        fig = px.line(df, x='Category', y='Value', color='Group', title=f'Line Chart (Data Range: {range_val}%)')
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Inter'),
        title_font=dict(size=16, family='Inter'),
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode='closest'
    )
    return fig, {'display': 'none'}

# ---------------------------
# PAGE 3 — ABOUT (Enhanced with more engaging content and links)
# ---------------------------
dash.register_page(
    "about",
    path="/about",
    title="About",
    layout=html.Div([
        html.H1("ℹ️ About Stock Intelligence Demo",
                style={'textAlign': 'center', 'marginBottom': '40px', 'fontWeight': '700'}),
        html.Div([
            html.Div([
                html.H3("🚀 What is This?", style={'color': '#00d2ff', 'marginBottom': '15px'}),
                html.P("""
                    This is an advanced multi-page Dash application showcasing a sleek, 
                    glassmorphism-inspired dark theme for financial analytics. Built with Plotly Dash, 
                    it demonstrates smooth client-side routing, interactive visualizations, and 
                    responsive design. All data is mock-generated for demo purposes, simulating 
                    real-time stock insights like portfolio yields, market regimes, and forecasts.
                """, style={'fontSize': '16px', 'lineHeight': '1.6', 'color': '#ccc', 'marginBottom': '20px'}),
                
                html.H3("✨ Features", style={'color': '#00d2ff', 'marginBottom': '15px'}),
                html.Ul([
                    html.Li("Ultra-smooth page transitions with CSS animations.", style={'color': '#aaa', 'marginBottom': '5px'}),
                    html.Li("Interactive charts with dynamic updates and loading states.", style={'color': '#aaa', 'marginBottom': '5px'}),
                    html.Li("Glassmorphism cards with hover effects and gradients.", style={'color': '#aaa', 'marginBottom': '5px'}),
                    html.Li("Stock-like mock data generation for realistic demos.", style={'color': '#aaa', 'marginBottom': '5px'}),
                    html.Li("Custom navbar with active states and shimmer effects.", style={'color': '#aaa'})
                ], style={'fontSize': '15px', 'lineHeight': '1.6', 'color': '#ccc'}),
                
                html.Div([
                    html.A("View Source on GitHub", href="https://github.com", target="_blank", 
                           style={'color': '#00d2ff', 'textDecoration': 'none', 'fontWeight': '600'}),
                    html.P(" | Built with ❤️ using Dash & Plotly", style={'textAlign': 'center', 'color': '#888', 'fontSize': '14px', 'marginTop': '20px'})
                ], style={'textAlign': 'center'})
            ], style={'padding': '30px'})
        ], className='card')
    ])
)

# Main layout with navbar included for reactivity
app.layout = html.Div([
    html.Div(className="navbar", children=[
        html.A(href="/", id="nav-home", className="nav-link", children="🏠 Dashboard"),
        html.A(href="/analytics", id="nav-analytics", className="nav-link", children="📊 Analytics"),
        html.A(href="/about", id="nav-about", className="nav-link", children="ℹ️ About")
    ]),
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content", className="page-content")
])

# Global callback for active nav highlighting and page animations
@app.callback(
    [Output("page-content", "children"),
     Output("nav-home", "className"),
     Output("nav-analytics", "className"),
     Output("nav-about", "className")],
    [Input("url", "pathname")]
)
def display_page(pathname):
    if pathname == "/":
        page_layout = dash.page_registry["home"]["layout"]
        home_class = "nav-link active"
        analytics_class = "nav-link"
        about_class = "nav-link"
    elif pathname == "/analytics":
        page_layout = dash.page_registry["analytics"]["layout"]
        home_class = "nav-link"
        analytics_class = "nav-link active"
        about_class = "nav-link"
    elif pathname == "/about":
        page_layout = dash.page_registry["about"]["layout"]
        home_class = "nav-link"
        analytics_class = "nav-link"
        about_class = "nav-link active"
    else:
        page_layout = html.Div("404: Page Not Found", style={'textAlign': 'center', 'padding': '50px'})
        home_class = "nav-link"
        analytics_class = "nav-link"
        about_class = "nav-link"
    
    # Wrap in div with key for animation trigger
    return html.Div(page_layout, key=pathname or "home", id="dynamic-content"), home_class, analytics_class, about_class

# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True, port=8050, dev_tools_hot_reload=True)