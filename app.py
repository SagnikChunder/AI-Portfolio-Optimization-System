import math
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

# -----------------------------
# Configuration / Universe
# -----------------------------
STOCK_UNIVERSE = {
    "Technology": ["AAPL", "MSFT", "NVDA", "ADBE", "INTC"],
    "Finance": ["JPM", "BAC", "C", "WFC", "GS"],
    "Energy": ["XOM", "CVX", "BP", "TTE", "COP"],  # Changed TOT to TTE (TotalEnergies ticker change)
    "Consumer": ["AMZN", "WMT", "PG", "KO", "MCD"]
}

# -----------------------------
# Utility: fetch market data
# -----------------------------
def get_market_data(tickers, period="1y", interval="1d"):
    """
    Download adjusted close prices for given tickers using yfinance.
    Returns DataFrame indexed by date with columns equal to tickers.
    """
    if not tickers:
        return pd.DataFrame()

    try:
        # Download data
        df = yf.download(tickers, period=period, interval=interval, progress=False)
        
        # Handle the MultiIndex columns returned by new yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            # If 'Adj Close' exists in the top level, use it
            if 'Adj Close' in df.columns.get_level_values(0):
                df = df['Adj Close']
            elif 'Close' in df.columns.get_level_values(0):
                df = df['Close']
        else:
            # Handle single ticker flat DataFrame
            if 'Adj Close' in df.columns:
                df = df[['Adj Close']]
            elif 'Close' in df.columns:
                df = df[['Close']]
                
    except Exception as e:
        print(f"yfinance download error: {e}")
        return pd.DataFrame()

    # If single ticker, df might be a Series, convert to DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame()
        # Rename column to the single ticker if necessary
        if len(tickers) == 1:
            df.columns = tickers

    # Clean data
    df = df.dropna(axis=1, how="all") # Drop tickers with no data
    df = df.ffill().dropna()          # Forward fill and drop remaining NaNs

    return df

# -----------------------------
# Monte Carlo optimizer
# -----------------------------
def monte_carlo_simulation(tickers, num_portfolios=5000, risk_tolerance=0.2, seed=42):
    np.random.seed(int(seed))

    if not tickers:
        return None, None, None, None, None

    # Cap tickers to prevent timeout on free tiers
    if len(tickers) > 12:
        tickers = tickers[:12]

    data = get_market_data(tickers)
    
    if data.empty or len(data.columns) < 2:
        return None, None, None, None, None

    # Calculate Daily Returns
    returns = data.pct_change().dropna()
    if returns.empty:
        return None, None, None, None, None

    # Annualize stats
    trading_days = 252
    mean_returns = returns.mean().values * trading_days
    cov_matrix = returns.cov().values * trading_days
    num_assets = len(data.columns) # Use actual columns from data, not input tickers

    # Validate num_portfolios
    try:
        num_portfolios = int(num_portfolios)
    except:
        num_portfolios = 5000
    num_portfolios = max(100, min(num_portfolios, 20000))

    # Generate Random Weights
    weights = np.random.random((num_portfolios, num_assets))
    weights /= np.sum(weights, axis=1)[:, np.newaxis]

    # Portfolio Metrics
    port_returns = np.dot(weights, mean_returns)
    port_variance = np.sum(np.dot(weights, cov_matrix) * weights, axis=1)
    port_std = np.sqrt(np.maximum(port_variance, 0)) # Ensure non-negative before sqrt

    # Avoid divide by zero
    port_std = np.where(port_std == 0, 1e-9, port_std)

    risk_free = 0.04 
    sharpe_ratios = (port_returns - risk_free) / port_std

    # Construct Results DataFrame
    results = {
        "Return": port_returns,
        "Volatility": port_std,
        "Sharpe Ratio": sharpe_ratios
    }
    
    # Add weights to results
    # data.columns ensures we map weights to the correct valid tickers
    for i, ticker in enumerate(data.columns):
        results[f"{ticker}_Weight"] = weights[:, i]

    df = pd.DataFrame(results)

    # Filter by Risk Tolerance
    try:
        risk_tolerance = float(risk_tolerance)
    except:
        risk_tolerance = 0.2

    valid_risk = df[df["Volatility"] <= risk_tolerance]
    
    if not valid_risk.empty:
        optimal_idx = valid_risk["Sharpe Ratio"].idxmax()
        optimal_portfolio = valid_risk.loc[optimal_idx]
    else:
        # Fallback if no portfolio meets risk criteria
        optimal_idx = df["Sharpe Ratio"].idxmax()
        optimal_portfolio = df.loc[optimal_idx]

    return df, optimal_portfolio, returns, data, list(data.columns)

# -----------------------------
# Build Dash app layout
# -----------------------------
app = Dash(__name__)
# EXPOSE SERVER FOR DEPLOYMENT (Required by Gunicorn/Render/Heroku)
server = app.server 

app.title = "Monte Carlo Portfolio Optimizer"

sector_options = [{"label": s, "value": s} for s in STOCK_UNIVERSE.keys()]

app.layout = html.Div([
    html.H2("Monte Carlo Portfolio Optimizer", style={"textAlign": "center"}),

    html.Div([
        html.Div([
            html.Label("Select Sectors"),
            dcc.Checklist(
                id="sector-selection", 
                options=sector_options, 
                value=["Technology"],
                style={"display": "flex", "flexDirection": "column"}
            ),

            html.Br(),
            html.Label("Risk Tolerance (annual volatility)"),
            dcc.Slider(id="risk-tolerance", min=0.01, max=1.0, step=0.01, value=0.2,
                       marks={0.05: "5%", 0.2: "20%", 0.4: "40%", 0.6: "60%"}),

            html.Br(),
            html.Label("Number of Portfolios (Simulations)"),
            dcc.Input(id="num-portfolios", type="number", value=4000, min=100, max=20000, step=100, style={"width": "100%"}),

            html.Br(), html.Br(),
            html.Label("Investment Amount ($)"),
            dcc.Input(id="investment-amount", type="number", value=100000, min=0, step=1000, style={"width": "100%"}),

            html.Br(), html.Br(),
            html.Button("Optimize", id="optimize-button", n_clicks=0,
                        style={"backgroundColor": "#3498db", "color": "white", "padding": "10px", "width": "100%", "border": "none", "cursor": "pointer"}),

            html.Div(id="status-output", style={"marginTop": "10px"})
        ], style={"width": "25%", "display": "inline-block", "verticalAlign": "top", 
                  "padding": "20px", "boxShadow": "0 2px 6px rgba(0,0,0,0.1)", "backgroundColor": "#f9f9f9", "borderRadius": "5px"}),

        html.Div([
            html.Div(id="summary-output"),
            html.H4("Optimal Allocation"),
            html.Div(id="allocation-table"),
        ], style={"width": "70%", "display": "inline-block", "padding": "20px", "verticalAlign": "top", "marginLeft": "2%"})
    ], style={"display": "flex", "flexWrap": "wrap"}),

    html.Hr(),

    html.Div([
        dcc.Graph(id="frontier-plot", style={"width": "48%", "display": "inline-block"}),
        dcc.Graph(id="correlation-plot", style={"width": "48%", "display": "inline-block", "float": "right"})
    ]),

    html.Div([dcc.Graph(id="performance-plot")]),

    html.Footer("Generated by Monte Carlo Optimizer", style={"textAlign": "center", "marginTop": "20px", "color": "#7f8c8d"})
], style={"fontFamily": "Arial, sans-serif", "margin": "20px", "maxWidth": "1200px", "marginLeft": "auto", "marginRight": "auto"})

# -----------------------------
# Callback: optimize portfolio
# -----------------------------
@app.callback(
    [Output("summary-output", "children"),
     Output("allocation-table", "children"),
     Output("frontier-plot", "figure"),
     Output("correlation-plot", "figure"),
     Output("performance-plot", "figure"),
     Output("status-output", "children"),
     Output("status-output", "style")],
    [Input("optimize-button", "n_clicks")],
    [State("sector-selection", "value"),
     State("risk-tolerance", "value"),
     State("num-portfolios", "value"),
     State("investment-amount", "value")]
)
def optimize_portfolio(n_clicks, selected_sectors, risk_tolerance, num_portfolios, investment_amount):
    empty_fig = go.Figure()
    
    # Don't run on initial load
    if not n_clicks:
        raise PreventUpdate

    if not selected_sectors:
        status_msg = "Please select at least one sector."
        status_style = {"backgroundColor": "#e74c3c", "color": "white", "padding": "10px", "borderRadius": "3px"}
        return "", "", empty_fig, empty_fig, empty_fig, status_msg, status_style

    # Build ticker list
    all_tickers = []
    for sector in selected_sectors:
        all_tickers.extend(STOCK_UNIVERSE.get(sector, []))
    all_tickers = list(dict.fromkeys(all_tickers)) # remove duplicates

    # Validate inputs
    try:
        num_portfolios = int(num_portfolios) if num_portfolios else 4000
        investment_amount = float(investment_amount) if investment_amount else 0.0
        risk_tolerance = float(risk_tolerance) if risk_tolerance else 0.2
    except ValueError:
        return "", "", empty_fig, empty_fig, empty_fig, "Invalid input numbers", {}

    # Run Simulation
    try:
        df, optimal_portfolio, returns, data, used_tickers = monte_carlo_simulation(
            all_tickers, num_portfolios, risk_tolerance
        )
    except Exception as e:
        print(f"Simulation Critical Error: {e}")
        df = None

    if df is None or data is None or data.empty:
        status_msg = "Simulation failed. Could not fetch data or insufficient data points."
        status_style = {"backgroundColor": "#e74c3c", "color": "white", "padding": "10px", "borderRadius": "3px"}
        return "", "", empty_fig, empty_fig, empty_fig, status_msg, status_style

    # 1. Summary
    summary = html.Div([
        html.H3("Results", style={"color": "#2c3e50", "marginTop": "0"}),
        html.Div([
            html.P([html.Strong("Expected Annual Return: "), f"{optimal_portfolio['Return']:.2%}"]),
            html.P([html.Strong("Portfolio Volatility: "), f"{optimal_portfolio['Volatility']:.2%}"]),
            html.P([html.Strong("Sharpe Ratio: "), f"{optimal_portfolio['Sharpe Ratio']:.2f}"]),
            html.P([html.Strong("Total Investment: "), f"${investment_amount:,.2f}"]),
        ], style={"backgroundColor": "#ecf0f1", "padding": "15px", "borderRadius": "5px"})
    ])

    # 2. Allocation Table
    allocation_rows = []
    current_prices = data.iloc[-1]

    for ticker in used_tickers:
        weight = float(optimal_portfolio.get(f"{ticker}_Weight", 0.0))
        if weight > 0.001:  # Show only > 0.1% allocation
            price = float(current_prices.get(ticker, 0))
            alloc_value = weight * investment_amount
            shares = int(alloc_value / price) if price > 0 else 0
            
            allocation_rows.append({
                "Ticker": ticker,
                "Weight": weight, # Keep raw for sorting
                "Allocation": alloc_value,
                "Price": price,
                "Shares": shares
            })

    alloc_df = pd.DataFrame(allocation_rows).sort_values("Weight", ascending=False)
    
    # Format for display
    display_data = alloc_df.copy()
    display_data['Weight'] = display_data['Weight'].apply(lambda x: f"{x:.2%}")
    display_data['Allocation'] = display_data['Allocation'].apply(lambda x: f"${x:,.2f}")
    display_data['Price'] = display_data['Price'].apply(lambda x: f"${x:.2f}")

    allocation_table = dash_table.DataTable(
        data=display_data.to_dict("records"),
        columns=[
            {"name": "Ticker", "id": "Ticker"},
            {"name": "Weight", "id": "Weight"},
            {"name": "Allocation Value", "id": "Allocation"},
            {"name": "Current Price", "id": "Price"},
            {"name": "Shares", "id": "Shares"}
        ],
        style_cell={"textAlign": "left", "padding": "10px", "fontFamily": "Arial"},
        style_header={"backgroundColor": "#3498db", "color": "white", "fontWeight": "bold"},
        style_data_conditional=[
            {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}
        ]
    )

    # 3. Efficient Frontier Plot
    frontier_fig = px.scatter(
        df, x="Volatility", y="Return", color="Sharpe Ratio",
        title="Efficient Frontier", hover_data=["Sharpe Ratio"],
        color_continuous_scale="Viridis"
    )
    frontier_fig.add_trace(go.Scatter(
        x=[optimal_portfolio["Volatility"]],
        y=[optimal_portfolio["Return"]],
        mode="markers",
        marker=dict(size=14, color="red", symbol="star", line=dict(width=2, color="black")),
        name="Optimal Portfolio"
    ))
    frontier_fig.update_layout(template="plotly_white")

    # 4. Correlation Plot
    corr_matrix = returns.corr()
    corr_fig = px.imshow(
        corr_matrix, 
        text_auto=".2f",
        aspect="auto",
        title="Correlation Matrix",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1
    )
    corr_fig.update_layout(template="plotly_white")

    # 5. Performance Plot (Normalized)
    normalized_prices = data / data.iloc[0] * 100
    perf_fig = px.line(normalized_prices, title="Historical Price Evolution (Normalized to 100)")
    perf_fig.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="Normalized Price")

    status_msg = "Optimization Complete."
    status_style = {"backgroundColor": "#2ecc71", "color": "white", "padding": "10px", "borderRadius": "3px"}

    return summary, allocation_table, frontier_fig, corr_fig, perf_fig, status_msg, status_style

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    # Debug=False is better for Docker/Deployment to avoid reloader issues
    # host='0.0.0.0' is required for containerized environments
    app.run_server(debug=True, host='0.0.0.0', port=8050))
