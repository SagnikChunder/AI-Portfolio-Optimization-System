# dash_portfolio_app.py
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
    "Energy": ["XOM", "CVX", "BP", "TOT", "COP"],
    "Consumer": ["AMZN", "WMT", "PG", "KO", "MCD"]
}

# -----------------------------
# Utility: fetch market data
# -----------------------------
def get_market_data(tickers, period="1y", interval="1d"):
    """
    Download adjusted close prices for given tickers using yfinance.
    Returns DataFrame indexed by date with columns equal to tickers.
    Returns empty DataFrame on failure.
    """
    if not tickers:
        return pd.DataFrame()

    # yfinance accepts list or comma separated string
    try:
        df = yf.download(tickers, period=period, interval=interval, progress=False)
        # select Adjusted Close when available
        if "Adj Close" in df.columns:
            df = df["Adj Close"]
        elif ("Adj Close",) == tuple(df.columns[:1]):
            # fallback unlikely
            df = df.iloc[:, 0]
    except Exception as e:
        print("yfinance download error:", e)
        return pd.DataFrame()

    # If single ticker, make sure DataFrame has 1 column
    if isinstance(df, pd.Series):
        df = df.to_frame()

    # Drop columns that are completely NaN
    df = df.dropna(axis=1, how="all")

    # Forward-fill small gaps and drop leading NaNs
    df = df.ffill().dropna()

    return df

# -----------------------------
# Monte Carlo optimizer
# -----------------------------
def monte_carlo_simulation(tickers, num_portfolios=5000, risk_tolerance=0.2, seed=42):
    """
    Returns:
      df (DataFrame) : simulated portfolios with Return, Volatility, Sharpe Ratio and weights
      optimal_portfolio (Series) : the selected optimal portfolio row from df
      returns (DataFrame) : daily returns used
      data (DataFrame) : price data used
      tickers_used (list) : actual tickers used
    On failure returns (None,)*5
    """
    np.random.seed(int(seed))

    if not tickers:
        return None, None, None, None, None

    # cap tickers for performance
    if len(tickers) > 12:
        tickers = tickers[:12]

    data = get_market_data(tickers)
    if data.empty or len(data.columns) < 2:
        return None, None, None, None, None

    returns = data.pct_change().dropna()
    if returns.empty:
        return None, None, None, None, None

    trading_days = 252
    mean_returns = returns.mean().values * trading_days
    cov_matrix = returns.cov().values * trading_days
    num_assets = len(tickers)

    # Validate num_portfolios
    try:
        num_portfolios = int(num_portfolios)
    except Exception:
        num_portfolios = 5000
    num_portfolios = max(100, min(num_portfolios, 20000))

    # Random weights generation
    weights = np.random.random((num_portfolios, num_assets))
    weights /= np.sum(weights, axis=1)[:, np.newaxis]

    port_returns = np.dot(weights, mean_returns)
    port_variance = np.sum(np.dot(weights, cov_matrix) * weights, axis=1)
    port_std = np.sqrt(np.maximum(port_variance, 0))

    # avoid divide by zero
    port_std = np.where(port_std == 0, 1e-9, port_std)

    risk_free = 0.04  # can be parameterized
    sharpe_ratios = (port_returns - risk_free) / port_std

    results = {
        "Return": port_returns,
        "Volatility": port_std,
        "Sharpe Ratio": sharpe_ratios
    }
    for i, ticker in enumerate(tickers):
        results[f"{ticker}_Weight"] = weights[:, i]

    df = pd.DataFrame(results)

    # Ensure risk_tolerance is float
    try:
        risk_tolerance = float(risk_tolerance)
    except Exception:
        risk_tolerance = 0.2

    valid_risk = df[df["Volatility"] <= risk_tolerance]
    if not valid_risk.empty:
        optimal_idx = valid_risk["Sharpe Ratio"].idxmax()
        optimal_portfolio = valid_risk.loc[optimal_idx]
    else:
        optimal_idx = df["Sharpe Ratio"].idxmax()
        optimal_portfolio = df.loc[optimal_idx]

    return df, optimal_portfolio, returns, data, tickers

# -----------------------------
# Build Dash app layout
# -----------------------------
app = Dash(__name__)
app.title = "Monte Carlo Portfolio Optimizer"

sector_options = [{"label": s, "value": s} for s in STOCK_UNIVERSE.keys()]

app.layout = html.Div([
    html.H2("Monte Carlo Portfolio Optimizer", style={"textAlign": "center"}),

    html.Div([
        html.Div([
            html.Label("Select Sectors"),
            dcc.Checklist(id="sector-selection", options=sector_options, value=["Technology"]),

            html.Br(),
            html.Label("Risk Tolerance (annual volatility)"),
            dcc.Slider(id="risk-tolerance", min=0.01, max=1.0, step=0.01, value=0.2,
                       marks={0.05: "5%", 0.2: "20%", 0.4: "40%", 0.6: "60%"}),

            html.Br(),
            html.Label("Number of Portfolios (Monte Carlo simulations)"),
            dcc.Input(id="num-portfolios", type="number", value=4000, min=100, max=20000, step=100),

            html.Br(),
            html.Label("Investment Amount"),
            dcc.Input(id="investment-amount", type="number", value=100000, min=0, step=1000),

            html.Br(), html.Br(),
            html.Button("Optimize", id="optimize-button", n_clicks=0,
                        style={"backgroundColor": "#3498db", "color": "white", "padding": "10px"}),

            html.Div(id="status-output", style={"marginTop": "10px"})
        ], style={"width": "28%", "display": "inline-block", "verticalAlign": "top",
                  "padding": "20px", "boxShadow": "0 2px 6px rgba(0,0,0,0.1)"}),

        html.Div([
            html.Div(id="summary-output"),
            html.H4("Allocation"),
            html.Div(id="allocation-table"),
        ], style={"width": "70%", "display": "inline-block", "padding": "20px", "verticalAlign": "top"})
    ]),

    html.Hr(),

    html.Div([
        dcc.Graph(id="frontier-plot", style={"width": "49%", "display": "inline-block"}),
        dcc.Graph(id="correlation-plot", style={"width": "49%", "display": "inline-block"})
    ]),

    html.Div([dcc.Graph(id="performance-plot")]),

    html.Footer("Generated by Monte Carlo Optimizer", style={"textAlign": "center", "marginTop": "20px", "color": "#7f8c8d"})
], style={"fontFamily": "Arial, sans-serif", "margin": "20px"})

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

    # don't run on initial load
    if not n_clicks or n_clicks == 0:
        raise PreventUpdate

    if not selected_sectors:
        status_msg = "Select at least one sector"
        status_style = {"backgroundColor": "#e74c3c", "color": "white", "padding": "10px"}
        return "", "", empty_fig, empty_fig, empty_fig, status_msg, status_style

    # Build ticker list (unique, keep order)
    all_tickers = []
    for sector in selected_sectors:
        all_tickers.extend(STOCK_UNIVERSE.get(sector, []))
    all_tickers = list(dict.fromkeys(all_tickers))

    # Validate inputs safely
    try:
        num_portfolios = int(num_portfolios) if num_portfolios is not None else 4000
    except Exception:
        num_portfolios = 4000
    try:
        investment_amount = float(investment_amount) if investment_amount is not None else 0.0
    except Exception:
        investment_amount = 0.0
    try:
        risk_tolerance = float(risk_tolerance) if risk_tolerance is not None else 0.2
    except Exception:
        risk_tolerance = 0.2

    # Run simulation inside try/except for clearer error reporting
    try:
        df, optimal_portfolio, returns, data, used_tickers = monte_carlo_simulation(
            all_tickers, num_portfolios, risk_tolerance
        )
    except Exception as e:
        print("Simulation error:", e)
        df = optimal_portfolio = returns = data = used_tickers = None

    if df is None or optimal_portfolio is None or returns is None or data is None or not used_tickers:
        status_msg = ("Data fetch or simulation failed. "
                      "Possible causes: insufficient data for chosen tickers/period, network error, or too small universe. "
                      "Try selecting fewer sectors, increasing the data period, or reducing number of portfolios.")
        status_style = {"backgroundColor": "#e74c3c", "color": "white", "padding": "10px"}
        return "", "", empty_fig, empty_fig, empty_fig, status_msg, status_style

    # Build summary
    try:
        summary = html.Div([
            html.H3("Portfolio Summary", style={"color": "#2c3e50"}),
            html.P([html.Strong("Expected Annual Return: "), f"{optimal_portfolio['Return']:.2%}"]),
            html.P([html.Strong("Portfolio Volatility: "), f"{optimal_portfolio['Volatility']:.2%}"]),
            html.P([html.Strong("Sharpe Ratio: "), f"{optimal_portfolio['Sharpe Ratio']:.2f}"]),
            html.P([html.Strong("Total Investment: "), f"{investment_amount:,.2f}"]),
        ])
    except Exception:
        summary = html.Div([html.P("Error building summary.")])

    # Allocation table (use numeric columns for correct sorting)
    allocation_rows = []
    current_prices = data.iloc[-1]

    for ticker in used_tickers:
        weight = float(optimal_portfolio.get(f"{ticker}_Weight", 0.0))
        if weight > 0.001:  # threshold to show only meaningful allocations
            price = current_prices.get(ticker, np.nan)
            try:
                price = float(price)
            except Exception:
                price = float("nan")
            allocation = weight * float(investment_amount)
            shares = int(allocation / price) if (price and not math.isnan(price) and price > 0) else 0

            allocation_rows.append({
                "Ticker": ticker,
                "Weight": round(weight * 100, 2),
                "Allocation": round(allocation, 2),
                "Price": round(price, 2) if not math.isnan(price) else None,
                "Shares": shares
            })

    if allocation_rows:
        allocation_df = pd.DataFrame(allocation_rows).sort_values("Allocation", ascending=False)

        allocation_table = dash_table.DataTable(
            data=allocation_df.to_dict("records"),
            columns=[
                {"name": "Ticker", "id": "Ticker"},
                {"name": "Weight (%)", "id": "Weight", "type": "numeric"},
                {"name": "Allocation", "id": "Allocation", "type": "numeric"},
                {"name": "Price", "id": "Price", "type": "numeric"},
                {"name": "Shares", "id": "Shares", "type": "numeric"}
            ],
            sort_action="native",
            style_cell={"textAlign": "left", "padding": "8px"},
            style_header={"backgroundColor": "#3498db", "color": "white"},
            page_size=10
        )
    else:
        allocation_table = html.P("No significant weights found for this configuration.")

    # Efficient frontier plot
    frontier_fig = px.scatter(
        df, x="Volatility", y="Return", color="Sharpe Ratio",
        title="Efficient Frontier", hover_data=["Sharpe Ratio"]
    )
    frontier_fig.add_trace(
        go.Scatter(
            x=[optimal_portfolio["Volatility"]],
            y=[optimal_portfolio["Return"]],
            mode="markers",
            marker=dict(size=15, color="red", symbol="star"),
            name="Optimal Portfolio"
        )
    )

    # Correlation heatmap
    try:
        corr_fig = px.imshow(
            returns.corr(), aspect="auto",
            title="Stock Correlation Matrix",
            color_continuous_scale="RdBu_r"
        )
    except Exception:
        corr_fig = go.Figure(data=go.Heatmap(z=returns.corr().values, x=returns.columns, y=returns.columns))
        corr_fig.update_layout(title="Stock Correlation Matrix")

    # Performance: normalized price evolution
    normalized_prices = data / data.iloc[0] * 100
    perf_fig = px.line(normalized_prices, title="Price Evolution (Normalized)")

    status_msg = "Success. Portfolio optimized."
    status_style = {"backgroundColor": "#2ecc71", "color": "white", "padding": "10px"}

    return summary, allocation_table, frontier_fig, corr_fig, perf_fig, status_msg, status_style

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
