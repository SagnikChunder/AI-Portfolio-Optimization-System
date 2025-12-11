import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# -----------------------------------------------------------
# Data and helper functions
# -----------------------------------------------------------

STOCK_UNIVERSE = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"],
    "Finance": ["JPM", "BAC", "WFC", "GS", "MS", "C"],
    "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "Consumer": ["PG", "KO", "PEP", "WMT", "HD", "MCD"],
    "Industrial": ["BA", "CAT", "GE", "MMM", "UPS"],
}


def get_market_data(tickers, period="1y"):
    """Fetch market data (Adj Close or Close) for given tickers."""
    if not tickers:
        return pd.DataFrame()

    try:
        data = yf.download(tickers, period=period, progress=False, threads=True)
        if data.empty:
            return pd.DataFrame()

        # Single ticker: yf returns a Series-like DataFrame
        if len(tickers) == 1:
            col = "Adj Close" if "Adj Close" in data.columns else "Close"
            return pd.DataFrame({tickers[0]: data[col]})

        # Multiple tickers: yf returns multi-index columns
        if "Adj Close" in data.columns:
            return data["Adj Close"]
        elif "Close" in data.columns:
            return data["Close"]
        else:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def monte_carlo_simulation(tickers, num_portfolios=5000, risk_tolerance=0.2):
    """Simple Monte Carlo portfolio optimization."""
    if len(tickers) < 2:
        return None, None, None, None

    # Limit to keep things light
    tickers = list(tickers)[:8]

    data = get_market_data(tickers, period="1y")
    if data.empty:
        return None, None, None, None

    data = data.dropna()
    if len(data) < 100:
        return None, None, None, None

    returns = data.pct_change().dropna()
    if returns.empty:
        return None, None, None, None

    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    if np.any(~np.isfinite(cov_matrix.values)):
        return None, None, None, None

    results = np.zeros((num_portfolios, 3 + len(tickers)))

    for i in range(num_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)

        port_ret = np.sum(mean_returns * weights)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = 0.0 if port_vol == 0 else (port_ret - 0.04) / port_vol

        results[i, 0] = port_ret
        results[i, 1] = port_vol
        results[i, 2] = sharpe
        results[i, 3:] = weights

    cols = ["Return", "Volatility", "Sharpe"] + [f"{t}_Weight" for t in tickers]
    df = pd.DataFrame(results, columns=cols)

    df = df[(df["Volatility"] > 0) & (df["Volatility"] < 1) & np.isfinite(df["Sharpe"])]
    if df.empty:
        return None, None, None, None

    filtered = df[df["Volatility"] <= risk_tolerance]
    if not filtered.empty:
        opt = filtered.loc[filtered["Sharpe"].idxmax()]
    else:
        opt = df.loc[df["Sharpe"].idxmax()]

    return df, opt, returns, data


def make_frontier_figure(df, optimal):
    fig = go.Figure()
    if df is None or df.empty:
        fig.update_layout(title="Efficient Frontier (no data)")
        return fig

    fig.add_trace(
        go.Scatter(
            x=df["Volatility"],
            y=df["Return"],
            mode="markers",
            marker=dict(
                size=4,
                color=df["Sharpe"],
                colorscale="Viridis",
                colorbar=dict(title="Sharpe"),
            ),
            name="Portfolios",
        )
    )

    if optimal is not None:
        fig.add_trace(
            go.Scatter(
                x=[optimal["Volatility"]],
                y=[optimal["Return"]],
                mode="markers",
                marker=dict(size=14, color="red", symbol="star"),
                name="Optimal",
            )
        )

    fig.update_layout(
        title="Efficient Frontier",
        xaxis_title="Volatility (Risk)",
        yaxis_title="Expected Return",
        height=450,
    )
    return fig


def make_allocation_figure(optimal, tickers):
    fig = go.Figure()
    if optimal is None:
        fig.update_layout(title="Allocation (no data)")
        return fig

    labels = []
    values = []
    for t in tickers:
        col = f"{t}_Weight"
        if col in optimal and optimal[col] > 0.01:
            labels.append(t)
            values.append(optimal[col])

    if not values:
        fig.update_layout(title="Allocation (no significant weights)")
        return fig

    fig.add_trace(go.Pie(labels=labels, values=values, hole=0.3))
    fig.update_layout(title="Optimal Portfolio Allocation", height=450)
    return fig


def make_correlation_figure(returns, tickers):
    fig = go.Figure()
    if returns is None or returns.empty:
        fig.update_layout(title="Correlation (no data)")
        return fig

    corr = returns.corr()
    tickers_in_data = [t for t in tickers if t in corr.columns]

    fig.add_trace(
        go.Heatmap(
            z=corr.loc[tickers_in_data, tickers_in_data].values,
            x=tickers_in_data,
            y=tickers_in_data,
            colorscale="RdBu",
            zmid=0,
        )
    )
    fig.update_layout(title="Stock Correlation Matrix", height=500)
    return fig


def make_allocation_table(optimal, tickers, investment):
    if optimal is None or investment is None or investment <= 0:
        return pd.DataFrame()

    rows = []
    for t in tickers:
        col = f"{t}_Weight"
        w = float(optimal[col]) if col in optimal else 0.0
        alloc = w * investment

        price = 100.0
        shares = 0
        try:
            hist = yf.Ticker(t).history(period="5d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        except Exception:
            price = 100.0
        if price > 0:
            shares = int(alloc // price)

        rows.append(
            {
                "Ticker": t,
                "Weight (%)": f"{w * 100:.2f}",
                "Allocation ($)": f"{alloc:,.2f}",
                "Price ($)": f"{price:.2f}",
                "Shares": shares,
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------
# Dash app
# -----------------------------------------------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        html.Br(),
        html.H2("AI Portfolio Management (Simplified)", className="text-center"),
        html.Hr(),

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4("Configuration"),
                        html.Label("Sectors"),
                        dcc.Checklist(
                            id="sector-select",
                            options=[
                                {"label": s, "value": s} for s in STOCK_UNIVERSE.keys()
                            ],
                            value=["Technology", "Finance"],
                            labelStyle={"display": "block"},
                        ),
                        html.Br(),
                        html.Label("Risk tolerance (max volatility)"),
                        dcc.Slider(
                            id="risk-tolerance",
                            min=0.05,
                            max=0.5,
                            step=0.01,
                            value=0.2,
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Br(),
                        html.Label("Number of simulations"),
                        dcc.Slider(
                            id="num-portfolios",
                            min=1000,
                            max=20000,
                            step=1000,
                            value=5000,
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Br(),
                        html.Label("Investment amount ($)"),
                        dcc.Input(
                            id="investment-amount",
                            type="number",
                            value=10000,
                            min=1000,
                            step=500,
                            style={"width": "100%"},
                        ),
                        html.Br(),
                        html.Br(),
                        dbc.Button(
                            "Run optimization",
                            id="run-btn",
                            color="primary",
                            className="w-100",
                        ),
                        html.Br(),
                        html.Br(),
                        html.Div(id="status-text", className="text-danger"),
                    ],
                    width=3,
                ),

                dbc.Col(
                    [
                        html.H4("Summary"),
                        html.Div(id="summary-div"),
                        html.Br(),

                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(id="frontier-graph"),
                                    width=6,
                                ),
                                dbc.Col(
                                    dcc.Graph(id="allocation-graph"),
                                    width=6,
                                ),
                            ]
                        ),

                        html.Br(),
                        dcc.Graph(id="correlation-graph"),
                    ],
                    width=9,
                ),
            ]
        ),

        html.Hr(),
        html.P(
            "Disclaimer: Educational use only. This is not financial advice.",
            className="text-center text-muted",
        ),
    ],
    fluid=True,
)


@app.callback(
    [
        Output("summary-div", "children"),
        Output("frontier-graph", "figure"),
        Output("allocation-graph", "figure"),
        Output("correlation-graph", "figure"),
        Output("status-text", "children"),
    ],
    Input("run-btn", "n_clicks"),
    State("sector-select", "value"),
    State("risk-tolerance", "value"),
    State("num-portfolios", "value"),
    State("investment-amount", "value"),
)
def run_optimization(n_clicks, sectors, risk_tol, num_ports, investment):
    if not n_clicks:
        return "Click 'Run optimization' to start.", go.Figure(), go.Figure(), go.Figure(), ""

    if not sectors:
        return "Please select at least one sector.", go.Figure(), go.Figure(), go.Figure(), "No sectors selected."

    if investment is None or investment <= 0:
        return "Enter a positive investment amount.", go.Figure(), go.Figure(), go.Figure(), "Invalid investment amount."

    # Build ticker list
    tickers = []
    for s in sectors:
        tickers.extend(STOCK_UNIVERSE.get(s, []))
    tickers = sorted(set(tickers))

    if len(tickers) < 2:
        return "Need at least two stocks.", go.Figure(), go.Figure(), go.Figure(), "Insufficient stocks."

    df, optimal, returns, data = monte_carlo_simulation(
        tickers, int(num_ports), float(risk_tol)
    )

    if df is None or optimal is None:
        return "Optimization failed. Try different settings.", go.Figure(), go.Figure(), go.Figure(), "Optimization failed."

    frontier_fig = make_frontier_figure(df, optimal)
    alloc_fig = make_allocation_figure(optimal, tickers)
    corr_fig = make_correlation_figure(returns, tickers)

    table_df = make_allocation_table(optimal, tickers, float(investment))

    summary_children = [
        html.P(f"Expected annual return: {optimal['Return']:.2%}"),
        html.P(f"Portfolio volatility: {optimal['Volatility']:.2%}"),
        html.P(f"Sharpe ratio: {optimal['Sharpe']:.4f}"),
        html.P(f"Total investment: ${investment:,.2f}"),
        html.Br(),
        dbc.Table.from_dataframe(table_df, striped=True, bordered=True, hover=True),
    ]

    status = f"Optimization complete using {len(tickers)} stocks from {len(sectors)} sectors."

    return summary_children, frontier_fig, alloc_fig, corr_fig, status


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
