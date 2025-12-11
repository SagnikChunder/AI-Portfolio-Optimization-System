import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from dash import Dash, html, dcc, Input, Output, State
import plotly.graph_objects as go
import plotly.tools as tls

# ---------------------------------------------------------
# STOCK UNIVERSE
# ---------------------------------------------------------

STOCK_UNIVERSE = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'],
    'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C'],
    'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
    'Consumer': ['PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD'],
    'Industrial': ['BA', 'CAT', 'GE', 'MMM', 'UPS']
}

# ---------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------

def get_market_data(tickers, period='1y'):
    try:
        for attempt in range(3):
            try:
                data = yf.download(tickers, period=period, progress=False, threads=True)
                if not data.empty:
                    if len(tickers) == 1:
                        col = "Adj Close" if "Adj Close" in data.columns else "Close"
                        return pd.DataFrame({tickers[0]: data[col]})
                    return data["Adj Close"] if "Adj Close" in data.columns else data["Close"]
            except:
                continue
        return pd.DataFrame()
    except:
        return pd.DataFrame()


def monte_carlo_simulation(tickers, num_portfolios=2000, risk_tolerance=0.2):

    if len(tickers) > 10:
        tickers = tickers[:10]

    data = get_market_data(tickers, period='1y')

    if data.empty:
        return None, None, None, None

    data = data.dropna()
    if len(data) < 100:
        return None, None, None, None

    returns = data.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    results = np.zeros((num_portfolios, 3 + len(tickers)))

    for i in range(num_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)

        p_return = np.sum(mean_returns * weights)
        p_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (p_return - 0.04) / p_std if p_std != 0 else 0

        results[i, 0] = p_return
        results[i, 1] = p_std
        results[i, 2] = sharpe
        results[i, 3:] = weights

    df = pd.DataFrame(
        results,
        columns=['Return', 'Volatility', 'Sharpe Ratio'] + [f"{t}_Weight" for t in tickers]
    )

    df = df[(df["Volatility"] > 0) & (np.isfinite(df["Sharpe Ratio"]))]

    if df.empty:
        return None, None, None, None

    filtered = df[df["Volatility"] <= risk_tolerance]
    optimal = filtered.loc[filtered["Sharpe Ratio"].idxmax()] if not filtered.empty else df.loc[df["Sharpe Ratio"].idxmax()]

    return df, optimal, returns, data


def fig_from_matplotlib(mat_fig):
    return tls.mpl_to_plotly(mat_fig)


def create_efficient_frontier_plot(df, optimal_portfolio):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(df['Volatility'], df['Return'], c=df['Sharpe Ratio'],
               cmap='viridis', alpha=0.6, s=15)

    ax.scatter(optimal_portfolio['Volatility'],
               optimal_portfolio['Return'],
               c='red', marker='*', s=300, edgecolor='black')

    ax.set_title("Efficient Frontier")
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Return")

    plt.tight_layout()
    return fig


def create_correlation_heatmap(returns):
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = returns.corr()

    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")

    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_yticklabels(corr.columns)

    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------
# DASH APP UI
# ---------------------------------------------------------

app = Dash(__name__)

app.layout = html.Div([
    html.H1("AI Portfolio Optimization Dashboard"),

    html.Label("Select Stocks"),
    dcc.Dropdown(
        id="stock-selector",
        options=[{"label": f"{sector}: {ticker}", "value": ticker}
                 for sector in STOCK_UNIVERSE for ticker in STOCK_UNIVERSE[sector]],
        multi=True
    ),

    html.Label("Risk Tolerance (Volatility Limit)"),
    dcc.Slider(id="risk-slider", min=0.05, max=0.5, step=0.01, value=0.2),

    html.Button("Run Simulation", id="run-btn"),

    html.H2("Efficient Frontier"),
    dcc.Loading(dcc.Graph(id="frontier-graph")),

    html.H2("Correlation Heatmap"),
    dcc.Loading(dcc.Graph(id="heatmap-graph")),

    html.H2("Optimal Portfolio Allocation"),
    dcc.Loading(dcc.Graph(id="allocation-graph")),
])

# ---------------------------------------------------------
# CALLBACKS
# ---------------------------------------------------------

@app.callback(
    Output("frontier-graph", "figure"),
    Output("heatmap-graph", "figure"),
    Output("allocation-graph", "figure"),
    Input("run-btn", "n_clicks"),
    State("stock-selector", "value"),
    State("risk-slider", "value"),
)
def run_simulation(n_clicks, tickers, risk):
    if not tickers:
        return go.Figure(), go.Figure(), go.Figure()

    df, optimal, returns, data = monte_carlo_simulation(tickers, 2000, risk)

    if df is None:
        return go.Figure(), go.Figure(), go.Figure()

    # Efficient Frontier
    ef_fig = fig_from_matplotlib(create_efficient_frontier_plot(df, optimal))

    # Correlation heatmap
    heatmap_fig = fig_from_matplotlib(create_correlation_heatmap(returns))

    # Allocation pie chart
    weights = {t: optimal[f"{t}_Weight"] for t in tickers}
    alloc_fig = go.Figure(data=[go.Pie(labels=list(weights.keys()), values=list(weights.values()))])
    alloc_fig.update_layout(title="Portfolio Allocation")

    return ef_fig, heatmap_fig, alloc_fig


# ---------------------------------------------------------
# RUN APP
# ---------------------------------------------------------

if __name__ == "__main__":
    app.run_server(debug=True)
