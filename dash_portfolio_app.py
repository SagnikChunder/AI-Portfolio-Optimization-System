import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
import yfinance as yf
import warnings
import json

warnings.filterwarnings('ignore')

# --- Global Data and Constants ---

# Expanded stock universe with sectors
STOCK_UNIVERSE = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'],
    'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C'],
    'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
    'Consumer': ['PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD'],
    'Industrial': ['BA', 'CAT', 'GE', 'MMM', 'UPS']
}
RISK_FREE_RATE = 0.04

# --- Core Portfolio Optimization Logic (Adapted for Dash) ---

def get_market_data(tickers, period='1y'):
    """Fetch market data with robust error handling"""
    if not tickers:
        return pd.DataFrame()
    try:
        # Try downloading data with multiple attempts
        for attempt in range(3):
            try:
                # Use adjusted period for yfinance
                data = yf.download(tickers, period=period, progress=False, threads=True)

                if data.empty:
                    continue

                # Handle single ticker case
                if len(tickers) == 1:
                    if 'Adj Close' in data.columns:
                        return pd.DataFrame({tickers[0]: data['Adj Close']})
                    else:
                        return pd.DataFrame({tickers[0]: data['Close']})

                # Handle multiple tickers
                if 'Adj Close' in data.columns:
                    # Select only 'Adj Close' for all tickers
                    adj_close_data = data['Adj Close']
                    # Drop any columns that are all NaN (sometimes happens with yfinance)
                    adj_close_data = adj_close_data.dropna(axis=1, how='all')
                    return adj_close_data
                else:
                    return data['Close']

            except Exception:
                if attempt < 2:
                    import time
                    time.sleep(2)
                continue
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def monte_carlo_simulation(tickers, num_portfolios=10000, risk_tolerance=0.2):
    """Enhanced Monte Carlo simulation"""
    try:
        # Limit number of tickers for stability
        if len(tickers) > 10:
            tickers = tickers[:10]

        data = get_market_data(tickers, period='1y')

        if data.empty:
            return None, None, None, None

        # Clean data
        data = data.dropna(axis=0)
        if len(data) < 100 or len(data.columns) < 2:
            return None, None, None, None

        returns = data.pct_change().dropna()

        if returns.empty:
            return None, None, None, None

        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
            return None, None, None, None

        results = np.zeros((num_portfolios, 3 + len(tickers)))

        for i in range(num_portfolios):
            weights = np.random.random(len(tickers))
            weights /= np.sum(weights)

            # Calculate portfolio metrics
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            sharpe_ratio = (portfolio_return - RISK_FREE_RATE) / portfolio_std if portfolio_std != 0 else 0

            results[i, 0] = portfolio_return
            results[i, 1] = portfolio_std
            results[i, 2] = sharpe_ratio
            results[i, 3:] = weights

        columns = ['Return', 'Volatility', 'Sharpe Ratio'] + [f'{ticker}_Weight' for ticker in data.columns]
        df = pd.DataFrame(results, columns=columns)

        # Remove invalid results
        df = df[(df['Volatility'] > 0) & (df['Volatility'] < 1) & (np.isfinite(df['Sharpe Ratio']))]

        if df.empty:
            return None, None, None, None

        # Find optimal portfolio
        optimal_idx = df['Sharpe Ratio'].idxmax()
        optimal_portfolio = df.loc[optimal_idx]

        return df, optimal_portfolio, returns, data

    except Exception:
        return None, None, None, None

def generate_investment_report(optimal_portfolio, tickers, investment_amount):
    """Generate detailed investment report and summary stats"""
    try:
        # Get weights safely, matching the order of tickers used in the simulation
        weights_dict = {f'{t}_Weight': optimal_portfolio.get(f'{t}_Weight', 0) for t in tickers}
        weights = [weights_dict[f'{t}_Weight'] for t in tickers]
        allocations = [weight * investment_amount for weight in weights]

        # Get current prices for share calculations
        current_prices = {}
        shares = {}

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                # Use info for faster retrieval of current price
                price = stock.info.get('regularMarketPrice')
                if price is not None and price > 0:
                    current_prices[ticker] = price
                    shares[ticker] = max(0, int(allocations[tickers.index(ticker)] / current_prices[ticker]))
                else:
                    current_prices[ticker] = 100.0  # Default fallback
                    shares[ticker] = max(0, int(allocations[tickers.index(ticker)] / current_prices[ticker]))
            except Exception:
                current_prices[ticker] = 100.0
                shares[ticker] = max(0, int(allocations[tickers.index(ticker)] / current_prices[ticker]))

        # Create report dataframe
        report_data = []
        for i, ticker in enumerate(tickers):
            if weights[i] > 0.001: # Only include tickers with significant weight
                report_data.append({
                    'Ticker': ticker,
                    'Weight (%)': f"{weights[i]*100:.2f}%",
                    'Allocation ($)': f"${allocations[i]:,.2f}",
                    'Current Price': f"${current_prices[ticker]:.2f}",
                    'Shares to Buy': shares[ticker]
                })

        report_df = pd.DataFrame(report_data)

        # Summary statistics markdown
        volatility = optimal_portfolio['Volatility']
        risk_level = "Low Risk" if volatility < 0.15 else "Moderate Risk" if volatility < 0.25 else "High Risk"

        summary_stats = f"""
        #### Portfolio Summary

        | Metric | Value |
        | :--- | :--- |
        | **Expected Annual Return** | {optimal_portfolio['Return']:.2%} |
        | **Portfolio Volatility (Risk)** | {optimal_portfolio['Volatility']:.2%} |
        | **Sharpe Ratio** | {optimal_portfolio['Sharpe Ratio']:.4f} |
        | **Total Investment** | ${investment_amount:,.2f} |

        #### Risk Assessment
        **Risk Level:** {risk_level}

        #### Performance Expectations
        * **Best Case (95% confidence):** {(optimal_portfolio['Return'] + 2*optimal_portfolio['Volatility'])*100:.1f}% annual return
        * **Expected Case:** {optimal_portfolio['Return']*100:.1f}% annual return
        * **Worst Case (5% confidence):** {(optimal_portfolio['Return'] - 2*optimal_portfolio['Volatility'])*100:.1f}% annual return
        """

        return report_df, summary_stats
    except Exception as e:
        return pd.DataFrame(), "Error generating report."

# --- Plotly Visualization Functions ---

def create_efficient_frontier_plot(df, optimal_portfolio, tickers):
    """Create efficient frontier visualization using Plotly"""
    if df is None or df.empty:
        return go.Figure().update_layout(title="Efficient Frontier - Insufficient Data")

    fig = px.scatter(
        df,
        x='Volatility',
        y='Return',
        color='Sharpe Ratio',
        color_continuous_scale=px.colors.sequential.Viridis,
        title='Efficient Frontier - Risk vs Return',
        labels={'Volatility': 'Volatility (Risk)', 'Return': 'Expected Return'},
        height=500
    )

    # Add optimal portfolio point
    if optimal_portfolio is not None:
        fig.add_trace(go.Scatter(
            x=[optimal_portfolio['Volatility']],
            y=[optimal_portfolio['Return']],
            mode='markers',
            marker=dict(
                size=15,
                color='red',
                symbol='star',
                line=dict(width=1, color='black')
            ),
            name='Optimal Portfolio'
        ))

    fig.update_layout(
        template="plotly_white",
        legend_title_text='Sharpe Ratio'
    )
    return fig

def create_allocation_pie_chart(optimal_portfolio, tickers):
    """Create portfolio allocation pie chart using Plotly"""
    if optimal_portfolio is None:
        return go.Figure().update_layout(title="Allocation - No Optimal Portfolio")

    try:
        weights = []
        valid_tickers = []
        other_weight = 0

        for ticker in tickers:
            weight_col = f'{ticker}_Weight'
            weight = optimal_portfolio.get(weight_col, 0)
            if weight > 0.001:
                if weight > 0.02:  # Show only weights > 2%
                    weights.append(weight)
                    valid_tickers.append(ticker)
                else:
                    other_weight += weight

        if other_weight > 0.001:
            weights.append(other_weight)
            valid_tickers.append('Others')

        if not weights:
            return go.Figure().update_layout(title="Allocation - No Significant Weights")

        fig = go.Figure(data=[go.Pie(
            labels=valid_tickers,
            values=weights,
            hovertemplate='%{label}<br>Weight: %{value:.2%}<extra></extra>'
        )])

        fig.update_layout(
            title_text='Optimal Portfolio Allocation',
            template="plotly_white",
            height=500
        )
        return fig
    except Exception:
        return go.Figure().update_layout(title="Allocation - Error")

def create_correlation_heatmap(returns, tickers):
    """Create correlation heatmap using Plotly"""
    if returns is None or returns.empty:
        return go.Figure().update_layout(title="Correlation Matrix - No Data")

    try:
        correlation_matrix = returns.corr()
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='coolwarm',
            zmin=-1, zmax=1
        ))

        fig.update_layout(
            title='Stock Correlation Matrix',
            xaxis_title='Ticker',
            yaxis_title='Ticker',
            template="plotly_white",
            height=600
        )
        return fig
    except Exception:
        return go.Figure().update_layout(title="Correlation Matrix - Error")

def create_performance_plot(data):
    """Create normalized price evolution plot using Plotly"""
    if data is None or data.empty or len(data.columns) < 2:
        return go.Figure().update_layout(title="Price Evolution - No Data")

    try:
        data_clean = data.dropna()
        if len(data_clean) < 10:
             return go.Figure().update_layout(title="Price Evolution - Insufficient Data Points")

        normalized_prices = data_clean / data_clean.iloc[0] * 100

        fig = go.Figure()

        # Plot up to 5 tickers for readability
        for ticker in normalized_prices.columns[:5]:
            fig.add_trace(go.Scatter(x=normalized_prices.index, y=normalized_prices[ticker],
                                     mode='lines', name=ticker))

        fig.update_layout(
            title='Normalized Stock Price Evolution (Base = 100)',
            xaxis_title='Date',
            yaxis_title='Normalized Price',
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig
    except Exception:
        return go.Figure().update_layout(title="Price Evolution - Error")


# --- Dash App Layout and Callbacks ---

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
app.title = "AI Portfolio Optimization"

# Define layout
app.layout = html.Div(style={'padding': '20px'}, children=[
    html.H1("AI-Powered Portfolio Management Assistant", style={'textAlign': 'center'}),
    html.P("Professional-grade portfolio optimization using Modern Portfolio Theory and Monte Carlo simulation.", style={'textAlign': 'center', 'marginBottom': '20px'}),

    # Configuration Row
    html.Div(className='row', style={'marginBottom': '20px'}, children=[
        html.Div(className='six columns', children=[
            html.Label("Select Sectors"),
            dcc.Checklist(
                id='sector-selection',
                options=[{'label': sector, 'value': sector} for sector in STOCK_UNIVERSE.keys()],
                value=['Technology', 'Finance'],
                inline=True,
                style={'paddingLeft': '10px'}
            ),
        ]),
        html.Div(className='three columns', children=[
            html.Label("Risk Tolerance (Max Volatility)"),
            dcc.Slider(
                id='risk-tolerance',
                min=0.05, max=0.50, step=0.01, value=0.20,
                marks={i/10: f'{i/10:.2f}' for i in range(1, 6)},
            ),
        ]),
        html.Div(className='three columns', children=[
            html.Label("Monte Carlo Simulations"),
            dcc.Input(id='num-portfolios', type='number', value=10000, min=1000, max=50000, step=1000, style={'width': '100%'}),
            html.Label("Investment Amount ($)"),
            dcc.Input(id='investment-amount', type='number', value=10000, min=1, step=1000, style={'width': '100%'})
        ])
    ]),

    # Optimization Button and Status
    html.Div(className='row', style={'textAlign': 'center', 'marginBottom': '20px'}, children=[
        html.Button('Optimize Portfolio', id='optimize-btn', n_clicks=0, style={'backgroundColor': '#1E90FF', 'color': 'white', 'padding': '10px 20px', 'fontSize': '16px', 'border': 'none', 'borderRadius': '5px'}),
    ]),

    html.Div(id='status-output', style={'textAlign': 'center', 'color': 'red', 'marginBottom': '20px'}),

    # Summary and Allocation
    html.Div(className='row', children=[
        html.Div(className='five columns', children=[
            dcc.Markdown(id='summary-output', style={'border': '1px solid #ccc', 'padding': '10px', 'borderRadius': '5px'})
        ]),
        html.Div(className='seven columns', children=[
            html.H4("Investment Allocation", style={'textAlign': 'center'}),
            dash_table.DataTable(
                id='allocation-output',
                columns=[
                    {"name": "Ticker", "id": "Ticker"},
                    {"name": "Weight (%)", "id": "Weight (%)"},
                    {"name": "Allocation ($)", "id": "Allocation ($)"},
                    {"name": "Current Price", "id": "Current Price"},
                    {"name": "Shares to Buy", "id": "Shares to Buy"},
                ],
                data=[],
                style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
                style_table={'overflowX': 'auto'}
            )
        ])
    ]),

    html.Hr(),

    # Plots
    html.Div(className='row', children=[
        html.Div(className='six columns', children=[
            dcc.Graph(id='frontier-plot')
        ]),
        html.Div(className='six columns', children=[
            dcc.Graph(id='allocation-plot')
        ])
    ]),

    html.Div(className='row', children=[
        html.Div(className='six columns', children=[
            dcc.Graph(id='correlation-plot')
        ]),
        html.Div(className='six columns', children=[
            dcc.Graph(id='performance-plot')
        ])
    ]),

    html.Hr(),
    html.Div(style={'textAlign': 'center', 'fontSize': '0.8em', 'color': '#777'}, children=[
        html.P("Disclaimer: This tool is for educational purposes only. Past performance does not guarantee future results. Always consult with a financial advisor before making investment decisions.")
    ])
])

# Single callback to run the optimization and update all components
@app.callback(
    [Output('status-output', 'children'),
     Output('summary-output', 'children'),
     Output('allocation-output', 'data'),
     Output('frontier-plot', 'figure'),
     Output('allocation-plot', 'figure'),
     Output('correlation-plot', 'figure'),
     Output('performance-plot', 'figure')],
    [Input('optimize-btn', 'n_clicks')],
    [dash.State('sector-selection', 'value'),
     dash.State('risk-tolerance', 'value'),
     dash.State('num-portfolios', 'value'),
     dash.State('investment-amount', 'value')]
)
def update_output(n_clicks, selected_sectors, risk_tolerance, num_portfolios, investment_amount):
    if n_clicks is None or n_clicks == 0:
        # Initial state: return empty figures
        empty_fig = go.Figure().update_layout(title="Run Optimization to See Results")
        return "", "", [], empty_fig, empty_fig, empty_fig, empty_fig

    # Input validation
    if not selected_sectors:
        return "Error: Please select at least one sector.", "", [], go.Figure(), go.Figure(), go.Figure(), go.Figure()
    if investment_amount is None or investment_amount <= 0:
        return "Error: Investment amount must be positive.", "", [], go.Figure(), go.Figure(), go.Figure(), go.Figure()
    if num_portfolios is None or num_portfolios < 1000:
        return "Error: Number of simulations must be at least 1000.", "", [], go.Figure(), go.Figure(), go.Figure(), go.Figure()

    # Get selected tickers
    all_tickers = []
    for sector in selected_sectors:
        if sector in STOCK_UNIVERSE:
            all_tickers.extend(STOCK_UNIVERSE[sector])

    tickers = list(set(all_tickers))

    if len(tickers) < 2:
        return "Error: Please select sectors with at least 2 different stocks.", "", [], go.Figure(), go.Figure(), go.Figure(), go.Figure()

    # Limit tickers for efficiency
    if len(tickers) > 8:
        tickers = tickers[:8]

    # Run Monte Carlo simulation
    df, optimal_portfolio, returns, data = monte_carlo_simulation(tickers, num_portfolios, risk_tolerance)

    if df is None or optimal_portfolio is None:
        error_fig = go.Figure().update_layout(title="Data Error: Could not fetch data or optimize portfolio.")
        return "Error: Unable to fetch market data or optimize portfolio. Try again with different sectors.", "", [], error_fig, error_fig, error_fig, error_fig

    # Generate Report
    report_df, summary_stats = generate_investment_report(optimal_portfolio, tickers, investment_amount)

    # Generate Plots (using Plotly functions)
    frontier_fig = create_efficient_frontier_plot(df, optimal_portfolio, tickers)
    allocation_fig = create_allocation_pie_chart(optimal_portfolio, tickers)
    correlation_fig = create_correlation_heatmap(returns, tickers)
    performance_fig = create_performance_plot(data)

    status_message = f"Analysis complete for {len(tickers)} stocks. Optimal Portfolio Sharpe Ratio: {optimal_portfolio['Sharpe Ratio']:.4f}"

    return status_message, summary_stats, report_df.to_dict('records'), frontier_fig, allocation_fig, correlation_fig, performance_fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
