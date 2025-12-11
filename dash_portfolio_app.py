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
                    # Use 'Adj Close' or 'Close' if 'Adj Close' is missing
                    if 'Adj Close' in data.columns:
                        return pd.DataFrame({tickers[0]: data['Adj Close']})
                    else:
                        return pd.DataFrame({tickers[0]: data['Close']})

                # Handle multiple tickers
                if 'Adj Close' in data.columns and isinstance(data['Adj Close'], pd.DataFrame):
                    # Select only 'Adj Close' for all tickers
                    adj_close_data = data['Adj Close']
                    # Drop any columns that are all NaN (sometimes happens with yfinance)
                    adj_close_data = adj_close_data.dropna(axis=1, how='all')
                    return adj_close_data
                elif 'Close' in data.columns and isinstance(data['Close'], pd.DataFrame):
                    return data['Close'].dropna(axis=1, how='all')
                else:
                    # Fallback for unexpected yfinance data structure
                    return pd.DataFrame()

            except Exception:
                if attempt < 2:
                    import time
                    time.sleep(2)
                continue
        return pd.DataFrame()
    except Exception as e:
        print(f"Global error in get_market_data: {e}")
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

        # --- CRITICAL FIX: Ensure Data is Cleaned and Sufficient ---
        # 1. Drop rows with any NaN (align dates/data)
        data = data.dropna(axis=0) 
        # 2. Drop columns (tickers) with too many NaNs (less than 90% valid data)
        data = data.dropna(axis=1, thresh=len(data)*0.9) 

        # Get the actual tickers that survived data cleaning
        sim_tickers = data.columns.tolist() 

        # Check for sufficient data after cleaning
        if len(data) < 100 or len(sim_tickers) < 2:
            print("Insufficient clean data points or tickers after cleanup.")
            return None, None, None, None

        returns = data.pct_change().dropna()

        # Check for sufficient returns data
        if returns.empty or len(returns.columns) < 2:
            print("Returns data is empty or has less than 2 columns.")
            return None, None, None, None

        # Recalculate mean/cov based on cleaned returns
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        # Check for non-finite values in covariance matrix
        if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
            print("Non-finite values in covariance matrix.")
            return None, None, None, None

        num_assets = len(sim_tickers)
        results = np.zeros((num_portfolios, 3 + num_assets))

        for i in range(num_portfolios):
            # Generate random weights
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)

            # Calculate portfolio metrics
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            sharpe_ratio = (portfolio_return - RISK_FREE_RATE) / portfolio_std if portfolio_std != 0 else 0

            results[i, 0] = portfolio_return
            results[i, 1] = portfolio_std
            results[i, 2] = sharpe_ratio
            results[i, 3:] = weights

        # CRITICAL FIX: Use sim_tickers for column naming
        columns = ['Return', 'Volatility', 'Sharpe Ratio'] + [f'{ticker}_Weight' for ticker in sim_tickers]
        df = pd.DataFrame(results, columns=columns)

        # Remove invalid results
        df = df[(df['Volatility'] > 0) & (df['Volatility'] < 1) & (np.isfinite(df['Sharpe Ratio']))]

        if df.empty:
            print("No valid portfolios generated.")
            return None, None, None, None

        # Find optimal portfolio (Maximum Sharpe Ratio)
        optimal_idx = df['Sharpe Ratio'].idxmax()
        optimal_portfolio = df.loc[optimal_idx]

        return df, optimal_portfolio, returns, data

    except Exception as e:
        print(f"Error in monte_carlo_simulation: {e}")
        return None, None, None, None

def generate_investment_report(optimal_portfolio, all_input_tickers, investment_amount):
    """Generate detailed investment report and summary stats."""
    try:
        # Check if optimization failed
        if optimal_portfolio is None:
            return pd.DataFrame(), "Error: Optimal portfolio data is missing."

        # Get weights safely, including only tickers that were successfully optimized
        weights_dict = {col: optimal_portfolio.get(col, 0) for col in optimal_portfolio.index if col.endswith('_Weight')}
        
        # Extract the list of tickers that were actually used in the simulation
        sim_tickers = [col.replace('_Weight', '') for col in weights_dict.keys()]
        weights = [weights_dict[f'{t}_Weight'] for t in sim_tickers]
        allocations = [weight * investment_amount for weight in weights]

        # Get current prices for share calculations
        current_prices = {}
        shares = {}

        for ticker in sim_tickers:
            try:
                stock = yf.Ticker(ticker)
                price = stock.info.get('regularMarketPrice')
                if price is not None and price > 0:
                    current_prices[ticker] = price
                    # Find the corresponding allocation index
                    idx = sim_tickers.index(ticker)
                    # Use floor division for whole shares
                    shares[ticker] = max(0, int(allocations[idx] / current_prices[ticker]))
                else:
                    current_prices[ticker] = np.nan
                    shares[ticker] = 0
            except Exception:
                current_prices[ticker] = np.nan
                shares[ticker] = 0

        # Create report dataframe
        report_data = []
        for i, ticker in enumerate(sim_tickers):
            if weights[i] > 0.001: # Only include tickers with significant weight
                price_str = f"${current_prices[ticker]:.2f}" if not np.isnan(current_prices[ticker]) else "N/A"
                report_data.append({
                    'Ticker': ticker,
                    'Weight (%)': f"{weights[i]*100:.2f}%",
                    'Allocation ($)': f"${allocations[i]:,.2f}",
                    'Current Price': price_str,
                    'Shares to Buy': shares[ticker]
                })

        report_df = pd.DataFrame(report_data)

        # Summary statistics markdown
        volatility = optimal_portfolio.get('Volatility', 0.0)
        risk_level = "Low Risk" if volatility < 0.15 else "Moderate Risk" if volatility < 0.25 else "High Risk"
        expected_return = optimal_portfolio.get('Return', 0.0)
        sharpe_ratio = optimal_portfolio.get('Sharpe Ratio', 0.0)

        summary_stats = f"""
        #### Portfolio Summary

        | Metric | Value |
        | :--- | :--- |
        | **Expected Annual Return** | {expected_return:.2%} |
        | **Portfolio Volatility (Risk)** | {volatility:.2%} |
        | **Sharpe Ratio** | {sharpe_ratio:.4f} |
        | **Total Investment** | ${investment_amount:,.2f} |

        #### Risk Assessment
        **Risk Level:** {risk_level}

        #### Performance Expectations
        * **Best Case (95% confidence):** {(expected_return + 2*volatility)*100:.1f}% annual return
        * **Expected Case:** {expected_return*100:.1f}% annual return
        * **Worst Case (5% confidence):** {(expected_return - 2*volatility)*100:.1f}% annual return
        """

        return report_df, summary_stats
    except Exception as e:
        print(f"Error generating investment report: {e}")
        return pd.DataFrame(), "Error generating report."

# --- Plotly Visualization Functions ---

def create_efficient_frontier_plot(df, optimal_portfolio):
    """Create efficient frontier visualization using Plotly"""
    
    if df is None or df.empty:
        return go.Figure().update_layout(title="Efficient Frontier - Insufficient Data", height=500)

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
    if optimal_portfolio is not None and 'Volatility' in optimal_portfolio and 'Return' in optimal_portfolio:
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
        legend_title_text='Sharpe Ratio',
        xaxis_tickformat=".2%",
        yaxis_tickformat=".2%"
    )
    return fig

def create_allocation_pie_chart(optimal_portfolio):
    """Create portfolio allocation pie chart using Plotly"""
    

[Image of a portfolio allocation pie chart]

    if optimal_portfolio is None:
        return go.Figure().update_layout(title="Allocation - No Optimal Portfolio", height=500)

    try:
        weights = []
        valid_tickers = []
        other_weight = 0

        # Filter and normalize weights for the pie chart
        weights_series = optimal_portfolio[optimal_portfolio.index.str.endswith('_Weight')]
        weights_series = weights_series[weights_series > 0.001].sort_values(ascending=False)
        
        # Determine number of major slices to show
        major_slices = 5
        
        for i, (col, weight) in enumerate(weights_series.items()):
            ticker = col.replace('_Weight', '')
            if i < major_slices:
                weights.append(weight)
                valid_tickers.append(ticker)
            else:
                other_weight += weight

        if other_weight > 0.001:
            weights.append(other_weight)
            valid_tickers.append('Others')

        if not weights:
            return go.Figure().update_layout(title="Allocation - No Significant Weights", height=500)

        fig = go.Figure(data=[go.Pie(
            labels=valid_tickers,
            values=weights,
            hovertemplate='%{label}<br>Weight: %{value:.2%}<extra></extra>',
            textinfo='label+percent'
        )])

        fig.update_layout(
            title_text='Optimal Portfolio Allocation',
            template="plotly_white",
            height=500
        )
        return fig
    except Exception:
        return go.Figure().update_layout(title="Allocation - Error", height=500)

def create_correlation_heatmap(returns):
    """Create correlation heatmap using Plotly"""
    
    # **CRITICAL FIX: Check for returns validity here**
    if returns is None or returns.empty or len(returns.columns) < 2:
        return go.Figure().update_layout(title="Stock Correlation Matrix - No Data (Need at least 2 stocks)", height=600)

    try:
        correlation_matrix = returns.corr()

        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='coolwarm',
            zmin=-1, zmax=1
        ))

        # Add annotations (correlation values)
        annotations = []
        for i in range(len(correlation_matrix.index)):
            for j in range(len(correlation_matrix.columns)):
                annotations.append(go.layout.Annotation(
                    text=f'{correlation_matrix.iloc[i, j]:.2f}',
                    x=correlation_matrix.columns[j],
                    y=correlation_matrix.index[i],
                    xref='x1', yref='y1',
                    showarrow=False,
                    font=dict(color='black' if abs(correlation_matrix.iloc[i, j]) < 0.5 else 'white', size=10)
                ))

        fig.update_layout(
            title='Stock Correlation Matrix',
            xaxis_title='Ticker',
            yaxis_title='Ticker',
            template="plotly_white",
            annotations=annotations,
            height=600
        )
        return fig
    except Exception as e:
        print(f"Error creating correlation heatmap: {e}")
        return go.Figure().update_layout(title="Stock Correlation Matrix - Error", height=600)

def create_performance_plot(data):
    """Create normalized price evolution plot using Plotly"""
    
    if data is None or data.empty or len(data.columns) < 1:
        # Check for at least 1 column for a basic plot
        return go.Figure().update_layout(title="Price Evolution - No Data", height=500)

    try:
        data_clean = data.dropna()
        if len(data_clean) < 10:
             return go.Figure().update_layout(title="Price Evolution - Insufficient Data Points", height=500)

        # Normalize prices to 100
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
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500
        )
        return fig
    except Exception as e:
        print(f"Error creating performance plot: {e}")
        return go.Figure().update_layout(title="Price Evolution - Error", height=500)


# --- Dash App Layout and Callbacks (Layout remains the same) ---

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
            html.Br(),
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
    # Default empty figure
    initial_message = "Click 'Optimize Portfolio' to run the analysis."
    error_fig = go.Figure().update_layout(title=initial_message, height=500)
    error_corr_fig = go.Figure().update_layout(title=initial_message, height=600)

    if n_clicks is None or n_clicks == 0:
        return "", "", [], error_fig, error_fig, error_corr_fig, error_fig

    # Input validation
    if not selected_sectors:
        return "Error: Please select at least one sector.", "", [], error_fig, error_fig, error_corr_fig, error_fig
    if investment_amount is None or investment_amount <= 0:
        return "Error: Investment amount must be positive.", "", [], error_fig, error_fig, error_corr_fig, error_fig
    if num_portfolios is None or num_portfolios < 1000:
        return "Error: Number of simulations must be at least 1000.", "", [], error_fig, error_fig, error_corr_fig, error_fig

    # Get selected tickers
    all_tickers = []
    for sector in selected_sectors:
        if sector in STOCK_UNIVERSE:
            all_tickers.extend(STOCK_UNIVERSE[sector])

    tickers = list(set(all_tickers))

    if len(tickers) < 2:
        return "Error: Please select sectors with at least 2 different stocks.", "", [], error_fig, error_fig, error_corr_fig, error_fig

    # Limit tickers for efficiency
    if len(tickers) > 8:
        tickers = tickers[:8]

    # Run Monte Carlo simulation
    df, optimal_portfolio, returns, data = monte_carlo_simulation(tickers, num_portfolios, risk_tolerance)

    if df is None or optimal_portfolio is None:
        # Create specific error figures for a clear user message
        data_error_title = "Data Error: Could not fetch sufficient market data or optimize portfolio. Try again with fewer sectors or different time."
        data_error_fig = go.Figure().update_layout(title=data_error_title, height=500)
        data_error_corr_fig = go.Figure().update_layout(title=data_error_title, height=600)
        return "Error: Unable to fetch sufficient market data or optimize portfolio.", "", [], data_error_fig, data_error_fig, data_error_corr_fig, data_error_fig

    # Generate Report
    report_df, summary_stats = generate_investment_report(optimal_portfolio, tickers, investment_amount)

    # Generate Plots (using Plotly functions)
    frontier_fig = create_efficient_frontier_plot(df, optimal_portfolio)
    allocation_fig = create_allocation_pie_chart(optimal_portfolio)
    
    # Use returns for correlation matrix
    correlation_fig = create_correlation_heatmap(returns)
    performance_fig = create_performance_plot(data)

    status_message = f"Analysis complete for {len(data.columns)} stocks. Optimal Portfolio Sharpe Ratio: {optimal_portfolio['Sharpe Ratio']:.4f}"

    return status_message, summary_stats, report_df.to_dict('records'), frontier_fig, allocation_fig, correlation_fig, performance_fig

# Run the app
if __name__ == '__main__':
    # Use a specific port to avoid conflicts
    app.run_server(debug=True, port=8050)
