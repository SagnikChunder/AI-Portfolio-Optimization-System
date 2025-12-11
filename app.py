import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

# Set risk-free rate
RISK_FREE_RATE = 0.04

# Expanded stock universe with sectors
STOCK_UNIVERSE = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'],
    'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C'],
    'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
    'Consumer': ['PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD'],
    'Industrial': ['BA', 'CAT', 'GE', 'MMM', 'UPS']
}

# --- Data Fetching and Core Logic (Simplified/Modified for Dash) ---

def get_market_data(tickers, period='1y'):
    """Fetch market data with robust error handling (period fixed to 1y for speed)"""
    try:
        # Fetching data directly for a single attempt
        data = yf.download(tickers, period=period, progress=False, threads=True)

        if data.empty:
            return pd.DataFrame()

        # Handle single/multiple ticker case
        if len(tickers) == 1:
            if 'Adj Close' in data.columns:
                return pd.DataFrame({tickers[0]: data['Adj Close']})
            else:
                return pd.DataFrame({tickers[0]: data['Close']})
        
        if 'Adj Close' in data.columns:
            return data['Adj Close']
        else:
            return data['Close']
            
    except Exception as e:
        print(f"Error fetching market data: {str(e)}")
        return pd.DataFrame()

def monte_carlo_simulation(tickers, num_portfolios, risk_tolerance):
    """Core Monte Carlo simulation function"""
    
    # 1. Fetch and Prepare Data
    data = get_market_data(tickers)
    
    if data.empty:
        return None, None
    
    data = data.dropna()
    if len(data) < 100:
        return None, None
    
    returns = data.pct_change().dropna()
    if returns.empty:
        return None, None

    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    # 2. Simulation Loop
    results = np.zeros((num_portfolios, 3 + len(tickers)))
    
    for i in range(num_portfolios):
        # Generate random weights
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

    # 3. Create DataFrame and Find Optimal Portfolio
    columns = ['Return', 'Volatility', 'Sharpe Ratio'] + [f'{ticker}_Weight' for ticker in tickers]
    df = pd.DataFrame(results, columns=columns)
    
    # Filter for valid results
    df = df[(df['Volatility'] > 0) & (df['Volatility'] < 1) & (np.isfinite(df['Sharpe Ratio']))]

    if df.empty:
        return None, None

    # Filter by risk tolerance (Max Volatility)
    filtered_df = df[df['Volatility'] <= risk_tolerance]

    # Find optimal portfolio (highest Sharpe Ratio)
    if not filtered_df.empty:
        optimal_portfolio = filtered_df.loc[filtered_df['Sharpe Ratio'].idxmax()]
    else:
        # Fallback to the overall highest Sharpe Ratio if no portfolio meets the max volatility constraint
        optimal_portfolio = df.loc[df['Sharpe Ratio'].idxmax()]

    # Attach market data for performance analysis
    optimal_portfolio['market_data'] = data
    optimal_portfolio['returns'] = returns
    
    return df, optimal_portfolio

# --- Plotly Dash Plotting Functions ---

def create_efficient_frontier_plot(df, optimal_portfolio, tickers):
    """Create Efficient Frontier and Allocation plots using Plotly"""
    if df is None or optimal_portfolio is None or df.empty:
        return go.Figure().update_layout(title="Efficient Frontier - Insufficient Data")

    # Efficient Frontier Scatter Plot
    frontier_scatter = go.Scatter(
        x=df['Volatility'],
        y=df['Return'],
        mode='markers',
        marker=dict(
            size=5,
            color=df['Sharpe Ratio'],
            colorscale='Viridis',
            colorbar=dict(title='Sharpe Ratio')
        ),
        name='Simulated Portfolios'
    )

    # Optimal Portfolio Marker
    optimal_marker = go.Scatter(
        x=[optimal_portfolio['Volatility']],
        y=[optimal_portfolio['Return']],
        mode='markers',
        marker=dict(size=20, symbol='star', color='red', line=dict(width=2, color='black')),
        name='Optimal Portfolio'
    )
    
    fig1 = go.Figure(data=[frontier_scatter, optimal_marker])
    fig1.update_layout(
        title='Efficient Frontier - Risk vs Return',
        xaxis_title='Volatility (Risk)',
        yaxis_title='Expected Return',
        height=500
    )
    
    # Portfolio Allocation Pie Chart
    weights = [optimal_portfolio[f'{ticker}_Weight'] for ticker in tickers if f'{ticker}_Weight' in optimal_portfolio]
    labels = [ticker for ticker in tickers if f'{ticker}_Weight' in optimal_portfolio]

    # Combine small weights into 'Others'
    df_weights = pd.Series(weights, index=labels)
    df_weights = df_weights[df_weights > 0.001] # Filter out near-zero weights

    if df_weights.empty:
        pie_data = pd.Series([1], index=['No Allocation']).reset_index()
        pie_data.columns = ['Ticker', 'Weight']
    else:
        small_weights = df_weights[df_weights < 0.02].sum() # Weights < 2%
        large_weights = df_weights[df_weights >= 0.02]
        
        if small_weights > 0.01: # Only add 'Others' if the sum is meaningful
            large_weights['Others'] = small_weights
            
        pie_data = large_weights.reset_index()
        pie_data.columns = ['Ticker', 'Weight']

    fig2 = go.Figure(data=[go.Pie(
        labels=pie_data['Ticker'], 
        values=pie_data['Weight'],
        hovertemplate='%{label}<br>%{value:.2%}<extra></extra>'
    )])

    fig2.update_layout(
        title='Optimal Portfolio Allocation',
        height=500
    )

    return html.Div([
        dcc.Graph(figure=fig1, style={'width': '50%', 'display': 'inline-block'}),
        dcc.Graph(figure=fig2, style={'width': '50%', 'display': 'inline-block'})
    ])

def create_correlation_heatmap(returns, tickers):
    """Create Correlation Heatmap using Plotly"""
    if returns is None or returns.empty:
        return go.Figure().update_layout(title="Stock Correlation Matrix - No Data")
    
    # Use only the tickers actually present in the returns data
    valid_tickers = [t for t in tickers if t in returns.columns]
    if len(valid_tickers) < 2:
        return go.Figure().update_layout(title="Stock Correlation Matrix - Not Enough Stocks")

    correlation_matrix = returns[valid_tickers].corr()

    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        hoverongaps=False
    ))

    fig.update_layout(
        title='Stock Correlation Matrix',
        xaxis=dict(tickangle=-45),
        yaxis=dict(autorange='reversed'),
        height=550
    )

    return dcc.Graph(figure=fig)

def create_performance_plot(data, optimal_portfolio, tickers):
    """Create Performance Analysis Plot (Normalized Price Evolution)"""
    if data is None or data.empty or optimal_portfolio is None:
        return go.Figure().update_layout(title="Performance Analysis - Insufficient Data")
    
    data_clean = data.dropna()
    if len(data_clean) < 10:
        return go.Figure().update_layout(title="Performance Analysis - Not Enough Data Points")

    # Normalized Stock Prices
    normalized_prices = data_clean / data_clean.iloc[0] * 100
    
    fig = go.Figure()
    
    valid_tickers = [t for t in tickers if t in normalized_prices.columns]
    
    # 1. Individual Stock Prices (Limit to 5 for readability)
    for ticker in valid_tickers[:5]:
        fig.add_trace(go.Scatter(
            x=normalized_prices.index, 
            y=normalized_prices[ticker], 
            mode='lines', 
            name=ticker, 
            opacity=0.5
        ))

    # 2. Optimal Portfolio Performance
    weights = np.array([optimal_portfolio.get(f'{ticker}_Weight', 0) for ticker in tickers])
    
    valid_indices = [i for i, ticker in enumerate(tickers) if ticker in data_clean.columns]
    valid_weights = weights[valid_indices]
    
    if np.sum(valid_weights) > 0:
        valid_weights = valid_weights / np.sum(valid_weights)
        portfolio_data = data_clean[[tickers[i] for i in valid_indices]]
        normalized_portfolio = portfolio_data / portfolio_data.iloc[0] * 100
        portfolio_value = (normalized_portfolio * valid_weights).sum(axis=1)

        fig.add_trace(go.Scatter(
            x=portfolio_value.index, 
            y=portfolio_value, 
            mode='lines', 
            name='Optimal Portfolio', 
            line=dict(color='red', width=3)
        ))

    fig.update_layout(
        title='Normalized Performance: Individual Stocks vs Optimal Portfolio',
        xaxis_title='Date',
        yaxis_title='Normalized Value (Base = 100)',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=550
    )

    return dcc.Graph(figure=fig)

# --- Reporting Functions (Markdown and Data Table) ---

def generate_investment_report(optimal_portfolio, tickers, investment_amount):
    """Generate detailed investment report and summary markdown"""
    if optimal_portfolio is None:
        return pd.DataFrame(), "### Error Generating Report"

    # 1. Allocation Table
    weights = [optimal_portfolio.get(f'{ticker}_Weight', 0) for ticker in tickers]
    allocations = [weight * investment_amount for weight in weights]

    report_data = []
    current_prices = {}
    
    for i, ticker in enumerate(tickers):
        if weights[i] > 0.001: # Only include significant weights
            # Try to get current price (simplified for Dash)
            try:
                hist = yf.Ticker(ticker).history(period='5d')
                price = hist['Close'].iloc[-1] if not hist.empty else 100.0
            except:
                price = 100.0
                
            current_prices[ticker] = price
            shares = max(0, int(allocations[i] / price))
            
            report_data.append({
                'Ticker': ticker,
                'Weight (%)': f"{weights[i]*100:.2f}%",
                'Allocation ($)': f"${allocations[i]:,.2f}",
                'Current Price': f"${price:.2f}",
                'Shares to Buy': shares
            })

    report_df = pd.DataFrame(report_data)

    # 2. Summary Markdown
    portfolio_return = optimal_portfolio['Return']
    portfolio_volatility = optimal_portfolio['Volatility']
    sharpe_ratio = optimal_portfolio['Sharpe Ratio']
    
    risk_level = (
        "🟢 Low Risk" if portfolio_volatility < 0.15 
        else "🟡 Moderate Risk" if portfolio_volatility < 0.25 
        else "🔴 High Risk"
    )

    summary_stats = f"""
        ### Portfolio Summary

        | Metric | Value |
        | :--- | :--- |
        | **Expected Annual Return** | {portfolio_return:.2%} |
        | **Portfolio Volatility** | {portfolio_volatility:.2%} |
        | **Sharpe Ratio** | {sharpe_ratio:.4f} |
        | **Total Investment** | ${investment_amount:,.2f} |
        | **Risk Assessment** | {risk_level} |

        ### Performance Expectations

        * **Expected Case:** {portfolio_return*100:.1f}% annual return
        * **Best Case (95% confidence):** {(portfolio_return + 2*portfolio_volatility)*100:.1f}% annual return
        * **Worst Case (5% confidence):** {(portfolio_return - 2*portfolio_volatility)*100:.1f}% annual return
    """
    
    return report_df, summary_stats

# --- Dash App Setup ---

app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div(style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '20px'}, children=[
    html.H1("AI-Powered Portfolio Management Assistant", style={'textAlign': 'center'}),
    html.P("Professional-grade portfolio optimization using Monte Carlo simulation."),

    html.Hr(),

    # --- Configuration Panel ---
    html.Div(className='config-panel', style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'marginBottom': '20px'}, children=[
        html.H3("Configuration"),
        
        html.Div([
            html.Label("Select Sectors (1-8 stocks will be used max):"),
            dcc.Checklist(
                id='sector-selection',
                options=[{'label': k, 'value': k} for k in STOCK_UNIVERSE.keys()],
                value=['Technology', 'Finance'],
                inline=True,
                style={'marginBottom': '10px'}
            ),
        ]),

        html.Div([
            html.Label("Risk Tolerance (Max Volatility):"),
            dcc.Slider(
                id='risk-tolerance',
                min=0.05, max=0.50, step=0.01, value=0.20,
                marks={i/100: f'{i}%' for i in range(5, 51, 5)}
            ),
        ], style={'marginBottom': '20px'}),

        html.Div([
            html.Label("Monte Carlo Simulations (More = Better, Slower):"),
            dcc.Slider(
                id='num-portfolios',
                min=1000, max=50000, step=1000, value=10000,
                marks={i: f'{i:,}' for i in [1000, 10000, 25000, 50000]}
            ),
        ], style={'marginBottom': '20px'}),
        
        html.Div([
            html.Label("Investment Amount ($):"),
            dcc.Input(id='investment-amount', type='number', value=10000, min=1, style={'width': '100%'})
        ], style={'marginBottom': '20px'}),

        html.Button('Optimize Portfolio', id='optimize-button', n_clicks=0, style={'width': '100%', 'padding': '10px', 'backgroundColor': '#007bff', 'color': 'white', 'border': 'none', 'borderRadius': '5px'}),
        
        html.Div(id='status-output', style={'marginTop': '10px', 'fontWeight': 'bold'})
    ]),

    # --- Results Section ---
    html.Hr(),
    html.H2("Optimization Results"),
    
    dcc.Loading(id="loading-1", type="default", children=[
        html.Div(id='summary-output', className='summary-markdown', style={'marginBottom': '20px'}),
        
        html.H3("Investment Allocation Plan"),
        dash_table.DataTable(
            id='allocation-output',
            columns=[
                {"name": "Ticker", "id": "Ticker"},
                {"name": "Weight (%)", "id": "Weight (%)"},
                {"name": "Allocation ($)", "id": "Allocation ($)"},
                {"name": "Current Price", "id": "Current Price"},
                {"name": "Shares to Buy", "id": "Shares to Buy"}
            ],
            data=[],
            sort_action="native",
            style_table={'overflowX': 'auto'},
            style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
        ),
        
        html.H3("Efficient Frontier and Allocation"),
        html.Div(id='frontier-plot-container', children=dcc.Graph(id='frontier-plot', figure=go.Figure().update_layout(height=500))),

        html.H3("Stock Correlation Matrix"),
        html.Div(id='correlation-plot-container', children=dcc.Graph(id='correlation-plot', figure=go.Figure().update_layout(height=550))),

        html.H3("Performance Analysis"),
        html.Div(id='performance-plot-container', children=dcc.Graph(id='performance-plot', figure=go.Figure().update_layout(height=550))),
    ]),

    # --- Disclaimer ---
    html.Hr(),
    html.Div([
        html.P("Disclaimer: This tool is for educational purposes only. Past performance does not guarantee future results. Always consult with a financial advisor before making investment decisions.", style={'fontSize': 'small', 'textAlign': 'center', 'color': 'gray'})
    ])
])


# --- Dash Callbacks ---

@app.callback(
    [
        Output('summary-output', 'children'),
        Output('allocation-output', 'data'),
        Output('frontier-plot-container', 'children'),
        Output('correlation-plot-container', 'children'),
        Output('performance-plot-container', 'children'),
        Output('status-output', 'children'),
    ],
    [Input('optimize-button', 'n_clicks')],
    [
        Input('sector-selection', 'value'),
        Input('risk-tolerance', 'value'),
        Input('num-portfolios', 'value'),
        Input('investment-amount', 'value')
    ]
)
def update_output(n_clicks, selected_sectors, risk_tolerance, num_portfolios, investment_amount):
    if n_clicks == 0 or not selected_sectors:
        return (
            "Select sectors and click 'Optimize Portfolio' to begin.", 
            [], 
            dcc.Graph(figure=go.Figure().update_layout(height=500)),
            dcc.Graph(figure=go.Figure().update_layout(height=550)),
            dcc.Graph(figure=go.Figure().update_layout(height=550)),
            "Status: Ready"
        )
    
    if investment_amount is None or investment_amount <= 0:
        return (
            "### Error", 
            [], 
            dcc.Graph(figure=go.Figure().update_layout(height=500)),
            dcc.Graph(figure=go.Figure().update_layout(height=550)),
            dcc.Graph(figure=go.Figure().update_layout(height=550)),
            "❌ Investment amount must be positive."
        )

    # 1. Get Tickers
    tickers = []
    for sector in selected_sectors:
        tickers.extend(STOCK_UNIVERSE.get(sector, []))
    
    tickers = list(set(tickers))
    
    if len(tickers) < 2:
        return (
            "### Error", 
            [], 
            dcc.Graph(figure=go.Figure().update_layout(height=500)),
            dcc.Graph(figure=go.Figure().update_layout(height=550)),
            dcc.Graph(figure=go.Figure().update_layout(height=550)),
            "❌ Please select sectors with at least 2 different stocks."
        )

    # Limit for efficiency/API stability
    if len(tickers) > 8:
        tickers = tickers[:8]

    status_message = f"Status: Running Monte Carlo simulation with {len(tickers)} stocks and {num_portfolios:,} portfolios..."

    # 2. Run Optimization
    df, optimal_portfolio = monte_carlo_simulation(tickers, num_portfolios, risk_tolerance)

    if df is None or optimal_portfolio is None:
        error_msg = "❌ Unable to fetch market data or optimize portfolio. Try again or check the selected stocks."
        return (
            "### Error", 
            [], 
            dcc.Graph(figure=go.Figure().update_layout(height=500)),
            dcc.Graph(figure=go.Figure().update_layout(height=550)),
            dcc.Graph(figure=go.Figure().update_layout(height=550)),
            error_msg
        )
    
    # Extract data for plotting/reporting
    market_data = optimal_portfolio.pop('market_data')
    returns = optimal_portfolio.pop('returns')
    
    # 3. Generate Outputs
    report_df, summary_stats = generate_investment_report(optimal_portfolio, tickers, investment_amount)
    frontier_plot = create_efficient_frontier_plot(df, optimal_portfolio, tickers)
    correlation_plot = create_correlation_heatmap(returns, tickers)
    performance_plot = create_performance_plot(market_data, optimal_portfolio, tickers)
    
    success_message = f"✅ Analysis complete! Optimized portfolio using {len(tickers)} stocks."
    
    return (
        dcc.Markdown(summary_stats),
        report_df.to_dict('records'),
        frontier_plot,
        correlation_plot,
        performance_plot,
        success_message
    )

if __name__ == '__main__':
    # Use custom ports/host for deployment environment compatibility
    app.run_server(debug=True, host='0.0.0.0', port=7860)
