import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import warnings
import time

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Global Configuration ---

# Expanded stock universe with sectors (Slightly reduced for speed in a web app)
STOCK_UNIVERSE = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META'],
    'Finance': ['JPM', 'BAC', 'GS', 'MS'],
    'Healthcare': ['JNJ', 'UNH', 'PFE', 'MRK'],
    'Energy': ['XOM', 'CVX', 'COP'],
    'Consumer': ['PG', 'KO', 'WMT', 'MCD'],
}

# Risk-free rate for Sharpe Ratio calculation
RISK_FREE_RATE = 0.04

# --- Data Fetching and Core Logic Functions ---

def get_market_data(tickers, period='1y'):
    """Fetch market data with error handling for a fixed period (1 year)"""
    try:
        data = yf.download(tickers, period=period, progress=False, threads=True)
        if data.empty:
            return pd.DataFrame()
        
        # Extract Adjusted Close, or Close if Adj Close is missing
        if 'Adj Close' in data.columns:
            # Handle single ticker case
            if len(tickers) == 1 and isinstance(data['Adj Close'], pd.Series):
                return pd.DataFrame({tickers[0]: data['Adj Close']})
            return data['Adj Close']
        else:
            if len(tickers) == 1 and isinstance(data['Close'], pd.Series):
                return pd.DataFrame({tickers[0]: data['Close']})
            return data['Close']
            
    except Exception as e:
        print(f"Error fetching market data: {str(e)}")
        return pd.DataFrame()

def calculate_portfolio_metrics(weights, returns):
    """Calculate portfolio performance metrics (annualized)"""
    # 252 trading days per year
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    
    # Handle potential division by zero for Sharpe Ratio
    sharpe_ratio = (portfolio_return - RISK_FREE_RATE) / portfolio_std if portfolio_std != 0 else 0
    
    return portfolio_return, portfolio_std, sharpe_ratio

def monte_carlo_simulation(tickers, num_portfolios, risk_tolerance):
    """Monte Carlo simulation for portfolio optimization"""
    
    if len(tickers) < 2:
        return None, None, "Must select at least two stocks."

    # Limit tickers to 8 for web app performance
    tickers = tickers[:8] 
    
    # Get data
    data = get_market_data(tickers)
    
    if data.empty:
        return None, None, "Could not fetch sufficient market data."

    data = data.dropna(axis=1) # Drop columns (tickers) with missing data
    data = data.dropna(axis=0)  # Drop rows (dates) with missing data
    
    # Update tickers list based on available data
    tickers = data.columns.tolist()
    
    if len(tickers) < 2:
        return None, None, "Insufficient clean data for two or more selected stocks."
        
    returns = data.pct_change().dropna()
    
    if returns.empty or len(returns) < 100:
        return None, None, "Insufficient daily returns data."

    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    results = np.zeros((num_portfolios, 3 + len(tickers)))

    for i in range(num_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        
        # Calculate metrics
        portfolio_return, portfolio_std, sharpe_ratio = calculate_portfolio_metrics(weights, returns)
        
        results[i, 0] = portfolio_return
        results[i, 1] = portfolio_std
        results[i, 2] = sharpe_ratio
        results[i, 3:] = weights
        
    # Create DataFrame of all simulated portfolios
    columns = ['Return', 'Volatility', 'Sharpe Ratio'] + [f'{ticker}_Weight' for ticker in tickers]
    df = pd.DataFrame(results, columns=columns)

    # Find optimal portfolio (highest Sharpe Ratio, constrained by Volatility)
    filtered_df = df[df['Volatility'] <= risk_tolerance]
    
    if not filtered_df.empty:
        optimal_idx = filtered_df['Sharpe Ratio'].idxmax()
        optimal_portfolio = filtered_df.loc[optimal_idx]
    else:
        # If no portfolio meets the risk tolerance, use the global max Sharpe Ratio portfolio
        optimal_idx = df['Sharpe Ratio'].idxmax()
        optimal_portfolio = df.loc[optimal_idx]

    return df, optimal_portfolio, returns, tickers, data

# --- Visualization Functions ---

def create_efficient_frontier_plot(df, optimal_portfolio, tickers):
    """Creates a Plotly Efficient Frontier and Portfolio Allocation chart."""
    if df is None or optimal_portfolio is None or df.empty:
        return go.Figure().update_layout(title="No Data for Efficient Frontier")

    # Efficient Frontier Scatter Plot
    frontier_fig = px.scatter(
        df, 
        x='Volatility', 
        y='Return', 
        color='Sharpe Ratio', 
        color_continuous_scale=px.colors.sequential.Viridis,
        title='Efficient Frontier: Risk vs Return',
        labels={'Volatility': 'Volatility (Annualized)', 'Return': 'Expected Return (Annualized)'},
        hover_data=[col for col in df.columns if 'Weight' in col],
    )
    
    # Add Optimal Portfolio Star Marker
    frontier_fig.add_trace(go.Scatter(
        x=[optimal_portfolio['Volatility']], 
        y=[optimal_portfolio['Return']], 
        mode='markers',
        marker=dict(symbol='star', size=20, color='red', line=dict(width=2, color='black')),
        name='Optimal Portfolio'
    ))
    
    frontier_fig.update_layout(
        template="plotly_white", 
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # Pie Chart Data (for allocation)
    weights = {f'{t}': optimal_portfolio[f'{t}_Weight'] for t in tickers if optimal_portfolio[f'{t}_Weight'] > 0.001}
    
    # Combine small weights into 'Others' for cleaner pie chart
    total_weight = sum(weights.values())
    pie_weights = {}
    other_weight = 0
    
    for ticker, weight in weights.items():
        if weight > 0.02: # Show only weights > 2%
            pie_weights[ticker] = weight
        else:
            other_weight += weight
            
    if other_weight > 0:
        pie_weights['Others'] = other_weight

    pie_df = pd.DataFrame(list(pie_weights.items()), columns=['Ticker', 'Weight'])
    
    # Pie Chart Figure
    pie_fig = px.pie(
        pie_df, 
        values='Weight', 
        names='Ticker', 
        title='Optimal Portfolio Allocation',
        hole=0.3
    )
    pie_fig.update_layout(
        template="plotly_white", 
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return frontier_fig, pie_fig

def create_correlation_heatmap(returns, tickers):
    """Creates a Plotly Correlation Heatmap."""
    if returns is None or returns.empty:
        return go.Figure().update_layout(title="No Returns Data for Heatmap")

    correlation_matrix = returns.corr()
    
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=tickers,
        y=tickers,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        hoverongaps=False
    ))

    # Add annotations (correlation values)
    annotations = []
    for i, row in enumerate(correlation_matrix.values):
        for j, val in enumerate(row):
            annotations.append(go.layout.Annotation(
                x=tickers[j],
                y=tickers[i],
                text=f"{val:.2f}",
                showarrow=False,
                font=dict(color="white" if abs(val) > 0.5 else "black")
            ))
            
    heatmap_fig.update_layout(
        title='Stock Correlation Matrix',
        xaxis_title='Ticker',
        yaxis_title='Ticker',
        template="plotly_white",
        annotations=annotations,
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return heatmap_fig

# --- Report and Summary Functions ---

def generate_investment_report(optimal_portfolio, tickers, investment_amount):
    """Generates the summary and allocation table data."""
    if optimal_portfolio is None:
        return "No optimal portfolio found.", pd.DataFrame()

    report_data = []
    total_invested = 0
    
    # Fetch latest prices (simplified: use historical close for demonstration)
    # In a production app, use real-time market data
    price_data = get_market_data(tickers, period='1d')
    current_prices = {t: price_data.iloc[-1][t] if t in price_data.columns else 100.0 for t in tickers}

    for ticker in tickers:
        weight_col = f'{ticker}_Weight'
        if weight_col in optimal_portfolio:
            weight = optimal_portfolio[weight_col]
            allocation = weight * investment_amount
            price = current_prices.get(ticker, 100.0)
            shares = max(0, int(allocation / price))
            
            report_data.append({
                'Ticker': ticker,
                'Weight (%)': f"{weight*100:.2f}%",
                'Allocation ($)': f"${allocation:,.2f}",
                'Current Price': f"${price:.2f}",
                'Shares to Buy': shares
            })
            total_invested += shares * price

    report_df = pd.DataFrame(report_data)

    # Summary statistics
    risk_level = "Low Risk (Vol < 15%)" if optimal_portfolio['Volatility'] < 0.15 else "Moderate Risk (15% < Vol < 25%)" if optimal_portfolio['Volatility'] < 0.25 else "High Risk (Vol > 25%)"

    summary_stats = f"""
    ### Portfolio Summary

    | Metric | Value |
    | :--- | :--- |
    | **Expected Annual Return** | {optimal_portfolio['Return']:.2%} |
    | **Portfolio Volatility** | {optimal_portfolio['Volatility']:.2%} |
    | **Sharpe Ratio** | {optimal_portfolio['Sharpe Ratio']:.4f} |
    | **Risk Assessment** | {risk_level} |
    | **Total Investment** | ${investment_amount:,.2f} |
    | **Total Shares Cost** | ${total_invested:,.2f} |
    
    ### Performance Expectations
    
    * **Expected Case:** {optimal_portfolio['Return']*100:.1f}% annual return
    * **Worst Case (5% confidence):** {(optimal_portfolio['Return'] - 2*optimal_portfolio['Volatility'])*100:.1f}% annual return
    """
    
    return summary_stats, report_df

# --- Dash App Setup ---

app = dash.Dash(__name__)
server = app.server 

app.layout = html.Div(style={'backgroundColor': '#f9f9f9', 'padding': '20px'}, children=[
    html.H1("AI Portfolio Optimization Assistant", style={'textAlign': 'center', 'color': '#333'}),
    html.P("Modern Portfolio Theory and Monte Carlo simulation to find the optimal risk-adjusted portfolio.", style={'textAlign': 'center', 'marginBottom': '30px'}),

    # Configuration Panel
    html.Div(style={'display': 'flex', 'gap': '20px', 'marginBottom': '30px', 'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '8px', 'backgroundColor': 'white'}, children=[
        
        # Sector Selection
        html.Div(style={'flex': 1}, children=[
            html.Label("Select Sectors:", style={'fontWeight': 'bold'}),
            dcc.Checklist(
                id='sector-selection',
                options=[{'label': sector, 'value': sector} for sector in STOCK_UNIVERSE.keys()],
                value=['Technology', 'Finance'],
                inline=False,
                style={'marginTop': '5px'}
            )
        ]),

        # Sliders and Inputs
        html.Div(style={'flex': 1}, children=[
            html.Label("Risk Tolerance (Max Volatility):", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='risk-tolerance',
                min=0.05, max=0.50, step=0.01, value=0.20,
                marks={i/10: f'{i/10:.0%}' for i in range(1, 6)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Br(),
            html.Label("Monte Carlo Simulations:", style={'fontWeight': 'bold', 'marginTop': '15px'}),
            dcc.Slider(
                id='num-portfolios',
                min=1000, max=10000, step=1000, value=5000,
                marks={i: str(i) for i in range(1000, 10001, 3000)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ]),
        
        # Investment and Button
        html.Div(style={'flex': 1, 'alignSelf': 'center'}, children=[
            html.Label("Investment Amount ($):", style={'fontWeight': 'bold'}),
            dcc.Input(id='investment-amount', type='number', value=10000, min=100, style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc'}),
            html.Br(),
            html.Button('🚀 Optimize Portfolio', id='optimize-button', n_clicks=0, style={'backgroundColor': '#007bff', 'color': 'white', 'padding': '10px 20px', 'border': 'none', 'borderRadius': '5px', 'marginTop': '20px', 'cursor': 'pointer', 'width': '100%'})
        ])
    ]),
    
    # Status and Report
    html.Div(id='status-output', style={'textAlign': 'center', 'color': 'green', 'fontWeight': 'bold', 'marginBottom': '20px'}),
    html.Div(dcc.Loading(id="loading-1", type="default", children=html.Div(id="loading-output-1")), style={'minHeight': '50px'}),
    
    html.Div(id='summary-output', className='markdown-body', style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '8px', 'backgroundColor': 'white', 'marginBottom': '20px'}),
    
    # Allocation Table
    html.H2("Investment Allocation", style={'textAlign': 'center', 'marginTop': '30px'}),
    html.Div(id='allocation-table', style={'padding': '10px', 'backgroundColor': 'white', 'borderRadius': '8px'}),
    
    # Plots
    html.Div(style={'display': 'flex', 'gap': '20px', 'marginTop': '30px'}, children=[
        html.Div(style={'flex': 1}, children=[
            dcc.Graph(id='efficient-frontier-plot'),
             html.P("The Efficient Frontier plots thousands of possible portfolios (dots) by their risk (Volatility) and return. The curve represents the best possible portfolios for any given level of risk. The optimal portfolio is the point with the highest Sharpe Ratio, limited by your Max Volatility target."),
            html.Br()
        ]),
        html.Div(style={'flex': 1}, children=[
            dcc.Graph(id='allocation-pie-plot'),
            html.P("This pie chart shows the percentage allocation for each stock in the final Optimal Portfolio. Allocation is determined by maximizing the Sharpe Ratio while respecting your maximum risk tolerance."),
            html.Br()
        ]),
    ]),
    
    dcc.Graph(id='correlation-heatmap-plot'),
    html.P("The Correlation Matrix shows how the stocks move in relation to each other. Lower (negative) correlation is beneficial for diversification, as it means stocks do not all fall (or rise) at the same time."),
    
    # Disclaimer
    html.Hr(style={'marginTop': '50px'}),
    html.P("Disclaimer: This tool is for educational purposes only. Past performance does not guarantee future results. Always consult with a financial advisor before making investment decisions.", style={'textAlign': 'center', 'fontSize': '0.8em', 'color': '#777'})
])

# --- Dash Callback ---

@app.callback(
    [
        Output('status-output', 'children'),
        Output('summary-output', 'children'),
        Output('allocation-table', 'children'),
        Output('efficient-frontier-plot', 'figure'),
        Output('allocation-pie-plot', 'figure'),
        Output('correlation-heatmap-plot', 'figure'),
        Output("loading-output-1", "children") # To clear the loading spinner
    ],
    [Input('optimize-button', 'n_clicks')],
    [
        State('sector-selection', 'value'),
        State('risk-tolerance', 'value'),
        State('num-portfolios', 'value'),
        State('investment-amount', 'value')
    ]
)
def update_output(n_clicks, selected_sectors, risk_tolerance, num_portfolios, investment_amount):
    if n_clicks is None or n_clicks == 0:
        # Initial state return empty figures
        empty_fig = go.Figure().update_layout(title="Click 'Optimize Portfolio' to begin")
        return "", "", "", empty_fig, empty_fig, empty_fig, ""

    if not selected_sectors:
        empty_fig = go.Figure().update_layout(title="No Data")
        return "❌ Please select at least one sector", "No summary available.", "", empty_fig, empty_fig, empty_fig, ""

    if investment_amount is None or investment_amount <= 0:
        empty_fig = go.Figure().update_layout(title="No Data")
        return "❌ Investment amount must be positive", "No summary available.", "", empty_fig, empty_fig, empty_fig, ""

    # 1. Collect all tickers from selected sectors
    tickers = []
    for sector in selected_sectors:
        if sector in STOCK_UNIVERSE:
            tickers.extend(STOCK_UNIVERSE[sector])
    tickers = list(set(tickers))
    
    if len(tickers) < 2:
        empty_fig = go.Figure().update_layout(title="No Data")
        return "❌ Please select sectors with at least 2 different stocks", "No summary available.", "", empty_fig, empty_fig, empty_fig, ""

    status_message = f"Optimizing portfolio with {len(tickers)} stocks and {num_portfolios} simulations..."

    # 2. Run Monte Carlo simulation
    df, optimal_portfolio, returns, final_tickers, data = monte_carlo_simulation(tickers, int(num_portfolios), risk_tolerance)
    
    if df is None:
        empty_fig = go.Figure().update_layout(title="Data Error")
        return f"❌ Optimization failed. {optimal_portfolio}", "No summary available.", "", empty_fig, empty_fig, empty_fig, ""

    # 3. Create Report and Allocation Table
    summary_stats, report_df = generate_investment_report(optimal_portfolio, final_tickers, investment_amount)

    allocation_table_html = dash.dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in report_df.columns],
        data=report_df.to_dict('records'),
        style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
        style_data_conditional=[
            {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}
        ]
    )

    # 4. Create Visualizations
    frontier_fig, pie_fig = create_efficient_frontier_plot(df, optimal_portfolio, final_tickers)
    heatmap_fig = create_correlation_heatmap(returns, final_tickers)

    success_message = f"✅ Analysis complete! Optimized portfolio using {len(final_tickers)} stocks."
    
    return success_message, dcc.Markdown(summary_stats), allocation_table_html, frontier_fig, pie_fig, heatmap_fig, ""

if __name__ == '__main__':
    # Set host and port for deployment
    app.run_server(debug=True)
