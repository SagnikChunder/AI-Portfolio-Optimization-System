import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Stock universe organized by sectors
STOCK_UNIVERSE = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'],
    'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C'],
    'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
    'Consumer': ['PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD'],
    'Industrial': ['BA', 'CAT', 'GE', 'MMM', 'UPS']
}

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "AI Portfolio Optimizer"

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("AI-Powered Portfolio Management Assistant", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
        html.P("Professional-grade portfolio optimization using Modern Portfolio Theory and Monte Carlo simulation",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '16px'})
    ], style={'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '10px', 'margin': '20px'}),
    
    # Main content
    html.Div([
        # Left sidebar - Controls
        html.Div([
            html.H3(" Configuration", style={'color': '#2c3e50'}),
            
            html.Label(" Select Sectors:", style={'fontWeight': 'bold', 'marginTop': '15px'}),
            dcc.Checklist(
                id='sector-selection',
                options=[{'label': sector, 'value': sector} for sector in STOCK_UNIVERSE.keys()],
                value=['Technology', 'Finance'],
                style={'marginBottom': '20px'}
            ),
            
            html.Label(" Risk Tolerance (Max Volatility):", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='risk-tolerance',
                min=0.05,
                max=0.50,
                step=0.01,
                value=0.20,
                marks={0.05: '5%', 0.15: '15%', 0.25: '25%', 0.35: '35%', 0.50: '50%'},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.P("Higher values allow for more aggressive portfolios", 
                   style={'fontSize': '12px', 'color': '#7f8c8d', 'marginBottom': '20px'}),
            
            html.Label(" Monte Carlo Simulations:", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='num-portfolios',
                min=1000,
                max=50000,
                step=1000,
                value=10000,
                marks={1000: '1K', 10000: '10K', 25000: '25K', 50000: '50K'},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.P("More simulations = better optimization (but slower)", 
                   style={'fontSize': '12px', 'color': '#7f8c8d', 'marginBottom': '20px'}),
            
            html.Label(" Investment Amount ($):", style={'fontWeight': 'bold'}),
            dcc.Input(
                id='investment-amount',
                type='number',
                value=10000,
                style={'width': '100%', 'padding': '10px', 'marginBottom': '20px', 'borderRadius': '5px', 'border': '1px solid #bdc3c7'}
            ),
            
            html.Button(' Optimize Portfolio', 
                       id='optimize-button', 
                       n_clicks=0,
                       style={
                           'width': '100%', 
                           'padding': '15px', 
                           'backgroundColor': '#3498db', 
                           'color': 'white', 
                           'border': 'none', 
                           'borderRadius': '5px',
                           'fontSize': '16px',
                           'fontWeight': 'bold',
                           'cursor': 'pointer'
                       }),
            
            html.Div(id='status-output', 
                    style={'marginTop': '20px', 'padding': '10px', 'borderRadius': '5px'})
            
        ], style={
            'width': '25%', 
            'padding': '20px', 
            'backgroundColor': '#f8f9fa', 
            'borderRadius': '10px',
            'marginRight': '20px'
        }),
        
        # Right content - Results
        html.Div([
            # Summary section
            html.Div(id='summary-output', 
                    style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),
            
            # Allocation table
            html.Div([
                html.H4(" Investment Allocation", style={'color': '#2c3e50'}),
                html.Div(id='allocation-table')
            ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),
            
            # Efficient Frontier plot
            dcc.Graph(id='frontier-plot', style={'marginBottom': '20px'}),
            
            # Correlation heatmap
            dcc.Graph(id='correlation-plot', style={'marginBottom': '20px'}),
            
            # Performance analysis
            dcc.Graph(id='performance-plot', style={'marginBottom': '20px'})
            
        ], style={'width': '73%'})
        
    ], style={'display': 'flex', 'padding': '20px'}),
    
    # Footer
    html.Div([
        html.P(" Disclaimer: This tool is for educational purposes only. Past performance does not guarantee future results. Always consult with a financial advisor before making investment decisions.",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '14px'})
    ], style={'padding': '20px', 'borderTop': '1px solid #bdc3c7', 'marginTop': '20px'})
    
], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#ecf0f1', 'minHeight': '100vh'})


# Helper functions
def get_market_data(tickers, period='1y'):
    """Fetch market data with error handling"""
    try:
        data = yf.download(tickers, period=period, progress=False, threads=True)
        
        if data.empty:
            return pd.DataFrame()
        
        if len(tickers) == 1:
            return pd.DataFrame({tickers[0]: data['Adj Close'] if 'Adj Close' in data.columns else data['Close']})
        
        return data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    except:
        return pd.DataFrame()


def monte_carlo_simulation(tickers, num_portfolios=10000, risk_tolerance=0.2):
    """Run Monte Carlo simulation for portfolio optimization"""
    try:
        if len(tickers) > 8:
            tickers = tickers[:8]
        
        data = get_market_data(tickers, period='1y')
        
        if data.empty or len(data) < 100:
            return None, None, None, None
        
        data = data.dropna()
        returns = data.pct_change().dropna()
        
        if returns.empty:
            return None, None, None, None
        
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        # Monte Carlo simulation
        results = np.zeros((num_portfolios, 3 + len(tickers)))
        
        for i in range(num_portfolios):
            weights = np.random.random(len(tickers))
            weights /= np.sum(weights)
            
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - 0.04) / portfolio_std if portfolio_std > 0 else 0
            
            results[i, 0] = portfolio_return
            results[i, 1] = portfolio_std
            results[i, 2] = sharpe_ratio
            results[i, 3:] = weights
        
        columns = ['Return', 'Volatility', 'Sharpe Ratio'] + [f'{ticker}_Weight' for ticker in tickers]
        df = pd.DataFrame(results, columns=columns)
        
        df = df[(df['Volatility'] > 0) & (df['Volatility'] < 1) & (np.isfinite(df['Sharpe Ratio']))]
        
        if df.empty:
            return None, None, None, None
        
        filtered_df = df[df['Volatility'] <= risk_tolerance]
        
        if not filtered_df.empty:
            optimal_idx = filtered_df['Sharpe Ratio'].idxmax()
            optimal_portfolio = filtered_df.loc[optimal_idx]
        else:
            optimal_idx = df['Sharpe Ratio'].idxmax()
            optimal_portfolio = df.loc[optimal_idx]
        
        return df, optimal_portfolio, returns, data
        
    except Exception as e:
        print(f"Error in simulation: {e}")
        return None, None, None, None


# Callback for optimization
@app.callback(
    [Output('summary-output', 'children'),
     Output('allocation-table', 'children'),
     Output('frontier-plot', 'figure'),
     Output('correlation-plot', 'figure'),
     Output('performance-plot', 'figure'),
     Output('status-output', 'children'),
     Output('status-output', 'style')],
    [Input('optimize-button', 'n_clicks')],
    [State('sector-selection', 'value'),
     State('risk-tolerance', 'value'),
     State('num-portfolios', 'value'),
     State('investment-amount', 'value')]
)
def optimize_portfolio(n_clicks, selected_sectors, risk_tolerance, num_portfolios, investment_amount):
    if n_clicks == 0:
        return "", "", {}, {}, {}, "", {}
    
    # Validate inputs
    if not selected_sectors:
        return "", "", {}, {}, {}, " Please select at least one sector", {'backgroundColor': '#e74c3c', 'color': 'white'}
    
    if investment_amount <= 0:
        return "", "", {}, {}, {}, " Investment amount must be positive", {'backgroundColor': '#e74c3c', 'color': 'white'}
    
    # Get tickers
    tickers = []
    for sector in selected_sectors:
        tickers.extend(STOCK_UNIVERSE[sector])
    tickers = list(set(tickers))
    
    if len(tickers) < 2:
        return "", "", {}, {}, {}, "❌ Please select sectors with at least 2 stocks", {'backgroundColor': '#e74c3c', 'color': 'white'}
    
    # Run optimization
    df, optimal_portfolio, returns, data = monte_carlo_simulation(tickers, int(num_portfolios), risk_tolerance)
    
    if df is None or optimal_portfolio is None:
        return "", "", {}, {}, {}, "❌ Unable to fetch market data. Please try again.", {'backgroundColor': '#e74c3c', 'color': 'white'}
    
    # Create summary
    summary = html.Div([
        html.H3("📊 Portfolio Summary", style={'color': '#2c3e50'}),
        html.P([html.Strong("Expected Annual Return: "), f"{optimal_portfolio['Return']:.2%}"]),
        html.P([html.Strong("Portfolio Volatility: "), f"{optimal_portfolio['Volatility']:.2%}"]),
        html.P([html.Strong("Sharpe Ratio: "), f"{optimal_portfolio['Sharpe Ratio']:.4f}"]),
        html.P([html.Strong("Total Investment: "), f"${investment_amount:,.2f}"]),
        html.Hr(),
        html.H4("🎯 Risk Assessment", style={'color': '#2c3e50'}),
        html.P("🟢 Low Risk" if optimal_portfolio['Volatility'] < 0.15 else "🟡 Moderate Risk" if optimal_portfolio['Volatility'] < 0.25 else "🔴 High Risk",
               style={'fontSize': '18px', 'fontWeight': 'bold'}),
        html.Hr(),
        html.H4("📈 Performance Expectations", style={'color': '#2c3e50'}),
        html.P([html.Strong("Best Case (95%): "), f"{(optimal_portfolio['Return'] + 2*optimal_portfolio['Volatility'])*100:.1f}% annual return"]),
        html.P([html.Strong("Expected Case: "), f"{optimal_portfolio['Return']*100:.1f}% annual return"]),
        html.P([html.Strong("Worst Case (5%): "), f"{(optimal_portfolio['Return'] - 2*optimal_portfolio['Volatility'])*100:.1f}% annual return"])
    ])
    
    # Create allocation table
    allocation_data = []
    for ticker in tickers:
        weight = optimal_portfolio[f'{ticker}_Weight']
        allocation = weight * investment_amount
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='5d')
            price = hist['Close'].iloc[-1] if not hist.empty else 100.0
        except:
            price = 100.0
        
        shares = int(allocation / price)
        
        allocation_data.append({
            'Ticker': ticker,
            'Weight': f"{weight*100:.2f}%",
            'Allocation': f"${allocation:,.2f}",
            'Current Price': f"${price:.2f}",
            'Shares': shares
        })
    
    allocation_df = pd.DataFrame(allocation_data)
    allocation_table = dash_table.DataTable(
        data=allocation_df.to_dict('records'),
        columns=[{'name': col, 'id': col} for col in allocation_df.columns],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '10px'},
        style_header={'backgroundColor': '#3498db', 'color': 'white', 'fontWeight': 'bold'}
    )
    
    # Create Efficient Frontier plot
    frontier_fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Efficient Frontier - Risk vs Return', 'Optimal Portfolio Allocation'),
        specs=[[{'type': 'scatter'}, {'type': 'pie'}]]
    )
    
    frontier_fig.add_trace(
        go.Scatter(
            x=df['Volatility'],
            y=df['Return'],
            mode='markers',
            marker=dict(size=5, color=df['Sharpe Ratio'], colorscale='Viridis', showscale=True, colorbar=dict(title="Sharpe Ratio")),
            name='Portfolios'
        ),
        row=1, col=1
    )
    
    frontier_fig.add_trace(
        go.Scatter(
            x=[optimal_portfolio['Volatility']],
            y=[optimal_portfolio['Return']],
            mode='markers',
            marker=dict(size=20, color='red', symbol='star', line=dict(width=2, color='black')),
            name='Optimal Portfolio'
        ),
        row=1, col=1
    )
    
    # Pie chart
    weights = [optimal_portfolio[f'{ticker}_Weight'] for ticker in tickers]
    frontier_fig.add_trace(
        go.Pie(labels=tickers, values=weights, textinfo='label+percent'),
        row=1, col=2
    )
    
    frontier_fig.update_xaxes(title_text="Volatility (Risk)", row=1, col=1)
    frontier_fig.update_yaxes(title_text="Expected Return", row=1, col=1)
    frontier_fig.update_layout(height=500, showlegend=True)
    
    # Create correlation heatmap
    correlation_matrix = returns.corr()
    correlation_fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=tickers,
        y=tickers,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    correlation_fig.update_layout(
        title='Stock Correlation Matrix',
        height=500,
        xaxis={'side': 'bottom'}
    )
    
    # Create performance analysis
    performance_fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Normalized Price Evolution', 'Optimal Portfolio Performance', 
                       'Portfolio Returns Distribution', 'Risk-Return Analysis'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'histogram'}, {'type': 'scatter'}]]
    )
    
    # Normalized prices
    normalized_prices = data / data.iloc[0] * 100
    for ticker in tickers[:5]:
        performance_fig.add_trace(
            go.Scatter(x=normalized_prices.index, y=normalized_prices[ticker], name=ticker, mode='lines'),
            row=1, col=1
        )
    
    # Portfolio performance
    weights = np.array([optimal_portfolio[f'{ticker}_Weight'] for ticker in tickers])
    portfolio_value = (normalized_prices * weights).sum(axis=1)
    performance_fig.add_trace(
        go.Scatter(x=portfolio_value.index, y=portfolio_value, name='Portfolio', line=dict(color='red', width=3)),
        row=1, col=2
    )
    
    # Returns distribution
    portfolio_returns = (returns * weights).sum(axis=1)
    performance_fig.add_trace(
        go.Histogram(x=portfolio_returns, name='Returns', nbinsx=30),
        row=2, col=1
    )
    
    # Risk-Return scatter
    individual_returns = returns.mean() * 252
    individual_volatility = returns.std() * np.sqrt(252)
    
    performance_fig.add_trace(
        go.Scatter(
            x=individual_volatility,
            y=individual_returns,
            mode='markers+text',
            text=tickers,
            textposition='top center',
            marker=dict(size=10),
            name='Stocks'
        ),
        row=2, col=2
    )
    
    performance_fig.add_trace(
        go.Scatter(
            x=[optimal_portfolio['Volatility']],
            y=[optimal_portfolio['Return']],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            name='Portfolio'
        ),
        row=2, col=2
    )
    
    performance_fig.update_layout(height=800, showlegend=True)
    
    status_msg = f"Analysis complete! Optimized portfolio using {len(tickers)} stocks from {len(selected_sectors)} sectors."
    status_style = {'backgroundColor': '#2ecc71', 'color': 'white', 'padding': '10px', 'borderRadius': '5px'}
    
    return summary, allocation_table, frontier_fig, correlation_fig, performance_fig, status_msg, status_style


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
