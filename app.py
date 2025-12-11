import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import warnings
from datetime import datetime

# Suppress pandas/yfinance warnings
warnings.filterfilterwarnings('ignore')

# --- 1. CONFIGURATION ---
STOCK_UNIVERSE = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'],
    'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C'],
    'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
    'Consumer': ['PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD'],
    'Industrial': ['BA', 'CAT', 'GE', 'MMM', 'UPS']
}
DEFAULT_SECTORS = ['Technology', 'Finance']
RISK_FREE_RATE = 0.04
PERIOD = '1y'  # Reduced period for faster execution

# --- 2. BACKEND FUNCTIONS (Simplified) ---

def get_market_data(tickers, period=PERIOD):
    """Fetch market data using yfinance."""
    print(f"Fetching data for {len(tickers)} tickers for {period}")
    try:
        data = yf.download(tickers, period=period, progress=False)
        if data.empty:
            return pd.DataFrame()
        
        # Ensure we get 'Adj Close' column for multiple or single tickers
        if 'Adj Close' in data.columns and len(tickers) > 1:
            return data['Adj Close'].dropna()
        elif 'Adj Close' in data.columns and len(tickers) == 1:
            return pd.DataFrame({tickers[0]: data['Adj Close']}).dropna()
        elif 'Close' in data.columns and len(tickers) == 1:
            return pd.DataFrame({tickers[0]: data['Close']}).dropna()
        elif 'Close' in data.columns and len(tickers) > 1:
            return data['Close'].dropna()
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def monte_carlo_simulation(tickers, num_portfolios, risk_tolerance):
    """
    Performs Monte Carlo simulation for portfolio optimization.
    Returns: df (results), optimal_portfolio (Series), returns (DataFrame), tickers (list)
    """
    
    if len(tickers) < 2:
        return None, None, None, []

    # Limit to a maximum of 10 tickers for stability
    if len(tickers) > 10:
        tickers = tickers[:10]
        
    data = get_market_data(tickers)
    
    if data.empty or len(data) < 100:
        print("Data is empty or insufficient for analysis.")
        return None, None, None, []
        
    returns = data.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    results = np.zeros((num_portfolios, 3 + len(tickers)))
    
    for i in range(num_portfolios):
        # Generate random weights
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        
        # Calculate portfolio metrics
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Sharpe Ratio (using Risk_Free_Rate)
        sharpe_ratio = (portfolio_return - RISK_FREE_RATE) / portfolio_std if portfolio_std != 0 else 0
        
        results[i, 0] = portfolio_return
        results[i, 1] = portfolio_std
        results[i, 2] = sharpe_ratio
        results[i, 3:] = weights
    
    # Create DataFrame
    columns = ['Return', 'Volatility', 'Sharpe Ratio'] + [f'{ticker}_Weight' for ticker in tickers]
    df = pd.DataFrame(results, columns=columns)
    df = df[(df['Volatility'] > 0) & (np.isfinite(df['Sharpe Ratio']))]
    
    if df.empty:
        return None, None, None, []
        
    # Find Optimal Portfolio (Max Sharpe, constrained by risk_tolerance)
    filtered_df = df[df['Volatility'] <= risk_tolerance]
    if not filtered_df.empty:
        optimal_portfolio = filtered_df.loc[filtered_df['Sharpe Ratio'].idxmax()]
    else:
        # Fallback to Max Sharpe from unconstrained set
        optimal_portfolio = df.loc[df['Sharpe Ratio'].idxmax()]

    return df, optimal_portfolio, returns, tickers

def create_efficient_frontier_plot(df, optimal_portfolio):
    """Creates an interactive Plotly Efficient Frontier scatter plot."""
    if df is None or optimal_portfolio is None or df.empty:
        return go.Figure().update_layout(title="Efficient Frontier (No Data)", height=400)

    # Convert the optimal portfolio to a DataFrame for scatter trace
    optimal_df = optimal_portfolio.to_frame().T
    
    fig = px.scatter(
        df, 
        x='Volatility', 
        y='Return', 
        color='Sharpe Ratio',
        hover_data={'Volatility': ':.2%', 'Return': ':.2%', 'Sharpe Ratio': ':.4f'},
        title="Efficient Frontier: Risk vs. Return",
        labels={'Volatility': 'Annual Volatility (Risk)', 'Return': 'Expected Annual Return'},
        color_continuous_scale=px.colors.sequential.Viridis,
        height=500
    )
    
    # Add Optimal Portfolio point
    fig.add_trace(go.Scatter(
        x=optimal_df['Volatility'], 
        y=optimal_df['Return'], 
        mode='markers',
        marker=dict(size=15, color='red', symbol='star', line=dict(width=1, color='black')),
        name=f"Optimal Portfolio (Sharpe: {optimal_df['Sharpe Ratio'].iloc[0]:.4f})",
        hoverinfo='text',
        hovertext=f"Return: {optimal_df['Return'].iloc[0]:.2%}<br>Volatility: {optimal_df['Volatility'].iloc[0]:.2%}<br>Sharpe: {optimal_df['Sharpe Ratio'].iloc[0]:.4f}"
    ))

    fig.update_layout(legend_title_text='Sharpe Ratio', margin={"t":40, "l":20, "b":20, "r":20})
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    
    return fig

def create_allocation_pie_chart(optimal_portfolio, tickers):
    """Creates a Plotly pie chart for optimal asset allocation."""
    if optimal_portfolio is None:
        return go.Figure().update_layout(title="Portfolio Allocation (No Data)", height=400)

    weights = []
    labels = []
    other_weight = 0
    
    for ticker in tickers:
        weight_col = f'{ticker}_Weight'
        if weight_col in optimal_portfolio:
            weight = optimal_portfolio[weight_col]
            if weight > 0.02:  # Show only weights > 2%
                weights.append(weight)
                labels.append(ticker)
            else:
                other_weight += weight
                
    if other_weight > 0:
        weights.append(other_weight)
        labels.append('Others')
        
    if not weights:
        return go.Figure().update_layout(title="Portfolio Allocation (No Significant Weights)", height=400)
        
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=weights, 
        hole=.3, 
        marker={'colors': px.colors.sequential.Sunset_r}
    )])
    
    fig.update_layout(
        title_text="Optimal Portfolio Allocation",
        margin={"t":40, "l":20, "b":20, "r":20},
        height=500
    )
    
    return fig

def create_correlation_heatmap(returns, tickers):
    """Creates a Plotly correlation heatmap."""
    if returns is None or returns.empty:
        return go.Figure().update_layout(title="Stock Correlation (No Data)", height=400)
    
    correlation_matrix = returns.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='coolwarm',
        zmin=-1,
        zmax=1,
        text=correlation_matrix.apply(lambda x: [f'{val:.2f}' for val in x], axis=1).values,
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Stock Correlation Matrix',
        xaxis_nticks=len(tickers),
        yaxis_nticks=len(tickers),
        height=500,
        margin={"t":40, "l":20, "b":20, "r":20}
    )
    return fig

# --- 3. DASH APPLICATION LAYOUT ---

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

# Style for better look and feel
main_style = {
    'fontFamily': 'Arial, sans-serif',
    'maxWidth': '1200px',
    'margin': 'auto',
    'padding': '20px'
}

app.layout = html.Div(style=main_style, children=[
    html.H1("AI Portfolio Management Assistant (Dash)", style={'textAlign': 'center'}),
    html.Div("Professional-grade portfolio optimization using Monte Carlo simulation and Modern Portfolio Theory.", style={'textAlign': 'center', 'marginBottom': '30px'}),

    # --- INPUT CONTROLS ---
    html.Div(className='row', children=[
        html.Div(className='six columns', children=[
            html.Label("Select Sectors (Stocks will be limited to 8 max):"),
            dcc.Checklist(
                id='sector-selection',
                options=[{'label': sector, 'value': sector} for sector in STOCK_UNIVERSE.keys()],
                value=DEFAULT_SECTORS,
                inline=True,
                style={'marginBottom': '20px'}
            ),
        ]),
        html.Div(className='three columns', children=[
            html.Label(id='risk-tolerance-label'),
            dcc.Slider(
                id='risk-tolerance-slider',
                min=0.05,
                max=0.50,
                step=0.01,
                value=0.20,
                marks={i / 100: f'{i}%' for i in range(5, 51, 5)}
            ),
        ]),
        html.Div(className='three columns', children=[
            html.Label(id='num-portfolios-label'),
            dcc.Slider(
                id='num-portfolios-slider',
                min=1000,
                max=20000, # Max reduced for stability
                step=1000,
                value=10000,
                marks={i: f'{i/1000}k' for i in range(1000, 20001, 5000)}
            ),
        ]),
    ], style={'marginBottom': '20px'}),
    
    html.Button('Optimize Portfolio', id='optimize-button', n_clicks=0, className='button-primary', style={'width': '100%', 'marginBottom': '30px'}),

    dcc.Loading(
        id="loading-1",
        type="default",
        children=[
            html.Div(id='status-message', style={'textAlign': 'center', 'fontWeight': 'bold', 'marginBottom': '20px'}),
            
            # --- OUTPUT GRAPHS ---
            html.Div(className='row', children=[
                html.Div(className='six columns', children=[
                    dcc.Graph(id='efficient-frontier-graph')
                ]),
                html.Div(className='six columns', children=[
                    dcc.Graph(id='allocation-pie-graph')
                ]),
            ]),
            
            html.Div(className='row', children=[
                html.Div(className='six columns', children=[
                    dcc.Graph(id='correlation-heatmap-graph')
                ]),
                html.Div(className='six columns', children=[
                    html.H4("Portfolio Summary", style={'textAlign': 'center'}),
                    dcc.Markdown(id='summary-markdown'),
                    html.H4("Investment Allocation", style={'textAlign': 'center', 'marginTop': '20px'}),
                    html.Div(id='allocation-table-div')
                ]),
            ])
        ]
    )
])

# --- 4. DASH CALLBACKS ---

# Callback for updating slider labels
@app.callback(
    [Output('risk-tolerance-label', 'children'),
     Output('num-portfolios-label', 'children')],
    [Input('risk-tolerance-slider', 'value'),
     Input('num-portfolios-slider', 'value')]
)
def update_slider_labels(risk_tolerance, num_portfolios):
    risk_label = f"Risk Tolerance (Max Volatility: {risk_tolerance:.0%})"
    num_label = f"Monte Carlo Simulations: {num_portfolios:,}"
    return risk_label, num_label

# Main Optimization Callback
@app.callback(
    [Output('status-message', 'children'),
     Output('efficient-frontier-graph', 'figure'),
     Output('allocation-pie-graph', 'figure'),
     Output('correlation-heatmap-graph', 'figure'),
     Output('summary-markdown', 'children'),
     Output('allocation-table-div', 'children')],
    [Input('optimize-button', 'n_clicks')],
    [dash.dependencies.State('sector-selection', 'value'),
     dash.dependencies.State('risk-tolerance-slider', 'value'),
     dash.dependencies.State('num-portfolios-slider', 'value')]
)
def update_output(n_clicks, selected_sectors, risk_tolerance, num_portfolios):
    # Initial status message and empty plots
    initial_status = "Click 'Optimize Portfolio' to start the analysis."
    empty_fig = go.Figure().update_layout(title="Awaiting Data", height=400)
    empty_df = pd.DataFrame()
    
    if n_clicks == 0:
        return initial_status, empty_fig, empty_fig, empty_fig, "", dash_table.DataTable(id='empty-table', columns=[{'name': i, 'id': i} for i in ["Ticker", "Weight (%)"]], data=empty_df.to_dict('records'))

    if not selected_sectors:
        return "❌ Please select at least one sector.", empty_fig, empty_fig, empty_fig, "", dash_table.DataTable(id='empty-table', columns=[{'name': i, 'id': i} for i in ["Ticker", "Weight (%)"]], data=empty_df.to_dict('records'))

    # Build ticker list
    tickers = []
    for sector in selected_sectors:
        tickers.extend(STOCK_UNIVERSE.get(sector, []))
    tickers = list(set(tickers))
    
    if len(tickers) < 2:
        return "❌ Please select sectors with at least 2 different stocks.", empty_fig, empty_fig, empty_fig, "", dash_table.DataTable(id='empty-table', columns=[{'name': i, 'id': i} for i in ["Ticker", "Weight (%)"]], data=empty_df.to_dict('records'))

    # Run Optimization
    try:
        df, optimal_portfolio, returns, final_tickers = monte_carlo_simulation(tickers, num_portfolios, risk_tolerance)
        
        if df is None:
            return "❌ Unable to fetch market data or perform simulation. Try again or check sector selection.", empty_fig, empty_fig, empty_fig, "", dash_table.DataTable(id='empty-table', columns=[{'name': i, 'id': i} for i in ["Ticker", "Weight (%)"]], data=empty_df.to_dict('records'))

        # 1. Plots
        frontier_plot = create_efficient_frontier_plot(df, optimal_portfolio)
        allocation_plot = create_allocation_pie_chart(optimal_portfolio, final_tickers)
        correlation_plot = create_correlation_heatmap(returns, final_tickers)

        # 2. Summary
        expected_return = optimal_portfolio['Return']
        volatility = optimal_portfolio['Volatility']
        sharpe_ratio = optimal_portfolio['Sharpe Ratio']
        
        risk_level = "🟢 Low Risk" if volatility < 0.15 else "🟡 Moderate Risk" if volatility < 0.25 else "🔴 High Risk"
        
        summary_markdown = f"""
- **Expected Annual Return:** {expected_return:.2%}
- **Portfolio Volatility:** {volatility:.2%} ({risk_level})
- **Sharpe Ratio:** {sharpe_ratio:.4f}
- **Assets Analyzed:** {len(final_tickers)}
- **Simulation Count:** {num_portfolios:,}
"""

        # 3. Allocation Table
        allocation_data = []
        for ticker in final_tickers:
            weight_col = f'{ticker}_Weight'
            if weight_col in optimal_portfolio:
                allocation_data.append({
                    'Ticker': ticker,
                    'Weight (%)': f"{optimal_portfolio[weight_col]*100:.2f}"
                })
        
        allocation_df = pd.DataFrame(allocation_data).sort_values(by='Weight (%)', ascending=False)
        
        allocation_table = dash_table.DataTable(
            id='allocation-table',
            columns=[
                {"name": "Ticker", "id": "Ticker"},
                {"name": "Weight (%)", "id": "Weight (%)"},
            ],
            data=allocation_df.to_dict('records'),
            style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
            style_data_conditional=[
                {'if': {'column_id': 'Weight (%)'}, 'textAlign': 'right'},
            ]
        )
        
        status_message = f"✅ Analysis complete! Optimal Portfolio found with Sharpe Ratio: {sharpe_ratio:.4f}"
        
        return status_message, frontier_plot, allocation_plot, correlation_plot, summary_markdown, allocation_table

    except Exception as e:
        error_message = f"❌ An unexpected error occurred: {str(e)}"
        print(f"Error in main callback: {e}")
        return error_message, empty_fig, empty_fig, empty_fig, "", dash_table.DataTable(id='empty-table', columns=[{'name': i, 'id': i} for i in ["Ticker", "Weight (%)"]], data=empty_df.to_dict('records'))

if __name__ == '__main__':
    app.run_server(debug=True)
