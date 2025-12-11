import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

# Stock universe with sectors
STOCK_UNIVERSE = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'],
    'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C'],
    'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
    'Consumer': ['PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD'],
    'Industrial': ['BA', 'CAT', 'GE', 'MMM', 'UPS']
}

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Portfolio Management Dashboard"),
    html.Div([
        html.Label("Select Sectors"),
        dcc.Checklist(
            id='sector-selection',
            options=[{'label': k, 'value': k} for k in STOCK_UNIVERSE.keys()],
            value=['Technology', 'Finance']
        ),
        html.Label("Risk Tolerance"),
        dcc.Slider(
            id='risk-tolerance',
            min=0.05, max=0.5, step=0.01, value=0.2,
            marks={i/10: str(i/10) for i in range(5, 51, 5)}
        ),
        html.Label("Number of Monte Carlo Simulations"),
        dcc.Slider(
            id='num-simulations',
            min=1000, max=50000, step=1000, value=10000,
            marks={i: str(i) for i in range(1000, 50001, 5000)}
        ),
        html.Label("Investment Amount"),
        dcc.Input(id='investment-amount', type='number', value=10000),
        html.Button('Run Optimization', id='run-button')
    ], style={'columnCount': 1, 'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    html.Div([
        html.H4("Portfolio Summary"),
        html.Div(id='portfolio-summary'),
        html.H4("Investment Allocation"),
        dash_table.DataTable(id='allocation-table', columns=[
            {'name': 'Ticker', 'id': 'Ticker'},
            {'name': 'Weight (%)', 'id': 'Weight (%)'},
            {'name': 'Allocation ($)', 'id': 'Allocation ($)'},
            {'name': 'Current Price', 'id': 'Current Price'},
            {'name': 'Shares to Buy', 'id': 'Shares to Buy'}
        ]),
        html.H4("Efficient Frontier & Allocation"),
        dcc.Graph(id='frontier-plot'),
        html.H4("Stock Correlation Matrix"),
        dcc.Graph(id='correlation-plot'),
        html.H4("Performance Analysis"),
        dcc.Graph(id='performance-plot'),
        html.Div(id='status')
    ], style={'columnCount': 1, 'width': '65%', 'display': 'inline-block', 'paddingLeft': '20px'})
])

def get_market_data(tickers, period='1y'):
    try:
        data = yf.download(tickers, period=period, progress=False, threads=True)
        if 'Adj Close' in data.columns:
            return data['Adj Close']
        else:
            return data['Close']
    except:
        return pd.DataFrame()

def monte_carlo_simulation(tickers, num_portfolios, risk_tolerance):
    data = get_market_data(tickers, period='1y')
    if data.empty:
        return None, None, None
    data = data.dropna()
    returns = data.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    results = []

    for _ in range(num_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        port_return = np.sum(mean_returns * weights)
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_return - 0.04) / port_std if port_std != 0 else 0
        results.append({
            'Return': port_return,
            'Volatility': port_std,
            'Sharpe': sharpe,
            **{f'{ticker}_Weight': weight for ticker, weight in zip(tickers, weights)}
        })
    df = pd.DataFrame(results)
    df = df[(df['Volatility'] > 0) & (df['Volatility'] < 1)]
    filtered_df = df[df['Volatility'] <= risk_tolerance]
    if not filtered_df.empty:
        optimal_idx = filtered_df['Sharpe'].idxmax()
        optimal_portfolio = filtered_df.loc[optimal_idx]
    else:
        optimal_portfolio = df.loc[df['Sharpe'].idxmax()] if not df.empty else None
    return df, optimal_portfolio, returns

def create_frontier_figure(df, optimal_portfolio):
    fig = go.Figure()
    if df is not None and not df.empty:
        fig.add_trace(go.Scatter(
            x=df['Volatility'], y=df['Return'],
            mode='markers', marker=dict(color=df['Sharpe'], colorscale='Viridis', size=8),
            text=df['Sharpe'], name='Portfolios'
        ))
        if optimal_portfolio is not None:
            fig.add_trace(go.Scatter(
                x=[optimal_portfolio['Volatility']],
                y=[optimal_portfolio['Return']],
                mode='markers', marker=dict(color='red', size=20, symbol='star'),
                name='Optimal Portfolio'
            ))
        fig.update_layout(title='Efficient Frontier', xaxis_title='Risk (Volatility)', yaxis_title='Return')
    return fig

def create_correlation_heatmap(returns):
    corr = returns.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale='RdBu', zmin=-1, zmax=1))
    fig.update_layout(title='Correlation Matrix')
    return fig

def create_performance_analysis(data, optimal_portfolio, tickers):
    fig = go.Figure()
    if data.empty or optimal_portfolio is None:
        return fig
    data = data.dropna()
    normalized_prices = data / data.iloc[0] * 100
    for ticker in tickers:
        if ticker in normalized_prices.columns:
            fig.add_trace(go.Scatter(
                x=normalized_prices.index, y=normalized_prices[ticker],
                mode='lines', name=ticker))
    # Portfolio performance
    weights = [optimal_portfolio.get(f'{ticker}_Weight', 0) for ticker in tickers]
    weights = np.array(weights)
    valid_tickers = [ticker for i, ticker in enumerate(tickers) if ticker in normalized_prices.columns]
    if len(valid_tickers) > 0 and sum(weights) > 0:
        weights /= sum(weights)
        portfolio_value = (normalized_prices[valid_tickers] * weights).sum(axis=1)
        fig.add_trace(go.Scatter(
            x=normalized_prices.index, y=portfolio_value,
            mode='lines', name='Optimal Portfolio', line=dict(color='black', width=3)))
    fig.update_layout(title='Performance Analysis', yaxis_title='Normalized Price')
    return fig

@app.callback(
    [Output('portfolio-summary', 'children'),
     Output('allocation-table', 'data'),
     Output('frontier-plot', 'figure'),
     Output('correlation-plot', 'figure'),
     Output('performance-plot', 'figure'),
     Output('status', 'children')],
    [Input('run-button', 'n_clicks')],
    [State('sector-selection', 'value'),
     State('risk-tolerance', 'value'),
     State('num-simulations', 'value'),
     State('investment-amount', 'value')]
)
def run_analysis(n_clicks, sectors, risk_tolerance, num_simulations, investment_amount):
    if n_clicks is None:
        return '', [], {}, {}, {}, ''
    
    # Collect tickers
    tickers = []
    for sec in sectors:
        tickers.extend(STOCK_UNIVERSE.get(sec, []))
    tickers = list(set(tickers))
    if len(tickers) < 2:
        return 'Select at least two stocks.', [], {}, {}, {}, ''
    
    df, optimal_portfolio, returns = monte_carlo_simulation(tickers, num_simulations, risk_tolerance)
    if optimal_portfolio is None:
        return 'Failed to optimize portfolio.', [], {}, {}, {}, ''
    
    frontier_fig = create_frontier_figure(df, optimal_portfolio)
    corr_fig = create_correlation_heatmap(returns)
    data = get_market_data(tickers, period='1y')
    performance_fig = create_performance_analysis(data, optimal_portfolio, tickers)
    # Prepare summary
    summary = f"""
    Expected Return: {optimal_portfolio['Return']:.2%}
    Volatility: {optimal_portfolio['Volatility']:.2%}
    Sharpe Ratio: {optimal_portfolio['Sharpe']:.2f}
    """
    # Prepare allocation data
    alloc_data = []
    total_investment = investment_amount
    for ticker in tickers:
        weight = optimal_portfolio.get(f'{ticker}_Weight', 0)
        alloc = weight * total_investment
        try:
            current_price = yf.Ticker(ticker).history(period='5d')['Close'].iloc[-1]
        except:
            current_price = 100
        shares = int(alloc / current_price)
        alloc_data.append({
            'Ticker': ticker,
            'Weight (%)': f"{weight*100:.2f}",
            'Allocation ($)': f"${alloc:,.2f}",
            'Current Price': f"${current_price:.2f}",
            'Shares to Buy': shares
        })
    return summary, alloc_data, frontier_fig, corr_fig, performance_fig, ''

if __name__ == '__main__':
    app.run_server(debug=True)
