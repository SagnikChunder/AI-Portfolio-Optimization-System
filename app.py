import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import datetime
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


def get_market_data(tickers, period='1y'):
    """Fetch market data with error handling"""
    try:
        for attempt in range(3):
            try:
                data = yf.download(tickers, period=period, progress=False, threads=True)
                
                if data.empty:
                    continue
                
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
                if attempt < 2:
                    import time
                    time.sleep(2)
                continue
                
        return pd.DataFrame()
        
    except Exception as e:
        return pd.DataFrame()


def monte_carlo_simulation(tickers, num_portfolios=10000, risk_tolerance=0.2):
    """Monte Carlo simulation for portfolio optimization"""
    try:
        if len(tickers) > 10:
            tickers = tickers[:10]
        
        data = get_market_data(tickers, period='1y')
        
        if data.empty:
            fallback_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
            data = get_market_data(fallback_tickers, period='1y')
            if not data.empty:
                tickers = fallback_tickers
            else:
                return None, None, None, None
        
        data = data.dropna()
        if len(data) < 100:
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
            
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            if portfolio_std == 0:
                sharpe_ratio = 0
            else:
                sharpe_ratio = (portfolio_return - 0.04) / portfolio_std
            
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
        return None, None, None, None


def create_efficient_frontier(df, optimal_portfolio, tickers):
    """Create efficient frontier visualization"""
    fig = go.Figure()
    
    if df is None or df.empty:
        return fig
    
    valid_data = df[(df['Volatility'] > 0) & (df['Return'].notna()) & (df['Sharpe Ratio'].notna())]
    
    if len(valid_data) > 0:
        fig.add_trace(go.Scatter(
            x=valid_data['Volatility'],
            y=valid_data['Return'],
            mode='markers',
            marker=dict(
                color=valid_data['Sharpe Ratio'],
                colorscale='Viridis',
                size=5,
                colorbar=dict(title="Sharpe Ratio")
            ),
            name='Portfolios'
        ))
        
        if optimal_portfolio is not None:
            fig.add_trace(go.Scatter(
                x=[optimal_portfolio['Volatility']],
                y=[optimal_portfolio['Return']],
                mode='markers',
                marker=dict(color='red', size=20, symbol='star'),
                name='Optimal Portfolio'
            ))
    
    fig.update_layout(
        title='Efficient Frontier - Risk vs Return',
        xaxis_title='Volatility (Risk)',
        yaxis_title='Expected Return',
        hovermode='closest',
        height=500
    )
    
    return fig


def create_allocation_pie(optimal_portfolio, tickers):
    """Create portfolio allocation pie chart"""
    fig = go.Figure()
    
    if optimal_portfolio is None:
        return fig
    
    weights = []
    valid_tickers = []
    
    for ticker in tickers:
        weight_col = f'{ticker}_Weight'
        if weight_col in optimal_portfolio and optimal_portfolio[weight_col] > 0.02:
            weights.append(optimal_portfolio[weight_col])
            valid_tickers.append(ticker)
    
    if weights:
        fig.add_trace(go.Pie(
            labels=valid_tickers,
            values=weights,
            hole=0.3
        ))
        
        fig.update_layout(
            title='Optimal Portfolio Allocation',
            height=500
        )
    
    return fig


def create_correlation_heatmap(returns, tickers):
    """Create correlation heatmap"""
    fig = go.Figure()
    
    if returns is None or returns.empty:
        return fig
    
    correlation_matrix = returns.corr()
    
    fig.add_trace(go.Heatmap(
        z=correlation_matrix.values,
        x=tickers,
        y=tickers,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title='Stock Correlation Matrix',
        height=600
    )
    
    return fig


def generate_investment_report(optimal_portfolio, tickers, investment_amount):
    """Generate investment allocation table"""
    try:
        weights = [optimal_portfolio[f'{ticker}_Weight'] for ticker in tickers]
        allocations = [weight * investment_amount for weight in weights]
        
        current_prices = {}
        shares = {}
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period='5d')
                if not hist.empty:
                    current_prices[ticker] = hist['Close'].iloc[-1]
                    shares[ticker] = max(0, int(allocations[tickers.index(ticker)] / current_prices[ticker]))
                else:
                    current_prices[ticker] = 100.0
                    shares[ticker] = max(0, int(allocations[tickers.index(ticker)] / current_prices[ticker]))
            except:
                current_prices[ticker] = 100.0
                shares[ticker] = max(0, int(allocations[tickers.index(ticker)] / current_prices[ticker]))
        
        report_data = []
        for i, ticker in enumerate(tickers):
            report_data.append({
                'Ticker': ticker,
                'Weight': f"{weights[i]*100:.2f}%",
                'Allocation': f"${allocations[i]:,.2f}",
                'Current Price': f"${current_prices[ticker]:.2f}",
                'Shares': shares[ticker]
            })
        
        return pd.DataFrame(report_data)
        
    except Exception as e:
        return pd.DataFrame()


# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout
app.layout = dbc.Container([
    html.H1("AI-Powered Portfolio Management Assistant", className="text-center my-4"),
    html.P("Professional portfolio optimization using Modern Portfolio Theory and Monte Carlo simulation",
           className="text-center text-muted mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.H4("Configuration"),
            html.Hr(),
            
            html.Label("Select Sectors:"),
            dcc.Checklist(
                id='sector-selection',
                options=[{'label': sector, 'value': sector} for sector in STOCK_UNIVERSE.keys()],
                value=['Technology', 'Finance'],
                labelStyle={'display': 'block', 'margin': '5px'}
            ),
            
            html.Br(),
            html.Label("Risk Tolerance (Max Volatility):"),
            dcc.Slider(
                id='risk-tolerance',
                min=0.05,
                max=0.50,
                step=0.01,
                value=0.20,
                marks={0.05: '0.05', 0.20: '0.20', 0.35: '0.35', 0.50: '0.50'},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            
            html.Br(),
            html.Label("Monte Carlo Simulations:"),
            dcc.Slider(
                id='num-portfolios',
                min=1000,
                max=50000,
                step=1000,
                value=10000,
                marks={1000: '1K', 10000: '10K', 30000: '30K', 50000: '50K'},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            
            html.Br(),
            html.Label("Investment Amount ($):"),
            dcc.Input(
                id='investment-amount',
                type='number',
                value=10000,
                style={'width': '100%'}
            ),
            
            html.Br(),
            html.Br(),
            dbc.Button("Optimize Portfolio", id="optimize-btn", color="primary", className="w-100"),
            
            html.Br(),
            html.Br(),
            html.Div(id='status-message')
            
        ], width=3),
        
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="Summary", tab_id="summary-tab"),
                dbc.Tab(label="Efficient Frontier", tab_id="frontier-tab"),
                dbc.Tab(label="Allocation", tab_id="allocation-tab"),
                dbc.Tab(label="Correlation", tab_id="correlation-tab"),
            ], id="tabs", active_tab="summary-tab"),
            
            html.Div(id='tab-content', className="mt-3")
            
        ], width=9)
    ]),
    
    html.Hr(),
    html.P("Disclaimer: This tool is for educational purposes only. Past performance does not guarantee future results. "
           "Always consult with a financial advisor before making investment decisions.",
           className="text-center text-muted small")
    
], fluid=True)


@app.callback(
    [Output('tab-content', 'children'),
     Output('status-message', 'children')],
    [Input('optimize-btn', 'n_clicks'),
     Input('tabs', 'active_tab')],
    [State('sector-selection', 'value'),
     State('risk-tolerance', 'value'),
     State('num-portfolios', 'value'),
     State('investment-amount', 'value')]
)
def update_content(n_clicks, active_tab, selected_sectors, risk_tolerance, num_portfolios, investment_amount):
    """Main callback for portfolio optimization"""
    
    if n_clicks is None:
        return html.Div("Configure settings and click 'Optimize Portfolio' to begin"), ""
    
    if not selected_sectors:
        return html.Div("Please select at least one sector"), html.Div("Error: No sectors selected", className="text-danger")
    
    if investment_amount is None or investment_amount <= 0:
        return html.Div("Please enter a valid investment amount"), html.Div("Error: Invalid investment amount", className="text-danger")
    
    # Get selected tickers
    tickers = []
    for sector in selected_sectors:
        if sector in STOCK_UNIVERSE:
            tickers.extend(STOCK_UNIVERSE[sector])
    
    tickers = list(set(tickers))
    
    if len(tickers) < 2:
        return html.Div("Please select sectors with at least 2 different stocks"), html.Div("Error: Insufficient stocks", className="text-danger")
    
    if len(tickers) > 8:
        tickers = tickers[:8]
    
    # Run Monte Carlo simulation
    df, optimal_portfolio, returns, data = monte_carlo_simulation(tickers, int(num_portfolios), risk_tolerance)
    
    if df is None or optimal_portfolio is None:
        return html.Div("Unable to fetch market data or optimize portfolio. Please try again."), html.Div("Error: Optimization failed", className="text-danger")
    
    # Generate content based on active tab
    if active_tab == "summary-tab":
        report_df = generate_investment_report(optimal_portfolio, tickers, investment_amount)
        
        summary_stats = dbc.Card([
            dbc.CardBody([
                html.H4("Portfolio Summary"),
                html.Hr(),
                html.P([
                    html.Strong("Expected Annual Return: "),
                    f"{optimal_portfolio['Return']:.2%}"
                ]),
                html.P([
                    html.Strong("Portfolio Volatility: "),
                    f"{optimal_portfolio['Volatility']:.2%}"
                ]),
                html.P([
                    html.Strong("Sharpe Ratio: "),
                    f"{optimal_portfolio['Sharpe Ratio']:.4f}"
                ]),
                html.P([
                    html.Strong("Total Investment: "),
                    f"${investment_amount:,.2f}"
                ]),
                html.Hr(),
                html.H5("Investment Allocation"),
                dbc.Table.from_dataframe(report_df, striped=True, bordered=True, hover=True)
            ])
        ])
        
        status = html.Div(f"Analysis complete! Optimized portfolio using {len(tickers)} stocks from {len(selected_sectors)} sectors.", 
                         className="text-success")
        
        return summary_stats, status
    
    elif active_tab == "frontier-tab":
        fig = create_efficient_frontier(df, optimal_portfolio, tickers)
        status = html.Div(f"Efficient frontier generated with {int(num_portfolios)} simulations", className="text-success")
        return dcc.Graph(figure=fig), status
    
    elif active_tab == "allocation-tab":
        fig = create_allocation_pie(optimal_portfolio, tickers)
        status = html.Div("Portfolio allocation visualization ready", className="text-success")
        return dcc.Graph(figure=fig), status
    
    elif active_tab == "correlation-tab":
        fig = create_correlation_heatmap(returns, tickers)
        status = html.Div("Correlation analysis complete", className="text-success")
        return dcc.Graph(figure=fig), status
    
    return html.Div("Select a tab to view results"), ""


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
