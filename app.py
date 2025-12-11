from dash.exceptions import PreventUpdate

def monte_carlo_simulation(tickers, num_portfolios=10000, risk_tolerance=0.2):
    if len(tickers) == 0:
        return None, None, None, None, None

    if len(tickers) > 10:
        tickers = tickers[:10]

    data = get_market_data(tickers)
    if data.empty or len(data.columns) < 2:
        return None, None, None, None, None

    returns = data.pct_change().dropna()
    if returns.empty:
        return None, None, None, None, None

    mean_returns = returns.mean().values * 252
    cov_matrix = returns.cov().values * 252
    num_assets = len(tickers)

    weights = np.random.random((num_portfolios, num_assets))
    weights /= np.sum(weights, axis=1)[:, np.newaxis]

    port_returns = np.dot(weights, mean_returns)
    port_variance = np.sum(np.dot(weights, cov_matrix) * weights, axis=1)
    port_std = np.sqrt(port_variance)

    sharpe_ratios = (port_returns - 0.04) / port_std

    results = {
        'Return': port_returns,
        'Volatility': port_std,
        'Sharpe Ratio': sharpe_ratios
    }
    for i, ticker in enumerate(tickers):
        results[f'{ticker}_Weight'] = weights[:, i]

    df = pd.DataFrame(results)

    valid_risk = df[df['Volatility'] <= risk_tolerance]
    if not valid_risk.empty:
        optimal_idx = valid_risk['Sharpe Ratio'].idxmax()
        optimal_portfolio = valid_risk.loc[optimal_idx]
    else:
        optimal_idx = df['Sharpe Ratio'].idxmax()
        optimal_portfolio = df.loc[optimal_idx]

    return df, optimal_portfolio, returns, data, tickers


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
    empty_fig = go.Figure()

    if n_clicks == 0:
        return "", "", empty_fig, empty_fig, empty_fig, "", {'display': 'none'}

    if not selected_sectors:
        status_msg = "Select at least one sector"
        status_style = {'backgroundColor': '#e74c3c', 'color': 'white', 'padding': '10px'}
        return "", "", empty_fig, empty_fig, empty_fig, status_msg, status_style

    all_tickers = []
    for sector in selected_sectors:
        all_tickers.extend(STOCK_UNIVERSE[sector])
    all_tickers = list(set(all_tickers))

    df, optimal_portfolio, returns, data, used_tickers = monte_carlo_simulation(
        all_tickers, int(num_portfolios), risk_tolerance
    )

    if df is None or optimal_portfolio is None or returns is None or data is None or not used_tickers:
        status_msg = "Data fetch or simulation failed. Try fewer stocks or another period."
        status_style = {'backgroundColor': '#e74c3c', 'color': 'white', 'padding': '10px'}
        return "", "", empty_fig, empty_fig, empty_fig, status_msg, status_style

    summary = html.Div([
        html.H3("Portfolio Summary", style={'color': '#2c3e50'}),
        html.P([html.Strong("Expected Annual Return: "), f"{optimal_portfolio['Return']:.2%}"]),
        html.P([html.Strong("Portfolio Volatility: "), f"{optimal_portfolio['Volatility']:.2%}"]),
        html.P([html.Strong("Sharpe Ratio: "), f"{optimal_portfolio['Sharpe Ratio']:.2f}"]),
        html.P([html.Strong("Total Investment: "), f"{investment_amount:,.2f}"]),
    ])

    allocation_data = []
    current_prices = data.iloc[-1]

    for ticker in used_tickers:
        weight = float(optimal_portfolio.get(f'{ticker}_Weight', 0.0))
        if weight > 0.001:
            price = float(current_prices.get(ticker, 0.0))
            allocation = weight * investment_amount
            shares = int(allocation / price) if price > 0 else 0

            allocation_data.append({
                'Ticker': ticker,
                'Weight': f"{weight * 100:.1f}%",
                'Allocation': f"{allocation:,.2f}",
                'Price': f"{price:.2f}",
                'Shares': shares
            })

    if allocation_data:
        allocation_df = pd.DataFrame(allocation_data).sort_values('Allocation', ascending=False)
        allocation_table = dash_table.DataTable(
            data=allocation_df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in allocation_df.columns],
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_header={'backgroundColor': '#3498db', 'color': 'white'}
        )
    else:
        allocation_table = html.P("No significant weights found for this configuration.")

    frontier_fig = px.scatter(
        df, x='Volatility', y='Return', color='Sharpe Ratio',
        title='Efficient Frontier', hover_data=['Sharpe Ratio']
    )
    frontier_fig.add_trace(
        go.Scatter(
            x=[optimal_portfolio['Volatility']],
            y=[optimal_portfolio['Return']],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            name='Optimal Portfolio'
        )
    )

    corr_fig = px.imshow(
        returns.corr(), aspect="auto",
        title="Stock Correlation Matrix", color_continuous_scale='RdBu_r'
    )

    normalized_prices = data / data.iloc[0] * 100
    perf_fig = px.line(normalized_prices, title="Price Evolution (Normalized)")

    status_msg = "Success. Portfolio optimized."
    status_style = {'backgroundColor': '#2ecc71', 'color': 'white', 'padding': '10px'}

    return summary, allocation_table, frontier_fig, corr_fig, perf_fig, status_msg, status_style
