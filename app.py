import gradio as gr
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

def optimize_portfolio(ticker_input, num_simulations):
    # 1. Parse Tickers & Fetch Data
    tickers = [t.strip().upper() for t in ticker_input.split(',')]
    if len(tickers) < 2:
        return "Please enter at least 2 tickers.", None

    try:
        data = yf.download(tickers, period='2y', progress=False)['Close']
        returns = data.pct_change().dropna()
    except Exception as e:
        return f"Error fetching data: {str(e)}", None

    # 2. Vectorized Monte Carlo Simulation (Faster & Simpler)
    mean_returns = returns.mean().to_numpy() * 252
    cov_matrix = returns.cov().to_numpy() * 252
    num_assets = len(tickers)

    # Generate random weights for all simulations at once
    weights = np.random.random((num_simulations, num_assets))
    weights /= weights.sum(axis=1)[:, None]

    # Calculate metrics in bulk
    port_returns = np.dot(weights, mean_returns)
    port_vols = np.sqrt(np.einsum('ij,ji->i', np.dot(weights, cov_matrix), weights.T))
    sharpe_ratios = (port_returns - 0.04) / port_vols

    # Create DataFrame
    df = pd.DataFrame({'Volatility': port_vols, 'Return': port_returns, 'Sharpe': sharpe_ratios})
    
    # Add weights to DataFrame for hover info
    weight_df = pd.DataFrame(weights, columns=[f"w_{t}" for t in tickers])
    df = pd.concat([df, weight_df], axis=1)

    # 3. Find Optimal Portfolio (Max Sharpe)
    best_idx = df['Sharpe'].idxmax()
    best_port = df.iloc[best_idx]

    # 4. Create Interactive Plot
    fig = px.scatter(df, x='Volatility', y='Return', color='Sharpe',
                     title='Efficient Frontier (Interactive)',
                     hover_data=[c for c in df.columns if 'w_' in c],
                     color_continuous_scale='Viridis')
    
    # Mark the best portfolio
    fig.add_scatter(x=[best_port['Volatility']], y=[best_port['Return']], 
                    marker=dict(color='red', size=15, symbol='star'),
                    name='Max Sharpe')

    # 5. Generate Text Summary
    summary = (f"**Optimal Portfolio (Max Sharpe: {best_port['Sharpe']:.2f})**\n\n"
               f"Expected Return: {best_port['Return']:.2%}\n"
               f"Annual Risk:     {best_port['Volatility']:.2%}\n\n"
               "**Allocation:**\n")
    
    for t in tickers:
        weight = best_port[f"w_{t}"]
        if weight > 0.01: # Only show significant allocations
            summary += f"- {t}: {weight:.1%}\n"

    return summary, fig

# --- Simple Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Simple Portfolio Optimizer")
    
    with gr.Row():
        t_input = gr.Textbox(value="AAPL, MSFT, GOOGL, AMZN", label="Tickers (comma separated)")
        n_input = gr.Slider(1000, 20000, value=5000, step=1000, label="Simulations")
    
    btn = gr.Button("Optimize", variant="primary")
    
    with gr.Row():
        txt_output = gr.Markdown(label="Best Allocation")
        plot_output = gr.Plot(label="Efficient Frontier")

    btn.click(optimize_portfolio, inputs=[t_input, n_input], outputs=[txt_output, plot_output])

if __name__ == "__main__":
    demo.launch()
