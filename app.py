import gradio as gr
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

def optimize_portfolio(ticker_input, num_simulations):
    # 1. Parse Tickers & Fetch Data
    tickers = [t.strip().upper() for t in ticker_input.split(',')]
    if len(tickers) < 2:
        return "Please enter at least 2 tickers.", None
    
    try:
        data = yf.download(tickers, period='2y', progress=False)['Close']
        if data.empty:
            return "Error: No data retrieved. Check ticker symbols.", None
        
        # Handle single vs multiple tickers
        if len(tickers) == 1:
            data = data.to_frame()
        
        returns = data.pct_change().dropna()
        
        if returns.empty or len(returns) < 50:
            return "Error: Insufficient data for analysis.", None
            
    except Exception as e:
        return f"Error fetching data: {str(e)}", None
    
    # 2. Vectorized Monte Carlo Simulation
    try:
        mean_returns = returns.mean().to_numpy() * 252
        cov_matrix = returns.cov().to_numpy() * 252
        num_assets = len(tickers)
        
        # Generate random weights
        weights = np.random.random((int(num_simulations), num_assets))
        weights /= weights.sum(axis=1)[:, None]
        
        # Calculate metrics
        port_returns = np.dot(weights, mean_returns)
        port_vols = np.sqrt(np.einsum('ij,jk,ik->i', weights, cov_matrix, weights))
        sharpe_ratios = (port_returns - 0.04) / port_vols
        
        # Create DataFrame
        df = pd.DataFrame({
            'Volatility': port_vols, 
            'Return': port_returns, 
            'Sharpe': sharpe_ratios
        })
        
        # Add weights
        weight_df = pd.DataFrame(weights, columns=[f"w_{t}" for t in tickers])
        df = pd.concat([df, weight_df], axis=1)
        
    except Exception as e:
        return f"Error in optimization: {str(e)}", None
    
    # 3. Find Optimal Portfolio
    best_idx = df['Sharpe'].idxmax()
    best_port = df.iloc[best_idx]
    
    # 4. Create Interactive Plot
    try:
        fig = px.scatter(
            df, 
            x='Volatility', 
            y='Return', 
            color='Sharpe',
            title='Efficient Frontier (Interactive)',
            hover_data=[c for c in df.columns if 'w_' in c],
            color_continuous_scale='Viridis',
            labels={'Volatility': 'Risk (Volatility)', 'Return': 'Expected Return'}
        )
        
        # Mark the best portfolio
        fig.add_scatter(
            x=[best_port['Volatility']], 
            y=[best_port['Return']], 
            marker=dict(color='red', size=15, symbol='star'),
            name='Max Sharpe',
            showlegend=True
        )
        
        fig.update_layout(height=500)
        
    except Exception as e:
        return f"Error creating plot: {str(e)}", None
    
    # 5. Generate Text Summary
    summary = (
        f"**Optimal Portfolio (Max Sharpe: {best_port['Sharpe']:.2f})**\n\n"
        f"Expected Annual Return: {best_port['Return']:.2%}\n"
        f"Annual Risk (Volatility): {best_port['Volatility']:.2%}\n\n"
        "**Allocation:**\n"
    )
    
    for t in tickers:
        weight = best_port[f"w_{t}"]
        if weight > 0.01:
            summary += f"- **{t}**: {weight:.1%}\n"
    
    return summary, fig

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(), title="Portfolio Optimizer") as demo:
    gr.Markdown("# 📊 Simple Portfolio Optimizer")
    gr.Markdown("Enter stock tickers and optimize your portfolio using Modern Portfolio Theory")
    
    with gr.Row():
        t_input = gr.Textbox(
            value="AAPL, MSFT, GOOGL, AMZN", 
            label="Stock Tickers (comma separated)",
            placeholder="e.g., AAPL, MSFT, TSLA"
        )
        n_input = gr.Slider(
            1000, 20000, 
            value=5000, 
            step=1000, 
            label="Number of Simulations"
        )
    
    btn = gr.Button("🚀 Optimize Portfolio", variant="primary", size="lg")
    
    with gr.Row():
        with gr.Column(scale=1):
            txt_output = gr.Markdown(label="Best Allocation")
        with gr.Column(scale=2):
            plot_output = gr.Plot(label="Efficient Frontier")
    
    btn.click(
        optimize_portfolio, 
        inputs=[t_input, n_input], 
        outputs=[txt_output, plot_output]
    )
    
    gr.Markdown("**Disclaimer:** Educational purposes only. Not financial advice.")

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
