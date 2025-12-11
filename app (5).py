import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Expanded stock universe with sectors
STOCK_UNIVERSE = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'],
    'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C'],
    'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
    'Consumer': ['PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD'],
    'Industrial': ['BA', 'CAT', 'GE', 'MMM', 'UPS']
}

def get_market_data(tickers, period='2y'):
    """Fetch market data with robust error handling"""
    try:
        # Try downloading data with multiple attempts
        for attempt in range(3):
            try:
                print(f"Fetching data for {len(tickers)} tickers (attempt {attempt + 1})")
                data = yf.download(tickers, period=period, progress=False, threads=True)
                
                if data.empty:
                    print(f"No data returned for tickers: {tickers}")
                    continue
                
                # Handle single ticker case
                if len(tickers) == 1:
                    if 'Adj Close' in data.columns:
                        return pd.DataFrame({tickers[0]: data['Adj Close']})
                    else:
                        return pd.DataFrame({tickers[0]: data['Close']})
                
                # Handle multiple tickers
                if 'Adj Close' in data.columns:
                    return data['Adj Close']
                else:
                    return data['Close']
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < 2:  # Not the last attempt
                    import time
                    time.sleep(2)  # Wait before retry
                continue
        
        print("All attempts failed, returning empty DataFrame")
        return pd.DataFrame()
        
    except Exception as e:
        print(f"Critical error in get_market_data: {str(e)}")
        return pd.DataFrame()

def calculate_portfolio_metrics(weights, returns, risk_free_rate=0.04):
    """Calculate portfolio performance metrics"""
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
    
    return {
        'return': portfolio_return,
        'volatility': portfolio_std,
        'sharpe_ratio': sharpe_ratio
    }

def monte_carlo_simulation(tickers, num_portfolios=10000, risk_tolerance=0.2):
    """Enhanced Monte Carlo simulation with better error handling"""
    
    try:
        # Limit number of tickers to prevent API issues
        if len(tickers) > 10:
            tickers = tickers[:10]
            print(f"Limited to first 10 tickers: {tickers}")
        
        # Get data with retry mechanism
        data = get_market_data(tickers, period='1y')  # Reduced period for faster loading
        
        if data.empty:
            print("No data received, trying with default tickers")
            # Fallback to reliable tickers
            fallback_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
            data = get_market_data(fallback_tickers, period='1y')
            if not data.empty:
                tickers = fallback_tickers
            else:
                return None, None, None, None
        
        # Clean data
        data = data.dropna()
        if len(data) < 100:  # Need sufficient data points
            print(f"Insufficient data points: {len(data)}")
            return None, None, None, None
        
        # Calculate returns
        returns = data.pct_change().dropna()
        
        if returns.empty:
            print("No returns data available")
            return None, None, None, None
        
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        # Validate covariance matrix
        if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
            print("Invalid covariance matrix")
            return None, None, None, None
        
        # Monte Carlo simulation
        results = np.zeros((num_portfolios, 3 + len(tickers)))
        
        print(f"Running {num_portfolios} simulations for {len(tickers)} assets...")
        
        for i in range(num_portfolios):
            # Generate random weights
            weights = np.random.random(len(tickers))
            weights /= np.sum(weights)
            
            # Calculate portfolio metrics
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # Handle potential division by zero
            if portfolio_std == 0:
                sharpe_ratio = 0
            else:
                sharpe_ratio = (portfolio_return - 0.04) / portfolio_std
            
            results[i, 0] = portfolio_return
            results[i, 1] = portfolio_std
            results[i, 2] = sharpe_ratio
            results[i, 3:] = weights
        
        # Create DataFrame
        columns = ['Return', 'Volatility', 'Sharpe Ratio'] + [f'{ticker}_Weight' for ticker in tickers]
        df = pd.DataFrame(results, columns=columns)
        
        # Remove invalid results
        df = df[(df['Volatility'] > 0) & (df['Volatility'] < 1) & (np.isfinite(df['Sharpe Ratio']))]
        
        if df.empty:
            print("No valid portfolio results generated")
            return None, None, None, None
        
        # Filter by risk tolerance
        filtered_df = df[df['Volatility'] <= risk_tolerance]
        
        # Find optimal portfolio
        if not filtered_df.empty:
            optimal_idx = filtered_df['Sharpe Ratio'].idxmax()
            optimal_portfolio = filtered_df.loc[optimal_idx]
        else:
            optimal_idx = df['Sharpe Ratio'].idxmax()
            optimal_portfolio = df.loc[optimal_idx]
        
        print(f"Optimization complete. Best Sharpe ratio: {optimal_portfolio['Sharpe Ratio']:.4f}")
        
        return df, optimal_portfolio, returns, data
        
    except Exception as e:
        print(f"Error in monte_carlo_simulation: {str(e)}")
        return None, None, None, None

def create_efficient_frontier_plot(df, optimal_portfolio, tickers):
    """Create enhanced efficient frontier visualization with error handling"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Validate data
        if df.empty or len(df) < 10:
            ax1.text(0.5, 0.5, 'Insufficient data for efficient frontier', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Efficient Frontier - Insufficient Data')
            ax2.text(0.5, 0.5, 'No allocation data available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Portfolio Allocation - No Data')
            return fig
        
        # Efficient Frontier
        valid_data = df[(df['Volatility'] > 0) & (df['Return'].notna()) & (df['Sharpe Ratio'].notna())]
        
        if len(valid_data) > 0:
            scatter = ax1.scatter(valid_data['Volatility'], valid_data['Return'], 
                                c=valid_data['Sharpe Ratio'], cmap='viridis', alpha=0.6, s=20)
            
            # Plot optimal portfolio
            if optimal_portfolio is not None:
                ax1.scatter(optimal_portfolio['Volatility'], optimal_portfolio['Return'], 
                           c='red', marker='*', s=500, label='Optimal Portfolio', edgecolors='black')
            
            ax1.set_xlabel('Volatility (Risk)', fontsize=12)
            ax1.set_ylabel('Expected Return', fontsize=12)
            ax1.set_title('Efficient Frontier - Risk vs Return', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Add colorbar safely
            try:
                cbar = plt.colorbar(scatter, ax=ax1)
                cbar.set_label('Sharpe Ratio', fontsize=10)
            except:
                pass
        else:
            ax1.text(0.5, 0.5, 'No valid data points for frontier', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Efficient Frontier - No Valid Data')
        
        # Portfolio Allocation Pie Chart
        try:
            if optimal_portfolio is not None:
                weights = []
                valid_tickers = []
                
                for ticker in tickers:
                    weight_col = f'{ticker}_Weight'
                    if weight_col in optimal_portfolio and optimal_portfolio[weight_col] > 0:
                        weights.append(optimal_portfolio[weight_col])
                        valid_tickers.append(ticker)
                
                if weights:
                    # Filter out very small weights for cleaner visualization
                    display_weights = []
                    display_tickers = []
                    other_weight = 0
                    
                    for ticker, weight in zip(valid_tickers, weights):
                        if weight > 0.02:  # Show only weights > 2%
                            display_weights.append(weight)
                            display_tickers.append(ticker)
                        else:
                            other_weight += weight
                    
                    if other_weight > 0:
                        display_weights.append(other_weight)
                        display_tickers.append('Others')
                    
                    if display_weights:
                        colors = plt.cm.Set3(np.linspace(0, 1, len(display_weights)))
                        wedges, texts, autotexts = ax2.pie(display_weights, labels=display_tickers, 
                                                          autopct='%1.1f%%', colors=colors, startangle=90)
                        ax2.set_title('Optimal Portfolio Allocation', fontsize=14, fontweight='bold')
                        
                        # Enhance pie chart text safely
                        for autotext in autotexts:
                            try:
                                autotext.set_color('white')
                                autotext.set_fontweight('bold')
                            except:
                                pass
                    else:
                        ax2.text(0.5, 0.5, 'No significant allocations', 
                                ha='center', va='center', transform=ax2.transAxes)
                        ax2.set_title('Portfolio Allocation - No Significant Weights')
                else:
                    ax2.text(0.5, 0.5, 'No allocation data available', 
                            ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title('Portfolio Allocation - No Data')
            else:
                ax2.text(0.5, 0.5, 'No optimal portfolio found', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Portfolio Allocation - No Optimal Portfolio')
                
        except Exception as e:
            print(f"Error creating pie chart: {e}")
            ax2.text(0.5, 0.5, 'Error creating allocation chart', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Portfolio Allocation - Error')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Error in create_efficient_frontier_plot: {e}")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Error creating efficient frontier plot: {str(e)}', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Efficient Frontier - Error')
        return fig

def create_correlation_heatmap(returns, tickers):
    """Create correlation heatmap using matplotlib"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    correlation_matrix = returns.corr()
    
    # Create heatmap using matplotlib
    im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation', rotation=270, labelpad=20)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(tickers)))
    ax.set_yticks(np.arange(len(tickers)))
    ax.set_xticklabels(tickers, rotation=45, ha='right')
    ax.set_yticklabels(tickers)
    
    # Add correlation values as text
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                         ha="center", va="center", color="black" if abs(correlation_matrix.iloc[i, j]) < 0.5 else "white")
    
    ax.set_title('Stock Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def create_performance_analysis(data, optimal_portfolio, tickers):
    """Create performance analysis charts with robust error handling"""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Validate input data
        if data.empty or optimal_portfolio is None:
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(0.5, 0.5, 'Insufficient data for analysis', 
                       ha='center', va='center', transform=ax.transAxes)
            fig.suptitle('Performance Analysis - Insufficient Data')
            return fig
        
        # Clean data
        data_clean = data.dropna()
        if len(data_clean) < 10:
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(0.5, 0.5, 'Not enough data points', 
                       ha='center', va='center', transform=ax.transAxes)
            fig.suptitle('Performance Analysis - Insufficient Data Points')
            return fig
        
        # 1. Normalized price evolution
        try:
            normalized_prices = data_clean / data_clean.iloc[0] * 100
            
            # Plot only valid tickers that exist in data
            valid_tickers = [t for t in tickers if t in normalized_prices.columns]
            
            if valid_tickers:
                for ticker in valid_tickers[:5]:  # Limit to 5 lines for readability
                    if ticker in normalized_prices.columns:
                        ax1.plot(normalized_prices.index, normalized_prices[ticker], 
                                label=ticker, linewidth=2, alpha=0.8)
                
                ax1.set_title('Normalized Stock Price Evolution', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Normalized Price (Base = 100)')
                ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'No valid ticker data for price evolution', 
                        ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Price Evolution - No Valid Data')
        except Exception as e:
            print(f"Error in price evolution plot: {e}")
            ax1.text(0.5, 0.5, 'Error creating price evolution plot', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Price Evolution - Error')
        
        # 2. Portfolio performance
        try:
            # Get weights safely
            weights = []
            for ticker in tickers:
                weight_col = f'{ticker}_Weight'
                if weight_col in optimal_portfolio:
                    weights.append(optimal_portfolio[weight_col])
                else:
                    weights.append(0)
            
            weights = np.array(weights)
            
            # Calculate portfolio performance only for valid tickers
            valid_indices = []
            valid_weights = []
            
            for i, ticker in enumerate(tickers):
                if ticker in data_clean.columns:
                    valid_indices.append(i)
                    valid_weights.append(weights[i])
            
            if valid_indices and sum(valid_weights) > 0:
                # Normalize weights
                valid_weights = np.array(valid_weights) / sum(valid_weights)
                
                # Calculate portfolio value
                portfolio_data = data_clean[[tickers[i] for i in valid_indices]]
                normalized_portfolio = portfolio_data / portfolio_data.iloc[0] * 100
                portfolio_value = (normalized_portfolio * valid_weights).sum(axis=1)
                
                ax2.plot(portfolio_value.index, portfolio_value, 
                        color='red', linewidth=3, label='Optimal Portfolio')
                ax2.set_title('Optimal Portfolio Performance', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Portfolio Value (Base = 100)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No valid data for portfolio performance', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Portfolio Performance - No Valid Data')
        except Exception as e:
            print(f"Error in portfolio performance plot: {e}")
            ax2.text(0.5, 0.5, 'Error creating portfolio performance plot', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Portfolio Performance - Error')
        
        # 3. Returns distribution
        try:
            returns = data_clean.pct_change().dropna()
            
            if not returns.empty and len(valid_indices) > 0:
                # Calculate portfolio returns
                portfolio_returns = (returns[[tickers[i] for i in valid_indices]] * valid_weights).sum(axis=1)
                
                if len(portfolio_returns) > 10:
                    ax3.hist(portfolio_returns, bins=min(30, len(portfolio_returns)//2), 
                            alpha=0.7, color='skyblue', edgecolor='black')
                    ax3.axvline(portfolio_returns.mean(), color='red', linestyle='--', 
                               linewidth=2, label=f'Mean: {portfolio_returns.mean():.4f}')
                    ax3.set_title('Portfolio Daily Returns Distribution', fontsize=12, fontweight='bold')
                    ax3.set_xlabel('Daily Returns')
                    ax3.set_ylabel('Frequency')
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
                else:
                    ax3.text(0.5, 0.5, 'Insufficient returns data', 
                            ha='center', va='center', transform=ax3.transAxes)
                    ax3.set_title('Returns Distribution - Insufficient Data')
            else:
                ax3.text(0.5, 0.5, 'No returns data available', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Returns Distribution - No Data')
        except Exception as e:
            print(f"Error in returns distribution plot: {e}")
            ax3.text(0.5, 0.5, 'Error creating returns distribution plot', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Returns Distribution - Error')
        
        # 4. Risk-Return scatter
        try:
            returns = data_clean.pct_change().dropna()
            
            if not returns.empty and len(valid_indices) > 0:
                # Calculate individual stock metrics
                individual_returns = returns.mean() * 252
                individual_volatility = returns.std() * np.sqrt(252)
                
                # Plot individual stocks
                for ticker in [tickers[i] for i in valid_indices]:
                    if ticker in individual_returns.index:
                        ax4.scatter(individual_volatility[ticker], individual_returns[ticker], 
                                   s=100, alpha=0.7, label=ticker)
                
                # Add portfolio point
                if 'Return' in optimal_portfolio and 'Volatility' in optimal_portfolio:
                    ax4.scatter(optimal_portfolio['Volatility'], optimal_portfolio['Return'], 
                               c='red', s=200, marker='*', label='Optimal Portfolio', 
                               edgecolors='black', zorder=5)
                
                ax4.set_xlabel('Volatility (Annual)')
                ax4.set_ylabel('Expected Return (Annual)')
                ax4.set_title('Individual Stocks vs Portfolio Risk-Return', fontsize=12, fontweight='bold')
                ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'No risk-return data available', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Risk-Return Analysis - No Data')
        except Exception as e:
            print(f"Error in risk-return scatter plot: {e}")
            ax4.text(0.5, 0.5, 'Error creating risk-return scatter plot', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Risk-Return Analysis - Error')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Error in create_performance_analysis: {e}")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Error creating performance analysis: {str(e)}', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Performance Analysis - Error')
        return fig

def generate_investment_report(optimal_portfolio, tickers, investment_amount):
    """Generate detailed investment report with better error handling"""
    
    try:
        # Calculate investment allocation
        weights = [optimal_portfolio[f'{ticker}_Weight'] for ticker in tickers]
        allocations = [weight * investment_amount for weight in weights]
        
        # Get current prices for share calculations
        current_prices = {}
        shares = {}
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period='5d')  # Get last 5 days to ensure data
                if not hist.empty:
                    current_prices[ticker] = hist['Close'].iloc[-1]
                    shares[ticker] = max(0, int(allocations[tickers.index(ticker)] / current_prices[ticker]))
                else:
                    # Fallback price estimation
                    current_prices[ticker] = 100.0  # Default price
                    shares[ticker] = max(0, int(allocations[tickers.index(ticker)] / current_prices[ticker]))
            except Exception as e:
                print(f"Error getting price for {ticker}: {e}")
                current_prices[ticker] = 100.0  # Default price
                shares[ticker] = max(0, int(allocations[tickers.index(ticker)] / current_prices[ticker]))
        
        # Create report dataframe
        report_data = []
        for i, ticker in enumerate(tickers):
            report_data.append({
                'Ticker': ticker,
                'Weight (%)': f"{weights[i]*100:.2f}%",
                'Allocation ($)': f"${allocations[i]:,.2f}",
                'Current Price': f"${current_prices[ticker]:.2f}",
                'Shares to Buy': shares[ticker]
            })
        
        report_df = pd.DataFrame(report_data)
        
        # Summary statistics
        summary_stats = f"""
        ## 📊 Portfolio Summary
        
        **Expected Annual Return:** {optimal_portfolio['Return']:.2%}
        **Portfolio Volatility:** {optimal_portfolio['Volatility']:.2%}
        **Sharpe Ratio:** {optimal_portfolio['Sharpe Ratio']:.4f}
        **Total Investment:** ${investment_amount:,.2f}
        
        ## 🎯 Risk Assessment
        {"🟢 Low Risk" if optimal_portfolio['Volatility'] < 0.15 else "🟡 Moderate Risk" if optimal_portfolio['Volatility'] < 0.25 else "🔴 High Risk"}
        
        ## 📈 Performance Expectations
        **Best Case (95% confidence):** {(optimal_portfolio['Return'] + 2*optimal_portfolio['Volatility'])*100:.1f}% annual return
        **Expected Case:** {optimal_portfolio['Return']*100:.1f}% annual return  
        **Worst Case (5% confidence):** {(optimal_portfolio['Return'] - 2*optimal_portfolio['Volatility'])*100:.1f}% annual return
        """
        
        return report_df, summary_stats
        
    except Exception as e:
        print(f"Error generating investment report: {str(e)}")
        return pd.DataFrame(), "Error generating report"

def run_portfolio_optimization(selected_sectors, risk_tolerance, num_portfolios, investment_amount):
    """Main function to run portfolio optimization with robust error handling"""
    
    try:
        # Validate inputs
        if not selected_sectors:
            return "❌ Please select at least one sector", pd.DataFrame(), None, None, None, ""
        
        if investment_amount <= 0:
            return "❌ Investment amount must be positive", pd.DataFrame(), None, None, None, ""
        
        # Get selected tickers
        tickers = []
        for sector in selected_sectors:
            if sector in STOCK_UNIVERSE:
                tickers.extend(STOCK_UNIVERSE[sector])
        
        # Remove duplicates and limit to prevent API issues
        tickers = list(set(tickers))
        
        if len(tickers) < 2:
            return "❌ Please select sectors with at least 2 different stocks", pd.DataFrame(), None, None, None, ""
        
        # Limit number of tickers for API efficiency
        if len(tickers) > 8:
            tickers = tickers[:8]
            print(f"Limited analysis to 8 stocks: {tickers}")
        
        print(f"Starting optimization with {len(tickers)} stocks: {tickers}")
        
        # Run Monte Carlo simulation
        df, optimal_portfolio, returns, data = monte_carlo_simulation(tickers, num_portfolios, risk_tolerance)
        
        if df is None or optimal_portfolio is None:
            return "❌ Unable to fetch market data or optimize portfolio. Please try again with different sectors.", pd.DataFrame(), None, None, None, ""
        
        # Create visualizations
        try:
            frontier_plot = create_efficient_frontier_plot(df, optimal_portfolio, tickers)
        except Exception as e:
            print(f"Error creating frontier plot: {e}")
            frontier_plot = plt.figure()
            plt.text(0.5, 0.5, 'Error creating efficient frontier plot', ha='center', va='center')
            plt.title('Efficient Frontier - Error')
        
        try:
            correlation_plot = create_correlation_heatmap(returns, tickers)
        except Exception as e:
            print(f"Error creating correlation plot: {e}")
            correlation_plot = plt.figure()
            plt.text(0.5, 0.5, 'Error creating correlation plot', ha='center', va='center')
            plt.title('Correlation Matrix - Error')
        
        try:
            performance_plot = create_performance_analysis(data, optimal_portfolio, tickers)
        except Exception as e:
            print(f"Error creating performance plot: {e}")
            performance_plot = plt.figure()
            plt.text(0.5, 0.5, 'Error creating performance analysis', ha='center', va='center')
            plt.title('Performance Analysis - Error')
        
        # Generate investment report
        report_df, summary_stats = generate_investment_report(optimal_portfolio, tickers, investment_amount)
        
        success_message = f"✅ Analysis complete! Optimized portfolio using {len(tickers)} stocks from {len(selected_sectors)} sectors."
        
        return summary_stats, report_df, frontier_plot, correlation_plot, performance_plot, success_message
        
    except Exception as e:
        error_message = f"❌ Unexpected error: {str(e)}. Please try again with different settings."
        print(f"Error in run_portfolio_optimization: {str(e)}")
        return error_message, pd.DataFrame(), None, None, None, ""

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="🧠 AI Portfolio Management Assistant", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # 🧠 AI-Powered Portfolio Management Assistant
        
        **Professional-grade portfolio optimization using Modern Portfolio Theory and Monte Carlo simulation**
        
        Select your preferred sectors, set your risk tolerance, and let AI find the optimal portfolio allocation for you!
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 🎛️ Configuration")
                
                sector_selection = gr.CheckboxGroup(
                    choices=list(STOCK_UNIVERSE.keys()),
                    value=['Technology', 'Finance'],
                    label="📊 Select Sectors",
                    info="Choose sectors to include in your portfolio"
                )
                
                risk_tolerance = gr.Slider(
                    minimum=0.05,
                    maximum=0.50,
                    value=0.20,
                    step=0.01,
                    label="🎯 Risk Tolerance (Max Volatility)",
                    info="Higher values allow for more aggressive portfolios"
                )
                
                num_portfolios = gr.Slider(
                    minimum=1000,
                    maximum=50000,
                    value=10000,
                    step=1000,
                    label="🔄 Monte Carlo Simulations",
                    info="More simulations = better optimization (but slower)"
                )
                
                investment_amount = gr.Number(
                    value=10000,
                    label="💰 Investment Amount ($)",
                    info="Total amount you plan to invest"
                )
                
                optimize_btn = gr.Button("🚀 Optimize Portfolio", variant="primary", size="lg")
                
                status_output = gr.Textbox(label="Status", interactive=False)
        
        with gr.Row():
            with gr.Column():
                summary_output = gr.Markdown(label="Portfolio Summary")
                
        with gr.Row():
            allocation_output = gr.Dataframe(
                label="📋 Investment Allocation",
                headers=["Ticker", "Weight (%)", "Allocation ($)", "Current Price", "Shares to Buy"]
            )
        
        with gr.Row():
            with gr.Column():
                frontier_plot = gr.Plot(label="📈 Efficient Frontier & Allocation")
        
        with gr.Row():
            with gr.Column():
                correlation_plot = gr.Plot(label="🔗 Stock Correlation Matrix")
        
        with gr.Row():
            with gr.Column():
                performance_plot = gr.Plot(label="📊 Performance Analysis")
        
        # Connect the optimization function
        optimize_btn.click(
            fn=run_portfolio_optimization,
            inputs=[sector_selection, risk_tolerance, num_portfolios, investment_amount],
            outputs=[summary_output, allocation_output, frontier_plot, correlation_plot, performance_plot, status_output]
        )
        
        # Add footer
        gr.Markdown("""
        ---
        **⚠️ Disclaimer:** This tool is for educational purposes only. Past performance does not guarantee future results. 
        Always consult with a financial advisor before making investment decisions.
        """)
    
    return demo

# Launch the app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)