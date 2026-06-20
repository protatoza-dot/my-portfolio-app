"""
Mean-Variance Portfolio Optimization (Markowitz Model)
Production-ready Streamlit Web Application

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Mean-Variance Portfolio Optimization")
st.markdown("*Markowitz Model — Efficient Frontier Analysis*")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Inputs
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Stock tickers input - โครงสร้างเดิม พิมพ์คั่นด้วยคอมมา และเพิ่ม RKLB เป็นค่าเริ่มต้นให้เห็นชัดเจน
    tickers_input = st.text_input(
        "Stock Tickers (comma-separated)",
        value="AAPL, MSFT, GOOG, AMZN, RKLB",
        help="Enter stock symbols separated by commas (e.g., AAPL, MSFT or PTT.BK, BDMS.BK)"
    )
    
    st.markdown("---")
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=3*365),
            max_value=datetime.now() - timedelta(days=30)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now()
        )
    
    st.markdown("---")
    
    # Risk-free rate
    risk_free_rate = st.slider(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.1,
        help="Annual risk-free rate for Sharpe Ratio calculation"
    ) / 100
    
    st.markdown("---")
    
    # Run optimization button
    run_optimization = st.button("🚀 Run Optimization", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def parse_tickers(tickers_str: str) -> list[str]:
    """Parse and clean ticker symbols from user input."""
    tickers = [t.strip().upper() for t in tickers_str.split(",")]
    return [t for t in tickers if t]  # Remove empty strings


def fetch_price_data(tickers: list[str], start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch adjusted close prices from Yahoo Finance safely for ALL global stocks."""
    combined_data = pd.DataFrame()
    
    for ticker in tickers:
        try:
            # ดึงข้อมูลแยกรายตัวแบบ auto_adjust เพื่อความทนทานต่อหุ้นทุกตัวในโลกที่ IPO แล้ว
            asset_data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if not asset_data.empty:
                # สกัดเอาคอลัมน์ราคาปิด 'Close' ไม่ว่าจะเป็น Series หรือ DataFrame ซ้อนคอลัมน์
                if isinstance(asset_data.columns, pd.MultiIndex):
                    if 'Close' in asset_data.columns.get_level_values(0):
                        combined_data[ticker] = asset_data['Close'].iloc[:, 0]
                else:
                    if 'Close' in asset_data.columns:
                        combined_data[ticker] = asset_data['Close']
                    else:
                        combined_data[ticker] = asset_data.iloc[:, 0]
        except Exception:
            continue
            
    return combined_data


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def portfolio_performance(weights: np.ndarray, mean_returns: np.ndarray, 
                          cov_matrix: np.ndarray, trading_days: int = 252) -> tuple:
    """Calculate annualized return, volatility, and Sharpe ratio."""
    portfolio_return = np.sum(mean_returns * weights) * trading_days
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(trading_days)
    return portfolio_return, portfolio_volatility


def negative_sharpe_ratio(weights: np.ndarray, mean_returns: np.ndarray, 
                          cov_matrix: np.ndarray, risk_free_rate: float) -> float:
    """Negative Sharpe ratio for minimization."""
    p_return, p_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_return - risk_free_rate) / p_volatility


def portfolio_volatility(weights: np.ndarray, mean_returns: np.ndarray, 
                         cov_matrix: np.ndarray) -> float:
    """Portfolio volatility for minimum variance optimization."""
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]


def optimize_portfolio(mean_returns: np.ndarray, cov_matrix: np.ndarray, 
                       risk_free_rate: float, objective: str = "sharpe") -> np.ndarray:
    """Optimize portfolio weights using scipy.optimize."""
    num_assets = len(mean_returns)
    initial_weights = np.array([1/num_assets] * num_assets)
    
    # Constraints: weights sum to 1
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    
    # Bounds: each weight between 0 and 1 (long only)
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    if objective == "sharpe":
        result = minimize(
            negative_sharpe_ratio,
            initial_weights,
            args=(mean_returns, cov_matrix, risk_free_rate),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
    else:  # minimum variance
        result = minimize(
            portfolio_volatility,
            initial_weights,
            args=(mean_returns, cov_matrix),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
    
    return result.x


def generate_efficient_frontier(mean_returns: np.ndarray, cov_matrix: np.ndarray, 
                                num_portfolios: int = 100) -> tuple:
    """Generate points along the efficient frontier safely."""
    num_assets = len(mean_returns)
    
    min_var_weights = optimize_portfolio(mean_returns, cov_matrix, 0, "min_var")
    min_return = portfolio_performance(min_var_weights, mean_returns, cov_matrix)[0]
    
    max_return = np.max(mean_returns) * 252
    
    # ป้องกันการคำนวณพังกรณีค่าช่วงขอบบนขอบล่างเท่ากันหรือใกล้กันเกินไปสำหรับหุ้นที่ข้อมูลน้อย
    if min_return >= max_return:
        max_return = min_return + 0.05

    target_returns = np.linspace(min_return, max_return, num_portfolios)
    frontier_volatilities = []
    frontier_returns = []
    
    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: portfolio_performance(w, mean_returns, cov_matrix)[0] - t}
        ]
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_weights = np.array([1/num_assets] * num_assets)
        
        result = minimize(
            portfolio_volatility,
            initial_weights,
            args=(mean_returns, cov_matrix),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            ret, vol = portfolio_performance(result.x, mean_returns, cov_matrix)
            frontier_returns.append(ret)
            frontier_volatilities.append(vol)
    
    return np.array(frontier_returns), np.array(frontier_volatilities)


# ─────────────────────────────────────────────────────────────────────────────
# Main Application Logic
# ─────────────────────────────────────────────────────────────────────────────

if run_optimization:
    try:
        # Parse tickers
        tickers = parse_tickers(tickers_input)
        
        if len(tickers) < 2:
            st.error("⚠️ Please enter at least 2 stock tickers for portfolio optimization.")
            st.stop()
        
        # Validate date range
        if start_date >= end_date:
            st.error("⚠️ Start date must be before end date.")
            st.stop()
        
        # Fetch data with progress indicator
        with st.spinner(f"Fetching data for {len(tickers)} stocks..."):
            prices = fetch_price_data(tickers, start_date, end_date)
        
        # Check for missing data
        if prices.empty:
            st.error("⚠️ No data retrieved. Please check ticker symbols and date range.")
            st.stop()
        
        # Check for tickers with no data
        missing_tickers = [t for t in tickers if t not in prices.columns]
        if missing_tickers:
            st.warning(f"⚠️ No data found for: {', '.join(missing_tickers)}. Proceeding with available tickers.")
            tickers = [t for t in tickers if t in prices.columns]
        
        if len(tickers) < 2:
            st.error("⚠️ Less than 2 valid tickers remaining. Cannot optimize.")
            st.stop()
        
        # Handle missing values safely (FFill แถวว่างกรณีวันหยุดเหลื่อมล้ำกันของหุ้นแต่ละสัญชาติตลาด)
        prices = prices[tickers].ffill().bfill().dropna()
        
        if len(prices) < 10:  # ปรับลดขั้นต่ำให้รองรับหุ้นกากๆ หรือหุ้นปั่นที่เปิดเทรดน้อยวัน
            st.error("⚠️ Insufficient data points. Please select a longer date range.")
            st.stop()
        
        # Calculate returns and statistics
        returns = calculate_returns(prices)
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        
        # Optimize portfolios
        with st.spinner("Optimizing portfolios..."):
            max_sharpe_weights = optimize_portfolio(mean_returns, cov_matrix, risk_free_rate, "sharpe")
            min_var_weights = optimize_portfolio(mean_returns, cov_matrix, risk_free_rate, "min_var")
            
            # Calculate portfolio metrics
            ms_return, ms_volatility = portfolio_performance(max_sharpe_weights, mean_returns, cov_matrix)
            ms_sharpe = (ms_return - risk_free_rate) / ms_volatility if ms_volatility > 0 else 0
            
            mv_return, mv_volatility = portfolio_performance(min_var_weights, mean_returns, cov_matrix)
            mv_sharpe = (mv_return - risk_free_rate) / mv_volatility if mv_volatility > 0 else 0
            
            # Generate efficient frontier
            frontier_returns, frontier_volatilities = generate_efficient_frontier(mean_returns, cov_matrix)
        
        st.success("✅ Optimization complete!")
        st.markdown("---")
        
        # ─────────────────────────────────────────────────────────────────────
        # Display Results
        # ─────────────────────────────────────────────────────────────────────
        
        # Summary Metrics Table
        st.subheader("📋 Portfolio Comparison")
        
        summary_df = pd.DataFrame({
            "Metric": ["Expected Annual Return", "Annual Volatility (Risk)", "Sharpe Ratio"],
            "Max Sharpe Ratio": [f"{ms_return:.2%}", f"{ms_volatility:.2%}", f"{ms_sharpe:.3f}"],
            "Minimum Variance": [f"{mv_return:.2%}", f"{mv_volatility:.2%}", f"{mv_sharpe:.3f}"]
        })
        
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("---")
        
        # Pie Charts for Portfolio Weights
        st.subheader("🥧 Portfolio Allocation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Max Sharpe Pie Chart
            fig_pie_sharpe = go.Figure(data=[go.Pie(
                labels=tickers,
                values=max_sharpe_weights,
                hole=0.4,
                textinfo="label+percent",
                textposition="outside",
                marker=dict(colors=["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", 
                                    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"])
            )])
            fig_pie_sharpe.update_layout(
                title=dict(text="Max Sharpe Ratio Portfolio", x=0.5, xanchor="center"),
                showlegend=True,
                height=400,
                margin=dict(t=60, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_pie_sharpe, use_container_width=True)
        
        with col2:
            # Min Variance Pie Chart
            fig_pie_minvar = go.Figure(data=[go.Pie(
                labels=tickers,
                values=min_var_weights,
                hole=0.4,
                textinfo="label+percent",
                textposition="outside",
                marker=dict(colors=["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
                                    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"])
            )])
            fig_pie_minvar.update_layout(
                title=dict(text="Minimum Variance Portfolio", x=0.5, xanchor="center"),
                showlegend=True,
                height=400,
                margin=dict(t=60, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_pie_minvar, use_container_width=True)
        
        st.markdown("---")
        
        # Efficient Frontier Chart
        st.subheader("📈 Efficient Frontier")
        
        fig_frontier = go.Figure()
        
        # Plot efficient frontier line
        if len(frontier_volatilities) > 0 and len(frontier_returns) > 0:
            fig_frontier.add_trace(go.Scatter(
                x=frontier_volatilities * 100,
                y=frontier_returns * 100,
                mode="lines",
                name="Efficient Frontier",
                line=dict(color="#636EFA", width=3)
            ))
        
        # Plot individual assets
        individual_returns = mean_returns * 252
        individual_volatilities = np.sqrt(np.diag(cov_matrix)) * np.sqrt(252)
        
        fig_frontier.add_trace(go.Scatter(
            x=individual_volatilities * 100,
            y=individual_returns * 100,
            mode="markers+text",
            name="Individual Assets",
            text=tickers,
            textposition="top center",
            marker=dict(size=10, color="#B6E880", symbol="diamond")
        ))
        
        # Mark Max Sharpe Portfolio
        fig_frontier.add_trace(go.Scatter(
            x=[ms_volatility * 100],
            y=[ms_return * 100],
            mode="markers",
            name=f"Max Sharpe (SR={ms_sharpe:.2f})",
            marker=dict(size=18, color="#EF553B", symbol="star", line=dict(width=2, color="white"))
        ))
        
        # Mark Minimum Variance Portfolio
        fig_frontier.add_trace(go.Scatter(
            x=[mv_volatility * 100],
            y=[mv_return * 100],
            mode="markers",
            name=f"Min Variance (SR={mv_sharpe:.2f})",
            marker=dict(size=18, color="#00CC96", symbol="star", line=dict(width=2, color="white"))
        ))
        
        fig_frontier.update_layout(
            title=dict(text="Efficient Frontier with Optimal Portfolios", x=0.5, xanchor="center"),
            xaxis_title="Annual Volatility (%)",
            yaxis_title="Expected Annual Return (%)",
            height=500,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            hovermode="closest"
        )
        
        st.plotly_chart(fig_frontier, use_container_width=True)
        
        # ─────────────────────────────────────────────────────────────────────
        # Detailed Weights Table
        # ─────────────────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Detailed Portfolio Weights")
        
        weights_df = pd.DataFrame({
            "Ticker": tickers,
            "Max Sharpe (%)": [f"{w*100:.2f}%" for w in max_sharpe_weights],
            "Min Variance (%)": [f"{w*100:.2f}%" for w in min_var_weights]
        })
        
        st.dataframe(weights_df, use_container_width=True, hide_index=True)
        
        # Data info
        st.markdown("---")
        st.caption(f"📅 Data period: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')} ({len(prices)} trading days)")

    except Exception as e:
        st.error(f"⚠️ An error occurred: {str(e)}")
        st.info("Please check your inputs and try again. Common issues:\n"
                "- Invalid ticker symbols\n"
                "- Date range with no trading data\n"
                "- Network connectivity issues")

else:
    # Initial state - show instructions
    st.info("👈 Configure your portfolio parameters in the sidebar and click **Run Optimization** to begin.")
    
    with st.expander("ℹ️ How to Use This Tool"):
        st.markdown("""
        **Step 1:** Enter stock ticker symbols separated by commas
        - US stocks: AAPL, MSFT, GOOG, AMZN
        - Thai stocks: PTT.BK, BDMS.BK, AOT.BK
        
        **Step 2:** Select your date range for historical data
        
        **Step 3:** Adjust the risk-free rate (default 2%)
        
        **Step 4:** Click "Run Optimization"
        
        ---
        
        **Output includes:**
        - **Max Sharpe Ratio Portfolio:** Highest risk-adjusted return
        - **Minimum Variance Portfolio:** Lowest risk
        - **Efficient Frontier:** Visual representation of optimal portfolios
        """)