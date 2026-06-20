"""
Mean-Variance Portfolio Optimization (Markowitz Model)
Production-ready Streamlit Web Application

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import requests  # เพิ่มสำหรับดึงข้อมูล Autocomplete จาก Yahoo Finance API
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
    
    # ระบบค้นหาคำใกล้เคียง Real-time จากฐานข้อมูล Yahoo Finance ทั่วโลก
    st.markdown("**🔍 Search Global Stocks**")
    search_query = st.text_input("Type stock name or ticker to search:", value="", placeholder="e.g. N, NV, NOK, RKLB")
    
    # คลังเก็บคำแนะนำแบบไดนามิก (เปลี่ยนไปตามสิ่งที่ผู้ใช้พิมพ์)
    dynamic_options = []
    
    if search_query.strip():
        try:
            # ยิง API ไปถาม Yahoo Finance ตรงๆ ว่าคำที่พี่พิมพ์มา มีหุ้นอะไรในโลกบ้างที่ใกล้เคียง
            url = f"https://query2.finance.yahoo.com/v1/finance/search?q={search_query}&quotesCount=10&newsCount=0"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            response = requests.get(url, headers=headers, timeout=5).json()
            
            # สกัดเอาชื่อหุ้นทั้งหมดที่ค้นเจอมาทำเป็นคำแนะนำย้อยลงมา
            for quote in response.get("quotes", []):
                symbol = quote.get("symbol")
                shortname = quote.get("shortname", "")
                exch = quote.get("exchange", "")
                if symbol:
                    dynamic_options.append(f"{symbol} ({shortname} - {exch})")
        except Exception:
            pass

    # ระบบจำหุ้นที่เลือกในเซสชัน เพื่อรักษาหน้าตาแบบกล่องเดียวของพี่ไว้
    if "selected_portfolio_tickers" not in st.session_state:
        st.session_state.selected_portfolio_tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "RKLB"]

    # แนะนำรายการที่พี่ค้นหาเจอเพื่อให้คลิกเลือกเพิ่มเข้าไป
    if dynamic_options:
        st.markdown("**💡 Matching Results (Click to add):**")
        chosen_search = st.selectbox("Select stock to add to portfolio:", ["-- Click to choose --"] + dynamic_options, index=0)
        if chosen_search != "-- Click to choose --":
            target_symbol = chosen_search.split(" ")[0].strip().upper()
            if target_symbol not in st.session_state.selected_portfolio_tickers:
                st.session_state.selected_portfolio_tickers.append(target_symbol)
            st.rerun()

    st.markdown("---")
    st.markdown("**🍰 Current Portfolio Assets**")
    
    # กล่องหลักเดี่ยวแสดงผลและจัดการหุ้นในพอร์ต (UI เหมือนเดิม คล่องตัวเหมือนเดิม ลบออกได้อิสระ)
    # คราวนี้บรรจุหุ้นอะไรบนโลกก็ได้แล้วครับ ไม่ติดขัดคลังคำแนะนำล็อกตายอีกต่อไป
    selected_tickers = st.multiselect(
        "Stock Tickers",
        options=st.session_state.selected_portfolio_tickers,
        default=st.session_state.selected_portfolio_tickers,
        help="This box shows your active portfolio. Use the search tool above to find and add any global asset."
    )
    
    # อัปเดตลิสต์หุ้นให้ตรงกับการกดลบออกของผู้ใช้
    st.session_state.selected_portfolio_tickers = selected_tickers

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
# Helper Functions (คงลอจิกคณิตศาสตร์และการดึงราคาอันแข็งแกร่งเดิมไว้ 100%)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_price_data(tickers: list[str], start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch adjusted close prices from Yahoo Finance safely for ALL global stocks."""
    combined_data = pd.DataFrame()
    
    for ticker in tickers:
        try:
            asset_data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if not asset_data.empty:
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
    
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
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
    else:
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
# Main Application Logic (UI ตารางและกราฟวงกลม + กราฟโค้ง คงเดิม 100%)
# ─────────────────────────────────────────────────────────────────────────────

if run_optimization:
    try:
        tickers = [t.strip().upper() for t in selected_tickers if t.strip()]
        
        if len(tickers) < 2:
            st.error("⚠️ Please enter at least 2 stock tickers for portfolio optimization.")
            st.stop()
        
        if start_date >= end_date:
            st.error("⚠️ Start date must be before end date.")
            st.stop()
        
        with st.spinner(f"Fetching data for {len(tickers)} stocks..."):
            prices = fetch_price_data(tickers, start_date, end_date)
        
        if prices.empty:
            st.error("⚠️ No data retrieved. Please check ticker symbols and date range.")
            st.stop()
        
        missing_tickers = [t for t in tickers if t not in prices.columns]
        if missing_tickers:
            st.warning(f"⚠️ No data found for: {', '.join(missing_tickers)}. Proceeding with available tickers.")
            tickers = [t for t in tickers if t in prices.columns]
        
        if len(tickers) < 2:
            st.error("⚠️ Less than 2 valid tickers remaining. Cannot optimize.")
            st.stop()
        
        prices = prices[tickers].ffill().bfill().dropna()
        
        if len(prices) < 10:
            st.error("⚠️ Insufficient data points. Please select a longer date range.")
            st.stop()
        
        returns = calculate_returns(prices)
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        
        with st.spinner("Optimizing portfolios..."):
            max_sharpe_weights = optimize_portfolio(mean_returns, cov_matrix, risk_free_rate, "sharpe")
            min_var_weights = optimize_portfolio(mean_returns, cov_matrix, risk_free_rate, "min_var")
            
            ms_return, ms_volatility = portfolio_performance(max_sharpe_weights, mean_returns, cov_matrix)
            ms_sharpe = (ms_return - risk_free_rate) / ms_volatility if ms_volatility > 0 else 0
            
            mv_return, mv_volatility = portfolio_performance(min_var_weights, mean_returns, cov_matrix)
            mv_sharpe = (mv_return - risk_free_rate) / mv_volatility if mv_volatility > 0 else 0
            
            frontier_returns, frontier_volatilities = generate_efficient_frontier(mean_returns, cov_matrix)
        
        st.success("✅ Optimization complete!")
        st.markdown("---")
        
        # Summary Metrics Table
        st.subheader("📋 Portfolio Comparison")
        summary_df = pd.DataFrame({
            "Metric": ["Expected Annual Return", "Annual Volatility (Risk)", "Sharpe Ratio"],
            "Max Sharpe Ratio": [f"{ms_return:.2%}", f"{ms_volatility:.2%}", f"{ms_sharpe:.3f}"],
            "Minimum Variance": [f"{mv_return:.2%}", f"{mv_volatility:.2%}", f"{mv_sharpe:.3f}"]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Pie Charts for Portfolio Weights
        st.subheader("🥧 Portfolio Allocation")
        col1, col2 = st.columns(2)
        
        with col1:
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
                showlegend=True, height=400, margin=dict(t=60, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_pie_sharpe, use_container_width=True)
        
        with col2:
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
                showlegend=True, height=400, margin=dict(t=60, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_pie_minvar, use_container_width=True)
        
        st.markdown("---")
        
        # Efficient Frontier Chart
        st.subheader("📈 Efficient Frontier")
        fig_frontier = go.Figure()
        
        if len(frontier_volatilities) > 0 and len(frontier_returns) > 0:
            fig_frontier.add_trace(go.Scatter(
                x=frontier_volatilities * 100, y=frontier_returns * 100,
                mode="lines", name="Efficient Frontier", line=dict(color="#636EFA", width=3)
            ))
        
        individual_returns = mean_returns * 252
        individual_volatilities = np.sqrt(np.diag(cov_matrix)) * np.sqrt(252)
        
        fig_frontier.add_trace(go.Scatter(
            x=individual_volatilities * 100, y=individual_returns * 100,
            mode="markers+text", name="Individual Assets", text=tickers, textposition="top center",
            marker=dict(size=10, color="#B6E880", symbol="diamond")
        ))
        
        fig_frontier.add_trace(go.Scatter(
            x=[ms_volatility * 100], y=[ms_return * 100], mode="markers",
            name=f"Max Sharpe (SR={ms_sharpe:.2f})",
            marker=dict(size=18, color="#EF553B", symbol="star", line=dict(width=2, color="white"))
        ))
        
        fig_frontier.add_trace(go.Scatter(
            x=[mv_volatility * 100], y=[mv_return * 100], mode="markers",
            name=f"Min Variance (SR={mv_sharpe:.2f})",
            marker=dict(size=18, color="#00CC96", symbol="star", line=dict(width=2, color="white"))
        ))
        
        fig_frontier.update_layout(
            title=dict(text="Efficient Frontier with Optimal Portfolios", x=0.5, xanchor="center"),
            xaxis_title="Annual Volatility (%)", yaxis_title="Expected Annual Return (%)",
            height=500, showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            hovermode="closest"
        )
        st.plotly_chart(fig_frontier, use_container_width=True)
        
        # Detailed Weights Table
        st.markdown("---")
        st.subheader("📊 Detailed Portfolio Weights")
        weights_df = pd.DataFrame({
            "Ticker": tickers,
            "Max Sharpe (%)": [f"{w*100:.2f}%" for w in max_sharpe_weights],
            "Min Variance (%)": [f"{w*100:.2f}%" for w in min_var_weights]
        })
        st.dataframe(weights_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.caption(f"📅 Data period: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')} ({len(prices)} trading days)")

    except Exception as e:
        st.error(f"⚠️ An error occurred: {str(e)}")
        st.info("Please check your inputs and try again.")

else:
    st.info("👈 Configure your portfolio parameters in the sidebar and click **Run Optimization** to begin.")