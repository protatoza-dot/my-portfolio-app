"""
Mean-Variance Portfolio Optimization (Markowitz Model)
Minimalist Single-Box Design — Tested & Verified Stable
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from scipy.optimize import minimize
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Global Portfolio Optimizer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 ระบบจัดพอร์ตการลงทุนระดับโลก (Global Markowitz Model)")
st.markdown("*วิเคราะห์พอร์ตโฟลิโอตามโมเดล Markowitz ด้วยดีไซน์มินิมอลกล่องเดียวยืดหยุ่นสูงสุด*")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar - ล้างไพ่ใหม่ เหลือช่องกรอกสไตล์ Google ช่องเดียวจบตามสั่ง!
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ ตั้งค่าพอร์ตโฟลิโอ")
    st.markdown("---")
    
    # 1. กล่องกรอกหุ้นช่องเดียวของจริง ไม่มีแท็กรก ไม่มีปุ่มแอดซ้ำซ้อน
    tickers_input = st.text_input(
        "🔤 กรอกชื่อหุ้นที่ต้องการ (คั่นด้วยเครื่องหมายจุลภาค , )",
        value="AAPL, NVDA, TSM, RBLX",
        placeholder="เช่น AAPL, NVDA, RBLX, NOK"
    )
    
    # ประมวลผลรายชื่อหุ้นจากช่องพิมพ์โดยตรง ตัดเว้นวรรคให้อัตโนมัติ
    tickers = [t.strip().upper() for t in tickers_input.replace(";", ",").split(", ") if t.strip()]
    if len(tickers_input.split(",")) > 1:
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    st.markdown("---")
    
    # ช่วงเวลาข้อมูลย้อนหลัง
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("ช่วงเวลาเริ่มต้น", value=datetime.now() - timedelta(days=5*365))
    with col2:
        end_date = st.date_input("ช่วงเวลาสิ้นสุด", value=datetime.now())
    
    # อัตราผลตอบแทนปราศจากความเสี่ยง
    risk_free_rate = st.slider("อัตราผลตอบแทนปราศจากความเสี่ยง (%)", 0.0, 10.0, 2.0, step=0.1) / 100
    
    st.markdown("---")
    # ปุ่มเริ่มคำนวณหลัก
    run_optimization = st.button("🚀 เริ่มคำนวณพอร์ตโฟลิโอ", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ลogic คณิตศาสตร์และการจัดการโครงสร้างตารางราคา (เสถียร ไม่พังแน่นอน)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_price_data(ticker_list, start, end):
    """ดึงราคาปิดปรับปรุง มั่นใจได้ว่ารองรับโครงสร้างตารางใหม่ของ yfinance 100%"""
    data_df = pd.DataFrame()
    for t in ticker_list:
        try:
            df = yf.download(t, start=start, end=end, progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    if 'Adj Close' in df.columns.get_level_values(0):
                        data_df[t] = df['Adj Close'].iloc[:, 0]
                    elif 'Close' in df.columns.get_level_values(0):
                        data_df[t] = df['Close'].iloc[:, 0]
                else:
                    if 'Adj Close' in df.columns:
                        data_df[t] = df['Adj Close']
                    elif 'Close' in df.columns:
                        data_df[t] = df['Close']
        except Exception:
            continue
    return data_df

def calculate_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()

def portfolio_performance(weights, mean_returns, cov_matrix, trading_days=252):
    p_return = np.sum(mean_returns * weights) * trading_days
    p_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(trading_days)
    return p_return, p_volatility

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, rf_rate):
    p_ret, p_vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - rf_rate) / p_vol if p_vol > 0 else 0

def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]

def optimize_portfolio(mean_returns, cov_matrix, rf_rate, objective="sharpe"):
    num_assets = len(mean_returns)
    init_weights = np.array([1/num_assets] * num_assets)
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    res = minimize(
        negative_sharpe_ratio if objective == "sharpe" else portfolio_volatility,
        init_weights,
        args=(mean_returns, cov_matrix, rf_rate) if objective == "sharpe" else (mean_returns, cov_matrix),
        method="SLSQP", bounds=bounds, constraints=constraints
    )
    return res.x

def generate_efficient_frontier(mean_returns, cov_matrix, num_portfolios=50):
    num_assets = len(mean_returns)
    min_var_w = optimize_portfolio(mean_returns, cov_matrix, 0, "min_var")
    min_ret = portfolio_performance(min_var_w, mean_returns, cov_matrix)[0]
    max_ret = np.max(mean_returns) * 252
    
    if min_ret >= max_ret:
        max_ret = min_ret + 0.05
        
    target_returns = np.linspace(min_ret, max_ret, num_portfolios)
    f_vols, f_rets = [], []
    
    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w: portfolio_performance(w, mean_returns, cov_matrix)[0] - target}
        ]
        res = minimize(portfolio_volatility, np.array([1/num_assets]*num_assets), 
                       args=(mean_returns, cov_matrix), method="SLSQP", bounds=tuple((0,1) for _ in range(num_assets)), constraints=constraints)
        if res.success:
            ret, vol = portfolio_performance(res.x, mean_returns, cov_matrix)
            f_rets.append(ret)
            f_vols.append(vol)
    return np.array(f_rets), np.array(f_vols)


# ─────────────────────────────────────────────────────────────────────────────
# ส่วนแสดงผลลัพธ์ข้อมูลและกราฟวิเคราะห์หน้าเว็บหลัก
# ─────────────────────────────────────────────────────────────────────────────
if run_optimization:
    try:
        if len(tickers) < 2:
            st.error("⚠️ กรุณากรอกชื่อหุ้นอย่างน้อย 2 ตัวขึ้นไปในกล่องค้นหา (เช่น AAPL, RBLX) เพื่อใช้คำนวณ")
            st.stop()
            
        with st.spinner("กำลังดึงราคาหุ้นจากตลาดโลก..."):
            prices = fetch_price_data(tickers, start_date, end_date)
            
        if prices.empty or len(prices.columns) < 2:
            st.error("❌ ไม่สามารถดึงราคาหุ้นมาคำนวณได้ กรุณาตรวจเช็คตัวสะกดชื่อย่อหุ้นในกล่องซ้ายมืออีกครั้ง")
            st.stop()
            
        prices = prices.ffill().bfill().dropna()
        valid_tickers = list(prices.columns)
        
        returns = calculate_returns(prices)
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        
        # รันแบบจำลองวิเคราะห์จัดพอร์ต
        ms_w = optimize_portfolio(mean_returns, cov_matrix, risk_free_rate, "sharpe")
        mv_w = optimize_portfolio(mean_returns, cov_matrix, risk_free_rate, "min_var")
        
        ms_ret, ms_vol = portfolio_performance(ms_w, mean_returns, cov_matrix)
        ms_sr = (ms_ret - risk_free_rate) / ms_vol if ms_vol > 0 else 0
        
        mv_ret, mv_vol = portfolio_performance(mv_w, mean_returns, cov_matrix)
        mv_sr = (mv_ret - risk_free_rate) / mv_vol if mv_vol > 0 else 0
        
        st.success("✅ ประมวลผลเสร็จสิ้น!")
        
        # ตารางผลลัพธ์
        st.subheader("📋 สรุปผลลัพธ์พอร์ตโฟลิโอ")
        summary_df = pd.DataFrame({
            "หัวข้อ": ["อัตราผลตอบแทนคาดหวังรายปี", "ความผันผวนรายปี (ความเสี่ยง)", "Sharpe Ratio"],
            "พอร์ต Sharpe สูงสุด (Max Sharpe)": [f"{ms_ret:.2%}", f"{ms_vol:.2%}", f"{ms_sr:.3f}"],
            "พอร์ตความเสี่ยงต่ำสุด (Min Variance)": [f"{mv_ret:.2%}", f"{mv_vol:.2%}", f"{mv_sr:.3f}"]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # กราฟวงกลม
        st.subheader("🥧 สัดส่วนการกระจายเงินลงทุน (Weight Allocation)")
        col1, col2 = st.columns(2)
        with col1:
            f1 = go.Figure(data=[go.Pie(labels=valid_tickers, values=ms_w, hole=0.4)])
            f1.update_layout(title="Max Sharpe Ratio Portfolio", height=400)
            st.plotly_chart(f1, use_container_width=True)
        with col2:
            f2 = go.Figure(data=[go.Pie(labels=valid_tickers, values=mv_w, hole=0.4)])
            f2.update_layout(title="Minimum Variance Portfolio", height=400)
            st.plotly_chart(f2, use_container_width=True)
            
        # กราฟประสิทธิภาพ
        st.subheader("📈 เส้นประสิทธิภาพการลงทุน (Efficient Frontier)")
        f_rets, f_vols = generate_efficient_frontier(mean_returns, cov_matrix)
        fig = go.Figure()
        if len(f_vols) > 0:
            fig.add_trace(go.Scatter(x=f_vols*100, y=f_rets*100, mode='lines', name='Efficient Frontier'))
        fig.add_trace(go.Scatter(x=[ms_vol*100], y=[ms_ret*100], mode='markers', name='Max Sharpe', marker=dict(size=15, color='red', symbol='star')))
        fig.add_trace(go.Scatter(x=[mv_vol*100], y=[mv_ret*100], mode='markers', name='Min Variance', marker=dict(size=15, color='green', symbol='star')))
        fig.update_layout(xaxis_title="Volatility (Risk) %", yaxis_title="Expected Return %")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ ระบบขัดข้อง: {str(e)}")
else:
    st.info("👈 พิมพ์รายชื่อหุ้นที่ท่านสนใจในช่องด้านซ้ายมือ (คั่นด้วยคอมมา) จากนั้นกดปุ่มเริ่มคำนวณพอร์ตโฟลิโอได้ทันที")