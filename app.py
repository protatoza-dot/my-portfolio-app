import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from scipy.optimize import minimize
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# 1. การตั้งค่าหน้าเว็บและระบบสลับภาษา (Bilingual UI Support)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Portfolio Optimizer", page_icon="📊", layout="wide")

# Custom CSS รองรับการแสดงผลบนมือถือ (iOS/Android) และ Desktop
st.markdown("""
<style>
    .reportview-container .main .block-container { max-width: 1200px; padding: 2rem; }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
    @media (max-width: 768px) {
        .stPlotlyChart { width: 100% !important; }
    }
</style>
""", unsafe_allow_html=True)

# พจนานุกรมสำหรับระบบ 2 ภาษา
LANG_DICT = {
    "ไทย": {
        "title": "📊 ระบบจัดพอร์ตการลงทุน (Markowitz Model)",
        "subtitle": "แบบจำลอง Mean-Variance และเส้นประสิทธิภาพ (Efficient Frontier)",
        "sidebar_header": "⚙️ ตั้งค่าพอร์ตฟอลิโอ",
        "lang_label": "เปลี่ยนภาษา / Language",
        "ticker_label": "เลือกหรือพิมพ์ชื่อหุ้น (เช่น AAPL, NVDA, PTT.BK)",
        "ticker_help": "พิมพ์ชื่อย่อหุ้นแล้วกดเลือก หรือพิมพ์ชื่อหุ้นอื่นนอกเหนือจากรายการแล้วกด Enter ได้เลย",
        "date_label": "ช่วงเวลาข้อมูลย้อนหลัง",
        "rf_label": "อัตราผลตอบแทนปราศจากความเสี่ยง (%)",
        "btn_run": "🚀 เริ่มคำนวณพอร์ตฟอลิโอ",
        "tab_weights": "🍰 สัดส่วนการลงทุน (Weights)",
        "tab_frontier": "📈 เส้นประสิทธิภาพ Efficient Frontier",
        "tab_summary": "📋 ตารางสรุปผลลัพธ์",
        "metric_return": "ผลตอบแทนคาดหวังรายปี",
        "metric_risk": "ความผันผวนรายปี (ความเสี่ยง)",
        "metric_sharpe": "Sharpe Ratio",
        "max_sharpe": "พอร์ต Sharpe สูงสุด (Max Sharpe)",
        "min_var": "พอร์ตความเสี่ยงต่ำสุด (Min Variance)",
        "asset": "ชื่อหุ้น / สินทรัพย์",
        "weight_pct": "สัดส่วนที่ควรลงทุน (%)",
        "err_min": "⚠️ กรุณาเลือกหุ้นอย่างน้อย 2 ตัวขึ้นไปเพื่อคำนวณจัดพอร์ต",
        "err_fetch": "❌ เกิดข้อผิดพลาด: ไม่สามารถดึงข้อมูลหุ้นได้ กรุณาตรวจสอบชื่อหุ้นหรือช่วงเวลาที่เลือกอีกครั้ง"
    },
    "English": {
        "title": "📊 Mean-Variance Portfolio Optimization",
        "subtitle": "Markowitz Model — Efficient Frontier Analysis",
        "sidebar_header": "⚙️ Configuration",
        "lang_label": "Language / เปลี่ยนภาษา",
        "ticker_label": "Search or Select Tickers (e.g., AAPL, NVDA, PTT.BK)",
        "ticker_help": "Type to search popular tickers or type a custom symbol and press Enter.",
        "date_label": "Historical Date Range",
        "rf_label": "Risk-Free Rate (%)",
        "btn_run": "🚀 Run Optimization",
        "tab_weights": "🍰 Asset Allocation Weights",
        "tab_frontier": "📈 Efficient Frontier Curve",
        "tab_summary": "📋 Performance Summary Table",
        "metric_return": "Expected Annual Return",
        "metric_risk": "Annual Volatility (Risk)",
        "metric_sharpe": "Sharpe Ratio",
        "max_sharpe": "Maximum Sharpe Ratio Portfolio",
        "min_var": "Minimum Variance Portfolio",
        "asset": "Asset Ticker",
        "weight_pct": "Optimal Weight (%)",
        "err_min": "⚠️ Please select at least 2 tickers for optimization.",
        "err_fetch": "❌ Error fetching data. Please verify the tickers or date range."
    }
}

# รายชื่อหุ้นยอดนิยมสำหรับระบบแนะนำ (Autocomplete Helper)
POPULAR_TICKERS = [
    "AAPL", "MSFT", "GOOG", "NVDA", "TSLA", "AMD", "META", "AMZN", "NFLX",
    "PTT.BK", "CPALL.BK", "BDMS.BK", "SCC.BK", "AOT.BK", "KBANK.BK", "SCB.BK"
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. ส่วนแสดงผลแถบข้าง (Sidebar)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    lang = st.selectbox("Language/ภาษา", ["ไทย", "English"])
    T = LANG_DICT[lang] # ตัวแปรย่อสำหรับดึงข้อความภาษาที่เลือก
    
    st.markdown("---")
    st.header(T["sidebar_header"])
    
    # ระบบค้นหาและแนะนำหุ้น (Autocomplete) รองรับการพิมพ์เพิ่มเอง
    selected_tickers = st.multiselect(
        T["ticker_label"],
        options=list(set(POPULAR_TICKERS)),
        default=["AAPL", "MSFT", "GOOG", "NVDA"],
        help=T["ticker_help"]
    )
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(f"{T['date_label']} (Start)", datetime.today() - timedelta(days=5*365))
    with col2:
        end_date = st.date_input(f"{T['date_label']} (End)", datetime.today())
        
    rf_rate = st.number_input(T["rf_label"], min_value=0.0, max_value=20.0, value=2.0, step=0.1) / 100

# ─────────────────────────────────────────────────────────────────────────────
# 3. ส่วนคำนวณทางคณิตศาสตร์และการเงิน (Financial Engine)
# ─────────────────────────────────────────────────────────────────────────────
def get_portfolio_stats(weights, returns_mean, returns_cov, rf_rate):
    port_return = np.sum(returns_mean * weights) * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(returns_cov * 252, weights)))
    sharpe_ratio = (port_return - rf_rate) / port_vol
    return port_return, port_vol, sharpe_ratio

def minimize_sharpe(weights, returns_mean, returns_cov, rf_rate):
    return -get_portfolio_stats(weights, returns_mean, returns_cov, rf_rate)[2]

def minimize_variance(weights, returns_mean, returns_cov, rf_rate):
    return get_portfolio_stats(weights, returns_mean, returns_cov, rf_rate)[1]**2

# ─────────────────────────────────────────────────────────────────────────────
# 4. ส่วนหน้าจอหลักและการแสดงผล (Main UI Logic)
# ─────────────────────────────────────────────────────────────────────────────
st.title(T["title"])
st.markdown(f"*{T['subtitle']}*")

if st.sidebar.button(T["btn_run"]):
    if len(selected_tickers) < 2:
        st.error(T["err_min"])
    else:
        with st.spinner('Calculating...'):
            try:
                # จัดการข้อมูลนำเข้า: ตัดช่องว่างและแปลงเป็นตัวพิมพ์ใหญ่
                tickers = [t.strip().upper() for t in selected_tickers]
                
                # ดึงราคาย้อนหลังจาก yfinance
                df = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
                
                if df.empty or (len(tickers) > 1 and df.shape[1] != len(tickers)):
                    raise ValueError()
                
                # คำนวณผลตอบแทนและความแปรปรวนร่วม
                returns = df.pct_change().dropna()
                returns_mean = returns.mean()
                returns_cov = returns.cov()
                
                num_assets = len(tickers)
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                bounds = tuple((0, 1) for _ in range(num_assets))
                init_guess = num_assets * [1. / num_assets]
                
                #คำนวณ พอร์ต Max Sharpe
                opt_sharpe = minimize(minimize_sharpe, init_guess, args=(returns_mean, returns_cov, rf_rate), method='SLSQP', bounds=bounds, constraints=constraints)
                w_sharpe = opt_sharpe['x']
                ret_sh, vol_sh, sr_sh = get_portfolio_stats(w_sharpe, returns_mean, returns_cov, rf_rate)
                
                # คำนวณ พอร์ต Min Variance
                opt_var = minimize(minimize_variance, init_guess, args=(returns_mean, returns_cov, rf_rate), method='SLSQP', bounds=bounds, constraints=constraints)
                w_var = opt_var['x']
                ret_v, vol_v, sr_v = get_portfolio_stats(w_var, returns_mean, returns_cov, rf_rate)
                
                # คำนวณเส้น Efficient Frontier Curve
                target_returns = np.linspace(ret_v, returns_mean.max() * 252, 30)
                frontier_vols = []
                for target in target_returns:
                    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                            {'type': 'eq', 'fun': lambda x: get_portfolio_stats(x, returns_mean, returns_cov, rf_rate)[0] - target})
                    res = minimize(minimize_variance, init_guess, args=(returns_mean, returns_cov, rf_rate), method='SLSQP', bounds=bounds, constraints=cons)
                    frontier_vols.append(np.sqrt(res['fun']))

                # แสดงผลแบ่งเป็น 3 Tabs
                tab1, tab2, tab3 = st.tabs([T["tab_weights"], T["tab_frontier"], T["tab_summary"]])
                
                with tab1: # กราฟวงกลมแสดงสัดส่วน
                    c1, c2 = st.columns(2)
                    with c1:
                        fig1 = go.Figure(data=[go.Pie(labels=tickers, values=w_sharpe, textinfo='percent+label')])
                        fig1.update_layout(title=T["max_sharpe"], height=400)
                        st.plotly_chart(fig1, use_container_width=True)
                    with c2:
                        fig2 = go.Figure(data=[go.Pie(labels=tickers, values=w_var, textinfo='percent+label')])
                        fig2.update_layout(title=T["min_var"], height=400)
                        st.plotly_chart(fig2, use_container_width=True)
                        
                with tab2: # กราฟ Efficient Frontier
                    fig_ef = go.Figure()
                    fig_ef.add_trace(go.Scatter(x=frontier_vols, y=target_returns, mode='lines', name='Efficient Frontier', line=dict(color='black', width=2)))
                    for i, ticker in enumerate(tickers):
                        stk_vol = np.sqrt(returns_cov.iloc[i, i] * 252)
                        stk_ret = returns_mean.iloc[i] * 252
                        fig_ef.add_trace(go.Scatter(x=[stk_vol], y=[stk_ret], mode='markers+text', name=ticker, text=[ticker], textposition="top center"))
                    fig_ef.add_trace(go.Scatter(x=[vol_sh], y=[ret_sh], mode='markers', name=T["max_sharpe"], marker=dict(color='gold', size=14, symbol='star', line=dict(color='black', width=1))))
                    fig_ef.add_trace(go.Scatter(x=[vol_v], y=[ret_v], mode='markers', name=T["min_var"], marker=dict(color='red', size=14, symbol='diamond', line=dict(color='black', width=1))))
                    fig_ef.update_layout(xaxis_title=T["metric_risk"], yaxis_title=T["metric_return"], height=500)
                    st.plotly_chart(fig_ef, use_container_width=True)
                    
                with tab3: # ตารางสรุปข้อมูลคำนวณ
                    summary_df = pd.DataFrame({
                        "Metric": [T["metric_return"], T["metric_risk"], T["metric_sharpe"]],
                        T["max_sharpe"]: [f"{ret_sh*100:.2f}%", f"{vol_sh*100:.2f}%", f"{sr_sh:.2f}"],
                        T["min_var"]: [f"{ret_v*100:.2f}%", f"{vol_v*100:.2f}%", f"{sr_v:.2f}"]
                    })
                    st.table(summary_df)
                    
                    st.subheader(T["tab_weights"])
                    weights_df = pd.DataFrame({
                        T["asset"]: tickers,
                        T["max_sharpe"]: [f"{w*100:.2f}%" for w in w_sharpe],
                        T["min_var"]: [f"{w*100:.2f}%" for w in w_var]
                    })
                    st.dataframe(weights_df, use_container_width=True)

            except Exception:
                st.error(T["err_fetch"])