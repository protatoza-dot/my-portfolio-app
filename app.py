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
st.set_page_config(page_title="Global Portfolio Optimizer", page_icon="📊", layout="wide")

# พจนานุกรมข้อความสำหรับระบบ 2 ภาษา
LANG_DICT = {
    "ไทย": {
        "title": "📊 ระบบจัดพอร์ตการลงทุนระดับโลก (Global Markowitz Model)",
        "subtitle": "ดึงข้อมูลจาก Yahoo Finance API ค้นหาและพิมพ์เพิ่มหุ้นได้ทุกตัวในโลกที่ IPO แล้ว",
        "sidebar_header": "⚙️ ตั้งค่าพอร์ตฟอลิโอ",
        "ticker_label": "พิมพ์หรือเลือกชื่อหุ้น (พิมพ์ชื่อย่อเสร็จแล้วกด Enter เพื่อเพิ่มได้)",
        "ticker_help": "พิมพ์ชื่อย่อหุ้นตัวไหนก็ได้ในโลกแล้วกด Enter เพื่อเพิ่มเข้าพอร์ต เช่น TSM, ASML, NVDA, PTT.BK",
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
        "err_min": "⚠️ กรุณาเลือกหรือพิมพ์ชื่อหุ้นอย่างน้อย 2 ตัวขึ้นไปเพื่อคำนวณจัดพอร์ต",
        "err_fetch": "❌ เกิดข้อผิดพลาด: ไม่สามารถดึงข้อมูลหุ้นบางตัวได้ กรุณาตรวจสอบตัวย่อหุ้นให้ถูกต้องตามหลัก Yahoo Finance หรือขยายช่วงเวลาข้อมูลย้อนหลัง"
    },
    "English": {
        "title": "📊 Global Portfolio Optimization",
        "subtitle": "Access any IPO'd stock worldwide via Yahoo Finance API with custom inputs",
        "sidebar_header": "⚙️ Configuration",
        "ticker_label": "Search or Type Tickers (Press Enter to add custom symbols)",
        "ticker_help": "Type any valid global ticker and press Enter to insert into the portfolio.",
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
        "err_fetch": "❌ Error fetching data. Please verify ticker symbols or date range."
    }
}

# กำหนดรายชื่อหุ้นยอดนิยมเริ่มต้นในระบบเดาคำ (Autocomplete Base)
if "popular_tickers" not in st.session_state:
    st.session_state.popular_tickers = ["AAPL", "NVDA", "TSM", "ASML", "MSFT", "GOOG", "AMZN", "META", "PTT.BK", "CPALL.BK"]

# ─────────────────────────────────────────────────────────────────────────────
# 2. แถบข้าง (Sidebar) พร้อมช่องกรอกหุ้นเดาคำ + พิมพ์กด Enter เพิ่มเองได้อิสระ
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    lang = st.selectbox("Language/ภาษา", ["ไทย", "English"])
    T = LANG_DICT[lang]
    
    st.markdown("---")
    st.header(T["sidebar_header"])
    
    if "selected_list" not in st.session_state:
        st.session_state.selected_list = ["AAPL", "NVDA", "TSM", "ASML"]
        
    # กล่องข้อความแบบพิมพ์แล้วเดาคำ หรือสามารถพิมพ์ตัวย่อหุ้นสากลตัวไหนก็ได้แล้วกด Enter
    selected_tickers = st.multiselect(
        T["ticker_label"],
        options=st.session_state.popular_tickers,
        default=st.session_state.selected_list,
        help=T["ticker_help"]
    )
    
    # ระบบดักจับอัจฉริยะ: ถ้าผู้ใช้พิมพ์หุ้นแปลกใหม่นอกรายการแล้วกด Enter จะเอาไปใส่เพิ่มในระบบทันที
    updated = False
    for ticker in selected_tickers:
        if ticker not in st.session_state.popular_tickers:
            st.session_state.popular_tickers.append(ticker)
            updated = True
    if updated:
        st.session_state.selected_list = selected_tickers
        st.rerun()

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(f"{T['date_label']} (Start)", datetime.today() - timedelta(days=5*365))
    with col2:
        end_date = st.date_input(f"{T['date_label']} (End)", datetime.today())
        
    rf_rate = st.number_input(T["rf_label"], min_value=0.0, max_value=20.0, value=2.0, step=0.1) / 100

# ─────────────────────────────────────────────────────────────────────────────
# 3. ฟังก์ชันคำนวณทางคณิตศาสตร์การเงิน (Financial Calculation Core)
# ─────────────────────────────────────────────────────────────────────────────
def get_portfolio_stats(weights, returns_mean, returns_cov, rf_rate):
    # ผลตอบแทนรายปี = สัดส่วนการลงทุนคูณผลตอบแทนเฉลี่ยรายวันคูณ 252 วันทำการ
    port_return = np.sum(returns_mean * weights) * 252
    # ความผันผวนรายปี = สแควร์รูทของ Variance พอร์ตรายปี
    port_vol = np.sqrt(np.dot(weights.T, np.dot(returns_cov * 252, weights)))
    sharpe_ratio = (port_return - rf_rate) / port_vol
    return port_return, port_vol, sharpe_ratio

def minimize_sharpe(weights, returns_mean, returns_cov, rf_rate):
    return -get_portfolio_stats(weights, returns_mean, returns_cov, rf_rate)[2]

def minimize_variance(weights, returns_mean, returns_cov, rf_rate):
    return get_portfolio_stats(weights, returns_mean, returns_cov, rf_rate)[1]**2

# ─────────────────────────────────────────────────────────────────────────────
# 4. ส่วนแสดงผลหลักและการเชื่อมต่อข้อมูล (Main UI & Secure Fetching)
# ─────────────────────────────────────────────────────────────────────────────
st.title(T["title"])
st.markdown(f"*{T['subtitle']}*")

if st.sidebar.button(T["btn_run"]):
    clean_tickers = [t.strip().upper() for t in selected_tickers if t.strip()]
    
    if len(clean_tickers) < 2:
        st.error(T["err_min"])
    else:
        with st.spinner('Processing global market data...'):
            try:
                # วิธีดึงราคาสดที่ปลอดภัยที่สุด: สั่งดึงเฉพาะราคาแบบปิดตัวจริง (group_by='ticker') ป้องกันคอลัมน์พัง
                df_raw = yf.download(clean_tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
                
                if df_raw.empty:
                    raise ValueError("No records found")
                
                # สกัดข้อมูลราคาเฉพาะช่องข้อมูลปิดตลาดของหุ้นแต่ละตัวอย่างปลอดภัย
                df = pd.DataFrame()
                if len(clean_tickers) == 1:
                    df[clean_tickers[0]] = df_raw["Close"]
                else:
                    for ticker in clean_tickers:
                        if ticker in df_raw.columns.get_level_values(0) or ticker in df_raw.columns:
                            # ดึงข้อมูลกรณีมี Multi-index หรือโครงสร้างปกติ
                            if isinstance(df_raw.columns, pd.MultiIndex):
                                df[ticker] = df_raw[ticker]["Close"]
                            else:
                                df[ticker] = df_raw["Close"]
                
                # ลบคอลัมน์ที่ว่างออกและสกัดรายชื่อหุ้นที่ดึงสำเร็จจริง ๆ
                df = df.dropna(axis=1, how='all').dropna()
                actual_tickers = list(df.columns)
                
                if len(actual_tickers) < 2:
                    st.error(T["err_min"])
                    st.stop()
                
                # คำนวณเปอร์เซ็นต์ผลตอบแทนแบบ Log Returns มีความแม่นยำทางสถิติสูงตรงตามโค้ดเดิม
                returns = np.log(df / df.shift(1)).dropna()
                returns_mean = returns.mean().values
                returns_cov = returns.cov().values
                
                num_assets = len(actual_tickers)
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                bounds = tuple((0, 1) for _ in range(num_assets))
                init_guess = num_assets * [1. / num_assets]
                
                # 1. ประมวลผลจุดพอร์ตสัดส่วน Sharpe Ratio สูงสุด
                opt_sharpe = minimize(minimize_sharpe, init_guess, args=(returns_mean, returns_cov, rf_rate), method='SLSQP', bounds=bounds, constraints=constraints)
                w_sharpe = opt_sharpe['x']
                ret_sh, vol_sh, sr_sh = get_portfolio_stats(w_sharpe, returns_mean, returns_cov, rf_rate)
                
                # 2. ประมวลผลจุดพอร์ตสัดส่วนความเสี่ยงต่ำที่สุด (Min Variance)
                opt_var = minimize(minimize_variance, init_guess, args=(returns_mean, returns_cov, rf_rate), method='SLSQP', bounds=bounds, constraints=constraints)
                w_var = opt_var['x']
                ret_v, vol_v, sr_v = get_portfolio_stats(w_var, returns_mean, returns_cov, rf_rate)
                
                # 3. คำนวณประกอบจุดโค้งประสิทธิภาพพอร์ตฟอลิโอ (Efficient Frontier)
                target_returns = np.linspace(ret_v, np.max(returns_mean) * 252, 30)
                frontier_vols = []
                for target in target_returns:
                    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                            {'type': 'eq', 'fun': lambda x: get_portfolio_stats(x, returns_mean, returns_cov, rf_rate)[0] - target})
                    res = minimize(minimize_variance, init_guess, args=(returns_mean, returns_cov, rf_rate), method='SLSQP', bounds=bounds, constraints=cons)
                    if res.success:
                        frontier_vols.append(np.sqrt(res['fun'] * 252))
                    else:
                        frontier_vols.append(np.nan)

                valid_frontier = [(v, r) for v, r in zip(frontier_vols, target_returns) if not np.isnan(v)]
                f_vols, f_rets = zip(*valid_frontier) if valid_frontier else ([], [])

                # จำแนกแท็บแสดงผลกราฟและสถิติแยกย่อย
                tab1, tab2, tab3 = st.tabs([T["tab_weights"], T["tab_frontier"], T["tab_summary"]])
                
                with tab1:
                    c1, c2 = st.columns(2)
                    with c1:
                        fig1 = go.Figure(data=[go.Pie(labels=actual_tickers, values=w_sharpe, hole=0.4, textinfo='label+percent')])
                        fig1.update_layout(title=T["max_sharpe"], height=400)
                        st.plotly_chart(fig1, use_container_width=True)
                    with c2:
                        fig2 = go.Figure(data=[go.Pie(labels=actual_tickers, values=w_var, hole=0.4, textinfo='label+percent')])
                        fig2.update_layout(title=T["min_var"], height=400)
                        st.plotly_chart(fig2, use_container_width=True)
                        
                with tab2:
                    fig_ef = go.Figure()
                    if f_vols:
                        fig_ef.add_trace(go.Scatter(x=np.array(f_vols)*100, y=np.array(f_rets)*100, mode='lines', name='Efficient Frontier', line=dict(color='#636EFA', width=3)))
                    
                    # พล็อตตำแหน่งหุ้นแต่ละตัวเดี่ยว ๆ บนแผนผัง
                    for i, ticker in enumerate(actual_tickers):
                        stk_vol = np.sqrt(returns_cov[i, i] * 252)
                        stk_ret = returns_mean[i] * 252
                        fig_ef.add_trace(go.Scatter(x=[stk_vol*100], y=[stk_ret*100], mode='markers+text', name=ticker, text=[ticker], textposition="top center"))
                    
                    fig_ef.add_trace(go.Scatter(x=[vol_sh*100], y=[ret_sh*100], mode='markers', name=T["max_sharpe"], marker=dict(color='gold', size=16, symbol='star', line=dict(color='black', width=1))))
                    fig_ef.add_trace(go.Scatter(x=[vol_v*100], y=[ret_v*100], mode='markers', name=T["min_var"], marker=dict(color='red', size=16, symbol='diamond', line=dict(color='black', width=1))))
                    fig_ef.update_layout(xaxis_title="Annual Volatility (%)", yaxis_title="Expected Annual Return (%)", height=500)
                    st.plotly_chart(fig_ef, use_container_width=True)
                    
                with tab3:
                    summary_df = pd.DataFrame({
                        "Metric": [T["metric_return"], T["metric_risk"], T["metric_sharpe"]],
                        T["max_sharpe"]: [f"{ret_sh:.2%}", f"{vol_sh:.2%}", f"{sr_sh:.3f}"],
                        T["min_var"]: [f"{ret_v:.2%}", f"{vol_v:.2%}", f"{sr_v:.3f}"]
                    })
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    
                    st.subheader(T["tab_weights"])
                    weights_df = pd.DataFrame({
                        T["asset"]: actual_tickers,
                        T["max_sharpe"]: [f"{w*100:.2f}%" for w in w_sharpe],
                        T["min_var"]: [f"{w*100:.2f}%" for w in w_var]
                    })
                    st.dataframe(weights_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(T["err_fetch"])
                st.caption(f"Debug Info: {str(e)}")