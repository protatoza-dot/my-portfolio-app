import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("📊 Portfolio Optimizer")

# 1. ช่องรับข้อมูลหุ้นแบบพิมพ์เอง (คลีนที่สุด)
ticker_input = st.text_input("พิมพ์ชื่อหุ้น (คั่นด้วยคอมม่า):", "RBLX, NVDA, AMD")
run_btn = st.button("🚀 คำนวณพอร์ตโฟลิโอ")

if run_btn:
    tickers = [t.strip().upper() for t in ticker_input.split(",")]
    
    with st.spinner("กำลังดึงข้อมูล..."):
        # ใช้แค่ yf.download แบบธรรมดาที่สุดเพื่อลดโอกาสค้าง
        data = yf.download(tickers, period="2y")['Adj Close']
        
    if data.empty:
        st.error("ไม่พบข้อมูลหุ้น กรุณาตรวจสอบชื่อหุ้น")
    else:
        # คำนวณเบื้องต้น
        returns = data.pct_change().dropna()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # แสดงผลลัพธ์
        st.success("คำนวณเสร็จแล้ว!")
        st.write("ผลตอบแทนรายวันเฉลี่ย:", mean_returns)
        
        # ตัวอย่างกราฟแบบเรียบง่าย
        fig = go.Figure(data=[go.Bar(x=mean_returns.index, y=mean_returns.values)])
        fig.update_layout(title="ผลตอบแทนเฉลี่ย")
        st.plotly_chart(fig)