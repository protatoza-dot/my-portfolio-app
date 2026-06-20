import streamlit as st
import yfinance as yf
import requests
from streamlit_searchbox import st_searchbox

# ฟังก์ชันดึงรายชื่อหุ้นจาก Yahoo แบบกูเกิ้ล
def search_stocks(search_term):
    if not search_term: return []
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={search_term}&quotesCount=10"
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).json()
        return [f"{q['symbol']} - {q.get('shortname', '')}" for q in res.get('quotes', [])]
    except:
        return []

st.title("📊 Portfolio Optimizer")

# กล่องค้นหาแบบ Google-style
selected_stock = st_searchbox(
    search_stocks,
    placeholder="🔍 พิมพ์ชื่อหุ้น เช่น RBLX, NVDA...",
    key="searchbox"
)

# เก็บหุ้นที่เลือกไว้ใน Session
if "my_portfolio" not in st.session_state:
    st.session_state.my_portfolio = []

if selected_stock:
    symbol = selected_stock.split(" - ")[0]
    if symbol not in st.session_state.my_portfolio:
        st.session_state.my_portfolio.append(symbol)

# แสดงรายชื่อหุ้นแบบสะอาดๆ ไม่รก
st.write("### 💼 หุ้นในพอร์ตของท่าน:")
cols = st.columns(4)
for i, ticker in enumerate(st.session_state.my_portfolio):
    if cols[i % 4].button(f"❌ {ticker}"):
        st.session_state.my_portfolio.remove(ticker)
        st.rerun()

# ปุ่มคำนวณ (เอาไว้ด้านล่างสุด)
if st.button("🚀 คำนวณพอร์ตโฟลิโอ"):
    st.write("กำลังรันโมเดล...")
    # (โค้ดคำนวณ Markowitz ของพี่ใส่ตรงนี้ต่อได้เลยครับ)