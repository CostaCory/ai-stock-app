import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import requests
import time
import os
from datetime import datetime

# 載入模型
model = joblib.load("ai_model_3day_retrained.pkl")

# 自動交易紀錄清單
auto_trade_records = []

# 自動記錄函數
def auto_log_trade_record(ticker, price, rsi, macd, golden_cross, ai_pred):
    stop = round(price * 0.95, 2)
    target = round(price * 1.05, 2)
    suggestion = "黃金交叉 + RSI < 60" if golden_cross and rsi < 60 else "不符技術條件"
    ai_direction = "升" if ai_pred == 1 else "跌"
    action = "✅ 值得入市" if ai_pred == 1 and golden_cross and rsi < 60 else "❌ 建議觀望"

    record = {
        "日期": datetime.today().strftime("%Y-%m-%d"),
        "股票代號": ticker,
        "入市價": price,
        "止蝕價": stop,
        "目標價": target,
        "AI 預測方向": ai_direction,
        "RSI": round(rsi, 2),
        "MACD": round(macd, 2),
        "策略條件": suggestion,
        "建議狀態": action,
        "交易狀態": "等待入市"
    }
    return record

# Streamlit 頁面設定
st.set_page_config(page_title="AI 股票分析工具（含交易紀錄）", layout="wide")
st.title("📊 AI 股票分析 + 自動交易紀錄工具")

watchlist = st.sidebar.text_area("輸入股票代號（用逗號分隔）", value="TSLA,NVDA")
watchlist = [s.strip().upper() for s in watchlist.split(",") if s.strip()]

st.sidebar.header("📁 可選：上傳你自己 CSV 資料")
uploaded_file = st.sidebar.file_uploader("選擇 CSV 檔（收市欄需叫做「收市」）")

records = []

for symbol in watchlist:
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "收市" in df.columns:
            df.rename(columns={"收市": "Close"}, inplace=True)
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df.dropna(subset=["Close"], inplace=True)
    else:
        df = yf.download(symbol, period="6mo", interval="1d")
    
    if df.empty or "Close" not in df.columns:
        st.warning(f"{symbol} 資料無效，跳過")
        continue

    df["SMA10"] = df["Close"].rolling(window=10).mean()
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df.dropna(inplace=True)
    latest = df.iloc[-1]
    features = ["RSI", "MACD", "MACD_signal", "SMA10", "SMA20", "Close"]
    X = latest[features].values.reshape(1, -1)
    prediction = model.predict(X)[0]
    golden_cross = latest["SMA10"] > latest["SMA20"] and df.iloc[-2]["SMA10"] <= df.iloc[-2]["SMA20"]
    
    r = auto_log_trade_record(
        ticker=symbol,
        price=round(latest["Close"], 2),
        rsi=latest["RSI"],
        macd=latest["MACD"],
        golden_cross=golden_cross,
        ai_pred=prediction
    )
    auto_trade_records.append(r)

# 顯示交易紀錄表格
df_records = pd.DataFrame(auto_trade_records)
st.subheader("🧾 自動交易紀錄表")
st.dataframe(df_records)

# 允許下載 CSV
csv = df_records.to_csv(index=False).encode('utf-8-sig')
st.download_button("⬇️ 下載交易紀錄 CSV", csv, "trade_records.csv", "text/csv")
