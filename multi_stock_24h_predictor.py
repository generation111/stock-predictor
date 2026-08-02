import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import time
import requests

# 1. 初始化 NLTK Vader 詞庫 (自動下載並防錯)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

st.set_page_config(page_title="美股 24H 實時走勢預測工具", layout="wide")

# 2. 側邊欄：股票代號與系統參數設定
st.sidebar.header("🔍 美股代號與參數設定")

quick_tickers = ["ABVC", "TSLA", "NVDA", "AAPL", "AMD", "AMZN", "MSFT", "GOOGL", "META"]
selected_quick = st.sidebar.selectbox("🔥 熱門個股快速選擇", options=["自訂輸入"] + quick_tickers, index=1)

default_ticker = "ABVC" if selected_quick == "自訂輸入" else selected_quick
ticker_input = st.sidebar.text_input("輸入美股股票代號 (例如: ABVC, TSLA)", value=default_ticker).upper().strip()

st.sidebar.markdown("---")
include_extended = st.sidebar.checkbox("開啟夜盤與延長交易時段 (24H Extended)", value=True)
interval = st.sidebar.selectbox("K線時間間隔", ["1m", "5m", "15m", "1d"], index=1)
period = "5d" if interval in ["1m", "5m", "15m"] else "1y"
refresh_rate = st.sidebar.slider("自動更新頻率 (秒)", min_value=10, max_value=60, value=20)

# 3. 新聞情緒分析模組 (支援新舊版 yfinance API 與相容抓取)
def get_news_sentiment(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        news_list = ticker.news
        
        if not news_list:
            return 0.0, []
        
        sia = SentimentIntensityAnalyzer()
        compound_scores = []
        processed_news = []
        
        for item in news_list[:8]:
            title = ""
            publisher = "Unknown"
            link = "#"

            # 兼容 yfinance 新版 (content 字典) 與舊版格式
            if isinstance(item, dict) and 'content' in item:
                content = item.get('content', {})
                title = content.get('title', '')
                publisher = content.get('provider', {}).get('displayName', 'Unknown')
                click_through_url = content.get('clickThroughUrl', {})
                canonical_url = content.get('canonicalUrl', {})
                link = click_through_url.get('url') or canonical_url.get('url') or '#'
            else:
                title = item.get('title', '')
                publisher = item.get('publisher', 'Unknown')
                link = item.get('link', '#')

            if not title:
                continue
            
            score = sia.polarity_scores(title)['compound']
            compound_scores.append(score)
            processed_news.append({
                'title': title,
                'publisher': publisher,
                'link': link,
                'score': score
            })
        
        avg_score = float(np.mean(compound_scores)) if compound_scores else 0.0
        return avg_score, processed_news
    except Exception:
        return 0.0, []

# 4. 進階特徵工程 (技術指標 + 波動率 + 布林通道 + 情緒)
def compute_indicators(df, sentiment_score):
    data = df.copy()
    
    # 價格報酬率
    data['Returns'] = data['Close'].pct_change()
    
    # 均線指標
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    
    # RSI (相對強弱指標)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # 布林通道與位置指標 (BB_%B)
    std_20 = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['SMA_20'] + (std_20 * 2)
    data['BB_Lower'] = data['SMA_20'] - (std_20 * 2)
    data['BB_PctB'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'] + 1e-9)
    
    # 波動率 (針對夜盤與生技股短線波動優化)
    data['Volatility'] = data['Returns'].rolling(window=10).std()
    
    # 新聞情緒特徵
    data['Sentiment'] = sentiment_score
    
    # 預測目標：下一期收盤價高於當期為 1 (漲)，否則為 0 (跌)
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    return data.dropna()

# 5. 數據讀取與機器學習建模
@st.cache_data(ttl=10)
def fetch_and_predict_dynamic(ticker, period="5d", interval="5m", extended=True):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval, prepost=extended)
        
        if df.empty or len(df) < 30:
            return None, None, None, None, 0.0, [], None
            
        try:
            company_name = stock.info.get('longName', ticker)
        except Exception:
            company_name = ticker
        
        sentiment_score, news_data = get_news_sentiment(ticker)
        df = compute_indicators(df, sentiment_score)
        
        feature_cols = ['Returns', 'SMA_5', 'SMA_20', 'RSI', 'MACD', 'MACD_Signal', 'BB_PctB', 'Volatility', 'Sentiment']
        
        X = df[feature_cols][:-1]
        y = df['Target'][:-1]
        
        if len(X) < 20:
            return None, None, None, None, 0.0, [], None
            
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        test_acc = accuracy_score(y_test, model.predict(X_test))
        
        latest_feature = df[feature_cols].iloc[[-1]]
        latest_pred = int(model.predict(latest_feature)[0])
        latest_probs = model.predict_proba(latest_feature)[0]
        
        latest_prob = float(latest_probs[latest_pred])
        
        return df, latest_pred, latest_prob, test_acc, sentiment_score, news_data, company_name
    except Exception:
        return None, None, None, None, 0.0, [], None

# 6. 主畫面局部刷新模組
@st.fragment(run_every=refresh_rate)
def render_dashboard(symbol, p_period, p_interval, p_extended):
    if not symbol:
        st.warning("請在左側輸入美股股票代號。")
        return

    df, latest_pred, latest_prob, test_acc, sentiment_score, news_data, company_name = fetch_and_predict_dynamic(
        symbol, p_period, p_interval, p_extended
    )

    if df is not None:
        st.title(f"⚡ {symbol} ({company_name}) 全時段實時走勢預測")
        
        latest_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        price_change = latest_price - prev_price
        pct_change = (price_change / prev_price) * 100

        # 頂部關鍵資訊指標
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"{symbol} 當前報價 (含夜盤)", f"${latest_price:.2f}", f"{price_change:+.2f} ({pct_change:+.2f}%)")
        
        pred_text = "🟢 看漲 (UP)" if latest_pred == 1 else "🔴 看跌 (DOWN)"
        
        col2.metric("下一週期預測方向", pred_text)
        col3.metric("預測信心度", f"{latest_prob * 100:.1f}%")
        
        sentiment_label = "中性 🟡"
        if sentiment_score > 0.05: sentiment_label = "偏多 🟢"
        elif sentiment_score < -0.05: sentiment_label = "偏空 🔴"
        
        col4.metric("即時新聞情緒指數", f"{sentiment_score:+.3f}", sentiment_label)

        st.markdown("---")

        # K 線圖
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'], name=symbol
        ))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_5'], line=dict(color='orange', width=1), name="SMA 5"))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='cyan', width=1), name="SMA 20"))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', width=1, dash='dash'), name="布林上軌"))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', width=1, dash='dash'), name="布林下軌"))
        
        fig.update_layout(
            title=f"{symbol} 全時段即時 K 線圖 (時間間隔: {p_interval})",
            xaxis_rangeslider_visible=False,
            height=500,
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

        # 底部資訊區
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.subheader(f"📰 {symbol} 最新新聞與情緒得分")
            if news_data:
                for item in news_data[:5]:
                    score_color = "green" if item['score'] > 0 else ("red" if item['score'] < 0 else "gray")
                    st.markdown(f"• [{item['title']}]({item['link']})")
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;來源: *{item['publisher']}* | 情緒分數: <span style='color:{score_color}'>{item['score']:+.2f}</span>", unsafe_allow_html=True)
            else:
                st.info("暫無該個股之即時新聞數據。")

        with col_right:
            st.subheader("💡 預測模式與指標說明")
            st.info(f"""
            * **預測目標股票**：當前為 **{symbol} ({company_name})**。
            * **全時段夜盤支援**：開啟延長交易後，系統會自動捕捉隔夜與盤前盤後跳動，提供 24 小時不間斷預測。
            * **歷史測試集準確率**：目前特徵模型在 {symbol} 歷史驗證集的測試準確率約為 **{test_acc * 100:.1f}%**。
            """)
    else:
        st.error(f"⚠️ 無法找到代號 **{symbol}** 的交易數據。請檢查股票代號是否正確，或嘗試更換時間間隔。")

# 執行主畫面渲染
render_dashboard(ticker_input, period, interval, include_extended)
