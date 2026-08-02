import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import datetime
from deep_translator import GoogleTranslator

# 1. 初始化 NLTK Vader 詞庫
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

st.set_page_config(page_title="華人跨國市場 24H 實時走勢預測工具", layout="wide")

# 初始化 Session State
if 'quick_tickers' not in st.session_state:
    st.session_state.quick_tickers = [
        "2330.TW",   # 台積電
        "2454.TW",   # 聯發科
        "0700.HK",   # 騰訊控股
        "9988.HK",   # 阿里巴巴
        "600519.SS", # 貴州茅台
        "ABVC",      # 美股 ABVC
        "NVDA",      # 美股 輝達
        "BTC-USD"    # 比特幣
    ]

if 'selected_quick' not in st.session_state:
    st.session_state.selected_quick = "2330.TW"

if 'ticker_input' not in st.session_state:
    st.session_state.ticker_input = "2330.TW"

# 代號自動正規化輔助函式
def normalize_ticker(symbol):
    symbol = symbol.upper().strip()
    if not symbol:
        return ""
    # 若為純數字，依長度預設自動補上台股或港股字尾
    if symbol.isdigit():
        if len(symbol) == 4:
            return f"{symbol}.TW"  # 預設台股上市
        elif len(symbol) == 5:
            return f"{symbol}.HK"  # 預設港股
        elif len(symbol) == 6:
            if symbol.startswith('6'):
                return f"{symbol}.SS" # 滬股
            else:
                return f"{symbol}.SZ" # 深股
    return symbol

# Callback: 當文字輸入框改變時，立即更新 Session State 及歷史清單
def on_ticker_input_change():
    raw_ticker = st.session_state.ticker_input_key
    new_ticker = normalize_ticker(raw_ticker)
    
    if new_ticker:
        st.session_state.ticker_input = new_ticker
        if new_ticker not in st.session_state.quick_tickers:
            st.session_state.quick_tickers.append(new_ticker)
            if len(st.session_state.quick_tickers) > 50:
                st.session_state.quick_tickers = st.session_state.quick_tickers[-50:]
        st.session_state.selected_quick = new_ticker

# Callback: 當下拉選單選擇改變時
def on_selectbox_change():
    sel = st.session_state.selectbox_key
    if sel != "自訂輸入":
        st.session_state.ticker_input = sel

# 2. 翻譯輔助函式
@st.cache_data(ttl=3600, show_spinner=False)
def translate_text(text, target_lang='zh-TW'):
    if not text:
        return ""
    try:
        translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
        return translated
    except Exception:
        return text

# 3. 側邊欄：華人熱門市場與參數設定
st.sidebar.header("🔍 全球華人市場標的設定")

options = ["自訂輸入"] + st.session_state.quick_tickers
default_idx = options.index(st.session_state.selected_quick) if st.session_state.selected_quick in options else 0

st.sidebar.selectbox(
    "🔥 熱門與歷史搜尋 (含台/港/A股/美股/加密貨幣)", 
    options=options, 
    index=default_idx,
    key="selectbox_key",
    on_change=on_selectbox_change
)

st.sidebar.text_input(
    "輸入代號 (台股如 2330, 港股如 0700, 美股如 NVDA)", 
    value=st.session_state.ticker_input,
    key="ticker_input_key",
    on_change=on_ticker_input_change
)

st.sidebar.markdown("---")
st.sidebar.caption("💡 **代號輸入指南**：")
st.sidebar.caption("• 台股：`2330` 或 `2330.TW` (上櫃 `8069.TWO`) \n• 港股：`0700.HK` \n• A股：`600519.SS` (滬) / `000001.SZ` (深)\n• 加密貨幣：`BTC-USD`")

include_extended = st.sidebar.checkbox("開啟延長交易/夜盤 (限美股與加密貨幣)", value=True)
interval = st.sidebar.selectbox("K線時間間隔", ["1m", "5m", "15m", "1d"], index=1)
period = "5d" if interval in ["1m", "5m", "15m"] else "1y"
refresh_rate = st.sidebar.slider("自動更新頻率 (秒)", min_value=10, max_value=60, value=20)

# 4. 新聞情緒分析模組
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
            pub_time_str = "未知時間"

            if isinstance(item, dict) and 'content' in item:
                content = item.get('content', {})
                title = content.get('title', '')
                publisher = content.get('provider', {}).get('displayName', 'Unknown')
                click_through_url = content.get('clickThroughUrl', {})
                canonical_url = content.get('canonicalUrl', {})
                link = click_through_url.get('url') or canonical_url.get('url') or '#'
                
                pub_date = content.get('pubDate')
                if pub_date:
                    try:
                        dt = datetime.datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                        pub_time_str = dt.strftime('%Y-%m-%d %H:%M')
                    except Exception:
                        pub_time_str = str(pub_date)[:16]
            else:
                title = item.get('title', '')
                publisher = item.get('publisher', 'Unknown')
                link = item.get('link', '#')
                
                pub_ts = item.get('providerPublishTime')
                if pub_ts:
                    try:
                        pub_time_str = datetime.datetime.fromtimestamp(pub_ts).strftime('%Y-%m-%d %H:%M')
                    except Exception:
                        pass

            if not title:
                continue
            
            # 若標題為英文，翻譯成繁體中文呈現；若原本為中文，則計算前轉英發送給 Vader
            title_zh = translate_text(title, target_lang='zh-TW')
            title_en_for_vader = translate_text(title, target_lang='en')
            
            score = sia.polarity_scores(title_en_for_vader)['compound']
            compound_scores.append(score)
            processed_news.append({
                'title': title,
                'title_zh': title_zh,
                'publisher': publisher,
                'link': link,
                'score': score,
                'pub_time': pub_time_str
            })
        
        avg_score = float(np.mean(compound_scores)) if compound_scores else 0.0
        return avg_score, processed_news
    except Exception:
        return 0.0, []

# 5. 技術指標計算模組
def compute_indicators(df, sentiment_score):
    data = df.copy()
    data['Returns'] = data['Close'].pct_change()
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    data['RSI'] = 100 - (100 / (1 + rs))
    
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    std_20 = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['SMA_20'] + (std_20 * 2)
    data['BB_Lower'] = data['SMA_20'] - (std_20 * 2)
    data['BB_PctB'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'] + 1e-9)
    
    data['Volatility'] = data['Returns'].rolling(window=10).std()
    data['Sentiment'] = sentiment_score
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    return data.dropna()

# 6. 數據擷取與預測建模
@st.cache_data(ttl=15, show_spinner=False)
def fetch_and_predict_dynamic(ticker, period="5d", interval="5m", extended=True):
    try:
        stock = yf.Ticker(ticker)
        
        # 僅美股與加密貨幣開啟 prepost，避免台/港/A股報錯
        use_prepost = extended if ("." not in ticker or ticker.endswith("-USD")) else False
        df = stock.history(period=period, interval=interval, prepost=use_prepost)
        
        if df.empty or len(df) < 20:
            return None, None, None, None, 0.0, [], None
            
        try:
            company_name = stock.info.get('longName') or stock.info.get('shortName') or ticker
        except Exception:
            company_name = ticker
        
        sentiment_score, news_data = get_news_sentiment(ticker)
        df = compute_indicators(df, sentiment_score)
        
        feature_cols = ['Returns', 'SMA_5', 'SMA_20', 'RSI', 'MACD', 'MACD_Signal', 'BB_PctB', 'Volatility', 'Sentiment']
        
        X = df[feature_cols][:-1]
        y = df['Target'][:-1]
        
        if len(X) < 15:
            return None, None, None, None, 0.0, [], None
            
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        test_acc = accuracy_score(y_test, model.predict(X_test)) if len(y_test) > 0 else 0.5
        
        latest_feature = df[feature_cols].iloc[[-1]]
        latest_pred = int(model.predict(latest_feature)[0])
        latest_probs = model.predict_proba(latest_feature)[0]
        
        latest_prob = float(latest_probs[latest_pred])
        
        return df, latest_pred, latest_prob, test_acc, sentiment_score, news_data, company_name
    except Exception:
        return None, None, None, None, 0.0, [], None

# 7. 主畫面局部刷新渲染模組
@st.fragment(run_every=refresh_rate)
def render_dashboard(symbol, p_period, p_interval, p_extended):
    if not symbol:
        st.warning("請在左側輸入股票或加密貨幣代號。")
        return

    df, latest_pred, latest_prob, test_acc, sentiment_score, news_data, company_name = fetch_and_predict_dynamic(
        symbol, p_period, p_interval, p_extended
    )

    if df is not None:
        st.title(f"⚡ {symbol} ({company_name}) 實時走勢與 AI 預測")
        
        latest_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        price_change = latest_price - prev_price
        pct_change = (price_change / prev_price) * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"{symbol} 最新報價", f"${latest_price:.2f}", f"{price_change:+.2f} ({pct_change:+.2f}%)")
        
        pred_text = "🟢 看漲 (UP)" if latest_pred == 1 else "🔴 看跌 (DOWN)"
        col2.metric("下一週期預測方向", pred_text)
        col3.metric("預測信心度", f"{latest_prob * 100:.1f}%")
        
        sentiment_label = "中性 🟡"
        if sentiment_score > 0.05: sentiment_label = "偏多 🟢"
        elif sentiment_score < -0.05: sentiment_label = "偏空 🔴"
        col4.metric("即時新聞情緒指數", f"{sentiment_score:+.3f}", sentiment_label)

        st.markdown("---")

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
            title=f"{symbol} 即時 K 線圖 (時間間隔: {p_interval})",
            xaxis_rangeslider_visible=False,
            height=500,
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.subheader(f"📰 {symbol} 最新相關新聞")
            if news_data:
                for item in news_data[:5]:
                    score_color = "green" if item['score'] > 0 else ("red" if item['score'] < 0 else "gray")
                    st.markdown(f"• [{item['title']}]({item['link']})")
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;<b>🌐 譯/標：{item['title_zh']}</b>", unsafe_allow_html=True)
                    st.markdown(
                        f"&nbsp;&nbsp;&nbsp;&nbsp;來源: *{item['publisher']}* | "
                        f"時間: `{item['pub_time']}` | "
                        f"情緒分數: <span style='color:{score_color}'>{item['score']:+.2f}</span>", 
                        unsafe_allow_html=True
                    )
                    st.markdown("<br>", unsafe_allow_html=True)
            else:
                st.info("暫無該標的之即時新聞數據。")

        with col_right:
            st.subheader("💡 市場涵蓋與指標說明")
            st.info(f"""
            * **當前選定標的**：**{symbol} ({company_name})**。
            * **華人全市場支援**：
              - **台股**：請輸入代號 (例如 `2330` 或 `2330.TW`)
              - **港股**：請輸入代號加 `.HK` (例如 `0700.HK`)
              - **A 股**：滬股 `.SS` / 深股 `.SZ` (例如 `600519.SS`)
              - **美股與加密貨幣**：支援 24 小時盤前盤後跳動與即時預測
            * **模型歷史測試準確率**：約為 **{test_acc * 100:.1f}%**。
            """)
    else:
        st.error(f"⚠️ 無法找到代號 **{symbol}** 的交易數據。請確認代號格式是否正確（例如台股是否加上 `.TW`，港股 `.HK`）。")

# 8. 執行主畫面渲染
render_dashboard(st.session_state.ticker_input, period, interval, include_extended)
