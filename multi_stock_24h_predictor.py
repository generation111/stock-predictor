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

st.set_page_config(page_title="華人跨國市場 24H 實時走勢預測與回測系統", layout="wide")

# 初始化 Session State
if 'quick_tickers' not in st.session_state:
    st.session_state.quick_tickers = [
        "2330.TW",   # 台積電
        "2454.TW",   # 聯發科
        "0700.HK",   # 騰訊控股
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
    if symbol.isdigit():
        if len(symbol) == 4:
            return f"{symbol}.TW"
        elif len(symbol) == 5:
            return f"{symbol}.HK"
        elif len(symbol) == 6:
            return f"{symbol}.SS" if symbol.startswith('6') else f"{symbol}.SZ"
    return symbol

# Callbacks
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

def on_selectbox_change():
    sel = st.session_state.selectbox_key
    if sel != "自訂輸入":
        st.session_state.ticker_input = sel

# 翻譯輔助函式
@st.cache_data(ttl=3600, show_spinner=False)
def translate_text(text, target_lang='zh-TW'):
    if not text:
        return ""
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception:
        return text

# 側邊欄：市場與策略回測設定
st.sidebar.header("🔍 全球標的與回測條件設定")

options = ["自訂輸入"] + st.session_state.quick_tickers
default_idx = options.index(st.session_state.selected_quick) if st.session_state.selected_quick in options else 0

st.sidebar.selectbox(
    "🔥 熱門與歷史搜尋", 
    options=options, 
    index=default_idx,
    key="selectbox_key",
    on_change=on_selectbox_change
)

st.sidebar.text_input(
    "輸入代號 (台/港/A股/美股/加密貨幣)", 
    value=st.session_state.ticker_input,
    key="ticker_input_key",
    on_change=on_ticker_input_change
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 策略與風控參數設定")
hold_periods = st.sidebar.slider("最長持倉 K 棒數 (Periods)", min_value=1, max_value=10, value=3)
stop_loss_pct = st.sidebar.slider("停損百分比 (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.1) / 100.0
take_profit_pct = st.sidebar.slider("停利百分比 (%)", min_value=0.5, max_value=10.0, value=2.0, step=0.1) / 100.0
tx_fee = st.sidebar.number_input("預估交易摩擦成本/單邊 (%)", value=0.05, step=0.01) / 100.0

st.sidebar.markdown("---")
include_extended = st.sidebar.checkbox("開啟延長交易/夜盤 (限美股與加密貨幣)", value=True)
interval = st.sidebar.selectbox("K線時間間隔", ["1m", "5m", "15m", "1d"], index=1)
period = "5d" if interval in ["1m", "5m", "15m"] else "1y"
refresh_rate = st.sidebar.slider("自動更新頻率 (秒)", min_value=10, max_value=60, value=20)

# 技術指標計算
def compute_indicators(df, sentiment_score=0.0):
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
    data['Target'] = (data['Close'].shift(-hold_periods) > data['Close']).astype(int)
    
    return data.dropna()

# 回測核心引擎 (Backtest Engine)
def run_backtest(df, model, feature_cols, hold_p, sl, tp, fee):
    df_bt = df.copy()
    X = df_bt[feature_cols]
    
    # 取得模型的看漲機率
    probs = model.predict_proba(X)[:, 1]
    df_bt['Prob_Up'] = probs
    
    # 策略進場訊號 (1: 做多, -1: 做空, 0: 觀望)
    signals = []
    for idx in range(len(df_bt)):
        prob = df_bt['Prob_Up'].iloc[idx]
        close = df_bt['Close'].iloc[idx]
        sma5 = df_bt['SMA_5'].iloc[idx]
        
        # 濾波條件：機率高於 55% 且價格站在短均線上才做多
        if prob > 0.55 and close > sma5:
            signals.append(1)
        elif prob < 0.45 and close < sma5:
            signals.append(-1)
        else:
            signals.append(0)
            
    df_bt['Signal'] = signals
    
    # 模擬交易執行
    trades = []
    equity_curve = [1.0] # 初始資金標準化為 1.0
    
    i = 0
    while i < len(df_bt) - hold_p:
        sig = df_bt['Signal'].iloc[i]
        if sig != 0:
            entry_price = df_bt['Close'].iloc[i]
            entry_time = df_bt.index[i]
            
            trade_ret = 0.0
            exit_reason = "Time Exit"
            
            # 檢查未來 hold_p 個週期內的動態停損停利
            for step in range(1, hold_p + 1):
                curr_price = df_bt['Close'].iloc[i + step]
                price_ret = (curr_price - entry_price) / entry_price if sig == 1 else (entry_price - curr_price) / entry_price
                
                if price_ret <= -sl:
                    trade_ret = -sl
                    exit_reason = "Stop Loss"
                    break
                elif price_ret >= tp:
                    trade_ret = tp
                    exit_reason = "Take Profit"
                    break
                else:
                    trade_ret = price_ret
            
            # 扣除雙邊交易摩擦成本
            net_ret = trade_ret - (fee * 2)
            trades.append({
                'entry_time': entry_time,
                'type': 'LONG' if sig == 1 else 'SHORT',
                'entry_price': entry_price,
                'ret': net_ret,
                'exit_reason': exit_reason
            })
            equity_curve.append(equity_curve[-1] * (1 + net_ret))
            i += hold_p # 避免重疊進場
        else:
            i += 1
            
    # 計算績效指標
    if trades:
        trade_df = pd.DataFrame(trades)
        win_rate = (trade_df['ret'] > 0).mean() * 100.0
        total_ret = (equity_curve[-1] - 1.0) * 100.0
        
        # 買入持有報酬率 (Buy & Hold)
        bh_ret = ((df_bt['Close'].iloc[-1] - df_bt['Close'].iloc[0]) / df_bt['Close'].iloc[0]) * 100.0
        
        # 最大回撤 (MDD)
        eq_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(eq_arr)
        drawdown = (eq_arr - peak) / peak
        max_dd = np.min(drawdown) * 100.0
        
        # 夏普比率 (Sharpe Ratio)
        returns_arr = trade_df['ret'].values
        sharpe = (np.mean(returns_arr) / (np.std(returns_arr) + 1e-9)) * np.sqrt(252) if len(returns_arr) > 1 else 0.0
    else:
        trade_df = pd.DataFrame()
        win_rate, total_ret, bh_ret, max_dd, sharpe = 0.0, 0.0, 0.0, 0.0, 0.0

    metrics = {
        'total_trades': len(trades),
        'win_rate': win_rate,
        'total_ret': total_ret,
        'bh_ret': bh_ret,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'equity_curve': equity_curve
    }
    return metrics, trade_df

# 數據擷取與預測建模
@st.cache_data(ttl=15, show_spinner=False)
def fetch_and_predict_dynamic(ticker, period="5d", interval="5m", extended=True):
    try:
        stock = yf.Ticker(ticker)
        use_prepost = extended if ("." not in ticker or ticker.endswith("-USD")) else False
        df = stock.history(period=period, interval=interval, prepost=use_prepost)
        
        if df.empty or len(df) < 30:
            return None, None, None, None, 0.0, company_name if 'company_name' in locals() else ticker
            
        try:
            company_name = stock.info.get('longName') or stock.info.get('shortName') or ticker
        except Exception:
            company_name = ticker
        
        df = compute_indicators(df)
        feature_cols = ['Returns', 'SMA_5', 'SMA_20', 'RSI', 'MACD', 'MACD_Signal', 'BB_PctB', 'Volatility']
        
        X = df[feature_cols][:-hold_periods]
        y = df['Target'][:-hold_periods]
        
        if len(X) < 20:
            return None, None, None, None, 0.0, company_name
            
        # 樣本拆分：前 70% 訓練，後 30% 進行 Walk-Forward 測試驗證
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        model = RandomForestClassifier(n_estimators=150, max_depth=4, min_samples_leaf=5, random_state=42)
        model.fit(X_train, y_train)
        
        # 執行樣本外 (Out-of-Sample) 策略回測
        test_df = df.iloc[split_idx:]
        bt_metrics, trade_history = run_backtest(
            test_df, model, feature_cols, hold_periods, stop_loss_pct, take_profit_pct, tx_fee
        )
        
        # 當前最新週期預測
        latest_feature = df[feature_cols].iloc[[-1]]
        latest_pred = int(model.predict(latest_feature)[0])
        latest_probs = model.predict_proba(latest_feature)[0]
        latest_prob = float(latest_probs[latest_pred])
        
        return df, latest_pred, latest_prob, bt_metrics, trade_history, company_name
    except Exception:
        return None, None, None, None, None, ticker

# 主畫面局部刷新渲染模組
@st.fragment(run_every=refresh_rate)
def render_dashboard(symbol, p_period, p_interval, p_extended):
    if not symbol:
        st.warning("請在左側輸入股票代號。")
        return

    df, latest_pred, latest_prob, bt_metrics, trade_history, company_name = fetch_and_predict_dynamic(
        symbol, p_period, p_interval, p_extended
    )

    if df is not None and bt_metrics is not None:
        st.title(f"⚡ {symbol} ({company_name}) 實時預測與策略回測驗證")
        
        latest_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        price_change = latest_price - prev_price
        pct_change = (price_change / prev_price) * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"{symbol} 當前最新報價", f"${latest_price:.2f}", f"{price_change:+.2f} ({pct_change:+.2f}%)")
        
        pred_text = "🟢 看漲 (UP)" if latest_pred == 1 else "🔴 看跌 (DOWN)"
        col2.metric("下一週期方向預估", pred_text)
        col3.metric("模型信心度", f"{latest_prob * 100:.1f}%")
        col4.metric("樣本外測試集勝率", f"{bt_metrics['win_rate']:.1f}%", f"總交易 {bt_metrics['total_trades']} 筆")

        st.markdown("---")

        # 頁籤分類：技術圖表 vs 回測績效報告
        tab1, tab2 = st.tabs(["📈 即時 K 線與指標圖", "📊 策略歷史回測驗證 (Out-of-Sample)"])

        with tab1:
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
                height=480,
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("📋 策略回測績效總覽 (過濾低信心訊號 + 扣除手續費)")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("策略總累積報酬率", f"{bt_metrics['total_ret']:+.2f}%")
            m2.metric("買入持有 (B&H) 報酬率", f"{bt_metrics['bh_ret']:+.2f}%")
            m3.metric("最大歷史回撤 (MDD)", f"{bt_metrics['max_dd']:.2f}%")
            m4.metric("年化夏普比率", f"{bt_metrics['sharpe']:.2f}")

            st.markdown("<br>", unsafe_allow_html=True)
            
            if not trade_history.empty:
                st.subheader("📝 近期進出場詳細交易明細")
                display_trades = trade_history.copy()
                display_trades['ret'] = display_trades['ret'].apply(lambda x: f"{x*100:+.2f}%")
                display_trades['entry_price'] = display_trades['entry_price'].apply(lambda x: f"${x:.2f}")
                display_trades.columns = ['進場時間', '方向', '進場價格', '淨報酬率 (含扣費)', '出場原因']
                st.dataframe(display_trades.tail(10), use_container_width=True)
            else:
                st.info("在當前風控與信心度門檻下，歷史測試區間未觸發進場交易條件。")

            st.info(f"""
            📌 **當前回測條件邏輯**：
            * **做多進場**：模型預估看漲信心 $> 55\%$ 且 價格在 SMA 5 之上。
            * **做空進場**：模型預估看跌信心 $> 55\%$（看漲信心 $< 45\%$）且 價格在 SMA 5 之下。
            * **風控條件**：固定持倉不超過 `{hold_periods}` 個 K 棒，設置停損 `{stop_loss_pct*100:.1f}%`、停利 `{take_profit_pct*100:.1f}%`，每筆扣除交易成本 `{tx_fee*100:.2f}%`。
            """)
    else:
        st.error(f"⚠️ 無法取得 **{symbol}** 的交易資料或無法完成回測，請確認代號格式是否正確。")

# 執行主畫面渲染
render_dashboard(st.session_state.ticker_input, period, interval, include_extended)
