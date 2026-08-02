import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

# 1. 頁面配置與 CSS 注入
st.set_page_config(
    page_title="24H 實時走勢預測與回測系統", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* 修正頂部邊界 */
    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 1.5rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    
    /* 股票標題區塊 */
    .title-wrapper {
        margin-top: 0.2rem !important;
        margin-bottom: 1rem !important;
        line-height: 1.4 !important;
    }
    
    .symbol-text {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: #1E88E5 !important;
        margin-right: 8px !important;
        display: inline-block !important;
    }
    
    .name-text {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        color: #424242 !important;
        display: inline-block !important;
        word-break: break-word !important;
    }

    /* 指標數據卡片優化 */
    [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 8px 12px !important;
        border: 1px solid rgba(0, 0, 0, 0.08);
    }

    @media (max-width: 900px) {
        .main .block-container {
            padding-top: 2.2rem !important;
            padding-left: 0.6rem !important;
            padding-right: 0.6rem !important;
        }
        
        .symbol-text {
            font-size: 1.4rem !important;
        }
        
        .name-text {
            font-size: 1.1rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# 2. 熱門與常見標的繁體中文名稱字典映射
STOCK_NAME_MAP = {
    # 台股 (上市/上櫃)
    "2330.TW": "台積電", "2454.TW": "聯發科", "2317.TW": "鴻海", "2308.TW": "台達電",
    "2382.TW": "廣達", "3231.TW": "緯創", "6669.TW": "緯穎", "2379.TW": "瑞昱",
    "3008.TW": "大立光", "2303.TW": "聯電", "2881.TW": "富邦金", "2882.TW": "國泰金",
    "2886.TW": "兆豐金", "0050.TW": "元大台灣50", "0056.TW": "元大高股息",
    "00878.TW": "國泰永續高股息", "00919.TW": "群益台灣精選高息", "00929.TW": "復華台灣科技優息",
    
    # 美股熱門
    "NVDA": "輝達", "TSLA": "特斯拉", "AAPL": "蘋果", "MSFT": "微軟",
    "GOOGL": "谷歌 (Alphabet)", "AMZN": "亞馬遜", "META": "Meta", "AMD": "超微半導體",
    "INTC": "英特爾", "TSM": "台積電 ADR", "ABVC": "ABVC BioPharma", "PLTR": "Palantir",
    "NFLX": "網飛", "DIS": "迪士尼", "BA": "波音", "BABA": "阿里巴巴 ADR",
    
    # 港股 / A股 / 加密貨幣
    "0700.HK": "騰訊控股", "9988.HK": "阿里巴巴", "3690.HK": "美團", "1810.HK": "小米集團", "9888.HK": "百度",
    "600519.SS": "貴州茅台", "000858.SZ": "五糧液",
    "BTC-USD": "比特幣", "ETH-USD": "以太幣", "SOL-USD": "Solana", "BNB-USD": "幣安幣",
    "XRP-USD": "瑞波幣", "DOGE-USD": "狗狗幣", "AVAX-USD": "Avalanche", "LINK-USD": "Chainlink"
}

# 預設 50 筆無重複的熱門與歷史標的清單
DEFAULT_50_TICKERS = [
    "2330.TW", "2454.TW", "2317.TW", "2308.TW", "2382.TW", "3231.TW", "6669.TW", "2379.TW",
    "3008.TW", "2303.TW", "2881.TW", "2882.TW", "2886.TW", "0050.TW", "0056.TW", "00878.TW",
    "00919.TW", "00929.TW", "NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "AMD", "INTC", "TSM", "ABVC", "PLTR", "NFLX", "DIS", "BA", "BABA", "0700.HK",
    "9988.HK", "3690.HK", "1810.HK", "9888.HK", "600519.SS", "000858.SZ", "BTC-USD",
    "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD", "AVAX-USD", "LINK-USD"
]

def get_company_name(ticker, stock_info=None):
    ticker_upper = ticker.upper().strip()
    if ticker_upper in STOCK_NAME_MAP:
        return STOCK_NAME_MAP[ticker_upper]
    if stock_info:
        name = stock_info.get('shortName') or stock_info.get('longName')
        if name:
            return name
    return ticker_upper

# 保持順序的清單去重函式
def deduplicate_list(input_list):
    seen = set()
    return [x for x in input_list if not (x in seen or seen.add(x))]

# 初始化 Session State
if 'quick_tickers' not in st.session_state:
    st.session_state.quick_tickers = deduplicate_list(DEFAULT_50_TICKERS)

if 'selected_quick' not in st.session_state:
    st.session_state.selected_quick = "2454.TW"

if 'ticker_input' not in st.session_state:
    st.session_state.ticker_input = "2454.TW"

# 代號正規化
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
        # 移除重複並推到最前方
        if new_ticker in st.session_state.quick_tickers:
            st.session_state.quick_tickers.remove(new_ticker)
        st.session_state.quick_tickers.insert(0, new_ticker)
        st.session_state.quick_tickers = deduplicate_list(st.session_state.quick_tickers)[:50]
        st.session_state.selected_quick = new_ticker

def on_selectbox_change():
    sel = st.session_state.selectbox_key
    if sel != "自訂輸入":
        clean_ticker = sel.split(" ")[0]
        st.session_state.ticker_input = clean_ticker

# 刪除指定歷史紀錄的 Callback
def delete_ticker(ticker_to_remove):
    if ticker_to_remove in st.session_state.quick_tickers:
        st.session_state.quick_tickers.remove(ticker_to_remove)
        if st.session_state.ticker_input == ticker_to_remove:
            st.session_state.ticker_input = st.session_state.quick_tickers[0] if st.session_state.quick_tickers else ""
            st.session_state.selected_quick = st.session_state.ticker_input

# 側邊欄設定
st.sidebar.header("🔍 全球標的與回測設定")

# 確保列表隨時無重複
st.session_state.quick_tickers = deduplicate_list(st.session_state.quick_tickers)

# 選單項目加入繁體中文名稱備註
formatted_options = ["自訂輸入"] + [
    f"{t} ({STOCK_NAME_MAP.get(t, '')})" if t in STOCK_NAME_MAP else t 
    for t in st.session_state.quick_tickers
]

current_selected_formatted = f"{st.session_state.selected_quick} ({STOCK_NAME_MAP.get(st.session_state.selected_quick, '')})" if st.session_state.selected_quick in STOCK_NAME_MAP else st.session_state.selected_quick
default_idx = formatted_options.index(current_selected_formatted) if current_selected_formatted in formatted_options else 0

st.sidebar.selectbox(
    f"🔥 熱門與歷史搜尋 (共 {len(st.session_state.quick_tickers)} 筆)", 
    options=formatted_options, 
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

# 歷史紀錄管理摺疊選單 (加上索引 i 確保 key 唯一性)
with st.sidebar.expander("🗑️ 管理歷史搜尋清單", expanded=False):
    st.caption("點擊按鈕可自清單中移除特定代號：")
    for i, t in enumerate(st.session_state.quick_tickers):
        c1, c2 = st.columns([3, 1])
        c1.text(f"{t} ({STOCK_NAME_MAP.get(t, '')})")
        if c2.button("刪除", key=f"del_{i}_{t}"):
            delete_ticker(t)
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 策略與風控參數")
hold_periods = st.sidebar.slider("最長持倉 K 棒數", min_value=1, max_value=10, value=3)
stop_loss_pct = st.sidebar.slider("停損百分比 (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.1) / 100.0
take_profit_pct = st.sidebar.slider("停利百分比 (%)", min_value=0.5, max_value=10.0, value=2.0, step=0.1) / 100.0
tx_fee = st.sidebar.number_input("單邊交易手續費/滑價 (%)", value=0.05, step=0.01) / 100.0

st.sidebar.markdown("---")
include_extended = st.sidebar.checkbox("開啟延長交易/夜盤", value=True)
interval = st.sidebar.selectbox("K線時間間隔", ["1m", "5m", "15m", "1d"], index=1)
period = "5d" if interval in ["1m", "5m", "15m"] else "1y"
refresh_rate = st.sidebar.slider("自動更新頻率 (秒)", min_value=10, max_value=60, value=20)

# 指標計算
def compute_indicators(df):
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
    data['Target'] = (data['Close'].shift(-hold_periods) > data['Close']).astype(int)
    
    return data.dropna()

# 回測引擎
def run_backtest(df, model, feature_cols, hold_p, sl, tp, fee):
    df_bt = df.copy()
    X = df_bt[feature_cols]
    
    probs = model.predict_proba(X)[:, 1]
    df_bt['Prob_Up'] = probs
    
    signals = []
    for idx in range(len(df_bt)):
        prob = df_bt['Prob_Up'].iloc[idx]
        close = df_bt['Close'].iloc[idx]
        sma5 = df_bt['SMA_5'].iloc[idx]
        
        if prob > 0.55 and close > sma5:
            signals.append(1)
        elif prob < 0.45 and close < sma5:
            signals.append(-1)
        else:
            signals.append(0)
            
    df_bt['Signal'] = signals
    trades = []
    equity_curve = [1.0]
    
    i = 0
    while i < len(df_bt) - hold_p:
        sig = df_bt['Signal'].iloc[i]
        if sig != 0:
            entry_price = df_bt['Close'].iloc[i]
            entry_time = df_bt.index[i]
            trade_ret = 0.0
            exit_reason = "Time Exit"
            
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
            
            net_ret = trade_ret - (fee * 2)
            trades.append({
                'entry_time': entry_time,
                'type': 'LONG' if sig == 1 else 'SHORT',
                'entry_price': entry_price,
                'ret': net_ret,
                'exit_reason': exit_reason
            })
            equity_curve.append(equity_curve[-1] * (1 + net_ret))
            i += hold_p
        else:
            i += 1
            
    if trades:
        trade_df = pd.DataFrame(trades)
        win_rate = (trade_df['ret'] > 0).mean() * 100.0
        total_ret = (equity_curve[-1] - 1.0) * 100.0
        bh_ret = ((df_bt['Close'].iloc[-1] - df_bt['Close'].iloc[0]) / df_bt['Close'].iloc[0]) * 100.0
        
        eq_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(eq_arr)
        drawdown = (eq_arr - peak) / peak
        max_dd = np.min(drawdown) * 100.0
        
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
        'sharpe': sharpe
    }
    return metrics, trade_df

# 數據擷取與預測建模
@st.cache_data(ttl=15, show_spinner=False)
def fetch_and_predict_dynamic(ticker, period="5d", interval="5m", extended=True):
    try:
        stock = yf.Ticker(ticker)
        use_prepost = extended if ("." not in ticker or ticker.endswith("-USD")) else False
        df = stock.history(period=period, interval=interval, prepost=use_prepost)
        
        stock_info = None
        try:
            stock_info = stock.info
        except Exception:
            pass
        company_name = get_company_name(ticker, stock_info)

        if df.empty or len(df) < 30:
            return None, None, None, None, None, company_name
            
        df = compute_indicators(df)
        feature_cols = ['Returns', 'SMA_5', 'SMA_20', 'RSI', 'MACD', 'MACD_Signal', 'BB_PctB', 'Volatility']
        
        X = df[feature_cols][:-hold_periods]
        y = df['Target'][:-hold_periods]
        
        if len(X) < 20:
            return None, None, None, None, None, company_name
            
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        model = RandomForestClassifier(n_estimators=150, max_depth=4, min_samples_leaf=5, random_state=42)
        model.fit(X_train, y_train)
        
        test_df = df.iloc[split_idx:]
        bt_metrics, trade_history = run_backtest(
            test_df, model, feature_cols, hold_periods, stop_loss_pct, take_profit_pct, tx_fee
        )
        
        latest_feature = df[feature_cols].iloc[[-1]]
        latest_pred = int(model.predict(latest_feature)[0])
        latest_probs = model.predict_proba(latest_feature)[0]
        latest_prob = float(latest_probs[latest_pred])
        
        return df, latest_pred, latest_prob, bt_metrics, trade_history, company_name
    except Exception:
        company_name = STOCK_NAME_MAP.get(ticker.upper().strip(), ticker)
        return None, None, None, None, None, company_name

# 主畫面渲染
@st.fragment(run_every=refresh_rate)
def render_dashboard(symbol, p_period, p_interval, p_extended):
    if not symbol:
        st.warning("請在左側選單輸入或選擇股票代號。")
        return

    df, latest_pred, latest_prob, bt_metrics, trade_history, company_name = fetch_and_predict_dynamic(
        symbol, p_period, p_interval, p_extended
    )

    if df is not None and bt_metrics is not None:
        st.markdown(f"""
        <div class="title-wrapper">
            <span class="symbol-text">⚡ {symbol}</span>
            <span class="name-text">({company_name})</span>
        </div>
        """, unsafe_allow_html=True)
        
        latest_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        price_change = latest_price - prev_price
        pct_change = (price_change / prev_price) * 100

        col1, col2 = st.columns(2)
        with col1:
            st.metric("最新報價", f"${latest_price:.2f}", f"{price_change:+.2f} ({pct_change:+.2f}%)")
            pred_text = "🟢 看漲 (UP)" if latest_pred == 1 else "🔴 看跌 (DOWN)"
            st.metric("下一週期方向", pred_text)
        with col2:
            st.metric("模型信心度", f"{latest_prob * 100:.1f}%")
            st.metric("歷史測試勝率", f"{bt_metrics['win_rate']:.1f}%", f"共 {bt_metrics['total_trades']} 筆交易")

        st.markdown("---")

        tab1, tab2 = st.tabs(["📈 K 線圖表", "📊 策略回測績效"])

        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name=company_name
            ))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_5'], line=dict(color='orange', width=1), name="SMA5"))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='cyan', width=1), name="SMA20"))
            
            fig.update_layout(
                margin=dict(l=10, r=10, t=25, b=10),
                xaxis_rangeslider_visible=False,
                height=350,
                template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with tab2:
            st.markdown("#### 📋 策略樣本外驗證 (Out-of-Sample)")
            
            m1, m2 = st.columns(2)
            m1.metric("策略累積報酬", f"{bt_metrics['total_ret']:+.2f}%")
            m2.metric("買入持有 (B&H)", f"{bt_metrics['bh_ret']:+.2f}%")
            
            m3, m4 = st.columns(2)
            m3.metric("最大歷史回撤", f"{bt_metrics['max_dd']:.2f}%")
            m4.metric("夏普比率", f"{bt_metrics['sharpe']:.2f}")

            if not trade_history.empty:
                st.markdown("##### 📝 最近交易明細")
                display_trades = trade_history.copy()
                display_trades['ret'] = display_trades['ret'].apply(lambda x: f"{x*100:+.2f}%")
                display_trades['entry_price'] = display_trades['entry_price'].apply(lambda x: f"${x:.2f}")
                display_trades.columns = ['時間', '方向', '進場價', '淨報酬', '原因']
                
                st.dataframe(display_trades.tail(8), use_container_width=True, height=220)
            else:
                st.info("當前條件下，歷史測試區間未觸發交易。")

    else:
        st.error(f"⚠️ 無法取得 **{symbol} ({company_name})** 資料，請確認代號與網路狀態。")

# 執行主畫面渲染
render_dashboard(st.session_state.ticker_input, period, interval, include_extended)
