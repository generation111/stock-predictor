import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import datetime

# 1. 頁面配置與 CSS 流動自適應注入
st.set_page_config(
    page_title="24H 實時走勢預測與回測系統", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 注入修復股票名稱顯示不完整的 CSS
st.markdown("""
<style>
    /* 修正頂部容器 Margin / Padding */
    .main .block-container {
        padding-top: 2.2rem !important;
        padding-bottom: 1.5rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
    }
    
    /* 股票標題彈性佈局容器 */
    .stock-header-wrapper {
        display: flex;
        flex-wrap: wrap;
        align-items: baseline;
        gap: 6px 10px;
        margin-top: 0.5rem !important;
        margin-bottom: 1rem !important;
        width: 100%;
        overflow: hidden;
    }
    
    .stock-symbol-text {
        font-size: 1.6rem !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        line-height: 1.2 !important;
        white-space: nowrap;
    }
    
    .stock-name-text {
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        color: #A0A0A0 !important;
        line-height: 1.2 !important;
        word-break: break-word; /* 允許長文字自動換行 */
        overflow-wrap: break-word;
        max-width: 100%;
    }

    /* 針對手機和平板直立畫面的 RWD 調優 */
    @media (max-width: 900px) {
        .main .block-container {
            padding-top: 2.5rem !important;
            padding-left: 0.6rem !important;
            padding-right: 0.6rem !important;
        }
        
        .stock-symbol-text {
            font-size: 1.35rem !important;
        }
        
        .stock-name-text {
            font-size: 0.95rem !important;
        }
        
        /* 讓 metric 卡片在直立螢幕上有清晰的卡片外觀 */
        [data-testid="stMetric"] {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 8px 12px !important;
            margin-bottom: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* 頁籤微調為適合觸控點擊 */
        .stTabs [data-baseweb="tab"] {
            height: 42px !important;
            font-size: 14px !important;
            padding-left: 8px !important;
            padding-right: 8px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

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

# 側邊欄：市場與策略回測設定
st.sidebar.header("🔍 全球標的與回測設定")

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

# 技術指標計算
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

# 回測核心引擎 (Backtest Engine)
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
        
        if df.empty or len(df) < 30:
            return None, None, None, None, None, ticker
            
        try:
            company_name = stock.info.get('shortName') or stock.info.get('longName') or ticker
        except Exception:
            company_name = ticker
        
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
        return None, None, None, None, None, ticker

# 主畫面渲染 (修復股票名稱顯示完整性)
@st.fragment(run_every=refresh_rate)
def render_dashboard(symbol, p_period, p_interval, p_extended):
    if not symbol:
        st.warning("請在左側選單輸入股票代號。")
        return

    df, latest_pred, latest_prob, bt_metrics, trade_history, company_name = fetch_and_predict_dynamic(
        symbol, p_period, p_interval, p_extended
    )

    if df is not None and bt_metrics is not None:
        # 使用自適應 Flex 容器，確保代號與過長的公司名稱皆可彈性自動換行並完整顯示
        st.markdown(f"""
        <div class="stock-header-wrapper">
            <span class="stock-symbol-text">⚡ {symbol}</span>
            <span class="stock-name-text">({company_name})</span>
        </div>
        """, unsafe_allow_html=True)
        
        latest_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        price_change = latest_price - prev_price
        pct_change = (price_change / prev_price) * 100

        # 直立式自適應網格：2x2 排版
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
                low=df['Low'], close=df['Close'], name=symbol
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
        st.error(f"⚠️ 無法取得 **{symbol}** 資料，請確認代號與網路狀態。")

# 執行主畫面渲染
render_dashboard(st.session_state.ticker_input, period, interval, include_extended)
