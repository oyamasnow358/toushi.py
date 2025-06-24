import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# --- 1. データ取得関数 (キャッシュ効率化) ---
@st.cache_data(ttl=3600) # 1時間キャッシュ
def get_stock_data(ticker_symbol):
    """yfinanceから全期間の株価データを取得"""
    return yf.download(ticker_symbol, period="max")

@st.cache_data(ttl=3600)
def get_fundamental_data(ticker_symbol):
    """yfinanceからファンダメンタルズデータを取得"""
    info = yf.Ticker(ticker_symbol).info
    fundamentals = {
        'pbr': info.get('priceToBook', 'N/A'),
        'per': info.get('trailingPE', 'N/A'),
        'peg': info.get('pegRatio', 'N/A'),
        'psr': info.get('priceToSalesTrailing12Months', 'N/A'),
        'forward_per': info.get('forwardPE', 'N/A')
    }
    return fundamentals

# --- 2. テクニカル指標を計算する関数 ---
def calculate_indicators(df):
    """データフレームにテクニカル指標を追加する"""
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    df['BB_UPPER'] = df['SMA20'] + (rolling_std * 2)
    df['BB_LOWER'] = df['SMA20'] - (rolling_std * 2)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rs = rs.fillna(0)
    df['RSI'] = 100 - (100 / (1 + rs))
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    df['High_60d'] = df['High'].rolling(window=60).max()
    
    df_weekly = df.resample('W-FRI').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
    exp1_w = df_weekly['Close'].ewm(span=12, adjust=False).mean()
    exp2_w = df_weekly['Close'].ewm(span=26, adjust=False).mean()
    df_weekly['MACD_W'] = exp1_w - exp2_w
    
    return df.dropna(subset=['SMA200']), df_weekly.dropna()

# --- 3. 分析ロジック関数 ---
def analyze_signals(df, df_weekly):
    """最新のデータに基づいて売買シグナル、トレンド、戦略を分析する"""
    if len(df) < 2 or len(df_weekly) < 1: return None 
    latest = df.iloc[-1]; previous = df.iloc[-2]; latest_weekly = df_weekly.iloc[-1]
    signals = {'buy': [], 'sell': []}; trends = {}; advice = {'buy_targets': [], 'sell_targets': []}; score = 0
    trends['long'] = "🟢 上昇基調" if latest['Close'] > latest['SMA200'] and latest_weekly['MACD_W'] > 0 else "🔴 下降基調" if latest['Close'] < latest['SMA200'] else "🟡 中立"
    trends['mid'] = "🟢 上昇" if latest['Close'] > latest['SMA50'] else "🔴 下降"
    trends['short'] = "🟢 上昇" if latest['Close'] > latest['SMA20'] else "🔴 調整/下降"
    if trends['long'] == "🟢 上昇基調":
        signals['buy'].append("✅ 長期トレンド良好"); score += 1
        if latest['Close'] <= latest['BB_LOWER']: signals['buy'].append("✅ Bバンド-2σ (売られすぎ)"); score += 1
        if 30 <= latest['RSI'] <= 45: signals['buy'].append(f"✅ RSI {latest['RSI']:.1f} (押し目)"); score += 1
    else: signals['sell'].append("❌ 長期トレンドが買い推奨ではない")
    if previous['MACD'] < previous['MACD_SIGNAL'] and latest['MACD'] > latest['MACD_SIGNAL']:
        signals['buy'].append("✅ MACDゴールデンクロス"); score += 1
        if latest['Volume'] > latest['Volume_MA20'] * 1.5: signals['buy'].append("🔥 出来高急増"); score += 1
    if latest['Close'] >= latest['BB_UPPER']: signals['sell'].append(f"⚠️ Bバンド+2σ (過熱感)")
    if latest['RSI'] >= 70: signals['sell'].append(f"⚠️ RSI {latest['RSI']:.1f} (買われすぎ)")
    if previous['MACD'] > previous['MACD_SIGNAL'] and latest['MACD'] < latest['MACD_SIGNAL']: signals['sell'].append("❌ MACDデッドクロス")
    if previous['Close'] > previous['SMA200'] and latest['Close'] < latest['SMA200']: signals['sell'].append("🚨 200日線下抜け(損切り警告)")
    advice['buy_targets'] = [f"Bバンド -2σ: **{latest['BB_LOWER']:.2f}**", f"50日線: **{latest['SMA50']:.2f}**"] if trends['long'] == "🟢 上昇基調" else ["長期トレンド下降のため押し目買い非推奨"]
    advice['sell_targets'] = [f"Bバンド +2σ: **{latest['BB_UPPER']:.2f}**", f"直近60日高値: **{latest['High_60d']:.2f}**"]
    final_score = min(score, 5) if trends['long'] == "🟢 上昇基調" else 0
    star_rating = "★" * final_score + "☆" * (5 - final_score)
    comment = "複数の買いシグナルが点灯。絶好の買い場が近い可能性。" if final_score >= 4 else "買いを検討できる良いタイミング。" if final_score >= 2 else "長期トレンド下降のため買いはリスク高。" if trends['long'] == "🔴 下降基調" else "明確な方向性なし。様子見が賢明。"
    return {"star_rating": star_rating, "score": final_score, "comment": comment, "signals": signals, "trends": trends, "advice": advice}

# --- 4. チャート描画関数 ---
def plot_chart(df, ticker):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.1, 0.15, 0.15])
    # 全ての描画を1つのtry-exceptブロックで囲む
    try:
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='ローソク足'), row=1, col=1)
        for sma, color, width, name in [(200, 'red', 2, 'SMA 200'), (50, 'green', 1.5, 'SMA 50'), (20, 'orange', 1, 'SMA 20')]:
            fig.add_trace(go.Scatter(x=df.index, y=df[f'SMA{sma}'], line=dict(color=color, width=width), name=name), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_UPPER'], line=dict(color='rgba(100,100,100,0.5)', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_LOWER'], line=dict(color='rgba(100,100,100,0.5)', width=1), fill='tonexty', fillcolor='rgba(100,100,100,0.1)'), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='出来高', marker_color='lightblue'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Volume_MA20'], line=dict(color='grey', width=1, dash='dash')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=3, col=1)
        fig.add_shape(type='line', x0=df.index[0], y0=70, x1=df.index[-1], y1=70, line=dict(color='red', dash='dash'), row=3, col=1)
        fig.add_shape(type='line', x0=df.index[0], y0=30, x1=df.index[-1], y1=30, line=dict(color='green', dash='dash'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_SIGNAL'], name='Signal', line=dict(color='orange')), row=4, col=1)
        fig.update_layout(title_text=f'{ticker} テクニカル分析チャート', height=800, xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        for i, title in enumerate(['価格', '出来高', 'RSI', 'MACD'], 1): fig.update_yaxes(title_text=title, row=i, col=1)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"チャート描画中にエラーが発生しました: {e}")

# --- 5. StreamlitアプリのメインUI ---
st.set_page_config(layout="wide"); st.title('📈 総合投資分析アプリ (テクニカル & ファンダメンタルズ)')
st.sidebar.header('分析設定'); ticker = st.sidebar.text_input('ティッカーシンボル (例: AAPL, 7203.T)', 'AAPL')
period_options = {'1ヶ月': 30, '3ヶ月': 90, '6ヶ月': 182, '1年': 365, '5年': 1825, '全期間': None}
selected_period = st.sidebar.radio("チャート表示期間", options=period_options.keys(), horizontal=True)

if ticker:
    raw_data = get_stock_data(ticker); fundamentals = get_fundamental_data(ticker)
    if raw_data.empty: st.error("ティッカーが見つからないか、データがありません。")
    else:
        data = pd.DataFrame({'Open': raw_data['Open'], 'High': raw_data['High'], 'Low': raw_data['Low'], 'Close': raw_data['Close'], 'Volume': raw_data['Volume']})
        analyzed_df, weekly_df = calculate_indicators(data.copy())
        analysis_result = analyze_signals(analyzed_df, weekly_df)
        display_df = analyzed_df[analyzed_df.index >= datetime.now() - timedelta(days=period_options[selected_period])] if period_options[selected_period] is not None else analyzed_df
        if analysis_result is None or display_df.empty: st.warning("分析データが不足しています。")
        else:
            st.header(f'分析結果：{ticker}')
            col1, col2 = st.columns([1.2, 1.8])
            with col1:
                st.subheader("📊 テクニカル総合評価"); st.metric(label="買いシグナル強度", value=analysis_result['star_rating'])
                st.info(f"**コメント：** {analysis_result['comment']}")
                
                st.subheader("📈 トレンド分析"); st.markdown(f"- 長期: {analysis_result['trends']['long']} / 中期: {analysis_result['trends']['mid']} / 短期: {analysis_result['trends']['short']}")
                
                # --- ファンダメンタルズ分析セクション ---
                st.subheader("💰 バリュエーション分析 (割安性)")
                with st.container(border=True):
                    for key, name in [('pbr', 'PBR (純資産)'), ('per', 'PER (利益)'), ('peg', 'PEG (成長性)')]:
                        val = fundamentals[key]
                        if isinstance(val, (int, float)):
                            if key == 'pbr': text = f"**{val:.2f} 倍** (1倍未満で割安)"
                            elif key == 'per': text = f"**{val:.2f} 倍** (15倍未満で割安目安)"
                            elif key == 'peg': text = f"**{val:.2f}** (1倍未満で割安)"
                            st.markdown(f"- **{name}:** {text}")
                        else: st.markdown(f"- **{name}:** データなし")

                st.subheader("🎯 売買戦略アドバイス")
                with st.container(border=True): st.markdown("**押し目買い目標 (目安)**"); [st.markdown(f" - {t}") for t in analysis_result['advice']['buy_targets']]
                with st.container(border=True): st.markdown("**利益確定目標 (目安)**"); [st.markdown(f" - {t}") for t in analysis_result['advice']['sell_targets']]
                with st.expander("🔍 シグナルの詳細な根拠を見る"): [st.markdown(f"  - {s}") for s in analysis_result['signals']['buy'] + analysis_result['signals']['sell']]

            with col2: plot_chart(display_df, ticker)

st.sidebar.markdown("---"); st.sidebar.markdown("**免責事項**"); st.sidebar.info("本アプリは教育目的で作成されたものであり、特定の金融商品を推奨するものではありません。表示される価格や指標はあくまで分析に基づく目安であり、利益を保証するものではありません。投資の最終的な判断はご自身の責任で行ってください。")